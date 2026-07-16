"""M6-R5 on-policy ranking distillation for constraint-ranker-v6."""

from __future__ import annotations

import argparse
import gc
import json
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from trainer.config import ExperimentConfig
from trainer.data import generate_on_policy_teacher_dataset
from trainer.data.hard_teacher import generate_component_balanced_teacher_dataset
from trainer.encoding_v3 import NUM_CHANNELS_V3
from trainer.env import DIFFICULTIES
from trainer.evaluation.axial import TorchAxialPolicy, evaluate_axial_policy
from trainer.evaluation.suites import get_suite
from trainer.models import build_policy_model
from trainer.pretrain_posterior import PaddedTeacherView
from trainer.ranker_loss import ranker_losses, ranker_metrics
from trainer.storage import save_checkpoint


class RankerPretrainer:
    """CUDA-friendly trainer that leaves legacy teacher objectives unchanged."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        learning_rate: float,
        mixed_precision: bool,
        rank_margin: float,
        risk_temperature: float,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        self.use_amp = mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.rank_margin = rank_margin
        self.risk_temperature = risk_temperature

    def train_epoch(self, dataset: Dataset, batch_size: int) -> dict[str, float]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        totals: dict[str, float] = {}
        examples = 0
        self.model.train()
        for batch in loader:
            count = int(batch["state"].shape[0])
            state = batch["state"].to(self.device, non_blocking=True)
            legal = (
                batch["legal_mask"]
                .to(
                    self.device,
                    non_blocking=True,
                )
                .bool()
            )
            target_policy = batch["policy_target"].to(
                self.device,
                non_blocking=True,
            )
            target_risk = batch["risk_target"].to(
                self.device,
                non_blocking=True,
            )
            target_mine = batch["mine_target"].to(
                self.device,
                non_blocking=True,
            )
            target_certainty = batch["certainty_target"].to(
                self.device,
                non_blocking=True,
            )
            target_value = batch["value_target"].to(
                self.device,
                non_blocking=True,
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                output = self.model(state, legal)
                losses = ranker_losses(
                    output,
                    legal=legal,
                    target_policy=target_policy,
                    target_risk=target_risk,
                    target_mine=target_mine,
                    target_certainty=target_certainty,
                    target_value=target_value,
                    rank_margin=self.rank_margin,
                    risk_temperature=self.risk_temperature,
                )
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * count
            examples += count
        return {name: value / max(examples, 1) for name, value in totals.items()}


def transfer_v5_weights(
    model: nn.Module,
    checkpoint: Path,
) -> tuple[str, int, dict]:
    """Load the complete v5 trunk and heads, excluding only new ranker controls."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload.get("config", {}))
    if config.get("architecture") != "constraint-posterior-v5":
        raise ValueError("constraint-ranker-v6 requires a v5 bootstrap checkpoint")
    incompatible = model.load_state_dict(payload["model"], strict=False)
    expected_missing = {
        "certainty_policy_scale",
        "preference_policy_limit",
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise ValueError(f"unexpected v5 transfer gaps: {sorted(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected v5 transfer parameters: {sorted(incompatible.unexpected_keys)}"
        )
    return (
        str(payload.get("checkpoint_id", "unknown")),
        len(payload["model"]),
        config,
    )


def _smoke(model: nn.Module, device: torch.device) -> dict[str, dict]:
    suite = get_suite("smoke")
    policy = TorchAxialPolicy(model, device)
    return {
        difficulty: evaluate_axial_policy(
            policy,
            *spec,
            seeds=suite.seeds,
            suite_name=suite.name,
            batch_size=32,
        ).to_dict()
        for difficulty, spec in DIFFICULTIES.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default="storage/runs/m6-r4/bootstrap.pt")
    parser.add_argument("--out", default="storage/runs/m6-r5/bootstrap.pt")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--boards", type=int, default=1_200)
    parser.add_argument("--states-per-round", type=int, default=16_000)
    parser.add_argument("--rehearsal-states", type=int, default=4_000)
    parser.add_argument("--validation-states", type=int, default=2_000)
    parser.add_argument("--epochs-per-round", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--dagger-weight", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--risk-temperature", type=float, default=0.5)
    parser.add_argument("--posterior-policy-weight", type=float, default=1.0)
    parser.add_argument("--certainty-policy-weight", type=float, default=4.0)
    parser.add_argument("--preference-policy-limit", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.rounds,
        args.boards,
        args.states_per_round,
        args.rehearsal_states,
        args.validation_states,
        args.epochs_per_round,
        args.batch_size,
        args.rollout_batch_size,
        args.dagger_weight,
    )
    if min(positive) < 1:
        raise ValueError("ranker generation and training arguments must be positive")
    if (
        min(
            args.lr,
            args.risk_temperature,
            args.posterior_policy_weight,
            args.certainty_policy_weight,
        )
        <= 0
    ):
        raise ValueError("ranker rates, temperatures, and scales must be positive")
    if args.rank_margin < 0:
        raise ValueError("rank margin must be non-negative")
    if not 0 <= args.preference_policy_limit <= 0.25:
        raise ValueError("preference policy limit must be in [0, 0.25]")


def _checkpoint_config(
    args: argparse.Namespace,
    source_config: dict,
) -> dict:
    experiment = ExperimentConfig(
        algorithm="ppo",
        architecture="constraint-ranker-v6",
        graph_rounds=int(source_config["graph_rounds"]),
        seed=args.seed,
        device=args.device,
        learning_rate=args.lr,
        width=int(source_config["width"]),
        blocks=int(source_config["blocks"]),
        mixed_precision=torch.device(args.device).type == "cuda",
    )
    return experiment.to_dict() | {
        "input_channels": int(source_config.get("input_channels", NUM_CHANNELS_V3)),
        "posterior_policy_weight": args.posterior_policy_weight,
        "certainty_policy_weight": args.certainty_policy_weight,
        "preference_policy_limit": args.preference_policy_limit,
        "ranker_teacher": {
            "rounds": args.rounds,
            "boards": args.boards,
            "states_per_round": args.states_per_round,
            "rehearsal_states": args.rehearsal_states,
            "validation_states": args.validation_states,
            "epochs_per_round": args.epochs_per_round,
            "dagger_weight": args.dagger_weight,
            "rank_margin": args.rank_margin,
            "risk_temperature": args.risk_temperature,
        },
    }


def _save_candidate(
    path: Path,
    *,
    trainer: RankerPretrainer,
    config: dict,
    run_id: str,
    parent_id: str,
    round_index: int,
    smoke: dict[str, dict],
    calibration: dict[str, float | int],
    train_metrics: dict[str, float],
    states: int,
) -> str:
    manifest = save_checkpoint(
        path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        config=config,
        run_id=run_id,
        git_sha="working-tree",
        episode=round_index * int(config["ranker_teacher"]["boards"]),
        global_step=round_index * states,
        current_eval=smoke["expert"],
        extra_state={
            "stage": "teacher-pretrain-constraint-ranker",
            "round": round_index,
            "parent_checkpoint_id": parent_id,
            "train": train_metrics,
            "calibration": calibration,
            "smoke": smoke,
            "amp_scaler": trainer.scaler.state_dict(),
        },
    )
    return manifest.checkpoint_id


def _append_metric(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    bootstrap = Path(args.bootstrap)
    source_payload = torch.load(
        bootstrap,
        map_location="cpu",
        weights_only=False,
    )
    source_config = dict(source_payload.get("config", {}))
    config = _checkpoint_config(args, source_config)
    model = build_policy_model(config)
    parent_id, transferred, _ = transfer_v5_weights(model, bootstrap)
    model.set_posterior_policy_weight(args.posterior_policy_weight)
    trainer = RankerPretrainer(
        model,
        device=device,
        learning_rate=args.lr,
        mixed_precision=device.type == "cuda",
        rank_margin=args.rank_margin,
        risk_temperature=args.risk_temperature,
    )
    print(
        f"bootstrap parent={parent_id} transferred_parameters={transferred}",
        flush=True,
    )

    target_shape = DIFFICULTIES["expert"][:2]
    rehearsal: dict[str, PaddedTeacherView] = {}
    for index, (difficulty, spec) in enumerate(DIFFICULTIES.items()):
        dataset = generate_component_balanced_teacher_dataset(
            *spec,
            boards=args.boards,
            seed_base=700_000_000 + index * 20_000_000 + args.seed * 10_000,
            max_states=args.rehearsal_states,
            oversample_factor=2,
        )
        rehearsal[difficulty] = PaddedTeacherView(dataset, target_shape)
        print(
            f"rehearsal difficulty={difficulty} states={len(dataset)}",
            flush=True,
        )

    validation = generate_component_balanced_teacher_dataset(
        *DIFFICULTIES["expert"],
        boards=args.boards,
        seed_base=780_000_000 + args.seed * 10_000,
        max_states=args.validation_states,
        oversample_factor=1,
    )
    validation_view = PaddedTeacherView(validation, target_shape)
    run_id = f"pretrain-ranker-{uuid.uuid4()}"
    output = Path(args.out)
    metrics_path = output.with_name("ranker.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("", encoding="utf-8")
    baseline_smoke = _smoke(trainer.model, device)
    baseline_calibration = ranker_metrics(
        trainer.model,
        validation_view,
        device=device,
        batch_size=args.batch_size,
    )
    best_rate = float(baseline_smoke["expert"]["win_rate"])
    checkpoint_id = _save_candidate(
        output,
        trainer=trainer,
        config=config,
        run_id=run_id,
        parent_id=parent_id,
        round_index=0,
        smoke=baseline_smoke,
        calibration=baseline_calibration,
        train_metrics={},
        states=0,
    )
    baseline_record = {
        "type": "round",
        "round": 0,
        "checkpoint_id": checkpoint_id,
        "calibration": baseline_calibration,
        "smoke": baseline_smoke,
    }
    _append_metric(metrics_path, baseline_record)
    print(json.dumps(baseline_record, sort_keys=True), flush=True)

    for round_index in range(1, args.rounds + 1):
        dagger = generate_on_policy_teacher_dataset(
            trainer.model,
            device=device,
            rows=DIFFICULTIES["expert"][0],
            cols=DIFFICULTIES["expert"][1],
            mines=DIFFICULTIES["expert"][2],
            boards=args.boards,
            seed_base=800_000_000 + round_index * 1_000_000 + args.seed * 10_000,
            max_states=args.states_per_round,
            batch_size=args.rollout_batch_size,
        )
        dagger_view = PaddedTeacherView(dagger, target_shape)
        mixed = ConcatDataset(
            [
                rehearsal["beginner"],
                rehearsal["intermediate"],
                rehearsal["expert"],
                *([dagger_view] * args.dagger_weight),
            ]
        )
        train_metrics: dict[str, float] = {}
        for epoch in range(1, args.epochs_per_round + 1):
            train_metrics = trainer.train_epoch(mixed, args.batch_size)
            epoch_record = {
                "type": "epoch",
                "round": round_index,
                "epoch": epoch,
                "states": len(mixed),
                "train": train_metrics,
            }
            _append_metric(metrics_path, epoch_record)
            print(json.dumps(epoch_record, sort_keys=True), flush=True)
        calibration = ranker_metrics(
            trainer.model,
            validation_view,
            device=device,
            batch_size=args.batch_size,
        )
        smoke = _smoke(trainer.model, device)
        rate = float(smoke["expert"]["win_rate"])
        last_checkpoint_id = _save_candidate(
            output.with_name("last.pt"),
            trainer=trainer,
            config=config,
            run_id=run_id,
            parent_id=parent_id,
            round_index=round_index,
            smoke=smoke,
            calibration=calibration,
            train_metrics=train_metrics,
            states=len(mixed),
        )
        record = {
            "type": "round",
            "round": round_index,
            "dagger_states": len(dagger),
            "train": train_metrics,
            "calibration": calibration,
            "smoke": smoke,
            "saved": rate >= best_rate,
            "last_checkpoint_id": last_checkpoint_id,
        }
        if rate >= best_rate:
            best_rate = rate
            checkpoint_id = _save_candidate(
                output,
                trainer=trainer,
                config=config,
                run_id=run_id,
                parent_id=parent_id,
                round_index=round_index,
                smoke=smoke,
                calibration=calibration,
                train_metrics=train_metrics,
                states=len(mixed),
            )
            record["checkpoint_id"] = checkpoint_id
        _append_metric(metrics_path, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        del mixed, dagger_view, dagger
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_record = {
        "type": "complete",
        "best_expert_smoke": best_rate,
        "checkpoint": str(output),
        "checkpoint_id": checkpoint_id,
    }
    _append_metric(metrics_path, final_record)
    print(json.dumps(final_record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
