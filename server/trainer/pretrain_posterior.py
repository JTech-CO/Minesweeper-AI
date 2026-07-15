"""M6-R4 constraint-posterior-v5 teacher distillation."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from trainer.config import ExperimentConfig
from trainer.data import TeacherDataset
from trainer.data.hard_teacher import generate_component_balanced_teacher_dataset
from trainer.encoding_v3 import NUM_CHANNELS_V3
from trainer.env import DIFFICULTIES
from trainer.models import build_policy_model
from trainer.pretrain import TeacherPretrainer
from trainer.storage import load_checkpoint, save_checkpoint


class PaddedTeacherView(Dataset):
    """Pad samples on demand instead of duplicating the in-memory dataset."""

    def __init__(
        self,
        dataset: TeacherDataset,
        target_shape: tuple[int, int],
    ) -> None:
        self.dataset = dataset
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        rows, cols = item["legal_mask"].shape
        target_rows, target_cols = self.target_shape
        padding = (0, target_cols - cols, 0, target_rows - rows)
        if min(padding) < 0:
            raise ValueError("target shape cannot crop a teacher sample")
        return {
            "state": F.pad(item["state"], padding),
            "legal_mask": F.pad(item["legal_mask"], padding),
            "policy_target": F.pad(item["policy_target"], padding),
            "risk_target": F.pad(item["risk_target"], padding),
            "mine_target": F.pad(item["mine_target"], padding),
            "certainty_target": F.pad(
                item["certainty_target"],
                padding,
                value=-100,
            ),
            "value_target": item["value_target"],
        }


def transfer_v4_weights(model: torch.nn.Module, checkpoint: Path) -> tuple[str, int]:
    """Transfer only shape-compatible visible-policy parameters from v4."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload["model"]
    target = model.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    for target_name, target_value in target.items():
        source_name = target_name
        if target_name.startswith("stem_conv."):
            source_name = target_name.replace("stem_conv.", "stem.0.", 1)
        elif target_name.startswith("stem_norm."):
            source_name = target_name.replace("stem_norm.", "stem.1.", 1)
        elif target_name.startswith("preference_head."):
            source_name = target_name.replace("preference_head.", "policy_head.", 1)
        elif target_name.startswith(("posterior_head.", "mine_head.")):
            source_name = target_name.split(".", 1)[1]
            source_name = f"risk_head.{source_name}"
        candidate = source.get(source_name)
        if candidate is not None and candidate.shape == target_value.shape:
            transferred[target_name] = candidate

    missing, unexpected = model.load_state_dict(transferred, strict=False)
    if unexpected:
        raise ValueError(f"unexpected v4 transfer parameters: {unexpected}")
    required = {
        "stem_conv.weight",
        "preference_head.weight",
        "posterior_head.weight",
        "mine_head.weight",
    }
    if not required <= transferred.keys():
        absent = sorted(required - transferred.keys())
        raise ValueError(f"v4 checkpoint lacks required transfer parameters: {absent}")

    with torch.no_grad():
        model.constraint_graph.clue_scale.zero_()
        model.constraint_graph.hidden_scale.zero_()
        for block in model.blocks:
            block.scale.fill_(3.0)
    return str(payload.get("checkpoint_id", "unknown")), len(transferred)


def bootstrap_teacher_weights(
    model: torch.nn.Module,
    checkpoint: Path,
) -> tuple[str, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = str(payload.get("config", {}).get("architecture", ""))
    if architecture == "constraint-posterior-v5":
        model.load_state_dict(payload["model"])
        return str(payload.get("checkpoint_id", "unknown")), len(payload["model"])
    return transfer_v4_weights(model, checkpoint)


@torch.no_grad()
def posterior_metrics(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    nll_sum = 0.0
    brier_sum = 0.0
    legal_count = 0
    predicted_safe = 0
    correct_safe = 0
    for batch in loader:
        state = batch["state"].to(device)
        legal = batch["legal_mask"].to(device).bool()
        target_risk = batch["risk_target"].to(device)
        target_mine = batch["mine_target"].to(device)
        output = model(state, legal)
        legal_flat = legal.flatten(start_dim=1)
        logits = output["posterior_logits"][legal_flat]
        risks = target_risk.flatten(start_dim=1)[legal_flat]
        probabilities = torch.sigmoid(logits)
        nll_sum += float(
            F.binary_cross_entropy_with_logits(
                logits,
                risks,
                reduction="sum",
            ).item()
        )
        brier_sum += float((probabilities - risks).square().sum().item())
        legal_count += int(legal_flat.sum().item())

        predicted = output["certainty_logits"].argmax(dim=1)
        safe = (predicted == 1) & legal
        predicted_safe += int(safe.sum().item())
        correct_safe += int(((target_mine == 0) & safe).sum().item())

    denominator = max(legal_count, 1)
    return {
        "posterior_nll": nll_sum / denominator,
        "posterior_brier": brier_sum / denominator,
        "certain_safe_precision": correct_safe / max(predicted_safe, 1),
        "certain_safe_predictions": predicted_safe,
        "legal_cells": legal_count,
    }


def _teacher_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "boards": args.boards,
        "beginner_states": args.beginner_states,
        "intermediate_states": args.intermediate_states,
        "expert_states": args.expert_states,
        "expert_weight": args.expert_weight,
        "oversample_factor": args.oversample_factor,
        "batch_size": args.batch_size,
        "validation_states": args.validation_states,
        "mixed_precision": args.device == "cuda",
        "posterior_policy_weight": args.posterior_policy_weight,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=4_000)
    parser.add_argument("--beginner-states", type=int, default=6_000)
    parser.add_argument("--intermediate-states", type=int, default=10_000)
    parser.add_argument("--expert-states", type=int, default=20_000)
    parser.add_argument("--expert-weight", type=int, default=3)
    parser.add_argument("--oversample-factor", type=int, default=2)
    parser.add_argument("--validation-states", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--graph-rounds", type=int, default=8)
    parser.add_argument("--posterior-policy-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--bootstrap",
        default="storage/runs/m4-r3-graph/validated.pt",
    )
    parser.add_argument(
        "--out",
        default="storage/runs/m6-r4/bootstrap.pt",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.boards,
        args.beginner_states,
        args.intermediate_states,
        args.expert_states,
        args.expert_weight,
        args.oversample_factor,
        args.validation_states,
        args.epochs,
        args.batch_size,
    ) < 1:
        raise ValueError("teacher generation and training arguments must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = ExperimentConfig(
        algorithm="ppo",
        architecture="constraint-posterior-v5",
        graph_rounds=args.graph_rounds,
        seed=args.seed,
        device=args.device,
        learning_rate=args.lr,
        width=args.width,
        blocks=args.blocks,
    )
    checkpoint_config = config.to_dict() | {
        "input_channels": NUM_CHANNELS_V3,
        "posterior_policy_weight": args.posterior_policy_weight,
        "teacher": _teacher_config(args),
    }
    model = build_policy_model(checkpoint_config)
    output = Path(args.out)
    trainer = TeacherPretrainer(
        model,
        device=device,
        learning_rate=args.lr,
        mixed_precision=device.type == "cuda",
    )

    if args.resume:
        payload, _ = load_checkpoint(
            output,
            model=model,
            optimizer=trainer.optimizer,
            expected_config=checkpoint_config,
            restore_rng=True,
            map_location=device,
        )
        extra = payload.get("extra_state", {})
        start_epoch = int(extra.get("epoch", 0))
        parent_id = str(extra.get("parent_checkpoint_id", "unknown"))
        scaler_state = extra.get("amp_scaler")
        if scaler_state:
            trainer.scaler.load_state_dict(scaler_state)
        run_id = str(payload["run_id"])
        print(
            f"resume epoch={start_epoch} checkpoint={payload['checkpoint_id']}",
            flush=True,
        )
    else:
        parent_id, transferred = bootstrap_teacher_weights(
            model,
            Path(args.bootstrap),
        )
        model.set_posterior_policy_weight(args.posterior_policy_weight)
        model.to(device)
        start_epoch = 0
        run_id = f"pretrain-posterior-{uuid.uuid4()}"
        print(
            f"bootstrap parent={parent_id} transferred_parameters={transferred}",
            flush=True,
        )

    target_shape = DIFFICULTIES["expert"][:2]
    state_limits = {
        "beginner": args.beginner_states,
        "intermediate": args.intermediate_states,
        "expert": args.expert_states,
    }
    views: dict[str, PaddedTeacherView] = {}
    counts: dict[str, int] = {}
    for index, (difficulty, spec) in enumerate(DIFFICULTIES.items()):
        limit = state_limits[difficulty]
        print(
            f"generate difficulty={difficulty} boards<={args.boards} "
            f"states<={limit}",
            flush=True,
        )
        dataset = generate_component_balanced_teacher_dataset(
            *spec,
            boards=args.boards,
            seed_base=900_000_000 + index * 10_000_000 + args.seed * 100_000,
            max_states=limit,
            oversample_factor=args.oversample_factor,
        )
        views[difficulty] = PaddedTeacherView(dataset, target_shape)
        counts[difficulty] = len(dataset)
        print(f"generated difficulty={difficulty} states={len(dataset)}", flush=True)

    validation = generate_component_balanced_teacher_dataset(
        *DIFFICULTIES["expert"],
        boards=args.boards,
        seed_base=990_000_000 + args.seed * 100_000,
        max_states=args.validation_states,
        oversample_factor=1,
    )
    validation_view = PaddedTeacherView(validation, target_shape)
    mixed = ConcatDataset(
        [
            views["beginner"],
            views["intermediate"],
            *([views["expert"]] * args.expert_weight),
        ]
    )

    for epoch in range(start_epoch, args.epochs):
        metrics = trainer.train_epoch(mixed, args.batch_size)
        calibration = posterior_metrics(
            model,
            validation_view,
            device=device,
            batch_size=args.batch_size,
        )
        record = {
            "epoch": epoch + 1,
            "mixed_states": len(mixed),
            "train": metrics,
            "validation": calibration,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        manifest = save_checkpoint(
            output,
            model=model,
            optimizer=trainer.optimizer,
            config=checkpoint_config,
            run_id=run_id,
            git_sha="working-tree",
            episode=(epoch + 1) * args.boards * len(DIFFICULTIES),
            global_step=(epoch + 1) * len(mixed),
            current_eval=calibration,
            extra_state={
                "stage": "teacher-pretrain-constraint-posterior",
                "epoch": epoch + 1,
                "parent_checkpoint_id": parent_id,
                "mixed_difficulties": True,
                "state_counts": counts,
                "metrics": metrics,
                "calibration": calibration,
                "amp_scaler": trainer.scaler.state_dict(),
            },
        )
        print(
            f"checkpoint epoch={epoch + 1} id={manifest.checkpoint_id}",
            flush=True,
        )


if __name__ == "__main__":
    main()
