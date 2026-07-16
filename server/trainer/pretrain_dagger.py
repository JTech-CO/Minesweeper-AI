"""M6-R4-3 on-policy solver distillation for constraint-posterior-v5."""

from __future__ import annotations

import argparse
import gc
import json
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from trainer.data import generate_on_policy_teacher_dataset
from trainer.data.hard_teacher import generate_component_balanced_teacher_dataset
from trainer.env import DIFFICULTIES
from trainer.evaluation.axial import TorchAxialPolicy, evaluate_axial_policy
from trainer.evaluation.suites import get_suite
from trainer.models import build_policy_model
from trainer.pretrain import TeacherPretrainer
from trainer.pretrain_posterior import PaddedTeacherView, posterior_metrics
from trainer.storage import load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="storage/runs/m6-r4/bootstrap.pt")
    parser.add_argument("--collector", default="storage/runs/m6-r4/last.pt")
    parser.add_argument("--out", default="storage/runs/m6-r4/bootstrap-dagger.pt")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--boards", type=int, default=500)
    parser.add_argument("--states-per-round", type=int, default=12_000)
    parser.add_argument("--rehearsal-states", type=int, default=3_000)
    parser.add_argument("--validation-states", type=int, default=1_000)
    parser.add_argument("--epochs-per-round", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--dagger-weight", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def _load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    model = build_policy_model(config)
    _, manifest = load_checkpoint(path, model=model, map_location=device)
    model.to(device)
    return model, config, manifest.checkpoint_id


def _smoke(model: torch.nn.Module, device: torch.device) -> dict[str, dict]:
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


def main() -> None:
    args = parse_args()
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
    if min(positive) < 1 or args.lr <= 0:
        raise ValueError("DAgger training arguments must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    target_shape = DIFFICULTIES["expert"][:2]

    student_path = Path(args.student)
    collector_path = Path(args.collector)
    student_payload = torch.load(student_path, map_location="cpu", weights_only=False)
    checkpoint_config = dict(student_payload["config"])
    if checkpoint_config.get("architecture") != "constraint-posterior-v5":
        raise ValueError("DAgger requires a constraint-posterior-v5 student")
    student = build_policy_model(checkpoint_config)
    _, student_manifest = load_checkpoint(student_path, model=student)

    rehearsal = {}
    for index, (difficulty, spec) in enumerate(DIFFICULTIES.items()):
        dataset = generate_component_balanced_teacher_dataset(
            *spec,
            boards=args.boards,
            seed_base=850_000_000 + index * 10_000_000 + args.seed * 100_000,
            max_states=args.rehearsal_states,
            oversample_factor=1,
        )
        rehearsal[difficulty] = PaddedTeacherView(dataset, target_shape)
        print(f"rehearsal difficulty={difficulty} states={len(dataset)}", flush=True)

    validation = generate_component_balanced_teacher_dataset(
        *DIFFICULTIES["expert"],
        boards=args.boards,
        seed_base=890_000_000 + args.seed * 100_000,
        max_states=args.validation_states,
        oversample_factor=1,
    )
    validation_view = PaddedTeacherView(validation, target_shape)
    trainer: TeacherPretrainer | None = None
    collector_id = ""
    run_id = f"pretrain-dagger-{uuid.uuid4()}"
    output = Path(args.out)

    for round_index in range(args.rounds):
        if round_index == 0:
            collector, _, collector_id = _load_model(collector_path, device)
        else:
            assert trainer is not None
            collector = trainer.model
            collector_id = f"round-{round_index}"
        dagger = generate_on_policy_teacher_dataset(
            collector,
            device=device,
            rows=DIFFICULTIES["expert"][0],
            cols=DIFFICULTIES["expert"][1],
            mines=DIFFICULTIES["expert"][2],
            boards=args.boards,
            seed_base=870_000_000 + round_index * 1_000_000 + args.seed * 10_000,
            max_states=args.states_per_round,
            batch_size=args.rollout_batch_size,
        )
        print(
            f"dagger round={round_index + 1} collector={collector_id} states={len(dagger)}",
            flush=True,
        )
        if round_index == 0:
            del collector
            if device.type == "cuda":
                torch.cuda.empty_cache()
            trainer = TeacherPretrainer(
                student,
                device=device,
                learning_rate=args.lr,
                mixed_precision=device.type == "cuda",
            )
        assert trainer is not None
        dagger_view = PaddedTeacherView(dagger, target_shape)
        mixed = ConcatDataset(
            [
                rehearsal["beginner"],
                rehearsal["intermediate"],
                rehearsal["expert"],
                *([dagger_view] * args.dagger_weight),
            ]
        )
        metrics = {}
        for epoch in range(args.epochs_per_round):
            metrics = trainer.train_epoch(mixed, args.batch_size)
            print(
                json.dumps(
                    {
                        "round": round_index + 1,
                        "epoch": epoch + 1,
                        "train": metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        calibration = posterior_metrics(
            trainer.model,
            validation_view,
            device=device,
            batch_size=args.batch_size,
        )
        smoke = _smoke(trainer.model, device)
        record = {
            "round": round_index + 1,
            "collector_checkpoint_id": collector_id,
            "dagger_states": len(dagger),
            "train": metrics,
            "calibration": calibration,
            "smoke": smoke,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        manifest = save_checkpoint(
            output,
            model=trainer.model,
            optimizer=trainer.optimizer,
            config=checkpoint_config,
            run_id=run_id,
            git_sha="working-tree",
            episode=(round_index + 1) * args.boards,
            global_step=(round_index + 1) * len(mixed),
            current_eval=smoke["expert"],
            extra_state={
                "stage": "teacher-pretrain-dagger",
                "round": round_index + 1,
                "student_checkpoint_id": student_manifest.checkpoint_id,
                **record,
                "amp_scaler": trainer.scaler.state_dict(),
            },
        )
        print(
            f"checkpoint round={round_index + 1} id={manifest.checkpoint_id}",
            flush=True,
        )
        del mixed, dagger_view, dagger
        gc.collect()


if __name__ == "__main__":
    main()
