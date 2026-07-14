"""M4-R2-B mixed-difficulty constraint-graph teacher pretraining."""

from __future__ import annotations

import argparse
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from trainer.config import ExperimentConfig
from trainer.data import TeacherDataset, TeacherSample
from trainer.data.hard_teacher import generate_component_balanced_teacher_dataset
from trainer.encoding_v3 import NUM_CHANNELS_V3
from trainer.env import DIFFICULTIES
from trainer.models import build_policy_model
from trainer.pretrain import TeacherPretrainer
from trainer.storage import load_checkpoint, save_checkpoint


def _pad_plane(
    value: np.ndarray,
    target_shape: tuple[int, int],
    *,
    fill: int | float = 0,
) -> np.ndarray:
    rows, cols = value.shape[-2:]
    target_rows, target_cols = target_shape
    shape = (*value.shape[:-2], target_rows, target_cols)
    padded = np.full(shape, fill, dtype=value.dtype)
    padded[..., :rows, :cols] = value
    return padded


def _pad_teacher_dataset(
    dataset: TeacherDataset,
    target_shape: tuple[int, int],
) -> TeacherDataset:
    samples: list[TeacherSample] = []
    for sample in dataset.samples:
        samples.append(
            replace(
                sample,
                state=_pad_plane(sample.state, target_shape),
                legal_mask=_pad_plane(sample.legal_mask, target_shape),
                policy_target=_pad_plane(sample.policy_target, target_shape),
                risk_target=_pad_plane(sample.risk_target, target_shape),
                mine_target=_pad_plane(sample.mine_target, target_shape),
                certainty_target=_pad_plane(
                    sample.certainty_target,
                    target_shape,
                    fill=-100,
                ),
            )
        )
    return TeacherDataset(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=1_500)
    parser.add_argument("--max-states", type=int, default=20_000)
    parser.add_argument("--oversample-factor", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--graph-rounds", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--bootstrap",
        default="storage/checkpoints/m4-r2/axial-pretrained.pt",
    )
    parser.add_argument(
        "--out",
        default="storage/checkpoints/m4-r2/graph-pretrained.pt",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def transfer_axial_weights(model: torch.nn.Module, path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload["model"]
    target = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in target and target[name].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected or any(not name.startswith("constraint_graph.") for name in missing):
        raise ValueError(
            f"incompatible axial bootstrap: missing={missing}, unexpected={unexpected}"
        )
    return str(payload.get("checkpoint_id", "unknown"))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = ExperimentConfig(
        algorithm="ppo",
        architecture="constraint-graph-v4",
        graph_rounds=args.graph_rounds,
        seed=args.seed,
        learning_rate=args.lr,
        width=args.width,
        blocks=args.blocks,
    )
    model = build_policy_model(config.to_dict() | {"input_channels": NUM_CHANNELS_V3})
    output = Path(args.out)
    checkpoint_config = config.to_dict() | {
        "input_channels": NUM_CHANNELS_V3,
        "teacher": {
            "boards": args.boards,
            "max_states": args.max_states,
            "oversample_factor": args.oversample_factor,
            "batch_size": args.batch_size,
        },
    }
    if args.resume:
        if not output.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {output}")
        trainer = TeacherPretrainer(model, device=device, learning_rate=args.lr)
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
        run_id = str(payload["run_id"])
        print(f"resume epoch={start_epoch} checkpoint={payload['checkpoint_id']}", flush=True)
    else:
        parent_id = transfer_axial_weights(model, Path(args.bootstrap))
        trainer = TeacherPretrainer(model, device=device, learning_rate=args.lr)
        start_epoch = 0
        run_id = f"pretrain-graph-{uuid.uuid4()}"

    datasets = []
    counts: dict[str, int] = {}
    target_shape = DIFFICULTIES["expert"][:2]
    for index, (difficulty, spec) in enumerate(DIFFICULTIES.items()):
        print(
            f"generate mixed difficulty={difficulty} boards<={args.boards} "
            f"states<={args.max_states}",
            flush=True,
        )
        dataset = generate_component_balanced_teacher_dataset(
            *spec,
            boards=args.boards,
            seed_base=600_000_000 + index * 10_000_000 + args.seed * 100_000,
            max_states=args.max_states,
            oversample_factor=args.oversample_factor,
        )
        datasets.append(_pad_teacher_dataset(dataset, target_shape))
        counts[difficulty] = len(dataset)
        print(f"generated difficulty={difficulty} states={len(dataset)}", flush=True)

    mixed = ConcatDataset(datasets)
    for epoch in range(start_epoch, args.epochs):
        metrics = trainer.train_epoch(mixed, args.batch_size)
        print(
            f"epoch={epoch + 1} mixed_states={len(mixed)} "
            f"loss={metrics['loss']:.4f} policy={metrics['policy']:.4f} "
            f"risk={metrics['risk']:.4f}",
            flush=True,
        )
        manifest = save_checkpoint(
            output,
            model=model,
            optimizer=trainer.optimizer,
            config=checkpoint_config,
            run_id=run_id,
            git_sha="working-tree",
            episode=(epoch + 1) * args.boards * len(DIFFICULTIES),
            global_step=(epoch + 1) * sum(counts.values()),
            extra_state={
                "stage": "teacher-pretrain-constraint-graph",
                "epoch": epoch + 1,
                "parent_checkpoint_id": parent_id,
                "mixed_difficulties": True,
                "state_counts": counts,
                "metrics": metrics,
            },
        )
        print(
            f"checkpoint epoch={epoch + 1} id={manifest.checkpoint_id}",
            flush=True,
        )


if __name__ == "__main__":
    main()
