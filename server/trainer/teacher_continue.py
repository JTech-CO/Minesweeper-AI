"""Generic continuation CLI for lineage-checked teacher checkpoints."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import torch

from trainer.config import ExperimentConfig
from trainer.data import generate_teacher_dataset
from trainer.encoding_v2 import NUM_CHANNELS_V2
from trainer.env import DIFFICULTIES
from trainer.models import SpatialPolicyValueNet
from trainer.pretrain import TeacherPretrainer
from trainer.storage import load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-seed", type=int, required=True)
    parser.add_argument("--parent-lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--boards", type=int, default=2_000)
    parser.add_argument("--max-states", type=int, default=12_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = SpatialPolicyValueNet(NUM_CHANNELS_V2, args.width, args.blocks)
    pretrainer = TeacherPretrainer(model, device=device, learning_rate=args.lr)
    parent_config = ExperimentConfig(
        seed=args.parent_seed,
        learning_rate=args.parent_lr,
        width=args.width,
        blocks=args.blocks,
    )
    load_checkpoint(
        Path(args.input),
        model=model,
        expected_config=parent_config.to_dict(),
        map_location=device,
    )

    datasets = {}
    for index, (difficulty, spec) in enumerate(DIFFICULTIES.items()):
        print(f"generate difficulty={difficulty} states<={args.max_states}", flush=True)
        dataset = generate_teacher_dataset(
            *spec,
            boards=args.boards,
            seed_base=300_000_000 + index * 10_000_000 + args.seed * 100_000,
            max_states=args.max_states,
        )
        datasets[difficulty] = dataset
        print(f"generated difficulty={difficulty} states={len(dataset)}", flush=True)

    for epoch in range(args.epochs):
        order = list(DIFFICULTIES)
        if epoch % 2:
            order.reverse()
        for difficulty in order:
            metrics = pretrainer.train_epoch(datasets[difficulty], args.batch_size)
            print(
                f"epoch={epoch + 1} difficulty={difficulty} "
                f"loss={metrics['loss']:.4f} policy={metrics['policy']:.4f} "
                f"risk={metrics['risk']:.4f}",
                flush=True,
            )

    config = ExperimentConfig(
        seed=args.seed,
        learning_rate=args.lr,
        width=args.width,
        blocks=args.blocks,
    )
    save_checkpoint(
        Path(args.output),
        model=model,
        optimizer=pretrainer.optimizer,
        config=config.to_dict(),
        run_id=f"teacher-continue-{uuid.uuid4()}",
        git_sha="working-tree",
        episode=args.boards * len(DIFFICULTIES),
        global_step=sum(len(dataset) for dataset in datasets.values()),
        extra_state={"stage": "teacher-continue", "parent": args.input},
    )


if __name__ == "__main__":
    main()
