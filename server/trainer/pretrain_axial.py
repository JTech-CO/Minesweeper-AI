"""M4-R2-A hard-state pretraining for the axial policy."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import torch

from trainer.config import ExperimentConfig
from trainer.data.hard_teacher import generate_component_balanced_teacher_dataset
from trainer.encoding_v3 import NUM_CHANNELS_V3
from trainer.env import DIFFICULTIES
from trainer.models.policy_value_axial import AxialPolicyValueNet
from trainer.pretrain import TeacherPretrainer
from trainer.storage import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=[*DIFFICULTIES, "all"], default="all")
    parser.add_argument("--boards", type=int, default=1_500)
    parser.add_argument("--max-states", type=int, default=20_000)
    parser.add_argument("--oversample-factor", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="storage/checkpoints/m4-r2/axial-pretrained.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = AxialPolicyValueNet(NUM_CHANNELS_V3, args.width, args.blocks)
    trainer = TeacherPretrainer(model, device=device, learning_rate=args.lr)
    difficulties = list(DIFFICULTIES) if args.difficulty == "all" else [args.difficulty]
    datasets = {}
    for index, difficulty in enumerate(difficulties):
        print(
            f"generate balanced difficulty={difficulty} boards<={args.boards} "
            f"states<={args.max_states}",
            flush=True,
        )
        dataset = generate_component_balanced_teacher_dataset(
            *DIFFICULTIES[difficulty],
            boards=args.boards,
            seed_base=300_000_000 + index * 10_000_000 + args.seed * 100_000,
            max_states=args.max_states,
            oversample_factor=args.oversample_factor,
        )
        datasets[difficulty] = dataset
        print(f"generated difficulty={difficulty} states={len(dataset)}", flush=True)

    for epoch in range(args.epochs):
        order = difficulties if epoch % 2 == 0 else list(reversed(difficulties))
        for difficulty in order:
            metrics = trainer.train_epoch(datasets[difficulty], args.batch_size)
            print(
                f"epoch={epoch + 1} difficulty={difficulty} "
                f"loss={metrics['loss']:.4f} policy={metrics['policy']:.4f} "
                f"risk={metrics['risk']:.4f}",
                flush=True,
            )

    config = ExperimentConfig(
        algorithm="ppo",
        seed=args.seed,
        learning_rate=args.lr,
        width=args.width,
        blocks=args.blocks,
    ).to_dict()
    config.update({"architecture": "axial-v3", "input_channels": NUM_CHANNELS_V3})
    save_checkpoint(
        Path(args.out),
        model=model,
        optimizer=trainer.optimizer,
        config=config,
        run_id=f"pretrain-axial-{uuid.uuid4()}",
        git_sha="working-tree",
        episode=args.boards * len(difficulties),
        global_step=sum(len(dataset) for dataset in datasets.values()),
        extra_state={
            "stage": "teacher-pretrain-axial",
            "difficulties": difficulties,
            "balanced": True,
        },
    )


if __name__ == "__main__":
    main()

