"""Solver-teacher supervised pretraining for the M4-R2 spatial model."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from trainer.config import ExperimentConfig
from trainer.data import TeacherDataset, generate_teacher_dataset
from trainer.encoding_v2 import NUM_CHANNELS_V2
from trainer.env import DIFFICULTIES
from trainer.models import SpatialPolicyValueNet
from trainer.storage import save_checkpoint


class TeacherPretrainer:
    def __init__(
        self,
        model: SpatialPolicyValueNet,
        *,
        device: torch.device,
        learning_rate: float,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )

    def train_epoch(self, dataset: TeacherDataset, batch_size: int) -> dict[str, float]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        totals = {
            "loss": 0.0,
            "policy": 0.0,
            "risk": 0.0,
            "certainty": 0.0,
            "value": 0.0,
        }
        batches = 0
        self.model.train()
        for batch in loader:
            state = batch["state"].to(self.device)
            legal = batch["legal_mask"].to(self.device).bool()
            target_policy = batch["policy_target"].to(self.device)
            target_risk = batch["risk_target"].to(self.device)
            target_mine = batch["mine_target"].to(self.device)
            target_certainty = batch["certainty_target"].to(self.device)
            target_value = batch["value_target"].to(self.device)

            output = self.model(state, legal)
            log_policy = F.log_softmax(output["policy_logits"], dim=1)
            policy_loss = -(target_policy.flatten(start_dim=1) * log_policy).sum(dim=1).mean()

            legal_flat = legal.flatten(start_dim=1)
            risk_logits = output["risk_logits"][legal_flat]
            risk_target = target_risk.flatten(start_dim=1)[legal_flat]
            mine_target = target_mine.flatten(start_dim=1)[legal_flat]
            risk_loss = F.binary_cross_entropy_with_logits(risk_logits, risk_target)
            mine_loss = F.binary_cross_entropy_with_logits(risk_logits, mine_target)
            certainty_loss = F.cross_entropy(
                output["certainty_logits"],
                target_certainty,
                ignore_index=-100,
            )
            value_loss = F.mse_loss(output["value"], target_value)
            loss = (
                policy_loss
                + risk_loss
                + 0.25 * mine_loss
                + 0.25 * certainty_loss
                + 0.1 * value_loss
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            values = {
                "loss": loss,
                "policy": policy_loss,
                "risk": risk_loss,
                "certainty": certainty_loss,
                "value": value_loss,
            }
            for name, value in values.items():
                totals[name] += float(value.detach().item())
            batches += 1
        return {name: value / max(batches, 1) for name, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", choices=[*DIFFICULTIES, "all"], default="all")
    parser.add_argument("--boards", type=int, default=300)
    parser.add_argument("--max-states", type=int, default=3_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="storage/checkpoints/m4-r2/pretrained.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = SpatialPolicyValueNet(NUM_CHANNELS_V2, args.width, args.blocks)
    trainer = TeacherPretrainer(model, device=device, learning_rate=args.lr)
    difficulties = list(DIFFICULTIES) if args.difficulty == "all" else [args.difficulty]
    datasets = {}
    for index, difficulty in enumerate(difficulties):
        print(
            f"generate difficulty={difficulty} boards<={args.boards} states<={args.max_states}",
            flush=True,
        )
        dataset = generate_teacher_dataset(
            *DIFFICULTIES[difficulty],
            boards=args.boards,
            seed_base=100_000_000 + index * 10_000_000 + args.seed * 100_000,
            max_states=args.max_states,
        )
        datasets[difficulty] = dataset
        print(f"generated difficulty={difficulty} states={len(dataset)}", flush=True)

    for epoch in range(args.epochs):
        for difficulty, dataset in datasets.items():
            metrics = trainer.train_epoch(dataset, args.batch_size)
            print(
                f"epoch={epoch + 1} difficulty={difficulty} states={len(dataset)} "
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
    )
    save_checkpoint(
        Path(args.out),
        model=model,
        optimizer=trainer.optimizer,
        config=config.to_dict(),
        run_id=f"pretrain-{uuid.uuid4()}",
        git_sha="working-tree",
        episode=args.boards * len(difficulties),
        global_step=sum(len(dataset) for dataset in datasets.values()),
        extra_state={"stage": "teacher-pretrain", "difficulties": difficulties},
    )


if __name__ == "__main__":
    main()
