"""Evaluate an R2-A axial checkpoint on immutable canonical seeds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from trainer.env import DIFFICULTIES
from trainer.evaluation.axial import TorchAxialPolicy, evaluate_axial_policy
from trainer.evaluation.suites import get_suite
from trainer.models import build_policy_model
from trainer.storage import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, required=True)
    parser.add_argument("--suite", choices=("smoke", "validation", "test"), default="validation")
    parser.add_argument("--games", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--risk-weight", type=float, default=0.0)
    parser.add_argument("--certainty-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload.get("config", {}))
    config.setdefault("width", args.width)
    config.setdefault("blocks", args.blocks)
    model = build_policy_model(config)
    _, manifest = load_checkpoint(checkpoint, model=model, map_location=device)
    model.to(device)
    suite = get_suite(args.suite, games=args.games)
    result = evaluate_axial_policy(
        TorchAxialPolicy(
            model,
            device,
            risk_weight=args.risk_weight,
            certainty_weight=args.certainty_weight,
        ),
        *DIFFICULTIES[args.difficulty],
        seeds=suite.seeds,
        suite_name=suite.name,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {"checkpoint_id": manifest.checkpoint_id, **asdict(result)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

