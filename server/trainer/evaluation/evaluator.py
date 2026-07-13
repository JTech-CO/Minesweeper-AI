"""Policy-only evaluation on immutable seed suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from trainer.encoding import encode
from trainer.env import MinesweeperEnv
from trainer.evaluation.metrics import wilson_interval


class Policy(Protocol):
    def act(self, state: np.ndarray, legal_mask: np.ndarray) -> int: ...


@dataclass(frozen=True)
class EvaluationResult:
    suite: str
    games: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    avg_steps: float
    seed_base: int
    opening: str

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_policy(
    policy: Policy,
    rows: int,
    cols: int,
    mines: int,
    *,
    seeds: tuple[int, ...],
    suite_name: str,
    opening: str = "center",
) -> EvaluationResult:
    if not seeds:
        raise ValueError("evaluation requires at least one seed")
    if opening not in {"center", "policy"}:
        raise ValueError("opening must be 'center' or 'policy'")

    wins = 0
    total_steps = 0
    center = (rows // 2) * cols + (cols // 2)
    for seed in seeds:
        env = MinesweeperEnv(rows, cols, mines, seed=seed, first_click_policy="safe-area")
        env.reset()
        if opening == "center":
            env.step(center)
        guard = 0
        while env.status in ("ready", "playing") and guard < rows * cols + 5:
            guard += 1
            action = int(policy.act(encode(env), env.legal_action_mask()))
            if env.legal_action_mask()[action] != 1:
                raise ValueError(f"policy selected illegal action {action}")
            env.step(action)
        wins += int(env.status == "won")
        total_steps += env.steps

    games = len(seeds)
    low, high = wilson_interval(wins, games)
    return EvaluationResult(
        suite=suite_name,
        games=games,
        wins=wins,
        win_rate=wins / games,
        ci_low=low,
        ci_high=high,
        avg_steps=total_steps / games,
        seed_base=seeds[0],
        opening=opening,
    )
