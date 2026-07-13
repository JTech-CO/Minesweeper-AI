"""Canonical evaluation for the constraint-aware axial policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from trainer.encoding_v3 import encode_v3
from trainer.env import MinesweeperEnv
from trainer.evaluation.evaluator import EvaluationResult
from trainer.evaluation.metrics import wilson_interval


@dataclass
class TorchAxialPolicy:
    model: nn.Module
    device: torch.device
    risk_weight: float = 0.0
    certainty_weight: float = 0.0

    @torch.no_grad()
    def act(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        x = torch.from_numpy(state[None]).to(self.device)
        legal = torch.from_numpy(legal_mask[None]).to(self.device).bool()
        output = self.model(x, legal)
        score = output["policy_logits"]
        if self.risk_weight:
            score = score - self.risk_weight * torch.sigmoid(output["risk_logits"])
        if self.certainty_weight:
            certainty = torch.softmax(output["certainty_logits"], dim=1)
            safe_minus_mine = certainty[:, 1] - certainty[:, 2]
            score = score + self.certainty_weight * safe_minus_mine.flatten(start_dim=1)
        return int(score.argmax(dim=1).item())


def evaluate_axial_policy(
    policy: TorchAxialPolicy,
    rows: int,
    cols: int,
    mines: int,
    *,
    seeds: tuple[int, ...],
    suite_name: str,
) -> EvaluationResult:
    wins = 0
    total_steps = 0
    center = (rows // 2) * cols + cols // 2
    policy.model.eval()
    for seed in seeds:
        env = MinesweeperEnv(rows, cols, mines, seed=seed, first_click_policy="safe-area")
        env.reset()
        env.step(center)
        guard = 0
        while env.status == "playing" and guard < rows * cols + 5:
            guard += 1
            legal = env.legal_action_mask().reshape(rows, cols)
            env.step(policy.act(encode_v3(env), legal))
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
        opening="center",
    )

