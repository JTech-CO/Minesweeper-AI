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
        return int(self.act_batch(state[None], legal_mask[None])[0])

    @torch.no_grad()
    def act_batch(
        self,
        states: np.ndarray,
        legal_masks: np.ndarray,
    ) -> np.ndarray:
        x = torch.from_numpy(states).to(self.device)
        legal = torch.from_numpy(legal_masks).to(self.device).bool()
        output = self.model(x, legal)
        score = output["policy_logits"]
        if self.risk_weight:
            score = score - self.risk_weight * torch.sigmoid(output["risk_logits"])
        if self.certainty_weight:
            certainty = torch.softmax(output["certainty_logits"], dim=1)
            safe_minus_mine = certainty[:, 1] - certainty[:, 2]
            score = score + self.certainty_weight * safe_minus_mine.flatten(start_dim=1)
        return score.argmax(dim=1).cpu().numpy()


def evaluate_axial_policy(
    policy: TorchAxialPolicy,
    rows: int,
    cols: int,
    mines: int,
    *,
    seeds: tuple[int, ...],
    suite_name: str,
    batch_size: int = 32,
) -> EvaluationResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    wins = 0
    total_steps = 0
    center = (rows // 2) * cols + cols // 2
    policy.model.eval()
    for start in range(0, len(seeds), batch_size):
        envs = [
            MinesweeperEnv(
                rows,
                cols,
                mines,
                seed=seed,
                first_click_policy="safe-area",
            )
            for seed in seeds[start : start + batch_size]
        ]
        for env in envs:
            env.reset()
            env.step(center)
        guards = [0] * len(envs)
        while True:
            active = [
                index
                for index, env in enumerate(envs)
                if env.status == "playing" and guards[index] < rows * cols + 5
            ]
            if not active:
                break
            states = np.stack([encode_v3(envs[index]) for index in active])
            legal = np.stack(
                [envs[index].legal_action_mask().reshape(rows, cols) for index in active]
            )
            for index, action in zip(active, policy.act_batch(states, legal), strict=True):
                guards[index] += 1
                envs[index].step(int(action))
        wins += sum(env.status == "won" for env in envs)
        total_steps += sum(env.steps for env in envs)
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
