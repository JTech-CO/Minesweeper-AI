"""Generate solver-supervised states without exposing solver features at inference."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.utils.data import Dataset

from trainer.encoding_v2 import encode_v2
from trainer.env import MinesweeperEnv
from trainer.solver import solve_step, view_from_arrays


@dataclass(frozen=True)
class TeacherSample:
    state: np.ndarray
    legal_mask: np.ndarray
    policy_target: np.ndarray
    risk_target: np.ndarray
    mine_target: np.ndarray
    certainty_target: np.ndarray
    value_target: float


class TeacherDataset(Dataset):
    def __init__(self, samples: list[TeacherSample]) -> None:
        if not samples:
            raise ValueError("teacher dataset cannot be empty")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "state": torch.from_numpy(sample.state),
            "legal_mask": torch.from_numpy(sample.legal_mask),
            "policy_target": torch.from_numpy(sample.policy_target),
            "risk_target": torch.from_numpy(sample.risk_target),
            "mine_target": torch.from_numpy(sample.mine_target),
            "certainty_target": torch.from_numpy(sample.certainty_target),
            "value_target": torch.tensor(sample.value_target, dtype=torch.float32),
        }


def _targets(env: MinesweeperEnv, step: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    legal = env.legal_action_mask().astype(bool)
    safe = {int(cell) for cell in step["safe"] if legal[int(cell)]}
    mines = {int(cell) for cell in step["mines"] if legal[int(cell)]}
    unknown = [int(cell) for cell in np.flatnonzero(legal) if int(cell) not in safe | mines]
    remaining = max(0, env.mines - len(mines))
    base_risk = min(1.0, remaining / max(len(unknown), 1))

    risk = np.full(env.n, 0.0, dtype=np.float32)
    risk[legal] = base_risk
    certainty = np.full(env.n, -100, dtype=np.int64)
    certainty[legal] = 0
    for cell in safe:
        risk[cell] = 0.0
        certainty[cell] = 1
    for cell in mines:
        risk[cell] = 1.0
        certainty[cell] = 2
    for cell, probability in (step["probabilities"] or {}).items():
        if legal[int(cell)]:
            risk[int(cell)] = float(probability)

    policy = np.zeros(env.n, dtype=np.float32)
    if safe:
        chosen = sorted(safe)
    else:
        candidates = [cell for cell in np.flatnonzero(legal) if int(cell) not in mines]
        if not candidates:
            candidates = list(np.flatnonzero(legal))
        min_risk = min(float(risk[int(cell)]) for cell in candidates)
        chosen = [int(cell) for cell in candidates if abs(float(risk[int(cell)]) - min_risk) < 1e-7]
    if chosen:
        policy[chosen] = 1.0 / len(chosen)
    return policy, risk, certainty


def generate_teacher_episode(
    rows: int,
    cols: int,
    mines: int,
    *,
    seed: int,
    max_steps: int | None = None,
) -> list[TeacherSample]:
    env = MinesweeperEnv(rows, cols, mines, seed=seed, first_click_policy="safe-area")
    env.reset()
    center = (rows // 2) * cols + (cols // 2)
    env.step(center)
    samples: list[TeacherSample] = []
    limit = max_steps or rows * cols

    while env.status == "playing" and len(samples) < limit:
        view = view_from_arrays(rows, cols, mines, env.revealed, env.adjacent, env.flagged)
        decision = solve_step(view)
        policy, risk, certainty = _targets(env, decision)
        legal = env.legal_action_mask()
        if not np.any(policy):
            break
        samples.append(
            TeacherSample(
                state=encode_v2(env),
                legal_mask=legal.reshape(rows, cols),
                policy_target=policy.reshape(rows, cols),
                risk_target=risk.reshape(rows, cols),
                mine_target=env.mine_layout.astype(np.float32).reshape(rows, cols),
                certainty_target=certainty.reshape(rows, cols),
                value_target=0.0,
            )
        )
        action = int(
            np.flatnonzero(policy == policy.max())[
                (seed + len(samples)) % np.count_nonzero(policy == policy.max())
            ]
        )
        env.step(action)

    value = 1.0 if env.status == "won" else -1.0
    return [replace(sample, value_target=value) for sample in samples]


def generate_teacher_dataset(
    rows: int,
    cols: int,
    mines: int,
    *,
    boards: int,
    seed_base: int,
    max_states: int | None = None,
) -> TeacherDataset:
    samples: list[TeacherSample] = []
    for board in range(boards):
        samples.extend(generate_teacher_episode(rows, cols, mines, seed=seed_base + board))
        if max_states is not None and len(samples) >= max_states:
            samples = samples[:max_states]
            break
    return TeacherDataset(samples)
