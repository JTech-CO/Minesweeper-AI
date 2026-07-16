"""DAgger teacher states collected from policy-only Minesweeper rollouts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from trainer.data.teacher import (
    TeacherDataset,
    TeacherSample,
    targets_from_solver_step,
)
from trainer.encoding_v3 import encode_v3
from trainer.env import MinesweeperEnv
from trainer.solver import solve_step, view_from_arrays


def _compact(sample: TeacherSample) -> TeacherSample:
    """Store immutable teacher data in compact dtypes; batches expand to float32."""

    return replace(
        sample,
        state=sample.state.astype(np.float16),
        legal_mask=sample.legal_mask.astype(np.uint8),
        policy_target=sample.policy_target.astype(np.float16),
        risk_target=sample.risk_target.astype(np.float16),
        mine_target=sample.mine_target.astype(np.uint8),
        certainty_target=sample.certainty_target.astype(np.int8),
    )


@torch.no_grad()
def generate_on_policy_teacher_dataset(
    model: nn.Module,
    *,
    device: torch.device,
    rows: int,
    cols: int,
    mines: int,
    boards: int,
    seed_base: int,
    max_states: int,
    batch_size: int = 16,
) -> TeacherDataset:
    """Collect model-visited states, then attach visible-only solver labels.

    Actions are computed before ``solve_step`` is called. Solver output therefore cannot
    influence the rollout and remains a training label rather than an inference feature.
    """

    if min(boards, max_states, batch_size) < 1:
        raise ValueError("boards, max_states, and batch_size must be positive")
    was_training = model.training
    model.eval()
    samples: list[TeacherSample] = []
    center = (rows // 2) * cols + cols // 2
    try:
        for start in range(0, boards, batch_size):
            count = min(batch_size, boards - start)
            envs = [
                MinesweeperEnv(
                    rows,
                    cols,
                    mines,
                    seed=seed_base + start + index,
                    first_click_policy="safe-area",
                )
                for index in range(count)
            ]
            sample_indices: list[list[int]] = [[] for _ in envs]
            for env in envs:
                env.reset()
                env.step(center)

            while True:
                active = [index for index, env in enumerate(envs) if env.status == "playing"]
                if not active:
                    break
                states = np.stack([encode_v3(envs[index]) for index in active])
                legal = np.stack(
                    [envs[index].legal_action_mask().reshape(rows, cols) for index in active]
                )
                output = model(
                    torch.from_numpy(states).to(device),
                    torch.from_numpy(legal).to(device).bool(),
                )
                actions = output["policy_logits"].argmax(dim=1).cpu().numpy()

                for position, env_index in enumerate(active):
                    env = envs[env_index]
                    if len(samples) < max_states:
                        view = view_from_arrays(
                            rows,
                            cols,
                            mines,
                            env.revealed,
                            env.adjacent,
                            env.flagged,
                        )
                        decision = solve_step(view)
                        policy, risk, certainty = targets_from_solver_step(env, decision)
                        sample_indices[env_index].append(len(samples))
                        samples.append(
                            _compact(
                                TeacherSample(
                                    state=states[position],
                                    legal_mask=legal[position],
                                    policy_target=policy.reshape(rows, cols),
                                    risk_target=risk.reshape(rows, cols),
                                    mine_target=env.mine_layout.reshape(rows, cols),
                                    certainty_target=certainty.reshape(rows, cols),
                                    value_target=0.0,
                                )
                            )
                        )
                    env.step(int(actions[position]))

            for env_index, env in enumerate(envs):
                value = 1.0 if env.status == "won" else -1.0
                for sample_index in sample_indices[env_index]:
                    samples[sample_index] = replace(
                        samples[sample_index],
                        value_target=value,
                    )
            if len(samples) >= max_states:
                break
    finally:
        model.train(was_training)
    if not samples:
        raise RuntimeError("on-policy teacher collection produced no states")
    return TeacherDataset(samples[:max_states])
