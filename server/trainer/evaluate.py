"""Greedy evaluation: win-rate / steps over fresh boards (기술백서 §2.4)."""

from __future__ import annotations

from trainer.dqn import DQNAgent
from trainer.encoding import encode
from trainer.env import MinesweeperEnv


def evaluate(
    agent: DQNAgent,
    rows: int,
    cols: int,
    mines: int,
    *,
    games: int,
    seed_base: int,
    first_click_policy: str = "safe-area",
) -> dict:
    """Play `games` boards greedily (ε=0) with action masking; report win-rate + avg steps."""
    wins = 0
    total_steps = 0
    for g in range(games):
        env = MinesweeperEnv(
            rows, cols, mines, seed=seed_base + g, first_click_policy=first_click_policy
        )
        env.reset()
        guard = 0
        while env.status in ("ready", "playing") and guard < rows * cols + 5:
            guard += 1
            action = agent.act(encode(env), env.legal_action_mask(), eps=0.0)
            env.step(action)
        wins += 1 if env.status == "won" else 0
        total_steps += env.steps
    return {"win_rate": wins / games, "avg_steps": total_steps / games, "games": games}
