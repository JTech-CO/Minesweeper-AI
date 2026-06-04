"""Reward presets (기술백서 §4.3). The canonical weights live in `env.RewardConfig`
(mirrored from the TS `DEFAULT_REWARD`); this module names a couple of variants.

- SHAPED  : small per-step progress signal (default) — speeds up early learning.
- SPARSE  : only terminal win/loss — cleaner objective, slower to learn.
"""

from __future__ import annotations

from trainer.env import RewardConfig

SHAPED = RewardConfig(win=1.0, loss=-1.0, progress=0.3, no_progress=-0.3, step=0.0)
SPARSE = RewardConfig(win=1.0, loss=-1.0, progress=0.0, no_progress=0.0, step=0.0)

_PRESETS = {"shaped": SHAPED, "sparse": SPARSE}


def make_reward(name: str) -> RewardConfig:
    if name not in _PRESETS:
        raise ValueError(f"unknown reward preset {name!r}; choose from {sorted(_PRESETS)}")
    return _PRESETS[name]
