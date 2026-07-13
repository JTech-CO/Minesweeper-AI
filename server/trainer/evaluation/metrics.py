"""Statistical metrics used for model selection and final reporting."""

from __future__ import annotations

import math


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a Bernoulli proportion."""
    if games < 1:
        raise ValueError("games must be positive")
    if not 0 <= wins <= games:
        raise ValueError("wins must be in 0..games")
    p = wins / games
    z2 = z * z
    denom = 1.0 + z2 / games
    center = (p + z2 / (2.0 * games)) / denom
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * games)) / games)
        / denom
    )
    low = 0.0 if wins == 0 else max(0.0, center - radius)
    high = 1.0 if wins == games else min(1.0, center + radius)
    return low, high
