"""Constraint-aware encoding for the M4-R2 axial policy.

The extra planes are derived only from the visible board. Solver deductions remain
teacher labels and are never exposed to the policy at inference time.
"""

from __future__ import annotations

import numpy as np

from trainer.encoding_v2 import NUM_CHANNELS_V2, VALID_CHANNEL, encode_v2
from trainer.env import neighbors

NUM_CHANNELS_V3 = 20


def upgrade_v2_to_v3(state: np.ndarray) -> np.ndarray:
    if state.ndim != 3 or state.shape[0] != NUM_CHANNELS_V2:
        raise ValueError(f"expected ({NUM_CHANNELS_V2}, H, W) state")
    rows, cols = state.shape[1:]
    out = np.zeros((NUM_CHANNELS_V3, rows, cols), dtype=np.float32)
    out[:NUM_CHANNELS_V2] = state
    hidden = state[0] > 0.5
    valid = state[VALID_CHANNEL] > 0.5

    clue_value = np.full((rows, cols), -1, dtype=np.int8)
    for number in range(1, 9):
        clue_value[state[1 + number] > 0.5] = number

    for row, col in np.argwhere(hidden & valid):
        cell = int(row) * cols + int(col)
        ratios: list[float] = []
        for clue_cell in neighbors(cell, rows, cols):
            clue_row, clue_col = divmod(clue_cell, cols)
            clue = int(clue_value[clue_row, clue_col])
            if clue < 0:
                continue
            unknown = sum(
                bool(hidden[nb // cols, nb % cols])
                for nb in neighbors(clue_cell, rows, cols)
            )
            if unknown:
                ratios.append(min(1.0, clue / unknown))
        if not ratios:
            continue
        out[15, row, col] = min(1.0, len(ratios) / 8.0)
        out[16, row, col] = min(ratios)
        out[17, row, col] = max(ratios)
        out[18, row, col] = float(np.mean(ratios))
        out[19, row, col] = max(ratios) - min(ratios)
    return out


def encode_v3(env, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    return upgrade_v2_to_v3(encode_v2(env, target_shape))

