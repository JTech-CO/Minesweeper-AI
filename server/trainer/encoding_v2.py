"""M4-R2 spatial encoding with explicit local and global context."""

from __future__ import annotations

import numpy as np

from trainer.env import neighbors

NUM_CHANNELS_V2 = 15
VALID_CHANNEL = 14


def encode_v2(env, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    rows, cols = env.rows, env.cols
    out_rows, out_cols = target_shape or (rows, cols)
    if out_rows < rows or out_cols < cols:
        raise ValueError("target_shape cannot crop the board")
    out = np.zeros((NUM_CHANNELS_V2, out_rows, out_cols), dtype=np.float32)
    local = np.zeros((NUM_CHANNELS_V2, rows, cols), dtype=np.float32)
    flat = local.reshape(NUM_CHANNELS_V2, rows * cols)

    hidden = env.revealed == 0
    flat[0, hidden] = 1.0
    revealed = env.revealed == 1
    for number in range(9):
        flat[1 + number, revealed & (env.adjacent == number)] = 1.0

    flat[10, :] = env.mines / env.n
    flat[11, :] = float(hidden.sum()) / env.n
    flat[12, :] = env.safe_revealed / max(env.safe_cells, 1)
    frontier = np.zeros(env.n, dtype=np.float32)
    for cell in np.flatnonzero(hidden):
        if any(env.revealed[nb] for nb in neighbors(int(cell), rows, cols)):
            frontier[cell] = 1.0
    flat[13, :] = frontier
    flat[VALID_CHANNEL, :] = 1.0
    out[:, :rows, :cols] = local
    return out


def legal_mask_v2(env, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    rows, cols = env.rows, env.cols
    out_rows, out_cols = target_shape or (rows, cols)
    mask = np.zeros((out_rows, out_cols), dtype=np.uint8)
    mask[:rows, :cols] = env.legal_action_mask().reshape(rows, cols)
    return mask
