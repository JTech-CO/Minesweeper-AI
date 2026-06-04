"""State encoding: board → (C, H, W) float32 tensor for the fully-conv CNN.

**Parity contract (M7):** `agent/encoding.ts` must produce the exact same channels in
the same order. Channel layout (C = 11):
  - 0:      hidden        (1 if the cell is covered / not revealed)
  - 1..9:   revealed-k    (channel 1+k = 1 if revealed with adjacent count k, k=0..8)
  - 10:     mine density  (mines / (rows*cols), broadcast — size-invariant context for
                           curriculum transfer)

Flags are NOT encoded: the policy net is reveal-only and the hybrid agent handles
solver flags separately. "hidden" therefore means simply "not revealed".
"""

from __future__ import annotations

import numpy as np

NUM_CHANNELS = 11


def encode(env) -> np.ndarray:  # noqa: ANN001 (env: trainer.env.MinesweeperEnv)
    """Encode the current visible board state as a (NUM_CHANNELS, rows, cols) float32 array."""
    h, w = env.rows, env.cols
    out = np.zeros((NUM_CHANNELS, env.n), dtype=np.float32)

    covered = env.revealed == 0
    out[0, covered] = 1.0

    revealed = env.revealed == 1
    adj = env.adjacent
    for k in range(9):
        out[1 + k, revealed & (adj == k)] = 1.0

    out[10, :] = env.mines / env.n
    return out.reshape(NUM_CHANNELS, h, w)
