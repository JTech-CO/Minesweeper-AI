"""Minesweeper environment for RL training.

**Parity invariant (CLAUDE.md §5):** the game rules here must match the TS engine in
`packages/core/src/board.ts` 1:1 — flat row-major indexing (`index = row*cols + col`),
first-click safety (`safe-area`: the clicked cell and its 8 neighbours are mine-free when
there is room, else `safe-cell`), BFS cascade through 0-cells that stops at flags, win
when every safe cell is revealed, and reveal-only action masking. The RNG need NOT match
TS (only within-Python determinism is required); mine *layouts* therefore differ across
languages, which is fine — M7 encoding parity feeds both sides identical explicit boards.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

GameStatus = Literal["ready", "playing", "won", "lost"]
FirstClickPolicy = Literal["safe-cell", "safe-area"]

# Standard presets (기술백서 §4.3): rows = height, cols = width.
DIFFICULTIES: dict[str, tuple[int, int, int]] = {
    "beginner": (9, 9, 10),
    "intermediate": (16, 16, 40),
    "expert": (16, 30, 99),
}


@dataclass
class RewardConfig:
    """Mirror of the TS `DEFAULT_REWARD` (packages/core/src/env.ts). Canonical for training."""

    win: float = 1.0
    loss: float = -1.0
    progress: float = 0.3
    no_progress: float = -0.3
    step: float = 0.0


def neighbors(index: int, rows: int, cols: int) -> list[int]:
    """Up-to-8 neighbours of a flat index, clamped to the grid."""
    row, col = divmod(index, cols)
    out: list[int] = []
    for dr in (-1, 0, 1):
        r = row + dr
        if r < 0 or r >= rows:
            continue
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            c = col + dc
            if 0 <= c < cols:
                out.append(r * cols + c)
    return out


class MinesweeperEnv:
    """Reveal-only RL environment. Action = flat index of the cell to open (H×W)."""

    def __init__(
        self,
        rows: int,
        cols: int,
        mines: int,
        *,
        seed: int = 0,
        first_click_policy: FirstClickPolicy = "safe-area",
        reward: RewardConfig | None = None,
    ) -> None:
        if rows < 1 or cols < 1:
            raise ValueError(f"invalid board size: {rows}x{cols}")
        if not (0 <= mines <= rows * cols - 1):
            raise ValueError(f"invalid mine count {mines} for {rows}x{cols}")
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.n = rows * cols
        self.first_click_policy = first_click_policy
        self.reward_cfg = reward or RewardConfig()
        self._seed = seed
        self._rng = random.Random(seed)
        self._reset_arrays()

    # -- construction / reset -------------------------------------------------

    def _reset_arrays(self) -> None:
        self.mine_layout = np.zeros(self.n, dtype=np.uint8)
        self.adjacent = np.zeros(self.n, dtype=np.uint8)
        self.revealed = np.zeros(self.n, dtype=np.uint8)
        self.flagged = np.zeros(self.n, dtype=np.uint8)
        self.status: GameStatus = "ready"
        self.mines_placed = False
        self.safe_revealed = 0
        self.exploded_at = -1
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self._reset_arrays()
        return self.observe()

    # -- mine placement -------------------------------------------------------

    def _compute_adjacency(self) -> None:
        self.adjacent.fill(0)
        for i in range(self.n):
            count = 0
            for nb in neighbors(i, self.rows, self.cols):
                if self.mine_layout[nb]:
                    count += 1
            self.adjacent[i] = count

    def _place_mines(self, safe_index: int) -> None:
        forbidden = {safe_index}
        if self.first_click_policy == "safe-area" and self.n - self.mines >= 9:
            forbidden.update(neighbors(safe_index, self.rows, self.cols))
        candidates = [i for i in range(self.n) if i not in forbidden]
        if self.mines > len(candidates):
            raise ValueError(f"cannot place {self.mines} mines in {len(candidates)} free cells")
        # Partial Fisher–Yates: pick `mines` distinct candidates deterministically.
        length = len(candidates)
        self.mine_layout.fill(0)
        for m in range(self.mines):
            j = m + self._rng.randrange(length - m)
            candidates[m], candidates[j] = candidates[j], candidates[m]
            self.mine_layout[candidates[m]] = 1
        self._compute_adjacency()
        self.mines_placed = True

    def set_mines(self, mine_indices: Iterable[int]) -> None:
        """Test/setup helper: place mines at explicit indices and make the board playable."""
        self._reset_arrays()
        for m in mine_indices:
            if not (0 <= m < self.n):
                raise ValueError(f"mine index out of range: {m}")
            self.mine_layout[m] = 1
        self._compute_adjacency()
        self.mines_placed = True
        self.status = "playing"

    # -- core actions ---------------------------------------------------------

    @property
    def safe_cells(self) -> int:
        return self.n - self.mines

    def _reveal(self, index: int) -> tuple[bool, int]:
        """Reveal a cell (mines must be placed). Returns (hit_mine, newly_revealed_count)."""
        if self.status != "playing":
            return False, 0
        if self.revealed[index] or self.flagged[index]:
            return False, 0
        if self.mine_layout[index]:
            self.status = "lost"
            self.exploded_at = index
            self.revealed[index] = 1
            return True, 0

        newly = 0
        stack = [index]
        while stack:
            cell = stack.pop()
            if self.revealed[cell] or self.flagged[cell]:
                continue
            self.revealed[cell] = 1
            self.safe_revealed += 1
            newly += 1
            if self.adjacent[cell] == 0:  # cascade only through 0-cells
                for nb in neighbors(cell, self.rows, self.cols):
                    if not self.revealed[nb] and not self.flagged[nb]:
                        stack.append(nb)

        if self.safe_revealed == self.safe_cells:
            self.status = "won"
        return False, newly

    def toggle_flag(self, index: int) -> bool:
        if self.status not in ("playing", "ready"):
            return False
        if self.revealed[index]:
            return False
        self.flagged[index] = 0 if self.flagged[index] else 1
        return True

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.status in ("won", "lost"):
            raise RuntimeError("step() on a terminal board; call reset() first")
        if not (0 <= action < self.n):
            raise ValueError(f"action out of range: {action}")

        self.steps += 1
        if not self.mines_placed:
            self._place_mines(action)
            self.status = "playing"

        hit_mine, newly = self._reveal(action)

        cfg = self.reward_cfg
        if self.status == "lost":
            reward = cfg.loss
        elif self.status == "won":
            reward = cfg.win
        elif newly > 0:
            reward = cfg.progress
        else:
            reward = cfg.no_progress
        reward += cfg.step

        done = self.status in ("won", "lost")
        info = {"steps": self.steps, "hit_mine": hit_mine, "won": self.status == "won"}
        return self.observe(), reward, done, info

    # -- observation ----------------------------------------------------------

    def legal_action_mask(self) -> np.ndarray:
        """1 where a cell may be revealed. Ready → all legal; terminal → none."""
        if self.status in ("won", "lost"):
            return np.zeros(self.n, dtype=np.uint8)
        if self.status == "ready":
            return np.ones(self.n, dtype=np.uint8)
        return ((self.revealed == 0) & (self.flagged == 0)).astype(np.uint8)

    def observe(self) -> np.ndarray:
        """Player-visible state per cell: -1 hidden, -2 flagged, else 0..8 (adjacent count).

        This is the raw view; channel encoding lives in encoding.py (M4) / agent (M7).
        """
        view = np.full(self.n, -1, dtype=np.int8)
        view[self.flagged == 1] = -2
        revealed_mask = self.revealed == 1
        view[revealed_mask] = self.adjacent[revealed_mask].astype(np.int8)
        return view

    def render(self) -> str:
        chars = []
        for i in range(self.n):
            if self.revealed[i]:
                if self.mine_layout[i]:
                    chars.append("*" if self.exploded_at == i else "X")
                else:
                    a = int(self.adjacent[i])
                    chars.append("." if a == 0 else str(a))
            elif self.flagged[i]:
                chars.append("F")
            else:
                chars.append("#")
        rows = [" ".join(chars[r * self.cols : (r + 1) * self.cols]) for r in range(self.rows)]
        return "\n".join(rows)
