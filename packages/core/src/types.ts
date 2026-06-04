/**
 * Core types for the Minesweeper engine (@msai/core).
 *
 * Pure data only — no DOM, no ONNX (CLAUDE.md §5 invariant). These types are the
 * shared contract that the RL env, the logic solver, and the dashboard all build on.
 */

/** Difficulty preset names. `custom` lets callers pass an explicit {@link BoardSpec}. */
export type Difficulty = 'beginner' | 'intermediate' | 'expert' | 'custom';

/** Board dimensions + mine count. Convention: `rows` = height, `cols` = width. */
export interface BoardSpec {
  rows: number;
  cols: number;
  mines: number;
}

/**
 * Game lifecycle.
 * - `ready`: created but mines not yet placed (mines are placed on the first reveal
 *   so first-click safety can be guaranteed).
 * - `playing`: in progress.
 * - `won` / `lost`: terminal.
 */
export type GameStatus = 'ready' | 'playing' | 'won' | 'lost';

/**
 * First-click safety policy.
 * - `safe-cell`: only the clicked cell is guaranteed mine-free.
 * - `safe-area`: the clicked cell *and its 8 neighbors* are mine-free, guaranteeing
 *   the first click opens a cascade (matches minesweeper.online default). Falls back
 *   to `safe-cell` automatically when the board has no room (mines > rows*cols - 9).
 */
export type FirstClickPolicy = 'safe-cell' | 'safe-area';

/**
 * Mutable board state. Cell data is stored in flat typed arrays of length `rows*cols`,
 * row-major (index = row * cols + col). Typed arrays keep the engine allocation-light
 * for training rollouts and give a layout that maps cleanly onto the Python env (M4).
 */
export interface Board {
  readonly rows: number;
  readonly cols: number;
  readonly mines: number;
  /** 1 = mine. All zero until mines are placed (see {@link GameStatus}). */
  mineLayout: Uint8Array;
  /** Number of adjacent mines per cell (0–8). Valid once mines are placed. */
  adjacent: Uint8Array;
  /** 1 = revealed (opened). */
  revealed: Uint8Array;
  /** 1 = flagged. */
  flagged: Uint8Array;
  status: GameStatus;
  /** Whether mines have been placed (happens on the first reveal). */
  minesPlaced: boolean;
  /** Count of revealed *safe* cells; used for the win check. */
  safeRevealed: number;
  /** Flat index of the exploded mine after a loss, else -1. */
  explodedAt: number;
}

/** What a single cell looks like to a player/agent (no hidden mine info leaked). */
export type CellView =
  | { state: 'hidden' }
  | { state: 'flagged' }
  | { state: 'revealed'; adjacent: number }
  /** The mine that was clicked (loss). */
  | { state: 'exploded' }
  /** A non-clicked mine, surfaced only after a loss for display. */
  | { state: 'mine' };

/** Result of a single {@link reveal} on the board layer. */
export interface RevealResult {
  /** false when the action was a no-op (already revealed/flagged, or game over). */
  ok: boolean;
  /** true if the revealed cell was a mine (terminal loss). */
  hitMine: boolean;
  /** Flat indices newly revealed by this action (includes cascade). */
  revealed: number[];
  won: boolean;
  lost: boolean;
}

/** An RL action: the flat index of the hidden cell to reveal. */
export type Action = number;

/** Reward weights for {@link MinesweeperEnv}. The canonical training reward lives in
 *  `server/trainer/reward.py` (M4); this mirror keeps the TS env usable on its own. */
export interface RewardConfig {
  /** Terminal reward for clearing the board. */
  win: number;
  /** Terminal reward for hitting a mine. */
  loss: number;
  /** Reward for a step that opens at least one new safe cell. */
  progress: number;
  /** Reward for a no-op step (e.g. an unmasked illegal action). */
  noProgress: number;
  /** Per-step shaping penalty (encourages speed). 0 disables it. */
  step: number;
}

/** Result of {@link MinesweeperEnv.step}. */
export interface StepResult {
  board: Board;
  reward: number;
  done: boolean;
  won: boolean;
  lost: boolean;
  /** Number of cells newly revealed this step. */
  revealed: number;
  info: {
    steps: number;
    hitMine: boolean;
    /** false if the action was a no-op. */
    ok: boolean;
  };
}
