/**
 * Board engine: creation, first-click-safe mine placement, reveal + BFS cascade,
 * flagging, and win/loss detection (파일트리: board.ts; 기술백서 §4.3).
 *
 * Cells are flat, row-major: `index = row * cols + col`. The board layer assumes mines
 * are already placed for {@link reveal}; first-click placement is orchestrated by the
 * env (env.ts) so the seeded RNG and click position are both available.
 */
import type { Board, BoardSpec, CellView, FirstClickPolicy, RevealResult } from './types';
import type { Rng } from './rng';

/** A fresh "nothing happened" result (no shared mutable array). */
function noop(): RevealResult {
  return { ok: false, hitMine: false, revealed: [], won: false, lost: false };
}

/** Flat index from row/col. */
export function toIndex(row: number, col: number, cols: number): number {
  return row * cols + col;
}

/** Row/col from a flat index. */
export function toCoords(index: number, cols: number): { row: number; col: number } {
  return { row: Math.floor(index / cols), col: index % cols };
}

/**
 * Invoke `fn` for each of the up-to-8 neighbors of `index` (clamped to the grid).
 * Hot path for adjacency, cascade, and 3BV — kept allocation-free.
 */
export function forEachNeighbor(
  index: number,
  rows: number,
  cols: number,
  fn: (neighbor: number) => void,
): void {
  const row = Math.floor(index / cols);
  const col = index % cols;
  for (let dr = -1; dr <= 1; dr++) {
    const r = row + dr;
    if (r < 0 || r >= rows) continue;
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const c = col + dc;
      if (c < 0 || c >= cols) continue;
      fn(r * cols + c);
    }
  }
}

/** Create an empty board in the `ready` state (mines not yet placed). */
export function createBoard(spec: BoardSpec): Board {
  const { rows, cols, mines } = spec;
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 1 || cols < 1) {
    throw new Error(`invalid board size: ${rows}x${cols}`);
  }
  const cells = rows * cols;
  if (!Number.isInteger(mines) || mines < 0 || mines > cells - 1) {
    throw new Error(`invalid mine count ${mines} for ${rows}x${cols} (need 0..${cells - 1})`);
  }
  return {
    rows,
    cols,
    mines,
    mineLayout: new Uint8Array(cells),
    adjacent: new Uint8Array(cells),
    revealed: new Uint8Array(cells),
    flagged: new Uint8Array(cells),
    status: 'ready',
    minesPlaced: false,
    safeRevealed: 0,
    explodedAt: -1,
  };
}

/** Recompute adjacency counts for every cell from the current mine layout. */
export function computeAdjacency(board: Board): void {
  const { rows, cols, mineLayout, adjacent } = board;
  const cells = rows * cols;
  adjacent.fill(0);
  for (let i = 0; i < cells; i++) {
    let count = 0;
    forEachNeighbor(i, rows, cols, (n) => {
      if (mineLayout[n]) count++;
    });
    adjacent[i] = count;
  }
}

/**
 * Place `board.mines` mines at random, avoiding a forbidden set, then compute adjacency.
 * Used by the env on the first reveal.
 *
 * @param safeIndex the just-clicked cell, always kept mine-free.
 * @param rng seeded RNG (determinism).
 * @param policy see {@link FirstClickPolicy}. `safe-area` also frees the 8 neighbors when
 *   there is room, so the first click opens a cascade.
 */
export function placeMines(
  board: Board,
  safeIndex: number,
  rng: Rng,
  policy: FirstClickPolicy = 'safe-area',
): void {
  const { rows, cols, mines } = board;
  const cells = rows * cols;

  const forbidden = new Set<number>([safeIndex]);
  if (policy === 'safe-area' && cells - mines >= 9) {
    forEachNeighbor(safeIndex, rows, cols, (n) => forbidden.add(n));
  }

  const candidates: number[] = [];
  for (let i = 0; i < cells; i++) {
    if (!forbidden.has(i)) candidates.push(i);
  }
  if (mines > candidates.length) {
    // Should never happen given createBoard validation + the safe-area room check above.
    throw new Error(`cannot place ${mines} mines in ${candidates.length} free cells`);
  }

  board.mineLayout.fill(0);
  // Partial Fisher–Yates: select `mines` distinct candidates deterministically.
  const len = candidates.length;
  for (let m = 0; m < mines; m++) {
    const j = m + Math.floor(rng() * (len - m));
    const pick = candidates[j]!;
    candidates[j] = candidates[m]!;
    candidates[m] = pick;
    board.mineLayout[pick] = 1;
  }

  computeAdjacency(board);
  board.minesPlaced = true;
}

/**
 * Test/setup helper: place mines at explicit indices and mark the board playable.
 * Lets tests build known layouts for cascade / 3BV assertions.
 */
export function setMines(board: Board, mineIndices: Iterable<number>): void {
  board.mineLayout.fill(0);
  board.adjacent.fill(0);
  board.revealed.fill(0);
  board.flagged.fill(0);
  board.safeRevealed = 0;
  board.explodedAt = -1;
  for (const m of mineIndices) {
    if (m < 0 || m >= board.rows * board.cols) throw new Error(`mine index out of range: ${m}`);
    board.mineLayout[m] = 1;
  }
  computeAdjacency(board);
  board.minesPlaced = true;
  board.status = 'playing';
}

/** Total number of safe (non-mine) cells. */
export function safeCellCount(board: Board): number {
  return board.rows * board.cols - board.mines;
}

/**
 * Reveal the cell at `index`. Assumes mines are already placed.
 *
 * - Revealing a mine → loss (records `explodedAt`).
 * - Revealing a 0 cell → BFS cascade opens the connected 0-region and its bordering
 *   numbers. The flood stops at flagged cells (a flag protects a cell from opening).
 * - Win when every safe cell has been revealed.
 *
 * No-op (returns `ok: false`) if the game is over or the cell is already revealed/flagged.
 */
export function reveal(board: Board, index: number): RevealResult {
  if (board.status !== 'playing') return noop();
  if (!board.minesPlaced) throw new Error('reveal() called before mines were placed');
  if (board.revealed[index] || board.flagged[index]) return noop();

  if (board.mineLayout[index]) {
    board.status = 'lost';
    board.explodedAt = index;
    board.revealed[index] = 1;
    return { ok: true, hitMine: true, revealed: [index], won: false, lost: true };
  }

  const { rows, cols } = board;
  const out: number[] = [];
  const stack: number[] = [index];
  while (stack.length > 0) {
    const cell = stack.pop()!;
    if (board.revealed[cell] || board.flagged[cell]) continue;
    board.revealed[cell] = 1;
    board.safeRevealed++;
    out.push(cell);
    // Cascade only through 0-cells; their neighbors are guaranteed non-mine.
    if (board.adjacent[cell] === 0) {
      forEachNeighbor(cell, rows, cols, (n) => {
        if (!board.revealed[n] && !board.flagged[n]) stack.push(n);
      });
    }
  }

  const won = board.safeRevealed === safeCellCount(board);
  if (won) board.status = 'won';
  return { ok: true, hitMine: false, revealed: out, won, lost: false };
}

/**
 * Toggle a flag on a hidden cell. No-op on revealed cells or after the game ends.
 * @returns true if the flag state changed.
 */
export function toggleFlag(board: Board, index: number): boolean {
  if (board.status !== 'playing' && board.status !== 'ready') return false;
  if (board.revealed[index]) return false;
  board.flagged[index] = board.flagged[index] ? 0 : 1;
  return true;
}

/** The player-visible state of a single cell (no hidden mine info unless lost). */
export function cellView(board: Board, index: number): CellView {
  if (board.revealed[index]) {
    if (board.mineLayout[index]) {
      return board.explodedAt === index ? { state: 'exploded' } : { state: 'mine' };
    }
    return { state: 'revealed', adjacent: board.adjacent[index]! };
  }
  if (board.flagged[index]) return { state: 'flagged' };
  return { state: 'hidden' };
}

/** Render the board as ASCII for debugging/tests. */
export function render(board: Board): string {
  const { rows, cols } = board;
  const lines: string[] = [];
  for (let r = 0; r < rows; r++) {
    const row: string[] = [];
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c;
      const view = cellView(board, i);
      switch (view.state) {
        case 'hidden':
          row.push('#');
          break;
        case 'flagged':
          row.push('F');
          break;
        case 'exploded':
          row.push('*');
          break;
        case 'mine':
          row.push('X');
          break;
        case 'revealed':
          row.push(view.adjacent === 0 ? '.' : String(view.adjacent));
          break;
      }
    }
    lines.push(row.join(' '));
  }
  return lines.join('\n');
}
