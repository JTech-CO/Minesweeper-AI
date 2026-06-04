/**
 * Solver view + shared deduction state (internal base module for solver/*).
 *
 * The solver must reason from PLAYER-VISIBLE information only — never from the ground
 * truth in `Board.mineLayout` (CLAUDE.md §5 false-positive-0 invariant). Critically,
 * `Board.adjacent` is filled for *every* cell (including hidden ones) by the engine, so
 * exposing it directly would leak hidden numbers. {@link toSolverView} therefore masks
 * adjacency to `-1` for any non-revealed cell — the solver structurally cannot cheat.
 */
import { forEachNeighbor } from '../board';
import type { Board } from '../types';

/** Player-visible projection of a board. No `mineLayout`; hidden numbers are masked. */
export interface SolverView {
  readonly rows: number;
  readonly cols: number;
  /** Total mines on the board (a number the player knows). */
  readonly mines: number;
  /** 1 if revealed. */
  readonly revealed: Uint8Array;
  /** 1 if flagged (treated as a known mine by the solver pipeline). */
  readonly flagged: Uint8Array;
  /** Adjacent mine count for REVEALED cells; -1 for covered cells (unknown). */
  readonly number: Int8Array;
}

/** Build a {@link SolverView} from a board, masking all hidden information. */
export function toSolverView(board: Board): SolverView {
  const n = board.rows * board.cols;
  const number = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    number[i] = board.revealed[i] ? board.adjacent[i]! : -1;
  }
  return {
    rows: board.rows,
    cols: board.cols,
    mines: board.mines,
    revealed: board.revealed,
    flagged: board.flagged,
    number,
  };
}

/**
 * Mutable working state accumulated across deduction rules within one analysis.
 * Both arrays refer to COVERED (not-yet-revealed) cells.
 */
export interface Deductions {
  /** 1 = proven mine. */
  mine: Uint8Array;
  /** 1 = proven safe. */
  safe: Uint8Array;
}

/** Fresh deduction state, seeded with flags as known mines. */
export function initDeductions(view: SolverView): Deductions {
  const n = view.rows * view.cols;
  const mine = new Uint8Array(n);
  const safe = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (view.flagged[i]) mine[i] = 1;
  return { mine, safe };
}

/** A cell is covered if not revealed. */
export function isCovered(view: SolverView, i: number): boolean {
  return view.revealed[i] === 0;
}

/** Undetermined = covered and not yet proven mine or safe. */
export function isUndetermined(view: SolverView, ded: Deductions, i: number): boolean {
  return view.revealed[i] === 0 && ded.mine[i] === 0 && ded.safe[i] === 0;
}

/** A constraint from a revealed number: its undetermined neighbors hold exactly `sum` mines. */
export interface Constraint {
  /** Undetermined neighbor cell indices. */
  cells: number[];
  /** Number of mines among `cells` (the revealed number minus already-known mines). */
  sum: number;
}

/**
 * Extract one constraint per revealed numbered cell that still has undetermined neighbors.
 * Known-mine neighbors are folded into `sum`; known-safe neighbors are dropped.
 */
export function buildConstraints(view: SolverView, ded: Deductions): Constraint[] {
  const { rows, cols } = view;
  const cells = rows * cols;
  const out: Constraint[] = [];
  for (let c = 0; c < cells; c++) {
    if (!view.revealed[c]) continue;
    const k = view.number[c]!;
    if (k < 0) continue;
    let knownMines = 0;
    const vars: number[] = [];
    forEachNeighbor(c, rows, cols, (nb) => {
      if (view.revealed[nb]) return;
      if (ded.mine[nb]) knownMines++;
      else if (!ded.safe[nb]) vars.push(nb);
    });
    if (vars.length > 0) out.push({ cells: vars, sum: k - knownMines });
  }
  return out;
}

/** Count cells flagged in `ded.mine`. */
export function knownMineCount(view: SolverView, ded: Deductions): number {
  const n = view.rows * view.cols;
  let count = 0;
  for (let i = 0; i < n; i++) if (ded.mine[i]) count++;
  return count;
}
