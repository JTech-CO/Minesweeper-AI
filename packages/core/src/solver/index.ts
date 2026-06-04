/**
 * Logic solver entry point (파일트리: solver/index.ts; 기술백서 §4.3).
 *
 * `analyze` runs the deduction fixpoint (single-point → subset → endgame → CSP) and
 * returns only *certain* safe/mine cells — the false-positive-0 invariant (CLAUDE.md §5).
 * `solveStep` adds a probability-based guess when no certainty exists. `solveBoard` is a
 * verification harness that drives a real (mine-bearing) board with the solver to prove
 * soundness and measure no-guess solvability. `generateNoGuessBoard` reject-samples boards
 * that the solver can clear without guessing.
 */
import { cloneBoard, createBoard, placeMines, reveal, toggleFlag } from '../board';
import type { Board, BoardSpec } from '../types';
import type { Rng } from '../rng';
import { cspCertain } from './csp';
import { subsetRule } from './patterns';
import { computeProbabilities } from './probability';
import { singlePoint } from './single-point';
import { initDeductions, toSolverView, type Deductions, type SolverView } from './view';

export interface SolveResult {
  /** Covered cells proven safe. */
  safe: number[];
  /** Covered cells proven to be mines. */
  mines: number[];
}

export interface SolveStep extends SolveResult {
  /** Lowest-risk cell to open when no safe cell is certain, else null. */
  guess: { cell: number; mineProb: number } | null;
  /** Mine probabilities for undetermined cells (only when guessing), else null. */
  probabilities: Map<number, number> | null;
}

export interface SolveOutcome {
  /** Board fully cleared by certain moves only. */
  solved: boolean;
  /** Solver ran out of certain moves before solving (a guess would be required). */
  neededGuess: boolean;
  steps: number;
  /** A claimed-safe cell was a mine, or a claimed-mine cell was not — must never happen. */
  falsePositive: boolean;
}

/** Endgame counting rule: resolve everything when remaining mines hit 0 or fill all cells. */
function endgame(view: SolverView, ded: Deductions): boolean {
  const n = view.rows * view.cols;
  let knownMines = 0;
  const undetermined: number[] = [];
  for (let i = 0; i < n; i++) {
    if (view.revealed[i]) continue;
    if (ded.mine[i]) {
      knownMines++;
      continue;
    }
    if (ded.safe[i]) continue;
    undetermined.push(i);
  }
  const remaining = view.mines - knownMines;
  let changed = false;
  if (remaining === 0) {
    for (const u of undetermined)
      if (!ded.safe[u]) {
        ded.safe[u] = 1;
        changed = true;
      }
  } else if (undetermined.length > 0 && remaining === undetermined.length) {
    for (const u of undetermined)
      if (!ded.mine[u]) {
        ded.mine[u] = 1;
        changed = true;
      }
  }
  return changed;
}

/** Run all deduction rules to a fixpoint. Cheap rules first, CSP only when they stall. */
function runDeductions(view: SolverView): Deductions {
  const ded = initDeductions(view);
  for (;;) {
    const a = singlePoint(view, ded);
    const b = subsetRule(view, ded);
    const c = endgame(view, ded);
    if (a || b || c) continue;
    if (!cspCertain(view, ded)) break;
  }
  return ded;
}

function collect(view: SolverView, ded: Deductions): SolveResult {
  const n = view.rows * view.cols;
  const safe: number[] = [];
  const mines: number[] = [];
  for (let i = 0; i < n; i++) {
    if (view.revealed[i]) continue;
    if (ded.safe[i]) safe.push(i);
    else if (ded.mine[i]) mines.push(i);
  }
  return { safe, mines };
}

/** All certain safe/mine cells deducible from the current view. */
export function analyze(view: SolverView): SolveResult {
  return collect(view, runDeductions(view));
}

/** Like {@link analyze}, but recommends a lowest-risk guess when no certainty remains. */
export function solveStep(view: SolverView): SolveStep {
  const ded = runDeductions(view);
  const { safe, mines } = collect(view, ded);
  if (safe.length > 0) return { safe, mines, guess: null, probabilities: null };

  const probabilities = computeProbabilities(view, ded);
  if (probabilities.size === 0) return { safe, mines, guess: null, probabilities: null };
  let bestCell = -1;
  let bestProb = Infinity;
  for (const [cell, p] of probabilities) {
    if (p < bestProb) {
      bestProb = p;
      bestCell = cell;
    }
  }
  return {
    safe,
    mines,
    guess: bestCell >= 0 ? { cell: bestCell, mineProb: bestProb } : null,
    probabilities,
  };
}

/**
 * Drive a real (mine-bearing) board using only certain moves. Reveals deduced-safe cells
 * and flags deduced mines until solved, stuck (would need a guess), or — if the solver
 * were ever wrong — until a claimed-safe cell explodes (`falsePositive`).
 *
 * This is a verifier, so it is allowed to read `board.mineLayout` to cross-check the
 * solver's mine claims; the solver itself only ever saw the masked {@link SolverView}.
 */
export function solveBoard(board: Board): SolveOutcome {
  let steps = 0;
  for (;;) {
    if (board.status === 'won') return { solved: true, neededGuess: false, steps, falsePositive: false };
    if (board.status === 'lost') return { solved: false, neededGuess: false, steps, falsePositive: true };

    const view = toSolverView(board);
    const { safe, mines } = analyze(view);

    for (const m of mines) {
      if (board.mineLayout[m] !== 1) {
        return { solved: false, neededGuess: false, steps, falsePositive: true };
      }
      if (!board.flagged[m] && !board.revealed[m]) toggleFlag(board, m);
    }

    if (safe.length === 0) {
      return { solved: false, neededGuess: true, steps, falsePositive: false };
    }

    for (const s of safe) {
      if (board.revealed[s] || board.flagged[s]) continue;
      const res = reveal(board, s);
      steps++;
      if (res.hitMine) return { solved: false, neededGuess: false, steps, falsePositive: true };
      if (res.won) break;
    }
  }
}

export interface GenerateOptions {
  /** Flat index of the guaranteed-safe first click. Defaults to the board center. */
  firstClick?: number;
  /** Max reject-sampling attempts before giving up. */
  maxAttempts?: number;
}

/**
 * Reject-sample a board the solver can clear with no guessing (an "NG" board). Returns the
 * board at its start state (mines placed, first click revealed, status `playing`), or null
 * if no such board was found within `maxAttempts`.
 */
export function generateNoGuessBoard(
  spec: BoardSpec,
  rng: Rng,
  options: GenerateOptions = {},
): Board | null {
  const firstClick = options.firstClick ?? Math.floor((spec.rows * spec.cols) / 2);
  const maxAttempts = options.maxAttempts ?? 5000;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const board = createBoard(spec);
    placeMines(board, firstClick, rng, 'safe-area');
    board.status = 'playing';
    reveal(board, firstClick);
    if (solveBoard(cloneBoard(board)).solved) {
      return cloneBoard(board);
    }
  }
  return null;
}

export { toSolverView, type SolverView } from './view';
export { computeProbabilities } from './probability';
