import { describe, expect, it } from 'vitest';
import {
  analyze,
  createBoard,
  DIFFICULTIES,
  generateNoGuessBoard,
  mulberry32,
  placeMines,
  reveal,
  solveBoard,
  solveStep,
  type BoardSpec,
  type SolverView,
} from '../src/index';

/** Build a SolverView directly from a sparse map of revealed cell → number (rest covered). */
function makeView(rows: number, cols: number, mines: number, revealed: Record<number, number>): SolverView {
  const n = rows * cols;
  const rev = new Uint8Array(n);
  const flagged = new Uint8Array(n);
  const number = new Int8Array(n).fill(-1);
  for (const [key, value] of Object.entries(revealed)) {
    const i = Number(key);
    rev[i] = 1;
    number[i] = value;
  }
  return { rows, cols, mines, revealed: rev, flagged, number };
}

/** Place a difficulty board, reveal a safe first click, return it ready for the solver. */
function startedBoard(spec: BoardSpec, seed: number, firstClick: number) {
  const board = createBoard(spec);
  placeMines(board, firstClick, mulberry32(seed), 'safe-area');
  board.status = 'playing';
  reveal(board, firstClick);
  return board;
}

describe('deduction rules (sound, on crafted views)', () => {
  it('single-point: a number equal to its hidden-neighbor count marks them all mines', () => {
    // 1x2, cell0 shows "1"; its only neighbor (cell1) must be a mine.
    const result = analyze(makeView(1, 2, 1, { 0: 1 }));
    expect(result.mines).toEqual([1]);
    expect(result.safe).toEqual([]);
  });

  it('single-point: a satisfied number marks its hidden neighbors safe', () => {
    // 1x3, cell1 shows "0"; neighbors 0 and 2 are safe.
    const result = analyze(makeView(1, 3, 0, { 1: 0 }));
    expect(result.safe).toEqual([0, 2]);
    expect(result.mines).toEqual([]);
  });

  it('subset rule deduces the difference cells safe (1-x pattern)', () => {
    //   0 1 2
    //   3 4 5     reveal 3="1" → {0,1}=1 ;  reveal 4="1" → {0,1,2,5}=1
    // {0,1} ⊆ {0,1,2,5} with equal sums ⇒ {2,5} are safe.
    const result = analyze(makeView(2, 3, 1, { 3: 1, 4: 1 }));
    expect([...result.safe].sort((a, b) => a - b)).toEqual([2, 5]);
  });
});

describe('CSP probabilities (match hand calculations)', () => {
  it('1 mine split across two cells → 0.5 each', () => {
    // 1x3, cell1="1"; the single mine is cell0 or cell2.
    const step = solveStep(makeView(1, 3, 1, { 1: 1 }));
    expect(step.safe).toEqual([]);
    expect(step.probabilities?.get(0)).toBeCloseTo(0.5);
    expect(step.probabilities?.get(2)).toBeCloseTo(0.5);
    expect(step.guess?.mineProb).toBeCloseTo(0.5);
  });

  it('1 mine among three frontier cells → 1/3 each', () => {
    // 2x2, cell0="1"; one mine among the other three cells.
    const step = solveStep(makeView(2, 2, 1, { 0: 1 }));
    for (const cell of [1, 2, 3]) expect(step.probabilities?.get(cell)).toBeCloseTo(1 / 3);
    expect(step.guess?.mineProb).toBeCloseTo(1 / 3);
  });

  it('weights frontier vs. sea by combinatorics (1/3 frontier < 1/2 sea)', () => {
    //   0 1 2
    //   3 4 5     cell0="1" ⇒ frontier {1,3,4} holds exactly one mine (1/3 each);
    // total mines 2 ⇒ the other mine is in the sea {2,5} (1/2 each).
    const step = solveStep(makeView(2, 3, 2, { 0: 1 }));
    for (const cell of [1, 3, 4]) expect(step.probabilities?.get(cell)).toBeCloseTo(1 / 3);
    for (const cell of [2, 5]) expect(step.probabilities?.get(cell)).toBeCloseTo(1 / 2);
    expect(step.guess?.mineProb).toBeCloseTo(1 / 3);
    expect([1, 3, 4]).toContain(step.guess?.cell);
  });
});

describe('DoD: false-positive 0 (the invariant)', () => {
  it('never reveals a mine it called safe, across many random boards', () => {
    let boards = 0;
    let falsePositives = 0;
    let solved = 0;
    const cases: Array<[keyof typeof DIFFICULTIES, number]> = [
      ['beginner', 300],
      ['intermediate', 80],
      ['expert', 40], // exercises large frontiers / CSP component cap
    ];
    for (const [diff, seeds] of cases) {
      const spec = DIFFICULTIES[diff];
      const first = Math.floor((spec.rows * spec.cols) / 2);
      for (let seed = 0; seed < seeds; seed++) {
        const board = startedBoard(spec, seed * 11 + 1, first);
        const outcome = solveBoard(board);
        boards++;
        if (outcome.falsePositive) falsePositives++;
        if (outcome.solved) solved++;
      }
    }
    // The hard invariant: not a single unsound deduction.
    expect(falsePositives).toBe(0);
    // Sanity: the solver actually solves a meaningful share of boards (not trivially stuck).
    expect(boards).toBeGreaterThan(300);
    expect(solved).toBeGreaterThan(0);
  });
});

describe('DoD: no-guess board generation + solver clears 100%', () => {
  it('generates beginner NG boards the solver clears with no guess', () => {
    const rng = mulberry32(20260604);
    const spec = DIFFICULTIES.beginner;
    const first = Math.floor((spec.rows * spec.cols) / 2);
    let made = 0;
    for (let i = 0; i < 10; i++) {
      const board = generateNoGuessBoard(spec, rng, { firstClick: first, maxAttempts: 5000 });
      expect(board).not.toBeNull();
      if (!board) continue;
      made++;
      const outcome = solveBoard(board);
      expect(outcome.solved).toBe(true);
      expect(outcome.neededGuess).toBe(false);
      expect(outcome.falsePositive).toBe(false);
    }
    expect(made).toBe(10);
  });
});
