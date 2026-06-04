import { describe, expect, it } from 'vitest';
import {
  createBoard,
  forEachNeighbor,
  mulberry32,
  placeMines,
  reveal,
  safeCellCount,
  setMines,
  toggleFlag,
  type Board,
} from '../src/index';

/** Flat indices of all currently-revealed cells. */
function revealedIndices(board: Board): number[] {
  const out: number[] = [];
  for (let i = 0; i < board.revealed.length; i++) if (board.revealed[i]) out.push(i);
  return out;
}

describe('createBoard', () => {
  it('starts in the ready state with no mines placed', () => {
    const b = createBoard({ rows: 9, cols: 9, mines: 10 });
    expect(b.status).toBe('ready');
    expect(b.minesPlaced).toBe(false);
    expect(b.mineLayout).toHaveLength(81);
    expect([...b.mineLayout].every((v) => v === 0)).toBe(true);
    expect(safeCellCount(b)).toBe(71);
  });

  it('rejects invalid dimensions and mine counts', () => {
    expect(() => createBoard({ rows: 0, cols: 5, mines: 1 })).toThrow();
    expect(() => createBoard({ rows: 3, cols: 3, mines: 9 })).toThrow(); // need >=1 safe cell
    expect(() => createBoard({ rows: 3, cols: 3, mines: -1 })).toThrow();
  });
});

describe('forEachNeighbor', () => {
  it('clamps to the grid at corners and edges', () => {
    const collect = (i: number): number[] => {
      const out: number[] = [];
      forEachNeighbor(i, 3, 3, (n) => out.push(n));
      return out.sort((a, b) => a - b);
    };
    expect(collect(0)).toEqual([1, 3, 4]); // top-left corner → 3 neighbors
    expect(collect(4)).toEqual([0, 1, 2, 3, 5, 6, 7, 8]); // center → 8
    expect(collect(8)).toEqual([4, 5, 7]); // bottom-right corner → 3
  });
});

describe('mine placement (determinism + first-click safety)', () => {
  it('same seed + same first click → identical mine layout', () => {
    const make = (): Uint8Array => {
      const b = createBoard({ rows: 9, cols: 9, mines: 10 });
      placeMines(b, 40, mulberry32(2026), 'safe-area');
      return b.mineLayout;
    };
    expect([...make()]).toEqual([...make()]);
  });

  it('places exactly the requested number of mines', () => {
    const b = createBoard({ rows: 16, cols: 30, mines: 99 });
    placeMines(b, 0, mulberry32(5), 'safe-area');
    expect([...b.mineLayout].reduce((a, v) => a + v, 0)).toBe(99);
  });

  it('safe-area keeps the clicked cell AND its neighbors mine-free (guarantees an opening)', () => {
    const click = 40; // center of a 9x9
    for (let seed = 0; seed < 200; seed++) {
      const b = createBoard({ rows: 9, cols: 9, mines: 10 });
      placeMines(b, click, mulberry32(seed), 'safe-area');
      expect(b.mineLayout[click]).toBe(0);
      forEachNeighbor(click, 9, 9, (n) => expect(b.mineLayout[n]).toBe(0));
      expect(b.adjacent[click]).toBe(0); // 0-cell ⇒ first click opens a cascade
    }
  });

  it('safe-cell policy keeps only the clicked cell mine-free', () => {
    let everNumbered = false;
    for (let seed = 0; seed < 200; seed++) {
      const b = createBoard({ rows: 9, cols: 9, mines: 10 });
      placeMines(b, 40, mulberry32(seed), 'safe-cell');
      expect(b.mineLayout[40]).toBe(0);
      if (b.adjacent[40]! > 0) everNumbered = true;
    }
    // With only the cell itself protected, the first click is sometimes a number.
    expect(everNumbered).toBe(true);
  });
});

describe('reveal: cascade, win, loss', () => {
  it('cascades through a connected 0-region and clears the board (win)', () => {
    // 3x3, single mine at corner (idx 0):
    //   M 1 .
    //   1 1 .
    //   . . .
    const b = createBoard({ rows: 3, cols: 3, mines: 1 });
    setMines(b, [0]);
    const res = reveal(b, 8); // click a 0-cell
    expect(res.hitMine).toBe(false);
    expect(res.won).toBe(true);
    expect(b.status).toBe('won');
    // Every safe cell (all but the mine at 0) is revealed in one click.
    expect(revealedIndices(b)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(b.safeRevealed).toBe(8);
  });

  it('a 0-cell flood reveals each bordering opening separately (partial reveal)', () => {
    // 1x5: 0 1 M 1 0  (mine at idx 2)
    const b = createBoard({ rows: 1, cols: 5, mines: 1 });
    setMines(b, [2]);
    const first = reveal(b, 0); // left opening
    expect(first.won).toBe(false);
    expect(revealedIndices(b)).toEqual([0, 1]);
    const second = reveal(b, 4); // right opening completes the board
    expect(second.won).toBe(true);
    expect(revealedIndices(b)).toEqual([0, 1, 3, 4]);
  });

  it('revealing a mine loses and records the explosion', () => {
    const b = createBoard({ rows: 3, cols: 3, mines: 1 });
    setMines(b, [0]);
    const res = reveal(b, 0);
    expect(res.hitMine).toBe(true);
    expect(res.lost).toBe(true);
    expect(b.status).toBe('lost');
    expect(b.explodedAt).toBe(0);
  });

  it('is a no-op on already-revealed cells and after the game ends', () => {
    const b = createBoard({ rows: 3, cols: 3, mines: 1 });
    setMines(b, [0]);
    reveal(b, 4); // numbered cell (1), no cascade
    expect(reveal(b, 4).ok).toBe(false); // already revealed
    reveal(b, 0); // hit mine → lost
    expect(reveal(b, 5).ok).toBe(false); // game over
  });
});

describe('flags', () => {
  it('a flag blocks a cascade from opening that cell', () => {
    const b = createBoard({ rows: 3, cols: 3, mines: 1 });
    setMines(b, [0]);
    expect(toggleFlag(b, 1)).toBe(true); // flag the "1" at idx 1
    const res = reveal(b, 8);
    expect(res.won).toBe(false); // idx 1 stays hidden, so not all safe cells open
    expect(b.revealed[1]).toBe(0);
    expect(b.flagged[1]).toBe(1);
    expect(revealedIndices(b)).toEqual([2, 3, 4, 5, 6, 7, 8]);
  });

  it('cannot flag a revealed cell', () => {
    const b = createBoard({ rows: 3, cols: 3, mines: 1 });
    setMines(b, [0]);
    reveal(b, 4);
    expect(toggleFlag(b, 4)).toBe(false);
  });
});
