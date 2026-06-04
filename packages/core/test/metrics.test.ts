import { describe, expect, it } from 'vitest';
import { bvPerSecond, compute3BV, createBoard, setMines } from '../src/index';

describe('compute3BV (known small cases)', () => {
  it('corner mine on 3x3 → 3BV 1 (one opening clears everything)', () => {
    // M 1 .
    // 1 1 .
    // . . .
    const b = createBoard({ rows: 3, cols: 3, mines: 1 });
    setMines(b, [0]);
    expect(compute3BV(b)).toBe(1);
  });

  it('center mine on 3x3 → 3BV 8 (no opening, every number clicked individually)', () => {
    // 1 1 1
    // 1 M 1
    // 1 1 1
    const b = createBoard({ rows: 3, cols: 3, mines: 1 });
    setMines(b, [4]);
    expect(compute3BV(b)).toBe(8);
  });

  it('1x5 split board → 3BV 2 (two separate openings)', () => {
    // 0 1 M 1 0
    const b = createBoard({ rows: 1, cols: 5, mines: 1 });
    setMines(b, [2]);
    expect(compute3BV(b)).toBe(2);
  });

  it('mine-free board → 3BV 1 (single opening)', () => {
    const b = createBoard({ rows: 3, cols: 3, mines: 0 });
    setMines(b, []);
    expect(compute3BV(b)).toBe(1);
  });

  it('requires mines to be placed', () => {
    const b = createBoard({ rows: 3, cols: 3, mines: 1 });
    expect(() => compute3BV(b)).toThrow();
  });
});

describe('bvPerSecond', () => {
  it('divides 3BV by elapsed seconds', () => {
    expect(bvPerSecond(2, 1000)).toBeCloseTo(2);
    expect(bvPerSecond(2, 500)).toBeCloseTo(4);
  });

  it('returns 0 for non-positive elapsed time', () => {
    expect(bvPerSecond(10, 0)).toBe(0);
    expect(bvPerSecond(10, -5)).toBe(0);
  });
});
