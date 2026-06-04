import { describe, expect, it } from 'vitest';
import { DIFFICULTIES, MinesweeperEnv } from '../src/index';

describe('MinesweeperEnv', () => {
  it('reset yields a ready board with every action legal', () => {
    const env = new MinesweeperEnv({ difficulty: 'beginner', seed: 1 });
    env.reset();
    expect(env.status).toBe('ready');
    const mask = env.legalActionMask();
    expect(mask).toHaveLength(81);
    expect([...mask].every((v) => v === 1)).toBe(true);
  });

  it('first step places mines safely and starts the game', () => {
    const env = new MinesweeperEnv({ difficulty: 'beginner', seed: 3 });
    const res = env.step(40);
    expect(res.lost).toBe(false);
    expect(env.state.minesPlaced).toBe(true);
    expect(env.state.mineLayout[40]).toBe(0);
    expect([...env.state.mineLayout].reduce((a, v) => a + v, 0)).toBe(10);
  });

  it('is deterministic: same seed + same first action → identical layout', () => {
    const a = new MinesweeperEnv({ difficulty: 'beginner', seed: 99 });
    const b = new MinesweeperEnv({ difficulty: 'beginner', seed: 99 });
    a.step(20);
    b.step(20);
    expect([...a.state.mineLayout]).toEqual([...b.state.mineLayout]);
  });

  it('legalActionMask drops revealed cells', () => {
    const env = new MinesweeperEnv({ difficulty: 'beginner', seed: 7 });
    env.step(40);
    const mask = env.legalActionMask();
    for (let i = 0; i < 81; i++) {
      if (env.state.revealed[i]) expect(mask[i]).toBe(0);
    }
    expect(env.state.revealed[40]).toBe(1);
    expect(mask[40]).toBe(0);
  });

  it('throws when stepping a terminal board, and masks everything once over', () => {
    // 1x2 with 1 mine: first click is safe (the lone other cell holds the mine),
    // revealing the single safe cell wins immediately.
    const env = new MinesweeperEnv({ preset: { rows: 1, cols: 2, mines: 1 }, seed: 0 });
    const res = env.step(0);
    expect(res.won).toBe(true);
    expect(env.status).toBe('won');
    expect([...env.legalActionMask()].every((v) => v === 0)).toBe(true);
    expect(() => env.step(1)).toThrow();
  });

  it('rejects out-of-range actions', () => {
    const env = new MinesweeperEnv({ difficulty: 'beginner' });
    expect(() => env.step(-1)).toThrow();
    expect(() => env.step(81)).toThrow();
  });

  it('render returns one line per row', () => {
    const env = new MinesweeperEnv({ difficulty: 'beginner', seed: 1 });
    env.step(40);
    expect(env.render().split('\n')).toHaveLength(9);
  });

  it('exposes the standard difficulty presets', () => {
    expect(DIFFICULTIES.beginner).toEqual({ rows: 9, cols: 9, mines: 10 });
    expect(DIFFICULTIES.intermediate).toEqual({ rows: 16, cols: 16, mines: 40 });
    expect(DIFFICULTIES.expert).toEqual({ rows: 16, cols: 30, mines: 99 });
  });
});
