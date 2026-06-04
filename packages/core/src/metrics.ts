/**
 * Board metrics: 3BV and 3BV/s (파일트리: metrics.ts; 기술백서 §4.3).
 *
 * 3BV (Bechtel's Board Benchmark Value) is the minimum number of left-clicks needed to
 * clear a board without flags or chording:
 *   3BV = (# of openings) + (# of number-cells not touching any opening)
 * where an "opening" is a connected component of 0-cells (8-connectivity). Clicking one
 * cell of an opening floods the whole component and reveals its bordering numbers for
 * free; any numbered cell not adjacent to a 0-cell must be clicked individually.
 */
import { forEachNeighbor } from './board';
import type { Board } from './types';

/**
 * Compute the 3BV of a board. Requires mines to be placed (uses the mine layout, not the
 * revealed state) — it measures the board itself, independent of play progress.
 */
export function compute3BV(board: Board): number {
  if (!board.minesPlaced) {
    throw new Error('compute3BV requires a board with mines placed');
  }
  const { rows, cols, mineLayout, adjacent } = board;
  const cells = rows * cols;
  const isZero = (i: number): boolean => mineLayout[i] === 0 && adjacent[i] === 0;

  // 1) Count connected components of 0-cells (each is one opening / one click).
  const visited = new Uint8Array(cells);
  let openings = 0;
  for (let i = 0; i < cells; i++) {
    if (!isZero(i) || visited[i]) continue;
    openings++;
    const stack: number[] = [i];
    visited[i] = 1;
    while (stack.length > 0) {
      const cell = stack.pop()!;
      forEachNeighbor(cell, rows, cols, (n) => {
        if (isZero(n) && !visited[n]) {
          visited[n] = 1;
          stack.push(n);
        }
      });
    }
  }

  // 2) Numbered cells (non-mine, adjacent>0) not bordering any opening need their own click.
  let isolated = 0;
  for (let i = 0; i < cells; i++) {
    if (mineLayout[i] || adjacent[i] === 0) continue;
    let touchesOpening = false;
    forEachNeighbor(i, rows, cols, (n) => {
      if (isZero(n)) touchesOpening = true;
    });
    if (!touchesOpening) isolated++;
  }

  return openings + isolated;
}

/** 3BV per second. Returns 0 for non-positive elapsed time. */
export function bvPerSecond(bv: number, elapsedMs: number): number {
  if (elapsedMs <= 0) return 0;
  return bv / (elapsedMs / 1000);
}
