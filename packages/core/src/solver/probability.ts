/**
 * Per-cell mine probability for the guessing fallback (기술백서 §4.3).
 *
 * Exact combinatorial weighting: each frontier component is enumerated into a polynomial
 * waysByMines[m]; the unconstrained "sea" of size S contributes C(S, s) for s mines.
 * Convolving all component polynomials with the sea polynomial and reading the coefficient
 * of x^R (R = remaining mines) gives the total weight W of all globally-consistent
 * configurations. A cell's mine probability is the weighted fraction of configurations in
 * which it is a mine. This is the standard exact Minesweeper probability; it matches hand
 * calculations on small boards (1-2-1, corner, …). Uses doubles — exact for the small
 * boards in tests; an approximation on very large frontiers (only used to pick a guess).
 */
import { connectedComponents, enumerateComponent, type ComponentEnum, COMPONENT_CAP } from './csp';
import { buildConstraints, type Deductions, type SolverView } from './view';

function binomial(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  const kk = Math.min(k, n - k);
  let result = 1;
  for (let i = 0; i < kk; i++) result = (result * (n - i)) / (i + 1);
  return result;
}

/** Polynomial multiplication: (a ⊗ b)[i+j] += a[i]·b[j]. */
function convolve(a: number[], b: number[]): number[] {
  const out = new Array<number>(a.length + b.length - 1).fill(0);
  for (let i = 0; i < a.length; i++) {
    if (a[i] === 0) continue;
    for (let j = 0; j < b.length; j++) out[i + j]! += a[i]! * b[j]!;
  }
  return out;
}

/**
 * Mine probability for every undetermined cell (frontier + sea). Determined cells are
 * omitted. Cells in skipped (too-large) components fall back to a local estimate.
 */
export function computeProbabilities(
  view: SolverView,
  ded: Deductions,
  cap: number = COMPONENT_CAP,
): Map<number, number> {
  const probs = new Map<number, number>();
  const cons = buildConstraints(view, ded);
  const comps = connectedComponents(cons);

  const enums: ComponentEnum[] = [];
  for (const comp of comps) {
    const e = enumerateComponent(comp.cells, comp.cons, cap);
    if (e && e.total > 0) {
      enums.push(e);
    } else {
      // Too large to enumerate: local per-constraint estimate, then move on.
      for (const con of comp.cons) {
        const p = con.cells.length > 0 ? con.sum / con.cells.length : 0;
        for (const c of con.cells) if (!probs.has(c)) probs.set(c, Math.max(0, Math.min(1, p)));
      }
    }
  }

  const polys = enums.map((e) => e.waysByMines);

  // Sea cells (covered, undetermined, not in any constraint).
  const n = view.rows * view.cols;
  const inComp = new Set<number>();
  for (const e of enums) for (const c of e.cells) inComp.add(c);
  let knownMines = 0;
  const sea: number[] = [];
  for (let i = 0; i < n; i++) {
    if (view.revealed[i]) continue;
    if (ded.mine[i]) {
      knownMines++;
      continue;
    }
    if (ded.safe[i] || inComp.has(i)) continue;
    sea.push(i);
  }
  const seaSize = sea.length;
  const remaining = view.mines - knownMines;

  const seaPoly: number[] = [];
  for (let s = 0; s <= Math.min(seaSize, remaining); s++) seaPoly.push(binomial(seaSize, s));
  if (seaPoly.length === 0) seaPoly.push(1); // seaSize 0 or remaining 0 → [1]

  // Product of all component polynomials, and the full product including the sea.
  let allComp: number[] = [1];
  for (const p of polys) allComp = convolve(allComp, p);
  const full = convolve(allComp, seaPoly);
  const W = remaining >= 0 && remaining < full.length ? full[remaining]! : 0;

  // Frontier component cells.
  for (let ci = 0; ci < enums.length; ci++) {
    const e = enums[ci]!;
    // rest = product of every other component poly, convolved with the sea poly.
    let rest: number[] = [1];
    for (let cj = 0; cj < polys.length; cj++) if (cj !== ci) rest = convolve(rest, polys[cj]!);
    rest = convolve(rest, seaPoly);
    for (let vi = 0; vi < e.cells.length; vi++) {
      const cell = e.cells[vi]!;
      const mineWays = e.mineWaysByVarMines[vi]!;
      let num = 0;
      for (let m = 0; m < mineWays.length; m++) {
        const w = mineWays[m]!;
        if (w === 0) continue;
        const r = remaining - m;
        if (r >= 0 && r < rest.length) num += w * rest[r]!;
      }
      probs.set(cell, W > 0 ? num / W : e.total > 0 ? e.mineCountByVar[vi]! / e.total : 0);
    }
  }

  // Sea probability (every sea cell is symmetric).
  if (seaSize > 0) {
    let expectedSeaMines = 0;
    for (let s = 0; s < seaPoly.length; s++) {
      const a = remaining - s;
      if (a >= 0 && a < allComp.length) expectedSeaMines += s * allComp[a]! * seaPoly[s]!;
    }
    const pSea = W > 0 ? expectedSeaMines / W / seaSize : remaining > 0 ? remaining / (seaSize + inComp.size) : 0;
    for (const c of sea) probs.set(c, pSea);
  }

  return probs;
}
