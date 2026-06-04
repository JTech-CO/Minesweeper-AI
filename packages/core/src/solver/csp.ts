/**
 * CSP frontier enumeration (기술백서 §4.3).
 *
 * The undetermined frontier is split into independent connected components (cells linked
 * by shared constraints). Each small component is enumerated exhaustively: a cell that is
 * a mine in *every* valid assignment is a certain mine; one that is a mine in *none* is
 * certainly safe. A global mine-budget bound (frontier min/max vs. remaining mines) lets
 * us also resolve the unconstrained "sea". All deductions here are sound (never a false
 * positive); enumeration is capped so a pathological component is skipped rather than
 * exploding — skipping only forgoes a deduction, it never produces a wrong one.
 */
import {
  buildConstraints,
  type Constraint,
  type Deductions,
  type SolverView,
} from './view';

/** Default cap on component size (variables). Above this, the component is left to guessing. */
export const COMPONENT_CAP = 20;

export interface ComponentEnum {
  /** Global cell indices, length V. */
  cells: number[];
  /** Number of valid assignments. */
  total: number;
  /** Per-variable count of assignments where it is a mine, length V. */
  mineCountByVar: number[];
  /** waysByMines[m] = number of assignments using exactly m mines, length V+1. */
  waysByMines: number[];
  /** mineWaysByVarMines[v][m] = assignments using m mines where var v is a mine. */
  mineWaysByVarMines: number[][];
}

/** Group constraints into connected components by shared cells (union-find). */
export function connectedComponents(cons: Constraint[]): { cells: number[]; cons: Constraint[] }[] {
  const parent = new Map<number, number>();
  const find = (x: number): number => {
    let root = x;
    while (parent.get(root) !== root) root = parent.get(root)!;
    while (parent.get(x) !== root) {
      const next = parent.get(x)!;
      parent.set(x, root);
      x = next;
    }
    return root;
  };
  const union = (a: number, b: number): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };

  for (const con of cons) {
    for (const c of con.cells) if (!parent.has(c)) parent.set(c, c);
    for (let i = 1; i < con.cells.length; i++) union(con.cells[0]!, con.cells[i]!);
  }

  const cellsByRoot = new Map<number, number[]>();
  for (const c of parent.keys()) {
    const r = find(c);
    let list = cellsByRoot.get(r);
    if (!list) {
      list = [];
      cellsByRoot.set(r, list);
    }
    list.push(c);
  }
  const consByRoot = new Map<number, Constraint[]>();
  for (const con of cons) {
    const r = find(con.cells[0]!);
    let list = consByRoot.get(r);
    if (!list) {
      list = [];
      consByRoot.set(r, list);
    }
    list.push(con);
  }

  const comps: { cells: number[]; cons: Constraint[] }[] = [];
  for (const [r, cells] of cellsByRoot) comps.push({ cells, cons: consByRoot.get(r) ?? [] });
  return comps;
}

/**
 * Exhaustively enumerate one component's valid mine assignments via pruned backtracking.
 * Returns null if the component exceeds `cap` variables (left to the guessing layer).
 */
export function enumerateComponent(
  cells: number[],
  cons: Constraint[],
  cap: number = COMPONENT_CAP,
): ComponentEnum | null {
  const V = cells.length;
  if (V > cap) return null;

  const local = new Map<number, number>();
  cells.forEach((c, i) => local.set(c, i));
  const conVars = cons.map((con) => con.cells.map((c) => local.get(c)!));
  const conSum = cons.map((con) => con.sum);
  const inCons: number[][] = Array.from({ length: V }, () => []);
  conVars.forEach((vars, ci) => vars.forEach((v) => inCons[v]!.push(ci)));

  const assign = new Int8Array(V);
  const conMines = new Int32Array(cons.length);
  const conUnassigned = conVars.map((vars) => vars.length);

  let total = 0;
  const mineCountByVar = new Array<number>(V).fill(0);
  const waysByMines = new Array<number>(V + 1).fill(0);
  const mineWaysByVarMines = Array.from({ length: V }, () => new Array<number>(V + 1).fill(0));

  const dfs = (v: number, mines: number): void => {
    if (v === V) {
      total++;
      waysByMines[mines]!++;
      for (let i = 0; i < V; i++) {
        if (assign[i] === 1) {
          mineCountByVar[i]!++;
          mineWaysByVarMines[i]![mines]!++;
        }
      }
      return;
    }
    const affected = inCons[v]!;
    for (let val = 0; val <= 1; val++) {
      assign[v] = val as 0 | 1;
      for (const ci of affected) {
        conMines[ci]! += val;
        conUnassigned[ci]!--;
      }
      let ok = true;
      for (const ci of affected) {
        // Prune: too many mines already, or not enough remaining slots to reach the target.
        if (conMines[ci]! > conSum[ci]! || conMines[ci]! + conUnassigned[ci]! < conSum[ci]!) {
          ok = false;
          break;
        }
      }
      if (ok) dfs(v + 1, mines + val);
      for (const ci of affected) {
        conMines[ci]! -= val;
        conUnassigned[ci]!++;
      }
    }
    assign[v] = 0;
  };
  dfs(0, 0);

  return { cells, total, mineCountByVar, waysByMines, mineWaysByVarMines };
}

/**
 * Derive certain safe/mine cells via per-component enumeration plus a global mine-budget
 * bound on the unconstrained sea. Returns true if any new deduction was made.
 */
export function cspCertain(view: SolverView, ded: Deductions, cap: number = COMPONENT_CAP): boolean {
  const cons = buildConstraints(view, ded);
  if (cons.length === 0) return false;

  const comps = connectedComponents(cons);
  let changed = false;
  let frontierMin = 0;
  let frontierMax = 0;
  const inFrontier = new Set<number>();

  for (const comp of comps) {
    const e = enumerateComponent(comp.cells, comp.cons, cap);
    for (const c of comp.cells) inFrontier.add(c);
    if (!e || e.total === 0) {
      // Unenumerable (too large) — stay sound by assuming the widest mine range.
      frontierMax += comp.cells.length;
      continue;
    }
    for (let i = 0; i < e.cells.length; i++) {
      const cell = e.cells[i]!;
      const mc = e.mineCountByVar[i]!;
      if (mc === 0) {
        if (!ded.safe[cell]) {
          ded.safe[cell] = 1;
          changed = true;
        }
      } else if (mc === e.total) {
        if (!ded.mine[cell]) {
          ded.mine[cell] = 1;
          changed = true;
        }
      }
    }
    let mn = Infinity;
    let mx = 0;
    for (let m = 0; m < e.waysByMines.length; m++) {
      if (e.waysByMines[m]! > 0) {
        if (m < mn) mn = m;
        if (m > mx) mx = m;
      }
    }
    frontierMin += mn === Infinity ? 0 : mn;
    frontierMax += mx;
  }

  // Sea = covered, undetermined cells not adjacent to any number.
  // Mines KNOWN OUTSIDE the frontier are counted here; mines inside the frontier are
  // already accounted for by frontierMin/Max. Counting frontier mines in both places
  // (e.g. cells just marked certain-mine above) would double-subtract them from the
  // remaining budget and make the sea bound unsound.
  const n = view.rows * view.cols;
  let knownOutsideFrontier = 0;
  const sea: number[] = [];
  for (let i = 0; i < n; i++) {
    if (view.revealed[i]) continue;
    if (inFrontier.has(i)) continue; // handled by frontierMin/Max
    if (ded.mine[i]) {
      knownOutsideFrontier++;
      continue;
    }
    if (ded.safe[i]) continue;
    sea.push(i);
  }
  if (sea.length > 0) {
    // mines among (frontier ∪ sea) = total − mines known outside the frontier.
    const remaining = view.mines - knownOutsideFrontier;
    // sea mines = remaining − (frontier mines ∈ [frontierMin, frontierMax]).
    const seaMax = remaining - frontierMin;
    const seaMin = remaining - frontierMax;
    if (seaMax <= 0) {
      for (const c of sea) {
        if (!ded.safe[c]) {
          ded.safe[c] = 1;
          changed = true;
        }
      }
    } else if (seaMin >= sea.length) {
      for (const c of sea) {
        if (!ded.mine[c]) {
          ded.mine[c] = 1;
          changed = true;
        }
      }
    }
  }

  return changed;
}
