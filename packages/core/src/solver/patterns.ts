/**
 * Subset rule (기술백서 §4.3 "부분집합 규칙", generalizes 1-2-1 / 1-2-2-1).
 *
 * For two constraints A ⊆ B over their undetermined neighbor sets, the cells in B\A must
 * contain exactly (sumB − sumA) mines. So if sumB − sumA == 0 they are all safe; if it
 * equals |B\A| they are all mines. This subsumes the classic 1-2-1 / 1-2-2-1 patterns and
 * is sound. Only constraints that share a cell can be subsets, so we scan candidate
 * supersets via a cell→constraint index instead of all pairs.
 */
import { buildConstraints, type Deductions, type SolverView } from './view';

/** Apply the subset rule once across overlapping constraint pairs. Returns true if changed. */
export function subsetRule(view: SolverView, ded: Deductions): boolean {
  const cons = buildConstraints(view, ded);
  const sets = cons.map((c) => new Set(c.cells));
  const byCell = new Map<number, number[]>();
  cons.forEach((con, ci) => {
    for (const cell of con.cells) {
      let list = byCell.get(cell);
      if (!list) {
        list = [];
        byCell.set(cell, list);
      }
      list.push(ci);
    }
  });

  let changed = false;
  for (let i = 0; i < cons.length; i++) {
    const a = cons[i]!;
    const aSet = sets[i]!;
    const considered = new Set<number>();
    for (const cell of a.cells) {
      for (const cj of byCell.get(cell)!) {
        if (cj === i || considered.has(cj)) continue;
        considered.add(cj);
        const b = cons[cj]!;
        if (b.cells.length <= a.cells.length) continue; // want strict superset
        const bSet = sets[cj]!;
        let isSubset = true;
        for (const x of a.cells) {
          if (!bSet.has(x)) {
            isSubset = false;
            break;
          }
        }
        if (!isSubset) continue;
        const diff = b.cells.filter((x) => !aSet.has(x));
        if (diff.length === 0) continue;
        const dRemaining = b.sum - a.sum;
        if (dRemaining === 0) {
          for (const c of diff)
            if (!ded.safe[c]) {
              ded.safe[c] = 1;
              changed = true;
            }
        } else if (dRemaining === diff.length) {
          for (const c of diff)
            if (!ded.mine[c]) {
              ded.mine[c] = 1;
              changed = true;
            }
        }
      }
    }
  }
  return changed;
}
