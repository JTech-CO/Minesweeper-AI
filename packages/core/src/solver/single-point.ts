/**
 * Single-point rule (기술백서 §4.3): for a revealed number, if its remaining mine count
 * (number minus already-known mines) is 0, every undetermined neighbor is safe; if it
 * equals the undetermined-neighbor count, they are all mines. Sound by construction.
 */
import { buildConstraints, type Deductions, type SolverView } from './view';

/** Apply the single-point rule once across the board. Returns true if anything changed. */
export function singlePoint(view: SolverView, ded: Deductions): boolean {
  const constraints = buildConstraints(view, ded);
  let changed = false;
  for (const con of constraints) {
    if (con.sum === 0) {
      for (const cell of con.cells) {
        if (!ded.safe[cell]) {
          ded.safe[cell] = 1;
          changed = true;
        }
      }
    } else if (con.sum === con.cells.length) {
      for (const cell of con.cells) {
        if (!ded.mine[cell]) {
          ded.mine[cell] = 1;
          changed = true;
        }
      }
    }
  }
  return changed;
}
