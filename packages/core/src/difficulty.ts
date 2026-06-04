/**
 * Standard difficulty presets (기술백서 §4.3, §6.2).
 *
 * Convention: `rows` = height, `cols` = width. "30×16" (width×height) Expert therefore
 * maps to rows=16, cols=30.
 */
import type { BoardSpec, Difficulty } from './types';

export const DIFFICULTIES: Record<Exclude<Difficulty, 'custom'>, BoardSpec> = {
  beginner: { rows: 9, cols: 9, mines: 10 },
  intermediate: { rows: 16, cols: 16, mines: 40 },
  expert: { rows: 16, cols: 30, mines: 99 },
};

/** Resolve a difficulty name to its preset. Throws for `custom` (pass a spec directly). */
export function presetFor(difficulty: Exclude<Difficulty, 'custom'>): BoardSpec {
  return DIFFICULTIES[difficulty];
}
