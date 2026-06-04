/**
 * @msai/core public API barrel.
 *
 * Pure Minesweeper engine + RL environment + (M2) logic solver. Zero runtime deps,
 * no DOM, no ONNX (CLAUDE.md §5). Consumed as source by @msai/agent and @msai/web.
 */

// Types
export type {
  Difficulty,
  BoardSpec,
  GameStatus,
  FirstClickPolicy,
  Board,
  CellView,
  RevealResult,
  Action,
  RewardConfig,
  StepResult,
} from './types';

// RNG
export { mulberry32, randInt } from './rng';
export type { Rng } from './rng';

// Difficulty presets
export { DIFFICULTIES, presetFor } from './difficulty';

// Board engine
export {
  toIndex,
  toCoords,
  forEachNeighbor,
  createBoard,
  cloneBoard,
  computeAdjacency,
  placeMines,
  setMines,
  safeCellCount,
  reveal,
  toggleFlag,
  cellView,
  render,
} from './board';

// Metrics
export { compute3BV, bvPerSecond } from './metrics';

// Environment
export { MinesweeperEnv, DEFAULT_REWARD } from './env';
export type { EnvOptions } from './env';

// Logic solver (M2)
export {
  analyze,
  solveStep,
  solveBoard,
  generateNoGuessBoard,
  toSolverView,
  computeProbabilities,
} from './solver/index';
export type {
  SolveResult,
  SolveStep,
  SolveOutcome,
  GenerateOptions,
  SolverView,
} from './solver/index';
