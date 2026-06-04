/**
 * RL-style environment wrapper (파일트리: env.ts; 기술백서 §4.3).
 *
 * Action space = "reveal a hidden cell" (H×W flat indices). Flagging is NOT part of the
 * RL action space — the agent only opens cells; flags are a solver/UI concern. Action
 * masking is mandatory for training (legalActionMask), so unmasked illegal actions only
 * occur by caller error and resolve to a no-op step.
 *
 * Rule parity note (CLAUDE.md §5): the game *rules* here (placement, cascade, win/loss,
 * masking) must stay identical to `server/trainer/env.py` (M4). The reward weights are a
 * convenience default; the canonical training reward lives in `reward.py`.
 */
import {
  createBoard,
  placeMines,
  reveal,
  render,
  toggleFlag,
} from './board';
import { DIFFICULTIES } from './difficulty';
import { mulberry32, type Rng } from './rng';
import type {
  Action,
  Board,
  BoardSpec,
  Difficulty,
  FirstClickPolicy,
  RewardConfig,
  StepResult,
} from './types';

export const DEFAULT_REWARD: RewardConfig = {
  win: 1,
  loss: -1,
  progress: 0.3,
  noProgress: -0.3,
  step: 0,
};

export interface EnvOptions {
  /** A named difficulty. Ignored if `preset` is given. Defaults to `beginner`. */
  difficulty?: Exclude<Difficulty, 'custom'>;
  /** Explicit board spec (overrides `difficulty`). */
  preset?: BoardSpec;
  /** Seed for the board RNG. Defaults to 0. */
  seed?: number;
  /** Reward overrides merged onto {@link DEFAULT_REWARD}. */
  reward?: Partial<RewardConfig>;
  /** First-click safety policy. Defaults to `safe-area`. */
  firstClickPolicy?: FirstClickPolicy;
}

export class MinesweeperEnv {
  readonly spec: BoardSpec;
  private readonly reward: RewardConfig;
  private readonly firstClickPolicy: FirstClickPolicy;
  private seed: number;
  private rng: Rng;
  private board: Board;
  private steps = 0;

  constructor(options: EnvOptions = {}) {
    this.spec = options.preset ?? DIFFICULTIES[options.difficulty ?? 'beginner'];
    this.seed = options.seed ?? 0;
    this.reward = { ...DEFAULT_REWARD, ...options.reward };
    this.firstClickPolicy = options.firstClickPolicy ?? 'safe-area';
    this.rng = mulberry32(this.seed);
    this.board = createBoard(this.spec);
  }

  /** Reset to a fresh board. Pass a seed to change it; otherwise the last seed is reused. */
  reset(seed?: number): Board {
    if (seed !== undefined) this.seed = seed;
    this.rng = mulberry32(this.seed);
    this.board = createBoard(this.spec);
    this.steps = 0;
    return this.board;
  }

  /**
   * Reveal the cell at flat index `action`. On the first step, mines are placed avoiding
   * the clicked cell (first-click safety). Throws if called on a terminal board.
   */
  step(action: Action): StepResult {
    const { status } = this.board;
    if (status === 'won' || status === 'lost') {
      throw new Error('step() called on a terminal board; call reset() first');
    }
    const cells = this.spec.rows * this.spec.cols;
    if (!Number.isInteger(action) || action < 0 || action >= cells) {
      throw new Error(`action out of range: ${action} (expected 0..${cells - 1})`);
    }

    this.steps++;
    if (!this.board.minesPlaced) {
      placeMines(this.board, action, this.rng, this.firstClickPolicy);
      this.board.status = 'playing';
    }

    const result = reveal(this.board, action);

    let reward: number;
    if (result.lost) reward = this.reward.loss;
    else if (result.won) reward = this.reward.win;
    else if (result.ok && result.revealed.length > 0) reward = this.reward.progress;
    else reward = this.reward.noProgress;
    reward += this.reward.step;

    return {
      board: this.board,
      reward,
      done: result.won || result.lost,
      won: result.won,
      lost: result.lost,
      revealed: result.revealed.length,
      info: { steps: this.steps, hitMine: result.hitMine, ok: result.ok },
    };
  }

  /** Toggle a flag (outside the RL action space; for UI/solver use). */
  flag(index: number): boolean {
    return toggleFlag(this.board, index);
  }

  /**
   * Mask of legal actions (1 = revealable). Before the first move every cell is legal;
   * after a terminal state nothing is legal. Otherwise: hidden and not flagged.
   */
  legalActionMask(): Uint8Array {
    const cells = this.spec.rows * this.spec.cols;
    const mask = new Uint8Array(cells);
    const { status } = this.board;
    if (status === 'won' || status === 'lost') return mask; // all zero
    if (status === 'ready') {
      mask.fill(1);
      return mask;
    }
    for (let i = 0; i < cells; i++) {
      mask[i] = !this.board.revealed[i] && !this.board.flagged[i] ? 1 : 0;
    }
    return mask;
  }

  render(): string {
    return render(this.board);
  }

  /** Current board (live reference — treat as read-only). */
  get state(): Board {
    return this.board;
  }

  get status(): Board['status'] {
    return this.board.status;
  }

  get stepCount(): number {
    return this.steps;
  }
}
