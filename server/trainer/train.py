"""Training loop: Double DQN + PER + n-step on the Minesweeper env (기술백서 §2.5, §4.3).

Run:
    python -m trainer.train --difficulty beginner --episodes 200000
    python -m trainer.train --rows 5 --cols 5 --mines 3 --episodes 4000   # sanity overfit

Logs train/eval win-rate, loss, and ε to stdout (flushed) and appends a metrics CSV under
--out so progress can be tailed while running in the background.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from trainer.dqn import DQNAgent, NStepBuffer
from trainer.encoding import NUM_CHANNELS, encode
from trainer.env import DIFFICULTIES, MinesweeperEnv
from trainer.evaluate import evaluate
from trainer.reward import make_reward


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--difficulty", choices=list(DIFFICULTIES), default=None)
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    p.add_argument("--mines", type=int, default=None)
    p.add_argument("--episodes", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--capacity", type=int, default=50_000)
    p.add_argument("--warmup", type=int, default=2_000)
    p.add_argument("--train-freq", type=int, default=1, help="learn once every N env steps")
    p.add_argument("--target-sync", type=int, default=1_000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=60_000)
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--beta-start", type=float, default=0.4)
    p.add_argument("--beta-steps", type=int, default=200_000)
    p.add_argument("--reward", default="shaped")
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate", type=float, default=0.85, help="stop early once eval win-rate ≥ gate")
    p.add_argument("--out", default="storage/checkpoints")
    p.add_argument("--tag", default="run")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def resolve_board(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.rows and args.cols and args.mines is not None:
        return args.rows, args.cols, args.mines
    diff = args.difficulty or "beginner"
    return DIFFICULTIES[diff]


def linear(start: float, end: float, steps: int, t: int) -> float:
    if t >= steps:
        return end
    return start + (end - start) * (t / steps)


def main() -> None:
    args = parse_args()
    rows, cols, mines = resolve_board(args)
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.tag}_metrics.csv"
    ckpt_path = out_dir / f"{args.tag}_best.pt"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["episode", "global_step", "train_wr", "eval_wr", "loss", "eps", "sps"]
        )

    n_actions = rows * cols
    env = MinesweeperEnv(rows, cols, mines, seed=args.seed, reward=make_reward(args.reward))
    agent = DQNAgent(
        NUM_CHANNELS,
        n_actions,
        device=device,
        width=args.width,
        blocks=args.blocks,
        lr=args.lr,
        gamma=args.gamma,
        n_step=args.n_step,
    )
    from trainer.replay import PrioritizedReplay  # local import keeps numpy seeding above tidy

    replay = PrioritizedReplay(
        args.capacity, (NUM_CHANNELS, rows, cols), n_actions, alpha=args.per_alpha
    )
    nstep = NStepBuffer(args.n_step, args.gamma)

    print(
        f"[train] board={rows}x{cols} mines={mines} device={device} "
        f"actions={n_actions} reward={args.reward} gate={args.gate}",
        flush=True,
    )

    recent_wins: deque[int] = deque(maxlen=500)
    global_step = 0
    last_loss = float("nan")
    best_eval = -1.0
    t0 = time.time()

    for episode in range(args.episodes):
        env.reset(seed=args.seed + 1 + episode)
        nstep.clear()
        state = encode(env)
        mask = env.legal_action_mask()
        done = False
        info = {"won": False}
        while not done:
            eps = linear(args.eps_start, args.eps_end, args.eps_decay_steps, global_step)
            action = agent.act(state, mask, eps)
            _, reward, done, info = env.step(action)
            next_state = encode(env)
            next_mask = env.legal_action_mask()
            nstep.push(state, action, reward, next_state, done, next_mask)
            for tr in nstep.pop_ready():
                replay.add(*tr)
            state, mask = next_state, next_mask
            global_step += 1

            if len(replay) >= args.warmup and global_step % args.train_freq == 0:
                beta = linear(args.beta_start, 1.0, args.beta_steps, global_step)
                batch = replay.sample(args.batch, beta)
                last_loss, td = agent.learn(batch, batch["weights"])
                replay.update_priorities(batch["indices"], td)
                if global_step % args.target_sync == 0:
                    agent.sync_target()

        recent_wins.append(1 if info["won"] else 0)

        if episode % args.log_every == 0:
            train_wr = float(np.mean(recent_wins)) if recent_wins else 0.0
            sps = global_step / max(time.time() - t0, 1e-9)
            print(
                f"ep {episode:>7} step {global_step:>9} train_wr {train_wr:.3f} "
                f"loss {last_loss:.4f} eps {eps:.3f} sps {sps:.0f}",
                flush=True,
            )

        if episode > 0 and episode % args.eval_every == 0:
            metrics = evaluate(
                agent, rows, cols, mines, games=args.eval_games, seed_base=10_000_000
            )
            eval_wr = metrics["win_rate"]
            print(
                f"  [eval] ep {episode} eval_wr {eval_wr:.3f} avg_steps {metrics['avg_steps']:.1f}",
                flush=True,
            )
            with csv_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        episode,
                        global_step,
                        f"{np.mean(recent_wins):.4f}",
                        f"{eval_wr:.4f}",
                        f"{last_loss:.4f}",
                        f"{eps:.4f}",
                        f"{global_step / max(time.time() - t0, 1e-9):.0f}",
                    ]
                )
            if eval_wr > best_eval:
                best_eval = eval_wr
                agent.save(
                    str(ckpt_path),
                    meta={
                        "episode": episode,
                        "eval_wr": eval_wr,
                        "rows": rows,
                        "cols": cols,
                        "mines": mines,
                        "in_channels": NUM_CHANNELS,
                        "width": args.width,
                        "blocks": args.blocks,
                    },
                )
                print(f"  [ckpt] saved {ckpt_path.name} eval_wr {eval_wr:.3f}", flush=True)
            if eval_wr >= args.gate:
                print(
                    f"[train] GATE reached: eval_wr {eval_wr:.3f} ≥ {args.gate} at ep {episode}",
                    flush=True,
                )
                break

    print(f"[train] done. best_eval_wr={best_eval:.3f}", flush=True)


if __name__ == "__main__":
    main()
