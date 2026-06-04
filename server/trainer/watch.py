"""Live terminal viewer for training + play (dev observability for M4).

Run in a second terminal while `trainer.train` runs in the background:

    python -m trainer.watch --tag beginner            # watch the Beginner run
    python -m trainer.watch --tag sanity --delay 0.15  # watch the sanity model

Left panel: the current best checkpoint plays boards one move at a time (greedy, masked),
classic Minesweeper colours, the chosen cell highlighted. Right panel: latest training
metrics parsed live from `<tag>.out` / `<tag>_metrics.csv`, with an eval win-rate sparkline.
The checkpoint is reloaded between games so the play reflects ongoing training. A real web
dashboard arrives in M8; this is a lightweight stand-in.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trainer.dqn import DQNAgent
from trainer.encoding import NUM_CHANNELS, encode
from trainer.env import DIFFICULTIES, MinesweeperEnv

# Classic Minesweeper number colours.
NUM_COLORS = {
    1: "blue",
    2: "green",
    3: "red",
    4: "magenta",
    5: "dark_red",
    6: "cyan",
    7: "grey85",
    8: "grey50",
}
SPARK = "▁▂▃▄▅▆▇█"
_EP_RE = re.compile(
    r"ep\s+(\d+)\s+step\s+(\d+)\s+train_wr\s+([\d.]+)\s+loss\s+([\d.nan]+)\s+eps\s+([\d.]+)\s+sps\s+([\d.]+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="beginner")
    p.add_argument("--dir", default="storage/checkpoints")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--difficulty", choices=list(DIFFICULTIES), default="beginner")
    p.add_argument("--delay", type=float, default=0.18, help="seconds between moves")
    p.add_argument("--end-delay", type=float, default=1.2, help="seconds to hold a finished board")
    p.add_argument("--max-games", type=int, default=None, help="exit after N games (testing)")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--no-screen", action="store_true", help="inline render (no alt-screen)")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def sparkline(values: list[float], width: int = 28) -> str:
    if not values:
        return ""
    vals = values[-width:]
    return "".join(
        SPARK[min(len(SPARK) - 1, max(0, int(round(v * (len(SPARK) - 1)))))] for v in vals
    )


def read_progress(log_path: Path, csv_path: Path) -> dict:
    """Parse the freshest train line from the log and the eval series from the CSV."""
    info: dict = {"evals": [], "updated_ago": None}
    if csv_path.exists():
        try:
            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            info["evals"] = [float(r["eval_wr"]) for r in rows if r.get("eval_wr")]
            if rows:
                last = rows[-1]
                info["last_eval_wr"] = float(last["eval_wr"])
                info["last_eval_ep"] = int(last["episode"])
        except (OSError, ValueError, KeyError):
            pass
    if log_path.exists():
        try:
            info["updated_ago"] = max(0.0, time.time() - log_path.stat().st_mtime)
            with log_path.open(encoding="utf-8", errors="ignore") as f:
                tail = f.readlines()[-40:]
            for line in reversed(tail):
                m = _EP_RE.search(line)
                if m:
                    info.update(
                        episode=int(m.group(1)),
                        global_step=int(m.group(2)),
                        train_wr=float(m.group(3)),
                        loss=m.group(4),
                        eps=float(m.group(5)),
                        sps=float(m.group(6)),
                    )
                    break
        except OSError:
            pass
    return info


def load_agent(ckpt_path: Path, device: torch.device, current: tuple) -> tuple:
    """Reload the checkpoint if present/changed. Returns (agent, board, meta, signature)."""
    agent, board, meta, sig = current
    if not ckpt_path.exists():
        return current
    try:
        new_sig = (ckpt_path.stat().st_mtime, ckpt_path.stat().st_size)
        if new_sig == sig:
            return current
        tmp = ckpt_path.with_suffix(".view.tmp")
        shutil.copyfile(ckpt_path, tmp)  # read a copy to minimise locking the live file
        ckpt = torch.load(tmp, map_location=device, weights_only=False)
        tmp.unlink(missing_ok=True)
        m = ckpt["meta"]
        new_agent = DQNAgent(
            m.get("in_channels", NUM_CHANNELS),
            m["rows"] * m["cols"],
            device=device,
            width=m["width"],
            blocks=m["blocks"],
        )
        new_agent.online.load_state_dict(ckpt["online"])
        new_agent.online.eval()
        return new_agent, (m["rows"], m["cols"], m["mines"]), m, new_sig
    except (OSError, KeyError, RuntimeError):
        return current


def render_board(env: MinesweeperEnv, last_action: int) -> Text:
    t = Text(justify="left")
    for r in range(env.rows):
        for c in range(env.cols):
            i = r * env.cols + c
            if env.revealed[i]:
                if env.mine_layout[i]:
                    ch, style = (
                        ("✸ ", "bold white on red") if env.exploded_at == i else ("✷ ", "red")
                    )
                else:
                    a = int(env.adjacent[i])
                    ch, style = (
                        ("· ", "grey37") if a == 0 else (f"{a} ", NUM_COLORS.get(a, "white"))
                    )
            elif env.flagged[i]:
                ch, style = "⚑ ", "yellow"
            else:
                ch, style = "■ ", "grey50"
            if i == last_action and not (env.revealed[i] and env.mine_layout[i]):
                style += " reverse"
            t.append(ch, style=style)
        t.append("\n")
    return t


def render_stats(prog: dict, meta: dict | None, games: int, wins: int, playing_wr: float) -> Table:
    g = Table.grid(padding=(0, 2))
    g.add_column(justify="right", style="grey70")
    g.add_column()

    def row(label: str, value: str) -> None:
        g.add_row(label, value)

    row("[bold]training[/]", "")
    row("episode", f"{prog.get('episode', '—')}")
    row("global step", f"{prog.get('global_step', '—')}")
    row("train win%", f"{prog.get('train_wr', float('nan')):.3f}" if "train_wr" in prog else "—")
    last_eval = prog.get("last_eval_wr")
    row("eval win%", f"[bold green]{last_eval:.3f}[/]" if last_eval is not None else "—")
    row("eval trend", f"[green]{sparkline(prog.get('evals', []))}[/]")
    row("loss", f"{prog.get('loss', '—')}")
    row("epsilon", f"{prog.get('eps', float('nan')):.3f}" if "eps" in prog else "—")
    row("steps/s", f"{prog.get('sps', float('nan')):.0f}" if "sps" in prog else "—")
    ago = prog.get("updated_ago")
    row("log age", f"{ago:.0f}s ago" if ago is not None else "—")
    g.add_row("", "")
    row("[bold]checkpoint[/]", "")
    if meta:
        row("ckpt eval%", f"{meta.get('eval_wr', float('nan')):.3f}")
        row("ckpt ep", f"{meta.get('episode', '—')}")
    else:
        row("ckpt", "[yellow]waiting…[/]")
    g.add_row("", "")
    row("[bold]viewer play[/]", "")
    row("games", f"{games}")
    row("won / lost", f"[green]{wins}[/] / [red]{games - wins}[/]")
    row("win%", f"{playing_wr:.3f}")
    return g


def update_frame(
    layout: Layout,
    env: MinesweeperEnv,
    last_action: int,
    board: tuple,
    games: int,
    wins: int,
    meta: dict | None,
    prog: dict,
    status_note: str,
) -> None:
    rows, cols, mines = board
    completed = games - 1
    pwr = wins / completed if completed > 0 else 0.0
    layout["header"].update(
        Panel(
            Align.center(
                Text.from_markup(
                    f"[bold]Minesweeper AI — live[/]   board [cyan]{rows}×{cols}[/] "
                    f"mines [cyan]{mines}[/]   game [bold]{games}[/]   {status_note}"
                )
            )
        )
    )
    layout["board"].update(
        Panel(Align.center(render_board(env, last_action)), title="play (greedy)")
    )
    layout["stats"].update(Panel(render_stats(prog, meta, completed, wins, pwr), title="metrics"))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = Path(args.dir)
    ckpt_path = Path(args.ckpt) if args.ckpt else base / f"{args.tag}_best.pt"
    log_path = base / f"{args.tag}.out"
    csv_path = base / f"{args.tag}_metrics.csv"

    default_board = DIFFICULTIES[args.difficulty]
    state = (None, default_board, None, None)  # agent, board, meta, signature

    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"))
    layout["body"].split_row(Layout(name="board", ratio=3), Layout(name="stats", ratio=2))

    games = wins = 0
    last_action = -1

    with Live(layout, screen=not args.no_screen, refresh_per_second=12, transient=False) as live:
        while args.max_games is None or games < args.max_games:
            state = load_agent(ckpt_path, device, state)
            agent, board, meta, _ = state
            rows, cols, mines = board
            games += 1
            env = MinesweeperEnv(rows, cols, mines, seed=args.seed + games)
            env.reset()
            last_action = -1

            update_frame(
                layout,
                env,
                last_action,
                board,
                games,
                wins,
                meta,
                read_progress(log_path, csv_path),
                "[grey70]start[/]",
            )
            live.refresh()
            guard = 0
            while env.status in ("ready", "playing") and guard < rows * cols + 5:
                guard += 1
                mask = env.legal_action_mask()
                if agent is not None:
                    action = agent.act(encode(env), mask, eps=0.0)
                else:
                    action = int(np.random.choice(np.flatnonzero(mask)))
                env.step(action)
                last_action = action
                note = {"won": "[bold green]WON[/]", "lost": "[bold red]LOST[/]"}.get(
                    env.status, "playing"
                )
                update_frame(
                    layout,
                    env,
                    last_action,
                    board,
                    games,
                    wins,
                    meta,
                    read_progress(log_path, csv_path),
                    note,
                )
                live.refresh()
                time.sleep(args.delay)

            if env.status == "won":
                wins += 1
            time.sleep(args.end_delay)


if __name__ == "__main__":
    main()
