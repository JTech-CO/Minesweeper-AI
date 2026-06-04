"""Web dashboard: watch the trained model play Minesweeper live in the browser.

    python -m trainer.dashboard --tag beginner        # opens http://127.0.0.1:8800

The server loads the latest checkpoint and plays boards move-by-move, streaming each
click (board state, chosen cell, elapsed solve time, win/loss) to the browser over a
WebSocket. A side panel polls the training metrics (win-rate curve, loss, ε) from the
background run's CSV/log. This is a dev preview that reuses the FastAPI stack; the polished
React dashboard with client-side ONNX inference is the M8 deliverable.

The page (dashboard.html) follows the dark "instrument" tokens (디자인백서 §5) and the
anti-cliché rules (no neon, no gradient text, no glassmorphism, no decorative emoji).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import re
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from trainer.encoding import encode
from trainer.env import DIFFICULTIES, MinesweeperEnv
from trainer.watch import load_agent  # checkpoint (re)loader shared with the terminal viewer

_EP_RE = re.compile(
    r"ep\s+(\d+)\s+step\s+(\d+)\s+train_wr\s+([\d.]+)\s+loss\s+([\d.nan]+)\s+eps\s+([\d.]+)\s+sps\s+([\d.]+)"
)
_HTML = Path(__file__).parent / "dashboard.html"


@dataclass
class Config:
    ckpt_path: Path
    csv_path: Path
    log_path: Path
    difficulty: str
    delay: float
    end_delay: float
    device: str


CFG: Config | None = None
app = FastAPI(title="Minesweeper AI Dashboard")


def serialize_board(env: MinesweeperEnv) -> list[int]:
    """Per-cell render code: -1 hidden, 0..8 revealed number, -2 mine, -3 exploded, -4 flag.
    On a loss, all mines are surfaced (classic reveal-on-loss)."""
    lost = env.status == "lost"
    out: list[int] = []
    for i in range(env.n):
        if env.revealed[i] and not env.mine_layout[i]:
            out.append(int(env.adjacent[i]))
        elif env.exploded_at == i:
            out.append(-3)
        elif lost and env.mine_layout[i]:
            out.append(-2)
        elif env.flagged[i]:
            out.append(-4)
        else:
            out.append(-1)
    return out


def read_metrics() -> dict:
    assert CFG is not None
    series: list[dict] = []
    latest: dict = {}
    if CFG.csv_path.exists():
        try:
            with CFG.csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                try:
                    series.append(
                        {
                            "episode": int(r["episode"]),
                            "train_wr": float(r["train_wr"]),
                            "eval_wr": float(r["eval_wr"]),
                        }
                    )
                except (ValueError, KeyError):
                    continue
            if rows:
                last = rows[-1]
                latest = {
                    "episode": int(last["episode"]),
                    "global_step": int(last["global_step"]),
                    "train_wr": float(last["train_wr"]),
                    "eval_wr": float(last["eval_wr"]),
                    "loss": last["loss"],
                    "eps": float(last["eps"]),
                    "sps": float(last["sps"]),
                }
        except OSError:
            pass
    if CFG.log_path.exists():
        try:
            with CFG.log_path.open(encoding="utf-8", errors="ignore") as f:
                tail = f.readlines()[-40:]
            for line in reversed(tail):
                m = _EP_RE.search(line)
                if m:
                    latest.update(
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
    return {"series": series, "latest": latest}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # Read fresh each request so the page can be tweaked without restarting the server.
    return _HTML.read_text(encoding="utf-8")


@app.get("/metrics")
def metrics() -> JSONResponse:
    return JSONResponse(read_metrics())


@app.websocket("/ws")
async def ws_play(ws: WebSocket) -> None:
    assert CFG is not None
    await ws.accept()
    device = torch.device(CFG.device)
    state = (None, DIFFICULTIES[CFG.difficulty], None, None)
    games = wins = 0
    try:
        while True:
            state = load_agent(CFG.ckpt_path, device, state)
            agent, board, meta, _ = state
            rows, cols, mines = board
            games += 1
            env = MinesweeperEnv(rows, cols, mines, seed=random.randrange(2**31))
            env.reset()
            await ws.send_json(
                {
                    "type": "start",
                    "rows": rows,
                    "cols": cols,
                    "mines": mines,
                    "game": games,
                    "ckpt": (
                        {"eval_wr": meta.get("eval_wr"), "episode": meta.get("episode")}
                        if meta
                        else None
                    ),
                }
            )
            start = time.monotonic()
            move = 0
            while env.status in ("ready", "playing"):
                mask = env.legal_action_mask()
                if agent is not None:
                    action = agent.act(encode(env), mask, eps=0.0)
                else:
                    action = int(np.random.choice(np.flatnonzero(mask)))
                env.step(action)
                move += 1
                await ws.send_json(
                    {
                        "type": "step",
                        "board": serialize_board(env),
                        "last": int(action),
                        "move": move,
                        "status": env.status,
                        "elapsedMs": round((time.monotonic() - start) * 1000, 1),
                    }
                )
                await asyncio.sleep(CFG.delay)
            if env.status == "won":
                wins += 1
            await ws.send_json(
                {
                    "type": "result",
                    "won": env.status == "won",
                    "moves": move,
                    "elapsedMs": round((time.monotonic() - start) * 1000, 1),
                    "games": games,
                    "wins": wins,
                }
            )
            await asyncio.sleep(CFG.end_delay)
    except WebSocketDisconnect:
        return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="beginner")
    p.add_argument("--dir", default="storage/checkpoints")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--difficulty", choices=list(DIFFICULTIES), default="beginner")
    p.add_argument("--delay", type=float, default=0.25, help="seconds between moves")
    p.add_argument("--end-delay", type=float, default=1.6, help="seconds to hold a finished board")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-open", action="store_true")
    return p.parse_args()


def main() -> None:
    global CFG
    args = parse_args()
    base = Path(args.dir)
    CFG = Config(
        ckpt_path=Path(args.ckpt) if args.ckpt else base / f"{args.tag}_best.pt",
        csv_path=base / f"{args.tag}_metrics.csv",
        log_path=base / f"{args.tag}.out",
        difficulty=args.difficulty,
        delay=args.delay,
        end_delay=args.end_delay,
        device=args.device,
    )
    url = f"http://127.0.0.1:{args.port}"
    print(f"[dashboard] {url}  ckpt={CFG.ckpt_path.name} device={CFG.device}", flush=True)
    if not args.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
