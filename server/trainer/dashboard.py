"""Integrated management dashboard (control plane).

Owns a TrainingManager (in-process training) so the browser can pause/resume training,
change hyperparameters live, watch GPU stats (read-only — no overclock), and run 1–8
parallel play sessions at a chosen difficulty. The model's lifetime trained-session count
is persisted by the manager and keeps counting across dashboard restarts.

    python -m trainer.dashboard            # resumes from beginner_best.pt, opens browser

The full React + client-ONNX dashboard is still the M8 deliverable; this reuses the
FastAPI stack and follows the dark instrument tokens (anti-cliché).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import re
import socket
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from trainer.encoding import encode
from trainer.env import DIFFICULTIES, MinesweeperEnv
from trainer.manager import TrainingManager

_EP_RE = re.compile(r"ep\s+(\d+)\s+step")
_HTML = Path(__file__).parent / "dashboard.html"

MANAGER: TrainingManager | None = None
PLAY: dict = {"parallel": 2, "difficulty": "beginner", "delay": 0.22}
# Subscribers = per-connection emit callables; the online player broadcasts to all of them.
_SUBSCRIBERS: set = set()
ONLINE: dict = {"player": None, "task": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the training manager on (re)start, reading config from env vars so the app
    can be served via an import string for uvicorn --reload (resumes from checkpoint)."""
    global MANAGER
    base = Path(os.environ.get("MSAI_DIR", "storage/checkpoints"))
    tag = os.environ.get("MSAI_BOOTSTRAP_TAG", "beginner")
    diff = os.environ.get("MSAI_DIFFICULTY", "beginner")
    device = os.environ.get("MSAI_DEVICE", "cuda")
    boot_ckpt = base / f"{tag}_best.pt"
    boot_ep = _read_latest_episode(base / f"{tag}.out", base / f"{tag}_metrics.csv")
    MANAGER = TrainingManager(
        str(base), tag="managed", difficulty=diff, device=device,
        bootstrap_ckpt=str(boot_ckpt) if boot_ckpt.exists() else None, bootstrap_episode=boot_ep,
    )
    PLAY["difficulty"] = diff
    if os.environ.get("MSAI_NO_TRAIN") != "1":
        MANAGER.start()
    print(
        f"[dashboard] manager ready: ep={MANAGER.cumulative_episodes} dev={MANAGER.device}",
        flush=True,
    )
    yield
    if MANAGER is not None:
        MANAGER.stop()


app = FastAPI(title="Minesweeper AI — Management", lifespan=lifespan)


def broadcast(frame: dict) -> None:
    for emit in list(_SUBSCRIBERS):
        try:
            emit(frame)
        except Exception:  # noqa: BLE001 - one slow client must not break others
            pass


def _read_latest_episode(log_path: Path, csv_path: Path) -> int:
    if log_path.exists():
        try:
            for line in reversed(
                log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-60:]
            ):
                m = _EP_RE.search(line)
                if m:
                    return int(m.group(1))
        except OSError:
            pass
    if csv_path.exists():
        try:
            rows = list(csv.DictReader(csv_path.open(newline="")))
            if rows:
                return int(rows[-1]["episode"])
        except (OSError, ValueError, KeyError):
            pass
    return 0


def serialize_board(env: MinesweeperEnv) -> list[int]:
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML.read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> JSONResponse:
    assert MANAGER is not None
    return JSONResponse({**MANAGER.snapshot(), "play": PLAY, "difficulties": list(DIFFICULTIES)})


@app.post("/api/control")
async def control(request: Request) -> JSONResponse:
    assert MANAGER is not None
    body = await request.json()
    action = body.get("action")
    if action == "pause":
        MANAGER.pause()
    elif action == "resume":
        MANAGER.resume()
    elif action == "start":
        MANAGER.start()
    elif action == "stop":
        MANAGER.stop()
    return JSONResponse({"status": MANAGER.status})


@app.post("/api/config")
async def config(request: Request) -> JSONResponse:
    assert MANAGER is not None
    body = await request.json()
    MANAGER.set_config(
        lr=body.get("lr"),
        batch=body.get("batch"),
        train_freq=body.get("train_freq"),
        eps_end=body.get("eps_end"),
    )
    from dataclasses import asdict

    return JSONResponse(asdict(MANAGER.cfg))


@app.post("/api/play")
async def set_play(request: Request) -> JSONResponse:
    body = await request.json()
    if "parallel" in body:
        PLAY["parallel"] = max(1, min(18, int(body["parallel"])))
    if "difficulty" in body and body["difficulty"] in DIFFICULTIES:
        PLAY["difficulty"] = body["difficulty"]
    return JSONResponse(PLAY)


@app.post("/api/online")
async def online_ctl(request: Request) -> JSONResponse:
    """Start/stop playing a real minesweeper.online game (personal demo; rate-limited)."""
    assert MANAGER is not None
    body = await request.json()
    action = body.get("action")
    if action == "start":
        url = (body.get("url") or "").strip()
        if not url.startswith("http"):
            return JSONResponse({"error": "provide a full https game URL"}, status_code=400)
        if ONLINE["player"] is not None:
            ONLINE["player"].request_stop()
        from trainer.online import OnlinePlayer

        player = OnlinePlayer(MANAGER.play_act, delay=float(body.get("delay", 0.5)))
        ONLINE["player"] = player
        ONLINE["task"] = asyncio.create_task(player.run(url, broadcast))
        return JSONResponse({"status": "starting", "url": url})
    if action == "stop" and ONLINE["player"] is not None:
        ONLINE["player"].request_stop()
        return JSONResponse({"status": "stopping"})
    return JSONResponse({"status": ONLINE["player"].status if ONLINE["player"] else "idle"})


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    assert MANAGER is not None
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=400)

    def emit(msg: dict) -> None:
        try:
            queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass  # drop frames if the client is slow

    _SUBSCRIBERS.add(emit)  # receive online-play broadcasts too

    async def board_task(slot: int) -> None:
        while True:
            diff = PLAY["difficulty"]
            r, c, m = DIFFICULTIES[diff]
            env = MinesweeperEnv(r, c, m, seed=random.randrange(2**31))
            env.reset()
            emit(
                {
                    "type": "board_start",
                    "slot": slot,
                    "rows": r,
                    "cols": c,
                    "mines": m,
                    "difficulty": diff,
                }
            )
            start = time.monotonic()
            move = 0
            while env.status in ("ready", "playing"):
                action = MANAGER.play_act(encode(env), env.legal_action_mask())
                env.step(action)
                move += 1
                emit(
                    {
                        "type": "board",
                        "slot": slot,
                        "cells": serialize_board(env),
                        "last": int(action),
                        "move": move,
                        "status": env.status,
                        "elapsedMs": round((time.monotonic() - start) * 1000, 1),
                    }
                )
                await asyncio.sleep(PLAY["delay"])
            emit(
                {
                    "type": "board_result",
                    "slot": slot,
                    "won": env.status == "won",
                    "moves": move,
                    "elapsedMs": round((time.monotonic() - start) * 1000, 1),
                }
            )
            await asyncio.sleep(1.0)

    async def status_task() -> None:
        while True:
            emit({"type": "status", **MANAGER.snapshot(), "play": PLAY})
            await asyncio.sleep(1.0)

    async def supervisor() -> None:
        boards: dict[int, asyncio.Task] = {}
        last_diff = PLAY["difficulty"]
        try:
            while True:
                if PLAY["difficulty"] != last_diff:  # difficulty changed → restart all boards
                    last_diff = PLAY["difficulty"]
                    for slot, t in list(boards.items()):
                        t.cancel()
                        emit({"type": "board_remove", "slot": slot})
                    boards.clear()
                target = PLAY["parallel"]
                for slot in range(target):
                    if slot not in boards or boards[slot].done():
                        boards[slot] = asyncio.create_task(board_task(slot))
                for slot in list(boards):
                    if slot >= target:
                        boards[slot].cancel()
                        del boards[slot]
                        emit({"type": "board_remove", "slot": slot})
                await asyncio.sleep(0.4)
        finally:
            for t in boards.values():
                t.cancel()

    async def writer() -> None:
        while True:
            await websocket.send_json(await queue.get())

    tasks = [
        asyncio.create_task(status_task()),
        asyncio.create_task(supervisor()),
        asyncio.create_task(writer()),
    ]
    try:
        while True:
            await websocket.receive_text()  # only used to detect disconnect
    except WebSocketDisconnect:
        pass
    finally:
        _SUBSCRIBERS.discard(emit)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _open_browser_when_ready(host: str, port: int, url: str) -> None:
    for _ in range(60):
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.25)
    print(f"[dashboard] ready  →  open {url}", flush=True)
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        print("[dashboard] (auto-open failed; open the URL above manually)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="storage/checkpoints")
    p.add_argument(
        "--bootstrap-tag", "--tag", default="beginner", help="run to resume weights/count from"
    )
    p.add_argument("--difficulty", choices=list(DIFFICULTIES), default="beginner")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--no-train", action="store_true", help="serve without starting training")
    p.add_argument("--no-reload", action="store_true", help="disable auto-reload on code changes")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Config is passed to the worker (which may be a reload subprocess) via the environment.
    os.environ["MSAI_DIR"] = args.dir
    os.environ["MSAI_BOOTSTRAP_TAG"] = args.bootstrap_tag
    os.environ["MSAI_DIFFICULTY"] = args.difficulty
    os.environ["MSAI_DEVICE"] = args.device
    if args.no_train:
        os.environ["MSAI_NO_TRAIN"] = "1"
    host = "127.0.0.1"
    url = f"http://{host}:{args.port}"
    reload_on = "off" if args.no_reload else "on"
    print(f"[dashboard] starting on {url}  (reload={reload_on})", flush=True)
    print("[dashboard] keep this window open; press Ctrl+C to stop.", flush=True)
    if not args.no_open:
        threading.Thread(
            target=_open_browser_when_ready, args=(host, args.port, url), daemon=True
        ).start()
    if args.no_reload:
        uvicorn.run(app, host=host, port=args.port, log_level="warning")
    else:
        uvicorn.run(
            "trainer.dashboard:app", host=host, port=args.port, log_level="warning",
            reload=True, reload_dirs=[str(Path(__file__).parent)],
        )


if __name__ == "__main__":
    main()
