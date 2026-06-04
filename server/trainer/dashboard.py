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
import sys
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
from trainer.solver import analyze, generate_no_guess_board, view_from_arrays

_EP_RE = re.compile(r"ep\s+(\d+)\s+step")
_HTML = Path(__file__).parent / "dashboard.html"

MANAGER: TrainingManager | None = None
PLAY: dict = {"parallel": 2, "difficulty": "beginner", "delay": 0.22, "no_guess": False}
# Boards grow with difficulty (fixed cell size), so fewer fit on screen → cap per difficulty.
MAX_PARALLEL = {"beginner": 18, "intermediate": 8, "expert": 4}
# Subscribers = per-connection emit callables; the online player broadcasts to all of them.
_SUBSCRIBERS: set = set()
ONLINE: dict = {"player": None, "thread": None, "chrome": None}
# Persistent Chrome profile so the online browser stays logged in across runs (gitignored).
ONLINE_PROFILE_DIR = str(Path(__file__).resolve().parent.parent / "storage" / "online_profile")
# Real-Chrome (CDP) login: Google blocks OAuth in automation browsers, so the user logs in
# in their own Chrome (launched with a debug port + a dedicated ASCII profile) and we attach.
CHROME_DEBUG_PORT = 9222
CHROME_CDP_ENDPOINT = f"http://127.0.0.1:{CHROME_DEBUG_PORT}"
CHROME_PROFILE_DIR = str(Path.home() / "msai-online-chrome")


def _find_chrome() -> str | None:
    """Locate the user's real Google Chrome (not Playwright's bundled Chromium)."""
    import shutil

    cands = [
        os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"), *_CHROME_REL),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"), *_CHROME_REL),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), *_CHROME_REL),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return shutil.which("chrome") or shutil.which("chrome.exe") or shutil.which("google-chrome")


_CHROME_REL = ("Google", "Chrome", "Application", "chrome.exe")


def _cdp_reachable() -> bool:
    """True if a Chrome with the debug port is up and accepting CDP connections."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{CHROME_CDP_ENDPOINT}/json/version", timeout=1.5):
            return True
    except Exception:  # noqa: BLE001
        return False


def _run_online_player(make_coro, server_loop: asyncio.AbstractEventLoop) -> None:
    """Drive Playwright on a dedicated subprocess-capable loop in this worker thread.

    uvicorn's reload worker runs on a Windows SelectorEventLoop, which cannot spawn the
    Playwright driver subprocess (NotImplementedError). We give the player its own
    ProactorEventLoop here and marshal every emit back to the server loop (which owns the
    WebSocket queues) via call_soon_threadsafe. Any failure surfaces as online_error.

    make_coro(emit) returns the coroutine to run (player.run or player.login).
    """

    def threadsafe_emit(frame: dict) -> None:
        server_loop.call_soon_threadsafe(broadcast, frame)

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(make_coro(threadsafe_emit))
    except Exception as e:  # noqa: BLE001 - never let a launch error vanish silently
        threadsafe_emit({"type": "online_error", "msg": f"{type(e).__name__}: {e}"})
        threadsafe_emit({"type": "online_status", "status": "stopped", "url": ""})
    finally:
        loop.close()


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
    return JSONResponse(
        {
            **MANAGER.snapshot(),
            "play": PLAY,
            "difficulties": list(DIFFICULTIES),
            "max_parallel": MAX_PARALLEL,
        }
    )


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
    from dataclasses import asdict

    from trainer.manager import LiveConfig

    body = await request.json()
    if body.get("reset"):  # back to LiveConfig defaults (authoritative source)
        d = LiveConfig()
        MANAGER.set_config(lr=d.lr, batch=d.batch, train_freq=d.train_freq, eps_end=d.eps_end)
    else:
        MANAGER.set_config(
            lr=body.get("lr"),
            batch=body.get("batch"),
            train_freq=body.get("train_freq"),
            eps_end=body.get("eps_end"),
        )
    return JSONResponse(asdict(MANAGER.cfg))


@app.post("/api/play")
async def set_play(request: Request) -> JSONResponse:
    body = await request.json()
    if "difficulty" in body and body["difficulty"] in DIFFICULTIES:
        PLAY["difficulty"] = body["difficulty"]
    if "parallel" in body:
        PLAY["parallel"] = int(body["parallel"])
    if "no_guess" in body:
        PLAY["no_guess"] = bool(body["no_guess"])
    cap = MAX_PARALLEL[PLAY["difficulty"]]
    PLAY["parallel"] = max(1, min(cap, PLAY["parallel"]))
    return JSONResponse({**PLAY, "max_parallel": cap})


@app.post("/api/online")
async def online_ctl(request: Request) -> JSONResponse:
    """Start/stop playing a real minesweeper.online game (personal demo; rate-limited)."""
    assert MANAGER is not None
    body = await request.json()
    action = body.get("action")
    from trainer.online import OnlinePlayer

    def _spawn(make_coro) -> None:
        # Playwright needs a subprocess-capable loop. On Windows the uvicorn reload worker
        # runs a SelectorEventLoop (can't spawn the driver) — there we hand the player its
        # own ProactorEventLoop in a worker thread. But we must NOT create a 2nd Proactor
        # loop alongside a Proactor main loop (it breaks Playwright's pipe → the browser
        # hands off and exits), so when the main loop already supports subprocesses
        # (Proactor on Windows, or any loop on Unix) we run the player directly on it.
        server_loop = asyncio.get_running_loop()
        if sys.platform == "win32" and not isinstance(server_loop, asyncio.ProactorEventLoop):
            thread = threading.Thread(
                target=_run_online_player, args=(make_coro, server_loop), daemon=True
            )
            ONLINE["thread"] = thread
            thread.start()
        else:
            task = asyncio.create_task(make_coro(broadcast))

            def _surface(t: asyncio.Task) -> None:  # never let a launch error vanish
                exc = None if t.cancelled() else t.exception()
                if exc is not None:
                    broadcast({"type": "online_error", "msg": f"{type(exc).__name__}: {exc}"})
                    broadcast({"type": "online_status", "status": "stopped", "url": ""})

            task.add_done_callback(_surface)
            ONLINE["thread"] = task

    if action == "open_chrome":
        # Launch the user's real Chrome with a debug port so they can log in normally
        # (Google allows it — no automation flags) and we attach to it via CDP for play.
        chrome = _find_chrome()
        if not chrome:
            return JSONResponse({"error": "Google Chrome not found"}, status_code=400)
        proc = ONLINE.get("chrome")
        if proc is None or proc.poll() is not None:
            import subprocess

            ONLINE["chrome"] = subprocess.Popen(  # noqa: S603 - fixed args, local launch
                [
                    chrome,
                    f"--remote-debugging-port={CHROME_DEBUG_PORT}",
                    f"--user-data-dir={CHROME_PROFILE_DIR}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "https://minesweeper.online/",
                ]
            )
        broadcast({"type": "online_status", "status": "chrome", "url": CHROME_CDP_ENDPOINT})
        return JSONResponse({"status": "chrome_open", "endpoint": CHROME_CDP_ENDPOINT})
    if action == "login":
        if ONLINE["player"] is not None:
            ONLINE["player"].request_stop()
        player = OnlinePlayer(MANAGER.play_act, profile_dir=ONLINE_PROFILE_DIR)
        ONLINE["player"] = player
        _spawn(lambda emit: player.login(emit))
        return JSONResponse({"status": "login"})
    if action == "start":
        url = (body.get("url") or "").strip()
        if not url.startswith("http"):
            return JSONResponse({"error": "provide a full https game URL"}, status_code=400)
        if ONLINE["player"] is not None:
            ONLINE["player"].request_stop()
        delay = max(0.0, min(2.0, float(body.get("delay", 0.3))))
        # Prefer attaching to the user's logged-in Chrome (CDP); else our own browser.
        cdp = CHROME_CDP_ENDPOINT if _cdp_reachable() else None
        player = OnlinePlayer(
            MANAGER.hybrid_act,
            delay=delay,
            profile_dir=None if cdp else ONLINE_PROFILE_DIR,
            cdp_endpoint=cdp,
        )
        ONLINE["player"] = player
        _spawn(lambda emit: player.run(url, emit))
        return JSONResponse({"status": "starting", "url": url, "delay": delay, "cdp": bool(cdp)})
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
        loop = asyncio.get_running_loop()
        while True:
            diff = PLAY["difficulty"]
            r, c, m = DIFFICULTIES[diff]
            ng = PLAY["no_guess"]
            if ng:
                # Reject-sample a no-guess board (solver clears it ~100%). CPU-bound, so run
                # it off the event loop. Falls back to a random board if none is found.
                seed = random.randrange(2**31)
                env = await loop.run_in_executor(None, generate_no_guess_board, r, c, m, seed)
                if env is None:
                    env = MinesweeperEnv(r, c, m, seed=random.randrange(2**31))
                    env.reset()
            else:
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
                    "ng": ng,
                }
            )
            start = time.monotonic()
            move = 0
            while env.status in ("ready", "playing"):
                # NG: reveal EVERY certain-safe cell each round (matches the no-guess
                # validator → clears ~100%). Otherwise the per-move hybrid (one action).
                if ng:
                    view = view_from_arrays(
                        env.rows, env.cols, env.mines, env.revealed, env.adjacent, env.flagged
                    )
                    safe, _ = analyze(view)
                    batch = [s for s in safe if not env.revealed[s]] or [
                        MANAGER.hybrid_act(encode(env), env.legal_action_mask())
                    ]
                else:
                    batch = [MANAGER.hybrid_act(encode(env), env.legal_action_mask())]
                for action in batch:
                    if env.status not in ("ready", "playing") or env.revealed[action]:
                        continue
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
            emit(
                {
                    "type": "status",
                    **MANAGER.snapshot(),
                    "play": PLAY,
                    "max_parallel": MAX_PARALLEL,
                }
            )
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
