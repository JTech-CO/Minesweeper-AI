"""Local-only M4-R3 worker control and policy-play dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
import webbrowser
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from trainer.config import ExperimentConfig
from trainer.encoding_v3 import NUM_CHANNELS_V3, encode_v3
from trainer.env import DIFFICULTIES, MinesweeperEnv
from trainer.models import build_policy_model
from trainer.models.policy_value_axial import AxialPolicyValueNet
from trainer.storage import load_checkpoint
from trainer.storage.state import atomic_write_json, read_json
from trainer.worker import WorkerController

SERVER_ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = Path(__file__).with_name("dashboard.html")
DEFAULT_RUN_DIR = SERVER_ROOT / "storage" / "runs" / "m4-r3-graph"
RUN_DIR = Path(os.environ.get("MSAI_RUN_DIR", DEFAULT_RUN_DIR)).resolve()
CONTROLLER = WorkerController(RUN_DIR)
MAX_PARALLEL = {"beginner": 8, "intermediate": 4, "expert": 2}
PLAY = {"difficulty": "beginner", "parallel": 4, "revision": 0}


def _legacy_lifetime() -> int:
    state = read_json(SERVER_ROOT / "storage" / "checkpoints" / "managed_state.json")
    return int(state.get("cumulative_episodes", 0))


def _lifetime_sessions() -> int:
    total = _legacy_lifetime()
    runs_root = SERVER_ROOT / "storage" / "runs"
    for lifetime_path in runs_root.glob("*/lifetime.json"):
        state = read_json(lifetime_path, {"episodes": 0})
        total += int(state.get("episodes", 0))
    return total


class PlayStats:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = read_json(path, {})
        for difficulty in DIFFICULTIES:
            self.data.setdefault(
                difficulty,
                {"wins": 0, "losses": 0, "win_ms": 0.0, "total_ms": 0.0},
            )

    def record(self, difficulty: str, won: bool, elapsed_ms: float) -> None:
        stats = self.data[difficulty]
        stats["wins" if won else "losses"] += 1
        stats["total_ms"] += elapsed_ms
        if won:
            stats["win_ms"] += elapsed_ms
        atomic_write_json(self.path, self.data)

    def snapshot(self) -> dict:
        result = {}
        for difficulty, values in self.data.items():
            wins = int(values["wins"])
            losses = int(values["losses"])
            games = wins + losses
            result[difficulty] = {
                "wins": wins,
                "losses": losses,
                "games": games,
                "win_rate": wins / games if games else 0.0,
                "avg_win_ms": float(values["win_ms"]) / wins if wins else None,
            }
        return result


class PolicyStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.lock = threading.Lock()
        self.model: AxialPolicyValueNet | None = None
        self.loaded_key: tuple[str, int] | None = None
        self.metadata: dict = {"status": "waiting", "error": None}

    def _source(self) -> Path | None:
        candidates = [
            self.run_dir / "last.pt",
            SERVER_ROOT / "storage" / "checkpoints" / "m4-r2" / "graph-pretrained.pt",
            SERVER_ROOT / "storage" / "checkpoints" / "m4-r2" / "axial-pretrained.pt",
        ]
        for path in candidates:
            if path.exists() and path.with_suffix(path.suffix + ".manifest.json").exists():
                return path
        return None

    def ensure_loaded(self) -> bool:
        source = self._source()
        if source is None:
            self.metadata = {"status": "waiting", "error": None}
            return False
        key = (str(source), source.stat().st_mtime_ns)
        if self.model is not None and key == self.loaded_key:
            return True
        with self.lock:
            if self.model is not None and key == self.loaded_key:
                return True
            try:
                payload = torch.load(source, map_location="cpu", weights_only=False)
                config = payload.get("config", {})
                model = build_policy_model(config | {"input_channels": NUM_CHANNELS_V3})
                _, manifest = load_checkpoint(source, model=model, map_location="cpu")
                model.eval()
                self.model = model
                self.loaded_key = key
                self.metadata = {
                    "status": "ready",
                    "error": None,
                    "path": str(source),
                    "checkpoint_id": manifest.checkpoint_id,
                    "episode": manifest.episode,
                    "global_step": manifest.global_step,
                    "eval": manifest.current_eval,
                }
            except Exception as exc:
                self.metadata = {
                    "status": "stale" if self.model is not None else "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return self.model is not None

    @torch.no_grad()
    def act(self, env: MinesweeperEnv) -> int:
        if not self.ensure_loaded() or self.model is None:
            raise RuntimeError("policy checkpoint is not ready")
        state = torch.from_numpy(encode_v3(env)[None])
        legal_np = env.legal_action_mask().reshape(env.rows, env.cols)
        legal = torch.from_numpy(legal_np[None]).bool()
        with self.lock:
            output = self.model(state, legal)
        return int(output["policy_logits"].argmax(dim=1).item())


STATS = PlayStats(RUN_DIR / "play_stats.json")
POLICY = PolicyStore(RUN_DIR)


class GpuMonitor:
    def __init__(self) -> None:
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.pynvml = None

    def snapshot(self) -> dict:
        if self.handle is None or self.pynvml is None:
            return {"available": False}
        try:
            memory = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "available": True,
                "name": self.pynvml.nvmlDeviceGetName(self.handle),
                "util": self.pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu,
                "vram_used": memory.used,
                "vram_total": memory.total,
                "temperature": self.pynvml.nvmlDeviceGetTemperature(
                    self.handle, self.pynvml.NVML_TEMPERATURE_GPU
                ),
            }
        except Exception:
            return {"available": False}


GPU = GpuMonitor()


def _metrics(limit: int = 600) -> list[dict]:
    path = RUN_DIR / "metrics.jsonl"
    if not path.exists():
        return []
    lines: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                lines.append(line)
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def _config() -> dict | None:
    if not CONTROLLER.config_path.exists():
        return None
    return ExperimentConfig.from_json(CONTROLLER.config_path.read_text(encoding="utf-8")).to_dict()


def _status() -> dict:
    worker = CONTROLLER.snapshot()
    lifetime = read_json(RUN_DIR / "lifetime.json", {"episodes": 0, "steps": 0})
    POLICY.ensure_loaded()
    return {
        "worker": worker,
        "config": _config(),
        "gpu": GPU.snapshot(),
        "metrics": _metrics(),
        "lifetime_sessions": _lifetime_sessions(),
        "ppo_sessions": int(lifetime.get("episodes", 0)),
        "play": dict(PLAY),
        "max_parallel": MAX_PARALLEL,
        "play_stats": STATS.snapshot(),
        "policy": dict(POLICY.metadata),
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Minesweeper AI R3 Control", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/status")
async def status() -> JSONResponse:
    return JSONResponse(_status())


@app.post("/api/training")
async def training(request: Request) -> JSONResponse:
    body = await request.json()
    action = body.get("action")
    if action == "start":
        if CONTROLLER.config_path.exists():
            config = ExperimentConfig.from_json(CONTROLLER.config_path.read_text(encoding="utf-8"))
        else:
            config = ExperimentConfig(
                algorithm="ppo",
                architecture="constraint-graph-v4",
                graph_rounds=4,
                difficulty=body.get("difficulty", "beginner"),
                device=body.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                learning_rate=float(body.get("learning_rate", 1e-4)),
                width=128,
                blocks=8,
                num_envs=int(body.get("num_envs", 32)),
                rollout_steps=int(body.get("rollout_steps", 64)),
                batch_size=int(body.get("batch_size", 512)),
                total_updates=int(body.get("total_updates", 10_000)),
                eval_every=int(body.get("eval_every", 50)),
                checkpoint_every=int(body.get("checkpoint_every", 50)),
                entropy_coef=float(body.get("entropy_coef", 0.01)),
            )
        CONTROLLER.start(config)
    elif action == "resume":
        config = _config()
        if config is None:
            raise HTTPException(409, "training run has not been initialized")
        CONTROLLER.start(ExperimentConfig.from_dict(config))
    elif action in {"pause", "stop"}:
        if not CONTROLLER.control_path.exists():
            raise HTTPException(409, "training run has not been initialized")
        CONTROLLER.command(action)
    else:
        raise HTTPException(400, "action must be start, resume, pause, or stop")
    return JSONResponse(_status())


@app.post("/api/config")
async def live_config(request: Request) -> JSONResponse:
    values = await request.json()
    allowed = {name: values[name] for name in ("learning_rate", "entropy_coef") if name in values}
    if not allowed:
        raise HTTPException(400, "no supported live values")
    control = read_json(CONTROLLER.control_path, {"action": "run"})
    CONTROLLER.command(str(control.get("action", "run")), live=allowed)
    return JSONResponse({"live": allowed})


@app.post("/api/play")
async def play_config(request: Request) -> JSONResponse:
    body = await request.json()
    difficulty = str(body.get("difficulty", PLAY["difficulty"]))
    if difficulty not in DIFFICULTIES:
        raise HTTPException(400, "unknown difficulty")
    parallel = int(body.get("parallel", PLAY["parallel"]))
    PLAY["difficulty"] = difficulty
    PLAY["parallel"] = max(1, min(MAX_PARALLEL[difficulty], parallel))
    PLAY["revision"] += 1
    return JSONResponse(dict(PLAY))


def _cells(env: MinesweeperEnv) -> list[int]:
    cells = env.observe().astype(np.int16)
    if env.status in {"won", "lost"}:
        cells[(env.mine_layout == 1) & (env.revealed == 0)] = 9
        if env.exploded_at >= 0:
            cells[env.exploded_at] = 10
    return [int(value) for value in cells]


@app.websocket("/ws")
async def websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    send_lock = asyncio.Lock()

    async def emit(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def status_loop() -> None:
        while True:
            await emit({"type": "status", "data": await asyncio.to_thread(_status)})
            await asyncio.sleep(1.0)

    async def board_loop(slot: int) -> None:
        while True:
            difficulty = str(PLAY["difficulty"])
            rows, cols, mines = DIFFICULTIES[difficulty]
            if not await asyncio.to_thread(POLICY.ensure_loaded):
                await emit({"type": "model_waiting", "slot": slot})
                await asyncio.sleep(1.0)
                continue
            seed = time.time_ns() ^ (slot << 20)
            env = MinesweeperEnv(
                rows,
                cols,
                mines,
                seed=seed,
                first_click_policy="safe-area",
            )
            env.reset()
            center = (rows // 2) * cols + cols // 2
            started = time.perf_counter()
            env.step(center)
            clicks = [center]
            await emit(
                {
                    "type": "board_start",
                    "slot": slot,
                    "difficulty": difficulty,
                    "rows": rows,
                    "cols": cols,
                    "mines": mines,
                    "cells": _cells(env),
                    "clicks": clicks,
                }
            )
            while env.status == "playing":
                action = await asyncio.to_thread(POLICY.act, env)
                env.step(action)
                clicks.append(action)
                elapsed_ms = (time.perf_counter() - started) * 1000
                await emit(
                    {
                        "type": "board",
                        "slot": slot,
                        "cells": _cells(env),
                        "move": len(clicks),
                        "last": action,
                        "clicks": clicks[-24:],
                        "elapsed_ms": elapsed_ms,
                        "cps": len(clicks) / max(elapsed_ms / 1000, 1e-6),
                    }
                )
                await asyncio.sleep(0.025)
            elapsed_ms = (time.perf_counter() - started) * 1000
            won = env.status == "won"
            STATS.record(difficulty, won, elapsed_ms)
            await emit(
                {
                    "type": "board_result",
                    "slot": slot,
                    "won": won,
                    "cells": _cells(env),
                    "moves": len(clicks),
                    "clicks": clicks,
                    "elapsed_ms": elapsed_ms,
                    "cps": len(clicks) / max(elapsed_ms / 1000, 1e-6),
                    "stats": STATS.snapshot(),
                }
            )
            await asyncio.sleep(0.4)

    status_task = asyncio.create_task(status_loop())
    boards: dict[int, asyncio.Task] = {}
    revision = -1
    try:
        while True:
            if revision != PLAY["revision"]:
                revision = int(PLAY["revision"])
                for task in boards.values():
                    task.cancel()
                boards.clear()
                await emit({"type": "boards_reset"})
            for slot in range(int(PLAY["parallel"])):
                if slot not in boards or boards[slot].done():
                    boards[slot] = asyncio.create_task(board_loop(slot))
            for slot in list(boards):
                if slot >= int(PLAY["parallel"]):
                    boards[slot].cancel()
                    del boards[slot]
                    await emit({"type": "board_remove", "slot": slot})
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
    finally:
        status_task.cancel()
        for task in boards.values():
            task.cancel()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--no-reload", action="store_true")
    parser.add_argument("--tag", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["MSAI_RUN_DIR"] = str(Path(args.run_dir).resolve())
    url = f"http://{args.host}:{args.port}"
    print(f"[dashboard] local R3 dashboard: {url}", flush=True)
    if not args.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        "trainer.dashboard:app",
        host=args.host,
        port=args.port,
        log_level="warning",
        reload=not args.no_reload,
        reload_dirs=[str(Path(__file__).parent)],
    )


if __name__ == "__main__":
    main()
