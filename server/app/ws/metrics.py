"""Batched M5 metric stream and checkpoint-event bridge."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.storage.weights import CheckpointRegistry

router = APIRouter()


class JsonlTail:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offset = 0

    def read_available(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        if self.path.stat().st_size < self.offset:
            self.offset = 0
        events = []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            handle.seek(self.offset)
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    handle.seek(position)
                    break
                self.offset = handle.tell()
                events.append(json.loads(line))
        return events


class MetricBroker:
    def __init__(self, history_size: int = 512) -> None:
        self.recent: deque[dict[str, Any]] = deque(maxlen=history_size)
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=8)
        self.subscribers.add(queue)
        if self.recent:
            queue.put_nowait(
                {"type": "metrics", "replay": True, "metrics": list(self.recent)}
            )
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.subscribers.discard(queue)

    async def publish(self, metrics: list[dict[str, Any]]) -> None:
        if not metrics:
            return
        self.recent.extend(metrics)
        message = {"type": "metrics", "replay": False, "metrics": metrics}
        for queue in tuple(self.subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(message)


class MetricsRuntime:
    def __init__(
        self,
        run_dir: str | Path | None,
        registry: CheckpointRegistry | None,
        *,
        poll_seconds: float = 0.05,
        throttle_seconds: float = 0.1,
    ) -> None:
        self.run_dir = None if run_dir is None else Path(run_dir).resolve()
        self.registry = registry
        self.poll_seconds = poll_seconds
        self.throttle_seconds = throttle_seconds
        self.broker = MetricBroker()
        self.last_error: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self.run_dir is not None and self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="m5-metrics-runtime")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None

    def status(self) -> dict[str, Any]:
        return {
            "run_dir": None if self.run_dir is None else str(self.run_dir),
            "connected": self.run_dir is not None,
            "error": self.last_error,
        }

    @staticmethod
    def _metric(event: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": float(event["time"]),
            "run_id": str(event["run_id"]),
            **dict(event["payload"]),
        }

    async def _run(self) -> None:
        assert self.run_dir is not None
        tail = JsonlTail(self.run_dir / "events.jsonl")
        pending_metrics: list[dict[str, Any]] = []
        pending_checkpoints: dict[str, dict[str, Any]] = {}
        last_flush = time.monotonic()
        while not self._stop.is_set():
            try:
                for event in tail.read_available():
                    event_type = event.get("type")
                    if event_type == "metric":
                        pending_metrics.append(self._metric(event))
                    elif event_type == "checkpoint":
                        checkpoint_id = str(event["manifest"]["checkpoint_id"])
                        pending_checkpoints[checkpoint_id] = event
                if self.registry is not None:
                    for checkpoint_id, event in tuple(pending_checkpoints.items()):
                        await asyncio.to_thread(self.registry.ingest, event)
                        del pending_checkpoints[checkpoint_id]
                now = time.monotonic()
                if pending_metrics and (
                    now - last_flush >= self.throttle_seconds
                    or len(pending_metrics) >= 128
                ):
                    await self.broker.publish(pending_metrics)
                    pending_metrics = []
                    last_flush = now
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass
        if pending_metrics:
            await self.broker.publish(pending_metrics)


@router.websocket("/ws/metrics")
async def metrics(websocket: WebSocket) -> None:
    await websocket.accept()
    runtime: MetricsRuntime = websocket.app.state.metrics_runtime
    queue = runtime.broker.subscribe()
    await websocket.send_json(
        {
            "type": "ready",
            "stream": "training_metrics",
            "throttle_ms": round(runtime.throttle_seconds * 1000),
            "epsilon_semantics": "null for non-epsilon algorithms such as PPO",
            **runtime.status(),
        }
    )
    try:
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        runtime.broker.unsubscribe(queue)
