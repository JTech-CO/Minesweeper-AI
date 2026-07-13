"""Algorithm-independent experiment orchestration.

The runner owns lifecycle and durable counters. Algorithm backends only implement one
training update and optional live configuration, so CLI and dashboard use the same loop.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from trainer.config import ExperimentConfig
from trainer.storage.state import atomic_write_json, read_json


class ExperimentBackend(Protocol):
    def train_update(self) -> dict: ...

    def apply_live_config(self, values: dict) -> None: ...

    def close(self) -> None: ...


@dataclass
class RunState:
    status: str = "starting"
    pid: int = 0
    run_id: str = ""
    config_hash: str = ""
    update: int = 0
    episode: int = 0
    global_step: int = 0
    train_win_rate: float = 0.0
    loss: float | None = None
    updated_at: float = 0.0
    error: str | None = None


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        backend: ExperimentBackend,
        run_dir: str | Path,
        *,
        run_id: str,
        pid: int,
    ) -> None:
        self.config = config
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.control_path = self.run_dir / "control.json"
        self.status_path = self.run_dir / "status.json"
        previous = read_json(self.status_path)
        self.state = RunState(
            status="starting",
            pid=pid,
            run_id=run_id,
            config_hash=config.hash,
            update=int(previous.get("update", 0)),
            episode=int(previous.get("episode", 0)),
            global_step=int(previous.get("global_step", 0)),
        )

    def _publish(self) -> None:
        self.state.updated_at = time.time()
        atomic_write_json(self.status_path, asdict(self.state))

    def run(self) -> None:
        self.state.status = "running"
        self._publish()
        try:
            while self.state.update < self.config.total_updates:
                control = read_json(
                    self.control_path,
                    {"action": "run", "revision": 0, "live": {}},
                )
                action = control.get("action", "run")
                if action == "stop":
                    break
                if action == "pause":
                    if self.state.status != "paused":
                        self.state.status = "paused"
                        self._publish()
                    time.sleep(0.05)
                    continue
                self.backend.apply_live_config(control.get("live") or {})
                self.state.status = "running"
                metrics = self.backend.train_update()
                self.state.update += 1
                self.state.episode += int(metrics.get("episodes", 0))
                self.state.global_step += int(metrics.get("steps", 0))
                self.state.train_win_rate = float(metrics.get("train_win_rate", 0.0))
                loss = metrics.get("loss")
                self.state.loss = None if loss is None else float(loss)
                self._publish()
        except Exception as exc:
            self.state.status = "failed"
            self.state.error = f"{type(exc).__name__}: {exc}"
            self._publish()
            raise
        finally:
            self.backend.close()
        if self.state.status != "failed":
            self.state.status = "stopped"
            self._publish()
