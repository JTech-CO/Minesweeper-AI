"""Detached training worker and reload-safe controller."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from trainer.config import ExperimentConfig
from trainer.experiment import ExperimentRunner
from trainer.storage.state import atomic_write_json, read_json


class CounterBackend:
    """Minimal deterministic backend used only by lifecycle tests."""

    def __init__(self) -> None:
        self.value = 0

    def train_update(self) -> dict:
        self.value += 1
        time.sleep(0.01)
        return {
            "episodes": 1,
            "steps": 2,
            "train_win_rate": (self.value % 10) / 10,
            "loss": 1.0 / self.value,
        }

    def apply_live_config(self, _values: dict) -> None:
        return

    def close(self) -> None:
        return


def _build_backend(config: ExperimentConfig, run_dir: Path):
    if config.algorithm == "counter":
        return CounterBackend()
    from trainer.algorithms.ppo import PPOBackend

    return PPOBackend(config, run_dir)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class WorkerController:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.config_path = self.run_dir / "config.json"
        self.control_path = self.run_dir / "control.json"
        self.status_path = self.run_dir / "status.json"
        self.pid_path = self.run_dir / "worker.pid"

    def initialize(self, config: ExperimentConfig) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            existing = ExperimentConfig.from_json(self.config_path.read_text(encoding="utf-8"))
            if existing.hash != config.hash:
                raise ValueError("run directory already contains a different config")
        else:
            atomic_write_json(self.config_path, config.to_dict())
        if not self.control_path.exists():
            atomic_write_json(self.control_path, {"action": "run", "revision": 0, "live": {}})

    def pid(self) -> int:
        try:
            return int(self.pid_path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, ValueError):
            return 0

    def is_running(self) -> bool:
        return _pid_exists(self.pid())

    def start(self, config: ExperimentConfig) -> int:
        self.initialize(config)
        if self.is_running():
            self.command("run")
            return self.pid()
        self.command("run")
        stdout = (self.run_dir / "worker.out").open("ab", buffering=0)
        stderr = (self.run_dir / "worker.err").open("ab", buffering=0)
        kwargs: dict = {
            "cwd": str(Path(__file__).resolve().parents[1]),
            "stdin": subprocess.DEVNULL,
            "stdout": stdout,
            "stderr": stderr,
            "close_fds": True,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "trainer.worker", "--run-dir", str(self.run_dir)],
                **kwargs,
            )
        finally:
            stdout.close()
            stderr.close()
        self.pid_path.write_text(str(process.pid), encoding="ascii")
        return process.pid

    def command(self, action: str, *, live: dict | None = None) -> None:
        if action not in {"run", "pause", "stop"}:
            raise ValueError(f"unknown worker action {action!r}")
        current = read_json(self.control_path, {"revision": 0, "live": {}})
        atomic_write_json(
            self.control_path,
            {
                "action": action,
                "revision": int(current.get("revision", 0)) + 1,
                "live": dict(current.get("live") or {}) | dict(live or {}),
            },
        )

    def snapshot(self) -> dict:
        state = read_json(self.status_path, {"status": "idle"})
        state["worker_alive"] = self.is_running()
        return state

    def wait_for(self, status: str, timeout: float = 10.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.snapshot()
            if state.get("status") == status:
                return state
            if state.get("status") == "failed":
                raise RuntimeError(f"worker failed: {state.get('error')}")
            time.sleep(0.05)
        raise TimeoutError(f"worker did not reach {status!r}: {self.snapshot()}")


def run_worker(run_dir: Path) -> None:
    config = ExperimentConfig.from_json((run_dir / "config.json").read_text(encoding="utf-8"))
    pid_path = run_dir / "worker.pid"
    pid_path.write_text(str(os.getpid()), encoding="ascii")
    status = read_json(run_dir / "status.json")
    run_id = str(status.get("run_id") or uuid.uuid4())
    backend = _build_backend(config, run_dir)
    runner = ExperimentRunner(config, backend, run_dir, run_id=run_id, pid=os.getpid())
    try:
        runner.run()
    finally:
        pid_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_worker(Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
