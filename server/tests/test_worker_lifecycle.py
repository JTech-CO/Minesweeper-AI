from __future__ import annotations

import time

import pytest

from trainer.config import ExperimentConfig
from trainer.worker import WorkerController


def test_config_validation_and_stable_hash():
    first = ExperimentConfig(algorithm="counter", total_updates=10)
    second = ExperimentConfig.from_dict(first.to_dict())
    assert first.hash == second.hash
    with pytest.raises(ValueError):
        ExperimentConfig(learning_rate=0)


def test_detached_worker_survives_controller_recreation(tmp_path):
    config = ExperimentConfig(
        algorithm="counter",
        total_updates=10_000,
        checkpoint_every=10,
        eval_every=10,
    )
    first = WorkerController(tmp_path)
    first.start(config)
    try:
        deadline = time.time() + 10
        while first.snapshot().get("episode", 0) < 3 and time.time() < deadline:
            time.sleep(0.05)
        before = first.snapshot()
        assert before["episode"] >= 3
        assert before["worker_alive"]

        second = WorkerController(tmp_path)
        assert second.pid() == first.pid()
        assert second.snapshot()["episode"] >= before["episode"]
        second.command("pause")
        paused = second.wait_for("paused")
        assert paused["episode"] >= before["episode"]
        second.command("run")
        second.wait_for("running")
        second.command("stop")
        stopped = second.wait_for("stopped")
        assert stopped["episode"] >= paused["episode"]
    finally:
        if first.is_running():
            first.command("stop")
            first.wait_for("stopped")
