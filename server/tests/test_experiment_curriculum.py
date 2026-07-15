from __future__ import annotations

import json

from trainer.config import ExperimentConfig
from trainer.experiment import ExperimentRunner


class CompleteBackend:
    def __init__(self) -> None:
        self.updates = 0
        self.closed = 0

    def train_update(self) -> dict:
        self.updates += 1
        return {
            "episodes": 2,
            "steps": 4,
            "difficulty": "expert",
            "stage_update": 12,
            "curriculum_complete": True,
            "transition": {"type": "complete"},
            "train_win_rate": 0.4,
            "loss": 0.1,
            "complete": True,
        }

    def apply_live_config(self, _values: dict) -> None:
        return

    def close(self) -> None:
        self.closed += 1


def test_experiment_runner_stops_on_curriculum_completion(tmp_path) -> None:
    backend = CompleteBackend()
    runner = ExperimentRunner(
        ExperimentConfig(algorithm="counter", total_updates=10),
        backend,
        tmp_path,
        run_id="curriculum-test",
        pid=1,
    )

    runner.run()

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert backend.updates == 1
    assert backend.closed == 1
    assert status["status"] == "stopped"
    assert status["difficulty"] == "expert"
    assert status["curriculum_complete"] is True


def test_completed_curriculum_restart_does_not_train_an_extra_update(tmp_path) -> None:
    (tmp_path / "status.json").write_text(
        json.dumps(
            {
                "status": "stopped",
                "difficulty": "expert",
                "curriculum_complete": True,
            }
        ),
        encoding="utf-8",
    )
    backend = CompleteBackend()
    runner = ExperimentRunner(
        ExperimentConfig(algorithm="counter", total_updates=10),
        backend,
        tmp_path,
        run_id="curriculum-test",
        pid=1,
    )

    runner.run()

    assert backend.updates == 0
    assert backend.closed == 1