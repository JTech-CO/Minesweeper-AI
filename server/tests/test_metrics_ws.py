"""M5 DoD: WebSocket metrics plus checkpoint file and DB persistence."""

from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import Difficulty, Model, RunStatus, TrainingRun
from app.main import app
from app.storage.weights import CheckpointRegistry, WeightStorage
from app.ws.metrics import MetricsRuntime
from trainer.events import emit_checkpoint, emit_metric


def test_metrics_websocket_and_checkpoint_registry(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"algorithm": "ppo", "difficulty": "beginner", "curriculum": True}),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "stopped"}),
        encoding="utf-8",
    )
    (run_dir / "validation.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "fedcba9876543210fedcba9876543210",
                "win_rate": 0.937,
            }
        ),
        encoding="utf-8",
    )
    engine = create_engine(
        f"sqlite:///{tmp_path / 'm5.db'}",
        connect_args={"check_same_thread": False},
    )
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(engine)
    storage = WeightStorage(tmp_path / "models")
    runtime = MetricsRuntime(
        run_dir,
        CheckpointRegistry(run_dir, storage, TestSession),
        poll_seconds=0.01,
        throttle_seconds=0.05,
    )
    app.state.metrics_runtime = runtime
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/metrics") as websocket:
                ready = websocket.receive_json()
                assert ready["type"] == "ready"
                assert ready["throttle_ms"] == 50

                emit_metric(
                    run_dir,
                    run_id="ppo-integration",
                    config={
                        "algorithm": "ppo",
                        "difficulty": "beginner",
                        "entropy_coef": 0.003,
                    },
                    metrics={
                        "time": 10.0,
                        "update": 4,
                        "episodes": 40,
                        "steps": 80,
                        "train_win_rate": 0.875,
                        "loss": 0.125,
                        "eval": None,
                    },
                )
                source = run_dir / "best.pt"
                source.write_bytes(b"small-test-checkpoint")
                checksum = hashlib.sha256(source.read_bytes()).hexdigest()
                emit_checkpoint(
                    run_dir,
                    source=source,
                    role="validated",
                    manifest={
                        "format_version": 2,
                        "run_id": "ppo-integration",
                        "checkpoint_id": "fedcba9876543210fedcba9876543210",
                        "created_at": "2026-07-15T00:00:00+00:00",
                        "git_sha": "test",
                        "config_hash": "config",
                        "arch_hash": "arch",
                        "weights_sha256": checksum,
                        "episode": 40,
                        "global_step": 80,
                        "current_eval": {"win_rate": 0.875},
                        "best_eval": {"win_rate": 0.875},
                    },
                )

                batch = websocket.receive_json()
                assert batch["type"] == "metrics"
                assert batch["replay"] is False
                metric = batch["metrics"][0]
                assert metric["episode"] == 40
                assert metric["win_rate"] == 0.875
                assert metric["loss"] == 0.125
                assert metric["epsilon"] is None

        with TestSession() as session:
            model = session.scalar(select(Model))
            run = session.scalar(select(TrainingRun))
            assert model is not None
            assert run is not None
            assert run.status is RunStatus.done
            assert run.difficulty is Difficulty.custom
            assert model.run_id == run.id
            assert model.episode == 40
            assert model.win_rate == 0.937
            stored = storage.root / f"{checksum}.pt"
            assert stored.read_bytes() == b"small-test-checkpoint"
            assert model.weights_uri == stored.as_uri()
    finally:
        del app.state.metrics_runtime
        engine.dispose()


def test_checkpoint_event_rejects_path_escape(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    engine = create_engine(f"sqlite:///{tmp_path / 'escape.db'}")
    TestSession = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    registry = CheckpointRegistry(run_dir, WeightStorage(tmp_path / "models"), TestSession)
    event = {
        "type": "checkpoint",
        "path": "../outside.pt",
        "manifest": {"checkpoint_id": "bad"},
    }

    try:
        registry.ingest(event)
    except ValueError as exc:
        assert "escapes run directory" in str(exc)
    else:
        raise AssertionError("path traversal event was accepted")
    finally:
        engine.dispose()
