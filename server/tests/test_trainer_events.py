"""Durable trainer event contract tests."""

from __future__ import annotations

import hashlib
import json
import os

from trainer.events import emit_checkpoint, emit_metric


def test_checkpoint_event_snapshots_mutable_source(tmp_path) -> None:
    source = tmp_path / "last.pt"
    source.write_bytes(b"checkpoint-v1")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "format_version": 2,
        "run_id": "ppo-test",
        "checkpoint_id": "0123456789abcdef0123456789abcdef",
        "created_at": "2026-07-15T00:00:00+00:00",
        "git_sha": "test",
        "config_hash": "config",
        "arch_hash": "arch",
        "weights_sha256": checksum,
        "episode": 12,
        "global_step": 34,
        "current_eval": {"win_rate": 0.75},
        "best_eval": {"win_rate": 0.75},
    }

    event = emit_checkpoint(tmp_path, source=source, manifest=manifest, role="last")
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"checkpoint-v2")
    os.replace(replacement, source)

    snapshot = tmp_path / event["path"]
    assert snapshot.read_bytes() == b"checkpoint-v1"
    assert event["manifest"]["weights_sha256"] == checksum
    persisted = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert persisted["type"] == "checkpoint"


def test_ppo_metric_event_keeps_epsilon_contract_explicit(tmp_path) -> None:
    event = emit_metric(
        tmp_path,
        run_id="ppo-test",
        config={
            "algorithm": "ppo",
            "difficulty": "beginner",
            "entropy_coef": 0.003,
        },
        metrics={
            "time": 1.0,
            "update": 2,
            "episodes": 10,
            "steps": 20,
            "train_win_rate": 0.8,
            "loss": 0.1,
            "eval": {"win_rate": 0.9},
        },
    )

    payload = event["payload"]
    assert payload["episode"] == 10
    assert payload["win_rate"] == 0.8
    assert payload["loss"] == 0.1
    assert payload["epsilon"] is None
    assert payload["entropy_coef"] == 0.003


def test_curriculum_metric_exposes_stage_state(tmp_path) -> None:
    event = emit_metric(
        tmp_path,
        run_id="curriculum-test",
        config={
            "algorithm": "ppo",
            "difficulty": "beginner",
            "current_difficulty": "intermediate",
            "entropy_coef": 0.003,
        },
        metrics={
            "update": 8,
            "episodes": 500,
            "steps": 1000,
            "difficulty": "intermediate",
            "current_difficulty": "expert",
            "stage_update": 0,
            "curriculum_complete": False,
            "transition": {"type": "promotion", "to": "expert"},
            "entropy_coef": 0.02,
        },
    )

    payload = event["payload"]
    assert payload["difficulty"] == "intermediate"
    assert payload["current_difficulty"] == "expert"
    assert payload["stage_update"] == 0
    assert payload["transition"] == {"type": "promotion", "to": "expert"}
    assert payload["entropy_coef"] == 0.02
