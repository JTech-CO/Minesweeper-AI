import sys
from pathlib import Path

from fastapi.testclient import TestClient

from trainer.dashboard import app, parse_args
from trainer.dashboard_app import DEFAULT_RUN_DIR, PLAY, PolicyStore, _lifetime_sessions


def test_dashboard_serves_local_policy_control_ui() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "R3 POLICY CONTROL" in response.text
        assert "minesweeper.online" not in response.text
        assert "playDirty||playPending" in response.text
        assert "syncPlayControls(s.play)" in response.text
        status = client.get("/api/status")
        assert status.status_code == 200
        payload = status.json()
        assert payload["lifetime_sessions"] >= 0
        assert payload["max_parallel"]["expert"] < payload["max_parallel"]["beginner"]


def test_play_settings_apply_difficulty_specific_board_limits(monkeypatch) -> None:
    monkeypatch.setitem(PLAY, "difficulty", "beginner")
    monkeypatch.setitem(PLAY, "parallel", 4)
    monkeypatch.setitem(PLAY, "revision", PLAY["revision"])

    with TestClient(app) as client:
        intermediate = client.post(
            "/api/play", json={"difficulty": "intermediate", "parallel": 3}
        )
        assert intermediate.status_code == 200
        assert intermediate.json()["difficulty"] == "intermediate"
        assert intermediate.json()["parallel"] == 3

        expert = client.post(
            "/api/play", json={"difficulty": "expert", "parallel": 4}
        )
        assert expert.status_code == 200
        assert expert.json()["difficulty"] == "expert"
        assert expert.json()["parallel"] == 2

        beginner = client.post(
            "/api/play", json={"difficulty": "beginner", "parallel": 7}
        )
        assert beginner.status_code == 200
        assert beginner.json()["difficulty"] == "beginner"
        assert beginner.json()["parallel"] == 7


def test_dashboard_cli_uses_canonical_graph_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["trainer.dashboard"])
    assert Path(parse_args().run_dir) == DEFAULT_RUN_DIR


def test_lifetime_sums_legacy_and_all_preserved_runs(tmp_path, monkeypatch) -> None:
    checkpoints = tmp_path / "storage" / "checkpoints"
    first_run = tmp_path / "storage" / "runs" / "first"
    second_run = tmp_path / "storage" / "runs" / "second"
    checkpoints.mkdir(parents=True)
    first_run.mkdir(parents=True)
    second_run.mkdir(parents=True)
    (checkpoints / "managed_state.json").write_text('{"cumulative_episodes":100}')
    (first_run / "lifetime.json").write_text('{"episodes":11}')
    (second_run / "lifetime.json").write_text('{"episodes":7}')
    monkeypatch.setattr("trainer.dashboard_app.SERVER_ROOT", tmp_path)

    assert _lifetime_sessions() == 118

def test_policy_store_prefers_run_bootstrap_before_global_fallback(tmp_path) -> None:
    bootstrap = tmp_path / "bootstrap.pt"
    bootstrap.write_bytes(b"checkpoint")
    bootstrap.with_suffix(".pt.manifest.json").write_text("{}", encoding="utf-8")

    assert PolicyStore(tmp_path)._source() == bootstrap
