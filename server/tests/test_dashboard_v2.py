import sys
from pathlib import Path

from fastapi.testclient import TestClient

from trainer.dashboard import app, parse_args
from trainer.dashboard_app import DEFAULT_RUN_DIR


def test_dashboard_serves_local_policy_control_ui() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "R3 POLICY CONTROL" in response.text
        assert "minesweeper.online" not in response.text
        status = client.get("/api/status")
        assert status.status_code == 200
        payload = status.json()
        assert payload["lifetime_sessions"] >= 0
        assert payload["max_parallel"]["expert"] < payload["max_parallel"]["beginner"]


def test_dashboard_cli_uses_canonical_graph_run(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["trainer.dashboard"])
    assert Path(parse_args().run_dir) == DEFAULT_RUN_DIR
