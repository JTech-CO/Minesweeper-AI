from fastapi.testclient import TestClient

from trainer.dashboard import app


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

