import time

import numpy as np
from fastapi.testclient import TestClient

import trainer.dashboard_app as dashboard


def test_dashboard_streams_policy_only_board(monkeypatch) -> None:
    monkeypatch.setattr(dashboard.POLICY, "ensure_loaded", lambda: True)
    monkeypatch.setattr(
        dashboard.POLICY,
        "act",
        lambda env: int(np.flatnonzero(env.legal_action_mask())[0]),
    )
    monkeypatch.setattr(dashboard.STATS, "record", lambda *_args: None)
    monkeypatch.setitem(dashboard.PLAY, "difficulty", "beginner")
    monkeypatch.setitem(dashboard.PLAY, "parallel", 1)
    monkeypatch.setitem(dashboard.PLAY, "revision", dashboard.PLAY["revision"] + 1)

    with TestClient(dashboard.app) as client:
        with client.websocket_connect("/ws") as websocket:
            message_types = set()
            for _ in range(12):
                message_types.add(websocket.receive_json()["type"])
                if "board_start" in message_types and "board" in message_types:
                    break
    assert "board_start" in message_types
    assert "board" in message_types



def test_dashboard_disconnect_does_not_recreate_failed_board_tasks(monkeypatch) -> None:
    loads = 0

    def ensure_loaded() -> bool:
        nonlocal loads
        loads += 1
        return True

    monkeypatch.setattr(dashboard.POLICY, "ensure_loaded", ensure_loaded)
    monkeypatch.setattr(
        dashboard.POLICY,
        "act",
        lambda env: int(np.flatnonzero(env.legal_action_mask())[0]),
    )
    monkeypatch.setattr(dashboard.STATS, "record", lambda *_args: None)
    monkeypatch.setitem(dashboard.PLAY, "difficulty", "beginner")
    monkeypatch.setitem(dashboard.PLAY, "parallel", 1)
    monkeypatch.setitem(dashboard.PLAY, "revision", dashboard.PLAY["revision"] + 1)

    with TestClient(dashboard.app) as client:
        with client.websocket_connect("/ws") as websocket:
            while websocket.receive_json()["type"] != "board_start":
                pass
        time.sleep(0.3)
        settled = loads
        time.sleep(0.5)
        assert loads == settled
