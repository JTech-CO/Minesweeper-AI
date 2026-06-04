"""M3 DoD: model-metadata CRUD round-trip (list / get / put)."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient


def test_model_crud_roundtrip(client: TestClient) -> None:
    model_id = str(uuid.uuid4())

    # Empty to start.
    assert client.get("/api/models").json() == []

    # PUT creates the record.
    body = {
        "episode": 1000,
        "win_rate": 0.85,
        "arch_hash": "deadbeef",
        "weights_uri": "file:///weights/x.pt",
        "format": "pt",
    }
    created = client.put(f"/api/models/{model_id}", json=body)
    assert created.status_code == 200
    assert created.json()["id"] == model_id
    assert created.json()["win_rate"] == 0.85

    # GET by id returns it.
    got = client.get(f"/api/models/{model_id}")
    assert got.status_code == 200
    assert got.json()["episode"] == 1000

    # List now has exactly one.
    listing = client.get("/api/models").json()
    assert len(listing) == 1
    assert listing[0]["id"] == model_id

    # PUT again updates in place (still one record).
    body["win_rate"] = 0.91
    updated = client.put(f"/api/models/{model_id}", json=body)
    assert updated.json()["win_rate"] == 0.91
    assert len(client.get("/api/models").json()) == 1

    # Unknown id → 404.
    assert client.get(f"/api/models/{uuid.uuid4()}").status_code == 404


def test_validation_rejects_out_of_range_win_rate(client: TestClient) -> None:
    resp = client.put(f"/api/models/{uuid.uuid4()}", json={"win_rate": 1.5})
    assert resp.status_code == 422
