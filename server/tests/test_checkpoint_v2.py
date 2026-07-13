from __future__ import annotations

import json

import numpy as np
import torch
from torch import nn

from trainer.storage import load_checkpoint, save_checkpoint


def test_atomic_checkpoint_round_trip_and_manifest(tmp_path):
    torch.manual_seed(7)
    np.random.seed(7)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = {"algorithm": "smoke", "seed": 7, "width": 8}
    path = tmp_path / "run.pt"

    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    manifest = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        config=config,
        run_id="run-test",
        git_sha="abc123",
        episode=12,
        global_step=34,
        current_eval={"suite": "validation", "win_rate": 0.5},
        best_eval={"suite": "validation", "win_rate": 0.6},
    )
    for value in model.parameters():
        value.data.zero_()

    payload, loaded_manifest = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_config=config,
    )
    assert loaded_manifest == manifest
    assert payload["episode"] == 12
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())
    on_disk = json.loads((tmp_path / "run.pt.manifest.json").read_text())
    assert on_disk["checkpoint_id"] == payload["checkpoint_id"]
    assert not list(tmp_path.glob("*.tmp"))
