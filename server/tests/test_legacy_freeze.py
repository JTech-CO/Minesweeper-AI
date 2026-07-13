from __future__ import annotations

import json

from trainer.storage.legacy import freeze_legacy


def test_legacy_freeze_is_idempotent_and_checksummed(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    for name, content in (
        ("managed_best.pt", b"best"),
        ("managed_last.pt", b"last"),
        ("managed_state.json", b"{}"),
    ):
        (source / name).write_bytes(content)

    first = freeze_legacy(source, target)
    second = freeze_legacy(source, target)
    assert [item["sha256"] for item in first["files"]] == [
        item["sha256"] for item in second["files"]
    ]
    manifest = json.loads((target / "legacy_manifest.json").read_text())
    assert manifest["format"] == "legacy-v1"
    assert {item["name"] for item in manifest["files"]} == {
        "managed_best.pt",
        "managed_last.pt",
        "managed_state.json",
    }
