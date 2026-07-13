"""Freeze legacy checkpoints into an immutable, checksummed local snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_legacy(source: Path, target: Path) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("managed_best.pt", "managed_last.pt", "managed_state.json"):
        src = source / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = target / name
        if dst.exists() and _sha256(dst) != _sha256(src):
            raise ValueError(f"legacy snapshot already exists with different content: {dst}")
        if not dst.exists():
            shutil.copy2(src, dst)
        records.append(
            {
                "name": name,
                "source": str(src.resolve()),
                "bytes": dst.stat().st_size,
                "sha256": _sha256(dst),
            }
        )
    manifest = {
        "format": "legacy-v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "files": records,
        "audit": {
            "historical_fixed_eval_best": 0.345,
            "independent_seed_base": 1_700_000_000,
            "best_beginner_games": 1_000,
            "best_beginner_win_rate": 0.226,
            "last_beginner_games": 1_000,
            "last_beginner_win_rate": 0.178,
            "best_intermediate_win_rate": 0.0,
            "best_expert_win_rate": 0.0,
        },
    }
    manifest_path = target / "legacy_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="storage/checkpoints")
    parser.add_argument("--target", default="storage/checkpoints/legacy-v1")
    args = parser.parse_args()
    manifest = freeze_legacy(Path(args.source), Path(args.target))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
