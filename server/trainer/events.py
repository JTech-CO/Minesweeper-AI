"""Durable trainer events consumed by the M5 API bridge."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

EVENT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_event(path: str | Path, event: Mapping[str, Any]) -> None:
    """Append one complete JSON line and make it visible to another process."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def emit_metric(
    run_dir: str | Path,
    *,
    run_id: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = metrics.get("eval")
    event = {
        "version": EVENT_VERSION,
        "type": "metric",
        "run_id": run_id,
        "time": float(metrics.get("time", time.time())),
        "payload": {
            "algorithm": config.get("algorithm"),
            "difficulty": metrics.get(
                "difficulty", config.get("current_difficulty", config.get("difficulty"))
            ),
            "current_difficulty": metrics.get(
                "current_difficulty",
                config.get("current_difficulty", config.get("difficulty")),
            ),
            "stage_update": metrics.get("stage_update"),
            "curriculum_complete": bool(
                metrics.get("curriculum_complete", False)
            ),
            "transition": metrics.get("transition"),
            "update": int(metrics.get("update", 0)),
            "episode": int(metrics.get("episodes", 0)),
            "episodes": int(metrics.get("episodes", 0)),
            "steps": int(metrics.get("steps", 0)),
            "win_rate": float(metrics.get("train_win_rate", 0.0)),
            "train_win_rate": float(metrics.get("train_win_rate", 0.0)),
            "eval_win_rate": (
                None if not evaluation else float(evaluation.get("win_rate", 0.0))
            ),
            "loss": metrics.get("loss"),
            # PPO has entropy exploration, not epsilon-greedy exploration.
            "epsilon": metrics.get("epsilon", metrics.get("eps")),
            "entropy_coef": metrics.get(
                "entropy_coef", config.get("entropy_coef")
            ),
        },
    }
    append_event(Path(run_dir) / "events.jsonl", event)
    return event


def _snapshot(source: Path, target: Path, expected_hash: str) -> None:
    if target.exists():
        if _sha256(target) != expected_hash:
            raise ValueError(f"checkpoint event snapshot hash mismatch: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source, temp)
        except OSError:
            shutil.copy2(source, temp)
        if _sha256(temp) != expected_hash:
            raise ValueError(f"checkpoint source hash mismatch: {source}")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def emit_checkpoint(
    run_dir: str | Path,
    *,
    source: str | Path,
    manifest: Any,
    role: str,
) -> dict[str, Any]:
    """Snapshot a mutable checkpoint path before publishing its durable event."""
    root = Path(run_dir).resolve()
    checkpoint = Path(source).resolve()
    data = asdict(manifest) if is_dataclass(manifest) else dict(manifest)
    checkpoint_id = str(data["checkpoint_id"])
    spool = root / ".events" / "checkpoints" / f"{checkpoint_id}.pt"
    _snapshot(checkpoint, spool, str(data["weights_sha256"]))
    manifest_path = spool.with_suffix(spool.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    event = {
        "version": EVENT_VERSION,
        "type": "checkpoint",
        "run_id": str(data["run_id"]),
        "time": time.time(),
        "role": role,
        "path": spool.relative_to(root).as_posix(),
        "manifest": data,
    }
    append_event(root / "events.jsonl", event)
    return event
