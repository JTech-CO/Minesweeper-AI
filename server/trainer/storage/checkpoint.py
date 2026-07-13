"""Atomic, self-describing checkpoints with integrity validation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

FORMAT_VERSION = 2


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config)).hexdigest()


def model_arch_hash(model: nn.Module) -> str:
    signature = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameters": [(name, list(value.shape)) for name, value in model.state_dict().items()],
    }
    return hashlib.sha256(_canonical_json(signature)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class CheckpointManifest:
    format_version: int
    run_id: str
    checkpoint_id: str
    created_at: str
    git_sha: str
    config_hash: str
    arch_hash: str
    weights_sha256: str
    episode: int
    global_step: int
    current_eval: dict[str, Any] | None
    best_eval: dict[str, Any] | None


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: dict[str, Any],
    run_id: str,
    git_sha: str,
    episode: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    current_eval: dict[str, Any] | None = None,
    best_eval: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> CheckpointManifest:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    checkpoint_id = uuid.uuid4().hex
    arch = model_arch_hash(model)
    cfg_hash = config_hash(config)
    payload = {
        "format_version": FORMAT_VERSION,
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": _rng_state(),
        "episode": int(episode),
        "global_step": int(global_step),
        "config": config,
        "config_hash": cfg_hash,
        "arch_hash": arch,
        "current_eval": current_eval,
        "best_eval": best_eval,
        "extra_state": extra_state or {},
    }
    try:
        torch.save(payload, tmp)
        # Windows requires a writable descriptor for FlushFileBuffers/os.fsync.
        with tmp.open("r+b") as handle:
            os.fsync(handle.fileno())
        checksum = _file_sha256(tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)

    manifest = CheckpointManifest(
        format_version=FORMAT_VERSION,
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        created_at=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        config_hash=cfg_hash,
        arch_hash=arch,
        weights_sha256=checksum,
        episode=int(episode),
        global_step=int(global_step),
        current_eval=current_eval,
        best_eval=best_eval,
    )
    _atomic_json(target.with_suffix(target.suffix + ".manifest.json"), asdict(manifest))
    return manifest


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    expected_config: dict[str, Any] | None = None,
    restore_rng: bool = False,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], CheckpointManifest]:
    target = Path(path)
    manifest_data = json.loads(
        target.with_suffix(target.suffix + ".manifest.json").read_text(encoding="utf-8")
    )
    manifest = CheckpointManifest(**manifest_data)
    if manifest.format_version != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint version {manifest.format_version}")
    if _file_sha256(target) != manifest.weights_sha256:
        raise ValueError("checkpoint checksum mismatch")
    if model_arch_hash(model) != manifest.arch_hash:
        raise ValueError("checkpoint architecture mismatch")
    if expected_config is not None and config_hash(expected_config) != manifest.config_hash:
        raise ValueError("checkpoint config mismatch")

    payload = torch.load(target, map_location=map_location, weights_only=False)
    if payload["checkpoint_id"] != manifest.checkpoint_id:
        raise ValueError("checkpoint manifest lineage mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        _restore_rng(payload["rng"])
    return payload, manifest
