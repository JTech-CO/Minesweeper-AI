"""Content-addressed checkpoint storage and idempotent DB registration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.db.models import (
    Difficulty,
    Model,
    ModelFormat,
    RunStatus,
    TrainingRun,
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class WeightStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def persist(self, source: Path, manifest: dict[str, Any]) -> Path:
        expected = str(manifest["weights_sha256"]).lower()
        target = self.root / f"{expected}.pt"
        self.root.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha256(target) != expected:
                raise ValueError(f"stored checkpoint checksum mismatch: {target}")
        else:
            if not source.exists() or file_sha256(source) != expected:
                raise ValueError(f"checkpoint source checksum mismatch: {source}")
            temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                try:
                    os.link(source, temp)
                except OSError:
                    shutil.copy2(source, temp)
                if file_sha256(temp) != expected:
                    raise ValueError(f"checkpoint copy checksum mismatch: {temp}")
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
        _atomic_json(target.with_suffix(".pt.manifest.json"), manifest)
        return target


def _stable_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"minesweeper-ai:{value}")


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(default)


def _run_status(value: str) -> RunStatus:
    return {
        "starting": RunStatus.queued,
        "running": RunStatus.running,
        "paused": RunStatus.paused,
        "stopped": RunStatus.done,
        "failed": RunStatus.failed,
    }.get(value, RunStatus.queued)


class CheckpointRegistry:
    def __init__(
        self,
        run_dir: str | Path,
        storage: WeightStorage,
        session_factory: sessionmaker[Session],
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.storage = storage
        self.session_factory = session_factory

    def _source(self, relative_path: str) -> Path:
        source = (self.run_dir / relative_path).resolve()
        try:
            source.relative_to(self.run_dir)
        except ValueError as exc:
            raise ValueError("checkpoint event path escapes run directory") from exc
        return source

    def ingest(self, event: dict[str, Any]) -> uuid.UUID:
        if event.get("type") != "checkpoint":
            raise ValueError("not a checkpoint event")
        manifest = dict(event["manifest"])
        source = self._source(str(event["path"]))
        target = self.storage.persist(source, manifest)
        run_name = str(manifest.get("run_id") or event.get("run_id"))
        run_id = _stable_uuid(run_name)
        model_id = _stable_uuid(str(manifest["checkpoint_id"]))
        config = _read_json(self.run_dir / "config.json", {})
        status = _read_json(self.run_dir / "status.json", {"status": "queued"})
        difficulty_value = str(config.get("difficulty", "beginner"))
        difficulty = (
            Difficulty(difficulty_value)
            if difficulty_value in Difficulty._value2member_map_
            else Difficulty.custom
        )
        evaluation = manifest.get("current_eval") or manifest.get("best_eval") or {}
        win_rate = float(evaluation.get("win_rate", 0.0))
        created_at = datetime.fromisoformat(str(manifest["created_at"]))
        with self.session_factory() as session:
            run = session.get(TrainingRun, run_id)
            if run is None:
                run = TrainingRun(
                    id=run_id,
                    difficulty=difficulty,
                    config=config,
                    status=_run_status(str(status.get("status", "queued"))),
                    summary={"trainer_run_id": run_name},
                )
                session.add(run)
            else:
                run.status = _run_status(str(status.get("status", run.status.value)))
            model = session.get(Model, model_id)
            if model is None:
                model = Model(
                    id=model_id,
                    run_id=run_id,
                    episode=int(manifest.get("episode", 0)),
                    win_rate=win_rate,
                    arch_hash=str(manifest.get("arch_hash", "")),
                    weights_uri=target.as_uri(),
                    format=ModelFormat.pt,
                    created_at=created_at,
                )
                session.add(model)
            else:
                model.weights_uri = target.as_uri()
                model.win_rate = win_rate
            session.commit()
        spool_root = self.run_dir / ".events" / "checkpoints"
        if spool_root in source.parents:
            source.unlink(missing_ok=True)
            source.with_suffix(".pt.manifest.json").unlink(missing_ok=True)
        return model_id
