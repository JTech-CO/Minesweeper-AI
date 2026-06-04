"""ORM entities (기술백서 §2.3). Portable column types so the same schema works on
SQLite (dev) and Postgres (CI/prod): `Uuid` (native on PG, CHAR(32) on SQLite),
`JSON`, and string-backed enums via `Enum(native_enum=False)`.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Uuid,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Difficulty(enum.StrEnum):
    beginner = "beginner"
    intermediate = "intermediate"
    expert = "expert"
    custom = "custom"


class RunStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    paused = "paused"
    done = "done"
    failed = "failed"


class ModelFormat(enum.StrEnum):
    pt = "pt"
    onnx = "onnx"


class BenchmarkMode(enum.StrEnum):
    standard = "standard"
    ng = "ng"


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _difficulty_col() -> Any:
    return SAEnum(Difficulty, native_enum=False, length=16)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    difficulty: Mapped[Difficulty] = mapped_column(_difficulty_col())
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[RunStatus] = mapped_column(
        SAEnum(RunStatus, native_enum=False, length=16), default=RunStatus.queued
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    models: Mapped[list[Model]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Model(Base):
    """A training checkpoint's metadata. Weights live on disk/object storage (weights_uri)."""

    __tablename__ = "models"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=True
    )
    episode: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    arch_hash: Mapped[str] = mapped_column(String(64), default="")
    weights_uri: Mapped[str] = mapped_column(String(512), default="")
    format: Mapped[ModelFormat] = mapped_column(
        SAEnum(ModelFormat, native_enum=False, length=8), default=ModelFormat.pt
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[TrainingRun | None] = relationship(back_populates="models")


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    mode: Mapped[BenchmarkMode] = mapped_column(SAEnum(BenchmarkMode, native_enum=False, length=8))
    difficulty: Mapped[Difficulty] = mapped_column(_difficulty_col())
    games: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_clear_ms: Mapped[float] = mapped_column(Float, default=0.0)
    avg_3bvps: Mapped[float] = mapped_column(Float, default=0.0)
    guess_rate: Mapped[float] = mapped_column(Float, default=0.0)
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
