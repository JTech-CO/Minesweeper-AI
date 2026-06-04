"""Pydantic schemas for training runs + the train config form (기술백서 §2.2, §2.3)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import Difficulty, RunStatus


class TrainConfig(BaseModel):
    """Client-submitted training configuration (validated server-side)."""

    difficulty: Difficulty = Difficulty.beginner
    episodes: int = Field(default=100_000, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)


class TrainingRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    difficulty: Difficulty
    config: dict[str, Any]
    status: RunStatus
    started_at: datetime | None
    ended_at: datetime | None
    summary: dict[str, Any] | None
    created_at: datetime
