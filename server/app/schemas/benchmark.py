"""Pydantic schemas for benchmark results (기술백서 §2.3, §2.4)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import BenchmarkMode, Difficulty


class BenchmarkResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    model_id: uuid.UUID | None
    mode: BenchmarkMode
    difficulty: Difficulty
    games: int
    win_rate: float
    avg_clear_ms: float
    avg_3bvps: float
    guess_rate: float
    seed: int | None
    created_at: datetime
