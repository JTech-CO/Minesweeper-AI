"""Pydantic schemas for Model (checkpoint) metadata.

Enums are reused from the ORM so the difficulty/format vocabularies are single-sourced
and stay consistent with the TS `@msai/core` `Difficulty` union (계약 일치, 기술백서 §3.2).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import ModelFormat


class ModelUpsert(BaseModel):
    """Body for PUT /api/models/{id} (create or update a checkpoint record)."""

    model_config = ConfigDict(protected_namespaces=())

    run_id: uuid.UUID | None = None
    episode: int = Field(default=0, ge=0)
    win_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    arch_hash: str = ""
    weights_uri: str = ""
    format: ModelFormat = ModelFormat.pt


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    run_id: uuid.UUID | None
    episode: int
    win_rate: float
    arch_hash: str
    weights_uri: str
    format: ModelFormat
    created_at: datetime
