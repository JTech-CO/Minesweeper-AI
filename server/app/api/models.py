"""Model (checkpoint) metadata API: list / get / put (기술백서 §2.3).

The trainer that produces real checkpoints arrives in M4–M5; for the M3 skeleton these
endpoints provide a full CRUD round-trip over the `models` table.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import get_db
from app.db.models import Model
from app.schemas.model import ModelRead, ModelUpsert

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("", response_model=list[ModelRead])
def list_models(session: Session = Depends(get_db)) -> list[Model]:
    return list(session.scalars(select(Model).order_by(Model.created_at.desc())))


@router.get("/{model_id}", response_model=ModelRead)
def get_model(model_id: uuid.UUID, session: Session = Depends(get_db)) -> Model:
    obj = session.get(Model, model_id)
    if obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="model not found")
    return obj


@router.put("/{model_id}", response_model=ModelRead)
def upsert_model(
    model_id: uuid.UUID, body: ModelUpsert, session: Session = Depends(get_db)
) -> Model:
    obj = session.get(Model, model_id)
    if obj is None:
        obj = Model(id=model_id, **body.model_dump())
        session.add(obj)
    else:
        for field, value in body.model_dump().items():
            setattr(obj, field, value)
    session.commit()
    session.refresh(obj)
    return obj
