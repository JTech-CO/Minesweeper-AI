"""Training runs API (read-only skeleton; start/stop/resume arrive in M4)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import get_db
from app.db.models import TrainingRun
from app.schemas.training import TrainingRunRead

router = APIRouter(prefix="/api/training", tags=["training"])


@router.get("/runs", response_model=list[TrainingRunRead])
def list_runs(session: Session = Depends(get_db)) -> list[TrainingRun]:
    return list(session.scalars(select(TrainingRun).order_by(TrainingRun.created_at.desc())))
