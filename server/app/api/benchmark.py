"""Benchmark results API (read-only skeleton; runs/leaderboard arrive in M10)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import get_db
from app.db.models import BenchmarkResult
from app.schemas.benchmark import BenchmarkResultRead

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


@router.get("", response_model=list[BenchmarkResultRead])
def list_benchmarks(session: Session = Depends(get_db)) -> list[BenchmarkResult]:
    return list(
        session.scalars(select(BenchmarkResult).order_by(BenchmarkResult.created_at.desc()))
    )
