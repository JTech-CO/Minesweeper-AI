"""FastAPI entry point (REST; WebSocket metrics arrive in M5). 기술백서 §3.1, §4.1."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import benchmark, models, training
from app.core.config import settings
from app.db.session import SessionLocal
from app.storage.weights import CheckpointRegistry, WeightStorage
from app.ws import metrics
from app.ws.metrics import MetricsRuntime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    injected = hasattr(app.state, "metrics_runtime")
    if injected:
        runtime = app.state.metrics_runtime
    else:
        run_dir = None if settings.trainer_run_dir is None else Path(settings.trainer_run_dir)
        registry = (
            None
            if run_dir is None
            else CheckpointRegistry(
                run_dir,
                WeightStorage(settings.model_storage_dir),
                SessionLocal,
            )
        )
        runtime = MetricsRuntime(
            run_dir,
            registry,
            poll_seconds=settings.metrics_poll_ms / 1000,
            throttle_seconds=settings.metrics_throttle_ms / 1000,
        )
        app.state.metrics_runtime = runtime
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()
        if not injected:
            del app.state.metrics_runtime


app = FastAPI(title="Minesweeper AI Server", version="0.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "msai-server", "version": "0.0.0"}


app.include_router(models.router)
app.include_router(training.router)
app.include_router(benchmark.router)
app.include_router(metrics.router)
