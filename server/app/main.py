"""FastAPI entry point (REST; WebSocket metrics arrive in M5). 기술백서 §3.1, §4.1."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import benchmark, models, training
from app.core.config import settings

app = FastAPI(title="Minesweeper AI Server", version="0.0.0")

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
