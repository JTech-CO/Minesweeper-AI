"""Application settings loaded from environment / .env (기술백서 §7)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces=() so a `model_*` field name (model_storage_dir) is allowed.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # Local dev defaults to SQLite so the stack runs with no Docker/Postgres.
    database_url: str = "sqlite:///./msai.db"
    cors_origins: str = "http://localhost:5173"
    model_storage_dir: str = "./storage/models"
    trainer_run_dir: str | None = None
    metrics_poll_ms: int = Field(default=50, ge=10)
    metrics_throttle_ms: int = Field(default=100, ge=50)
    api_token: str | None = None

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
