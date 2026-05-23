"""Runtime configuration — env-driven, validated by pydantic-settings.

Resolution order: explicit kwargs → environment variables → ``.env`` file → defaults.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """Top-level settings; instantiate once at process start."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GATEWAY_",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # ── Server ──────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")  # noqa: S104 — gateway is a server, bind all
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_timeout_s: float = Field(default=120.0, ge=1.0)

    model_high_quality: str = Field(
        default="llama3.2:3b",
        validation_alias=AliasChoices("GATEWAY_MODEL_HIGH_QUALITY", "model_high_quality"),
    )
    model_budget: str = Field(
        default="qwen2.5:0.5b",
        validation_alias=AliasChoices("GATEWAY_MODEL_BUDGET", "model_budget"),
    )
    model_embeddings: str = Field(
        default="nomic-embed-text",
        validation_alias=AliasChoices("GATEWAY_MODEL_EMBEDDINGS", "model_embeddings"),
    )

    # ── Redis ───────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── SQLite ──────────────────────────────────────────────────────────────
    db_path: Path = Field(default=Path("data/gateway.db"))

    # ── Auth ────────────────────────────────────────────────────────────────
    teams_yaml: Path = Field(default=Path("config/teams.yaml"))

    # ── Circuit breaker ─────────────────────────────────────────────────────
    cb_failure_rate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    cb_min_requests: int = Field(default=10, ge=1)
    cb_open_duration_s: float = Field(default=30.0, ge=1.0)

    # ── Observability ───────────────────────────────────────────────────────
    metrics_enabled: bool = Field(default=True)
    tracing_enabled: bool = Field(default=False)
