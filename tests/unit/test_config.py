from __future__ import annotations

from pathlib import Path

import pytest

from llm_gateway.config import LogLevel, Settings


def test_defaults_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GATEWAY_HOST",
        "GATEWAY_PORT",
        "GATEWAY_OLLAMA_BASE_URL",
        "GATEWAY_REDIS_URL",
        "GATEWAY_MODEL_HIGH_QUALITY",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.port == 8000
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.model_high_quality == "llama3.2:3b"
    assert settings.model_budget == "qwen2.5:0.5b"
    assert settings.model_embeddings == "nomic-embed-text"
    assert settings.log_level == LogLevel.INFO
    assert settings.metrics_enabled is True


def test_env_overrides_take_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_PORT", "9001")
    monkeypatch.setenv("GATEWAY_OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("GATEWAY_MODEL_HIGH_QUALITY", "llama3.1:8b")
    monkeypatch.setenv("GATEWAY_LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.port == 9001
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.model_high_quality == "llama3.1:8b"
    assert settings.log_level == LogLevel.DEBUG


def test_invalid_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_PORT", "70000")
    with pytest.raises(ValueError, match="less than or equal"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_db_path_parsed_as_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_DB_PATH", "/tmp/test/gateway.db")  # noqa: S108
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.db_path == Path("/tmp/test/gateway.db")  # noqa: S108


def test_circuit_breaker_threshold_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_CB_FAILURE_RATE_THRESHOLD", "1.5")
    with pytest.raises(ValueError, match="less than or equal"):
        Settings(_env_file=None)  # type: ignore[call-arg]
