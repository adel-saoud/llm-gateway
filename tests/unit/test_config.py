from __future__ import annotations

from pathlib import Path

import pytest

from llm_gateway.config import LogLevel, Settings, UnknownSettingError


def test_defaults_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GATEWAY_OLLAMA_BASE_URL",
        "GATEWAY_REDIS_URL",
        "GATEWAY_MODEL_HIGH_QUALITY",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.model_high_quality == "qwen3.5:4b"
    assert settings.model_budget == "qwen3.5:0.8b"
    assert settings.model_embeddings == "qwen3-embedding:0.6b"
    assert settings.log_level == LogLevel.INFO


def test_env_overrides_take_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("GATEWAY_MODEL_HIGH_QUALITY", "llama3.1:8b")
    monkeypatch.setenv("GATEWAY_LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.ollama_base_url == "http://ollama:11434"
    assert settings.model_high_quality == "llama3.1:8b"
    assert settings.log_level == LogLevel.DEBUG


def test_invalid_timeout_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_OLLAMA_TIMEOUT_S", "0.5")
    with pytest.raises(ValueError, match="greater than or equal"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_attribute_docstrings_become_field_descriptions() -> None:
    description = Settings.model_fields["cache_enabled"].description
    assert description is not None
    assert "semantic_cache" in description


def test_db_path_parsed_as_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_DB_PATH", "/tmp/test/gateway.db")  # noqa: S108
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.db_path == Path("/tmp/test/gateway.db")  # noqa: S108


def test_circuit_breaker_threshold_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_CIRCUIT_BREAKER_FAILURE_RATE_THRESHOLD", "1.5")
    with pytest.raises(ValueError, match="less than or equal"):
        Settings(_env_file=None)  # type: ignore[call-arg]


# ── Unknown GATEWAY_* names ─────────────────────────────────────────────────


def test_renamed_variable_points_to_its_new_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_CB_MIN_REQUESTS", "3")
    with pytest.raises(UnknownSettingError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]
    message = str(excinfo.value)
    assert "GATEWAY_CB_MIN_REQUESTS" in message
    assert "did you mean GATEWAY_CIRCUIT_BREAKER_MIN_REQUESTS?" in message


def test_misspelled_variable_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_REDIS_URI", "redis://elsewhere:6379/0")
    with pytest.raises(UnknownSettingError, match="did you mean GATEWAY_REDIS_URL"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_unknown_variable_without_close_match_gets_no_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_PORT", "9000")
    with pytest.raises(UnknownSettingError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert "GATEWAY_PORT (no setting with a close name)" in str(excinfo.value)


def test_unknown_variable_in_dotenv_is_rejected(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GATEWAY_CB_OPEN_DURATION_S=5\nOTEL_SERVICE_NAME=gw\n")
    with pytest.raises(UnknownSettingError, match="GATEWAY_CIRCUIT_BREAKER_OPEN_DURATION_S"):
        Settings(_env_file=env_file)  # type: ignore[call-arg]


def test_other_prefixes_are_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "gw")
    env_file = tmp_path / ".env"
    env_file.write_text("GATEWAY_LOG_LEVEL=DEBUG\nOTEL_EXPORTER_OTLP_ENDPOINT=http://x:4318\n")
    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]
    assert settings.log_level == LogLevel.DEBUG


def test_removed_variable_says_why_instead_of_a_close_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_AUTOPILOT_TIER_COMPLEX", "high_quality")
    with pytest.raises(UnknownSettingError, match="no complex tier") as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]
    assert "TIER_SIMPLE" not in str(excinfo.value)
