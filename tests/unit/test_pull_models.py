from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_gateway.config import Settings
from llm_gateway.pull_models import configured_models, main

OLLAMA = "http://ollama.test:11434"


def _settings(**overrides: str) -> Settings:
    return Settings(_env_file=None, ollama_base_url=OLLAMA, **overrides)  # type: ignore[call-arg]


def test_configured_models_follow_the_settings() -> None:
    settings = _settings(
        model_high_quality="big:4b", model_budget="small:1b", model_embeddings="embed:1b"
    )
    assert configured_models(settings) == ("big:4b", "small:1b", "embed:1b")


def test_a_model_shared_by_two_tiers_is_pulled_once() -> None:
    settings = _settings(model_high_quality="same:1b", model_budget="same:1b")
    assert configured_models(settings) == ("same:1b", "qwen3-embedding:0.6b")


@respx.mock
def test_main_pulls_every_configured_model() -> None:
    bodies: list[dict[str, object]] = []

    def record(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "success"})

    respx.post(f"{OLLAMA}/api/pull").mock(side_effect=record)
    assert main(_settings()) == 0
    assert [body["model"] for body in bodies] == [
        "qwen3.5:4b",
        "qwen3.5:0.8b",
        "qwen3-embedding:0.6b",
    ]
    assert all(body["stream"] is False for body in bodies)


@respx.mock
def test_main_stops_on_the_first_failed_pull(capsys: pytest.CaptureFixture[str]) -> None:
    respx.post(f"{OLLAMA}/api/pull").mock(
        return_value=httpx.Response(500, json={"error": "pull model manifest: file does not exist"})
    )
    assert main(_settings()) == 1
    assert "qwen3.5:4b: 500" in capsys.readouterr().err
