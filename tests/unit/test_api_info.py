"""Integration tests for ``GET /v1/models`` and ``GET /v1/stats``."""

from __future__ import annotations

from pathlib import Path

from llm_gateway.config import Settings
from tests.conftest import gateway_client_factory

AUTH = {"Authorization": "Bearer sk-test"}


def test_models_lists_registry_entries(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.get("/v1/models", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert ids == {"qwen3.5:4b", "qwen3.5:0.8b", "qwen3-embedding:0.6b"}
    embed = next(m for m in body["data"] if m["id"] == "qwen3-embedding:0.6b")
    assert embed["is_embedding_model"] is True
    assert embed["tier"] == "embeddings"


def test_stats_returns_budget_and_operations(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.get("/v1/stats", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["team"] == "t1"
    assert body["today_cost_usd"] == 0.0
    assert body["budget_utilisation"] == 0.0
    assert body["daily_cap_usd"] == 100.0
    assert "by_operation" in body


def test_stats_requires_auth(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.get("/v1/stats")
    assert response.status_code == 401
