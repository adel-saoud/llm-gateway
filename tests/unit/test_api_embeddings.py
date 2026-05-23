"""Integration tests for ``POST /v1/embeddings``."""

from __future__ import annotations

from pathlib import Path

from llm_gateway.config import Settings
from llm_gateway.providers.base import EmbeddingResult, TokenUsage
from tests.conftest import ScriptedProvider, gateway_client_factory, make_team

AUTH = {"Authorization": "Bearer sk-test"}


def _embedding(model: str = "nomic-embed-text") -> EmbeddingResult:
    return EmbeddingResult(
        model=model,
        embedding=(0.1, 0.2, 0.3, 0.4),
        usage=TokenUsage(prompt_tokens=3, completion_tokens=0),
        latency_ms=1.0,
    )


def test_embeddings_happy_path(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(embed_responses=[_embedding()])
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post(
            "/v1/embeddings",
            headers=AUTH,
            json={"model": "nomic-embed-text", "input": "hello world"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == "nomic-embed-text"
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3, 0.4]
    assert body["usage"]["prompt_tokens"] == 3
    assert response.headers["X-Gateway-Rag-Operation"] == "retrieval"


def test_embeddings_rejects_non_allowed_model(settings: Settings, tmp_path: Path) -> None:
    team = make_team(allowed_models=("llama3.2:3b",))
    provider = ScriptedProvider(embed_responses=[_embedding()])
    with gateway_client_factory(settings, tmp_path, team=team, provider=provider) as client:
        response = client.post(
            "/v1/embeddings",
            headers=AUTH,
            json={"model": "nomic-embed-text", "input": "hi"},
        )
    assert response.status_code == 403


def test_embeddings_rejects_unknown_model(settings: Settings, tmp_path: Path) -> None:
    # Team with no whitelist → model check falls through to registry lookup → 404.
    team = make_team(allowed_models=())
    provider = ScriptedProvider(embed_responses=[_embedding()])
    with gateway_client_factory(settings, tmp_path, team=team, provider=provider) as client:
        response = client.post(
            "/v1/embeddings",
            headers=AUTH,
            json={"model": "no-such-model", "input": "hi"},
        )
    assert response.status_code == 404


def test_embeddings_classifies_as_retrieval_by_default(settings: Settings, tmp_path: Path) -> None:
    """Embeddings always tag as RETRIEVAL even without the header."""
    provider = ScriptedProvider(embed_responses=[_embedding()])
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post(
            "/v1/embeddings",
            headers=AUTH,
            json={"model": "nomic-embed-text", "input": "hi"},
        )
    assert response.headers["X-Gateway-Rag-Operation"] == "retrieval"
