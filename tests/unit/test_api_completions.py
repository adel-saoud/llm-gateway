"""Integration tests for ``POST /v1/chat/completions``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_gateway.config import Settings
from llm_gateway.providers.base import (
    CompletionResult,
    ProviderUnavailableError,
    StreamChunk,
    TokenUsage,
)
from tests.conftest import ScriptedProvider, gateway_client_factory, make_team

AUTH = {"Authorization": "Bearer sk-test"}


def _completion(model: str = "qwen3.5:4b", *, content: str = "hello") -> CompletionResult:
    return CompletionResult(
        model=model,
        content=content,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        latency_ms=1.0,
        finish_reason="stop",
    )


# ── Happy path ─────────────────────────────────────────────────────────────


def test_chat_completion_happy_path(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(
        provider_name="ollama", chat_responses=[_completion(content="Hi there")]
    )
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "qwen3.5:4b"
    assert body["choices"][0]["message"]["content"] == "Hi there"
    assert body["usage"]["prompt_tokens"] == 10
    assert body["usage"]["completion_tokens"] == 5
    assert body["usage"]["total_tokens"] == 15
    # Gateway response headers
    assert response.headers["X-Gateway-Model-Used"] == "qwen3.5:4b"
    assert response.headers["X-Gateway-Rag-Operation"] == "generation"
    assert "X-Gateway-Request-Id" in response.headers
    assert "X-Gateway-Cost-USD" in response.headers


# ── Authorisation / model validation ───────────────────────────────────────


def test_chat_rejects_model_not_in_allowlist(settings: Settings, tmp_path: Path) -> None:
    team = make_team(allowed_models=("qwen3.5:0.8b",))
    provider = ScriptedProvider(chat_responses=[_completion()])
    with gateway_client_factory(settings, tmp_path, team=team, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert response.status_code == 403


def test_chat_rejects_unknown_model(settings: Settings, tmp_path: Path) -> None:
    team = make_team(allowed_models=())  # unrestricted whitelist
    provider = ScriptedProvider(chat_responses=[_completion()])
    with gateway_client_factory(settings, tmp_path, team=team, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "no-such-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert response.status_code == 404


# ── Budget enforcement ─────────────────────────────────────────────────────


def test_chat_blocks_when_budget_exceeded(settings: Settings, tmp_path: Path) -> None:
    """A team with daily_usd cap of 0.001 hits the cap quickly. Pre-seed the ledger so the
    enforcer immediately returns BLOCKED."""
    from datetime import UTC, datetime

    from llm_gateway.storage.db import CostLedger, CostLedgerEntry

    team = make_team(daily_usd=0.001)
    provider = ScriptedProvider(chat_responses=[_completion()])
    db_path = tmp_path / "test.db"
    ledger = CostLedger(db_path)

    async def _seed() -> None:
        await ledger.initialize()
        await ledger.record(
            CostLedgerEntry(
                ts=datetime.now(UTC),
                team_id="t1",
                model_id="qwen3.5:4b",
                operation_type="generation",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.002,
                request_id="seed",
            )
        )

    import asyncio

    asyncio.run(_seed())

    with gateway_client_factory(settings, tmp_path, team=team, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert response.status_code == 429


# ── Provider unavailable ───────────────────────────────────────────────────


def test_chat_returns_503_when_provider_unavailable(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(chat_responses=[ProviderUnavailableError("downstream down")] * 5)
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
    assert response.status_code == 503


# ── RAG tagging propagates ─────────────────────────────────────────────────


def test_chat_classifies_evaluation_via_system_prompt(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(chat_responses=[_completion()])
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "qwen3.5:4b",
                "messages": [
                    {"role": "system", "content": "You are a judge."},
                    {"role": "user", "content": "Score this."},
                ],
            },
        )
    assert response.status_code == 200
    assert response.headers["X-Gateway-Rag-Operation"] == "evaluation"


def test_chat_classifies_via_header_override(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(chat_responses=[_completion()])
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={**AUTH, "X-Rag-Operation": "reranking"},
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "rank these"}],
            },
        )
    assert response.headers["X-Gateway-Rag-Operation"] == "reranking"


# ── Streaming ───────────────────────────────────────────────────────────────


def _parse_sse_lines(text: str) -> list[str]:
    """Return the JSON payloads of each ``data:`` SSE line, excluding ``[DONE]``."""
    payloads: list[str] = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            payloads.append(line[len("data: ") :])
    return payloads


def test_chat_streaming_emits_sse_chunks(settings: Settings, tmp_path: Path) -> None:
    chunks = [
        StreamChunk(model="qwen3.5:4b", delta="Hello", done=False),
        StreamChunk(model="qwen3.5:4b", delta=" world", done=False),
        StreamChunk(
            model="qwen3.5:4b",
            delta="",
            done=True,
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2),
        ),
    ]
    provider = ScriptedProvider(stream_chunks=chunks)
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # Stream-only headers
    assert response.headers["X-Gateway-Rag-Operation"] == "generation"
    assert "X-Gateway-Request-Id" in response.headers

    body = response.text
    payloads = _parse_sse_lines(body)
    parsed = [json.loads(p) for p in payloads]
    # First chunk: role + content
    assert parsed[0]["choices"][0]["delta"] == {"role": "assistant", "content": "Hello"}
    # Subsequent content chunks
    assert parsed[1]["choices"][0]["delta"] == {"content": " world"}
    # Final chunk: empty delta + finish_reason
    final = parsed[-1]
    assert final["choices"][0]["delta"] == {}
    assert final["choices"][0]["finish_reason"] == "stop"
    # Body ends with [DONE]
    assert body.rstrip().endswith("data: [DONE]")


# ── Thinking ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reasoning_effort", "think"),
    [(None, False), ("none", False), ("minimal", True), ("high", True)],
)
def test_reasoning_effort_controls_thinking(
    settings: Settings, tmp_path: Path, reasoning_effort: str | None, think: bool
) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    body: dict[str, object] = {
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": "Say hi."}],
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert response.status_code == 200
    assert provider.think_requests == [think]


def test_unknown_reasoning_effort_is_rejected(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": "max",
            },
        )
    assert response.status_code == 422
    assert provider.chat_calls == 0


def test_streaming_passes_thinking_through(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = client.post(
            "/v1/chat/completions",
            headers=AUTH,
            json={
                "model": "qwen3.5:4b",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "reasoning_effort": "low",
            },
        )
    assert response.status_code == 200
    assert provider.think_requests == [True]
