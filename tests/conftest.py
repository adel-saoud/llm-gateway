"""Shared fixtures and test helpers.

Every external dependency (Ollama, Redis, SQLite filesystem) is mocked or
sandboxed so the suite stays hermetic — no network, no daemons, no shared state.

The helpers here are deliberately small and explicit. Tests assemble what they
need from these primitives rather than relying on heavyweight magic fixtures:

* :class:`FakeProvider` — minimal stand-in used by middleware tests that don't
  care about LLM behaviour.
* :class:`ScriptedProvider` — controllable stand-in: tests pre-supply chat,
  embed, and stream responses (or exceptions).
* :func:`make_team` — keyword-only :class:`TeamConfig` factory.
* :func:`gateway_client_factory` — context manager that builds a fully wired
  ``TestClient`` with hermetic Redis/SQLite/provider stand-ins.

Ollama HTTP traffic is intercepted by ``respx`` in the provider-level tests —
see the per-test ``@respx.mock`` decorator.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from pydantic import SecretStr

from llm_gateway.auth.models import BudgetConfig, RateLimitConfig, TeamConfig
from llm_gateway.auth.resolver import TeamResolver
from llm_gateway.config import Settings
from llm_gateway.main import create_app
from llm_gateway.providers.base import (
    ChatMessage,
    CompletionResult,
    EmbeddingResult,
    StreamChunk,
    TokenUsage,
)
from llm_gateway.ratelimit.token_bucket import TokenBucket
from llm_gateway.storage.db import CostLedger

OLLAMA_BASE_URL = "http://ollama-test:11434"


@pytest.fixture
def ollama_base_url() -> str:
    return OLLAMA_BASE_URL


# ── Shared stand-ins ────────────────────────────────────────────────────────


class FakeProvider:
    """No-network ``Provider`` for middleware / handler tests.

    Only ``health()`` does anything useful — the LLM methods raise so a test
    accidentally invoking them fails loud instead of silently passing.
    """

    name = "fake"

    def __init__(self, *, healthy: bool = True) -> None:
        self._healthy = healthy

    async def health(self) -> bool:
        return self._healthy

    async def chat(self, *_: Any, **__: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def stream_chat(self, *_: Any, **__: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def embed(self, *_: Any, **__: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


class ScriptedProvider:
    """A controllable provider for testing handlers and router fallback paths.

    Each method either consumes from a pre-supplied script (list of results or
    exceptions, exhausted in order) or returns a canned default. Streaming
    accepts an explicit list of :class:`StreamChunk` instances.
    """

    def __init__(
        self,
        *,
        provider_name: str = "scripted",
        chat_responses: Sequence[object] = (),
        embed_responses: Sequence[object] = (),
        stream_chunks: Sequence[StreamChunk] | None = None,
        healthy: bool = True,
    ) -> None:
        self._name = provider_name
        self._chat_responses = list(chat_responses)
        self._embed_responses = list(embed_responses)
        self._stream_chunks = list(stream_chunks) if stream_chunks is not None else None
        self._healthy = healthy
        self.chat_calls = 0
        self.embed_calls = 0
        self.stream_calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def health(self) -> bool:
        return self._healthy

    async def chat(
        self,
        model: str,
        _messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        self.chat_calls += 1
        if self._chat_responses:
            item = self._chat_responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, CompletionResult)
            return item
        return CompletionResult(
            model=model,
            content="scripted-response",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            latency_ms=1.0,
            finish_reason="stop",
        )

    async def stream_chat(
        self,
        model: str,
        _messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        chunks = (
            self._stream_chunks
            if self._stream_chunks is not None
            else [
                StreamChunk(model=model, delta="Hello", done=False),
                StreamChunk(model=model, delta=" world", done=False),
                StreamChunk(
                    model=model,
                    delta="",
                    done=True,
                    usage=TokenUsage(prompt_tokens=8, completion_tokens=2),
                ),
            ]
        )
        for chunk in chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    async def embed(self, model: str, _text: str) -> EmbeddingResult:
        self.embed_calls += 1
        if self._embed_responses:
            item = self._embed_responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, EmbeddingResult)
            return item
        return EmbeddingResult(
            model=model,
            embedding=(0.1, 0.2, 0.3),
            usage=TokenUsage(prompt_tokens=4, completion_tokens=0),
            latency_ms=1.0,
        )


def make_team(
    *,
    team_id: str = "t1",
    api_key: str = "sk-test",
    allowed_models: tuple[str, ...] = ("llama3.2:3b", "qwen2.5:0.5b", "nomic-embed-text"),
    requests_per_minute: int = 600,
    burst_capacity: int = 1000,
    daily_usd: float = 100.0,
    is_admin: bool = False,
) -> TeamConfig:
    """Build a :class:`TeamConfig` with high-throughput defaults — override per test."""
    return TeamConfig(
        team_id=team_id,
        api_key=SecretStr(api_key),
        allowed_models=allowed_models,
        rate_limits=RateLimitConfig(
            requests_per_minute=requests_per_minute,
            burst_capacity=burst_capacity,
        ),
        budget=BudgetConfig(daily_usd=daily_usd),
        is_admin=is_admin,
    )


# ── Settings + TestClient factory ───────────────────────────────────────────


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Default ``Settings`` with a fixed Ollama URL — env vars otherwise cleared."""
    monkeypatch.setenv("GATEWAY_OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    return Settings(_env_file=None)  # type: ignore[call-arg]


@contextmanager
def gateway_client_factory(
    settings: Settings,
    tmp_path: Path,
    *,
    team: TeamConfig | None = None,
    teams: Sequence[TeamConfig] | None = None,
    provider: FakeProvider | ScriptedProvider | None = None,
) -> Generator[TestClient]:
    """Yield a fully-wired ``TestClient`` for one test.

    Pass ``team=`` for a single-team setup or ``teams=`` for multi-team setups
    (used by admin tests). Pass ``provider=`` to control LLM behaviour.
    """
    if teams is None:
        teams = [team or make_team()]
    app = create_app(
        settings,
        provider=provider or FakeProvider(),  # type: ignore[arg-type]
        team_resolver=TeamResolver(teams),
        token_bucket=TokenBucket(FakeAsyncRedis()),  # type: ignore[arg-type]
        cost_ledger=CostLedger(tmp_path / "test.db"),
    )
    with TestClient(app) as client:
        yield client
