"""Provider router: fallback chain + circuit-breaker integration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from llm_gateway.circuit_breaker.monitor import FailureRateMonitor
from llm_gateway.circuit_breaker.state_machine import CircuitBreaker, CircuitBreakerState
from llm_gateway.providers.base import (
    ChatMessage,
    CompletionResult,
    EmbeddingResult,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    TokenUsage,
)
from llm_gateway.providers.router import (
    AllProvidersUnavailableError,
    ProviderRouter,
    RouteCandidate,
)
from llm_gateway.storage.db import CircuitStateRepository, CostLedger


class _ScriptedProvider:
    """A controllable provider for testing fallback paths.

    Each method consumes from a pre-supplied list of responses-or-exceptions.
    """

    def __init__(self, name: str, *, chat_script: Sequence[object] = ()) -> None:
        self._name = name
        self._chat_script = list(chat_script)
        self.chat_calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def health(self) -> bool:
        return True

    async def chat(self, *_: Any, **__: Any) -> CompletionResult:
        self.chat_calls += 1
        item = self._chat_script.pop(0)
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, CompletionResult)
        return item

    def stream_chat(self, *_: Any, **__: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    async def embed(self, *_: Any, **__: Any) -> EmbeddingResult:  # pragma: no cover
        raise NotImplementedError


def _make_result(*, model: str) -> CompletionResult:
    return CompletionResult(
        model=model,
        content="ok",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        latency_ms=1.0,
        finish_reason="stop",
    )


async def _build_breaker(
    tmp_path: Path,
    *,
    threshold: float = 0.5,
    min_requests: int = 2,
    open_duration_s: float = 30.0,
) -> CircuitBreaker:
    db_path = tmp_path / "router.db"
    ledger = CostLedger(db_path)
    await ledger.initialize()
    return CircuitBreaker(
        CircuitStateRepository(db_path),
        FailureRateMonitor(window_size=10),
        failure_rate_threshold=threshold,
        min_requests=min_requests,
        open_duration_s=open_duration_s,
    )


# ── Happy path ─────────────────────────────────────────────────────────────


async def test_first_candidate_succeeds(tmp_path: Path) -> None:
    breaker = await _build_breaker(tmp_path)
    router = ProviderRouter(breaker)
    primary = _ScriptedProvider("ollama", chat_script=[_make_result(model="qwen3.5:4b")])
    chain = [RouteCandidate(primary, "qwen3.5:4b")]
    result = await router.chat(chain, [ChatMessage(role="user", content="hi")])
    assert result.content == "ok"
    assert primary.chat_calls == 1
    assert await breaker.current_state("ollama", "qwen3.5:4b") is CircuitBreakerState.CLOSED


# ── Fallback on unavailable ────────────────────────────────────────────────


async def test_fallback_to_next_on_provider_unavailable(tmp_path: Path) -> None:
    breaker = await _build_breaker(tmp_path)
    router = ProviderRouter(breaker)

    primary = _ScriptedProvider("ollama-primary", chat_script=[ProviderUnavailableError("down")])
    fallback = _ScriptedProvider(
        "ollama-fallback", chat_script=[_make_result(model="qwen3.5:0.8b")]
    )
    chain = [
        RouteCandidate(primary, "qwen3.5:4b"),
        RouteCandidate(fallback, "qwen3.5:0.8b"),
    ]
    result = await router.chat(chain, [ChatMessage(role="user", content="hi")])
    assert result.model == "qwen3.5:0.8b"
    assert primary.chat_calls == 1
    assert fallback.chat_calls == 1


async def test_fallback_on_timeout(tmp_path: Path) -> None:
    breaker = await _build_breaker(tmp_path)
    router = ProviderRouter(breaker)

    primary = _ScriptedProvider("p1", chat_script=[ProviderTimeoutError("slow")])
    fallback = _ScriptedProvider("p2", chat_script=[_make_result(model="m")])
    chain = [RouteCandidate(primary, "m"), RouteCandidate(fallback, "m")]
    result = await router.chat(chain, [ChatMessage(role="user", content="hi")])
    assert result.content == "ok"


# ── 4xx propagates ─────────────────────────────────────────────────────────


async def test_provider_error_propagates_no_fallback(tmp_path: Path) -> None:
    """``ProviderError`` (e.g. 4xx) is a request-shape problem: surface it immediately."""
    breaker = await _build_breaker(tmp_path)
    router = ProviderRouter(breaker)
    primary = _ScriptedProvider("p1", chat_script=[ProviderError("400 bad request")])
    fallback = _ScriptedProvider("p2", chat_script=[_make_result(model="m")])
    chain = [RouteCandidate(primary, "m"), RouteCandidate(fallback, "m")]
    with pytest.raises(ProviderError, match="bad request"):
        await router.chat(chain, [ChatMessage(role="user", content="hi")])
    assert fallback.chat_calls == 0


# ── All providers fail ─────────────────────────────────────────────────────


async def test_all_unavailable_raises(tmp_path: Path) -> None:
    breaker = await _build_breaker(tmp_path)
    router = ProviderRouter(breaker)
    a = _ScriptedProvider("a", chat_script=[ProviderUnavailableError("a down")])
    b = _ScriptedProvider("b", chat_script=[ProviderUnavailableError("b down")])
    chain = [RouteCandidate(a, "m"), RouteCandidate(b, "m")]
    with pytest.raises(AllProvidersUnavailableError, match="all providers"):
        await router.chat(chain, [ChatMessage(role="user", content="hi")])


# ── Circuit breaker integration ────────────────────────────────────────────


async def test_open_circuit_skipped_in_chain(tmp_path: Path) -> None:
    """When the primary's circuit is OPEN, the router falls through immediately."""
    breaker = await _build_breaker(tmp_path, threshold=0.5, min_requests=2)
    router = ProviderRouter(breaker)

    # Pre-open the primary's circuit
    for _ in range(5):
        await breaker.record_failure("p1", "m")
    assert await breaker.current_state("p1", "m") is CircuitBreakerState.OPEN

    primary = _ScriptedProvider("p1", chat_script=[_make_result(model="m")])
    fallback = _ScriptedProvider("p2", chat_script=[_make_result(model="m")])
    chain = [RouteCandidate(primary, "m"), RouteCandidate(fallback, "m")]
    await router.chat(chain, [ChatMessage(role="user", content="hi")])
    assert primary.chat_calls == 0  # Skipped: circuit OPEN
    assert fallback.chat_calls == 1


async def test_repeated_failures_open_circuit_via_router(tmp_path: Path) -> None:
    """The router records each failure into the breaker, eventually opening the circuit."""
    breaker = await _build_breaker(tmp_path, threshold=0.5, min_requests=3)
    router = ProviderRouter(breaker)

    # Both providers fail every call; we'll only invoke once and let the router fail through.
    failing = _ScriptedProvider("p1", chat_script=[ProviderUnavailableError("x")] * 3)
    fallback = _ScriptedProvider("p2", chat_script=[_make_result(model="m")] * 3)
    chain = [RouteCandidate(failing, "m"), RouteCandidate(fallback, "m")]

    for _ in range(3):
        await router.chat(chain, [ChatMessage(role="user", content="hi")])

    # After 3 failures with threshold 0.5 & min_requests 3, the circuit opens.
    assert await breaker.current_state("p1", "m") is CircuitBreakerState.OPEN
