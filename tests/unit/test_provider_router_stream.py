"""Stream-chat tests for :class:`ProviderRouter`.

Streaming has a stricter contract than ``chat``: fallback only works *before*
the first chunk yields. These tests cover the pre-first-chunk and
mid-stream failure paths.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from llm_gateway.circuit_breaker.monitor import FailureRateMonitor
from llm_gateway.circuit_breaker.state_machine import CircuitBreaker
from llm_gateway.providers.base import (
    ChatMessage,
    ProviderUnavailableError,
    StreamChunk,
    TokenUsage,
)
from llm_gateway.providers.router import (
    AllProvidersUnavailableError,
    ProviderRouter,
    RouteCandidate,
)
from llm_gateway.storage.db import CircuitStateRepository, CostLedger


class _StreamingProvider:
    """A provider whose ``stream_chat`` either yields a sequence or raises."""

    def __init__(
        self,
        name: str,
        *,
        chunks: Sequence[object] = (),
    ) -> None:
        self._name = name
        self._chunks = list(chunks)
        self.stream_calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def health(self) -> bool:
        return True

    async def chat(self, *_: Any, **__: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def stream_chat(
        self,
        _model: str,
        _messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls += 1
        for item in self._chunks:
            if isinstance(item, BaseException):
                raise item
            assert isinstance(item, StreamChunk)
            yield item

    async def embed(self, *_: Any, **__: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


async def _build_breaker(tmp_path: Path) -> CircuitBreaker:
    db_path = tmp_path / "router-stream.db"
    ledger = CostLedger(db_path)
    await ledger.initialize()
    return CircuitBreaker(
        CircuitStateRepository(db_path),
        FailureRateMonitor(window_size=10),
        failure_rate_threshold=0.5,
        min_requests=2,
        open_duration_s=30.0,
    )


def _chunks(model: str = "m") -> list[StreamChunk]:
    return [
        StreamChunk(model=model, delta="Hello", done=False),
        StreamChunk(model=model, delta=" world", done=False),
        StreamChunk(
            model=model,
            delta="",
            done=True,
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2),
        ),
    ]


async def test_streaming_happy_path(tmp_path: Path) -> None:
    breaker = await _build_breaker(tmp_path)
    router = ProviderRouter(breaker)
    provider = _StreamingProvider("p1", chunks=_chunks())
    chain = [RouteCandidate(provider, "m")]  # type: ignore[arg-type]
    collected: list[StreamChunk] = [
        c async for c in router.stream_chat(chain, [ChatMessage(role="user", content="hi")])
    ]
    assert [c.delta for c in collected] == ["Hello", " world", ""]
    assert provider.stream_calls == 1


async def test_streaming_falls_back_when_first_provider_fails_before_yielding(
    tmp_path: Path,
) -> None:
    """Provider raises before any chunk → next candidate is tried."""
    breaker = await _build_breaker(tmp_path)
    router = ProviderRouter(breaker)
    failing = _StreamingProvider("p1", chunks=[ProviderUnavailableError("boom")])
    succeeding = _StreamingProvider("p2", chunks=_chunks())
    chain = [
        RouteCandidate(failing, "m"),  # type: ignore[arg-type]
        RouteCandidate(succeeding, "m"),  # type: ignore[arg-type]
    ]
    collected: list[StreamChunk] = [
        c async for c in router.stream_chat(chain, [ChatMessage(role="user", content="hi")])
    ]
    assert failing.stream_calls == 1
    assert succeeding.stream_calls == 1
    assert [c.delta for c in collected] == ["Hello", " world", ""]


async def test_streaming_mid_stream_failure_does_not_fall_back(tmp_path: Path) -> None:
    """A provider that yields then raises commits the router: no fallback."""
    breaker = await _build_breaker(tmp_path)
    router = ProviderRouter(breaker)
    mid_failing = _StreamingProvider(
        "p1",
        chunks=[
            StreamChunk(model="m", delta="Hello", done=False),
            ProviderUnavailableError("mid-stream boom"),
        ],
    )
    fallback = _StreamingProvider("p2", chunks=_chunks())
    chain = [
        RouteCandidate(mid_failing, "m"),  # type: ignore[arg-type]
        RouteCandidate(fallback, "m"),  # type: ignore[arg-type]
    ]
    collected: list[StreamChunk] = []

    async def _drain() -> None:
        async for chunk in router.stream_chat(chain, [ChatMessage(role="user", content="hi")]):
            collected.append(chunk)

    with pytest.raises(ProviderUnavailableError, match="mid-stream"):
        await _drain()
    assert [c.delta for c in collected] == ["Hello"]
    # Fallback was not tried: we were already committed.
    assert fallback.stream_calls == 0


async def test_streaming_all_unavailable_raises(tmp_path: Path) -> None:
    breaker = await _build_breaker(tmp_path)
    router = ProviderRouter(breaker)
    p1 = _StreamingProvider("p1", chunks=[ProviderUnavailableError("a")])
    p2 = _StreamingProvider("p2", chunks=[ProviderUnavailableError("b")])
    chain = [RouteCandidate(p1, "m"), RouteCandidate(p2, "m")]  # type: ignore[arg-type]
    with pytest.raises(AllProvidersUnavailableError):
        async for _ in router.stream_chat(chain, [ChatMessage(role="user", content="hi")]):
            pass
