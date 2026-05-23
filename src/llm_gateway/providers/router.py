"""Provider routing — tries each candidate in order, skipping broken circuits.

The router doesn't *own* the fallback chain — handlers in the API layer build
a :class:`RouteChain` per request based on the team's allowed models, the
requested model, and any operational policy (e.g. promote the budget tier for
large input contexts). The router executes whatever chain it's handed.

For each candidate it asks the circuit breaker for permission; on a
``ProviderUnavailableError`` or ``ProviderTimeoutError`` it records a failure
and moves to the next entry. ``ProviderError`` (4xx / malformed response) is
*not* treated as a circuit-relevant failure — it's a request-shape problem,
not a provider health problem, so it bubbles immediately.

Streaming has a stricter contract: fallback is only possible *before* the
first chunk yields. Once a chunk is delivered to the caller, the router is
committed to that provider — a mid-stream failure raises to the caller.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import structlog

from llm_gateway.circuit_breaker.state_machine import CircuitBreaker
from llm_gateway.providers.base import (
    ChatMessage,
    CompletionResult,
    EmbeddingResult,
    Provider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StreamChunk,
)

_log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    """One ``(provider, model)`` pair in the fallback chain."""

    provider: Provider
    model_id: str


RouteChain: TypeAlias = Sequence[RouteCandidate]

OnFallback: TypeAlias = Callable[[str, str, str], None]


class AllProvidersUnavailableError(ProviderError):
    """Every candidate in the chain was either broken or returned an error."""


class ProviderRouter:
    """Tries each candidate in order; skips OPEN circuits; records outcomes."""

    def __init__(self, breaker: CircuitBreaker, on_fallback: OnFallback | None = None) -> None:
        self._breaker = breaker
        self._on_fallback = on_fallback

    async def chat(
        self,
        chain: RouteChain,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        last_error: Exception | None = None
        for i, candidate in enumerate(chain):
            if not await self._breaker.is_allowed(candidate.provider.name, candidate.model_id):
                _log.debug(
                    "router_skip_open_circuit",
                    provider=candidate.provider.name,
                    model=candidate.model_id,
                )
                if self._on_fallback is not None and i + 1 < len(chain):
                    self._on_fallback(candidate.model_id, chain[i + 1].model_id, "circuit_open")
                continue
            try:
                result = await candidate.provider.chat(
                    candidate.model_id,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (ProviderUnavailableError, ProviderTimeoutError) as exc:
                last_error = exc
                await self._breaker.record_failure(candidate.provider.name, candidate.model_id)
                _log.info(
                    "router_provider_failed",
                    provider=candidate.provider.name,
                    model=candidate.model_id,
                    error=str(exc),
                )
                if self._on_fallback is not None and i + 1 < len(chain):
                    self._on_fallback(candidate.model_id, chain[i + 1].model_id, type(exc).__name__)
                continue
            await self._breaker.record_success(candidate.provider.name, candidate.model_id)
            return result
        raise AllProvidersUnavailableError(f"all providers exhausted; last error: {last_error}")

    async def stream_chat(
        self,
        chain: RouteChain,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chunks from the first non-broken candidate.

        Fallback semantics are stricter than ``chat()``: a provider that
        raises *before* yielding any chunk is treated as a regular failure
        and the next candidate is tried. A provider that raises *mid-stream*
        cannot be recovered (the client has already received partial bytes)
        — that error is recorded as a circuit failure and re-raised.
        """
        last_error: Exception | None = None
        for i, candidate in enumerate(chain):
            if not await self._breaker.is_allowed(candidate.provider.name, candidate.model_id):
                _log.debug(
                    "router_skip_open_circuit",
                    provider=candidate.provider.name,
                    model=candidate.model_id,
                )
                if self._on_fallback is not None and i + 1 < len(chain):
                    self._on_fallback(candidate.model_id, chain[i + 1].model_id, "circuit_open")
                continue

            stream = candidate.provider.stream_chat(
                candidate.model_id,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            yielded_any = False
            try:
                async for chunk in stream:
                    yielded_any = True
                    yield chunk
            except (ProviderUnavailableError, ProviderTimeoutError) as exc:
                last_error = exc
                await self._breaker.record_failure(candidate.provider.name, candidate.model_id)
                _log.info(
                    "router_stream_failed",
                    provider=candidate.provider.name,
                    model=candidate.model_id,
                    yielded=yielded_any,
                    error=str(exc),
                )
                if yielded_any:
                    raise
                if self._on_fallback is not None and i + 1 < len(chain):
                    self._on_fallback(candidate.model_id, chain[i + 1].model_id, type(exc).__name__)
                continue
            await self._breaker.record_success(candidate.provider.name, candidate.model_id)
            return
        raise AllProvidersUnavailableError(f"all providers exhausted; last error: {last_error}")

    async def embed(self, chain: RouteChain, text: str) -> EmbeddingResult:
        last_error: Exception | None = None
        for i, candidate in enumerate(chain):
            if not await self._breaker.is_allowed(candidate.provider.name, candidate.model_id):
                _log.debug(
                    "router_skip_open_circuit",
                    provider=candidate.provider.name,
                    model=candidate.model_id,
                )
                if self._on_fallback is not None and i + 1 < len(chain):
                    self._on_fallback(candidate.model_id, chain[i + 1].model_id, "circuit_open")
                continue
            try:
                result = await candidate.provider.embed(candidate.model_id, text)
            except (ProviderUnavailableError, ProviderTimeoutError) as exc:
                last_error = exc
                await self._breaker.record_failure(candidate.provider.name, candidate.model_id)
                _log.info(
                    "router_provider_failed",
                    provider=candidate.provider.name,
                    model=candidate.model_id,
                    error=str(exc),
                )
                if self._on_fallback is not None and i + 1 < len(chain):
                    self._on_fallback(candidate.model_id, chain[i + 1].model_id, type(exc).__name__)
                continue
            await self._breaker.record_success(candidate.provider.name, candidate.model_id)
            return result
        raise AllProvidersUnavailableError(f"all providers exhausted; last error: {last_error}")
