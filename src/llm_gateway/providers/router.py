"""Provider routing: tries each candidate in order, skipping broken circuits.

The router doesn't own the fallback chain: the routing policy plans it and
the handler filters it by the team's allowed models. The router executes
whatever chain it is handed.

For each candidate it asks the circuit breaker for permission; on a
``ProviderUnavailableError`` or ``ProviderTimeoutError`` it records a failure
and moves to the next entry. ``ProviderError`` (4xx / malformed response) is
*not* treated as a circuit-relevant failure: it's a request-shape problem,
not a provider health problem, so it bubbles immediately.

Streaming has a stricter contract: fallback is only possible *before* the
first chunk yields. Once a chunk is delivered to the caller, the router is
committed to that provider: a mid-stream failure raises to the caller.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass

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


type RouteChain = Sequence[RouteCandidate]
"""Candidates in preference order; the first healthy one serves the request."""

type OnFallback = Callable[[str, str, str], None]
"""Called with ``(from_model, to_model, reason)`` each time the router moves down the chain."""


class AllProvidersUnavailableError(ProviderError):
    """Every candidate in the chain was either broken or returned an error."""


class ProviderRouter:
    """Tries each candidate in order; skips OPEN circuits; records outcomes."""

    def __init__(
        self, circuit_breaker: CircuitBreaker, on_fallback: OnFallback | None = None
    ) -> None:
        self._circuit_breaker = circuit_breaker
        self._on_fallback = on_fallback

    async def chat(
        self,
        chain: RouteChain,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> CompletionResult:
        """Completion from the first candidate that is allowed and answers."""
        last_error: Exception | None = None
        for index, candidate in enumerate(chain):
            if not await self._is_allowed(chain, index):
                continue
            try:
                result = await candidate.provider.chat(
                    candidate.model_id,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    think=think,
                )
            except (ProviderUnavailableError, ProviderTimeoutError) as exc:
                last_error = exc
                await self._record_failure(chain, index, exc)
                continue
            await self._circuit_breaker.record_success(candidate.provider.name, candidate.model_id)
            return result
        raise AllProvidersUnavailableError(f"all providers exhausted; last error: {last_error}")

    async def stream_chat(
        self,
        chain: RouteChain,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chunks from the first non-broken candidate.

        Fallback semantics are stricter than ``chat()``: a provider that
        raises *before* yielding any chunk is treated as a regular failure
        and the next candidate is tried. A provider that raises *mid-stream*
        cannot be recovered (the client has already received partial bytes),
        so that error is recorded as a circuit failure and re-raised.
        """
        last_error: Exception | None = None
        for index, candidate in enumerate(chain):
            if not await self._is_allowed(chain, index):
                continue
            stream = candidate.provider.stream_chat(
                candidate.model_id,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                think=think,
            )
            yielded_any = False
            try:
                async for chunk in stream:
                    yielded_any = True
                    yield chunk
            except (ProviderUnavailableError, ProviderTimeoutError) as exc:
                last_error = exc
                if yielded_any:
                    await self._circuit_breaker.record_failure(
                        candidate.provider.name, candidate.model_id
                    )
                    _log.info(
                        "router_stream_failed_mid_stream",
                        provider=candidate.provider.name,
                        model=candidate.model_id,
                        error=str(exc),
                    )
                    raise
                await self._record_failure(chain, index, exc)
                continue
            await self._circuit_breaker.record_success(candidate.provider.name, candidate.model_id)
            return
        raise AllProvidersUnavailableError(f"all providers exhausted; last error: {last_error}")

    async def embed(self, chain: RouteChain, text: str) -> EmbeddingResult:
        """Embedding from the first candidate that is allowed and answers."""
        last_error: Exception | None = None
        for index, candidate in enumerate(chain):
            if not await self._is_allowed(chain, index):
                continue
            try:
                result = await candidate.provider.embed(candidate.model_id, text)
            except (ProviderUnavailableError, ProviderTimeoutError) as exc:
                last_error = exc
                await self._record_failure(chain, index, exc)
                continue
            await self._circuit_breaker.record_success(candidate.provider.name, candidate.model_id)
            return result
        raise AllProvidersUnavailableError(f"all providers exhausted; last error: {last_error}")

    async def _is_allowed(self, chain: RouteChain, index: int) -> bool:
        """Ask the breaker about ``chain[index]``; report a fallback when its circuit is open."""
        candidate = chain[index]
        if await self._circuit_breaker.is_allowed(candidate.provider.name, candidate.model_id):
            return True
        _log.debug(
            "router_skip_open_circuit",
            provider=candidate.provider.name,
            model=candidate.model_id,
        )
        self._report_fallback(chain, index, "circuit_open")
        return False

    async def _record_failure(self, chain: RouteChain, index: int, exc: ProviderError) -> None:
        """Count a transient failure against ``chain[index]`` and report the fallback."""
        candidate = chain[index]
        await self._circuit_breaker.record_failure(candidate.provider.name, candidate.model_id)
        _log.info(
            "router_provider_failed",
            provider=candidate.provider.name,
            model=candidate.model_id,
            error=str(exc),
        )
        self._report_fallback(chain, index, type(exc).__name__)

    def _report_fallback(self, chain: RouteChain, index: int, reason: str) -> None:
        """Fire ``on_fallback`` when a next candidate exists to fall back to."""
        if self._on_fallback is not None and index + 1 < len(chain):
            self._on_fallback(chain[index].model_id, chain[index + 1].model_id, reason)
