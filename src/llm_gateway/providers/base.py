"""Provider interface: structural Protocol over the wire contract.

The gateway speaks to one or more provider backends (today: Ollama). The
``Provider`` Protocol is intentionally minimal: chat, streaming chat, embed,
health: so swapping in a different backend is a matter of writing a new
adapter, not rewiring the routing layer.

Errors are normalised to two domain types so callers don't need to know about
``httpx.ConnectError`` vs ``httpx.ReadTimeout`` vs JSON decode errors: the
distinction the circuit breaker and router care about is *transient vs fatal*.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class ProviderError(Exception):
    """Base class for all provider-layer failures."""


class ProviderTimeoutError(ProviderError):
    """Request exceeded the per-call timeout: counts toward circuit-breaker failure rate."""


class ProviderUnavailableError(ProviderError):
    """Provider returned 5xx, connection refused, or otherwise could not service the call."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn in a chat conversation."""

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token accounting for a single completion."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        """OpenAI's ``total_tokens``."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """A non-streaming chat completion response."""

    model: str
    content: str
    usage: TokenUsage
    latency_ms: float
    finish_reason: str


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One delta in a streaming chat completion.

    The final chunk has ``done=True`` and may carry final ``usage`` figures
    and the provider's ``finish_reason`` (for example ``"length"`` when
    ``max_tokens`` cut the answer short).
    """

    model: str
    delta: str
    done: bool
    usage: TokenUsage | None = None
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """An embedding vector for a single input."""

    model: str
    embedding: tuple[float, ...]
    usage: TokenUsage
    latency_ms: float


@runtime_checkable
class Provider(Protocol):
    """Structural interface every provider adapter implements."""

    @property
    def name(self) -> str:
        """Provider id used in circuit keys and as ``gen_ai.provider.name`` on spans."""
        ...

    async def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> CompletionResult:
        """One non-streaming completion of ``messages`` by ``model``."""
        ...

    def stream_chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """Completion deltas; the last chunk has ``done=True`` and the usage."""
        ...

    async def embed(self, model: str, text: str) -> EmbeddingResult:
        """Embedding of one input string, with the input token count."""
        ...

    async def health(self) -> bool:
        """True when the backend answers; used by ``/health/ready`` only."""
        ...
