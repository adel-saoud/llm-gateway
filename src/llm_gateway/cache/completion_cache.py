"""Gateway-side orchestration of the semantic cache for one chat request.

:class:`CompletionCache` decides whether a request may use the cache, runs
the exact-then-semantic lookup (embedding the prompt only when semantic
matching is enabled, the exact lookup missed and the request is single-turn),
and stores fresh completions.

The cache fails open: if the embedding call fails, the request is served
normally and stored for exact matching only. A cache problem can cost a hit,
never a request.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Literal

import structlog

from llm_gateway.auth.models import TeamConfig
from llm_gateway.cache.semantic import (
    CachedCompletion,
    CacheHit,
    CachePartition,
    CachePrompt,
    SemanticCache,
)
from llm_gateway.providers.base import EmbeddingResult, ProviderError

_log = structlog.get_logger(__name__)

X_GATEWAY_CACHE_HEADER: Final[str] = "X-Gateway-Cache"
"""Response header carrying ``hit``, ``miss`` or ``bypass``."""
X_GATEWAY_CACHE_SIMILARITY_HEADER: Final[str] = "X-Gateway-Cache-Similarity"
"""Response header on hits: 1.0 for exact, the cosine similarity for semantic."""

BypassReason = Literal[
    "disabled", "team_opt_out", "no_cache_header", "stream", "temperature", "thinking"
]
"""``reason`` label of ``llm_gateway_semantic_cache_bypass_total``."""
CacheResult = Literal["hit_exact", "hit_semantic", "miss", "bypass"]
"""``result`` label of ``llm_gateway_semantic_cache_requests_total``."""

EmbedFunction = Callable[[str], Awaitable[EmbeddingResult]]
"""Embeds one prompt; injected so lookups go through the router's circuit breaker."""

_NO_CACHE_DIRECTIVES: Final[frozenset[str]] = frozenset({"no-cache", "no-store"})


def wants_no_cache(cache_control: str | None) -> bool:
    """True when the ``Cache-Control`` request header carries no-cache or no-store."""
    if not cache_control:
        return False
    directives = {part.strip().lower() for part in cache_control.split(",")}
    return not directives.isdisjoint(_NO_CACHE_DIRECTIVES)


@dataclass(frozen=True, slots=True)
class CacheLookup:
    """Outcome of one lookup, carried to :meth:`CompletionCache.store` on a miss.

    ``embedding_result`` is set only when the prompt was embedded; the store
    reuses its vector so a miss is never embedded twice, and the handler
    prices it for the lookup-cost metric.
    """

    hit: CacheHit | None
    embedding_result: EmbeddingResult | None
    top_similarity: float | None
    duration_s: float

    @property
    def result(self) -> CacheResult:
        """The ``result`` label of ``llm_gateway_semantic_cache_requests_total``."""
        if self.hit is None:
            return "miss"
        return "hit_exact" if self.hit.kind == "exact" else "hit_semantic"


class CompletionCache:
    """Policy and orchestration around a :class:`SemanticCache` store."""

    def __init__(
        self,
        store: SemanticCache,
        *,
        embed: EmbedFunction,
        enabled: bool,
        semantic_enabled: bool,
        embedding_prefix: str,
        max_temperature: float,
    ) -> None:
        self._store = store
        self._embed = embed
        self._enabled = enabled
        self._semantic_enabled = semantic_enabled
        self._embedding_prefix = embedding_prefix
        self._max_temperature = max_temperature

    def bypass_reason(
        self,
        *,
        team: TeamConfig,
        stream: bool,
        temperature: float,
        think: bool,
        cache_control: str | None,
    ) -> BypassReason | None:
        """Why this request skips the cache, or None if it may use it.

        The first reason that applies wins. Thinking requests skip it because the
        cache key has no thinking flag: a thought-through answer is what the client
        paid for and must never be served or stored in place of the default one.
        """
        reasons: tuple[tuple[bool, BypassReason], ...] = (
            (not self._enabled, "disabled"),
            (not team.semantic_cache, "team_opt_out"),
            (wants_no_cache(cache_control), "no_cache_header"),
            (stream, "stream"),
            (temperature > self._max_temperature, "temperature"),
            (think, "thinking"),
        )
        return next((reason for applies, reason in reasons if applies), None)

    async def lookup(self, partition: CachePartition, prompt: CachePrompt) -> CacheLookup:
        """Exact match first; embed and search semantically only if that missed and it applies."""
        start = time.perf_counter()
        exact = self._store.get_exact(partition, prompt)
        if exact is not None or prompt.semantic_text is None or not self._semantic_enabled:
            return CacheLookup(
                hit=exact,
                embedding_result=None,
                top_similarity=None,
                duration_s=time.perf_counter() - start,
            )
        try:
            embedding_result = await self._embed(self._embedding_prefix + prompt.semantic_text)
        except ProviderError as exc:
            _log.warning("semantic_cache_embed_failed", error_type=type(exc).__name__)
            return CacheLookup(
                hit=None,
                embedding_result=None,
                top_similarity=None,
                duration_s=time.perf_counter() - start,
            )
        hit, top = self._store.get_semantic(partition, prompt, embedding_result.embedding)
        return CacheLookup(
            hit=hit,
            embedding_result=embedding_result,
            top_similarity=top,
            duration_s=time.perf_counter() - start,
        )

    def store(
        self,
        partition: CachePartition,
        prompt: CachePrompt,
        lookup: CacheLookup,
        completion: CachedCompletion,
    ) -> None:
        """Store a fresh completion, with the lookup's embedding when it has one."""
        embedding_result = lookup.embedding_result
        embedding = embedding_result.embedding if embedding_result is not None else None
        self._store.put(partition, prompt, embedding, completion)
