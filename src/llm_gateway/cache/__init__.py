"""Semantic response cache for chat completions."""

from __future__ import annotations

from llm_gateway.cache.completion_cache import (
    X_GATEWAY_CACHE_HEADER,
    X_GATEWAY_CACHE_SIMILARITY_HEADER,
    CacheLookup,
    CompletionCache,
    wants_no_cache,
)
from llm_gateway.cache.semantic import (
    CachedCompletion,
    CacheHit,
    CachePartition,
    CachePrompt,
    SemanticCache,
    build_cache_key,
    extract_literals,
)

__all__ = [
    "X_GATEWAY_CACHE_HEADER",
    "X_GATEWAY_CACHE_SIMILARITY_HEADER",
    "CacheHit",
    "CacheLookup",
    "CachePartition",
    "CachePrompt",
    "CachedCompletion",
    "CompletionCache",
    "SemanticCache",
    "build_cache_key",
    "extract_literals",
    "wants_no_cache",
]
