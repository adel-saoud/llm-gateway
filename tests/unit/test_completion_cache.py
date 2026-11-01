"""Unit tests for the cache orchestration: bypass rules and fail-open lookups."""

from __future__ import annotations

import pytest

from llm_gateway.cache.completion_cache import CompletionCache, wants_no_cache
from llm_gateway.cache.semantic import CachedCompletion, SemanticCache, build_cache_key
from llm_gateway.providers.base import (
    ChatMessage,
    EmbeddingResult,
    ProviderUnavailableError,
    TokenUsage,
)
from tests.conftest import make_team


class RecordingEmbed:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self._fail = fail

    async def __call__(self, text: str) -> EmbeddingResult:
        self.calls.append(text)
        if self._fail:
            raise ProviderUnavailableError("embedding model down")
        return EmbeddingResult(
            model="qwen3-embedding:0.6b",
            embedding=(1.0, 0.0, 0.0),
            usage=TokenUsage(prompt_tokens=4, completion_tokens=0),
            latency_ms=1.0,
        )


def _cache(
    embed: RecordingEmbed, *, enabled: bool = True, semantic_enabled: bool = True
) -> CompletionCache:
    store = SemanticCache(
        similarity_threshold=0.95, ttl_s=60.0, max_entries_per_partition=10, max_partitions=10
    )
    return CompletionCache(
        store,
        embed=embed,
        enabled=enabled,
        semantic_enabled=semantic_enabled,
        embedding_prefix="task: ",
        max_temperature=0.0,
    )


def _completion() -> CachedCompletion:
    return CachedCompletion(
        model="qwen3.5:4b",
        content="answer",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
    )


def _key(*messages: ChatMessage):
    return build_cache_key(team_id="t1", user=None, model="qwen3.5:4b", messages=messages)


# ── Cache-Control parsing ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, False),
        ("", False),
        ("max-age=0", False),
        ("no-cache", True),
        ("No-Store", True),
        ("max-age=0, no-cache", True),
    ],
)
def test_wants_no_cache(header: str | None, expected: bool) -> None:
    assert wants_no_cache(header) is expected


# ── Bypass rules ──────────────────────────────────────────────────────────


def _bypass(cache: CompletionCache, **overrides: object) -> str | None:
    params: dict[str, object] = {
        "team": make_team(semantic_cache=True),
        "stream": False,
        "temperature": 0.0,
        "think": False,
        "cache_control": None,
    }
    params.update(overrides)
    return cache.bypass_reason(**params)  # type: ignore[arg-type]


def test_no_bypass_when_everything_allows_it() -> None:
    assert _bypass(_cache(RecordingEmbed())) is None


def test_bypass_reasons_in_priority_order() -> None:
    assert _bypass(_cache(RecordingEmbed(), enabled=False)) == "disabled"
    cache = _cache(RecordingEmbed())
    assert _bypass(cache, team=make_team(semantic_cache=False)) == "team_opt_out"
    assert _bypass(cache, cache_control="no-cache") == "no_cache_header"
    assert _bypass(cache, stream=True) == "stream"
    assert _bypass(cache, temperature=0.7) == "temperature"
    assert _bypass(cache, think=True) == "thinking"


# ── Lookups ───────────────────────────────────────────────────────────────


async def test_exact_hit_does_not_embed() -> None:
    embed = RecordingEmbed()
    cache = _cache(embed)
    partition, prompt = _key(ChatMessage(role="user", content="hello"))
    first = await cache.lookup(partition, prompt)
    cache.store(partition, prompt, first, _completion())
    embed.calls.clear()
    second = await cache.lookup(partition, prompt)
    assert second.result == "hit_exact"
    assert embed.calls == []


async def test_semantic_lookup_embeds_with_prefix_then_hits() -> None:
    embed = RecordingEmbed()
    cache = _cache(embed)
    partition, prompt = _key(ChatMessage(role="user", content="what is the return policy"))
    miss = await cache.lookup(partition, prompt)
    assert miss.result == "miss"
    assert miss.embedding_result is not None
    assert embed.calls == ["task: what is the return policy"]
    cache.store(partition, prompt, miss, _completion())

    _, paraphrase = _key(ChatMessage(role="user", content="how do returns work"))
    hit = await cache.lookup(partition, paraphrase)
    assert hit.result == "hit_semantic"
    assert hit.top_similarity is not None
    assert abs(hit.top_similarity - 1.0) < 1e-6


async def test_multi_turn_miss_does_not_embed() -> None:
    embed = RecordingEmbed()
    cache = _cache(embed)
    partition, prompt = _key(
        ChatMessage(role="user", content="a"),
        ChatMessage(role="assistant", content="b"),
        ChatMessage(role="user", content="c"),
    )
    lookup = await cache.lookup(partition, prompt)
    assert lookup.result == "miss"
    assert embed.calls == []


async def test_embedding_failure_fails_open_and_still_allows_exact_store() -> None:
    cache = _cache(RecordingEmbed(fail=True))
    partition, prompt = _key(ChatMessage(role="user", content="hello"))
    lookup = await cache.lookup(partition, prompt)
    assert lookup.result == "miss"
    assert lookup.embedding_result is None
    cache.store(partition, prompt, lookup, _completion())
    again = await cache.lookup(partition, prompt)
    assert again.result == "hit_exact"


async def test_semantic_disabled_never_embeds() -> None:
    embed = RecordingEmbed()
    cache = _cache(embed, semantic_enabled=False)
    partition, prompt = _key(ChatMessage(role="user", content="what is the return policy"))
    lookup = await cache.lookup(partition, prompt)
    cache.store(partition, prompt, lookup, _completion())
    _, paraphrase = _key(ChatMessage(role="user", content="how do returns work"))
    again = await cache.lookup(partition, paraphrase)
    assert again.result == "miss"
    assert embed.calls == []
