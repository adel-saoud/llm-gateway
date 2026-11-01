"""Unit tests for the in-process semantic cache store."""

from __future__ import annotations

import pytest

from llm_gateway.cache.semantic import (
    CachedCompletion,
    SemanticCache,
    build_cache_key,
    extract_literals,
)
from llm_gateway.providers.base import ChatMessage

VECTOR_A = (1.0, 0.0, 0.0)
VECTOR_NEAR_A = (0.99, 0.1, 0.0)  # cosine ~0.995 with VECTOR_A
VECTOR_B = (0.0, 1.0, 0.0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _cache(
    *,
    threshold: float = 0.95,
    ttl_s: float = 60.0,
    max_entries: int = 10,
    max_partitions: int = 10,
    clock: FakeClock | None = None,
) -> SemanticCache:
    return SemanticCache(
        similarity_threshold=threshold,
        ttl_s=ttl_s,
        max_entries_per_partition=max_entries,
        max_partitions=max_partitions,
        clock=clock or FakeClock(),
    )


def _completion(content: str = "cached answer") -> CachedCompletion:
    return CachedCompletion(
        model="qwen3.5:4b",
        content=content,
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
    )


def _key(
    text: str,
    *,
    team_id: str = "t1",
    user: str | None = None,
    model: str = "qwen3.5:4b",
    system: str | None = None,
):
    messages = [ChatMessage(role="user", content=text)]
    if system is not None:
        messages.insert(0, ChatMessage(role="system", content=system))
    return build_cache_key(team_id=team_id, user=user, model=model, messages=messages)


# ── Literal extraction ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cancel order 123", {"123"}),
        ("Cancel order A-123.", {"a-123"}),
        ("refund for 2024-05-01, please", {"2024-05-01"}),
        ('reset the "staging" password', {"staging"}),
        ("what is the return policy?", set[str]()),
        ("use `config.yaml` and v2", {"config.yaml", "v2"}),
        ("“Lyon” store hours", {"lyon"}),
    ],
)
def test_extract_literals(text: str, expected: set[str]) -> None:
    assert extract_literals(text) == frozenset(expected)


# ── Key building ──────────────────────────────────────────────────────────


def test_single_turn_prompt_is_semantic_candidate() -> None:
    _, prompt = _key("cancel order 123", system="be brief")
    assert prompt.semantic_text == "cancel order 123"
    assert prompt.literals == frozenset({"123"})


def test_multi_turn_prompt_is_exact_only() -> None:
    _, prompt = build_cache_key(
        team_id="t1",
        user=None,
        model="m",
        messages=[
            ChatMessage(role="user", content="cancel my order"),
            ChatMessage(role="assistant", content="which one?"),
            ChatMessage(role="user", content="yes, do it"),
        ],
    )
    assert prompt.semantic_text is None
    assert prompt.literals == frozenset()


def test_system_prompt_changes_partition_not_prompt_hash_input() -> None:
    p1, _ = _key("hi", system="persona A")
    p2, _ = _key("hi", system="persona B")
    assert p1 != p2


# ── Exact lookups ─────────────────────────────────────────────────────────


def test_exact_hit_after_put() -> None:
    cache = _cache()
    partition, prompt = _key("what is the return policy?")
    cache.put(partition, prompt, None, _completion())
    hit = cache.get_exact(partition, prompt)
    assert hit is not None
    assert hit.kind == "exact"
    assert hit.similarity == 1.0
    assert hit.completion.content == "cached answer"


def test_exact_miss_on_empty_cache() -> None:
    partition, prompt = _key("anything")
    assert _cache().get_exact(partition, prompt) is None


def test_exact_miss_on_other_prompt_in_same_partition() -> None:
    cache = _cache()
    partition, prompt = _key("first")
    cache.put(partition, prompt, None, _completion())
    _, other = _key("second")
    assert cache.get_exact(partition, other) is None


# ── Semantic lookups ──────────────────────────────────────────────────────


def test_semantic_hit_above_threshold() -> None:
    cache = _cache()
    partition, stored = _key("what is your return policy?")
    cache.put(partition, stored, VECTOR_A, _completion())
    _, query = _key("how do returns work?")
    hit, top = cache.get_semantic(partition, query, VECTOR_NEAR_A)
    assert hit is not None
    assert hit.kind == "semantic"
    assert abs(hit.similarity - 0.995) < 0.01
    assert top == hit.similarity


def test_semantic_miss_below_threshold_reports_top_similarity() -> None:
    cache = _cache()
    partition, stored = _key("what is your return policy?")
    cache.put(partition, stored, VECTOR_A, _completion())
    _, query = _key("store opening hours")
    hit, top = cache.get_semantic(partition, query, VECTOR_B)
    assert hit is None
    assert top is not None
    assert abs(top) < 1e-6


def test_literal_guard_rejects_different_entity_at_identical_embedding() -> None:
    cache = _cache()
    partition, stored = _key("cancel order 123")
    cache.put(partition, stored, VECTOR_A, _completion("order 123 cancelled"))
    _, query = _key("cancel order 124")
    hit, top = cache.get_semantic(partition, query, VECTOR_A)
    assert hit is None
    assert top is not None
    assert abs(top - 1.0) < 1e-6


def test_literal_guard_skips_to_next_candidate_with_matching_literals() -> None:
    cache = _cache()
    partition, first = _key("cancel order 123")
    cache.put(partition, first, VECTOR_A, _completion("order 123 cancelled"))
    _, second = _key("please cancel order 124")
    cache.put(partition, second, VECTOR_NEAR_A, _completion("order 124 cancelled"))
    _, query = _key("cancel my order 124")
    hit, _ = cache.get_semantic(partition, query, VECTOR_A)
    assert hit is not None
    assert hit.completion.content == "order 124 cancelled"


def test_semantic_lookup_ignores_multi_turn_query() -> None:
    cache = _cache()
    partition, stored = _key("cancel my order")
    cache.put(partition, stored, VECTOR_A, _completion())
    _, multi = build_cache_key(
        team_id="t1",
        user=None,
        model="qwen3.5:4b",
        messages=[
            ChatMessage(role="user", content="cancel my order"),
            ChatMessage(role="assistant", content="ok"),
            ChatMessage(role="user", content="cancel my order"),
        ],
    )
    assert cache.get_semantic(partition, multi, VECTOR_A) == (None, None)


def test_multi_turn_entries_are_not_semantic_candidates() -> None:
    cache = _cache()
    partition, multi = build_cache_key(
        team_id="t1",
        user=None,
        model="qwen3.5:4b",
        messages=[
            ChatMessage(role="user", content="a"),
            ChatMessage(role="assistant", content="b"),
            ChatMessage(role="user", content="yes, do it"),
        ],
    )
    cache.put(partition, multi, VECTOR_A, _completion())
    _, query = _key("yes, do it")
    assert cache.get_semantic(partition, query, VECTOR_A) == (None, None)


def test_zero_vector_never_matches() -> None:
    cache = _cache()
    partition, stored = _key("hello")
    cache.put(partition, stored, VECTOR_A, _completion())
    _, query = _key("hi")
    assert cache.get_semantic(partition, query, (0.0, 0.0, 0.0)) == (None, None)


def test_dimension_mismatch_is_a_miss() -> None:
    cache = _cache()
    partition, stored = _key("hello")
    cache.put(partition, stored, VECTOR_A, _completion())
    _, query = _key("hi")
    assert cache.get_semantic(partition, query, (1.0, 0.0)) == (None, None)


# ── Isolation ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "other",
    [
        {"team_id": "t2"},
        {"user": "end-user-b"},
        {"model": "qwen3.5:0.8b"},
        {"system": "other persona"},
    ],
)
def test_partitions_are_isolated(other: dict[str, str]) -> None:
    cache = _cache()
    partition, prompt = _key("what is the return policy?", user="end-user-a")
    cache.put(partition, prompt, VECTOR_A, _completion())
    other_kwargs: dict[str, str] = {"user": "end-user-a", **other}
    other_partition, other_prompt = _key("what is the return policy?", **other_kwargs)
    assert cache.get_exact(other_partition, other_prompt) is None
    assert cache.get_semantic(other_partition, other_prompt, VECTOR_A) == (None, None)


# ── TTL and eviction ──────────────────────────────────────────────────────


def test_entries_expire_after_ttl() -> None:
    clock = FakeClock()
    cache = _cache(ttl_s=10.0, clock=clock)
    partition, prompt = _key("hello")
    cache.put(partition, prompt, VECTOR_A, _completion())
    clock.now = 9.9
    assert cache.get_exact(partition, prompt) is not None
    clock.now = 10.0
    assert cache.get_exact(partition, prompt) is None
    assert cache.get_semantic(partition, prompt, VECTOR_A) == (None, None)


def test_expired_entry_dropped_while_live_one_still_matches() -> None:
    clock = FakeClock()
    cache = _cache(ttl_s=10.0, clock=clock)
    partition, old = _key("old question")
    cache.put(partition, old, VECTOR_B, _completion("old"))
    clock.now = 5.0
    _, fresh = _key("fresh question")
    cache.put(partition, fresh, VECTOR_A, _completion("fresh"))
    clock.now = 12.0
    _, query = _key("fresh question again")
    hit, _ = cache.get_semantic(partition, query, VECTOR_A)
    assert hit is not None
    assert hit.completion.content == "fresh"
    assert cache.get_exact(partition, old) is None


def test_lru_eviction_per_partition() -> None:
    cache = _cache(max_entries=2)
    partition, first = _key("first")
    _, second = _key("second")
    _, third = _key("third")
    cache.put(partition, first, None, _completion())
    cache.put(partition, second, None, _completion())
    assert cache.get_exact(partition, first) is not None  # refreshes "first"
    cache.put(partition, third, None, _completion())
    assert cache.get_exact(partition, second) is None
    assert cache.get_exact(partition, first) is not None
    assert cache.get_exact(partition, third) is not None


def test_lru_eviction_of_partitions() -> None:
    cache = _cache(max_partitions=1)
    p1, prompt1 = _key("hello", team_id="t1")
    p2, prompt2 = _key("hello", team_id="t2")
    cache.put(p1, prompt1, None, _completion())
    cache.put(p2, prompt2, None, _completion())
    assert cache.get_exact(p1, prompt1) is None
    assert cache.get_exact(p2, prompt2) is not None


def test_put_overwrites_same_prompt() -> None:
    cache = _cache()
    partition, prompt = _key("hello")
    cache.put(partition, prompt, VECTOR_A, _completion("v1"))
    cache.put(partition, prompt, VECTOR_A, _completion("v2"))
    hit = cache.get_exact(partition, prompt)
    assert hit is not None
    assert hit.completion.content == "v2"
