"""In-process semantic cache for chat completions.

Two lookups, cheapest first:

1. **Exact**: sha256 of every message, so a byte-identical request (any
   number of turns) is served without an embedding call.
2. **Semantic**: cosine similarity between the embedding of the request's
   single user message and stored single-turn prompts. Multi-turn requests
   never match semantically, because "yes, do it" after two different
   conversations embeds the same.

A semantic match is accepted only when its similarity reaches the threshold
**and** both prompts carry the same literals (numbers, ids, dates, quoted
strings). Embeddings place "cancel order 123" and "cancel order 124" almost on
top of each other; the literal guard is what keeps those apart. Named entities
without digits ("Paris" vs "Lyon") are not caught by it, which is one reason the
cache is opt-in per team and the threshold is calibrated
(``scripts/calibrate_cache.py``).

Entries are isolated per :class:`CachePartition` (team, end user, requested
model, system prompt), expire after ``ttl_s`` and are evicted LRU per
partition; partitions themselves are evicted LRU past ``max_partitions``.
The store lives in process memory, so each gateway replica has its own cache.

Not thread-safe by design: every method is synchronous and called from the
event loop, so no two calls interleave.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from llm_gateway.providers.base import ChatMessage

type Vector = NDArray[np.float32]
"""Embeddings are stored as float32, half the memory of float64."""

_DIGIT_TOKEN: Final[re.Pattern[str]] = re.compile(r"\S*\d\S*")
"""Tokens that contain a digit: 123, A-123, 2024-05-01, v2.1."""
_QUOTED_SPAN: Final[re.Pattern[str]] = re.compile(r"\"([^\"]+)\"|“([^”]+)”|`([^`]+)`")
"""Spans in straight, curly or backtick quotes, kept whole as one literal."""
_TRIM_CHARS: Final[str] = ".,;:!?()[]{}\"'“”`"
"""Punctuation stripped from token edges so "123," and "123" are the same literal."""


def extract_literals(text: str) -> frozenset[str]:
    """Return the exact-match literals of ``text``, lower-cased."""
    literals = {m.group(0).strip(_TRIM_CHARS).lower() for m in _DIGIT_TOKEN.finditer(text)}
    for match in _QUOTED_SPAN.finditer(text):
        span = next(g for g in match.groups() if g is not None)
        literals.add(span.strip().lower())
    literals.discard("")
    return frozenset(literals)


def _sha256_of_json(payload: object) -> str:
    """Stable digest of a JSON-serialisable value (compact separators, UTF-8)."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CachePartition:
    """Isolation boundary: no entry is ever visible outside its partition."""

    team_id: str
    user: str
    model: str
    system_prompt_hash: str


@dataclass(frozen=True, slots=True)
class CachePrompt:
    """The cache's view of one request's messages."""

    exact_hash: str
    semantic_text: str | None
    literals: frozenset[str]


def build_cache_key(
    *, team_id: str, user: str | None, model: str, messages: Sequence[ChatMessage]
) -> tuple[CachePartition, CachePrompt]:
    """Split a request into its partition and its prompt fingerprint."""
    system_prompts = [m.content for m in messages if m.role == "system"]
    conversation = [m for m in messages if m.role != "system"]
    partition = CachePartition(
        team_id=team_id,
        user=user or "",
        model=model,
        system_prompt_hash=_sha256_of_json(system_prompts),
    )
    single_user_turn = len(conversation) == 1 and conversation[0].role == "user"
    semantic_text = conversation[0].content if single_user_turn else None
    prompt = CachePrompt(
        exact_hash=_sha256_of_json([[m.role, m.content] for m in messages]),
        semantic_text=semantic_text,
        literals=extract_literals(semantic_text) if semantic_text is not None else frozenset(),
    )
    return partition, prompt


@dataclass(frozen=True, slots=True)
class CachedCompletion:
    """The parts of a completion needed to replay it."""

    model: str
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class CacheHit:
    """A replayable completion; ``similarity`` is 1.0 for exact hits."""

    completion: CachedCompletion
    kind: Literal["exact", "semantic"]
    similarity: float


@dataclass(slots=True)
class _Entry:
    """One cached completion; ``embedding`` is a unit vector, or None for multi-turn prompts."""

    prompt: CachePrompt
    embedding: Vector | None
    completion: CachedCompletion
    expires_at: float


@dataclass(slots=True)
class _PartitionStore:
    """Entries of one partition in LRU order, plus the stacked vectors for one-matmul search.

    ``matrix`` and ``matrix_keys`` are rebuilt lazily when ``dirty``, so a
    burst of writes costs one rebuild at the next read.
    """

    entries: OrderedDict[str, _Entry] = field(default_factory=lambda: OrderedDict())
    matrix: Vector | None = None
    matrix_keys: list[str] = field(default_factory=lambda: [])
    dirty: bool = True


def _unit_vector(vector: Sequence[float]) -> Vector | None:
    """``vector`` scaled to length 1, so a dot product is the cosine; None for a zero vector."""
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return None
    return array / norm


class SemanticCache:
    """Exact plus embedding-similarity cache, bounded by TTL and LRU."""

    def __init__(
        self,
        *,
        similarity_threshold: float,
        ttl_s: float,
        max_entries_per_partition: int,
        max_partitions: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = similarity_threshold
        self._ttl_s = ttl_s
        self._max_entries = max_entries_per_partition
        self._max_partitions = max_partitions
        self._clock = clock
        self._partitions: OrderedDict[CachePartition, _PartitionStore] = OrderedDict()

    def get_exact(self, partition: CachePartition, prompt: CachePrompt) -> CacheHit | None:
        """Byte-identical request in this partition, refreshed as most recently used."""
        store = self._live_store(partition)
        if store is None:
            return None
        entry = store.entries.get(prompt.exact_hash)
        if entry is None:
            return None
        store.entries.move_to_end(prompt.exact_hash)
        return CacheHit(completion=entry.completion, kind="exact", similarity=1.0)

    def get_semantic(
        self, partition: CachePartition, prompt: CachePrompt, embedding: Sequence[float]
    ) -> tuple[CacheHit | None, float | None]:
        """Return the best guarded match and the top raw similarity seen.

        The raw similarity is reported even when the guard rejects the match,
        so the ``llm_gateway_semantic_cache_similarity`` histogram shows how
        close near-misses get.
        """
        query = _unit_vector(embedding)
        store = self._live_store(partition)
        if query is None or prompt.semantic_text is None or store is None:
            return None, None
        matrix = self._matrix(store)
        if matrix is None or matrix.shape[1] != query.shape[0]:
            return None, None
        similarities = matrix @ query
        order = np.argsort(-similarities)
        top = float(similarities[order[0]])
        for index in order:
            similarity = float(similarities[index])
            if similarity < self._threshold:
                break
            key = store.matrix_keys[int(index)]
            entry = store.entries[key]
            if entry.prompt.literals == prompt.literals:
                store.entries.move_to_end(key)
                return CacheHit(
                    completion=entry.completion, kind="semantic", similarity=similarity
                ), top
        return None, top

    def put(
        self,
        partition: CachePartition,
        prompt: CachePrompt,
        embedding: Sequence[float] | None,
        completion: CachedCompletion,
    ) -> None:
        """Insert or replace an entry, then evict LRU entries and partitions over the caps.

        ``embedding`` is kept only for single-turn prompts, the only ones
        :meth:`get_semantic` can match.
        """
        store = self._partitions.get(partition)
        if store is None:
            store = _PartitionStore()
            self._partitions[partition] = store
            while len(self._partitions) > self._max_partitions:
                self._partitions.popitem(last=False)
        self._partitions.move_to_end(partition)
        unit = _unit_vector(embedding) if embedding is not None and prompt.semantic_text else None
        store.entries[prompt.exact_hash] = _Entry(
            prompt=prompt,
            embedding=unit,
            completion=completion,
            expires_at=self._clock() + self._ttl_s,
        )
        store.entries.move_to_end(prompt.exact_hash)
        while len(store.entries) > self._max_entries:
            store.entries.popitem(last=False)
        store.dirty = True

    def _live_store(self, partition: CachePartition) -> _PartitionStore | None:
        """The partition's store with expired entries dropped, or None if empty."""
        store = self._partitions.get(partition)
        if store is None:
            return None
        now = self._clock()
        expired = [k for k, e in store.entries.items() if e.expires_at <= now]
        for key in expired:
            del store.entries[key]
        if expired:
            store.dirty = True
        if not store.entries:
            del self._partitions[partition]
            return None
        self._partitions.move_to_end(partition)
        return store

    @staticmethod
    def _matrix(store: _PartitionStore) -> Vector | None:
        """Stacked semantic vectors, rebuilt only after a write so reads stay one matmul."""
        if store.dirty:
            keyed = [(k, e.embedding) for k, e in store.entries.items() if e.embedding is not None]
            store.matrix_keys = [k for k, _ in keyed]
            store.matrix = np.stack([v for _, v in keyed]) if keyed else None
            store.dirty = False
        return store.matrix
