"""RAG operation classifier: 5-priority decision tree.

First match wins:

1. Explicit ``X-Rag-Operation`` header: caller knows best.
2. Embedding-model call: always :data:`RagOperationType.RETRIEVAL`.
3. System prompt contains ``judge`` / ``evaluate`` / ``score`` →
   :data:`RagOperationType.EVALUATION`.
4. System prompt contains ``rerank`` / ``relevance`` →
   :data:`RagOperationType.RERANKING`.
5. Default: :data:`RagOperationType.GENERATION`.

The classifier is pure: handlers extract the three inputs from the request and
call :func:`classify`; nothing here touches state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from llm_gateway.rag.models import RagOperationType

EVALUATION_KEYWORDS: Final[tuple[str, ...]] = ("judge", "evaluate", "score")
"""Substrings of an LLM-as-judge system prompt; checked before the reranking ones."""
RERANKING_KEYWORDS: Final[tuple[str, ...]] = ("rerank", "relevance")
"""Substrings of a reranking system prompt."""

X_RAG_OPERATION_HEADER: Final[str] = "X-Rag-Operation"
"""Request header a caller sets to tag the operation explicitly; it beats every other signal."""


def classify(
    *,
    header_value: str | None = None,
    is_embedding_model: bool = False,
    system_prompts: Sequence[str] = (),
) -> RagOperationType:
    """Return the :class:`RagOperationType` for one request.

    Args:
        header_value: Raw value of the ``X-Rag-Operation`` HTTP header (any case).
            Unknown values are ignored: the function falls through to the
            other signals rather than returning ``UNKNOWN`` on a typo.
        is_embedding_model: True when the request targets an embedding model
            (e.g. ``/v1/embeddings`` or an embedding-tier model).
        system_prompts: Content of every ``system``-role message in the chat
            request, matched case-insensitively. Empty for embedding calls.
    """
    if header_value is not None:
        canonical = header_value.strip().lower()
        if canonical:
            try:
                return RagOperationType(canonical)
            except ValueError:
                # Unknown header value: fall through to the other signals.
                pass

    if is_embedding_model:
        return RagOperationType.RETRIEVAL

    if system_prompts:
        joined = " ".join(p.lower() for p in system_prompts)
        if any(kw in joined for kw in EVALUATION_KEYWORDS):
            return RagOperationType.EVALUATION
        if any(kw in joined for kw in RERANKING_KEYWORDS):
            return RagOperationType.RERANKING

    return RagOperationType.GENERATION
