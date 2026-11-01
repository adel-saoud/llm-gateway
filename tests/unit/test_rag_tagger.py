"""Tests for the 5-priority RAG operation classifier."""

from __future__ import annotations

import pytest

from llm_gateway.rag.models import RagOperationType
from llm_gateway.rag.tagger import classify

# ── Priority 1: explicit header ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("retrieval", RagOperationType.RETRIEVAL),
        ("RETRIEVAL", RagOperationType.RETRIEVAL),
        ("Generation", RagOperationType.GENERATION),
        ("  evaluation  ", RagOperationType.EVALUATION),
        ("reranking", RagOperationType.RERANKING),
        ("unknown", RagOperationType.UNKNOWN),
    ],
)
def test_header_takes_priority(raw_value: str, expected: RagOperationType) -> None:
    result = classify(
        header_value=raw_value,
        is_embedding_model=True,  # would otherwise force RETRIEVAL
        system_prompts=["You are a judge that scores."],  # would otherwise force EVALUATION
    )
    assert result is expected


def test_unrecognised_header_falls_through() -> None:
    """Typo in the header → fall through to other signals, don't return UNKNOWN."""
    result = classify(
        header_value="retreivl",  # typo
        is_embedding_model=True,
    )
    assert result is RagOperationType.RETRIEVAL


def test_empty_header_falls_through() -> None:
    result = classify(header_value="", is_embedding_model=True)
    assert result is RagOperationType.RETRIEVAL


def test_whitespace_only_header_falls_through() -> None:
    result = classify(header_value="   ", is_embedding_model=False)
    assert result is RagOperationType.GENERATION


# ── Priority 2: embedding model ────────────────────────────────────────────


def test_embedding_model_forces_retrieval() -> None:
    result = classify(
        is_embedding_model=True,
        system_prompts=["judge the answer"],  # ignored: embedding wins
    )
    assert result is RagOperationType.RETRIEVAL


# ── Priority 3: evaluation keywords ────────────────────────────────────────


@pytest.mark.parametrize(
    "system_prompt",
    [
        "You are a strict judge of LLM outputs.",
        "Please EVALUATE the response carefully.",
        "Score this between 1 and 5.",
        "judge",  # bare keyword
    ],
)
def test_evaluation_keyword_in_system_prompt(system_prompt: str) -> None:
    result = classify(system_prompts=[system_prompt])
    assert result is RagOperationType.EVALUATION


# ── Priority 4: reranking keywords ─────────────────────────────────────────


@pytest.mark.parametrize(
    "system_prompt",
    [
        "Rerank these chunks by importance.",
        "Decide on the relevance of each passage.",
        "RERANK the candidates.",
    ],
)
def test_reranking_keyword_in_system_prompt(system_prompt: str) -> None:
    result = classify(system_prompts=[system_prompt])
    assert result is RagOperationType.RERANKING


# ── Priority 5: default ────────────────────────────────────────────────────


def test_default_is_generation_with_no_signals() -> None:
    assert classify() is RagOperationType.GENERATION


def test_default_is_generation_with_neutral_prompt() -> None:
    result = classify(system_prompts=["You are a helpful assistant."])
    assert result is RagOperationType.GENERATION


# ── Precedence ──────────────────────────────────────────────────────────────


def test_evaluation_beats_reranking_when_both_present() -> None:
    # Keyword precedence: EVALUATION is checked before RERANKING.
    result = classify(system_prompts=["judge the relevance of each chunk"])
    assert result is RagOperationType.EVALUATION


def test_embedding_beats_keyword_signals() -> None:
    result = classify(
        is_embedding_model=True,
        system_prompts=["judge this rerank"],
    )
    assert result is RagOperationType.RETRIEVAL


def test_header_beats_everything() -> None:
    result = classify(
        header_value="reranking",
        is_embedding_model=True,
        system_prompts=["judge"],
    )
    assert result is RagOperationType.RERANKING


# ── Multi-prompt aggregation ────────────────────────────────────────────────


def test_keywords_match_across_multiple_system_prompts() -> None:
    result = classify(system_prompts=["You are an assistant.", "Now judge the answer."])
    assert result is RagOperationType.EVALUATION
