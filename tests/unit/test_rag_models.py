from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llm_gateway.rag.models import RagCostEvent, RagCostReport, RagOperationType


def test_operation_type_string_values() -> None:
    assert RagOperationType.RETRIEVAL == "retrieval"
    assert RagOperationType.RERANKING == "reranking"
    assert RagOperationType.GENERATION == "generation"
    assert RagOperationType.EVALUATION == "evaluation"
    assert RagOperationType.UNKNOWN == "unknown"


def test_operation_type_from_string() -> None:
    assert RagOperationType("retrieval") is RagOperationType.RETRIEVAL
    assert RagOperationType("generation") is RagOperationType.GENERATION


def test_operation_type_rejects_unknown_string() -> None:
    with pytest.raises(ValueError, match="'rerank-strict'"):
        RagOperationType("rerank-strict")


def test_cost_event_is_frozen() -> None:
    event = RagCostEvent(
        team_id="t",
        model_id="m",
        operation_type=RagOperationType.GENERATION,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        request_id="r",
        ts=datetime.now(UTC),
    )
    with pytest.raises(AttributeError):
        event.cost_usd = 99.0  # type: ignore[misc]


def test_cost_report_carries_iso_date() -> None:
    report = RagCostReport(
        team_id="t",
        date="2026-05-23",
        total_cost_usd=0.5,
        by_operation={"generation": 0.5},
    )
    assert report.date == "2026-05-23"
    assert report.total_cost_usd == 0.5
