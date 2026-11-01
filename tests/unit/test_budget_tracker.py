from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_gateway.budget.tracker import BudgetTracker
from llm_gateway.providers.registry import default_registry
from llm_gateway.rag.models import RagOperationType
from llm_gateway.storage.db import CostLedger


@pytest.fixture
async def tracker(tmp_path: Path) -> BudgetTracker:
    ledger = CostLedger(tmp_path / "budget.db")
    await ledger.initialize()
    registry = default_registry(
        high_quality_model="qwen3.5:4b",
        budget_model="qwen3.5:0.8b",
        embedding_model="qwen3-embedding:0.6b",
    )
    return BudgetTracker(ledger, registry)


async def test_record_computes_cost_from_registry(tracker: BudgetTracker) -> None:
    # qwen3.5:4b → 0.003/1k input + 0.015/1k output
    # 1000 input + 500 output → 0.003 + 0.0075 = 0.0105
    record = await tracker.record(
        team_id="t1",
        model_id="qwen3.5:4b",
        operation_type=RagOperationType.GENERATION,
        input_tokens=1000,
        output_tokens=500,
        request_id="req-1",
    )
    assert record.cost_usd == pytest.approx(0.0105)  # pyright: ignore[reportUnknownMemberType]
    assert record.operation_type == "generation"


async def test_record_is_stamped_in_utc(tracker: BudgetTracker) -> None:
    before = datetime.now(UTC)
    record = await tracker.record(
        team_id="t1",
        model_id="qwen3.5:4b",
        operation_type=RagOperationType.GENERATION,
        input_tokens=100,
        output_tokens=50,
        request_id="req-2",
    )
    assert record.ts.tzinfo is UTC
    assert before <= record.ts <= datetime.now(UTC)


async def test_record_unknown_model_zero_cost_no_crash(tracker: BudgetTracker) -> None:
    record = await tracker.record(
        team_id="t1",
        model_id="unknown-model",
        operation_type=RagOperationType.GENERATION,
        input_tokens=10,
        output_tokens=5,
        request_id="req-3",
    )
    assert record.cost_usd == 0.0


async def test_total_for_today_sums_records(tracker: BudgetTracker) -> None:
    for _ in range(3):
        await tracker.record(
            team_id="t1",
            model_id="qwen3.5:0.8b",
            operation_type=RagOperationType.GENERATION,
            input_tokens=1000,
            output_tokens=1000,
            request_id="req",
        )
    # qwen budget: 0.0001/1k input + 0.0002/1k output = 0.0003 per record
    total = await tracker.total_for_today("t1")
    assert total == pytest.approx(0.0009, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]
