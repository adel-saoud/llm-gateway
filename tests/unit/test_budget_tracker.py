from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_gateway.budget.tracker import BudgetTracker
from llm_gateway.providers.registry import default_registry
from llm_gateway.storage.db import CostLedger


@pytest.fixture
async def tracker(tmp_path: Path) -> BudgetTracker:
    ledger = CostLedger(tmp_path / "budget.db")
    await ledger.initialize()
    registry = default_registry(
        high_quality_model="llama3.2:3b",
        budget_model="qwen2.5:0.5b",
        embedding_model="nomic-embed-text",
    )
    return BudgetTracker(ledger, registry)


async def test_record_computes_cost_from_registry(tracker: BudgetTracker) -> None:
    # llama3.2:3b → 0.003/1k input + 0.015/1k output
    # 1000 input + 500 output → 0.003 + 0.0075 = 0.0105
    record = await tracker.record(
        team_id="t1",
        model_id="llama3.2:3b",
        operation_type="generation",
        input_tokens=1000,
        output_tokens=500,
        request_id="req-1",
    )
    assert record.cost_usd == pytest.approx(0.0105)  # pyright: ignore[reportUnknownMemberType]
    assert record.operation_type == "generation"


async def test_record_uses_supplied_timestamp(tracker: BudgetTracker) -> None:
    ts = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    record = await tracker.record(
        team_id="t1",
        model_id="llama3.2:3b",
        operation_type="generation",
        input_tokens=100,
        output_tokens=50,
        request_id="req-2",
        ts=ts,
    )
    assert record.ts == ts


async def test_record_unknown_model_zero_cost_no_crash(tracker: BudgetTracker) -> None:
    record = await tracker.record(
        team_id="t1",
        model_id="unknown-model",
        operation_type="generation",
        input_tokens=10,
        output_tokens=5,
        request_id="req-3",
    )
    assert record.cost_usd == 0.0


async def test_total_for_today_sums_records(tracker: BudgetTracker) -> None:
    for _ in range(3):
        await tracker.record(
            team_id="t1",
            model_id="qwen2.5:0.5b",
            operation_type="generation",
            input_tokens=1000,
            output_tokens=1000,
            request_id="req",
        )
    # qwen budget: 0.0001/1k input + 0.0002/1k output = 0.0003 per record
    total = await tracker.total_for_today("t1")
    assert total == pytest.approx(0.0009, rel=1e-6)  # pyright: ignore[reportUnknownMemberType]
