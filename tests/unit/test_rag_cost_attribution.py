from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llm_gateway.rag.cost_attribution import RagCostAttributor
from llm_gateway.rag.models import RagOperationType
from llm_gateway.storage.db import CostLedger, CostLedgerEntry


async def _record(
    ledger: CostLedger, *, team_id: str, op: RagOperationType, cost: float, ts: datetime
) -> None:
    await ledger.record(
        CostLedgerEntry(
            ts=ts,
            team_id=team_id,
            model_id="qwen3.5:4b",
            operation_type=op.value,
            input_tokens=0,
            output_tokens=0,
            cost_usd=cost,
            request_id="r",
        )
    )


@pytest.fixture
async def attributor(tmp_path: Path) -> RagCostAttributor:
    ledger = CostLedger(tmp_path / "rag.db")
    await ledger.initialize()
    return RagCostAttributor(ledger)


def _ledger(attributor: RagCostAttributor) -> CostLedger:
    return attributor._ledger  # pyright: ignore[reportPrivateUsage]


async def test_empty_day_has_zero_total_and_four_keys(attributor: RagCostAttributor) -> None:
    report = await attributor.report_for_day("t1")
    assert report.team_id == "t1"
    assert report.total_cost_usd == 0.0
    assert set(report.by_operation.keys()) == {
        "retrieval",
        "reranking",
        "generation",
        "evaluation",
    }
    assert all(v == 0.0 for v in report.by_operation.values())


async def test_report_aggregates_by_operation(attributor: RagCostAttributor) -> None:
    now = datetime.now(UTC)
    ledger = _ledger(attributor)
    await _record(ledger, team_id="t1", op=RagOperationType.GENERATION, cost=0.58, ts=now)
    await _record(ledger, team_id="t1", op=RagOperationType.RETRIEVAL, cost=0.12, ts=now)
    await _record(ledger, team_id="t1", op=RagOperationType.RERANKING, cost=0.08, ts=now)
    await _record(ledger, team_id="t1", op=RagOperationType.EVALUATION, cost=0.067, ts=now)

    report = await attributor.report_for_day("t1")
    assert report.by_operation["generation"] == pytest.approx(0.58)  # pyright: ignore[reportUnknownMemberType]
    assert report.by_operation["retrieval"] == pytest.approx(0.12)  # pyright: ignore[reportUnknownMemberType]
    assert report.by_operation["reranking"] == pytest.approx(0.08)  # pyright: ignore[reportUnknownMemberType]
    assert report.by_operation["evaluation"] == pytest.approx(0.067)  # pyright: ignore[reportUnknownMemberType]
    assert report.total_cost_usd == pytest.approx(0.847)  # pyright: ignore[reportUnknownMemberType]


async def test_unknown_traffic_included_when_nonzero(attributor: RagCostAttributor) -> None:
    now = datetime.now(UTC)
    ledger = _ledger(attributor)
    await _record(ledger, team_id="t1", op=RagOperationType.GENERATION, cost=0.5, ts=now)
    await _record(ledger, team_id="t1", op=RagOperationType.UNKNOWN, cost=0.1, ts=now)

    report = await attributor.report_for_day("t1")
    assert "unknown" in report.by_operation
    assert report.by_operation["unknown"] == pytest.approx(0.1)  # pyright: ignore[reportUnknownMemberType]
    assert report.total_cost_usd == pytest.approx(0.6)  # pyright: ignore[reportUnknownMemberType]


async def test_unknown_omitted_when_zero(attributor: RagCostAttributor) -> None:
    now = datetime.now(UTC)
    ledger = _ledger(attributor)
    await _record(ledger, team_id="t1", op=RagOperationType.GENERATION, cost=0.5, ts=now)

    report = await attributor.report_for_day("t1")
    assert "unknown" not in report.by_operation


async def test_report_scoped_to_team(attributor: RagCostAttributor) -> None:
    now = datetime.now(UTC)
    ledger = _ledger(attributor)
    await _record(ledger, team_id="t1", op=RagOperationType.GENERATION, cost=1.0, ts=now)
    await _record(ledger, team_id="t2", op=RagOperationType.GENERATION, cost=99.0, ts=now)

    report = await attributor.report_for_day("t1")
    assert report.by_operation["generation"] == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]


async def test_report_scoped_to_day(attributor: RagCostAttributor) -> None:
    today = datetime.now(UTC)
    yesterday = today - timedelta(days=1)
    ledger = _ledger(attributor)
    await _record(ledger, team_id="t1", op=RagOperationType.GENERATION, cost=1.0, ts=today)
    await _record(ledger, team_id="t1", op=RagOperationType.GENERATION, cost=99.0, ts=yesterday)

    today_report = await attributor.report_for_day("t1", day=today)
    assert today_report.by_operation["generation"] == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
    assert today_report.date == today.date().isoformat()

    yesterday_report = await attributor.report_for_day("t1", day=yesterday)
    assert yesterday_report.by_operation["generation"] == pytest.approx(99.0)  # pyright: ignore[reportUnknownMemberType]
    assert yesterday_report.date == yesterday.date().isoformat()
