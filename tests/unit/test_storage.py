from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llm_gateway.storage.db import CURRENT_SCHEMA_VERSION, CostLedger, CostLedgerEntry


@pytest.fixture
async def ledger(tmp_path: Path) -> CostLedger:
    ledger = CostLedger(tmp_path / "test.db")
    await ledger.initialize()
    return ledger


def _entry(
    *,
    team_id: str = "t",
    model_id: str = "llama3.2:3b",
    operation_type: str = "generation",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cost_usd: float = 0.001,
    request_id: str = "req-1",
    ts: datetime | None = None,
) -> CostLedgerEntry:
    return CostLedgerEntry(
        ts=ts or datetime.now(UTC),
        team_id=team_id,
        model_id=model_id,
        operation_type=operation_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        request_id=request_id,
    )


async def test_initialize_creates_schema(ledger: CostLedger) -> None:
    assert await ledger.schema_version() == CURRENT_SCHEMA_VERSION


async def test_record_then_total_for_day(ledger: CostLedger) -> None:
    now = datetime.now(UTC)
    await ledger.record(_entry(team_id="t1", cost_usd=0.10, ts=now))
    await ledger.record(_entry(team_id="t1", cost_usd=0.25, ts=now))
    await ledger.record(_entry(team_id="t2", cost_usd=0.99, ts=now))

    assert await ledger.total_for_day("t1", day=now) == pytest.approx(0.35)  # pyright: ignore[reportUnknownMemberType]
    assert await ledger.total_for_day("t2", day=now) == pytest.approx(0.99)  # pyright: ignore[reportUnknownMemberType]


async def test_total_for_day_ignores_other_days(ledger: CostLedger) -> None:
    today = datetime.now(UTC)
    yesterday = today - timedelta(days=1)
    await ledger.record(_entry(team_id="t1", cost_usd=1.00, ts=today))
    await ledger.record(_entry(team_id="t1", cost_usd=99.99, ts=yesterday))
    assert await ledger.total_for_day("t1", day=today) == pytest.approx(1.00)  # pyright: ignore[reportUnknownMemberType]


async def test_total_for_day_empty_team(ledger: CostLedger) -> None:
    assert await ledger.total_for_day("nobody", day=datetime.now(UTC)) == 0.0


async def test_totals_by_operation_for_day(ledger: CostLedger) -> None:
    now = datetime.now(UTC)
    await ledger.record(_entry(team_id="t1", operation_type="generation", cost_usd=0.50, ts=now))
    await ledger.record(_entry(team_id="t1", operation_type="generation", cost_usd=0.10, ts=now))
    await ledger.record(_entry(team_id="t1", operation_type="retrieval", cost_usd=0.05, ts=now))
    await ledger.record(_entry(team_id="t1", operation_type="evaluation", cost_usd=0.02, ts=now))

    totals = await ledger.totals_by_operation_for_day("t1", day=now)
    assert totals == {
        "generation": pytest.approx(0.60),  # pyright: ignore[reportUnknownMemberType]
        "retrieval": pytest.approx(0.05),  # pyright: ignore[reportUnknownMemberType]
        "evaluation": pytest.approx(0.02),  # pyright: ignore[reportUnknownMemberType]
    }


async def test_initialize_creates_parent_directory(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c.db"
    ledger = CostLedger(nested)
    await ledger.initialize()
    assert nested.exists()
