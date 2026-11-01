from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_gateway.storage.db import (
    CircuitStateRecord,
    CircuitStateRepository,
    CostLedger,
)


@pytest.fixture
async def repo(tmp_path: Path) -> CircuitStateRepository:
    db_path = tmp_path / "cb.db"
    # CostLedger.initialize creates the schema, which includes circuit_state.
    ledger = CostLedger(db_path)
    await ledger.initialize()
    return CircuitStateRepository(db_path)


async def test_get_returns_none_when_no_record(repo: CircuitStateRepository) -> None:
    assert await repo.get("ollama", "qwen3.5:4b") is None


async def test_upsert_then_get(repo: CircuitStateRepository) -> None:
    now = datetime.now(UTC)
    record = CircuitStateRecord(
        provider_name="ollama",
        model_id="qwen3.5:4b",
        state="open",
        opened_at=now,
        updated_at=now,
    )
    await repo.upsert(record)
    fetched = await repo.get("ollama", "qwen3.5:4b")
    assert fetched is not None
    assert fetched.state == "open"
    assert fetched.opened_at == now
    assert fetched.updated_at == now


async def test_upsert_overwrites_existing(repo: CircuitStateRepository) -> None:
    t1 = datetime(2026, 1, 1, tzinfo=UTC)
    t2 = datetime(2026, 1, 2, tzinfo=UTC)
    await repo.upsert(
        CircuitStateRecord(
            provider_name="ollama",
            model_id="m",
            state="open",
            opened_at=t1,
            updated_at=t1,
        )
    )
    await repo.upsert(
        CircuitStateRecord(
            provider_name="ollama",
            model_id="m",
            state="closed",
            opened_at=None,
            updated_at=t2,
        )
    )
    fetched = await repo.get("ollama", "m")
    assert fetched is not None
    assert fetched.state == "closed"
    assert fetched.opened_at is None
    assert fetched.updated_at == t2


async def test_independent_keys(repo: CircuitStateRepository) -> None:
    now = datetime.now(UTC)
    await repo.upsert(CircuitStateRecord("ollama", "a", "open", now, now))
    await repo.upsert(CircuitStateRecord("ollama", "b", "closed", None, now))
    a = await repo.get("ollama", "a")
    b = await repo.get("ollama", "b")
    assert a is not None
    assert a.state == "open"
    assert b is not None
    assert b.state == "closed"
