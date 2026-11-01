"""Integration tests for ``GET /v1/rag/cost-report``."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from llm_gateway.config import Settings
from llm_gateway.storage.db import CostLedger, CostLedgerEntry
from tests.conftest import gateway_client_factory, make_team

AUTH = {"Authorization": "Bearer sk-test"}


def _seed_ledger(db_path: Path, *, team_id: str = "t1") -> None:
    ledger = CostLedger(db_path)

    async def _do() -> None:
        await ledger.initialize()
        now = datetime.now(UTC)
        for op, cost in [
            ("generation", 0.58),
            ("retrieval", 0.12),
            ("reranking", 0.08),
            ("evaluation", 0.067),
        ]:
            await ledger.record(
                CostLedgerEntry(
                    ts=now,
                    team_id=team_id,
                    model_id="m",
                    operation_type=op,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=cost,
                    request_id="seed",
                )
            )

    asyncio.run(_do())


def test_cost_report_returns_breakdown(settings: Settings, tmp_path: Path) -> None:
    _seed_ledger(tmp_path / "test.db")
    with gateway_client_factory(settings, tmp_path, team=make_team()) as client:
        response = client.get("/v1/rag/cost-report", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["team"] == "t1"
    assert body["date"] == datetime.now(UTC).date().isoformat()
    assert body["by_operation"]["generation"] == 0.58
    assert body["by_operation"]["retrieval"] == 0.12
    assert body["by_operation"]["reranking"] == 0.08
    assert body["by_operation"]["evaluation"] == 0.067
    assert body["total_cost_usd"] == 0.847


def test_cost_report_empty_for_new_team(settings: Settings, tmp_path: Path) -> None:
    # No seeding: fresh ledger
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.get("/v1/rag/cost-report", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["total_cost_usd"] == 0.0
    assert set(body["by_operation"].keys()) == {
        "retrieval",
        "reranking",
        "generation",
        "evaluation",
    }


def test_cost_report_requires_auth(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.get("/v1/rag/cost-report")
    assert response.status_code == 401
