"""``GET /v1/rag/cost-report``: per-operation cost breakdown for the caller's team.

This is the gateway's differentiating endpoint: see
:class:`~llm_gateway.rag.cost_attribution.RagCostAttributor` for the
aggregation logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from llm_gateway.api.schemas import CostReportResponse
from llm_gateway.rag.cost_attribution import RagCostAttributor

if TYPE_CHECKING:
    from llm_gateway.auth.models import TeamConfig
    from llm_gateway.storage.db import CostLedger

router = APIRouter(tags=["rag"])


@router.get("/v1/rag/cost-report")
async def cost_report(request: Request) -> CostReportResponse:
    """Today's (UTC) spend for the calling team, split by RAG operation type."""
    team: TeamConfig = request.state.team
    ledger: CostLedger = request.app.state.cost_ledger
    attributor = RagCostAttributor(ledger)
    report = await attributor.report_for_day(team.team_id)
    return CostReportResponse(
        team=report.team_id,
        date=report.date,
        total_cost_usd=report.total_cost_usd,
        by_operation=dict(report.by_operation),
    )
