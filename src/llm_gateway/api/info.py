"""Informational endpoints — ``/v1/models`` and ``/v1/stats``.

These give clients a stable view of what the gateway knows about: the model
catalogue (with simulated prices) and the calling team's running spend.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Request

from llm_gateway.api.schemas import (
    ModelInfo,
    ModelsListResponse,
    StatsResponse,
)
from llm_gateway.rag.cost_attribution import RagCostAttributor

if TYPE_CHECKING:
    from llm_gateway.auth.models import TeamConfig
    from llm_gateway.budget.enforcer import BudgetEnforcer
    from llm_gateway.providers.registry import ProviderRegistry
    from llm_gateway.storage.db import CostLedger

_log = structlog.get_logger(__name__)

router = APIRouter(tags=["info"])


@router.get("/v1/models")
async def list_models(request: Request) -> ModelsListResponse:
    registry: ProviderRegistry = request.app.state.registry
    return ModelsListResponse(
        data=[
            ModelInfo(
                id=m.model_id,
                tier=m.tier.value,
                cost_per_1k_input_usd=m.cost_per_1k_input_usd,
                cost_per_1k_output_usd=m.cost_per_1k_output_usd,
                is_embedding_model=m.is_embedding_model,
            )
            for m in registry.all()
        ]
    )


@router.get("/v1/stats")
async def stats(request: Request) -> StatsResponse:
    team: TeamConfig = request.state.team
    enforcer: BudgetEnforcer = request.app.state.budget_enforcer
    ledger: CostLedger = request.app.state.cost_ledger

    decision = await enforcer.check(team)
    attributor = RagCostAttributor(ledger)
    report = await attributor.report_for_day(team.team_id, day=datetime.now(UTC))
    return StatsResponse(
        team=team.team_id,
        today_cost_usd=decision.today_usd,
        budget_utilisation=decision.utilisation,
        daily_cap_usd=decision.daily_cap_usd,
        by_operation=dict(report.by_operation),
    )
