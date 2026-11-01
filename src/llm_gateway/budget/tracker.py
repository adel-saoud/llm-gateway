"""Budget tracker: records cost events to the SQLite ledger.

Cost is computed from token counts and the per-1k prices in
:class:`~llm_gateway.providers.registry.ModelConfig`. The tracker is the only
component that writes to the ledger; the enforcer reads from it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from llm_gateway.providers.registry import ProviderRegistry
from llm_gateway.rag.models import RagCostEvent, RagOperationType
from llm_gateway.storage.db import CostLedger, CostLedgerEntry

_log = structlog.get_logger(__name__)


class BudgetTracker:
    """Records cost events and exposes day-totals to the enforcer."""

    def __init__(self, ledger: CostLedger, registry: ProviderRegistry) -> None:
        self._ledger = ledger
        self._registry = registry

    async def record(
        self,
        *,
        team_id: str,
        model_id: str,
        operation_type: RagOperationType,
        input_tokens: int,
        output_tokens: int,
        request_id: str,
    ) -> RagCostEvent:
        """Compute cost, persist, and return the structured event.

        Unknown models are recorded with zero cost: the gateway shouldn't
        crash on misconfiguration, but the structlog event records the miss
        so it shows up in dashboards.
        """
        recorded_at = datetime.now(UTC)
        cost_usd = 0.0
        if self._registry.has(model_id):
            cost_usd = self._registry.get(model_id).cost_for(input_tokens, output_tokens)
        else:
            _log.warning(
                "budget_unknown_model",
                team_id=team_id,
                model_id=model_id,
                request_id=request_id,
            )

        await self._ledger.record(
            CostLedgerEntry(
                ts=recorded_at,
                team_id=team_id,
                model_id=model_id,
                operation_type=operation_type.value,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                request_id=request_id,
            )
        )
        event = RagCostEvent(
            team_id=team_id,
            model_id=model_id,
            operation_type=operation_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
            ts=recorded_at,
        )
        _log.info(
            "cost_event",
            team_id=team_id,
            model_id=model_id,
            operation_type=operation_type.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            request_id=request_id,
        )
        return event

    async def total_for_today(self, team_id: str) -> float:
        """The team's spend so far on the current UTC calendar day."""
        return await self._ledger.total_for_day(team_id, day=datetime.now(UTC))
