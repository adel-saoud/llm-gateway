"""Per-operation cost attribution — composes ledger queries into a report.

The aggregator is intentionally thin: the SQLite ledger already groups by
operation type, so this layer's job is to shape the result into a stable API
contract (the four named operations are always present, ``unknown`` only when
it carries weight) and pick the date the report is for.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from llm_gateway.rag.models import RagCostReport, RagOperationType
from llm_gateway.storage.db import CostLedger

_KNOWN_OPERATIONS: Final[tuple[RagOperationType, ...]] = tuple(
    op for op in RagOperationType if op is not RagOperationType.UNKNOWN
)


class RagCostAttributor:
    """Builds :class:`RagCostReport` instances from the cost ledger."""

    def __init__(self, ledger: CostLedger) -> None:
        self._ledger = ledger

    async def report_for_day(
        self,
        team_id: str,
        *,
        day: datetime | None = None,
    ) -> RagCostReport:
        """Return the team's per-operation breakdown for a UTC calendar day.

        When ``day`` is omitted, the report covers *today* (UTC).
        """
        when = day or datetime.now(UTC)
        totals = await self._ledger.totals_by_operation_for_day(team_id, day=when)

        # Always include the 4 named operations (0.0 if absent). Carry
        # ``unknown`` through only when it has non-zero spend so the response
        # is stable for the common case but still surfaces opaque traffic.
        by_op: dict[str, float] = {op.value: 0.0 for op in _KNOWN_OPERATIONS}
        for op_str, cost in totals.items():
            by_op[op_str] = cost
        if by_op.get(RagOperationType.UNKNOWN.value, 0.0) == 0.0:
            by_op.pop(RagOperationType.UNKNOWN.value, None)

        return RagCostReport(
            team_id=team_id,
            date=when.astimezone(UTC).date().isoformat(),
            total_cost_usd=sum(by_op.values()),
            by_operation=by_op,
        )
