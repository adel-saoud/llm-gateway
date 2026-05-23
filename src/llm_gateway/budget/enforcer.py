"""Budget enforcer — warn at 80% of daily cap, block at 100%.

Called by the chat / embedding handlers *before* the LLM is invoked. The
enforcer is read-only (it queries the tracker for today's total) — the
tracker writes to the ledger after the call succeeds.

Every decision is logged with ``structlog`` so the budget audit trail is
complete: every block, every warn, every pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import structlog

from llm_gateway.auth.models import TeamConfig
from llm_gateway.budget.tracker import BudgetTracker

_log = structlog.get_logger(__name__)

WARN_THRESHOLD: Final[float] = 0.80
BLOCK_THRESHOLD: Final[float] = 1.00


class BudgetStatus(StrEnum):
    """Decision the enforcer returns."""

    OK = "ok"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Result of one ``check`` call."""

    status: BudgetStatus
    today_usd: float
    daily_cap_usd: float
    utilisation: float

    @property
    def allowed(self) -> bool:
        return self.status is not BudgetStatus.BLOCKED


class BudgetEnforcer:
    def __init__(self, tracker: BudgetTracker) -> None:
        self._tracker = tracker

    async def check(self, team: TeamConfig) -> BudgetDecision:
        """Read today's spend and classify it against the team's daily cap."""
        today = await self._tracker.total_for_today(team.team_id)
        cap = team.budget.daily_usd
        utilisation = today / cap if cap > 0.0 else 0.0

        if utilisation >= BLOCK_THRESHOLD:
            status = BudgetStatus.BLOCKED
        elif utilisation >= WARN_THRESHOLD:
            status = BudgetStatus.WARNING
        else:
            status = BudgetStatus.OK

        _log.info(
            "budget_check",
            team_id=team.team_id,
            today_usd=today,
            daily_cap_usd=cap,
            utilisation=utilisation,
            status=status.value,
        )
        return BudgetDecision(
            status=status,
            today_usd=today,
            daily_cap_usd=cap,
            utilisation=utilisation,
        )
