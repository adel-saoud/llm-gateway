from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import SecretStr

from llm_gateway.auth.models import BudgetConfig, RateLimitConfig, TeamConfig
from llm_gateway.budget.enforcer import BudgetEnforcer, BudgetStatus
from llm_gateway.budget.tracker import BudgetTracker
from llm_gateway.providers.registry import default_registry
from llm_gateway.storage.db import CostLedger


def _team(daily_cap_usd: float = 10.0) -> TeamConfig:
    return TeamConfig(
        team_id="t1",
        api_key=SecretStr("sk"),
        allowed_models=("qwen3.5:4b",),
        rate_limits=RateLimitConfig(requests_per_minute=60, burst_capacity=10),
        budget=BudgetConfig(daily_usd=daily_cap_usd),
    )


@dataclass
class _Bundle:
    tracker: BudgetTracker
    enforcer: BudgetEnforcer


@pytest.fixture
async def bundle(tmp_path: Path) -> _Bundle:
    ledger = CostLedger(tmp_path / "budget.db")
    await ledger.initialize()
    registry = default_registry(
        high_quality_model="qwen3.5:4b",
        budget_model="qwen3.5:0.8b",
        embedding_model="qwen3-embedding:0.6b",
    )
    tracker = BudgetTracker(ledger, registry)
    return _Bundle(tracker=tracker, enforcer=BudgetEnforcer(tracker))


async def _spend(bundle: _Bundle, usd: float) -> None:
    """Helper: record a synthetic event with a known cost via the unknown-model path."""
    # We bypass the registry by patching cost in-place via direct ledger insert.
    from datetime import UTC, datetime

    from llm_gateway.storage.db import CostLedgerEntry

    # Access ledger via tracker internals: single-purpose test helper.
    ledger: CostLedger = bundle.tracker._ledger  # pyright: ignore[reportPrivateUsage]
    await ledger.record(
        CostLedgerEntry(
            ts=datetime.now(UTC),
            team_id="t1",
            model_id="qwen3.5:4b",
            operation_type="generation",
            input_tokens=0,
            output_tokens=0,
            cost_usd=usd,
            request_id="synthetic",
        )
    )


async def test_ok_below_warn_threshold(bundle: _Bundle) -> None:
    await _spend(bundle, 5.0)  # 50% of 10
    decision = await bundle.enforcer.check(_team())
    assert decision.status is BudgetStatus.OK
    assert decision.allowed is True
    assert decision.utilisation == pytest.approx(0.5)  # pyright: ignore[reportUnknownMemberType]


async def test_warning_between_80_and_100(bundle: _Bundle) -> None:
    await _spend(bundle, 8.5)
    decision = await bundle.enforcer.check(_team())
    assert decision.status is BudgetStatus.WARNING
    assert decision.allowed is True


async def test_blocked_at_100_percent(bundle: _Bundle) -> None:
    await _spend(bundle, 10.0)
    decision = await bundle.enforcer.check(_team())
    assert decision.status is BudgetStatus.BLOCKED
    assert decision.allowed is False


async def test_blocked_over_100_percent(bundle: _Bundle) -> None:
    await _spend(bundle, 12.0)
    decision = await bundle.enforcer.check(_team())
    assert decision.status is BudgetStatus.BLOCKED
    assert decision.allowed is False
    assert decision.utilisation > 1.0


async def test_zero_spend_is_ok(bundle: _Bundle) -> None:
    decision = await bundle.enforcer.check(_team())
    assert decision.status is BudgetStatus.OK
    assert decision.today_usd == 0.0
