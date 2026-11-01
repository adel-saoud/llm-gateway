from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from llm_gateway.auth.models import BudgetConfig, RateLimitConfig, TeamConfig


def _make_team(**overrides: object) -> TeamConfig:
    base: dict[str, object] = {
        "team_id": "search-team",
        "api_key": SecretStr("sk-gw-x"),
        "allowed_models": ("qwen3.5:4b",),
        "rate_limits": RateLimitConfig(requests_per_minute=60, burst_capacity=100),
        "budget": BudgetConfig(daily_usd=10.0),
    }
    base.update(overrides)
    return TeamConfig.model_validate(base)


def test_refill_per_second_derived() -> None:
    rl = RateLimitConfig(requests_per_minute=60, burst_capacity=100)
    assert rl.refill_per_second == 1.0
    rl = RateLimitConfig(requests_per_minute=120, burst_capacity=200)
    assert rl.refill_per_second == 2.0


def test_empty_allowed_models_means_unrestricted() -> None:
    team = _make_team(allowed_models=())
    assert team.is_model_allowed("anything") is True
    assert team.is_model_allowed("qwen3.5:4b") is True


def test_allowed_models_whitelist_enforced() -> None:
    team = _make_team(allowed_models=("qwen3.5:4b", "qwen3-embedding:0.6b"))
    assert team.is_model_allowed("qwen3.5:4b") is True
    assert team.is_model_allowed("qwen3-embedding:0.6b") is True
    assert team.is_model_allowed("qwen3.5:0.8b") is False


def test_allowed_models_may_name_tiers() -> None:
    team = _make_team(allowed_models=("budget", "embeddings"))
    assert team.is_model_allowed("any-budget-model", "budget") is True
    assert team.is_model_allowed("any-embedder", "embeddings") is True
    assert team.is_model_allowed("any-big-model", "high_quality") is False
    # A model the registry does not know has no tier, so a tier entry never admits it.
    assert team.is_model_allowed("unknown-model", None) is False


def test_team_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        TeamConfig.model_validate(
            {
                "team_id": "x",
                "api_key": "sk-x",
                "allowed_models": [],
                "rate_limits": {"requests_per_minute": 1, "burst_capacity": 1},
                "budget": {"daily_usd": 1.0},
                "unknown_field": "boom",
            }
        )


def test_budget_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        BudgetConfig(daily_usd=0.0)
    with pytest.raises(ValidationError):
        BudgetConfig(daily_usd=-1.0)


def test_rate_limit_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RateLimitConfig(requests_per_minute=0, burst_capacity=1)
    with pytest.raises(ValidationError):
        RateLimitConfig(requests_per_minute=1, burst_capacity=0)


def test_team_config_is_frozen() -> None:
    team = _make_team()
    with pytest.raises(ValidationError):
        team.team_id = "other"  # type: ignore[misc]
