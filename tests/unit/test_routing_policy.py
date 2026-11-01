"""Unit tests for the static routing policy."""

from __future__ import annotations

from llm_gateway.providers.base import ChatMessage
from llm_gateway.providers.registry import ModelTier, default_registry
from llm_gateway.routing.policy import (
    AUTO_MODEL,
    RoutingPolicy,
    RoutingPolicyName,
    RoutingRequest,
    StaticTierPolicy,
    tier_chain,
)

REGISTRY = default_registry(
    high_quality_model="qwen3.5:4b",
    budget_model="qwen3.5:0.8b",
    embedding_model="qwen3-embedding:0.6b",
)


def _request(model: str) -> RoutingRequest:
    return RoutingRequest(
        request_id="r1",
        team_id="t1",
        requested_model=model,
        messages=[ChatMessage(role="user", content="hi")],
    )


def test_static_policy_satisfies_protocol() -> None:
    assert isinstance(StaticTierPolicy(REGISTRY), RoutingPolicy)


async def test_high_quality_model_gets_budget_fallback() -> None:
    plan = await StaticTierPolicy(REGISTRY).plan(_request("qwen3.5:4b"))
    assert plan.model_ids == ("qwen3.5:4b", "qwen3.5:0.8b")
    assert plan.policy is RoutingPolicyName.STATIC
    assert plan.reason == "requested_model"


async def test_budget_model_has_no_fallback() -> None:
    plan = await StaticTierPolicy(REGISTRY).plan(_request("qwen3.5:0.8b"))
    assert plan.model_ids == ("qwen3.5:0.8b",)


async def test_auto_defaults_to_high_quality_chain() -> None:
    plan = await StaticTierPolicy(REGISTRY).plan(_request(AUTO_MODEL))
    assert plan.model_ids == ("qwen3.5:4b", "qwen3.5:0.8b")
    assert plan.reason == "auto_default_high_quality"


def test_tier_chain_for_budget_and_embeddings() -> None:
    assert tier_chain(REGISTRY, ModelTier.BUDGET) == ("qwen3.5:0.8b",)
    assert tier_chain(REGISTRY, ModelTier.EMBEDDINGS) == ("qwen3-embedding:0.6b",)
