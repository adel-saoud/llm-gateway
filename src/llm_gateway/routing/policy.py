"""Routing policies: turn a requested model into an ordered fallback chain.

The :class:`~llm_gateway.providers.router.ProviderRouter` executes a chain; a
``RoutingPolicy`` decides which chain to hand it. Keeping the two apart means a
learned router (see :mod:`llm_gateway.routing.autopilot`) can be swapped in
without touching fallback or circuit-breaker code.

A client opts into policy-driven routing by sending the alias
:data:`AUTO_MODEL` as ``model``. A concrete model id is never overridden: a
caller that names a model gets that model (plus the budget fallback when it is
high-quality), whatever policy is configured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from llm_gateway.providers.base import ChatMessage
from llm_gateway.providers.registry import ModelTier, ProviderRegistry

AUTO_MODEL: Final[str] = "auto"
"""``model`` alias that hands the choice to the configured routing policy."""


class RoutingPolicyName(StrEnum):
    """Values accepted by ``GATEWAY_ROUTING_POLICY``."""

    STATIC = "static"
    AUTOPILOT = "autopilot"


@dataclass(frozen=True, slots=True)
class RoutingRequest:
    """What a policy sees about one chat request."""

    request_id: str
    team_id: str
    requested_model: str
    messages: Sequence[ChatMessage]


@dataclass(frozen=True, slots=True)
class RoutingPlan:
    """An ordered chain of model ids plus why it was chosen.

    ``reason`` feeds the ``X-Gateway-Routing-Reason`` header and the
    ``llm_gateway_routing_decisions_total`` metric, so it must stay a small,
    bounded vocabulary.
    """

    model_ids: tuple[str, ...]
    policy: RoutingPolicyName
    reason: str


@runtime_checkable
class RoutingPolicy(Protocol):
    """Structural interface every routing policy implements."""

    async def plan(self, request: RoutingRequest) -> RoutingPlan:
        """Ordered model chain for ``request``; must not raise for ``AUTO_MODEL``."""
        ...


def tier_chain(registry: ProviderRegistry, tier: ModelTier) -> tuple[str, ...]:
    """Models of ``tier`` first, then the budget tier as fallback for high quality."""
    primary = tuple(m.model_id for m in registry.by_tier(tier))
    if tier is not ModelTier.HIGH_QUALITY:
        return primary
    return primary + tuple(m.model_id for m in registry.by_tier(ModelTier.BUDGET))


class StaticTierPolicy:
    """Requested model, then the budget tier as fallback when it is high quality.

    For :data:`AUTO_MODEL` it serves the high-quality tier, the conservative
    choice when nothing has classified the prompt.
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    async def plan(self, request: RoutingRequest) -> RoutingPlan:
        """Chain for the request; raises ``KeyError`` for a model not in the registry."""
        if request.requested_model == AUTO_MODEL:
            return RoutingPlan(
                model_ids=tier_chain(self._registry, ModelTier.HIGH_QUALITY),
                policy=RoutingPolicyName.STATIC,
                reason="auto_default_high_quality",
            )
        config = self._registry.get(request.requested_model)
        chain = (config.model_id,)
        if config.tier is ModelTier.HIGH_QUALITY:
            chain += tuple(m.model_id for m in self._registry.by_tier(ModelTier.BUDGET))
        return RoutingPlan(
            model_ids=chain,
            policy=RoutingPolicyName.STATIC,
            reason="requested_model",
        )
