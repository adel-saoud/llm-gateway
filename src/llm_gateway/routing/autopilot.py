"""llm-cost-autopilot as a routing policy, over HTTP.

This is the only module that knows about llm-cost-autopilot. Autopilot runs
as a sidecar service (see ``docker-compose.autopilot.yml``) rather than as an
in-process plugin, because its classifiers need torch, sentence-transformers
and a DeBERTa model: an in-process import would put those in the gateway image
and in its startup time.

The wire shapes below mirror autopilot's route API v2 (``POST /v2/route``,
``docs/route-api.schema.json`` in that repository; a copy lives in
``tests/contracts/`` for the contract test). They are copied, not imported, so
the gateway has no dependency on the autopilot package. Decoded responses
ignore unknown fields, and unknown ``reason`` values are bucketed, because v2
allows additive changes.

The gateway stays in charge of what it serves:

* autopilot returns a complexity tier only; the gateway maps it onto its own
  registry through config,
* the team whitelist is applied afterwards by the handler, as for any policy,
* any failure (timeout, HTTP error, invalid body) falls back to the static
  policy, so autopilot can slow routing down by at most ``timeout_s`` and can
  never fail a request.

Autopilot v2 has no outcome endpoint, so nothing the gateway serves is sent
back to it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Final, Literal

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_gateway.providers.registry import ModelTier, ProviderRegistry
from llm_gateway.routing.policy import (
    AUTO_MODEL,
    RoutingPlan,
    RoutingPolicy,
    RoutingPolicyName,
    RoutingRequest,
    tier_chain,
)

_log = structlog.get_logger(__name__)

ROUTE_PATH: Final[str] = "/v2/route"
"""Autopilot route API v2 endpoint (v1 was removed with its ``complex`` tier)."""

AutopilotTier = Literal["simple", "moderate"]
"""Complexity tiers autopilot v2 returns; config maps each onto a gateway :class:`ModelTier`."""

_ROUTABLE_ROLES: Final[frozenset[str]] = frozenset({"system", "user", "assistant"})
"""Roles autopilot v2 accepts; tool turns carry no routing signal."""

_KNOWN_DECISION_REASONS: Final[frozenset[str]] = frozenset(
    {"stage_1_embedding", "stage_2_zero_shot", "correction_memory"}
)
"""Reasons kept verbatim in the routing-reason metric label; others become ``other``."""


# ── Wire shapes (mirror of autopilot route API v2) ─────────────────────────


class RouteMessage(BaseModel):
    """One chat turn as autopilot v1 accepts it."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class RouteRequest(BaseModel):
    """``POST /v2/route`` body; ``tenant_id`` is the team, so correction memory stays per team."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    messages: list[RouteMessage] = Field(min_length=1)


class RouteResponse(BaseModel):
    """``POST /v2/route`` answer; unknown fields are ignored because v2 allows additions."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    route_id: str
    tier: AutopilotTier
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


# ── HTTP client ────────────────────────────────────────────────────────────


class AutopilotClient:
    """Thin async client for the autopilot sidecar."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float,
        api_token: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(timeout_s), headers=headers
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        """Close the HTTP client, unless it was injected (then its owner closes it)."""
        if self._owns_client:
            await self._client.aclose()

    async def route(self, request: RouteRequest) -> RouteResponse:
        """``POST /v2/route``; raises ``httpx.HTTPError`` or ``ValidationError`` on failure."""
        response = await self._client.post(ROUTE_PATH, json=request.model_dump())
        response.raise_for_status()
        return RouteResponse.model_validate_json(response.content)


# ── Policy ─────────────────────────────────────────────────────────────────


class AutopilotPolicy:
    """Routes ``model="auto"`` requests by autopilot's complexity tier."""

    def __init__(
        self,
        *,
        client: AutopilotClient,
        registry: ProviderRegistry,
        fallback: RoutingPolicy,
        tier_map: Mapping[AutopilotTier, ModelTier],
        timeout_s: float,
    ) -> None:
        self._client = client
        self._registry = registry
        self._fallback = fallback
        self._tier_map = tier_map
        self._timeout_s = timeout_s

    async def plan(self, request: RoutingRequest) -> RoutingPlan:
        """Autopilot's tier for ``auto``; the fallback policy for named models or on failure."""
        if request.requested_model != AUTO_MODEL:
            return await self._fallback.plan(request)
        messages = [
            RouteMessage.model_validate({"role": m.role, "content": m.content})
            for m in request.messages
            if m.role in _ROUTABLE_ROLES
        ]
        if not messages:
            return await self._fallback_plan(request, "autopilot_no_routable_message")
        try:
            async with asyncio.timeout(self._timeout_s):
                decision = await self._client.route(
                    RouteRequest(tenant_id=request.team_id, messages=messages)
                )
        except (TimeoutError, httpx.HTTPError, ValidationError) as exc:
            _log.warning(
                "autopilot_route_failed",
                request_id=request.request_id,
                error_type=type(exc).__name__,
            )
            return await self._fallback_plan(request, "autopilot_unavailable")

        _log.debug(
            "autopilot_route_decision",
            request_id=request.request_id,
            route_id=decision.route_id,
            tier=decision.tier,
            confidence=decision.confidence,
        )
        reason = decision.reason if decision.reason in _KNOWN_DECISION_REASONS else "other"
        return RoutingPlan(
            model_ids=tier_chain(self._registry, self._tier_map[decision.tier]),
            policy=RoutingPolicyName.AUTOPILOT,
            reason=f"autopilot_{decision.tier}_{reason}",
        )

    async def _fallback_plan(self, request: RoutingRequest, reason: str) -> RoutingPlan:
        """The fallback policy's chain, relabelled so metrics show why autopilot was skipped."""
        plan = await self._fallback.plan(request)
        return RoutingPlan(model_ids=plan.model_ids, policy=plan.policy, reason=reason)
