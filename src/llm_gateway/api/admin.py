"""``PUT /v1/admin/teams/{team_id}`` — live team-config updates.

Requires the caller's :class:`TeamConfig` to have ``is_admin=True``. The
update is applied to the in-memory :class:`TeamResolver` atomically: we
construct a new resolver and swap it onto ``app.state.team_resolver`` —
existing requests already in flight keep their old reference; new requests
pick up the new resolver on their first auth lookup.

The brief says "no restart needed"; this is what that means in practice.
The YAML file on disk is *not* rewritten — the operator is expected to fold
the change into source control separately. Persistence-across-restart would
need a follow-on layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, HTTPException, Request

from llm_gateway.api.schemas import TeamUpdateRequest, TeamUpdateResponse
from llm_gateway.auth.models import BudgetConfig, RateLimitConfig, TeamConfig
from llm_gateway.auth.resolver import TeamResolver

if TYPE_CHECKING:
    from llm_gateway.auth.models import TeamConfig as _TeamConfig

_log = structlog.get_logger(__name__)

router = APIRouter(tags=["admin"])


def _require_admin(request: Request) -> None:
    team: _TeamConfig = request.state.team
    if not team.is_admin:
        raise HTTPException(
            status_code=403,
            detail="admin privilege required",
        )


def _merge(team: TeamConfig, body: TeamUpdateRequest) -> TeamConfig:
    """Produce a new :class:`TeamConfig` with the requested fields replaced."""
    rate_limits = team.rate_limits
    if body.rate_limits is not None:
        rate_limits = RateLimitConfig(
            requests_per_minute=body.rate_limits.requests_per_minute
            or team.rate_limits.requests_per_minute,
            burst_capacity=body.rate_limits.burst_capacity or team.rate_limits.burst_capacity,
        )
    budget = team.budget
    if body.budget is not None and body.budget.daily_usd is not None:
        budget = BudgetConfig(daily_usd=body.budget.daily_usd)
    allowed_models = team.allowed_models
    if body.allowed_models is not None:
        allowed_models = body.allowed_models
    return TeamConfig(
        team_id=team.team_id,
        api_key=team.api_key,
        allowed_models=allowed_models,
        rate_limits=rate_limits,
        budget=budget,
        is_admin=team.is_admin,
    )


@router.put("/v1/admin/teams/{team_id}")
async def update_team(
    team_id: str, body: TeamUpdateRequest, request: Request
) -> TeamUpdateResponse:
    _require_admin(request)
    resolver: TeamResolver = request.app.state.team_resolver
    target = resolver.get(team_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"unknown team: {team_id}")

    updated = _merge(target, body)
    new_resolver = TeamResolver(
        [updated if t.team_id == team_id else t for t in resolver.all_teams()]
    )
    request.app.state.team_resolver = new_resolver
    _log.info(
        "team_updated",
        team_id=team_id,
        rate_limits=updated.rate_limits.model_dump(),
        budget=updated.budget.model_dump(),
        allowed_models=list(updated.allowed_models),
    )
    return TeamUpdateResponse(
        team_id=updated.team_id,
        allowed_models=updated.allowed_models,
        rate_limits={
            "requests_per_minute": updated.rate_limits.requests_per_minute,
            "burst_capacity": updated.rate_limits.burst_capacity,
        },
        budget={"daily_usd": updated.budget.daily_usd},
    )
