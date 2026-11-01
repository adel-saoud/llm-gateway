"""``PUT /v1/admin/teams/{team_id}``: change a team's limits without a restart.

The caller's team must have ``is_admin=True``. The update builds a new
:class:`TeamResolver` and swaps it onto ``app.state.team_resolver`` in one
assignment: requests already in flight keep the old resolver, new requests
see the new one, and lookups never need a lock.

The YAML file on disk is not rewritten, so a restart reverts the change; the
operator folds it into the file separately.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request

from llm_gateway.api.schemas import TeamUpdateRequest, TeamUpdateResponse
from llm_gateway.auth.models import BudgetConfig, RateLimitConfig, TeamConfig
from llm_gateway.auth.resolver import TeamResolver

_log = structlog.get_logger(__name__)

router = APIRouter(tags=["admin"])


def _require_admin(request: Request) -> None:
    team: TeamConfig = request.state.team
    if not team.is_admin:
        raise HTTPException(status_code=403, detail="admin privilege required")


def _apply_team_update(team: TeamConfig, body: TeamUpdateRequest) -> TeamConfig:
    """Copy of ``team`` with only the fields present in ``body`` replaced.

    ``model_copy`` keeps every field the request does not mention (the API
    key, the admin flag, the cache opt-in), so a partial update never resets
    them to their defaults.
    """
    updates: dict[str, object] = {}
    if body.rate_limits is not None:
        updates["rate_limits"] = RateLimitConfig(
            requests_per_minute=body.rate_limits.requests_per_minute
            or team.rate_limits.requests_per_minute,
            burst_capacity=body.rate_limits.burst_capacity or team.rate_limits.burst_capacity,
        )
    if body.budget is not None and body.budget.daily_usd is not None:
        updates["budget"] = BudgetConfig(daily_usd=body.budget.daily_usd)
    if body.allowed_models is not None:
        updates["allowed_models"] = body.allowed_models
    return team.model_copy(update=updates)


@router.put("/v1/admin/teams/{team_id}")
async def update_team(
    team_id: str, body: TeamUpdateRequest, request: Request
) -> TeamUpdateResponse:
    """Replace a team's rate limits, budget cap or model whitelist in memory."""
    _require_admin(request)
    resolver: TeamResolver = request.app.state.team_resolver
    target = resolver.get(team_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"unknown team: {team_id}")

    updated = _apply_team_update(target, body)
    request.app.state.team_resolver = TeamResolver(
        [updated if team.team_id == team_id else team for team in resolver.all_teams()]
    )
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
        rate_limits=updated.rate_limits.model_dump(),
        budget=updated.budget.model_dump(),
    )
