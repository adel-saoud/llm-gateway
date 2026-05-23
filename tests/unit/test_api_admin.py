"""Integration tests for ``PUT /v1/admin/teams/{team_id}``."""

from __future__ import annotations

from pathlib import Path

from llm_gateway.auth.models import TeamConfig
from llm_gateway.config import Settings
from tests.conftest import gateway_client_factory, make_team

ADMIN_AUTH = {"Authorization": "Bearer sk-admin"}
USER_AUTH = {"Authorization": "Bearer sk-user"}


def _setup_teams() -> list[TeamConfig]:
    return [
        make_team(team_id="admin", api_key="sk-admin", is_admin=True),
        make_team(team_id="user", api_key="sk-user", is_admin=False),
    ]


def test_admin_can_update_budget(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path, teams=_setup_teams()) as client:
        response = client.put(
            "/v1/admin/teams/user",
            headers=ADMIN_AUTH,
            json={"budget": {"daily_usd": 250.0}},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["team_id"] == "user"
    assert body["budget"]["daily_usd"] == 250.0


def test_admin_can_update_rate_limits(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path, teams=_setup_teams()) as client:
        response = client.put(
            "/v1/admin/teams/user",
            headers=ADMIN_AUTH,
            json={"rate_limits": {"requests_per_minute": 120, "burst_capacity": 200}},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["rate_limits"]["requests_per_minute"] == 120
    assert body["rate_limits"]["burst_capacity"] == 200


def test_admin_can_update_allowed_models(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path, teams=_setup_teams()) as client:
        response = client.put(
            "/v1/admin/teams/user",
            headers=ADMIN_AUTH,
            json={"allowed_models": ["qwen2.5:0.5b"]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["allowed_models"] == ["qwen2.5:0.5b"]


def test_non_admin_is_forbidden(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path, teams=_setup_teams()) as client:
        response = client.put(
            "/v1/admin/teams/user",
            headers=USER_AUTH,
            json={"budget": {"daily_usd": 999.0}},
        )
    assert response.status_code == 403


def test_unknown_team_returns_404(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path, teams=_setup_teams()) as client:
        response = client.put(
            "/v1/admin/teams/does-not-exist",
            headers=ADMIN_AUTH,
            json={"budget": {"daily_usd": 10.0}},
        )
    assert response.status_code == 404


def test_invalid_field_rejected(settings: Settings, tmp_path: Path) -> None:
    """``extra="forbid"`` on the update schema means unknown fields → 422."""
    with gateway_client_factory(settings, tmp_path, teams=_setup_teams()) as client:
        response = client.put(
            "/v1/admin/teams/user",
            headers=ADMIN_AUTH,
            json={"budget": {"daily_usd": 10.0, "monthly_usd": 999.0}},
        )
    assert response.status_code == 422


def test_admin_update_takes_effect_on_subsequent_resolves(
    settings: Settings, tmp_path: Path
) -> None:
    """After PUT, the new resolver should see the updated team."""
    with gateway_client_factory(settings, tmp_path, teams=_setup_teams()) as client:
        client.put(
            "/v1/admin/teams/user",
            headers=ADMIN_AUTH,
            json={"budget": {"daily_usd": 250.0}},
        )
        # Subsequent admin GET (via existing endpoint) — we don't have GET /v1/admin/teams
        # but the next request from sk-user should see the new budget via /v1/stats
        response = client.get("/v1/stats", headers=USER_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["daily_cap_usd"] == 250.0
