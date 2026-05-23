"""Integration tests for AuthMiddleware via FastAPI TestClient."""

from __future__ import annotations

from pathlib import Path

from llm_gateway.config import Settings
from tests.conftest import gateway_client_factory, make_team


def test_health_is_public(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        assert client.get("/health").status_code == 200


def test_protected_route_returns_401_without_credentials(
    settings: Settings, tmp_path: Path
) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.post("/v1/does-not-exist", json={})
    assert response.status_code == 401
    assert response.json() == {"error": "missing_credentials"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_invalid_bearer_token_returns_401(settings: Settings, tmp_path: Path) -> None:
    team = make_team(api_key="sk-correct")
    with gateway_client_factory(settings, tmp_path, team=team) as client:
        response = client.post(
            "/v1/does-not-exist",
            json={},
            headers={"Authorization": "Bearer sk-wrong"},
        )
    assert response.status_code == 401
    assert response.json() == {"error": "invalid_credentials"}


def test_malformed_authorization_header_returns_401(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.post(
            "/v1/does-not-exist",
            json={},
            headers={"Authorization": "Basic sk-test"},
        )
    assert response.status_code == 401
    assert response.json() == {"error": "missing_credentials"}


def test_empty_bearer_token_returns_401(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.post(
            "/v1/does-not-exist",
            json={},
            headers={"Authorization": "Bearer "},
        )
    assert response.status_code == 401


def test_valid_token_passes_auth_reaches_404(settings: Settings, tmp_path: Path) -> None:
    # No matching route is registered — auth lets the request through and the
    # router responds 404 instead of 401, which proves the token was accepted.
    team = make_team(api_key="sk-valid")
    with gateway_client_factory(settings, tmp_path, team=team) as client:
        response = client.post(
            "/v1/does-not-exist",
            json={},
            headers={"Authorization": "Bearer sk-valid"},
        )
    assert response.status_code == 404
