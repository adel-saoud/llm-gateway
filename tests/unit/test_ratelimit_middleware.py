"""Integration tests for RateLimitMiddleware via FastAPI TestClient."""

from __future__ import annotations

from pathlib import Path

from llm_gateway.config import Settings
from tests.conftest import gateway_client_factory, make_team

AUTH = {"Authorization": "Bearer sk-test"}


def test_within_burst_capacity_passes(settings: Settings, tmp_path: Path) -> None:
    team = make_team(requests_per_minute=1, burst_capacity=3)
    with gateway_client_factory(settings, tmp_path, team=team) as client:
        for _ in range(3):
            response = client.post("/v1/does-not-exist", json={}, headers=AUTH)
            # No matching route is registered: 404 confirms the request
            # passed every middleware and only failed at the router.
            assert response.status_code == 404


def test_exceeding_burst_returns_429_with_retry_after(settings: Settings, tmp_path: Path) -> None:
    # 1 rpm, 2 burst → 3rd request inside the same second should be denied
    team = make_team(requests_per_minute=1, burst_capacity=2)
    with gateway_client_factory(settings, tmp_path, team=team) as client:
        for _ in range(2):
            client.post("/v1/does-not-exist", json={}, headers=AUTH)
        third = client.post("/v1/does-not-exist", json={}, headers=AUTH)
    assert third.status_code == 429
    body = third.json()
    assert body["error"] == "rate_limit_exceeded"
    assert "retry_after_s" in body
    assert int(third.headers["Retry-After"]) >= 1


def test_health_path_skips_rate_limit(settings: Settings, tmp_path: Path) -> None:
    team = make_team(requests_per_minute=1, burst_capacity=1)
    with gateway_client_factory(settings, tmp_path, team=team) as client:
        for _ in range(10):
            assert client.get("/health").status_code == 200
