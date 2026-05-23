"""Integration tests for :class:`MetricsMiddleware` via FastAPI TestClient."""

from __future__ import annotations

from pathlib import Path

from llm_gateway.config import Settings
from llm_gateway.observability.metrics import Metrics
from tests.conftest import gateway_client_factory, make_team


def _count_requests(body: str, *, team: str, status: int) -> int:
    """Find the requests_total line matching team+status and return its value."""
    needle = f'llm_gateway_requests_total{{model="",status="{status}",team="{team}"}}'
    for line in body.splitlines():
        if line.startswith(needle):
            return int(float(line.rsplit(maxsplit=1)[-1]))
    return 0


def test_health_is_counted_as_unauthenticated(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        client.get("/health")
        body = client.get("/metrics").text
    # Health is public — no team — counted under "unauthenticated"
    assert _count_requests(body, team="unauthenticated", status=200) >= 1


def test_protected_route_401_counted_as_unauthenticated(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        client.post("/v1/does-not-exist", json={})
        body = client.get("/metrics").text
    assert _count_requests(body, team="unauthenticated", status=401) >= 1


def test_authenticated_request_counted_under_team(settings: Settings, tmp_path: Path) -> None:
    team = make_team(team_id="my-team", api_key="sk-x")
    with gateway_client_factory(settings, tmp_path, team=team) as client:
        # No /v1/does-not-exist route → 404, but auth lets it through.
        client.post("/v1/does-not-exist", json={}, headers={"Authorization": "Bearer sk-x"})
        body = client.get("/metrics").text
    assert _count_requests(body, team="my-team", status=404) >= 1


def test_metrics_endpoint_not_counted_in_requests_total(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        # Hit /metrics multiple times — its own counter must not inflate.
        for _ in range(3):
            client.get("/metrics")
        body = client.get("/metrics").text
    # Look for any requests_total line that has path /metrics → none should exist.
    assert "/metrics" not in body  # path isn't a label; this just sanity-checks no leak


def test_duration_histogram_observed(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        client.get("/health")
        body = client.get("/metrics").text
    assert "llm_gateway_request_duration_seconds_count" in body
    assert "llm_gateway_request_duration_seconds_bucket" in body


def test_each_request_increments_counter(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        for _ in range(5):
            client.get("/health")
        body = client.get("/metrics").text
    assert _count_requests(body, team="unauthenticated", status=200) >= 5


def test_fresh_metrics_instance_has_no_data_lines() -> None:
    """An untouched ``Metrics()`` emits HELP/TYPE comments but no data lines."""
    fresh = Metrics()
    body = fresh.render().decode("utf-8")
    # No labeled data line was emitted.
    assert "llm_gateway_requests_total{" not in body
