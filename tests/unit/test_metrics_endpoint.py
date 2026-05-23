"""Integration tests for ``GET /metrics``."""

from __future__ import annotations

from pathlib import Path

from llm_gateway.config import Settings
from llm_gateway.observability.metrics import METRICS_CONTENT_TYPE
from tests.conftest import gateway_client_factory


def test_metrics_endpoint_is_public(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_endpoint_content_type(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.get("/metrics")
    assert response.headers["content-type"].startswith("text/plain")
    assert METRICS_CONTENT_TYPE.startswith("text/plain")


def test_metrics_body_is_prometheus_text(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        # Generate at least one observation first
        client.get("/health")
        response = client.get("/metrics")
    body = response.text
    assert "# HELP" in body
    assert "# TYPE" in body
    assert "llm_gateway_requests_total" in body
