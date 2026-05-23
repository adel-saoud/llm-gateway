from __future__ import annotations

from pathlib import Path

from llm_gateway.config import Settings
from tests.conftest import FakeProvider, gateway_client_factory


def test_health_endpoint(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_when_ollama_healthy(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path, provider=FakeProvider(healthy=True)) as client:
        response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ollama": True}


def test_ready_endpoint_when_ollama_unhealthy(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path, provider=FakeProvider(healthy=False)) as client:
        response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "ollama": False}
