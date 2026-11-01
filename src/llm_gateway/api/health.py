"""Health endpoints: liveness and readiness."""

from __future__ import annotations

from fastapi import APIRouter, Request

from llm_gateway.providers.base import Provider

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict[str, str]:
    """Liveness: 200 while the process is up."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness(request: Request) -> dict[str, str | bool]:
    """Readiness: true when the upstream Ollama provider responds."""
    provider: Provider = request.app.state.provider
    ok = await provider.health()
    return {"status": "ok" if ok else "degraded", "ollama": ok}
