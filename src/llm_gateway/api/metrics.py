"""``GET /metrics`` — Prometheus scrape endpoint.

Returns the current state of the gateway's metrics registry in the standard
text exposition format. Stays unauthenticated by intent: Prometheus pulls this
on an interval and shouldn't need credentials.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from llm_gateway.observability.metrics import METRICS_CONTENT_TYPE, Metrics

router = APIRouter(tags=["observability"])


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    collector: Metrics = request.app.state.metrics
    return Response(content=collector.render(), media_type=METRICS_CONTENT_TYPE)
