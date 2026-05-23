"""ASGI middleware that records HTTP-level metrics for every request.

The middleware is registered *outermost* so it sees every request, including
ones that auth/rate-limit reject. It increments :data:`Metrics.requests_total`
and observes :data:`Metrics.request_duration_seconds` with the team from
``request.state.team`` (set by :class:`~llm_gateway.auth.middleware.AuthMiddleware`)
or the placeholder ``"unauthenticated"`` when auth wasn't reached.

Pure ASGI: response bodies stream through unbuffered. The status code is
captured by wrapping the ``send`` channel and reading the
``http.response.start`` message.

The ``/metrics`` endpoint is exempt — instrumenting the scrape endpoint
would inflate its own counter on every Prometheus scrape.

The :class:`~llm_gateway.observability.metrics.Metrics` instance is read from
``app.state.metrics`` so tests can swap it without re-registering middleware.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Final, cast

import structlog
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway.auth.models import TeamConfig
from llm_gateway.observability.metrics import Metrics

_log = structlog.get_logger(__name__)

DEFAULT_EXEMPT_PATHS: Final[tuple[str, ...]] = ("/metrics",)
_UNAUTHENTICATED: Final[str] = "unauthenticated"


class MetricsMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        exempt_paths: Iterable[str] = DEFAULT_EXEMPT_PATHS,
    ) -> None:
        self._app = app
        self._exempt_paths = tuple(exempt_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive)
        if any(request.url.path == p for p in self._exempt_paths):
            await self._app(scope, receive, send)
            return

        metrics: Metrics | None = getattr(request.app.state, "metrics", None)
        if metrics is None:
            _log.error("metrics_collector_not_configured")
            await self._app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 0

        async def capture_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                raw_status = message.get("status")
                if isinstance(raw_status, int):
                    status_code = raw_status
            await send(message)

        try:
            await self._app(scope, receive, capture_send)
        finally:
            duration_s = time.perf_counter() - start
            team_id = _team_id_from_scope(scope)
            metrics.record_http_request(
                team=team_id,
                status=status_code or 500,
                duration_s=duration_s,
            )


def _team_id_from_scope(scope: Scope) -> str:
    raw_state = scope.get("state")
    if not isinstance(raw_state, dict):
        return _UNAUTHENTICATED
    state = cast(dict[str, Any], raw_state)
    team = state.get("team")
    if isinstance(team, TeamConfig):
        return team.team_id
    return _UNAUTHENTICATED
