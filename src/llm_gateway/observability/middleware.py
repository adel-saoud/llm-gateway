"""ASGI middleware that records HTTP-level metrics for every request.

The middleware is registered *outermost* so it sees every request, including
ones that auth/rate-limit reject. It increments :data:`Metrics.requests_total`
and observes :data:`Metrics.request_duration_seconds` with the team from
``request.state.team`` (set by :class:`~llm_gateway.auth.middleware.AuthMiddleware`)
or the placeholder ``"unauthenticated"`` when auth wasn't reached.

Pure ASGI: response bodies stream through unbuffered. The status code is
captured by wrapping the ``send`` channel and reading the
``http.response.start`` message.

When a handler reports its provider wait time in ``request.state.upstream_s``
(non-streaming completions do), the middleware also adds a ``Server-Timing``
header and observes :data:`Metrics.overhead_seconds`, the time the gateway
spent outside the provider call.

The ``/metrics`` endpoint is exempt: instrumenting the scrape endpoint
would inflate its own counter on every Prometheus scrape.

The :class:`~llm_gateway.observability.metrics.Metrics` instance is read from
``app.state.metrics`` so tests can swap it without re-registering middleware.
"""

from __future__ import annotations

import time
from typing import Any, Final, cast

import structlog
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_gateway.auth.models import TeamConfig
from llm_gateway.observability.metrics import Metrics

_log = structlog.get_logger(__name__)

_SCRAPE_PATH: Final[str] = "/metrics"
"""Not instrumented: counting the scrape would inflate its own counter every interval."""
_UNAUTHENTICATED_TEAM: Final[str] = "unauthenticated"
"""``team`` label for requests rejected before auth resolved a team."""
UPSTREAM_SECONDS_STATE_KEY: Final[str] = "upstream_s"
"""``request.state`` attribute a handler sets to report its provider wait time."""


class MetricsMiddleware:
    """Count and time every HTTP request, rejections included."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Time the request and record it once the response (or the exception) is done."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive)
        if request.url.path == _SCRAPE_PATH:
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
                message = _with_server_timing(message, scope, metrics, start)
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


def _with_server_timing(message: Message, scope: Scope, metrics: Metrics, start: float) -> Message:
    """Add ``Server-Timing`` and record overhead when a handler reported its upstream time.

    Handlers set ``request.state.upstream_s`` (seconds spent waiting on the
    provider; 0 for a cache hit). Overhead is everything else between the
    outermost middleware seeing the request and the response headers going
    out: auth, rate limiting, budget, cache, routing, cost recording.
    Streaming responses never set it, because their headers leave before the
    provider has finished.
    """
    raw_state = scope.get("state")
    if not isinstance(raw_state, dict):
        return message
    upstream_s = cast(dict[str, Any], raw_state).get(UPSTREAM_SECONDS_STATE_KEY)
    if not isinstance(upstream_s, float):
        return message
    overhead_s = time.perf_counter() - start - upstream_s
    metrics.record_overhead(team=_team_id_from_scope(scope), overhead_s=overhead_s)
    value = f"gateway;dur={overhead_s * 1000:.3f}, upstream;dur={upstream_s * 1000:.3f}"
    headers = [*cast(list[tuple[bytes, bytes]], message.get("headers", []))]
    headers.append((b"server-timing", value.encode("ascii")))
    return {**message, "headers": headers}


def _team_id_from_scope(scope: Scope) -> str:
    """The resolved team id, read from the raw scope because no ``Request`` exists at send time."""
    raw_state = scope.get("state")
    if not isinstance(raw_state, dict):
        return _UNAUTHENTICATED_TEAM
    state = cast(dict[str, Any], raw_state)
    team = state.get("team")
    if isinstance(team, TeamConfig):
        return team.team_id
    return _UNAUTHENTICATED_TEAM
