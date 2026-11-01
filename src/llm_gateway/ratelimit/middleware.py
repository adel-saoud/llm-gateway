"""Pure ASGI middleware: gates each request through the team's token bucket.

Runs after :class:`~llm_gateway.auth.middleware.AuthMiddleware`, which puts
the resolved team on ``request.state.team``. Health and metrics paths are
exempt.

Pure ASGI (not ``BaseHTTPMiddleware``) so streamed response bodies pass
through unbuffered. Reads :class:`~llm_gateway.ratelimit.token_bucket.TokenBucket`
from ``app.state.token_bucket``.

On rejection: 429 + ``Retry-After`` header (integer seconds, per RFC 9110).
"""

from __future__ import annotations

import structlog
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_gateway._http import send_json_response
from llm_gateway.auth.middleware import is_public_path
from llm_gateway.auth.models import TeamConfig
from llm_gateway.ratelimit.token_bucket import TokenBucket

_log = structlog.get_logger(__name__)


class RateLimitMiddleware:
    """Reject a team's request with 429 when its token bucket is empty."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass public and unauthenticated requests through; otherwise spend one token."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive)
        if is_public_path(request.url.path):
            await self._app(scope, receive, send)
            return

        team: TeamConfig | None = getattr(request.state, "team", None)
        if team is None:
            # Auth middleware should have produced a 401 before us. If it
            # didn't run, pass through: the route handler will fail closed.
            await self._app(scope, receive, send)
            return

        bucket: TokenBucket | None = getattr(request.app.state, "token_bucket", None)
        if bucket is None:
            _log.error("ratelimit_no_bucket_configured")
            await self._app(scope, receive, send)
            return

        result = await bucket.acquire(
            team.team_id,
            capacity=team.rate_limits.burst_capacity,
            refill_per_second=team.rate_limits.refill_per_second,
        )
        if not result.allowed:
            retry_after = result.retry_after_seconds_ceiled
            _log.info(
                "rate_limit_exceeded",
                team_id=team.team_id,
                path=request.url.path,
                retry_after_s=retry_after,
            )
            await send_json_response(
                send,
                status=429,
                body={"error": "rate_limit_exceeded", "retry_after_s": retry_after},
                extra_headers=[(b"retry-after", str(retry_after).encode("ascii"))],
            )
            return

        await self._app(scope, receive, send)
