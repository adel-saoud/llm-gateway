"""Pure ASGI middleware that resolves the caller's team from the Authorization header.

Health, metrics, and OpenAPI paths are exempt: they're scraped by Prometheus
and load balancers that don't carry credentials. Everything else requires
``Authorization: Bearer <api_key>`` and a matching team in the resolver.

Implemented as pure ASGI (not ``BaseHTTPMiddleware``) so streaming response
bodies pass through unbuffered. The middleware reads
:class:`~llm_gateway.auth.resolver.TeamResolver` from ``app.state.team_resolver``
and the resolved :class:`~llm_gateway.auth.models.TeamConfig` is attached to
``request.state.team`` via the ASGI scope dict.
"""

from __future__ import annotations

from typing import Final

import structlog
from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from llm_gateway._http import send_json_response
from llm_gateway.auth.resolver import TeamResolver

_log = structlog.get_logger(__name__)

PUBLIC_PATHS: Final[tuple[str, ...]] = (
    "/health",
    "/health/ready",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
)
"""Paths (and their sub-paths) served without credentials, to auth and rate limiting alike."""

_WWW_AUTHENTICATE_BEARER: Final[list[tuple[bytes, bytes]]] = [(b"www-authenticate", b"Bearer")]
"""RFC 9110 requires a 401 to name the expected scheme."""


class AuthMiddleware:
    """Validate the caller's API key and attach the team to request state."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass public paths through; answer 401 unless the Bearer key resolves to a team."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope, receive)
        if is_public_path(request.url.path):
            await self._app(scope, receive, send)
            return

        resolver: TeamResolver | None = getattr(request.app.state, "team_resolver", None)
        if resolver is None:
            # Misconfiguration: lifespan should always populate this. Fail
            # closed so we never serve protected routes without auth.
            _log.error("auth_no_resolver_configured")
            await send_json_response(
                send,
                status=401,
                body={"error": "auth_unavailable"},
                extra_headers=_WWW_AUTHENTICATE_BEARER,
            )
            return

        api_key = _extract_bearer_token(request.headers.get("authorization"))
        if api_key is None:
            await send_json_response(
                send,
                status=401,
                body={"error": "missing_credentials"},
                extra_headers=_WWW_AUTHENTICATE_BEARER,
            )
            return

        team = resolver.resolve(api_key)
        if team is None:
            _log.info("auth_rejected", path=request.url.path)
            await send_json_response(
                send,
                status=401,
                body={"error": "invalid_credentials"},
                extra_headers=_WWW_AUTHENTICATE_BEARER,
            )
            return

        request.state.team = team
        await self._app(scope, receive, send)


def is_public_path(path: str) -> bool:
    """True for a :data:`PUBLIC_PATHS` entry or anything below it."""
    return any(path == public or path.startswith(f"{public}/") for public in PUBLIC_PATHS)


def _extract_bearer_token(header_value: str | None) -> str | None:
    """The token of an ``Authorization: Bearer <token>`` header (scheme case-insensitive)."""
    if header_value is None:
        return None
    parts = header_value.split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None
