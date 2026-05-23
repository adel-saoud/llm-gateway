"""Tiny ASGI-level HTTP helpers shared by the auth and rate-limit middlewares.

These exist so the middlewares can be *pure ASGI* (no Starlette
``BaseHTTPMiddleware`` wrapper that would buffer streamed response bodies).
Streaming response bodies need to pass through every middleware without being
read into memory first, which is why we go straight to the ASGI ``send`` channel.
"""

from __future__ import annotations

import json
from typing import Any

from starlette.types import Send


async def send_json_response(
    send: Send,
    *,
    status: int,
    body: dict[str, Any],
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    """Send a JSON response over the raw ASGI ``send`` channel.

    Equivalent to constructing a :class:`starlette.responses.JSONResponse` and
    awaiting it, but without the indirection — used in error paths that don't
    have a Starlette ``Request`` instance handy.
    """
    payload = json.dumps(body).encode("utf-8")
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode("ascii")),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": payload})
