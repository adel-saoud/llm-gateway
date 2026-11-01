"""Ollama-compatible stub that answers instantly, for the overhead benchmark.

It implements the three endpoints the gateway calls (``/api/chat``
non-streaming, ``/api/embed``, ``/api/tags``) with fixed bodies, so the
time a request spends here is close to zero and the benchmark measures the
gateway, not a model.

Run::

    uv run uvicorn bench.stub_ollama:app --host 127.0.0.1 --port 18080 --log-level warning
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

# A fixed 1024-dimension vector, the size qwen3-embedding:0.6b returns.
# The stub only needs a stable shape.
_EMBEDDING = [0.01] * 1024


async def chat(request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(
        {
            "model": body.get("model", "stub"),
            "message": {"role": "assistant", "content": "stub answer"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 5,
        }
    )


async def embed(request: Request) -> JSONResponse:
    body = await request.json()
    return JSONResponse(
        {"model": body.get("model", "stub"), "embeddings": [_EMBEDDING], "prompt_eval_count": 8}
    )


async def tags(_: Request) -> JSONResponse:
    return JSONResponse({"models": []})


app = Starlette(
    routes=[
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/embed", embed, methods=["POST"]),
        Route("/api/tags", tags, methods=["GET"]),
    ]
)
