"""Async Ollama HTTP adapter.

Implements the ``Provider`` Protocol against the Ollama HTTP API. Two endpoints
are touched:

* ``POST /api/chat``: non-streaming returns a single JSON body; streaming
  returns one JSON object per line (NDJSON). The streaming branch yields one
  ``StreamChunk`` per line and a final chunk with ``done=True`` carrying token
  usage so downstream cost attribution stays accurate.
* ``POST /api/embed``: one input per call (the gateway's embeddings API takes
  one string). Unlike the superseded ``/api/embeddings``, it returns
  ``prompt_eval_count``, so embedding cost uses Ollama's own token count.

The minimum Ollama is :data:`MIN_OLLAMA_VERSION`: the first release whose notes
document ``/api/embed`` returning ``prompt_eval_count`` (0.3.3 added the count,
0.3.4 announced the endpoint). An older server fails with a message naming it,
instead of a bare 404 or embeddings priced at zero.

Errors are normalised: connection refused / timeout / 5xx all become
``ProviderUnavailableError`` or ``ProviderTimeoutError``. The router and circuit
breaker care about the *category* of failure, not the underlying exception type.

Response parsing is isolated in the ``_parse_*`` helpers at the bottom of this
module; the public methods stay focused on transport, timing, and error
normalisation.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Final, cast

import httpx
import structlog

from llm_gateway.providers.base import (
    ChatMessage,
    CompletionResult,
    EmbeddingResult,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StreamChunk,
    TokenUsage,
)

MIN_OLLAMA_VERSION: Final = "0.3.4"
_VERSION_HINT: Final = f"this gateway needs Ollama {MIN_OLLAMA_VERSION} or newer"

_log = structlog.get_logger(__name__)


class OllamaProvider:
    """HTTP adapter for an Ollama server."""

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        """The injected client, or one created on first use with a 10 s connect timeout.

        Connect fails fast so the router falls back quickly when Ollama is
        down; the read timeout stays long for slow CPU generation.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout_s, connect=10.0),
            )
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client, unless it was injected (then its owner closes it)."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> CompletionResult:
        """One ``/api/chat`` call; latency covers the HTTP round trip only."""
        payload = _build_chat_payload(
            model,
            messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
        )
        start = time.perf_counter()
        body = await self._post_json("/api/chat", payload)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _parse_chat_response(body, requested_model=model, latency_ms=latency_ms)

    async def stream_chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> AsyncIterator[StreamChunk]:
        """Yield one chunk per NDJSON line of a streaming ``/api/chat`` call."""
        payload = _build_chat_payload(
            model,
            messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
        )
        client = await self._get_client()
        try:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 500:
                    raise ProviderUnavailableError(
                        f"ollama: stream returned {response.status_code}"
                    )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    yield _parse_stream_line(line, requested_model=model)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"ollama: stream timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"ollama: stream HTTP error: {exc}") from exc

    async def embed(self, model: str, text: str) -> EmbeddingResult:
        """One ``/api/embed`` call for a single input string."""
        start = time.perf_counter()
        body = await self._post_json("/api/embed", {"model": model, "input": text})
        latency_ms = (time.perf_counter() - start) * 1000.0
        return _parse_embedding_response(body, requested_model=model, latency_ms=latency_ms)

    async def health(self) -> bool:
        """True iff the Ollama server responds 2xx to ``/api/tags``."""
        client = await self._get_client()
        try:
            response = await client.get("/api/tags")
        except httpx.HTTPError as exc:
            _log.debug("ollama_health_failed", error=str(exc))
            return False
        return response.is_success

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST and decode a JSON object; transport and status failures become provider errors."""
        client = await self._get_client()
        try:
            response = await client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"ollama: timeout calling {path}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"ollama: connection error to {path}: {exc}") from exc

        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"ollama: {path} returned {response.status_code}: {response.text[:200]}"
            )
        if response.status_code == 404 and not response.headers.get("content-type", "").startswith(
            "application/json"
        ):
            # Unknown model: JSON error body. Unknown route: plain-text 404 from
            # an Ollama too old to have it.
            raise ProviderError(f"ollama: {path} returned 404 (no such route); {_VERSION_HINT}")
        if response.status_code >= 400:
            raise ProviderError(
                f"ollama: {path} returned {response.status_code}: {response.text[:200]}"
            )

        try:
            parsed = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(f"ollama: {path} returned non-JSON body") from exc
        return _expect_dict(parsed, field=f"{path} response")


# ── Request payload builders ────────────────────────────────────────────────


def _build_chat_payload(
    model: str,
    messages: Sequence[ChatMessage],
    *,
    stream: bool,
    temperature: float,
    max_tokens: int | None,
    think: bool,
) -> dict[str, Any]:
    """Ollama ``/api/chat`` body; OpenAI's ``max_tokens`` maps to Ollama's ``num_predict``.

    ``think`` is always sent, ``false`` included: a thinking model left to its
    default thinks, and models without thinking ignore the field.
    """
    options: dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    return {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": stream,
        "think": think,
        "options": options,
    }


# ── Response parsers ────────────────────────────────────────────────────────


def _parse_chat_response(
    body: dict[str, Any], *, requested_model: str, latency_ms: float
) -> CompletionResult:
    """A non-streaming ``/api/chat`` body; ``done_reason`` becomes the OpenAI ``finish_reason``."""
    message = _expect_dict(body.get("message"), field="message")
    content = _expect_str(message.get("content"), field="message.content")
    return CompletionResult(
        model=str(body.get("model", requested_model)),
        content=content,
        usage=_usage_from_body(body),
        latency_ms=latency_ms,
        finish_reason=str(body.get("done_reason") or "stop"),
    )


def _parse_stream_line(line: str, *, requested_model: str) -> StreamChunk:
    """One NDJSON line; only the ``done`` line carries usage and a finish reason."""
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"ollama: invalid NDJSON line: {line!r}") from exc
    chunk = _expect_dict(parsed, field="NDJSON line")

    delta = ""
    raw_message = chunk.get("message")
    if isinstance(raw_message, dict):
        raw_content = cast(dict[str, Any], raw_message).get("content")
        if isinstance(raw_content, str):
            delta = raw_content

    done = bool(chunk.get("done", False))
    return StreamChunk(
        model=str(chunk.get("model", requested_model)),
        delta=delta,
        done=done,
        usage=_usage_from_body(chunk) if done else None,
        finish_reason=str(chunk.get("done_reason") or "stop") if done else None,
    )


def _parse_embedding_response(
    body: dict[str, Any],
    *,
    requested_model: str,
    latency_ms: float,
) -> EmbeddingResult:
    """The single vector of an ``/api/embed`` response, priced by ``prompt_eval_count``."""
    embeddings = body.get("embeddings")
    if not isinstance(embeddings, list) or len(cast(list[Any], embeddings)) != 1:
        raise ProviderError("ollama: embeddings is not a list of exactly one vector")
    embedding = _expect_numeric_list(cast(list[Any], embeddings)[0], field="embeddings[0]")
    if not isinstance(body.get("prompt_eval_count"), int):
        raise ProviderError(f"ollama: /api/embed sent no prompt_eval_count; {_VERSION_HINT}")
    return EmbeddingResult(
        model=str(body.get("model", requested_model)),
        embedding=embedding,
        usage=_usage_from_body(body),
        latency_ms=latency_ms,
    )


def _usage_from_body(body: dict[str, Any]) -> TokenUsage:
    """Extract token usage from an Ollama response body.

    The fields are ``prompt_eval_count`` (input) and ``eval_count`` (output);
    both are absent on streaming chunks that aren't the final one, and
    ``/api/embed`` sends only the first.
    """
    prompt_tokens = body.get("prompt_eval_count")
    completion_tokens = body.get("eval_count")
    return TokenUsage(
        prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else 0,
        completion_tokens=int(completion_tokens) if isinstance(completion_tokens, int) else 0,
    )


# ── Field validators ────────────────────────────────────────────────────────
# These raise :class:`ProviderError` with a precise field path so a malformed
# Ollama response surfaces in logs with enough context to diagnose the issue
# without re-running with extra logging.


def _expect_dict(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProviderError(f"ollama: {field} is not a JSON object")
    return cast(dict[str, Any], value)


def _expect_str(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ProviderError(f"ollama: {field} is not a string")
    return value


def _expect_numeric_list(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ProviderError(f"ollama: {field} is not a list")
    out: list[float] = []
    for item in cast(list[Any], value):
        if not isinstance(item, int | float):
            raise ProviderError(f"ollama: {field} contains non-numeric value")
        out.append(float(item))
    return tuple(out)
