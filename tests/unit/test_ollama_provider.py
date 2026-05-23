"""Hermetic tests for ``OllamaProvider``.

Every Ollama HTTP call is intercepted by ``respx`` — no daemon required.
We test the happy path for chat / stream_chat / embed / health, then exercise
the error normalisation paths: 5xx → ``ProviderUnavailableError``, timeout →
``ProviderTimeoutError``, malformed JSON / missing fields → ``ProviderError``.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_gateway.providers.base import (
    ChatMessage,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from llm_gateway.providers.ollama import OllamaProvider

OLLAMA = "http://ollama-test:11434"


def _make_provider() -> OllamaProvider:
    return OllamaProvider(base_url=OLLAMA, timeout_s=5.0)


# ── chat ───────────────────────────────────────────────────────────────────


@respx.mock
async def test_chat_happy_path() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama3.2:3b",
                "message": {"role": "assistant", "content": "Hello!"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 4,
            },
        )
    )
    provider = _make_provider()
    try:
        result = await provider.chat("llama3.2:3b", [ChatMessage(role="user", content="hi")])
        assert result.content == "Hello!"
        assert result.model == "llama3.2:3b"
        assert result.usage.prompt_tokens == 12
        assert result.usage.completion_tokens == 4
        assert result.usage.total_tokens == 16
        assert result.finish_reason == "stop"
        assert result.latency_ms >= 0.0
    finally:
        await provider.aclose()


@respx.mock
async def test_chat_passes_temperature_and_max_tokens() -> None:
    route = respx.post(f"{OLLAMA}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "llama3.2:3b",
                "message": {"role": "assistant", "content": "x"},
                "done": True,
            },
        )
    )
    provider = _make_provider()
    try:
        await provider.chat(
            "llama3.2:3b",
            [ChatMessage(role="user", content="hi")],
            temperature=0.7,
            max_tokens=128,
        )
    finally:
        await provider.aclose()
    sent_request = route.calls[0].request  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    body = json.loads(sent_request.content.decode("utf-8"))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert body["options"]["temperature"] == 0.7
    assert body["options"]["num_predict"] == 128
    assert body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "hi"}]


@respx.mock
async def test_chat_5xx_becomes_provider_unavailable() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(503, text="overloaded"))
    provider = _make_provider()
    try:
        with pytest.raises(ProviderUnavailableError, match="503"):
            await provider.chat("llama3.2:3b", [ChatMessage(role="user", content="hi")])
    finally:
        await provider.aclose()


@respx.mock
async def test_chat_4xx_becomes_provider_error() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(404, text="no such model"))
    provider = _make_provider()
    try:
        with pytest.raises(ProviderError, match="404"):
            await provider.chat("nope", [ChatMessage(role="user", content="hi")])
    finally:
        await provider.aclose()


@respx.mock
async def test_chat_timeout_becomes_provider_timeout() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    provider = _make_provider()
    try:
        with pytest.raises(ProviderTimeoutError):
            await provider.chat("llama3.2:3b", [ChatMessage(role="user", content="hi")])
    finally:
        await provider.aclose()


@respx.mock
async def test_chat_connection_error_becomes_provider_unavailable() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    provider = _make_provider()
    try:
        with pytest.raises(ProviderUnavailableError):
            await provider.chat("llama3.2:3b", [ChatMessage(role="user", content="hi")])
    finally:
        await provider.aclose()


@respx.mock
async def test_chat_malformed_body_raises_provider_error() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(200, json={"done": True}))
    provider = _make_provider()
    try:
        with pytest.raises(ProviderError, match="message"):
            await provider.chat("llama3.2:3b", [ChatMessage(role="user", content="hi")])
    finally:
        await provider.aclose()


# ── stream_chat ────────────────────────────────────────────────────────────


@respx.mock
async def test_stream_chat_emits_deltas_then_final_chunk() -> None:
    ndjson = (
        b'{"model":"llama3.2:3b","message":{"role":"assistant","content":"He"},"done":false}\n'
        b'{"model":"llama3.2:3b","message":{"role":"assistant","content":"llo"},"done":false}\n'
        b'{"model":"llama3.2:3b","message":{"role":"assistant","content":""},'
        b'"done":true,"prompt_eval_count":7,"eval_count":3}\n'
    )
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(200, content=ndjson))
    provider = _make_provider()
    try:
        chunks = [
            c
            async for c in provider.stream_chat(
                "llama3.2:3b", [ChatMessage(role="user", content="hi")]
            )
        ]
    finally:
        await provider.aclose()
    assert [c.delta for c in chunks] == ["He", "llo", ""]
    assert [c.done for c in chunks] == [False, False, True]
    final = chunks[-1]
    assert final.usage is not None
    assert final.usage.prompt_tokens == 7
    assert final.usage.completion_tokens == 3


@respx.mock
async def test_stream_chat_5xx_raises_provider_unavailable() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(503, text="down"))
    provider = _make_provider()
    try:
        with pytest.raises(ProviderUnavailableError):
            async for _ in provider.stream_chat(
                "llama3.2:3b", [ChatMessage(role="user", content="hi")]
            ):
                pass
    finally:
        await provider.aclose()


@respx.mock
async def test_stream_chat_invalid_ndjson_raises_provider_error() -> None:
    respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(200, content=b"not-json\n"))
    provider = _make_provider()
    try:
        with pytest.raises(ProviderError, match="NDJSON"):
            async for _ in provider.stream_chat(
                "llama3.2:3b", [ChatMessage(role="user", content="hi")]
            ):
                pass
    finally:
        await provider.aclose()


# ── embed ──────────────────────────────────────────────────────────────────


@respx.mock
async def test_embed_happy_path() -> None:
    respx.post(f"{OLLAMA}/api/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "nomic-embed-text",
                "embedding": [0.1, 0.2, 0.3, 0.4],
            },
        )
    )
    provider = _make_provider()
    try:
        result = await provider.embed("nomic-embed-text", "hello world")
        assert result.embedding == (0.1, 0.2, 0.3, 0.4)
        assert result.model == "nomic-embed-text"
        # "hello world" → 2 whitespace tokens
        assert result.usage.prompt_tokens == 2
        assert result.usage.completion_tokens == 0
    finally:
        await provider.aclose()


@respx.mock
async def test_embed_missing_field_raises_provider_error() -> None:
    respx.post(f"{OLLAMA}/api/embeddings").mock(
        return_value=httpx.Response(200, json={"model": "x"})
    )
    provider = _make_provider()
    try:
        with pytest.raises(ProviderError, match="embedding"):
            await provider.embed("x", "hello")
    finally:
        await provider.aclose()


# ── health ─────────────────────────────────────────────────────────────────


@respx.mock
async def test_health_true_on_2xx() -> None:
    respx.get(f"{OLLAMA}/api/tags").mock(return_value=httpx.Response(200, json={"models": []}))
    provider = _make_provider()
    try:
        assert await provider.health() is True
    finally:
        await provider.aclose()


@respx.mock
async def test_health_false_on_connection_error() -> None:
    respx.get(f"{OLLAMA}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    provider = _make_provider()
    try:
        assert await provider.health() is False
    finally:
        await provider.aclose()


# ── lifecycle ──────────────────────────────────────────────────────────────


async def test_external_client_not_closed_by_provider() -> None:
    """Caller-supplied clients have their lifecycle managed externally."""
    external = httpx.AsyncClient(base_url=OLLAMA)
    provider = OllamaProvider(base_url=OLLAMA, client=external)
    await provider.aclose()
    # External client remains usable — the provider didn't close it.
    assert external.is_closed is False
    await external.aclose()
