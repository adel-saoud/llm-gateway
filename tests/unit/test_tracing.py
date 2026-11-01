"""OTel GenAI spans from the chat and embeddings handlers (in-memory exporter)."""

from __future__ import annotations

import json
from collections.abc import Generator
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from llm_gateway import __version__
from llm_gateway.api.completions import _stream_chat  # pyright: ignore[reportPrivateUsage]
from llm_gateway.config import Settings
from llm_gateway.observability.tracing import (
    DEFAULT_SERVICE_NAME,
    GenAiOperationName,
    build_tracer_provider,
    get_tracer,
    start_genai_span,
)
from llm_gateway.providers.base import (
    ProviderError,
    ProviderUnavailableError,
    StreamChunk,
    TokenUsage,
)
from tests.conftest import ScriptedProvider, gateway_client_factory, make_team

AUTH = {"Authorization": "Bearer sk-test"}
MESSAGES = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Where is my order?"},
]
TRACE_ID_HEX = "4bf92f3577b34da6a3ce929d0e0e4736"
TRACEPARENT = f"00-{TRACE_ID_HEX}-00f067aa0ba902b7-01"

BASE_ATTRIBUTES = {
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.response.finish_reasons",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
}


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(exporter: InMemorySpanExporter) -> Generator[TracerProvider]:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider
    provider.shutdown()


def _capture(settings: Settings) -> Settings:
    return settings.model_copy(update={"tracing_capture_message_content": True})


def _chat(client: TestClient, **extra: Any) -> httpx.Response:
    headers = {**AUTH, **extra.pop("headers", {})}
    return client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "qwen3.5:4b", "messages": MESSAGES, **extra},
    )


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


# ── Chat, non-streaming ─────────────────────────────────────────────────────


def test_chat_span_carries_genai_attributes_and_no_content_by_default(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    with gateway_client_factory(
        settings, tmp_path, provider=ScriptedProvider(), tracer_provider=tracer_provider
    ) as client:
        response = _chat(client)
    assert response.status_code == 200
    span = _only_span(exporter)
    assert span.name == "chat qwen3.5:4b"
    assert span.kind is SpanKind.CLIENT
    assert span.status.status_code is StatusCode.UNSET
    attrs = _attrs(span)
    assert set(attrs) == BASE_ATTRIBUTES
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "scripted"
    assert attrs["gen_ai.request.model"] == "qwen3.5:4b"
    assert attrs["gen_ai.response.model"] == "qwen3.5:4b"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5


def test_chat_span_records_messages_when_capture_is_on(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    with gateway_client_factory(
        _capture(settings), tmp_path, provider=ScriptedProvider(), tracer_provider=tracer_provider
    ) as client:
        _chat(client)
    attrs = _attrs(_only_span(exporter))
    assert json.loads(attrs["gen_ai.input.messages"]) == [
        {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
        {"role": "user", "parts": [{"type": "text", "content": "Where is my order?"}]},
    ]
    assert json.loads(attrs["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "scripted-response"}],
            "finish_reason": "stop",
        }
    ]


def test_chat_span_joins_the_callers_trace(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    with gateway_client_factory(
        settings, tmp_path, provider=ScriptedProvider(), tracer_provider=tracer_provider
    ) as client:
        _chat(client, headers={"traceparent": TRACEPARENT})
    span = _only_span(exporter)
    assert span.context is not None
    assert trace.format_trace_id(span.context.trace_id) == TRACE_ID_HEX
    assert span.parent is not None
    assert trace.format_span_id(span.parent.span_id) == "00f067aa0ba902b7"


@pytest.mark.parametrize(
    ("error", "status_code", "error_type"),
    [
        (ProviderUnavailableError("down: secret prompt"), 503, "AllProvidersUnavailableError"),
        (ProviderError("bad request: secret prompt"), 502, "ProviderError"),
    ],
)
def test_chat_span_marks_provider_failures(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
    error: Exception,
    status_code: int,
    error_type: str,
) -> None:
    # Two failures: the static policy falls back from qwen3.5:4b to qwen3.5:0.8b.
    provider = ScriptedProvider(chat_responses=[error, error])
    with gateway_client_factory(
        settings, tmp_path, provider=provider, tracer_provider=tracer_provider
    ) as client:
        response = _chat(client)
    assert response.status_code == status_code
    span = _only_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert _attrs(span)["error.type"] == error_type
    # Exception text can echo the prompt, so it is never recorded.
    assert span.events == ()
    assert "secret prompt" not in (span.status.description or "")


def test_chat_span_marks_unexpected_failures(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    provider = ScriptedProvider(chat_responses=[RuntimeError("boom")])
    with (
        gateway_client_factory(
            settings, tmp_path, provider=provider, tracer_provider=tracer_provider
        ) as client,
        pytest.raises(RuntimeError),
    ):
        _chat(client)
    span = _only_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert _attrs(span)["error.type"] == "RuntimeError"


def test_cache_hit_emits_no_span(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    cache_settings = settings.model_copy(update={"cache_enabled": True})
    with gateway_client_factory(
        cache_settings,
        tmp_path,
        team=make_team(semantic_cache=True),
        provider=ScriptedProvider(),
        tracer_provider=tracer_provider,
    ) as client:
        _chat(client)
        second = _chat(client)
    assert second.headers["X-Gateway-Cache"] == "hit"
    assert len(exporter.get_finished_spans()) == 1


# ── Chat, streaming ─────────────────────────────────────────────────────────


def test_streaming_span_covers_the_whole_stream(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    with gateway_client_factory(
        _capture(settings), tmp_path, provider=ScriptedProvider(), tracer_provider=tracer_provider
    ) as client:
        response = _chat(client, stream=True)
    assert response.text.endswith("data: [DONE]\n\n")
    attrs = _attrs(_only_span(exporter))
    assert attrs["gen_ai.usage.input_tokens"] == 8
    assert attrs["gen_ai.usage.output_tokens"] == 2
    output = json.loads(attrs["gen_ai.output.messages"])
    assert output[0]["parts"][0]["content"] == "Hello world"


def test_streaming_span_records_the_provider_finish_reason(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    chunks = [
        StreamChunk(model="qwen3.5:4b", delta="Hel", done=False),
        StreamChunk(
            model="qwen3.5:4b",
            delta="",
            done=True,
            usage=TokenUsage(prompt_tokens=8, completion_tokens=1),
            finish_reason="length",
        ),
    ]
    provider = ScriptedProvider(stream_chunks=chunks)
    with gateway_client_factory(
        settings, tmp_path, provider=provider, tracer_provider=tracer_provider
    ) as client:
        response = _chat(client, stream=True, max_tokens=1)
    final_chunk = json.loads(response.text.split("data: ")[-2])
    assert final_chunk["choices"][0]["finish_reason"] == "length"
    assert _attrs(_only_span(exporter))["gen_ai.response.finish_reasons"] == ("length",)


class _OpenSpanCounter(SpanProcessor):
    """Counts spans started and not yet ended."""

    def __init__(self) -> None:
        self.open = 0

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        self.open += 1

    def on_end(self, span: ReadableSpan) -> None:
        self.open -= 1


async def test_streaming_span_is_not_left_open_when_the_stream_never_starts() -> None:
    """A client gone before the first chunk closes the generator unstarted.

    Only ``open_span`` matters: the body never runs, so the other arguments
    are placeholders.
    """
    counter = _OpenSpanCounter()
    provider = TracerProvider()
    provider.add_span_processor(counter)
    unused: Any = None
    stream = _stream_chat(
        open_span=partial(
            start_genai_span,
            get_tracer(provider),
            operation=GenAiOperationName.CHAT,
            provider_name="scripted",
            request_model="qwen3.5:4b",
            request_headers={},
        ),
        provider_router=unused,
        chain=[],
        messages=[],
        temperature=0.0,
        max_tokens=None,
        think=False,
        team=unused,
        operation=unused,
        request_id="never-started",
        tracker=unused,
        metrics=unused,
    )
    await stream.aclose()
    assert counter.open == 0


def test_streaming_span_marks_unavailable_providers(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    provider = ScriptedProvider(stream_chunks=[ProviderUnavailableError("down")])  # type: ignore[list-item]
    with gateway_client_factory(
        settings, tmp_path, provider=provider, tracer_provider=tracer_provider
    ) as client:
        response = _chat(client, stream=True)
    assert "providers_unavailable" in response.text
    span = _only_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert _attrs(span)["error.type"] == "AllProvidersUnavailableError"


def test_streaming_span_marks_mid_stream_failures(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    chunks: list[Any] = [
        StreamChunk(model="qwen3.5:4b", delta="Hel", done=False),
        ProviderUnavailableError("dropped"),
    ]
    provider = ScriptedProvider(stream_chunks=chunks)
    with (
        gateway_client_factory(
            settings, tmp_path, provider=provider, tracer_provider=tracer_provider
        ) as client,
        pytest.raises(ProviderUnavailableError),
    ):
        _chat(client, stream=True)
    span = _only_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert _attrs(span)["error.type"] == "ProviderUnavailableError"


# ── Embeddings ──────────────────────────────────────────────────────────────


def _embed(client: TestClient) -> httpx.Response:
    return client.post(
        "/v1/embeddings",
        headers=AUTH,
        json={"model": "qwen3-embedding:0.6b", "input": "my address is 1 rue de Rivoli"},
    )


def test_embeddings_span_has_input_tokens_and_no_content(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    with gateway_client_factory(
        _capture(settings), tmp_path, provider=ScriptedProvider(), tracer_provider=tracer_provider
    ) as client:
        response = _embed(client)
    assert response.status_code == 200
    span = _only_span(exporter)
    assert span.name == "embeddings qwen3-embedding:0.6b"
    assert span.kind is SpanKind.CLIENT
    assert _attrs(span) == {
        "gen_ai.operation.name": "embeddings",
        "gen_ai.provider.name": "scripted",
        "gen_ai.request.model": "qwen3-embedding:0.6b",
        "gen_ai.response.model": "qwen3-embedding:0.6b",
        "gen_ai.usage.input_tokens": 4,
    }


@pytest.mark.parametrize(
    ("error", "status_code", "error_type"),
    [
        (ProviderUnavailableError("down"), 503, "AllProvidersUnavailableError"),
        (ProviderError("bad"), 502, "ProviderError"),
    ],
)
def test_embeddings_span_marks_provider_failures(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
    error: Exception,
    status_code: int,
    error_type: str,
) -> None:
    provider = ScriptedProvider(embed_responses=[error])
    with gateway_client_factory(
        settings, tmp_path, provider=provider, tracer_provider=tracer_provider
    ) as client:
        response = _embed(client)
    assert response.status_code == status_code
    span = _only_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert _attrs(span)["error.type"] == error_type


def test_embeddings_span_marks_unexpected_failures(
    settings: Settings,
    tmp_path: Path,
    tracer_provider: TracerProvider,
    exporter: InMemorySpanExporter,
) -> None:
    provider = ScriptedProvider(embed_responses=[RuntimeError("boom")])
    with (
        gateway_client_factory(
            settings, tmp_path, provider=provider, tracer_provider=tracer_provider
        ) as client,
        pytest.raises(RuntimeError),
    ):
        _embed(client)
    assert _attrs(_only_span(exporter))["error.type"] == "RuntimeError"


# ── Span context ────────────────────────────────────────────────────────────


def test_client_disconnect_ends_the_span_without_marking_it_failed(
    tracer_provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    genai_span = start_genai_span(
        tracer_provider.get_tracer("test"),
        operation=GenAiOperationName.CHAT,
        provider_name="scripted",
        request_model="qwen3.5:4b",
        request_headers={},
    )
    with pytest.raises(GeneratorExit), genai_span:
        raise GeneratorExit
    span = _only_span(exporter)
    assert span.status.status_code is StatusCode.UNSET
    assert "error.type" not in _attrs(span)


# ── Provider setup ──────────────────────────────────────────────────────────


def test_tracing_off_uses_a_noop_tracer(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(settings, tmp_path) as client:
        tracer = client.app.state.tracer  # type: ignore[attr-defined]
    assert isinstance(tracer, trace.NoOpTracer)


def test_tracing_on_builds_an_otlp_provider_for_the_app_lifetime(
    settings: Settings, tmp_path: Path
) -> None:
    enabled = settings.model_copy(update={"tracing_enabled": True})
    with gateway_client_factory(enabled, tmp_path) as client:
        tracer = client.app.state.tracer  # type: ignore[attr-defined]
    assert not isinstance(tracer, trace.NoOpTracer)


def test_resource_defaults_service_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    provider = build_tracer_provider()
    try:
        attributes = provider.resource.attributes
        assert attributes[SERVICE_NAME] == DEFAULT_SERVICE_NAME
        assert attributes[SERVICE_VERSION] == __version__
    finally:
        provider.shutdown()


def test_resource_honours_otel_service_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "gateway-eu")
    provider = build_tracer_provider()
    try:
        assert provider.resource.attributes[SERVICE_NAME] == "gateway-eu"
    finally:
        provider.shutdown()
