"""OpenTelemetry GenAI client spans for the gateway's LLM calls.

One ``CLIENT`` span per upstream call (chat completion or embedding), named and
attributed per the OTel GenAI semantic conventions v1.37.0
(``docs/gen-ai/gen-ai-spans.md``). Downstream, trace-to-golden turns these
spans into eval cases, so the attribute names are its contract.

The GenAI conventions are still in Development status and their Python
constants live in the private ``opentelemetry.semconv._incubating`` package, so
the names are pinned here as literals: one module to touch when they change.

Privacy: message content is recorded only when
``GATEWAY_TRACING_CAPTURE_MESSAGE_CONTENT`` is on, and exception text is never
recorded (upstream errors can echo the prompt). No other attribute carries
user data: no team id, ``user`` field, client address or URL.

The tracer lives on ``app.state`` rather than in the global provider so each
app instance (and each test) owns its own pipeline. Parent context comes from
the request's W3C ``traceparent`` header, so a caller's trace (for example an
agent's ``invoke_agent`` span) contains the gateway span as a child.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Final, Self

from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

from llm_gateway import __version__
from llm_gateway.providers.base import ChatMessage

GEN_AI_OPERATION_NAME: Final = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME: Final = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL: Final = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL: Final = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS: Final = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"
GEN_AI_INPUT_MESSAGES: Final = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES: Final = "gen_ai.output.messages"
ERROR_TYPE: Final = "error.type"

DEFAULT_SERVICE_NAME: Final = "llm-gateway"
TRACER_NAME: Final = "llm_gateway"


class GenAiOperationName(StrEnum):
    """Well-known ``gen_ai.operation.name`` values the gateway emits."""

    CHAT = "chat"
    EMBEDDINGS = "embeddings"


def build_tracer_provider() -> TracerProvider:
    """Tracer provider exporting OTLP/HTTP in batches.

    Endpoint, headers, timeout and protocol options come from the standard
    ``OTEL_EXPORTER_OTLP_*`` environment variables, read by the exporter itself;
    ``OTEL_SERVICE_NAME`` / ``OTEL_RESOURCE_ATTRIBUTES`` by the resource. They
    must be real environment variables: the gateway's ``.env`` loader only
    feeds ``GATEWAY_*`` settings.
    """
    resource = Resource.create({SERVICE_VERSION: __version__})
    if str(resource.attributes.get(SERVICE_NAME, "")).startswith("unknown_service"):
        resource = resource.merge(Resource({SERVICE_NAME: DEFAULT_SERVICE_NAME}))
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider


def get_tracer(provider: TracerProvider | None) -> Tracer:
    """The gateway tracer, or a no-op tracer when tracing is off."""
    if provider is None:
        return trace.NoOpTracer()
    return provider.get_tracer(TRACER_NAME, __version__)


def _text_message(role: str, content: str) -> dict[str, object]:
    return {"role": role, "parts": [{"type": "text", "content": content}]}


def _input_messages_json(messages: Sequence[ChatMessage]) -> str:
    """``gen_ai.input.messages`` as a JSON string (the SDK has no structured attributes)."""
    return json.dumps([_text_message(m.role, m.content) for m in messages], ensure_ascii=False)


def _output_messages_json(content: str, finish_reason: str) -> str:
    """``gen_ai.output.messages`` as a JSON string; ``finish_reason`` is required by the schema."""
    message = _text_message("assistant", content)
    message["finish_reason"] = finish_reason
    return json.dumps([message], ensure_ascii=False)


@dataclass(slots=True)
class GenAiSpan:
    """An open GenAI client span plus the content-capture decision for it.

    Use it as a context manager around the upstream call: on exit it records
    any ``Exception`` through :meth:`record_error` and ends the span. A
    ``GeneratorExit`` or ``CancelledError`` (client gone) ends the span without
    marking it failed, because the provider did not fail.
    """

    span: Span
    capture_message_content: bool

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if isinstance(exc_value, Exception):
            self.record_error(exc_value)
        self.span.end()

    def record_chat_response(
        self,
        *,
        response_model: str,
        input_tokens: int,
        output_tokens: int,
        content: str,
        finish_reason: str,
    ) -> None:
        """Set the response attributes; ``content`` is recorded only when capture is on."""
        self.span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)
        self.span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
        self.span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        self.span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        if self.capture_message_content:
            self.span.set_attribute(
                GEN_AI_OUTPUT_MESSAGES, _output_messages_json(content, finish_reason)
            )

    def record_embeddings_response(self, *, response_model: str, input_tokens: int) -> None:
        """Set the response attributes; embedding input and vectors are never recorded."""
        self.span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)
        self.span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)

    def record_error(self, exc: BaseException) -> None:
        """Mark the span failed with a low-cardinality ``error.type`` and no message text."""
        error_type = type(exc).__name__
        self.span.set_attribute(ERROR_TYPE, error_type)
        self.span.set_status(Status(StatusCode.ERROR, error_type))


def start_genai_span(
    tracer: Tracer,
    *,
    operation: GenAiOperationName,
    provider_name: str,
    request_model: str,
    request_headers: Mapping[str, str],
    input_messages: Sequence[ChatMessage] = (),
    capture_message_content: bool = False,
) -> GenAiSpan:
    """Start a GenAI client span as a child of the caller's ``traceparent``, if any.

    The span is not made current and is ended by leaving the returned
    :class:`GenAiSpan` context. Starting it with plain ``start_span`` lets the
    streaming handler open the context inside its generator, so the span covers
    the whole stream. ``record_exception=False`` keeps exception text off the
    span even if it is ever used through the SDK's own context manager.
    """
    span = tracer.start_span(
        f"{operation.value} {request_model}",
        context=propagate.extract(request_headers),
        kind=SpanKind.CLIENT,
        attributes={
            GEN_AI_OPERATION_NAME: operation.value,
            GEN_AI_PROVIDER_NAME: provider_name,
            GEN_AI_REQUEST_MODEL: request_model,
        },
        record_exception=False,
    )
    if capture_message_content and input_messages:
        span.set_attribute(GEN_AI_INPUT_MESSAGES, _input_messages_json(input_messages))
    return GenAiSpan(span=span, capture_message_content=capture_message_content)
