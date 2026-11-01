# Tracing

With `GATEWAY_TRACING_ENABLED=true` the gateway exports one OpenTelemetry span per
upstream LLM call over OTLP/HTTP. Spans follow the
[GenAI semantic conventions v1.37.0](https://github.com/open-telemetry/semantic-conventions/blob/v1.37.0/docs/gen-ai/gen-ai-spans.md),
so [trace-to-golden](https://github.com/adel-saoud/trace-to-golden) can turn gateway
traffic into eval cases.

| Span | When | Attributes |
|:--|:--|:--|
| `chat {model}` (kind `CLIENT`) | `/v1/chat/completions`, streaming or not, cache miss or bypass | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.finish_reasons`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` |
| `embeddings {model}` (kind `CLIENT`) | `/v1/embeddings` | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens` |

- `gen_ai.request.model` is the model the client asked for, so it can be `auto`;
  `gen_ai.response.model` is the model that answered after routing and fallback.
- `gen_ai.response.finish_reasons` is the provider's reason (`stop`, `length`, ...), for
  streams too.
- A streaming span opens when the response body starts and closes when the stream ends,
  including on a client disconnect.
- On failure the span gets status `ERROR` and `error.type` set to the exception class
  (`AllProvidersUnavailableError`, `ProviderError`, ...). The exception message is not
  recorded: upstream errors can echo the prompt.
- A W3C `traceparent` request header makes the gateway span a child of the caller's
  trace, so an agent and its LLM calls share one trace.
- Prompts and answers are recorded only with
  `GATEWAY_TRACING_CAPTURE_MESSAGE_CONTENT=true`, as JSON strings in
  `gen_ai.input.messages` and `gen_ai.output.messages`
  (`[{"role": "user", "parts": [{"type": "text", "content": "..."}]}]`). They are not
  redacted; turn this on only where the trace backend may hold user data. No other
  attribute carries user data (no team id, `user` field, client address or URL).

## Exporter settings

The exporter reads the standard OpenTelemetry variables, which must be set in the
process environment (the `.env` file only feeds `GATEWAY_*` settings):

```bash
GATEWAY_TRACING_ENABLED=true \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
OTEL_SERVICE_NAME=llm-gateway \
uv run uvicorn llm_gateway.main:app
```

`OTEL_SERVICE_NAME` defaults to `llm-gateway`. With tracing off, handlers use a no-op
tracer.

## Not covered

- No HTTP server spans, and no spans for the cache lookup or the autopilot routing call.
  A cache hit makes no upstream call and emits no span.
- `gen_ai.conversation.id` is never set: the OpenAI request has no conversation field.
- The GenAI conventions are in Development status. Attribute names follow v1.37.0 and
  are pinned in one module, [`observability/tracing.py`](../src/llm_gateway/observability/tracing.py).
