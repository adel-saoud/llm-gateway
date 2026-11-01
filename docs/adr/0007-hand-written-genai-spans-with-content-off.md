---
status: accepted
date: 2026-10-31
---

# Hand-written GenAI spans, message content off by default

## Context and Problem Statement

[trace-to-golden](https://github.com/adel-saoud/trace-to-golden) turns gateway traffic into
eval cases, so it needs one span per upstream LLM call with the model asked for, the model
that answered, token counts and finish reason, linked to the caller's trace. Prompts and
answers are user data. How should the gateway produce those spans, and what should they
carry?

## Considered Options

* FastAPI and httpx auto-instrumentation
* Hand-written client spans following the OpenTelemetry GenAI semantic conventions

## Decision Outcome

Chosen option: "Hand-written client spans following the OpenTelemetry GenAI semantic
conventions" (v1.37.0), one `chat {model}` or `embeddings {model}` span per upstream call,
child of an incoming W3C `traceparent` ([`docs/tracing.md`](../tracing.md)). Prompts and
answers are recorded only with `GATEWAY_TRACING_CAPTURE_MESSAGE_CONTENT=true`; the
exception message is never recorded, because upstream errors can echo the prompt. The
attribute names are pinned as literals in one module,
[`observability/tracing.py`](../../src/llm_gateway/observability/tracing.py), because the
semconv constants live in a private `_incubating` package and the conventions are in
Development status ([`docs/architecture.md`](../architecture.md#why-these-choices)).

### Consequences

* Good, because `gen_ai.request.model` (can be `auto`) and `gen_ai.response.model` (after
  routing and fallback) are both on the span, which auto-instrumentation of the HTTP call
  cannot know.
* Good, because no attribute carries user data by default: no team id, `user` field,
  client address or URL.
* Bad, because there are no HTTP server spans and no spans for the cache lookup or the
  routing call, and a cache hit emits no span, so the eval pipeline does not see served-from-
  cache traffic ([`docs/tracing.md`](../tracing.md#not-covered)).
* Bad, because a convention rename is a manual edit in `observability/tracing.py`.
* Neutral: `opentelemetry-instrumentation-fastapi` was removed as an unused dependency
  ([`CHANGELOG.md`](../../CHANGELOG.md) 0.2.0, Removed).
