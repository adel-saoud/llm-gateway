---
status: accepted
date: 2026-10-31
---

# Own Ollama adapter instead of LiteLLM

## Context and Problem Statement

The gateway's job is to attribute cost per RAG stage, which needs exact token counts per
call, unbuffered streaming and one error hierarchy the circuit breaker and fallback chain
can act on. It needs a provider layer. Should that layer be a library that already speaks
to many providers, or an adapter written for the one provider the project runs?

## Considered Options

* LiteLLM as the provider layer
* A custom async adapter for Ollama behind a `Provider` Protocol

## Decision Outcome

Chosen option: "A custom async adapter for Ollama behind a `Provider` Protocol", because
it keeps the HTTP contract visible and testable: NDJSON streaming, token counts from
Ollama's `prompt_eval_count` / `eval_count`, and errors normalised to `ProviderError` /
`ProviderTimeoutError` / `ProviderUnavailableError`
([`docs/architecture.md`](../architecture.md#why-these-choices)). The `Provider` Protocol
is the seam where LiteLLM or another backend can come in later as one adapter plus
registry entries ([`docs/extending.md`](../extending.md)).

### Consequences

* Good, because token counting is exact and tested: a thinking run's 3,684 `eval_count`
  lands in `usage.completion_tokens` ([`tests/unit/test_ollama_provider.py`](../../tests/unit/test_ollama_provider.py)).
* Good, because the wire contract the breaker and fallback depend on is in this repo, not
  behind a dependency upgrade.
* Bad, because Ollama is the only provider (README Limits). Each new backend is code to
  write and test.
* Bad, because version requirements on Ollama are the gateway's to track: `/api/embed`
  with `prompt_eval_count` needs Ollama 0.3.4, and the adapter names that version in its
  error ([`CHANGELOG.md`](../../CHANGELOG.md), Unreleased, Added).

## Pros and Cons of the Options

### LiteLLM

* Good, because it already speaks to many providers and is a common production choice
  ([`docs/architecture.md`](../architecture.md#compared-with-litellm)).
* Bad, because the contract the cost ledger depends on (token fields, stream framing,
  error types) sits behind its abstraction.
