---
status: accepted
date: 2026-10-31
---

# Thinking off by default, billed when a client turns it on

## Context and Problem Statement

The default models (Qwen3.5) think before answering unless told not to. In a local test,
"Say hi." produced 3,684 output tokens with thinking and 12 without
([`docs/configuration.md`](../configuration.md#thinking)). Ollama counts thinking tokens in
`eval_count`, which the gateway bills ([`tests/unit/test_ollama_provider.py`](../../tests/unit/test_ollama_provider.py)).
Should the gateway pass the model's default through, or decide for the client?

## Considered Options

* Pass the model's default through (thinking on)
* Thinking off by default, opt in with OpenAI's `reasoning_effort`
* Thinking off and not offered

## Decision Outcome

Chosen option: "Thinking off by default, opt in with OpenAI's `reasoning_effort`". The
gateway sends `think: false` to Ollama unless `reasoning_effort` is present and not
`"none"` ([`api/schemas.py`](../../src/llm_gateway/api/schemas.py)). With thinking on, the
thinking tokens are completion tokens in `usage`, the cost header, the ledger and the team
budget, and the request bypasses the response cache
([`docs/configuration.md`](../configuration.md#thinking)).

### Consequences

* Good, because a client that did not ask for hidden tokens does not pay for them.
* Good, because the opt-in uses a field OpenAI clients already send.
* Good, because the bill stays honest: thinking is counted, not hidden.
* Bad, because Ollama takes on or off for Qwen, so the level (`low`, `high`, ...) is not
  forwarded; every level is the same.
* Bad, because `max_tokens` caps thinking and answer together, so a low cap can end with an
  empty answer and `finish_reason: "length"`.
* Neutral: thinking answers are never cached (cache bypass reason `thinking`,
  [`cache/completion_cache.py`](../../src/llm_gateway/cache/completion_cache.py)).
