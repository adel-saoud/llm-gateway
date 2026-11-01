---
status: accepted
date: 2026-10-31
---

# In-process response cache, exact matching on, semantic matching off

## Context and Problem Statement

A repeated question should not pay for a second model call. Version 0.1.0 listed semantic
caching as out of scope; 0.2.0 added a response cache inside the gateway, after the budget
check and RAG tagging and before routing, so a hit also skips the routing call
([`docs/cache.md`](../cache.md), [`CHANGELOG.md`](../../CHANGELOG.md) 0.2.0). Where should
the cache live, how should it match, and what should it never do?

## Decision Drivers

* A cached answer must never reach a different team or end user.
* A wrong answer is worse than a missed hit: "cancel order 123" must never answer "cancel
  order 124".
* A cache problem may cost a hit, never a request.
* The README cost numbers must stay true: a hit costs nothing.

## Considered Options

* Exact matching only
* Exact plus semantic matching on by default
* Exact matching on, semantic matching opt-in at a calibrated threshold
* A shared store (Redis, plain or with vector search) instead of in-process

## Decision Outcome

Chosen option: "Exact matching on, semantic matching opt-in at a calibrated threshold",
in process with numpy search. Entries are partitioned by team, OpenAI `user`, requested
model and system prompt; semantic matching is single-turn only and must carry the same
numbers, ids, dates and quoted strings (the literal guard); an embedding failure serves the
request normally; a hit returns `X-Gateway-Cost-USD: 0.000000` and is not written to the
ledger ([`docs/cache.md`](../cache.md)). The threshold, 0.92, is the lowest with no wrong
answer on the 48 labelled pairs in [`bench/cache_calibration.jsonl`](../../bench/cache_calibration.jsonl).
It serves too few paraphrases to turn on for everyone, so semantic matching ships off.
In process because one replica with bounded partitions
(`cache_max_entries_per_partition`, default 1,000, [`config.py`](../../src/llm_gateway/config.py))
needs no network hop ([`docs/architecture.md`](../architecture.md#why-these-choices)).

### Consequences

* Good, because at 0.92 the guarded sweep has 0 wrong answers out of 28 near-misses
  (`results/manifest.json`: `metrics.cache_calibration.sweep["0.92"].false_hits_guarded` 0).
* Bad, because only 30% of paraphrases are served at that threshold
  (`sweep["0.92"].hit_rate_guarded` 0.3), on a small hand-written set.
* Good, because the literal guard does real work: without it, 0.92 lets 4 near-misses
  through (`false_hits_raw` 4); other numbers score above paraphrases (median 0.932 against
  0.858, `median_cosine_by_category`).
* Neutral: the embedding instruction is put on both prompts, which the Qwen3-Embedding
  model card does not cover; that symmetric use is UNVERIFIED by the model card and was
  chosen by calibration ([`docs/cache.md`](../cache.md#why-semantic-matching-is-off-by-default)).
* Bad, because each replica has its own cache and a restart empties it (README Limits).
* Bad, because a cache hit emits no span, so traces under-count served traffic
  ([0007](0007-hand-written-genai-spans-with-content-off.md)).
* Neutral: the semantic miss costs 4.24 ms of gateway time p50 against 2.14 ms with the
  cache off (`metrics.overhead.sequential.gateway_semantic_cache_miss` and
  `gateway_cache_off`, `gateway_self_ms_p50`).

## Pros and Cons of the Options

### Exact plus semantic matching on by default

* Good, because more repeated questions are served free.
* Bad, because with the previous embedder (`nomic-embed-text`, the first one tried) near-misses scored
  higher than paraphrases and no threshold separated them ([`docs/cache.md`](../cache.md)).

### A shared store (Redis)

* Good, because replicas would share hits and survive restarts.
* Bad, because every lookup pays a network hop, and the vector search variant is one more
  service to run for a single-node deployment.
