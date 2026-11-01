# Response cache

A repeated question should not pay for a second model call. The cache sits after the
budget check and RAG tagging and before routing, so a hit also skips the routing call.

## How a lookup works

1. **Exact match.** sha256 over every message. A byte-identical request, any number of
   turns, is served without an embedding call.
2. **Semantic match** (only with `GATEWAY_CACHE_SEMANTIC_ENABLED=true`). The single user
   message is embedded with the embeddings model (`GATEWAY_MODEL_EMBEDDINGS`) and compared by cosine similarity with
   stored single-turn prompts. A match must reach `GATEWAY_CACHE_SIMILARITY_THRESHOLD`
   **and** carry the same literals: numbers, ids, dates and quoted strings. "Cancel
   order 123" never answers "cancel order 124", even at similarity 1.0. Multi-turn
   requests never match semantically, because "yes, do it" after two different
   conversations embeds the same.

**Isolation.** Entries are partitioned by team, the OpenAI `user` field, the requested
model and the system prompt. One end user's cached answer is never served to another
user of the same team.

**When it is skipped.** The cache is off unless `GATEWAY_CACHE_ENABLED=true` **and** the
team has `semantic_cache: true`. A request also bypasses it when it sends
`Cache-Control: no-cache` (or `no-store`), streams, asks for a `temperature` above
`GATEWAY_CACHE_MAX_TEMPERATURE` (default 0), or turns thinking on (`reasoning_effort`). If the embedding call fails, the request is
served normally: a cache problem can cost a hit, never a request.

**Cost.** A hit returns the original usage figures with `X-Gateway-Cost-USD: 0.000000`
and is not written to the cost ledger. `llm_gateway_semantic_cache_cost_saved_usd_total`
counts what the hits would have cost; `llm_gateway_semantic_cache_lookup_cost_usd_total`
counts what the lookups spent on embeddings. The Operations dashboard shows the hit rate
and the net saving.

## Why semantic matching is off by default

`scripts/calibrate_cache.py` embeds 48 labelled prompt pairs
([`bench/cache_calibration.jsonl`](../bench/cache_calibration.jsonl)): 20 paraphrases that
should hit, and 28 near-misses that should not (other order number, other city,
opposite intent, unrelated). It uses the gateway's own settings: `qwen3-embedding:0.6b`
with the Qwen3-Embedding duplicate-question instruction on both prompts
(`GATEWAY_CACHE_EMBEDDING_PREFIX`).

| Pair type | Pairs | Median cosine |
|:--|--:|--:|
| Paraphrase (should hit) | 20 | 0.858 |
| Other number or id | 8 | 0.932 |
| Other name (city, colour, team) | 8 | 0.830 |
| Opposite or different intent | 8 | 0.829 |
| Unrelated | 4 | 0.347 |

Names and intent flips now score below paraphrases. Numbers still score above them, and
the literal guard removes those:

| Threshold | Paraphrases served | Wrong answers, no guard | Wrong answers, with literal guard |
|--:|--:|--:|--:|
| 0.85 | 55% | 11 of 28 | 5 of 28 |
| 0.88 | 45% | 8 of 28 | 2 of 28 |
| 0.90 | 40% | 7 of 28 | 1 of 28 |
| 0.92 | 30% | 4 of 28 | 0 of 28 |
| 0.95 | 20% | 3 of 28 | 0 of 28 |

The default threshold, 0.92, is the lowest with no wrong answer on this set, and it
serves 30% of paraphrases. That is too few, on too small a set, to turn on for everyone:
the gateway ships exact matching on, semantic matching off, and this script for anyone
who wants to test their own traffic. The set is hand-written; treat it as a smoke test of
the method, not a benchmark of the model. Full output:
[`results/raw/calibration.json`](../results/raw/calibration.json), summary in
[`results/manifest.json`](../results/manifest.json).

**Instruction.** The Qwen3-Embedding model card puts the task instruction
(`Instruct: <task>\nQuery:`) on queries and not on documents. A cache lookup compares a
query with earlier queries, which the card does not cover, so the instruction goes on both
sides. That symmetric use is UNVERIFIED by the model card: it was chosen by calibration.
Three variants on the same pairs:

| Prefix on both prompts | Paraphrase median | Best threshold with 0 wrong answers (guarded) |
|:--|--:|:--|
| none | 0.810 | none up to 0.97 (1 wrong answer) |
| `Retrieve semantically similar text` | 0.950 | 0.97, serving 20% |
| `Given a question, retrieve questions that are semantically equivalent to the given question` (default) | 0.858 | 0.92, serving 30% |

**Before (first embedder tried, `nomic-embed-text` with `clustering:`).** Near-misses scored *higher*
than paraphrases (medians 0.936 to 0.985 against 0.907) and no threshold separated them,
even with the guard. The move to Qwen3-Embedding is what made a zero-error threshold exist.

```bash
uv run python scripts/calibrate_cache.py   # needs a host Ollama with the configured embedding model
```
