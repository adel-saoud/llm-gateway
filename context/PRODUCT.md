# Product Context

**Objective**: An LLM gateway with RAG-aware cost attribution that breaks down LLM spend by operation type (retrieval, reranking, generation, evaluation) so teams know exactly where their budget goes, not just which model consumed it.

## Objective & Vision

- **Elevator pitch**: A drop-in OpenAI-compatible gateway that classifies every request into a RAG operation type, records per-operation cost, and exposes a real-time per-team-per-operation cost report. Runs end-to-end with Ollama: no API keys.
- **Long-term vision**: The default cost-attribution layer for production RAG systems. The "Stripe Atlas" of LLM observability: pre-wired auth, rate limiting, budget caps, fallback routing, circuit breakers, and dashboards: drop the gateway in front of any model backend and operations are visible from day one.
- **Value proposition**: LiteLLM / PortKey / OpenRouter answer *"how much did I spend per model?"* This gateway answers a different question: *"which stage of my RAG pipeline is consuming my budget?"*

## Target Audience

- **Usage context**: Drop-in HTTP gateway in front of any OpenAI-compatible model backend; pairs with the included Grafana dashboards for live cost attribution.
- **Primary users**: ML / platform engineers operating production RAG systems who need per-operation cost visibility to make architecture decisions (cheaper embeddings? smaller reranker? colder cache?).
- **Secondary users**: Contributors adding new providers, new RAG operation types, new metrics, or new dashboards.

## Features & Scope

- **Key features**: OpenAI-compatible `chat/completions` and `embeddings` (streaming + non-streaming), per-team auth via YAML config, atomic Redis-backed token-bucket rate limiting, per-team daily budget enforcement (warn at 80% / block at 100%), 5-priority RAG operation classifier (header → embedding model → eval keywords → rerank keywords → default), SQLite-persisted three-state circuit breaker (CLOSED / OPEN / HALF_OPEN) per (provider, model), provider router with fallback chain, full Prometheus metric surface (standard + RAG-specific), 3 pre-provisioned Grafana dashboards, live admin endpoint for team config updates, opt-in per-team response cache (exact, plus calibrated semantic matching), `model: "auto"` routing with an optional llm-cost-autopilot sidecar, OpenTelemetry GenAI spans per LLM call, gateway overhead benchmark, demo + load-test + circuit-breaker resilience demo scripts.
- **Out of scope**: Real (paid) provider API keys (Ollama only, by design). Kubernetes / Helm charts. JWT / OAuth (API key is sufficient). Multi-tenant SaaS billing.
- **Future roadmap**: Persistent failure-rate windows (today the window resets on restart; the *state* survives). Per-operation rate limits (today only per-team). Pairwise provider scoring (auto-route to the cheaper provider when quality is comparable). Real-provider adapters (OpenAI, Anthropic) gated behind opt-in env vars.

## Business Logic & Rules

- **Core domain rules**: Every successful chat/embed call is tagged with exactly one `RagOperationType` value and persisted to the cost ledger. The classifier is pure: header > embedding-model signal > evaluation keywords > rerank keywords > default `GENERATION`. Unknown header values fall through rather than producing `UNKNOWN`: a typo doesn't poison the signal.
- **Edge cases**: A team's daily budget is checked before the LLM call (block at 100%) and the cost is recorded *after* the call succeeds. A request that overruns the budget mid-call (the LLM happens to use more tokens than estimated) is *recorded*, not retroactively blocked: accurate accounting beats half-applied enforcement.
- **Compliance / Privacy**: No real provider API keys (Ollama only, by design). All data stays on the local host. SQLite cost ledger is the single source of truth for billing: no third-party telemetry, no shipped logs.

## Success Criteria & Constraints

- **Usage limits**: $0. The default config runs entirely on local Ollama. Simulated cost tiers (qwen3.5:4b at $0.003/1k input · qwen3.5:0.8b at $0.0001/1k input · qwen3-embedding:0.6b at $0.0001/1k tokens) drive the routing logic and the dashboards; the dollar amounts are illustrative, the % savings are real.
- **Success metrics**: `uv run pytest` passes in under 5s (full suite hermetic: no Ollama daemon required). `docker-compose up` cold-starts the full stack in under 60s on a fresh machine. Gateway P50 overhead vs raw Ollama call < 10ms, measured by `scripts/bench_overhead.py` ([`docs/benchmark.md`](../docs/benchmark.md)).
- **Absolute constraints**: No paid-model defaults. No hardcoded model IDs anywhere outside `config.py` / `registry.py`. No `utils.py`. No bare `except Exception`. No `cast(Any, ...)` without a specific third-party typing gap to justify it.
