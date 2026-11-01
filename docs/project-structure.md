# Project structure

```
src/llm_gateway/
├── config.py                 Settings, all env-driven via pydantic-settings
├── main.py                   FastAPI factory · lifespan · middleware
├── _http.py                  ASGI helper: send_json_response for pure middleware
├── api/
│   ├── completions.py        POST /v1/chat/completions (streaming + non-streaming)
│   ├── embeddings.py         POST /v1/embeddings
│   ├── rag.py                GET  /v1/rag/cost-report   ← the differentiator
│   ├── info.py               GET  /v1/models · GET /v1/stats
│   ├── admin.py              PUT  /v1/admin/teams/{id}  (is_admin gated)
│   ├── health.py             GET  /health · GET /health/ready
│   ├── metrics.py            GET  /metrics              (Prometheus scrape)
│   └── schemas.py            Shared Pydantic request/response models
├── auth/
│   ├── models.py             TeamConfig · RateLimitConfig · BudgetConfig
│   ├── resolver.py           YAML-backed, constant-time api_key match
│   └── middleware.py         Pure-ASGI Bearer-token validation
├── ratelimit/
│   ├── token_bucket.py       Redis Lua atomic token bucket
│   └── middleware.py         Pure-ASGI 429 + Retry-After
├── budget/
│   ├── tracker.py            Cost = input_tokens·price + output_tokens·price
│   └── enforcer.py           Warn at 80% · block at 100% · log every event
├── cache/
│   ├── semantic.py           Exact + embedding-similarity store, literal guard, TTL, LRU
│   └── completion_cache.py   Bypass rules, fail-open lookup, store on miss
├── routing/
│   ├── policy.py             RoutingPolicy Protocol · StaticTierPolicy · model="auto"
│   └── autopilot.py          llm-cost-autopilot sidecar adapter (the only file that knows it)
├── rag/
│   ├── models.py             RagOperationType · RagCostEvent · RagCostReport
│   ├── tagger.py             5-priority classifier (pure function)
│   └── cost_attribution.py   Per-team-per-operation aggregator
├── providers/
│   ├── base.py               Provider Protocol + error hierarchy
│   ├── ollama.py             Async HTTP adapter · NDJSON streaming
│   ├── registry.py           ModelConfig · simulated cost tiers
│   └── router.py             Fallback chain + circuit-breaker integration
├── circuit_breaker/
│   ├── state_machine.py      CLOSED / OPEN / HALF_OPEN · SQLite-persisted
│   └── monitor.py            Rolling outcome window per (provider, model)
├── observability/
│   ├── metrics.py            Prometheus collectors (standard + RAG-specific)
│   ├── middleware.py         Auto-instrument every HTTP request
│   └── tracing.py            OTel GenAI spans + OTLP exporter setup
└── storage/
    └── db.py                 SQLite: cost_ledger · circuit_state · schema versioning

scripts/
    demo_rag.py               5-step RAG pipeline simulator
    load_test.py              N requests · prints cost savings % + P50/P95/P99
    demo_resilience.py        Circuit breaker demo: OPEN→HALF_OPEN→CLOSED cycle
    bench_overhead.py         Gateway overhead benchmark (sequential + k6 open loop)
    calibrate_cache.py        Semantic-cache threshold sweep on labelled pairs
    reproduce.py              Reruns all three measurements, writes results/manifest.json
    sync_schemas.py           Checks the pinned llm-cost-autopilot route API schema

bench/
    stub_ollama.py            Instant Ollama-compatible stub for the benchmark
    k6_open_loop.js           k6 constant-arrival-rate scenario
    cache_calibration.jsonl   Labelled paraphrase / near-miss pairs

results/
    manifest.json             Every README number, with environment and sample sizes
    raw/                      Full benchmark and calibration output

grafana/
    provisioning/             Auto-provisioned datasource + dashboard config
    dashboards/               3 pre-built dashboard JSONs

tests/                        Hermetic suite, 90% coverage floor
tests/contracts/              Pinned copy of the autopilot route API schema
context/                      PRODUCT / PLATFORM / PROCESS (read this first)
docs/                         Architecture, decisions, feature pages
bruno/                        Bruno API collection, all endpoints, Local environment
.github/workflows/ci.yml      Pre-commit, pytest, wheel build and smoke test
.github/workflows/bench.yml   Manual benchmark run on a GitHub runner
.github/dependabot.yml        Monthly grouped updates (uv, Actions, pre-commit)
CHANGELOG.md                  Release notes
```

## Dependencies

| Area | Library or tool | Role |
|:--|:--|:--|
| Core | `fastapi` + `uvicorn` | Async API, OpenAPI docs generated |
| | `pydantic` v2 + `pydantic-settings` | Validated models, env-driven config |
| | `httpx` | Async HTTP client for the Ollama adapter and the autopilot sidecar |
| | `redis[hiredis]` | Atomic token bucket via a Lua script |
| | `aiosqlite` | Cost ledger and circuit-breaker state |
| | `numpy` | Cosine similarity for the semantic cache |
| | `structlog` | Structured JSON logs |
| Observability | `prometheus-client` | Standard and RAG-specific metrics, per-instance registry |
| | `opentelemetry-sdk` + OTLP/HTTP exporter | One GenAI span per LLM call, off by default ([tracing](tracing.md)) |
| Dev | `uv` with the `uv_build` backend | Environments, lockfile, build |
| | `ruff` | Lint (including the `S` security rules) and format |
| | `pyright` strict | Type checking |
| | `pytest`, `pytest-asyncio`, `respx`, `fakeredis[lua]` | Hermetic tests: no daemon, no network |
| | `pre-commit`, `gitleaks`, `commitizen` | Local gate, secret scan, Conventional Commits |
