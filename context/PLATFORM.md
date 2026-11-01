# Platform Context

**Stack**: Python 3.12+ (developed on 3.14) · `uv` · `uv_build` · `src/` layout · FastAPI · SQLite + Redis. Check existing libraries before adding new dependencies.

## Core Technology Stack

- **Languages**: Python 3.12+ (uses `StrEnum`, `Self`, `|` union types, `match`, `tomllib`).
- **Tooling**: `uv` (env, run and build through the `uv_build` backend), `ruff` (lint + format, including the `S` security rules), `pyright` strict (type-check), `gitleaks` (secrets), `pre-commit`.
- **API surface**: `fastapi` + `uvicorn[standard]` for the HTTP server; pure-ASGI middleware (not `BaseHTTPMiddleware`) so streaming SSE bodies pass through unbuffered.
- **Data validation**: `pydantic` v2 + `pydantic-settings`: runtime validation across every API boundary; `frozen=True, extra="forbid"` on admin schemas.
- **HTTP client**: `httpx` (async) for the Ollama adapter; `respx` for hermetic test interception.
- **State**: `aiosqlite` for the cost ledger + circuit-breaker state (one file, schema-versioned). `redis.asyncio` for the token bucket (atomic Lua script through `register_script`, which reloads it after a Redis restart).
- **Provider**: Custom Ollama HTTP adapter: NDJSON streaming, token counts from Ollama's `prompt_eval_count` / `eval_count` (chat and `/api/embed`), error normalisation to `ProviderTimeoutError` / `ProviderUnavailableError` / `ProviderError`. See [`docs/architecture.md`](../docs/architecture.md) "Why these choices" for why-not-`litellm`.
- **Logging**: `structlog` with JSON renderer; `bind()` for request-scoped context (request_id, team_id).
- **Observability**: `prometheus_client` for metrics; pre-provisioned Grafana dashboards; OpenTelemetry SDK + OTLP/HTTP exporter for one GenAI span per LLM call (off by default, `GATEWAY_TRACING_ENABLED`).
- **Cache search**: `numpy` for the in-process semantic cache's cosine search.
- **Testing**: `pytest` + `pytest-asyncio` (auto mode) + `pytest-cov`; `respx` for httpx mocking; `fakeredis[lua]` for the token bucket.

## Architecture & Patterns

- **Data flow**: HTTP request → `MetricsMiddleware` (outermost) → `AuthMiddleware` → `RateLimitMiddleware` → route handler → `BudgetEnforcer.check` → `classify()` → `CompletionCache.lookup` (chat only; a hit returns here) → `RoutingPolicy.plan` → `ProviderRouter.chat/embed/stream_chat` (with circuit breaker, inside a GenAI span) → `BudgetTracker.record` → `Metrics.record_*` → JSON response with `X-Gateway-*` headers.
- **Design patterns**: Protocols (PEP 544) over ABCs: `Provider` is structural; `runtime_checkable` so tests can assert conformance. Frozen `@dataclass(frozen=True, slots=True)` for in-memory event/result types; pydantic `BaseModel` for API boundaries.
- **State management**: Immutable models across boundaries. SQLite as the single source of truth for billing; failure-rate window is in-memory and resets on restart while *circuit state* survives.
- **System architecture**: Single FastAPI process behind any reverse proxy; sidecars are Ollama, Redis, Prometheus, Grafana, all wired by `docker-compose.yml`. No multi-process / multi-worker support today (Redis bucket would still be correct; SQLite is single-writer).
- **Concurrency**: Pure asyncio. The token-bucket Lua script is atomic at the Redis level: the per-test concurrency assertion fires 40 simultaneous acquires at a bucket of capacity 20 and verifies exactly 20 ± 1 were allowed.

## Infrastructure & Deployment

- **Containerization**: Multi-stage `Dockerfile` (Python 3.14 slim) → non-root `gateway` user → `uvicorn` entrypoint. `docker-compose.yml` provisions Ollama (with one-shot `ollama-init` for model pulls), Redis, Prometheus, Grafana, and the gateway.
- **Environment**: Runs locally or on any single-host runner. All settings are env-driven via `pydantic-settings` with the `GATEWAY_*` prefix; host and port are uvicorn flags.
- **Secret management**: Env vars only. No real API keys today: Ollama runs locally. Team API keys live in `config/teams.yaml` (gitignored; `config/teams.example.yaml` is the template).
- **CI/CD pipelines**: GitHub Actions (`ci.yml`): pre-commit (ruff, pyright, gitleaks), pytest on Python 3.14, and a wheel build started in a clean venv. Coverage gate at 90%. Actions pinned by commit SHA; Dependabot updates them monthly.
- **Observability**: Prometheus scrapes `/metrics` every 5s; Grafana provisions 3 dashboards on first boot. `structlog` JSON to stdout for log aggregation. OpenTelemetry GenAI spans exported over OTLP/HTTP when `GATEWAY_TRACING_ENABLED=true`, configured by the standard `OTEL_*` variables; prompts and answers only with `GATEWAY_TRACING_CAPTURE_MESSAGE_CONTENT=true`.

## Third-Party Integrations

- **Routing**: optional llm-cost-autopilot sidecar (`docker-compose.autopilot.yml`) over its route API v2, only for `model: "auto"`; isolated in `routing/autopilot.py`.
- **Providers**: Ollama (only). The `Provider` Protocol is the seam: adding a real provider is an adapter + a `ModelConfig` entry in the registry.
- **Storage**: SQLite (`aiosqlite`) with explicit schema versioning. Two tables: `cost_ledger` (append-only events) and `circuit_state` (per-(provider, model) breaker state). `schema_meta` tracks `CURRENT_SCHEMA_VERSION`.
- **Cache / rate limit**: Redis (`redis.asyncio` + `hiredis`). Token-bucket Lua script is the only Redis touch: kept minimal so an in-process implementation could swap in for the single-host case.
- **Metrics**: Prometheus exposition via `prometheus_client`. Metric names and label vocabulary are defined in `observability/metrics.py`; recording helpers are the only public surface.
