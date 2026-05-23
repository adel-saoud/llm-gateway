# Platform Context

**Stack**: Python 3.11+ · `uv` · `hatchling` · `src/` layout · FastAPI · SQLite + Redis. Check existing libraries before adding new dependencies.

## Core Technology Stack

- **Languages**: Python 3.11+ (uses `StrEnum`, `Self`, `|` union types, `match`, `tomllib`).
- **Tooling**: `uv` (env + run), `hatchling` (build), `ruff` (lint + format), `pyright` strict (type-check), `pre-commit`.
- **API surface**: `fastapi` + `uvicorn[standard]` for the HTTP server; pure-ASGI middleware (not `BaseHTTPMiddleware`) so streaming SSE bodies pass through unbuffered.
- **Data validation**: `pydantic` v2 + `pydantic-settings` — runtime validation across every API boundary; `frozen=True, extra="forbid"` on admin schemas.
- **HTTP client**: `httpx` (async) for the Ollama adapter; `respx` for hermetic test interception.
- **State**: `aiosqlite` for the cost ledger + circuit-breaker state (one file, schema-versioned). `redis.asyncio` for the token bucket (atomic `EVALSHA`).
- **Provider**: Custom Ollama HTTP adapter — NDJSON streaming, error normalisation to `ProviderTimeoutError` / `ProviderUnavailableError` / `ProviderError`. See [`context/decisions/`](decisions/) for why-not-`litellm`.
- **Logging**: `structlog` with JSON renderer; `bind()` for request-scoped context (request_id, team_id).
- **Observability**: `prometheus_client` for metrics; pre-provisioned Grafana dashboards; OpenTelemetry SDK wired but not yet exporting spans by default.
- **Testing**: `pytest` + `pytest-asyncio` (auto mode) + `pytest-cov`; `respx` for httpx mocking; `fakeredis[lua]` for the token bucket.

## Architecture & Patterns

- **Data flow**: HTTP request → `MetricsMiddleware` (outermost) → `AuthMiddleware` → `RateLimitMiddleware` → route handler → `BudgetEnforcer.check` → `classify()` → `ProviderRouter.chat/embed/stream_chat` (with circuit breaker) → `BudgetTracker.record` → `Metrics.record_*` → JSON response with `X-Gateway-*` headers.
- **Design patterns**: Protocols (PEP 544) over ABCs — `Provider` is structural; `runtime_checkable` so tests can assert conformance. Frozen `@dataclass(frozen=True, slots=True)` for in-memory event/result types; pydantic `BaseModel` for API boundaries.
- **State management**: Immutable models across boundaries. SQLite as the single source of truth for billing; failure-rate window is in-memory and resets on restart while *circuit state* survives.
- **System architecture**: Single FastAPI process behind any reverse proxy; sidecars are Ollama, Redis, Prometheus, Grafana — all wired by `docker-compose.yml`. No multi-process / multi-worker support today (Redis bucket would still be correct; SQLite is single-writer).
- **Concurrency**: Pure asyncio. The token-bucket Lua script is atomic at the Redis level — the per-test concurrency assertion fires 40 simultaneous acquires at a bucket of capacity 20 and verifies exactly 20 ± 1 were allowed.

## Infrastructure & Deployment

- **Containerization**: Multi-stage `Dockerfile` (Python 3.12 slim) → non-root `gateway` user → `uvicorn` entrypoint. `docker-compose.yml` provisions Ollama (with one-shot `ollama-init` for model pulls), Redis, Prometheus, Grafana, and the gateway.
- **Environment**: Runs locally or on any single-host runner. All settings are env-driven via `pydantic-settings` with the `GATEWAY_*` prefix.
- **Secret management**: Env vars only. No real API keys today — Ollama runs locally. Team API keys live in `config/teams.yaml` (gitignored; `config/teams.example.yaml` is the template).
- **CI/CD pipelines**: GitHub Actions — `ruff check → ruff format --check → pyright → pytest`. Coverage gate at 85%.
- **Observability**: Prometheus scrapes `/metrics` every 5s; Grafana provisions 3 dashboards on first boot. `structlog` JSON to stdout for log aggregation. OpenTelemetry traces wired but disabled by default (`GATEWAY_TRACING_ENABLED=false`).

## Third-Party Integrations

- **Providers**: Ollama (only). The `Provider` Protocol is the seam — adding a real provider is an adapter + a `ModelConfig` entry in the registry.
- **Storage**: SQLite (`aiosqlite`) with explicit schema versioning. Two tables: `cost_ledger` (append-only events) and `circuit_state` (per-(provider, model) breaker state). `schema_meta` tracks `CURRENT_SCHEMA_VERSION`.
- **Cache / rate limit**: Redis (`redis.asyncio` + `hiredis`). Token-bucket Lua script is the only Redis touch — kept minimal so an in-process implementation could swap in for the single-host case.
- **Metrics**: Prometheus exposition via `prometheus_client`. Metric names and label vocabulary are defined in `observability/metrics.py`; recording helpers are the only public surface.
