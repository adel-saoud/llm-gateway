# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Breaking changes below, so the next release is 0.2.0 (minor bump while the
major version is 0).

### Added
- Startup check on `GATEWAY_*` names: an unknown variable, in the environment or
  in `.env`, stops the gateway with the closest valid name (a leftover
  `GATEWAY_CB_MIN_REQUESTS` points to `GATEWAY_CIRCUIT_BREAKER_MIN_REQUESTS`).
  Before, pydantic-settings ignored it and the default applied silently.
- Ollama 0.3.4 is the stated minimum (`/api/embed` with `prompt_eval_count`).
  A 404 with no such route, or an embed answer without the token count, fails
  with an error naming that version. README has a Requirements table.
- `scripts/sync_schemas.py` and the `sync-schemas` pre-commit hook and CI step:
  the pinned copy of llm-cost-autopilot's route API schema must match the
  pinned release. Contract tests validate the router client's requests and
  accepted responses against it (`jsonschema` dev dependency).
- `python -m llm_gateway.pull_models` pulls the models the settings name; the
  compose `ollama-init` service runs it, and both compose services read `.env`.
- `reasoning_effort` on chat completions. Thinking is off by default (`think:
  false` to Ollama) and on for any value other than `"none"`; thinking tokens are
  billed as completion tokens, and thinking requests bypass the response cache.
- Team `allowed_models` entries may be tiers (`high_quality`, `budget`,
  `embeddings`), resolved through the registry, so a model change needs no
  allowlist edit. The example teams and the Bruno collection use tiers, and
  Bruno reads the model ids from `GET /v1/models`.
- `scripts/reproduce.py` reruns the benchmark, cache calibration and RAG demo
  and writes `results/manifest.json` (environment, model digests, dataset hash,
  each README number with its sample size). Raw output moved from
  `bench/results/` to `results/raw/`.

- Response cache for chat completions (`llm_gateway.cache`). Exact matching on
  every message; optional embedding-similarity matching for single-turn
  prompts behind `GATEWAY_CACHE_SEMANTIC_ENABLED`, guarded by a literal check
  (numbers, ids, dates and quoted strings must match). Partitioned by team,
  OpenAI `user`, requested model and system prompt; TTL and LRU bounded.
  Opt-in per team (`semantic_cache: true`) and globally
  (`GATEWAY_CACHE_ENABLED`); `Cache-Control: no-cache` bypasses it per request.
  The semantic threshold defaults to 0.92 with the Qwen3-Embedding
  duplicate-question prefix, from the calibration sweep below.
- Cache metrics: `llm_gateway_semantic_cache_requests_total`,
  `_bypass_total`, `_cost_saved_usd_total`, `_lookup_cost_usd_total`,
  `_lookup_duration_seconds`, `_similarity`; hit-rate and net-saving panels on
  the Operations dashboard.
- `scripts/calibrate_cache.py` and `bench/cache_calibration.jsonl`: threshold
  sweep on labelled paraphrase and near-miss pairs.
- Gateway overhead measurement: `Server-Timing: gateway;dur=..., upstream;dur=...`
  on non-streaming completions, `llm_gateway_overhead_seconds` histogram, and
  `scripts/bench_overhead.py` (instant stub provider, sequential and k6
  open-loop runs). Results and method in `docs/benchmark.md`; a
  manual `Benchmark` workflow reruns it on a GitHub runner.
- Routing policies (`llm_gateway.routing`). `model: "auto"` lets the configured
  policy choose the model; `GATEWAY_ROUTING_POLICY=autopilot` delegates to the
  llm-cost-autopilot sidecar over its route API v2 (`POST /v2/route`,
  llm-cost-autopilot v0.2.0, `docker-compose.autopilot.yml`): `simple` maps to
  the budget tier and `moderate` to high quality, with a hard timeout, an
  optional Bearer token and fallback to the static policy. A contract test checks the wire
  shapes against autopilot's published JSON Schema.
- `X-Gateway-Cache`, `X-Gateway-Cache-Similarity` and `X-Gateway-Routing-Reason`
  response headers; `llm_gateway_routing_decisions_total` metric.
- Bruno requests "Auto routing" and "Cached (bypass)".
- `numpy` dependency (cache similarity search).
- OpenTelemetry tracing (`llm_gateway.observability.tracing`): one GenAI
  `CLIENT` span per chat or embeddings call, named and attributed per the GenAI
  semantic conventions v1.37.0, with `error.type` and status `ERROR` on
  failure and the caller's `traceparent` as parent. Exported over OTLP/HTTP
  when `GATEWAY_TRACING_ENABLED=true`, configured by the standard `OTEL_*`
  variables. Message content only with
  `GATEWAY_TRACING_CAPTURE_MESSAGE_CONTENT=true`.
- `opentelemetry-exporter-otlp-proto-http` dependency.

### Changed
- **Breaking:** default models are now `qwen3.5:4b` (high quality),
  `qwen3.5:0.8b` (budget) and `qwen3-embedding:0.6b` (embeddings, 1024
  dimensions), replacing `llama3.2:3b`, `qwen2.5:0.5b` and `nomic-embed-text`.
  Pull them with `python -m llm_gateway.pull_models`.
- README results re-measured on Python 3.14 with the new layout (overhead
  benchmark alone on the machine, cache calibration, RAG cost split), every number
  taken from `results/manifest.json`.
- Model ids come only from settings. The demo and load scripts ask the gateway
  (`GET /v1/models`) for each tier, the benchmark, calibration and reproduce
  script read `Settings`, and the k6 script needs `MODEL` from its caller. The
  load test's quality-vs-budget price ratio is read from the catalogue instead
  of a hard-coded 30.
- **Breaking:** Python 3.12 or newer is required (was 3.11). Development and CI
  run on 3.14.
- Build backend switched from hatchling to `uv_build`. Dev tools moved from the
  `dev` extra to the `dev` dependency group (`uv sync` installs them).
- Security lint comes from ruff's `S` rules only; coverage floor raised from
  85% to 90%.
- Pre-commit: hooks pinned by commit SHA, `ruff-check` hook id, pyright run in
  the project venv, gitleaks added. Commitizen configured for `cz bump`.
- CI: pre-commit, tests on Python 3.14 and a wheel build started in a clean venv;
  every Action pinned by commit SHA; read-only token. Dependabot updates uv,
  Actions and pre-commit hooks monthly.
- Docker image on `python:3.14-slim`, installed with `uv sync --locked`.
- README cut to one page; feature details moved to `docs/` (cache, routing,
  tracing, resilience, RAG cost attribution, benchmark, configuration, project
  structure, Bruno collection, development). The demo GIF and its vhs tape
  are removed: they showed retired models and a savings figure no benchmark
  supports. The README shows the quick start's real output instead.
- Demo scripts and the Bruno environment default to the API key in
  `config/teams.example.yaml`, so they work on a fresh copy.
- **Breaking:** circuit-breaker settings renamed from `GATEWAY_CB_*` to
  `GATEWAY_CIRCUIT_BREAKER_*` (`_FAILURE_RATE_THRESHOLD`, `_MIN_REQUESTS`,
  `_OPEN_DURATION_S`). No aliases: an old name now stops startup with the new
  name in the error.
- Embeddings call Ollama's `/api/embed` instead of the superseded
  `/api/embeddings`, and are priced from its `prompt_eval_count` instead of a
  whitespace word count.
- The token bucket runs its Lua script through redis-py's `register_script`.
- Each setting is documented by an attribute docstring in `config.py`, also
  exposed as the field description.

- Fallback models are now filtered by the team's `allowed_models` too. Before,
  a high-quality request could fall back to a budget model the team was not
  allowed to call.
- The FastAPI app version now comes from `llm_gateway.__version__`.

### Docs
- `docs/DECISIONS.md` is replaced by Architecture Decision Records in
  `docs/adr/` (MADR 4), indexed in `docs/adr/README.md`. All seven are
  accepted.

### Fixed
- The SQLite file now runs in WAL mode, the gateway holds one connection open
  for its lifetime (`CostLedger.keep_wal_open`) so the WAL is not checkpointed
  and deleted after every request, and every connection states its lock wait
  (`BUSY_TIMEOUT_S`, 5 s) in one helper. Before, budget reads blocked ledger
  writes, and at 500 req/s 8.8% of requests failed with `database is locked`;
  now 2.1% do, and one worker serves about 460 req/s instead of 300.
- Streaming chat spans were started before the response body, so a client that
  disconnected before the first chunk left the span open and never exported.
  The span now opens inside the stream.
- Streaming chat responses always reported `finish_reason: "stop"`, on the span
  and in the final SSE chunk. They now carry Ollama's `done_reason` (for example
  `length` when `max_tokens` cut the answer).
- A partial `PUT /v1/admin/teams/{id}` reset the team's `semantic_cache` opt-in
  to `false`. Fields the request does not mention are now kept.
- Rate limiting failed on every request after a Redis restart or
  `SCRIPT FLUSH` (`NoScriptError`): the script SHA was cached and never
  reloaded.

### Removed
- `Makefile`: its commands are documented in `docs/development.md`.
- `gateway` console script, which pointed at a module that did not exist.
- Unused `typer` dependency; `rich` is now a dev dependency (scripts only).
- Root `conftest.py` path workaround for hatchling's editable install.
- All em-dashes from code and docs.
- `GATEWAY_HOST`, `GATEWAY_PORT` and `GATEWAY_METRICS_ENABLED`: nothing read
  them (uvicorn takes `--host` / `--port`; metrics are always on).
- Unused code: `CircuitStateRepository.list_all`, `RagOperationType.from_string`
  (the tracker now takes the enum only), `ModelConfig.provider_name`,
  `Metrics.registry`, the `METRICS_CONTENT_TYPE` alias, the never-varied
  path parameters of the three middlewares, and the `create_app` injection
  parameters no test used.
- The README limitation "No semantic response caching".
- The README claim that OpenTelemetry tracing was wired: until this release
  only the `GATEWAY_TRACING_ENABLED` setting existed.
- `opentelemetry-instrumentation-fastapi` dependency (never used).

## [0.1.0] - 2026-05-23

First public release: OpenAI-compatible gateway with RAG cost attribution,
circuit breaker, Redis rate limiting, budget caps, Prometheus metrics and
three Grafana dashboards.
