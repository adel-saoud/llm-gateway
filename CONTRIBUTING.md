# Contributing

## Setup
```bash
uv sync --all-extras
uv pip install -e .
uv run pre-commit install
```

## Run quality gates locally
```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyright
uv run pytest
```

CI runs the same gates. Coverage threshold is **85%**.

## Commits
Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`).
`commitizen` is wired into `pre-commit` to enforce this.

## Adding an LLM provider
Today the gateway speaks to one backend (Ollama). Adding a second provider
(OpenAI, Anthropic, vLLM, etc.) is a structural change:

1. Implement the [`Provider` Protocol](src/llm_gateway/providers/base.py) —
   `chat`, `stream_chat`, `embed`, `health` plus the `name` property. Error
   normalisation must produce `ProviderTimeoutError` for timeouts,
   `ProviderUnavailableError` for transient transport failures, and
   `ProviderError` for request-shape problems (4xx / malformed body).
2. Register a `ModelConfig` for each model the new provider exposes in
   [`providers/registry.py`](src/llm_gateway/providers/registry.py), with the
   correct simulated cost tier.
3. Wire the provider into the lifespan in
   [`main.py`](src/llm_gateway/main.py) and add it to the default
   `RouteChain` for the request handlers that need it.
4. Add hermetic tests under `tests/unit/` — mock the HTTP layer with
   `respx`; no real network calls.

## Adding a new RAG operation type
1. Add the new value to
   [`RagOperationType`](src/llm_gateway/rag/models.py) — the storage layer
   accepts any string, so old rows with the new value will load cleanly via
   `from_string()`.
2. Add a classifier rule in
   [`rag/tagger.py`](src/llm_gateway/rag/tagger.py). Priority matters — see
   the docstring.
3. Update [`rag/cost_attribution.py`](src/llm_gateway/rag/cost_attribution.py)
   so the new operation appears in the report (the named-operations list is
   derived from the enum, so this is usually automatic).
4. Add Prometheus label handling — the existing `record_rag_event` already
   uses the enum value, so no change is required unless you add new metric
   dimensions.

## Adding a Prometheus metric
1. Define the collector in
   [`observability/metrics.py`](src/llm_gateway/observability/metrics.py)
   inside `Metrics.__init__`. Use the `llm_gateway_*` prefix.
2. Add a recording helper on the `Metrics` class so callers don't construct
   label dicts at the call site.
3. If the metric is dashboard-worthy, add a panel to the appropriate JSON
   file under [`grafana/dashboards/`](grafana/dashboards/). The
   `test_grafana_dashboards.py` checks will fail if you reference a metric
   that doesn't exist.

## Evolving stored data
SQLite tables in [`storage/db.py`](src/llm_gateway/storage/db.py) share one
file with a single `schema_version` row. When you add a non-additive change:

1. Bump `CURRENT_SCHEMA_VERSION`.
2. Add the new table / column to the `_SCHEMA` string. Prefer
   `CREATE TABLE IF NOT EXISTS` (idempotent) and additive columns over
   destructive `ALTER TABLE`.
3. Add a test in `tests/unit/test_storage.py` that exercises the new shape.
