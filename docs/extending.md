# Extending llm-gateway

Setup, the quality gate and commit rules are in the shared
[contributing guide](https://github.com/adel-saoud/.github/blob/main/CONTRIBUTING.md).
This page covers the extension points specific to this project.

## Adding an LLM provider

Today the gateway speaks to one backend (Ollama). Adding a second provider (OpenAI,
Anthropic, vLLM, etc.) is a structural change:

1. Implement the [`Provider` Protocol](../src/llm_gateway/providers/base.py): `chat`,
   `stream_chat`, `embed`, `health` plus the `name` property. Error normalisation must
   produce `ProviderTimeoutError` for timeouts, `ProviderUnavailableError` for transient
   transport failures, and `ProviderError` for request-shape problems (4xx or malformed
   body).
2. Register a `ModelConfig` for each model the new provider exposes in
   [`providers/registry.py`](../src/llm_gateway/providers/registry.py), with the correct
   simulated cost tier.
3. Wire the provider into the lifespan in [`main.py`](../src/llm_gateway/main.py) and
   into the chains the routing policies build
   ([`routing/policy.py`](../src/llm_gateway/routing/policy.py)). The provider's `name`
   becomes `gen_ai.provider.name` on trace spans, so use the OTel GenAI well-known value
   when one exists (`openai`, `anthropic`, ...).
4. Add hermetic tests under `tests/unit/`. Mock the HTTP layer with `respx`; no real
   network calls.

## Adding a new RAG operation type

1. Add the new value to [`RagOperationType`](../src/llm_gateway/rag/models.py). The
   ledger stores the value as a plain string, so no migration is needed.
2. Add a classifier rule in [`rag/tagger.py`](../src/llm_gateway/rag/tagger.py).
   Priority matters; see the docstring.
3. Update [`rag/cost_attribution.py`](../src/llm_gateway/rag/cost_attribution.py) so the
   new operation appears in the report. The named-operations list is derived from the
   enum, so this is usually automatic.
4. Check the Prometheus labels. `record_rag_event` already uses the enum value, so no
   change is needed unless you add new metric dimensions.

## Adding a Prometheus metric

1. Define the collector in
   [`observability/metrics.py`](../src/llm_gateway/observability/metrics.py) inside
   `Metrics.__init__`. Use the `llm_gateway_*` prefix.
2. Add a recording helper on the `Metrics` class so callers don't build label dicts at
   the call site.
3. If the metric is dashboard-worthy, add a panel to the matching JSON file under
   [`grafana/dashboards/`](../grafana/dashboards/). The `test_grafana_dashboards.py`
   checks fail if a panel references a metric that doesn't exist.

## Changing trace span attributes

GenAI attribute names are pinned in
[`observability/tracing.py`](../src/llm_gateway/observability/tracing.py) and are the
contract trace-to-golden reads. Change them only to follow a new version of the OTel
GenAI semantic conventions, update the version named in that module's docstring and in
[tracing.md](tracing.md), and keep message content behind
`GATEWAY_TRACING_CAPTURE_MESSAGE_CONTENT`. `tests/unit/test_tracing.py` asserts the
exact attribute set of each span.

## Evolving stored data

SQLite tables in [`storage/db.py`](../src/llm_gateway/storage/db.py) share one file with
a single `schema_version` row. For a non-additive change:

1. Bump `CURRENT_SCHEMA_VERSION`.
2. Add the new table or column to the `_SCHEMA` string. Prefer
   `CREATE TABLE IF NOT EXISTS` (idempotent) and additive columns over destructive
   `ALTER TABLE`.
3. Add a test in `tests/unit/test_storage.py` that exercises the new shape.
