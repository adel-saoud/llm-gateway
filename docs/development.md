# Development

Setup, commit rules and the review flow shared by all my repos are in the
[contributing guide](https://github.com/adel-saoud/.github/blob/main/CONTRIBUTING.md).
How to add a provider, an operation type, a metric or a storage migration is in
[extending.md](extending.md).

## Local gate

```bash
uv sync                                  # Python 3.14 from .python-version, dev group included
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run pre-commit run --all-files        # ruff, ruff format, pyright, gitleaks, file checks
uv run pytest                            # hermetic suite, fails under 90% coverage
uv build                                 # sdist + wheel with the uv_build backend
```

CI ([`ci.yml`](../.github/workflows/ci.yml)) runs the same hooks and tests on Python 3.14,
then installs the wheel in a clean venv and checks that the app answers `/health`.

The `sync-schemas` hook checks that `tests/contracts/` still matches the route API schema
of the pinned llm-cost-autopilot release ([routing](routing.md#contract-with-llm-cost-autopilot)).

## Models

The three model ids are settings: `GATEWAY_MODEL_HIGH_QUALITY`, `GATEWAY_MODEL_BUDGET` and
`GATEWAY_MODEL_EMBEDDINGS` ([configuration](configuration.md)). Nothing else names a model:
the compose `ollama-init` service pulls what the settings name, the demo and load scripts
ask the running gateway (`GET /v1/models`) for each tier, and the benchmark and calibration
read the same settings. To change a model, set it in `.env`, then pull it:

```bash
uv run python -m llm_gateway.pull_models   # pulls the configured models into GATEWAY_OLLAMA_BASE_URL
```

Team allowlists in `config/teams.yaml` name models too; update them with the settings.

## Full stack with Ollama

The Docker Compose stack runs the gateway with Redis, Prometheus, Grafana and Ollama.
Create the local config files first (both are gitignored):

```bash
cp config/teams.example.yaml config/teams.yaml
```

**Bundled Ollama** (pulls the configured models on first start, cached in a volume):

```bash
docker compose up -d --build
docker compose logs -f gateway           # wait for "Application startup complete"
```

**Host Ollama** (you already run Ollama with the models):

```bash
uv run python -m llm_gateway.pull_models
cp docker-compose.override.yml.example docker-compose.override.yml
docker compose up -d --build
```

The override replaces the bundled Ollama services with stubs and points the gateway at
the host. Other commands:

```bash
docker compose down                      # stop, keep volumes
docker compose down -v --remove-orphans  # stop and delete volumes (ledger, models, dashboards)
```

Grafana is on `http://localhost:3000`, Prometheus on `http://localhost:9090`.

## Demos

```bash
uv run python scripts/demo_rag.py                       # 5-step RAG pipeline + cost report
uv run python scripts/load_test.py --requests 200       # RAG traffic mix, latency percentiles
uv run python scripts/demo_resilience.py                # circuit breaker OPEN, HALF_OPEN, CLOSED
uv run python scripts/calibrate_cache.py                # semantic-cache threshold sweep (host Ollama)
```

Running the first three in order populates the three Grafana dashboards.
The benchmark has its own page: [benchmark.md](benchmark.md).

## Reproducing the README numbers

```bash
docker compose up -d redis
uv run python -m llm_gateway.pull_models   # the calibration needs the embedding model
uv run python scripts/reproduce.py
```

It reruns the overhead benchmark, the cache calibration and the RAG demo, copies the raw
output to `results/raw/` and writes [`results/manifest.json`](../results/manifest.json):
date, commit, Python and package versions, hardware, model digests, dataset hash, and each
README number with its sample size. Every number in the README comes from that file.

## Dashboard screenshots

The dashboard screenshots (`docs/dashboard-*.png`) are taken by hand from Grafana after
the three demos above.
