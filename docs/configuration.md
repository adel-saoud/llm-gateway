# Configuration

## Pointing a client at the gateway

The gateway speaks the OpenAI chat-completions and embeddings wire format. Point any
OpenAI client at `http://localhost:8000/v1` and pass the team's API key as the Bearer
token:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-gw-...")
response = client.chat.completions.create(
    model="qwen3.5:4b",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={"X-Rag-Operation": "generation"},  # optional explicit tag
)
```

### Thinking

The default models (Qwen3.5) think before answering unless told not to, and a thinking
run of "Say hi." produced 3,684 output tokens instead of 12 in a local test. The gateway
therefore sends `think: false` to Ollama unless the request asks otherwise with
OpenAI's `reasoning_effort`:

| `reasoning_effort` | Thinking |
|:--|:--|
| absent or `"none"` | off |
| `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` | on (Ollama takes on/off for Qwen, so the level is not forwarded) |

With thinking on, the thinking tokens count as completion tokens: they are in `usage`,
the cost header, the ledger and the team budget, and `max_tokens` caps thinking and answer
together (a low cap can end with an empty answer and `finish_reason: "length"`). The
thinking text itself is not returned. Thinking requests bypass the response cache.

## Response headers

| Header | Example | Meaning |
|:--|:--|:--|
| `X-Gateway-Cost-USD` | `0.001143` | Cost of this request (simulated, matches the ledger) |
| `X-Gateway-Model-Used` | `qwen3.5:4b` | Model that served the request (differs from the request if fallback fired) |
| `X-Gateway-Rag-Operation` | `generation` | Operation type the classifier assigned ([RAG cost attribution](rag-cost-attribution.md)) |
| `X-Gateway-Routing-Reason` | `requested_model` | Why this model chain was chosen ([routing](routing.md)) |
| `X-Gateway-Cache` | `hit` | `hit`, `miss` or `bypass` ([response cache](cache.md)) |
| `X-Gateway-Cache-Similarity` | `1.0000` | On a hit: 1.0 for an exact match, the cosine similarity for a semantic one |
| `X-Gateway-Request-Id` | `3f2c...` | Id written to the cost ledger |
| `Server-Timing` | `gateway;dur=2.4, upstream;dur=180.2` | Milliseconds in the gateway and waiting on the provider (non-streaming only) |

## Teams

Teams live in a YAML file, `config/teams.yaml` by default (`GATEWAY_TEAMS_YAML`). Copy
[`config/teams.example.yaml`](../config/teams.example.yaml), replace the keys and
restart. Each team has an API key, an allowed-model list, rate limits, a daily budget
cap, an admin flag and the `semantic_cache` opt-in. Allowed-model entries are tiers
(`high_quality`, `budget`, `embeddings`), which follow the `GATEWAY_MODEL_*` settings,
or model ids, for a team pinned to one model. `PUT /v1/admin/teams/{id}` (admin
key required) updates a team without a restart; the YAML file on disk is not rewritten.

## Models and prices

Model ids come from settings (`GATEWAY_MODEL_HIGH_QUALITY`, `GATEWAY_MODEL_BUDGET`,
`GATEWAY_MODEL_EMBEDDINGS`); nothing else hardcodes a model string. The per-1k prices in
[`providers/registry.py`](../src/llm_gateway/providers/registry.py) are simulated:

| Model | Role | Input per 1k | Output per 1k |
|:--|:--|--:|--:|
| `qwen3.5:4b` | High quality | $0.003 | $0.015 |
| `qwen3.5:0.8b` | Budget, fallback | $0.0001 | $0.0002 |
| `qwen3-embedding:0.6b` | Embeddings | $0.0001 | n/a |

## Environment variables

Every setting is a `GATEWAY_*` environment variable read by `pydantic-settings`
([`config.py`](../src/llm_gateway/config.py), where each setting's docstring explains it).
[`.env.example`](../.env.example) lists them all with their defaults, grouped by area:
logging, Ollama, Redis, SQLite, auth, circuit breaker, cache, routing and tracing. Host
and port are uvicorn flags (`--host`, `--port`), not settings. Tracing also reads the
standard `OTEL_*` variables, which must be real environment variables
([tracing](tracing.md)).

An unknown `GATEWAY_*` name, in the environment or in `.env`, stops startup with the
closest valid name, for example:

```
UnknownSettingError: unknown GATEWAY_* variable(s): GATEWAY_CB_MIN_REQUESTS (did you mean
GATEWAY_CIRCUIT_BREAKER_MIN_REQUESTS?). Valid names are listed in .env.example.
```

pydantic-settings would otherwise ignore the name and keep the default, so a typo or a
setting renamed between versions would go unnoticed. A setting removed without a
successor (`GATEWAY_AUTOPILOT_TIER_COMPLEX`) gets the reason it is gone instead of a
close name, which would point at an unrelated setting.
