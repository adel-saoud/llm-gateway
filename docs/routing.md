# Routing policies and llm-cost-autopilot

A client that names a model always gets that model, plus the budget fallback when it is
the high-quality one. A client that sends `model: "auto"` lets the configured
`RoutingPolicy` choose:

| `GATEWAY_ROUTING_POLICY` | What `"auto"` does |
|:--|:--|
| `static` (default) | High-quality tier, budget fallback |
| `autopilot` | Asks the [llm-cost-autopilot](https://github.com/adel-saoud/llm-cost-autopilot) sidecar to classify the prompt (`simple` / `moderate`) and maps the tier onto the gateway's own models (`GATEWAY_AUTOPILOT_TIER_*`) |

Whatever the policy returns is filtered by the team's `allowed_models`. The autopilot
call has a hard ceiling (`GATEWAY_AUTOPILOT_TIMEOUT_S`, default 250 ms); a timeout, an
HTTP error or an invalid body falls back to the static policy, so autopilot can slow
routing down but can never fail a request. Autopilot returns a tier, never a model id:
the gateway keeps its registry, whitelists and fallback chain.

`X-Gateway-Routing-Reason` on each response says why the chain was chosen.

## Integration

The integration lives in one module,
[`routing/autopilot.py`](../src/llm_gateway/routing/autopilot.py). It speaks autopilot's
route API v2 over HTTP: `POST /v2/route` with the team as `tenant_id`, so autopilot's
correction memory never crosses teams, and an optional Bearer token
(`GATEWAY_AUTOPILOT_API_TOKEN`). A contract test checks the gateway's wire shapes
against a copy of autopilot's published JSON Schema
([`tests/contracts/autopilot-route-api.schema.json`](../tests/contracts/autopilot-route-api.schema.json)).

Autopilot runs as a sidecar rather than in-process because its classifiers need torch
and transformer models that do not belong in the gateway image.

```bash
# llm-cost-autopilot checked out next to this repository
docker compose -f docker-compose.yml -f docker-compose.autopilot.yml up -d --build
```

### Contract with llm-cost-autopilot

The contract tests in [`test_autopilot_policy.py`](../tests/unit/test_autopilot_policy.py)
validate the request the client sends, and the responses it accepts, against the pinned
copy of the schema. [`scripts/sync_schemas.py`](../scripts/sync_schemas.py) keeps that copy
equal to the file in the pinned autopilot release: pre-commit compares it with the sibling
checkout, CI with the raw GitHub file at the tag. To move to a new autopilot release:

```bash
# set tag in CONTRACTS in scripts/sync_schemas.py, then
uv run python scripts/sync_schemas.py          # rewrite the copy
uv run pytest tests/unit/test_autopilot_policy.py
```

and add a CHANGELOG entry naming the new version.

Route API v2 has no outcome endpoint, so autopilot does not learn from gateway traffic
yet: its verifier would make a second model call outside the gateway's budget and cost
tracking.
