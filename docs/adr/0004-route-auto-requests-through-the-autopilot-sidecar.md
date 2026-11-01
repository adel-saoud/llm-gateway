---
status: accepted
date: 2026-10-31
---

# Route `model: "auto"` requests through the llm-cost-autopilot sidecar

## Context and Problem Statement

Some prompts do not need the high-quality model. llm-cost-autopilot classifies a prompt
into a tier and owns route API v2; its side of the contract is recorded in its own ADRs
([llm-cost-autopilot `docs/adr/`](https://github.com/adel-saoud/llm-cost-autopilot/blob/main/docs/adr/)).
This record covers the caller side: which requests the gateway sends to it, what it does
with the answer, and what happens when autopilot is slow or down.

## Decision Drivers

* A client that names a model must always get that model.
* Routing may slow a request down but must never fail it.
* Team allowlists, the registry and the fallback chain stay the gateway's.
* The gateway image stays free of torch and transformer models.

## Considered Options

* Autopilot in process, as a plugin
* Autopilot as an HTTP sidecar returning a model id
* Autopilot as an HTTP sidecar returning a tier the gateway maps to its own models

## Decision Outcome

Chosen option: "Autopilot as an HTTP sidecar returning a tier the gateway maps to its own
models". Only `model: "auto"` goes to autopilot, and only with
`GATEWAY_ROUTING_POLICY=autopilot`. The client calls `POST /v2/route` with the team as
`tenant_id`, maps `simple` to the budget tier and `moderate` to high quality
(`GATEWAY_AUTOPILOT_TIER_*`), and filters the result through the team's allowlist. A hard
ceiling of 250 ms (`autopilot_timeout_s`, [`config.py`](../../src/llm_gateway/config.py)),
an HTTP error or an invalid body falls back to the static policy
([`docs/routing.md`](../routing.md)). The whole contract lives in one module,
[`routing/autopilot.py`](../../src/llm_gateway/routing/autopilot.py).

### Consequences

* Good, because a pinned copy of autopilot's JSON Schema is checked in pre-commit and CI
  (`scripts/sync_schemas.py`), and contract tests validate the client's requests and
  accepted responses against it ([`docs/routing.md`](../routing.md#contract-with-llm-cost-autopilot)).
* Good, because `X-Gateway-Routing-Reason` says on each response why a chain was chosen.
* Bad, because a contract change is a coordinated release: v2 removed `/v1/route` and the
  `complex` tier with no fallback, a breaking change here
  ([`CHANGELOG.md`](../../CHANGELOG.md) Unreleased, Changed).
* Bad, because autopilot does not learn from gateway traffic: route API v2 has no outcome
  endpoint, since autopilot's verifier would make a model call outside the gateway's budget
  and cost tracking (README Limits).
* Neutral: the routing call emits no span ([`docs/tracing.md`](../tracing.md#not-covered)).

## Pros and Cons of the Options

### Autopilot in process

* Good, because no network hop.
* Bad, because its classifiers need torch and transformer models in the gateway image.

### Sidecar returning a model id

* Good, because one less mapping in the gateway.
* Bad, because autopilot would need the gateway's registry, allowlists and fallback chain
  to pick safely, and a model change would touch two services.
