---
status: accepted
date: 2026-10-31
---

# Model ids only in settings; team allowlists name tiers

## Context and Problem Statement

The default models changed from `llama3.2:3b`, `qwen2.5:0.5b` and `nomic-embed-text` to
`qwen3.5:4b`, `qwen3.5:0.8b` and `qwen3-embedding:0.6b` ([`CHANGELOG.md`](../../CHANGELOG.md)
Unreleased, Changed). Model ids had been repeated in scripts, compose, the Bruno collection,
the example teams and the docs, and team allowlists listed model ids, so a model change
meant editing every one of them. Where should a model id be written, and what should a team
allowlist name?

## Considered Options

* Model ids repeated wherever they are used, allowlists of model ids
* Model ids in settings only, allowlists of model ids
* Model ids in settings only, allowlists that accept tiers as well as model ids

## Decision Outcome

Chosen option: "Model ids in settings only, allowlists that accept tiers as well as model
ids". `GATEWAY_MODEL_HIGH_QUALITY`, `GATEWAY_MODEL_BUDGET` and `GATEWAY_MODEL_EMBEDDINGS`
are the only place a model id is set. Scripts ask the gateway (`GET /v1/models`) or read
`Settings`, compose pulls models with `python -m llm_gateway.pull_models`, and an allowlist
entry may be `high_quality`, `budget` or `embeddings`, resolved through the registry at
request time. A model id is still accepted, for a team pinned to one model
(`TeamConfig.is_model_allowed`, [`auth/models.py`](../../src/llm_gateway/auth/models.py)).

### Consequences

* Good, because a model swap is one setting; [`config/teams.example.yaml`](../../config/teams.example.yaml)
  and the Bruno collection need no edit.
* Good, because `pull_models` and the gateway read the same settings, so the pulled models
  cannot drift from the served ones ([`pull_models.py`](../../src/llm_gateway/pull_models.py)).
* Bad, because a tier entry silently follows the setting: a team allowed `high_quality`
  gets whichever model that tier serves today. A team that must not move pins a model id.
* Neutral: an unknown `GATEWAY_*` name stops startup with the closest valid name, so a
  mistyped model variable fails loudly instead of falling back to a default
  ([`config.py`](../../src/llm_gateway/config.py)).
