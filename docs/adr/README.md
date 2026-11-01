# Architecture Decision Records

One file per significant decision, in [MADR 4](https://adr.github.io/madr/) format.
`proposed` until the maintainer confirms the reasoning, then `accepted`.

| ADR | Title | Status |
|:--|:--|:--|
| [0001](0001-own-ollama-adapter-instead-of-litellm.md) | Own Ollama adapter instead of LiteLLM | accepted |
| [0002](0002-sqlite-ledger-in-wal-mode-with-one-held-connection.md) | SQLite cost ledger in WAL mode with one held connection | accepted |
| [0003](0003-in-process-response-cache-with-semantic-matching-off.md) | In-process response cache, exact matching on, semantic matching off | accepted |
| [0004](0004-route-auto-requests-through-the-autopilot-sidecar.md) | Route `model: "auto"` requests through the llm-cost-autopilot sidecar | accepted |
| [0005](0005-model-ids-only-in-settings-and-tier-allowlists.md) | Model ids only in settings; team allowlists name tiers | accepted |
| [0006](0006-thinking-off-by-default-billed-when-on.md) | Thinking off by default, billed when a client turns it on | accepted |
| [0007](0007-hand-written-genai-spans-with-content-off.md) | Hand-written GenAI spans, message content off by default | accepted |
