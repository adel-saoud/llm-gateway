# API collection (Bruno)

The [`bruno/`](../bruno/) folder is a [Bruno](https://www.usebruno.com/) collection that
covers every endpoint. Bruno is open source, stores requests as plain `.bru` files and
needs no account.

```
bruno/
├── environments/
│   └── Local.bru                       baseUrl and API keys
├── collection.bru                      reads the model per tier from GET /v1/models
├── 1. Health/
│   ├── Liveness.bru                    GET /health
│   └── Readiness.bru                   GET /health/ready (checks Ollama)
├── 2. Chat Completions/
│   ├── Generation.bru                  POST /v1/chat/completions, quality model, fallback chain
│   ├── Generation (streaming).bru      same, stream: true
│   ├── Reranking.bru                   budget model + X-Rag-Operation: reranking
│   ├── Evaluation (LLM-as-Judge).bru   budget model, judge system prompt
│   ├── Auto routing.bru                model: "auto", routed by the configured policy
│   └── Cached (bypass).bru             Cache-Control: no-cache, skips the response cache
├── 3. Embeddings/
│   └── Create Embedding.bru            POST /v1/embeddings (qwen3-embedding:0.6b)
├── 4. RAG/
│   └── Cost Report.bru                 GET /v1/rag/cost-report
├── 5. Info/
│   ├── List Models.bru                 GET /v1/models (catalogue + simulated prices)
│   └── Team Stats.bru                  GET /v1/stats (budget utilisation + spend)
├── 6. Admin/
│   ├── Update Rate Limits.bru          PUT /v1/admin/teams/{id}, admin key required
│   ├── Update Budget Cap.bru
│   └── Update Allowed Models.bru
└── 7. Observability/
    └── Prometheus Metrics.bru          GET /metrics
```

To use it: in Bruno, *Open Collection*, pick the `bruno/` folder, then the `Local`
environment. The gateway must be running ([Try it](../README.md#try-it) or the
[full stack](development.md#full-stack-with-ollama)). `Local.bru` uses the key from
`config/teams.example.yaml`; change it if your `config/teams.yaml` uses another.
The model variables (`qualityModel`, `budgetModel`, `embedModel`) are not stored in the
collection: a collection pre-request script fills them from `GET /v1/models` on the first
request, so they always match the gateway's settings.
