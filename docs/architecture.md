# Architecture

## High-level request lifecycle

```mermaid
flowchart LR
    Client([Client]) --> M[MetricsMiddleware<br/>counts every request]
    M --> Auth[AuthMiddleware<br/>resolves team from API key]
    Auth --> RL[RateLimitMiddleware<br/>Redis token bucket]
    RL --> H{Route}

    H -->|chat / embed| Budget[BudgetEnforcer.check<br/>warn at 80% · block at 100%]
    Budget --> Tag[classify<br/>header · model · keywords · default]
    Tag --> Router[ProviderRouter<br/>fallback chain · circuit breaker]
    Router --> Prov[Ollama HTTP]
    Prov --> Track[BudgetTracker.record<br/>SQLite cost ledger]
    Track --> Met[Metrics.record_rag_event]
    Met --> Resp[Response + X-Gateway-* headers]

    H -->|cost-report / stats / models| Read[Read-only<br/>ledger query]
    H -->|admin/teams| Admin[Resolver swap]

    style Met fill:#1f5fff20,stroke:#58a6ff
    style Track fill:#1f5fff20,stroke:#58a6ff
```

The middleware stack runs in **reverse** order of registration. The factory in
[`main.py`](../src/llm_gateway/main.py) registers `RateLimitMiddleware` first,
then `AuthMiddleware`, then `MetricsMiddleware` — so the actual flow is
metrics → auth → rate-limit → handler. Metrics is outermost so it observes
rejections too.

## Provider fallback chain

```mermaid
flowchart LR
    Call[router.chat / embed] --> CB{Circuit OK?<br/>per provider+model}
    CB -->|CLOSED / HALF_OPEN| Try[Try candidate]
    CB -->|OPEN| Skip[Skip to next]
    Try -->|success| OK[Record success]
    Try -->|ProviderUnavailable<br/>ProviderTimeout| Fail[Record failure]
    Try -->|ProviderError<br/>4xx / malformed| Raise[Bubble — no fallback]
    Fail --> Next[Next candidate]
    Skip --> Next
    Next -->|exhausted| Bail[AllProvidersUnavailableError]
```

Streaming has a stricter contract: once the first chunk yields to the caller
the router is **committed** — a mid-stream failure raises to the client
rather than re-routing (the client has already received partial SSE bytes).

## Circuit breaker state machine

```mermaid
flowchart LR
    CLOSED -->|failure rate ≥ threshold<br/>and samples ≥ min_requests| OPEN
    OPEN -->|open_duration_s elapsed<br/>on next is_allowed| HALF_OPEN
    HALF_OPEN -->|probe success<br/>reset window| CLOSED
    HALF_OPEN -->|probe failure| OPEN

    style CLOSED fill:#22c55e20,stroke:#22c55e
    style OPEN fill:#ef444420,stroke:#ef4444
    style HALF_OPEN fill:#eab30820,stroke:#eab308
```

State persists to SQLite (`circuit_state` table) so a restart while a
downstream is OPEN still skips that endpoint until the duration elapses, then
probes once via HALF_OPEN. The failure-rate **window** is in-memory and
resets on restart — that's intentional: the window is for opening decisions,
not for closing ones.

## RAG operation classifier — 5-priority decision tree

```mermaid
flowchart TD
    Start([Request]) --> H{X-Rag-Operation header?}
    H -->|known value| Header[Return header value]
    H -->|missing / typo| E{Embedding model?}
    E -->|yes| Ret[Return RETRIEVAL]
    E -->|no| K1{System prompt contains<br/>judge / evaluate / score?}
    K1 -->|yes| Eval[Return EVALUATION]
    K1 -->|no| K2{System prompt contains<br/>rerank / relevance?}
    K2 -->|yes| Rer[Return RERANKING]
    K2 -->|no| Gen[Return GENERATION]

    style Header fill:#1f5fff20,stroke:#58a6ff
    style Ret fill:#22c55e20,stroke:#22c55e
    style Eval fill:#22c55e20,stroke:#22c55e
    style Rer fill:#22c55e20,stroke:#22c55e
    style Gen fill:#22c55e20,stroke:#22c55e
```

The function is **pure** — handlers extract the three inputs and call
`classify(...)`. Unknown header values intentionally fall through to the
other signals so a typo doesn't bucket every request as `UNKNOWN`.

## Module boundaries

| Module | Responsibility | Key types |
|---|---|---|
| `config` | env-driven settings | `Settings`, `LogLevel` |
| `auth.models` | team-scoped policy | `TeamConfig`, `RateLimitConfig`, `BudgetConfig` |
| `auth.resolver` | YAML-backed `api_key → team` lookup, constant-time match | `TeamResolver` |
| `auth.middleware` | pure-ASGI Bearer-token validation | `AuthMiddleware` |
| `ratelimit.token_bucket` | atomic Redis Lua token bucket | `TokenBucket`, `AcquireResult` |
| `ratelimit.middleware` | 429 + `Retry-After` | `RateLimitMiddleware` |
| `budget.tracker` | computes cost, persists to ledger | `BudgetTracker` |
| `budget.enforcer` | warn at 80% / block at 100% | `BudgetEnforcer`, `BudgetDecision` |
| `rag.tagger` | 5-priority pure classifier | `classify` |
| `rag.cost_attribution` | per-day per-operation aggregator | `RagCostAttributor` |
| `rag.models` | typed enum + event/report shapes | `RagOperationType`, `RagCostEvent`, `RagCostReport` |
| `providers.base` | structural `Provider` Protocol + error hierarchy | `Provider`, `ChatMessage`, `CompletionResult`, `StreamChunk`, `EmbeddingResult` |
| `providers.ollama` | async HTTP adapter, NDJSON streaming, error normalisation | `OllamaProvider` |
| `providers.registry` | model catalogue + simulated prices | `ModelConfig`, `ProviderRegistry`, `ModelTier` |
| `providers.router` | fallback chain + circuit-breaker integration | `ProviderRouter`, `RouteCandidate`, `RouteChain` |
| `circuit_breaker.monitor` | in-memory rolling outcome window | `FailureRateMonitor` |
| `circuit_breaker.state_machine` | 3-state CB with SQLite persistence | `CircuitBreaker`, `CircuitBreakerState` |
| `observability.metrics` | Prometheus collectors + recording helpers | `Metrics` |
| `observability.middleware` | HTTP-level auto-instrumentation | `MetricsMiddleware` |
| `storage.db` | SQLite ledger + circuit-state repo, schema versioning | `CostLedger`, `CostLedgerEntry`, `CircuitStateRepository`, `CircuitStateRecord` |
| `api.completions` / `embeddings` / `rag` / `info` / `admin` / `health` / `metrics` | route modules — one file per concern | — |
| `main` | `create_app` factory + lifespan + middleware registration | — |

## Why these choices

- **Custom Ollama adapter (no `litellm`)** — building the adapter directly keeps the HTTP contract (streaming NDJSON, per-model token counting from `prompt_eval_count`/`eval_count`, normalised error hierarchy) visible and testable rather than hidden behind a router. `litellm` is a valid production choice; the `Provider` Protocol seam means swapping it in is an adapter and a registry entry, no churn elsewhere.
- **Pure ASGI middleware** — `BaseHTTPMiddleware` buffers response bodies, which would break the OpenAI streaming SSE response. Going straight to the ASGI `send` channel keeps streaming response bodies unbuffered.
- **Per-instance Prometheus `CollectorRegistry`** — tests construct a fresh `Metrics()` per assertion. The default global registry would leak counter values between tests; this pattern keeps the suite hermetic without resorting to `prometheus_client.REGISTRY.unregister`.
- **SQLite for circuit state, Redis for the token bucket** — they need different things. The token bucket is atomic-write-heavy and per-request; Redis Lua is the right tool. Circuit state is read-heavy with rare writes (state transitions) and must survive restart; SQLite is the right tool.
- **In-memory failure-rate window** — only used for *opening* decisions; the open state itself is persisted. Losing the window on restart causes a longer warm-up before the breaker can open again, which is the safe direction.
- **`@dataclass(frozen=True, slots=True)` everywhere** — slots save memory at scale; frozen makes the types safe to pass across threads / awaits without copy-on-mutation worries.
- **`RagOperationType` as `StrEnum`** — direct string equality works (`RagOperationType.GENERATION == "generation"`), so the storage layer can persist strings and tests can compare against literals without converting.
- **Atomic Lua token bucket** — the read/refill/decrement happens in a single Redis round-trip. The `test_concurrent_acquires_are_atomic` test fires 40 simultaneous acquires at capacity-20 and verifies exactly 20 ± 1 were allowed. Application-side locking would be racy.
- **Header-first classification with fall-through on typos** — the caller knows best when they bother to set `X-Rag-Operation`, but a typo shouldn't poison the signal. Returning `UNKNOWN` on bad header would silently break attribution.
