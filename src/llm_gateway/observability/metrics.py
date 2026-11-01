"""Prometheus collectors: standard and RAG-specific.

All metric updates flow through the :class:`Metrics` facade so:

* there is exactly one place that owns the collector definitions and their
  label vocabulary,
* tests construct a fresh :class:`prometheus_client.CollectorRegistry` per
  instance instead of polluting the global registry,
* the handler/middleware code that records events stays readable (no
  ``metrics.cost_usd_total.labels(team=..., model=...).inc(...)`` chains
  scattered across the codebase).

The metric names and labels are the dashboards' contract
(``tests/unit/test_grafana_dashboards.py`` checks every panel query against
them). HTTP-level events that never reached a model selection (auth
rejections, health checks) use the empty string for the ``model`` label, so
the dashboards can ``ignoring(model)`` over them.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from llm_gateway.circuit_breaker.state_machine import CircuitBreakerState
from llm_gateway.rag.models import RagOperationType

CIRCUIT_STATE_VALUES: Final[dict[CircuitBreakerState, float]] = {
    CircuitBreakerState.CLOSED: 0.0,
    CircuitBreakerState.HALF_OPEN: 1.0,
    CircuitBreakerState.OPEN: 2.0,
}
"""Gauge encoding of circuit states, ordered by severity so ``max()`` over models is meaningful."""


_FAST_BUCKETS_S: Final[tuple[float, ...]] = (
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
)
"""Sub-millisecond buckets: overhead and cache lookups sit below the default 5 ms floor."""
_SIMILARITY_BUCKETS: Final[tuple[float, ...]] = (
    0.5,
    0.6,
    0.7,
    0.8,
    0.85,
    0.9,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    1.0,
)
"""Dense above 0.9, where the calibration showed paraphrases and near-misses overlap."""


class Metrics:
    """Facade over Prometheus collectors. One instance per :class:`FastAPI` app."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry or CollectorRegistry()

        # ── Standard ───────────────────────────────────────────────────────
        self.requests_total = Counter(
            "llm_gateway_requests_total",
            "Total HTTP requests handled by the gateway.",
            ["team", "model", "status"],
            registry=self._registry,
        )
        self.request_duration_seconds = Histogram(
            "llm_gateway_request_duration_seconds",
            "End-to-end request duration in seconds.",
            ["team", "model"],
            registry=self._registry,
        )
        self.tokens_total = Counter(
            "llm_gateway_tokens_total",
            "Tokens processed by the gateway.",
            ["team", "model", "type"],
            registry=self._registry,
        )
        self.cost_usd_total = Counter(
            "llm_gateway_cost_usd_total",
            "Cumulative simulated cost in USD.",
            ["team", "model"],
            registry=self._registry,
        )
        self.rate_limit_hits_total = Counter(
            "llm_gateway_rate_limit_hits_total",
            "Requests rejected by the per-team rate limiter.",
            ["team"],
            registry=self._registry,
        )
        self.budget_utilisation = Gauge(
            "llm_gateway_budget_utilisation",
            "Per-team budget utilisation (0.0–1.0; >1.0 means over cap).",
            ["team"],
            registry=self._registry,
        )
        self.circuit_breaker_state = Gauge(
            "llm_gateway_circuit_breaker_state",
            "Circuit-breaker state per (provider, model). 0=CLOSED, 1=HALF_OPEN, 2=OPEN.",
            ["provider", "model"],
            registry=self._registry,
        )
        self.fallback_total = Counter(
            "llm_gateway_fallback_total",
            "Provider-router fallback events.",
            ["from_model", "to_model", "reason"],
            registry=self._registry,
        )

        # ── RAG-specific ───────────────────────────────────────────────────
        self.rag_cost_usd_total = Counter(
            "llm_gateway_rag_cost_usd_total",
            "RAG cost in USD per team and operation type.",
            ["team", "operation_type"],
            registry=self._registry,
        )
        self.rag_requests_total = Counter(
            "llm_gateway_rag_requests_total",
            "RAG requests per team and operation type.",
            ["team", "operation_type"],
            registry=self._registry,
        )
        self.rag_tokens_total = Counter(
            "llm_gateway_rag_tokens_total",
            "RAG tokens per team, operation type, and direction.",
            ["team", "operation_type", "token_type"],
            registry=self._registry,
        )
        self.rag_duration_seconds = Histogram(
            "llm_gateway_rag_duration_seconds",
            "RAG operation duration per team and operation type.",
            ["team", "operation_type"],
            registry=self._registry,
        )

        # ── Gateway overhead (time not spent waiting on the provider) ──────
        self.overhead_seconds = Histogram(
            "llm_gateway_overhead_seconds",
            "Gateway time per non-streaming completion, excluding the provider call.",
            ["team"],
            buckets=_FAST_BUCKETS_S,
            registry=self._registry,
        )

        # ── Semantic cache ─────────────────────────────────────────────────
        self.semantic_cache_requests_total = Counter(
            "llm_gateway_semantic_cache_requests_total",
            "Chat requests by cache result (hit_exact, hit_semantic, miss, bypass).",
            ["team", "model", "result"],
            registry=self._registry,
        )
        self.semantic_cache_bypass_total = Counter(
            "llm_gateway_semantic_cache_bypass_total",
            "Chat requests that skipped the cache, by reason.",
            ["team", "reason"],
            registry=self._registry,
        )
        self.semantic_cache_cost_saved_usd_total = Counter(
            "llm_gateway_semantic_cache_cost_saved_usd_total",
            "Simulated USD the cached completions would have cost (gross).",
            ["team", "model"],
            registry=self._registry,
        )
        self.semantic_cache_lookup_cost_usd_total = Counter(
            "llm_gateway_semantic_cache_lookup_cost_usd_total",
            "Simulated USD spent embedding prompts for cache lookups.",
            ["team"],
            registry=self._registry,
        )
        self.semantic_cache_lookup_duration_seconds = Histogram(
            "llm_gateway_semantic_cache_lookup_duration_seconds",
            "Cache lookup time, including the embedding call on exact misses.",
            ["team"],
            buckets=_FAST_BUCKETS_S,
            registry=self._registry,
        )
        self.semantic_cache_similarity = Histogram(
            "llm_gateway_semantic_cache_similarity",
            "Top cosine similarity per semantic lookup, accepted or not.",
            ["team"],
            buckets=_SIMILARITY_BUCKETS,
            registry=self._registry,
        )

        # ── Routing ────────────────────────────────────────────────────────
        self.routing_decisions_total = Counter(
            "llm_gateway_routing_decisions_total",
            "Routing plans by policy and reason.",
            ["team", "policy", "reason"],
            registry=self._registry,
        )

    # ── Recording helpers: single place for label values ───────────────────

    def record_http_request(
        self, *, team: str, status: int, duration_s: float, model: str = ""
    ) -> None:
        """Middleware-level: every HTTP request, including non-LLM endpoints."""
        self.requests_total.labels(team=team, model=model, status=str(status)).inc()
        self.request_duration_seconds.labels(team=team, model=model).observe(duration_s)

    def record_tokens(
        self, *, team: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """Add token counts; zero counts create no series."""
        if prompt_tokens:
            self.tokens_total.labels(team=team, model=model, type="prompt").inc(prompt_tokens)
        if completion_tokens:
            self.tokens_total.labels(team=team, model=model, type="completion").inc(
                completion_tokens
            )

    def record_cost(self, *, team: str, model: str, cost_usd: float) -> None:
        """Add simulated spend; a zero cost (cache hit, unknown model) creates no series."""
        if cost_usd:
            self.cost_usd_total.labels(team=team, model=model).inc(cost_usd)

    def record_rate_limit_hit(self, *, team: str) -> None:
        """Count one 429 from the token bucket."""
        self.rate_limit_hits_total.labels(team=team).inc()

    def set_budget_utilisation(self, *, team: str, utilisation: float) -> None:
        """Set the team's spend-to-cap ratio as last seen by the enforcer."""
        self.budget_utilisation.labels(team=team).set(utilisation)

    def set_circuit_state(self, *, provider: str, model: str, state: CircuitBreakerState) -> None:
        """Set the gauge from a breaker transition (see :data:`CIRCUIT_STATE_VALUES`)."""
        self.circuit_breaker_state.labels(provider=provider, model=model).set(
            CIRCUIT_STATE_VALUES[state]
        )

    def record_fallback(self, *, from_model: str, to_model: str, reason: str) -> None:
        """Count one hop down the chain; ``reason`` is an exception name or ``circuit_open``."""
        self.fallback_total.labels(from_model=from_model, to_model=to_model, reason=reason).inc()

    def record_rag_event(
        self,
        *,
        team: str,
        operation_type: RagOperationType,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
        duration_s: float,
    ) -> None:
        """Record one tagged LLM call: request count, cost, tokens and duration per operation."""
        op = operation_type.value
        self.rag_requests_total.labels(team=team, operation_type=op).inc()
        if cost_usd:
            self.rag_cost_usd_total.labels(team=team, operation_type=op).inc(cost_usd)
        if prompt_tokens:
            self.rag_tokens_total.labels(
                team=team,
                operation_type=op,
                token_type="prompt",  # noqa: S106  # Prometheus label, not a credential
            ).inc(prompt_tokens)
        if completion_tokens:
            self.rag_tokens_total.labels(
                team=team,
                operation_type=op,
                token_type="completion",  # noqa: S106  # Prometheus label, not a credential
            ).inc(completion_tokens)
        self.rag_duration_seconds.labels(team=team, operation_type=op).observe(duration_s)

    def record_overhead(self, *, team: str, overhead_s: float) -> None:
        """Observe gateway time outside the provider call, clamped at 0 against clock jitter."""
        self.overhead_seconds.labels(team=team).observe(max(overhead_s, 0.0))

    def record_cache_bypass(self, *, team: str, model: str, reason: str) -> None:
        """Count a request that skipped the cache, in both the result and the reason counters."""
        self.semantic_cache_requests_total.labels(team=team, model=model, result="bypass").inc()
        self.semantic_cache_bypass_total.labels(team=team, reason=reason).inc()

    def record_cache_lookup(
        self,
        *,
        team: str,
        model: str,
        result: str,
        duration_s: float,
        top_similarity: float | None,
        lookup_cost_usd: float,
        cost_saved_usd: float,
    ) -> None:
        """Record one cache lookup; ``top_similarity`` is None when no semantic search ran."""
        self.semantic_cache_requests_total.labels(team=team, model=model, result=result).inc()
        self.semantic_cache_lookup_duration_seconds.labels(team=team).observe(duration_s)
        if top_similarity is not None:
            self.semantic_cache_similarity.labels(team=team).observe(top_similarity)
        if lookup_cost_usd:
            self.semantic_cache_lookup_cost_usd_total.labels(team=team).inc(lookup_cost_usd)
        if cost_saved_usd:
            self.semantic_cache_cost_saved_usd_total.labels(team=team, model=model).inc(
                cost_saved_usd
            )

    def record_routing_decision(self, *, team: str, policy: str, reason: str) -> None:
        """Count one routing plan by policy and its bounded ``reason`` vocabulary."""
        self.routing_decisions_total.labels(team=team, policy=policy, reason=reason).inc()

    def render(self) -> bytes:
        """Serialise the current state in Prometheus exposition format."""
        return generate_latest(self._registry)
