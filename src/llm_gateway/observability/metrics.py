"""Prometheus collectors — standard and RAG-specific.

All metric updates flow through the :class:`Metrics` facade so:

* there is exactly one place that owns the collector definitions and their
  label vocabulary,
* tests construct a fresh :class:`prometheus_client.CollectorRegistry` per
  instance instead of polluting the global registry,
* the handler/middleware code that records events stays readable (no
  ``metrics.cost_usd_total.labels(team=..., model=...).inc(...)`` chains
  scattered across the codebase).

The brief's metric names and labels are honoured verbatim; the only place we
add a convention is the empty string ``""`` for the ``model`` label on
HTTP-level events that didn't reach a model selection (e.g. auth rejections),
so the dashboards can ``ignoring(model)`` over them.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from llm_gateway.circuit_breaker.state_machine import CircuitBreakerState
from llm_gateway.rag.models import RagOperationType

METRICS_CONTENT_TYPE: Final[str] = CONTENT_TYPE_LATEST

# Numeric encoding for the circuit-breaker state gauge.
CIRCUIT_STATE_VALUES: Final[dict[CircuitBreakerState, float]] = {
    CircuitBreakerState.CLOSED: 0.0,
    CircuitBreakerState.HALF_OPEN: 1.0,
    CircuitBreakerState.OPEN: 2.0,
}


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

    @property
    def registry(self) -> CollectorRegistry:
        return self._registry

    # ── Recording helpers — single place for label values ───────────────────

    def record_http_request(
        self, *, team: str, status: int, duration_s: float, model: str = ""
    ) -> None:
        """Middleware-level: every HTTP request, including non-LLM endpoints."""
        self.requests_total.labels(team=team, model=model, status=str(status)).inc()
        self.request_duration_seconds.labels(team=team, model=model).observe(duration_s)

    def record_tokens(
        self, *, team: str, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        if prompt_tokens:
            self.tokens_total.labels(team=team, model=model, type="prompt").inc(prompt_tokens)
        if completion_tokens:
            self.tokens_total.labels(team=team, model=model, type="completion").inc(
                completion_tokens
            )

    def record_cost(self, *, team: str, model: str, cost_usd: float) -> None:
        if cost_usd:
            self.cost_usd_total.labels(team=team, model=model).inc(cost_usd)

    def record_rate_limit_hit(self, *, team: str) -> None:
        self.rate_limit_hits_total.labels(team=team).inc()

    def set_budget_utilisation(self, *, team: str, utilisation: float) -> None:
        self.budget_utilisation.labels(team=team).set(utilisation)

    def set_circuit_state(self, *, provider: str, model: str, state: CircuitBreakerState) -> None:
        self.circuit_breaker_state.labels(provider=provider, model=model).set(
            CIRCUIT_STATE_VALUES[state]
        )

    def record_fallback(self, *, from_model: str, to_model: str, reason: str) -> None:
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
        op = operation_type.value
        self.rag_requests_total.labels(team=team, operation_type=op).inc()
        if cost_usd:
            self.rag_cost_usd_total.labels(team=team, operation_type=op).inc(cost_usd)
        if prompt_tokens:
            self.rag_tokens_total.labels(
                team=team,
                operation_type=op,
                token_type="prompt",  # noqa: S106 — Prometheus label, not a credential
            ).inc(prompt_tokens)
        if completion_tokens:
            self.rag_tokens_total.labels(
                team=team,
                operation_type=op,
                token_type="completion",  # noqa: S106 — Prometheus label, not a credential
            ).inc(completion_tokens)
        self.rag_duration_seconds.labels(team=team, operation_type=op).observe(duration_s)

    def render(self) -> bytes:
        """Serialise the current state in Prometheus exposition format."""
        return generate_latest(self._registry)
