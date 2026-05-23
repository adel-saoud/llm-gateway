"""Unit tests for the :class:`Metrics` facade.

We construct a fresh :class:`prometheus_client.CollectorRegistry` per test so
counters don't leak between tests, then either inspect the values directly via
``Collector.collect()`` or assert on the text exposition body.
"""

from __future__ import annotations

import pytest

from llm_gateway.circuit_breaker.state_machine import CircuitBreakerState
from llm_gateway.observability.metrics import (
    CIRCUIT_STATE_VALUES,
    METRICS_CONTENT_TYPE,
    Metrics,
)
from llm_gateway.rag.models import RagOperationType


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()


# ── HTTP-level ─────────────────────────────────────────────────────────────


def test_record_http_request_increments_counter_and_observes_duration(
    metrics: Metrics,
) -> None:
    metrics.record_http_request(team="t1", status=200, duration_s=0.05, model="m")
    body = metrics.render().decode("utf-8")
    assert 'llm_gateway_requests_total{model="m",status="200",team="t1"} 1.0' in body
    assert 'llm_gateway_request_duration_seconds_count{model="m",team="t1"} 1.0' in body


def test_status_label_is_string_form_of_int(metrics: Metrics) -> None:
    metrics.record_http_request(team="t1", status=429, duration_s=0.01)
    body = metrics.render().decode("utf-8")
    assert 'status="429"' in body


def test_default_model_label_is_empty_string(metrics: Metrics) -> None:
    metrics.record_http_request(team="t1", status=401, duration_s=0.01)
    body = metrics.render().decode("utf-8")
    assert 'model=""' in body


# ── Tokens / cost ──────────────────────────────────────────────────────────


def test_record_tokens_splits_prompt_and_completion(metrics: Metrics) -> None:
    metrics.record_tokens(team="t1", model="m", prompt_tokens=100, completion_tokens=50)
    body = metrics.render().decode("utf-8")
    assert 'llm_gateway_tokens_total{model="m",team="t1",type="prompt"} 100.0' in body
    assert 'llm_gateway_tokens_total{model="m",team="t1",type="completion"} 50.0' in body


def test_record_tokens_skips_zero(metrics: Metrics) -> None:
    metrics.record_tokens(team="t1", model="m", prompt_tokens=10, completion_tokens=0)
    body = metrics.render().decode("utf-8")
    assert 'type="prompt"' in body
    # No completion label since count was 0
    assert 'type="completion"' not in body


def test_record_cost_accumulates(metrics: Metrics) -> None:
    metrics.record_cost(team="t1", model="m", cost_usd=0.5)
    metrics.record_cost(team="t1", model="m", cost_usd=0.25)
    body = metrics.render().decode("utf-8")
    assert 'llm_gateway_cost_usd_total{model="m",team="t1"} 0.75' in body


def test_record_cost_skips_zero(metrics: Metrics) -> None:
    metrics.record_cost(team="t1", model="m", cost_usd=0.0)
    body = metrics.render().decode("utf-8")
    # The counter's HELP/TYPE lines may still appear because the collector is
    # registered; we only care that no data line was emitted.
    assert "llm_gateway_cost_usd_total{" not in body


# ── Rate limit / budget / circuit ──────────────────────────────────────────


def test_rate_limit_hit_counter(metrics: Metrics) -> None:
    metrics.record_rate_limit_hit(team="t1")
    body = metrics.render().decode("utf-8")
    assert 'llm_gateway_rate_limit_hits_total{team="t1"} 1.0' in body


def test_budget_utilisation_gauge(metrics: Metrics) -> None:
    metrics.set_budget_utilisation(team="t1", utilisation=0.85)
    body = metrics.render().decode("utf-8")
    assert 'llm_gateway_budget_utilisation{team="t1"} 0.85' in body
    # Setting again replaces the value (gauge semantics)
    metrics.set_budget_utilisation(team="t1", utilisation=0.95)
    body = metrics.render().decode("utf-8")
    assert 'llm_gateway_budget_utilisation{team="t1"} 0.95' in body


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (CircuitBreakerState.CLOSED, 0.0),
        (CircuitBreakerState.HALF_OPEN, 1.0),
        (CircuitBreakerState.OPEN, 2.0),
    ],
)
def test_circuit_state_gauge_encoding(
    metrics: Metrics, state: CircuitBreakerState, expected: float
) -> None:
    metrics.set_circuit_state(provider="ollama", model="m", state=state)
    body = metrics.render().decode("utf-8")
    assert f'llm_gateway_circuit_breaker_state{{model="m",provider="ollama"}} {expected}' in body


def test_circuit_state_values_table_exhaustive() -> None:
    assert set(CIRCUIT_STATE_VALUES) == set(CircuitBreakerState)


def test_fallback_total_labels(metrics: Metrics) -> None:
    metrics.record_fallback(from_model="m1", to_model="m2", reason="circuit_open")
    body = metrics.render().decode("utf-8")
    assert (
        'llm_gateway_fallback_total{from_model="m1",reason="circuit_open",to_model="m2"} 1.0'
        in body
    )


# ── RAG-specific ───────────────────────────────────────────────────────────


def test_record_rag_event_increments_all_dimensions(metrics: Metrics) -> None:
    metrics.record_rag_event(
        team="t1",
        operation_type=RagOperationType.GENERATION,
        cost_usd=0.12,
        prompt_tokens=200,
        completion_tokens=80,
        duration_s=0.5,
    )
    body = metrics.render().decode("utf-8")
    prompt_line = (
        'llm_gateway_rag_tokens_total{operation_type="generation",'
        'team="t1",token_type="prompt"} 200.0'
    )
    completion_line = (
        'llm_gateway_rag_tokens_total{operation_type="generation",'
        'team="t1",token_type="completion"} 80.0'
    )
    duration_line = (
        'llm_gateway_rag_duration_seconds_count{operation_type="generation",team="t1"} 1.0'
    )
    assert 'llm_gateway_rag_requests_total{operation_type="generation",team="t1"} 1.0' in body
    assert 'llm_gateway_rag_cost_usd_total{operation_type="generation",team="t1"} 0.12' in body
    assert prompt_line in body
    assert completion_line in body
    assert duration_line in body


def test_render_returns_bytes_in_prom_format(metrics: Metrics) -> None:
    metrics.record_http_request(team="t1", status=200, duration_s=0.01)
    body = metrics.render()
    assert isinstance(body, bytes)
    assert b"# HELP llm_gateway_requests_total" in body
    assert b"# TYPE llm_gateway_requests_total counter" in body


def test_metrics_content_type_is_prom_text() -> None:
    assert METRICS_CONTENT_TYPE.startswith("text/plain")


def test_each_metrics_instance_has_isolated_registry() -> None:
    m1 = Metrics()
    m2 = Metrics()
    m1.record_http_request(team="t1", status=200, duration_s=0.01)
    body2 = m2.render().decode("utf-8")
    # m2 has no observations — counter should be absent
    assert 'llm_gateway_requests_total{model="",status="200",team="t1"} 1.0' not in body2
