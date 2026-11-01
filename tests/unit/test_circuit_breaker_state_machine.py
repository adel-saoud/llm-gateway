"""State-machine and persistence tests for ``CircuitBreaker``.

Time is injected via the ``now`` kwarg on every call: no ``asyncio.sleep`` and
no flaky timing in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from llm_gateway.circuit_breaker.monitor import FailureRateMonitor
from llm_gateway.circuit_breaker.state_machine import CircuitBreaker, CircuitBreakerState
from llm_gateway.storage.db import CircuitStateRepository, CostLedger


@dataclass(frozen=True)
class _Setup:
    breaker: CircuitBreaker
    repo: CircuitStateRepository
    monitor: FailureRateMonitor
    db_path: Path


async def _build(
    tmp_path: Path,
    *,
    threshold: float = 0.5,
    min_requests: int = 5,
    open_duration_s: float = 30.0,
    window_size: int = 20,
) -> _Setup:
    db_path = tmp_path / "cb.db"
    ledger = CostLedger(db_path)
    await ledger.initialize()
    repo = CircuitStateRepository(db_path)
    monitor = FailureRateMonitor(window_size=window_size)
    breaker = CircuitBreaker(
        repo,
        monitor,
        failure_rate_threshold=threshold,
        min_requests=min_requests,
        open_duration_s=open_duration_s,
    )
    return _Setup(breaker=breaker, repo=repo, monitor=monitor, db_path=db_path)


# ── Initial state ──────────────────────────────────────────────────────────


async def test_unknown_endpoint_is_closed(tmp_path: Path) -> None:
    s = await _build(tmp_path)
    assert await s.breaker.current_state("p", "m") is CircuitBreakerState.CLOSED
    assert await s.breaker.is_allowed("p", "m") is True


# ── CLOSED → OPEN ──────────────────────────────────────────────────────────


async def test_below_min_requests_stays_closed(tmp_path: Path) -> None:
    s = await _build(tmp_path, min_requests=5)
    for _ in range(4):
        await s.breaker.record_failure("p", "m")
    assert await s.breaker.current_state("p", "m") is CircuitBreakerState.CLOSED


async def test_crossing_threshold_opens(tmp_path: Path) -> None:
    s = await _build(tmp_path, threshold=0.5, min_requests=5)
    for _ in range(10):
        await s.breaker.record_failure("p", "m")
    assert await s.breaker.current_state("p", "m") is CircuitBreakerState.OPEN


async def test_below_threshold_stays_closed(tmp_path: Path) -> None:
    s = await _build(tmp_path, threshold=0.5, min_requests=5)
    for _ in range(8):
        await s.breaker.record_success("p", "m")
    for _ in range(2):
        await s.breaker.record_failure("p", "m")
    assert await s.breaker.current_state("p", "m") is CircuitBreakerState.CLOSED


# ── OPEN denies requests ───────────────────────────────────────────────────


async def test_open_circuit_denies(tmp_path: Path) -> None:
    s = await _build(tmp_path)
    for _ in range(10):
        await s.breaker.record_failure("p", "m")
    assert await s.breaker.is_allowed("p", "m") is False


# ── OPEN → HALF_OPEN after duration ────────────────────────────────────────


async def test_open_transitions_to_half_open_after_duration(tmp_path: Path) -> None:
    s = await _build(tmp_path, open_duration_s=30.0)
    opened_at = datetime(2026, 5, 23, 12, 0, 0, tzinfo=UTC)
    for _ in range(10):
        await s.breaker.record_failure("p", "m", now=opened_at)
    assert await s.breaker.current_state("p", "m") is CircuitBreakerState.OPEN

    # 29s later: still OPEN
    too_soon = opened_at + timedelta(seconds=29)
    assert await s.breaker.is_allowed("p", "m", now=too_soon) is False
    assert await s.breaker.current_state("p", "m") is CircuitBreakerState.OPEN

    # 30s+ later: HALF_OPEN, probe allowed
    elapsed = opened_at + timedelta(seconds=31)
    assert await s.breaker.is_allowed("p", "m", now=elapsed) is True
    assert await s.breaker.current_state("p", "m") is CircuitBreakerState.HALF_OPEN


# ── HALF_OPEN → CLOSED on success, OPEN on failure ─────────────────────────


async def test_half_open_success_closes(tmp_path: Path) -> None:
    s = await _build(tmp_path, open_duration_s=10.0)
    t0 = datetime(2026, 5, 23, tzinfo=UTC)
    for _ in range(10):
        await s.breaker.record_failure("p", "m", now=t0)
    later = t0 + timedelta(seconds=20)
    await s.breaker.is_allowed("p", "m", now=later)  # transitions to HALF_OPEN
    await s.breaker.record_success("p", "m", now=later)
    assert await s.breaker.current_state("p", "m") is CircuitBreakerState.CLOSED
    # Monitor was reset: next is_allowed lets traffic through and the window
    # has no failures recorded against the now-closed circuit.
    assert s.monitor.sample_count("p", "m") == 0


async def test_half_open_failure_reopens(tmp_path: Path) -> None:
    s = await _build(tmp_path, open_duration_s=10.0)
    t0 = datetime(2026, 5, 23, tzinfo=UTC)
    for _ in range(10):
        await s.breaker.record_failure("p", "m", now=t0)
    later = t0 + timedelta(seconds=20)
    await s.breaker.is_allowed("p", "m", now=later)  # HALF_OPEN
    even_later = later + timedelta(seconds=1)
    await s.breaker.record_failure("p", "m", now=even_later)
    assert await s.breaker.current_state("p", "m") is CircuitBreakerState.OPEN


# ── Persistence: survives restart ─────────────────────────────────────────


async def test_state_persists_across_breaker_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "cb.db"
    ledger = CostLedger(db_path)
    await ledger.initialize()

    repo1 = CircuitStateRepository(db_path)
    cb1 = CircuitBreaker(
        repo1,
        FailureRateMonitor(window_size=20),
        failure_rate_threshold=0.5,
        min_requests=5,
        open_duration_s=30.0,
    )
    for _ in range(10):
        await cb1.record_failure("ollama", "x")
    assert await cb1.current_state("ollama", "x") is CircuitBreakerState.OPEN

    # Simulate a process restart: fresh objects, same DB.
    repo2 = CircuitStateRepository(db_path)
    cb2 = CircuitBreaker(
        repo2,
        FailureRateMonitor(window_size=20),  # fresh in-memory window
        failure_rate_threshold=0.5,
        min_requests=5,
        open_duration_s=30.0,
    )
    assert await cb2.current_state("ollama", "x") is CircuitBreakerState.OPEN


# ── Independent endpoints ──────────────────────────────────────────────────


async def test_endpoints_are_independent(tmp_path: Path) -> None:
    s = await _build(tmp_path)
    for _ in range(10):
        await s.breaker.record_failure("p1", "m")
    assert await s.breaker.current_state("p1", "m") is CircuitBreakerState.OPEN
    # p2 untouched
    assert await s.breaker.current_state("p2", "m") is CircuitBreakerState.CLOSED


# ── Constructor validation ─────────────────────────────────────────────────


async def test_invalid_threshold_rejected(tmp_path: Path) -> None:
    s = await _build(tmp_path)
    with pytest.raises(ValueError, match="failure_rate_threshold"):
        CircuitBreaker(
            s.repo,
            s.monitor,
            failure_rate_threshold=1.5,
            min_requests=1,
            open_duration_s=1.0,
        )


async def test_invalid_min_requests_rejected(tmp_path: Path) -> None:
    s = await _build(tmp_path)
    with pytest.raises(ValueError, match="min_requests"):
        CircuitBreaker(
            s.repo,
            s.monitor,
            failure_rate_threshold=0.5,
            min_requests=0,
            open_duration_s=1.0,
        )


async def test_invalid_open_duration_rejected(tmp_path: Path) -> None:
    s = await _build(tmp_path)
    with pytest.raises(ValueError, match="open_duration_s"):
        CircuitBreaker(
            s.repo,
            s.monitor,
            failure_rate_threshold=0.5,
            min_requests=1,
            open_duration_s=0.0,
        )
