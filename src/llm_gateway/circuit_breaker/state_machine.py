"""Three-state circuit breaker — CLOSED / OPEN / HALF_OPEN — per (provider, model).

State transitions:

* **CLOSED → OPEN** when the failure rate over the most recent traffic crosses
  ``failure_rate_threshold`` AND at least ``min_requests`` samples are
  available. Below ``min_requests`` we don't act — small-sample failure rates
  are too noisy.
* **OPEN → HALF_OPEN** automatically once ``open_duration_s`` has elapsed
  since the breaker opened. The next request is allowed through as a probe.
* **HALF_OPEN → CLOSED** on a successful probe; the rolling window is reset
  so prior failures don't immediately re-open the breaker.
* **HALF_OPEN → OPEN** on a failed probe; the open timer restarts.

State is persisted via :class:`~llm_gateway.storage.db.CircuitStateRepository`
so the breaker's verdict survives process restart. The failure-rate window
is in-memory only — see :mod:`llm_gateway.circuit_breaker.monitor`.

Caller pattern::

    if await breaker.is_allowed(provider, model):
        try:
            result = await provider.chat(model, ...)
        except (ProviderUnavailableError, ProviderTimeoutError):
            await breaker.record_failure(provider, model)
            raise
        await breaker.record_success(provider, model)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TypeAlias

import structlog

from llm_gateway.circuit_breaker.monitor import FailureRateMonitor
from llm_gateway.storage.db import CircuitStateRecord, CircuitStateRepository

_log = structlog.get_logger(__name__)


class CircuitBreakerState(StrEnum):
    """Three CB states; stored as strings in SQLite."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


OnStateChange: TypeAlias = Callable[[str, str, "CircuitBreakerState"], None]


class CircuitBreaker:
    """Per-(provider, model) circuit breaker with SQLite persistence."""

    def __init__(
        self,
        repo: CircuitStateRepository,
        monitor: FailureRateMonitor,
        *,
        failure_rate_threshold: float,
        min_requests: int,
        open_duration_s: float,
        on_state_change: OnStateChange | None = None,
    ) -> None:
        if not 0.0 <= failure_rate_threshold <= 1.0:
            raise ValueError(
                f"failure_rate_threshold must be in [0, 1], got {failure_rate_threshold}"
            )
        if min_requests < 1:
            raise ValueError(f"min_requests must be >= 1, got {min_requests}")
        if open_duration_s <= 0.0:
            raise ValueError(f"open_duration_s must be > 0, got {open_duration_s}")
        self._repo = repo
        self._monitor = monitor
        self._threshold = failure_rate_threshold
        self._min_requests = min_requests
        self._open_duration = timedelta(seconds=open_duration_s)
        self._on_state_change = on_state_change

    async def is_allowed(
        self,
        provider_name: str,
        model_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Check whether a request to ``(provider, model)`` should proceed.

        ``OPEN`` returns False unless the open-duration has elapsed, in which
        case we transition to ``HALF_OPEN`` and allow the probe.
        """
        when = now or datetime.now(UTC)
        record = await self._repo.get(provider_name, model_id)
        if record is None:
            return True

        state = CircuitBreakerState(record.state)
        if state is CircuitBreakerState.CLOSED:
            return True
        if state is CircuitBreakerState.OPEN:
            if record.opened_at is not None and (when - record.opened_at) >= self._open_duration:
                await self._transition_to(
                    provider_name, model_id, CircuitBreakerState.HALF_OPEN, when=when
                )
                return True
            return False
        # HALF_OPEN — allow the probe; the next record_success / record_failure
        # call decides the next state.
        return True

    async def record_success(
        self,
        provider_name: str,
        model_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        when = now or datetime.now(UTC)
        self._monitor.record_success(provider_name, model_id)
        state = await self._current_state(provider_name, model_id)
        if state is CircuitBreakerState.HALF_OPEN:
            await self._transition_to(
                provider_name, model_id, CircuitBreakerState.CLOSED, when=when
            )
            self._monitor.reset(provider_name, model_id)

    async def record_failure(
        self,
        provider_name: str,
        model_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        when = now or datetime.now(UTC)
        self._monitor.record_failure(provider_name, model_id)
        state = await self._current_state(provider_name, model_id)
        # Both paths transition to OPEN: a failed HALF_OPEN probe (we re-open
        # immediately) or a CLOSED endpoint whose failure rate just crossed
        # the threshold.
        should_open = state is CircuitBreakerState.HALF_OPEN or (
            state is CircuitBreakerState.CLOSED
            and self._monitor.should_open(
                provider_name,
                model_id,
                threshold=self._threshold,
                min_requests=self._min_requests,
            )
        )
        if should_open:
            await self._transition_to(provider_name, model_id, CircuitBreakerState.OPEN, when=when)

    async def current_state(self, provider_name: str, model_id: str) -> CircuitBreakerState:
        return await self._current_state(provider_name, model_id)

    async def _current_state(self, provider_name: str, model_id: str) -> CircuitBreakerState:
        record = await self._repo.get(provider_name, model_id)
        if record is None:
            return CircuitBreakerState.CLOSED
        return CircuitBreakerState(record.state)

    async def _transition_to(
        self,
        provider_name: str,
        model_id: str,
        new_state: CircuitBreakerState,
        *,
        when: datetime,
    ) -> None:
        opened_at = when if new_state is CircuitBreakerState.OPEN else None
        await self._repo.upsert(
            CircuitStateRecord(
                provider_name=provider_name,
                model_id=model_id,
                state=new_state.value,
                opened_at=opened_at,
                updated_at=when,
            )
        )
        _log.info(
            "circuit_state_transition",
            provider=provider_name,
            model=model_id,
            new_state=new_state.value,
        )
        if self._on_state_change is not None:
            self._on_state_change(provider_name, model_id, new_state)
