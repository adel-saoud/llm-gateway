"""In-memory rolling window of recent outcomes per ``(provider, model)``.

The window is intentionally in-memory: on process restart, failure-rate
accounting resets, but the *circuit state* (read from SQLite) is preserved.
A process that comes back up while a downstream is OPEN still skips it until
the open-duration elapses, then probes once via HALF_OPEN.

Outcomes are stored as booleans in a ``deque`` capped to ``window_size``; the
deque auto-evicts the oldest entry on append, so the failure-rate calculation
is always over the most recent traffic.
"""

from __future__ import annotations

from collections import deque

type _Outcome = bool
"""``True`` for a success, ``False`` for a failure."""


class FailureRateMonitor:
    """Per-(provider, model) rolling outcome window."""

    def __init__(self, *, window_size: int = 40) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._windows: dict[tuple[str, str], deque[_Outcome]] = {}

    def record_success(self, provider_name: str, model_id: str) -> None:
        """Append a success, evicting the oldest outcome once the window is full."""
        self._window(provider_name, model_id).append(True)

    def record_failure(self, provider_name: str, model_id: str) -> None:
        """Append a failure, evicting the oldest outcome once the window is full."""
        self._window(provider_name, model_id).append(False)

    def reset(self, provider_name: str, model_id: str) -> None:
        """Drop all history for one endpoint: used after a HALF_OPEN → CLOSED transition."""
        self._windows.pop((provider_name, model_id), None)

    def sample_count(self, provider_name: str, model_id: str) -> int:
        """Outcomes currently in the window, compared against ``min_requests``."""
        return len(self._windows.get((provider_name, model_id), ()))

    def failure_rate(self, provider_name: str, model_id: str) -> float:
        """Share of failures in the window; 0.0 for an endpoint with no history."""
        window = self._windows.get((provider_name, model_id))
        if not window:
            return 0.0
        failures = sum(1 for ok in window if not ok)
        return failures / len(window)

    def should_open(
        self,
        provider_name: str,
        model_id: str,
        *,
        threshold: float,
        min_requests: int,
    ) -> bool:
        """True iff we have enough samples AND failure rate ≥ threshold."""
        if self.sample_count(provider_name, model_id) < min_requests:
            return False
        return self.failure_rate(provider_name, model_id) >= threshold

    def _window(self, provider_name: str, model_id: str) -> deque[_Outcome]:
        key = (provider_name, model_id)
        existing = self._windows.get(key)
        if existing is not None:
            return existing
        new: deque[_Outcome] = deque(maxlen=self._window_size)
        self._windows[key] = new
        return new
