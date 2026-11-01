from __future__ import annotations

import pytest

from llm_gateway.circuit_breaker.monitor import FailureRateMonitor


def test_empty_window_zero_failure_rate() -> None:
    monitor = FailureRateMonitor()
    assert monitor.failure_rate("p", "m") == 0.0
    assert monitor.sample_count("p", "m") == 0


def test_all_failures_one_hundred_percent() -> None:
    monitor = FailureRateMonitor()
    for _ in range(5):
        monitor.record_failure("p", "m")
    assert monitor.failure_rate("p", "m") == 1.0
    assert monitor.sample_count("p", "m") == 5


def test_mixed_outcomes() -> None:
    monitor = FailureRateMonitor()
    monitor.record_success("p", "m")
    monitor.record_failure("p", "m")
    monitor.record_failure("p", "m")
    monitor.record_success("p", "m")
    assert monitor.failure_rate("p", "m") == 0.5


def test_window_evicts_oldest() -> None:
    monitor = FailureRateMonitor(window_size=3)
    monitor.record_failure("p", "m")
    monitor.record_failure("p", "m")
    monitor.record_failure("p", "m")
    # 3 failures: 100%
    assert monitor.failure_rate("p", "m") == 1.0
    # Next success evicts the oldest failure → 2 failures + 1 success
    monitor.record_success("p", "m")
    assert monitor.failure_rate("p", "m") == pytest.approx(2 / 3)  # pyright: ignore[reportUnknownMemberType]
    assert monitor.sample_count("p", "m") == 3


def test_independent_keys() -> None:
    monitor = FailureRateMonitor()
    monitor.record_failure("p1", "m")
    monitor.record_success("p2", "m")
    assert monitor.failure_rate("p1", "m") == 1.0
    assert monitor.failure_rate("p2", "m") == 0.0


def test_should_open_requires_minimum_samples() -> None:
    monitor = FailureRateMonitor()
    for _ in range(3):
        monitor.record_failure("p", "m")
    # 3 samples < min_requests=5 → no open
    assert monitor.should_open("p", "m", threshold=0.5, min_requests=5) is False


def test_should_open_when_threshold_crossed() -> None:
    monitor = FailureRateMonitor()
    for _ in range(10):
        monitor.record_failure("p", "m")
    assert monitor.should_open("p", "m", threshold=0.5, min_requests=5) is True


def test_should_open_below_threshold() -> None:
    monitor = FailureRateMonitor()
    for _ in range(8):
        monitor.record_success("p", "m")
    for _ in range(2):
        monitor.record_failure("p", "m")
    # 2/10 = 0.2 < 0.5
    assert monitor.should_open("p", "m", threshold=0.5, min_requests=5) is False


def test_reset_clears_history() -> None:
    monitor = FailureRateMonitor()
    for _ in range(5):
        monitor.record_failure("p", "m")
    assert monitor.sample_count("p", "m") == 5
    monitor.reset("p", "m")
    assert monitor.sample_count("p", "m") == 0
    assert monitor.failure_rate("p", "m") == 0.0


def test_window_size_validated() -> None:
    with pytest.raises(ValueError, match="window_size"):
        FailureRateMonitor(window_size=0)
