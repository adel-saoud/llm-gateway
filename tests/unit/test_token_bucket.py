"""Token bucket tests — hermetic via fakeredis (in-process Redis emulator)."""

from __future__ import annotations

import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from llm_gateway.ratelimit.token_bucket import TokenBucket


@pytest.fixture
def redis_client() -> FakeAsyncRedis:
    return FakeAsyncRedis()


@pytest.fixture
def bucket(redis_client: FakeAsyncRedis) -> TokenBucket:
    return TokenBucket(redis_client)  # type: ignore[arg-type]


async def test_first_acquire_allowed_with_full_capacity(bucket: TokenBucket) -> None:
    result = await bucket.acquire("t1", capacity=10, refill_per_second=1.0)
    assert result.allowed is True
    assert result.remaining == pytest.approx(9.0)  # pyright: ignore[reportUnknownMemberType]
    assert result.retry_after_s == 0.0


async def test_bucket_drains_then_rejects(bucket: TokenBucket) -> None:
    for _ in range(5):
        result = await bucket.acquire("t1", capacity=5, refill_per_second=0.001)
        assert result.allowed is True
    # 6th call has no tokens (refill rate is tiny)
    result = await bucket.acquire("t1", capacity=5, refill_per_second=0.001)
    assert result.allowed is False
    assert result.retry_after_s > 0.0
    assert result.retry_after_seconds_ceiled >= 1


async def test_refill_restores_tokens(bucket: TokenBucket) -> None:
    # Drain the bucket
    for _ in range(3):
        await bucket.acquire("t1", capacity=3, refill_per_second=1000.0)
    # High refill rate — after a short sleep, tokens are back
    await asyncio.sleep(0.05)
    result = await bucket.acquire("t1", capacity=3, refill_per_second=1000.0)
    assert result.allowed is True


async def test_independent_teams_have_independent_buckets(bucket: TokenBucket) -> None:
    # Drain t1
    for _ in range(3):
        await bucket.acquire("t1", capacity=3, refill_per_second=0.001)
    # t2 still has its full capacity
    result = await bucket.acquire("t2", capacity=3, refill_per_second=0.001)
    assert result.allowed is True
    assert result.remaining == pytest.approx(2.0)  # pyright: ignore[reportUnknownMemberType]


async def test_concurrent_acquires_are_atomic(bucket: TokenBucket) -> None:
    """Hammer the bucket with concurrent requests — total allowed must equal capacity."""
    capacity = 20
    tasks = [
        bucket.acquire("t_concurrent", capacity=capacity, refill_per_second=0.0001)
        for _ in range(40)
    ]
    results = await asyncio.gather(*tasks)
    allowed_count = sum(1 for r in results if r.allowed)
    # With a near-zero refill rate, exactly `capacity` (give or take 1 for the
    # tiny refill during the test) should have been allowed.
    assert capacity <= allowed_count <= capacity + 1


async def test_acquire_validates_arguments(bucket: TokenBucket) -> None:
    with pytest.raises(ValueError, match="requested must be"):
        await bucket.acquire("t", capacity=10, refill_per_second=1.0, requested=0)
    with pytest.raises(ValueError, match="capacity must be"):
        await bucket.acquire("t", capacity=0, refill_per_second=1.0)
    with pytest.raises(ValueError, match="refill_per_second must be"):
        await bucket.acquire("t", capacity=1, refill_per_second=0.0)


async def test_retry_after_seconds_ceiled_rounds_up() -> None:
    from llm_gateway.ratelimit.token_bucket import AcquireResult

    assert (
        AcquireResult(allowed=True, remaining=0.0, retry_after_s=0.0).retry_after_seconds_ceiled
        == 0
    )
    assert (
        AcquireResult(allowed=False, remaining=0.0, retry_after_s=0.1).retry_after_seconds_ceiled
        == 1
    )
    assert (
        AcquireResult(allowed=False, remaining=0.0, retry_after_s=1.5).retry_after_seconds_ceiled
        == 2
    )
    assert (
        AcquireResult(allowed=False, remaining=0.0, retry_after_s=10.0).retry_after_seconds_ceiled
        == 10
    )
