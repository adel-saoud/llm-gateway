"""Redis-backed token bucket — atomic Lua script.

A separate bucket per team is keyed under ``ratelimit:{team_id}``. The Lua
script is the only place that mutates bucket state, which gives us atomicity
without any application-side locking: the read, refill, and decrement happen
in a single round-trip, so two concurrent requests cannot both observe the
same pre-decrement balance.

State is laid out as a Redis hash so the script can read both fields with one
``HMGET`` and write both with one ``HSET``. The TTL on the key is set to
``2 × (capacity / refill_rate)`` seconds so idle buckets eventually evict
themselves without us tracking them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Final, cast

from redis.asyncio import Redis

# ── Lua script ──────────────────────────────────────────────────────────────
# KEYS[1]   bucket hash key
# ARGV[1]   capacity (int — max tokens the bucket can hold)
# ARGV[2]   refill_rate (float — tokens per second)
# ARGV[3]   now_ms (int — caller's clock, so the script doesn't depend on TIME)
# ARGV[4]   requested (int — tokens to consume)
#
# Returns: {allowed, remaining_tokens_x1000, retry_after_ms}
#   - allowed: 1 if requested tokens were consumed, 0 otherwise
#   - remaining_tokens_x1000: tokens left, scaled ×1000 to preserve fractions
#     across the integer Redis return type
#   - retry_after_ms: 0 when allowed, otherwise the ms until the bucket has
#     enough tokens for ``requested``
_LUA_SCRIPT: Final[str] = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate_per_s = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'last_refill_ms')
local tokens = tonumber(data[1])
local last_refill_ms = tonumber(data[2])

if tokens == nil then
  tokens = capacity
  last_refill_ms = now_ms
end

local elapsed_ms = math.max(0, now_ms - last_refill_ms)
local refill = (elapsed_ms / 1000.0) * rate_per_s
tokens = math.min(capacity, tokens + refill)

local ttl_s = math.max(1, math.ceil((capacity / rate_per_s) * 2))

if tokens >= requested then
  tokens = tokens - requested
  redis.call('HSET', key, 'tokens', tostring(tokens), 'last_refill_ms', tostring(now_ms))
  redis.call('EXPIRE', key, ttl_s)
  return {1, math.floor(tokens * 1000), 0}
else
  redis.call('HSET', key, 'tokens', tostring(tokens), 'last_refill_ms', tostring(now_ms))
  redis.call('EXPIRE', key, ttl_s)
  local needed = requested - tokens
  local retry_after_ms = math.ceil((needed / rate_per_s) * 1000)
  return {0, math.floor(tokens * 1000), retry_after_ms}
end
"""


@dataclass(frozen=True, slots=True)
class AcquireResult:
    """Outcome of a single ``acquire()`` call."""

    allowed: bool
    remaining: float
    retry_after_s: float

    @property
    def retry_after_seconds_ceiled(self) -> int:
        """Integer seconds, rounded up — the value used in the ``Retry-After`` header."""
        if self.retry_after_s <= 0.0:
            return 0
        return max(1, int(self.retry_after_s + 0.999))


class TokenBucket:
    """Per-team rate limiter."""

    KEY_PREFIX: Final[str] = "ratelimit:"

    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client
        self._script_sha: str | None = None

    async def _ensure_script_loaded(self) -> str:
        if self._script_sha is None:
            # redis-py's stubs don't expose script_load's true return shape.
            raw: object = await cast(Any, self._redis).script_load(_LUA_SCRIPT)
            # redis-py returns bytes on the async client; coerce to str for
            # consistent dict-key behaviour in tests across drivers.
            self._script_sha = raw.decode("ascii") if isinstance(raw, bytes) else str(raw)
        return self._script_sha

    async def acquire(
        self,
        team_id: str,
        *,
        capacity: int,
        refill_per_second: float,
        requested: int = 1,
    ) -> AcquireResult:
        """Try to consume ``requested`` tokens from the team's bucket atomically."""
        if requested < 1:
            raise ValueError(f"requested must be >= 1, got {requested}")
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if refill_per_second <= 0.0:
            raise ValueError(f"refill_per_second must be > 0, got {refill_per_second}")

        sha = await self._ensure_script_loaded()
        now_ms = int(time.time() * 1000)
        key = f"{self.KEY_PREFIX}{team_id}"
        # redis-py's type stub for evalsha returns `str` (the sha) rather than
        # the Lua result — that's wrong for our usage, so we erase the type
        # through Any before awaiting.
        result: object = await cast(Any, self._redis).evalsha(
            sha,
            1,
            key,
            str(capacity),
            str(refill_per_second),
            str(now_ms),
            str(requested),
        )
        if not isinstance(result, list):
            raise RuntimeError(f"redis evalsha returned non-list: {type(result).__name__}")
        triple = cast(list[object], result)
        allowed = _to_int(triple[0]) == 1
        remaining = _to_int(triple[1]) / 1000.0
        retry_after_s = _to_int(triple[2]) / 1000.0
        return AcquireResult(
            allowed=allowed,
            remaining=remaining,
            retry_after_s=retry_after_s,
        )


def _to_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, bytes):
        return int(value.decode("ascii"))
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"cannot coerce {type(value).__name__} to int")
