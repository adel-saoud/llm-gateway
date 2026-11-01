---
status: accepted
date: 2026-10-31
---

# SQLite cost ledger in WAL mode with one held connection

## Context and Problem Statement

Every request reads the team's spend for the day (budget check) and writes one row to
the cost ledger. The ledger and the circuit state live in one SQLite file
([`storage/db.py`](../../src/llm_gateway/storage/db.py)). The first open-loop run opened
a connection per call on a rollback-journal file: one worker topped out near 300 req/s,
8.8% of requests failed with `database is locked` at 500 req/s, and the 250 req/s p99
was 31 ms ([`docs/benchmark.md`](../benchmark.md), [`CHANGELOG.md`](../../CHANGELOG.md)
Unreleased, Fixed). How should the gateway open SQLite so the per-request read and write
stop blocking each other, without adding a server to run?

## Decision Drivers

* No new infrastructure for a single-node deployment.
* Budget reads must not wait behind ledger writes.
* A lock wait must be stated in one place, not left to each connection's default.

## Considered Options

* Keep the rollback journal and per-call connections
* WAL mode with per-call connections
* WAL mode with one connection held open for the gateway's lifetime
* Move the ledger to a server database (Postgres)

## Decision Outcome

Chosen option: "WAL mode with one connection held open for the gateway's lifetime".
`initialize()` sets `PRAGMA journal_mode=WAL`, every connection goes through `_connect()`
with `BUSY_TIMEOUT_S = 5.0`, and the app lifespan holds `CostLedger.keep_wal_open()`.
The held connection is what makes WAL pay off: when the last connection closes, SQLite
checkpoints and deletes the WAL, so per-call connections paid that on every request.
Measured on the benchmark machine, one budget read plus one ledger write took 0.88 ms with
the rollback journal, 1.73 ms with WAL and per-call connections, and 0.59 ms with WAL held
open ([`docs/benchmark.md`](../benchmark.md)).

### Consequences

* Good, because 250 req/s now runs with a 6 ms gateway p99 and 0 failures
  (`results/manifest.json`: `metrics.overhead.open_loop[1]`, `gateway_overhead_ms_p99`
  6.256, `failed_rate` 0).
* Good, because at 500 req/s failures fell from 8.8% to 2.1%
  (`metrics.overhead.open_loop[2].failed_rate` 0.0209) and one worker serves about 460
  req/s ([`docs/benchmark.md`](../benchmark.md)).
* Bad, because SQLite still takes one writer at a time: past about 460 req/s, latency
  climbs to seconds (`gateway_overhead_ms_p99` 4,458 at 500 req/s) and requests fail on
  the lock. README Limits names the next steps: batch ledger writes, or move the ledger to
  Postgres.
* Bad, because the held connection is one more resource the lifespan must close;
  aiosqlite runs non-daemon threads, so an unclosed connection hangs process exit, which is
  why it is a context manager and not part of `initialize()`.
* Neutral: the benchmark uses an instant stub provider, so these numbers are the gateway's
  own cost, not a model's ([`docs/benchmark.md`](../benchmark.md)).

## Pros and Cons of the Options

### WAL mode with per-call connections

* Good, because reads run alongside the writer.
* Bad, because it was slower than the rollback journal in sequential use (1.73 against
  0.88 ms), from the checkpoint and delete on every last close.

### Move the ledger to Postgres

* Good, because it removes the single-writer ceiling.
* Bad, because a portfolio-scale single node would need a second server to run and
  operate, for load it has not seen.
