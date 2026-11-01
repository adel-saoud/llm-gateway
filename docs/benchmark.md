# Benchmark: gateway overhead

What the gateway adds on top of the provider, measured with [`scripts/bench_overhead.py`](../scripts/bench_overhead.py).

**Method**

- **Provider:** an Ollama-compatible stub ([`bench/stub_ollama.py`](../bench/stub_ollama.py)) that answers instantly, so the numbers are the gateway, not a model.
- **Gateway:** a separate `uvicorn` process, 1 worker, uvloop, log level WARNING, real Redis, every gateway feature on: auth, Redis token bucket, budget check and cost-ledger write in SQLite, Prometheus metrics.
- **Sequential runs:** one keep-alive client, 200 warmup requests, then 2,000 requests per scenario, repeated 5 times. Two numbers per request: the client round trip, and the gateway's own time from its `Server-Timing` header (total minus the provider call). Values below are the median of the 5 runs.
- **Open-loop runs:** k6 (`grafana/k6:1.8.2` in Docker), constant arrival rate for 30 s per rate. Requests start on schedule whatever the response time, so a slow gateway is measured under the load it was asked to take.

**Hardware:** Apple M4 Pro (14 cores), 24 GB, macOS 27.0.1, Python 3.14.4, fastapi 0.136.1, uvicorn 0.47.0, uvloop 0.22.1, redis-py 7.4.0. Load generator, stub, gateway and Redis share the machine, and nothing else ran during the measurement. Redis runs in Docker: PING p50 is 0.29 ms, p99 1.17 ms, and every request pays at least one of those for rate limiting.

**Sequential (ms)**

| Scenario | Gateway time p50 / p95 / p99 | Round trip p50 / p95 / p99 |
|:--|:--|:--|
| Stub called directly (baseline) | n/a | 0.23 / 0.31 / 0.43 |
| Gateway, cache off | **2.14 / 3.03 / 4.08** | 2.98 / 4.40 / 5.84 |
| Gateway, exact cache miss | 2.02 / 2.79 / 3.42 | 2.88 / 3.89 / 4.77 |
| Gateway, exact cache hit | 2.33 / 2.66 / 3.17 | 2.70 / 3.10 / 3.59 |
| Gateway, semantic cache miss (embed + search over 1,000 entries of 1,024 dimensions) | 4.24 / 5.26 / 6.81 | 5.12 / 6.40 / 8.55 |

Across the 5 runs the cache-off gateway p50 ranged from 1.88 to 2.36 ms, and the exact-hit p50 from 1.64 to 2.36 ms: the hit's median landing above the miss's is run-to-run noise at this scale, not a slower path.

**Open loop, cache off (ms)**

| Arrival rate | Gateway time p50 / p95 / p99 | Round trip p50 / p95 / p99 | Failed | Not started (k6 `dropped_iterations`) |
|--:|:--|:--|--:|--:|
| 100 req/s | 3.84 / 4.60 / 5.31 | 5.44 / 6.66 / 7.75 | 0% | 0 |
| 250 req/s | 2.88 / 4.12 / 6.26 | 3.90 / 5.76 / 8.95 | 0% | 0 |
| 500 req/s | 29 / 2,521 / 4,458 | 36 / 3,275 / 5,240 | 2.1% | 1,085 |

One worker holds 250 req/s with a 6 ms p99 and saturates before 500 req/s: it serves about 460 req/s, latency climbs to seconds, and 2.1% of requests fail with `database is locked` from the SQLite cost ledger, which takes one writer at a time.

**What the SQLite settings changed.** The first run of this layout opened a connection per call on a rollback-journal file: at 500 req/s the worker topped out near 300 req/s with 8.8% of requests failing on the lock, and the 250 req/s p99 was 31 ms. WAL mode lets the per-request budget read run alongside the ledger write, and the gateway holds one connection open for its lifetime, because SQLite otherwise checkpoints and deletes the WAL each time the last connection closes, which alone doubled the sequential cost. Measured on this machine, one budget read plus one ledger write takes 0.88 ms with the rollback journal, 1.73 ms with WAL and per-call connections, and 0.59 ms with WAL held open. Past about 460 req/s the single writer is the limit: batched ledger writes or a server database would be the next step (see [Limits](../README.md#limits)).

Raw results: [`results/raw/overhead.json`](../results/raw/overhead.json). Environment, model digests and the per-scenario sample sizes: [`results/manifest.json`](../results/manifest.json). Both are written by `scripts/reproduce.py` ([development](development.md#reproducing-the-readme-numbers)).
