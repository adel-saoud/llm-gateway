"""Measure the latency the gateway adds on top of the provider.

Method (details in docs/benchmark.md):

* An Ollama-compatible stub (``bench/stub_ollama.py``) answers instantly, so
  the provider's own time is near zero.
* The gateway runs as a separate uvicorn process (1 worker, uvloop, logs at
  WARNING) against real Redis, with auth, rate limiting, budget checks, cost
  ledger writes and Prometheus metrics all on.
* **Sequential:** one keep-alive client sends N requests per scenario after a
  warmup, repeated R times. Two numbers per request: client round trip, and
  the gateway's own time from its ``Server-Timing`` header (total minus the
  provider call). The client-side overhead is the gateway scenario minus the
  same client calling the stub directly.
* **Open loop (optional):** k6's constant-arrival-rate executor, run from the
  ``grafana/k6`` Docker image, sends requests at fixed rates regardless of
  response time, so tail latency under load is not hidden by a client that
  slows down with the server.

Prerequisite: Redis on ``--redis-url`` (``docker compose up -d redis``).

Usage::

    uv run python scripts/bench_overhead.py
    uv run python scripts/bench_overhead.py --open-loop --rates 100,250,500
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from llm_gateway.config import Settings

ROOT = Path(__file__).resolve().parent.parent
STUB_PORT = 18080
GATEWAY_PORT = 18000
STUB_URL = f"http://127.0.0.1:{STUB_PORT}"
GATEWAY_URL = f"http://127.0.0.1:{GATEWAY_PORT}"
K6_IMAGE = "grafana/k6:1.8.2"
# The stub answers any id; the gateway's own budget model keeps the request
# identical to real traffic (registry lookup, pricing, ledger row).
MODEL = Settings().model_budget
KEY_NO_CACHE = "sk-bench-no-cache"
KEY_CACHE = "sk-bench-cache"
_GATEWAY_DUR = re.compile(r"gateway;dur=([0-9.]+)")


# ── Processes ──────────────────────────────────────────────────────────────


def _wait_ready(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(url, timeout=1.0).status_code < 500:
                return
        time.sleep(0.2)
    raise RuntimeError(f"not ready after {timeout_s}s: {url}")


@contextlib.contextmanager
def _process(argv: list[str], env: dict[str, str], ready_url: str) -> Generator[None]:
    proc = subprocess.Popen(argv, cwd=ROOT, env=env)  # noqa: S603 (fixed argv)
    try:
        _wait_ready(ready_url)
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _uvicorn(app: str, port: int, host: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        app,
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        "1",
        "--log-level",
        "warning",
        "--no-access-log",
    ]


def _gateway_env(redis_url: str, db_path: Path, *, semantic: bool) -> dict[str, str]:
    return {
        **os.environ,
        "GATEWAY_OLLAMA_BASE_URL": STUB_URL,
        "GATEWAY_REDIS_URL": redis_url,
        "GATEWAY_DB_PATH": str(db_path),
        "GATEWAY_TEAMS_YAML": str(ROOT / "bench" / "teams.yaml"),
        "GATEWAY_LOG_LEVEL": "WARNING",
        "GATEWAY_CACHE_ENABLED": "true",
        "GATEWAY_CACHE_SEMANTIC_ENABLED": "true" if semantic else "false",
    }


# ── Sequential measurement ─────────────────────────────────────────────────

Sender = Callable[[httpx.Client, int], httpx.Response]


def _gateway_sender(api_key: str, prompt: Callable[[int], str]) -> Sender:
    headers = {"Authorization": f"Bearer {api_key}"}

    def send(client: httpx.Client, i: int) -> httpx.Response:
        return client.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            headers=headers,
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt(i)}]},
        )

    return send


def _direct_sender(client: httpx.Client, _: int) -> httpx.Response:
    return client.post(
        f"{STUB_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "benchmark prompt"}],
            "stream": False,
            "options": {"temperature": 0.0},
        },
    )


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _run_scenario(send: Sender, *, requests: int, warmup: int, repeats: int) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    with httpx.Client(timeout=10.0) as client:
        for i in range(warmup):
            send(client, -1 - i).raise_for_status()
        for repeat in range(repeats):
            round_trip: list[float] = []
            gateway: list[float] = []
            for i in range(requests):
                start = time.perf_counter()
                response = send(client, repeat * requests + i)
                round_trip.append((time.perf_counter() - start) * 1000)
                response.raise_for_status()
                match = _GATEWAY_DUR.search(response.headers.get("server-timing", ""))
                if match:
                    gateway.append(float(match.group(1)))
            run: dict[str, Any] = {"round_trip_ms": _percentiles(round_trip)}
            if gateway:
                run["gateway_self_ms"] = _percentiles(gateway)
            runs.append(run)
    return {"runs": runs, "median": _median_of_runs(runs)}


def _median_of_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for measure in ("round_trip_ms", "gateway_self_ms"):
        if measure not in runs[0]:
            continue
        out[measure] = {
            p: {
                "median": float(np.median([r[measure][p] for r in runs])),
                "min": min(r[measure][p] for r in runs),
                "max": max(r[measure][p] for r in runs),
            }
            for p in ("p50", "p95", "p99")
        }
    return out


# ── Open loop (k6 in Docker) ───────────────────────────────────────────────


def _run_k6(rate: int, duration: str, api_key: str) -> dict[str, Any]:
    argv = [
        "docker",
        "run",
        "--rm",
        "--add-host=host.docker.internal:host-gateway",
        "-v",
        f"{ROOT / 'bench'}:/bench:ro",
        "-e",
        f"BASE_URL=http://host.docker.internal:{GATEWAY_PORT}",
        "-e",
        f"API_KEY={api_key}",
        "-e",
        f"RATE={rate}",
        "-e",
        f"DURATION={duration}",
        "-e",
        f"MODEL={MODEL}",
        K6_IMAGE,
        "run",
        "--quiet",
        "/bench/k6_open_loop.js",
    ]
    result = subprocess.run(argv, check=True, capture_output=True, text=True)  # noqa: S603
    return json.loads(result.stdout[result.stdout.index("{") :])


# ── Machine description ────────────────────────────────────────────────────


def _command(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _redis_round_trip_ms(redis_url: str, samples: int = 200) -> dict[str, float]:
    """Redis PING time from this machine: the floor under every rate-limit check."""
    from redis import Redis

    client = Redis.from_url(redis_url)
    try:
        timings: list[float] = []
        for _ in range(samples):
            start = time.perf_counter()
            client.ping()
            timings.append((time.perf_counter() - start) * 1000)
    finally:
        client.close()
    return _percentiles(timings)


def _machine() -> dict[str, Any]:
    if sys.platform == "darwin":
        cpu = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
        memory_bytes = int(_command(["sysctl", "-n", "hw.memsize"]) or 0)
    else:
        cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.splitlines()
                if "model name" in line
            ),
            platform.processor() or "unknown",
        )
        meminfo = Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else ""
        kb = next((int(line.split()[1]) for line in meminfo.splitlines() if "MemTotal" in line), 0)
        memory_bytes = kb * 1024
    return {
        "os": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "memory_gb": round(memory_bytes / 2**30, 1),
        "python": platform.python_version(),
        "packages": {
            name: metadata.version(name)
            for name in ("fastapi", "starlette", "uvicorn", "uvloop", "httpx", "redis", "numpy")
        },
        "git_commit": _command(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_command(["git", "status", "--porcelain"])),
    }


# ── Main ───────────────────────────────────────────────────────────────────


def _print_table(scenarios: dict[str, dict[str, Any]]) -> None:
    direct_p50 = scenarios["direct_to_stub"]["median"]["round_trip_ms"]["p50"]["median"]
    print(
        "\n| Scenario | Round trip p50 / p95 / p99 (ms) | Gateway self-time p50 / p95 / p99 (ms) |"
    )
    print("|---|---|---|")
    for name, result in scenarios.items():
        rt = result["median"]["round_trip_ms"]
        cells = " / ".join(f"{rt[p]['median']:.2f}" for p in ("p50", "p95", "p99"))
        gw = result["median"].get("gateway_self_ms")
        gw_cells = (
            " / ".join(f"{gw[p]['median']:.2f}" for p in ("p50", "p95", "p99")) if gw else "n/a"
        )
        print(f"| {name} | {cells} | {gw_cells} |")
    print(f"\nDirect-to-stub p50 baseline: {direct_p50:.2f} ms")


def _unique_prompt(i: int) -> str:
    """A prompt never seen before, so every request misses the cache."""
    return f"benchmark prompt {uuid.uuid4()} {i}"


def _fixed_prompt(_: int) -> str:
    return "benchmark prompt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the latency the gateway adds on top of the provider."
    )
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--redis-url", default="redis://localhost:6379/15")
    parser.add_argument("--open-loop", action="store_true")
    parser.add_argument("--rates", default="100,250,500")
    parser.add_argument("--duration", default="30s")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "raw")
    args = parser.parse_args()
    if args.open_loop and shutil.which("docker") is None:
        parser.error("--open-loop needs docker")

    sizes = {"requests": args.requests, "warmup": args.warmup, "repeats": args.repeats}
    scenarios: dict[str, dict[str, Any]] = {}
    open_loop: list[dict[str, Any]] = []
    # Open-loop runs reach the gateway from a container, so it binds all interfaces.
    host = "0.0.0.0" if args.open_loop else "127.0.0.1"  # noqa: S104

    with (
        tempfile.TemporaryDirectory() as tmp,
        _process(
            _uvicorn("bench.stub_ollama:app", STUB_PORT, "127.0.0.1"),
            dict(os.environ),
            f"{STUB_URL}/api/tags",
        ),
    ):
        print("scenario: direct_to_stub")
        scenarios["direct_to_stub"] = _run_scenario(_direct_sender, **sizes)

        env = _gateway_env(args.redis_url, Path(tmp) / "exact.db", semantic=False)
        with _process(
            _uvicorn("llm_gateway.main:app", GATEWAY_PORT, host), env, f"{GATEWAY_URL}/health"
        ):
            for name, key, prompt in (
                ("gateway_cache_off", KEY_NO_CACHE, _fixed_prompt),
                ("gateway_exact_cache_miss", KEY_CACHE, _unique_prompt),
                ("gateway_exact_cache_hit", KEY_CACHE, _fixed_prompt),
            ):
                print(f"scenario: {name}")
                scenarios[name] = _run_scenario(_gateway_sender(key, prompt), **sizes)
            if args.open_loop:
                for rate in (int(r) for r in args.rates.split(",")):
                    print(f"open loop: {rate} req/s for {args.duration}")
                    open_loop.append(_run_k6(rate, args.duration, KEY_NO_CACHE))

        env = _gateway_env(args.redis_url, Path(tmp) / "semantic.db", semantic=True)
        with _process(
            _uvicorn("llm_gateway.main:app", GATEWAY_PORT, host), env, f"{GATEWAY_URL}/health"
        ):
            print("scenario: gateway_semantic_cache_miss")
            scenarios["gateway_semantic_cache_miss"] = _run_scenario(
                _gateway_sender(KEY_CACHE, _unique_prompt), **sizes
            )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated_at": stamp,
        "method": {**sizes, "model": MODEL, "gateway_workers": 1, "log_level": "WARNING"},
        "machine": _machine(),
        "redis_ping_ms": _redis_round_trip_ms(args.redis_url),
        "sequential": scenarios,
        "open_loop": open_loop,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"overhead-{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    _print_table(scenarios)
    for run in open_loop:
        print(json.dumps(run))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
