"""Demo: circuit breaker and automatic fallback.

Shows the full CLOSED → OPEN → HALF_OPEN → CLOSED state cycle and the
automatic fallback from the high-quality model to the budget model:

1. Stops the Ollama backend (auto for bundled Docker, prompted for host Ollama).
2. Sends requests that fail → circuit breaker trips OPEN, fallback metric emits.
3. Waits for the open-duration to elapse (default 30 s).
4. Restores Ollama.
5. Sends a probe request → HALF_OPEN → CLOSED.

Watch the Operations dashboard while this runs:
    http://localhost:3000/d/operations

Run with the gateway up:

    docker-compose up -d          # bundled Ollama
    uv run python scripts/demo_resilience.py

    # or for host Ollama:
    uv run python scripts/demo_resilience.py --host-ollama
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx
from gateway_models import list_models, models_by_tier

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "sk-gw-prod-search-replace-me"

# Must exceed GATEWAY_CIRCUIT_BREAKER_MIN_REQUESTS (default 10) so the circuit can open.
FAILURE_REQUESTS = 14

# Must match GATEWAY_CIRCUIT_BREAKER_OPEN_DURATION_S (default 30 s).
OPEN_DURATION_S = 30


def _detect_compose_cmd() -> str:
    result = subprocess.run(
        ["docker", "compose", "version"],  # noqa: S607
        capture_output=True,
        check=False,
    )
    return "docker compose" if result.returncode == 0 else "docker-compose"


def _is_host_ollama() -> bool:
    """Return True when the override file points at host.docker.internal."""
    override = Path("docker-compose.override.yml")
    return override.exists() and "host.docker.internal" in override.read_text()


def _step(name: str) -> None:
    width = 60 - len(name)
    print(f"\n── {name} {'─' * max(width, 2)}")


def _send_requests(client: httpx.Client, n: int, model: str) -> None:
    for i in range(n):
        try:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
                timeout=5.0,
            )
            print(f"  [{i + 1:>2}] {r.status_code}")
        except httpx.HTTPError as exc:
            print(f"  [{i + 1:>2}] network error: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument(
        "--host-ollama",
        action="store_true",
        default=_is_host_ollama(),
        help="Use manual stop/start prompts instead of docker compose commands.",
    )
    parser.add_argument(
        "--quality-model",
        help="Model that triggers the fallback chain. Default: the gateway's high_quality tier.",
    )
    parser.add_argument(
        "--open-duration",
        type=int,
        default=OPEN_DURATION_S,
        help="Seconds to wait for the circuit to half-open (match the gateway setting).",
    )
    args = parser.parse_args(argv)

    compose_cmd = _detect_compose_cmd()
    client = httpx.Client(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {args.api_key}"},
    )
    args.quality_model = args.quality_model or models_by_tier(list_models(client))["high_quality"]

    print(
        f"\nLLM Gateway: resilience demo\n"
        f"Base URL:       {args.base_url}\n"
        f"Quality model:  {args.quality_model}\n"
        f"Ollama mode:    {'host (manual)' if args.host_ollama else 'bundled (auto)'}\n"
        f"Open duration:  {args.open_duration}s\n"
        f"\nOpen the Operations dashboard now:\n"
        f"  http://localhost:3000/d/operations"
    )

    # ── 1. Stop Ollama ─────────────────────────────────────────────────────
    _step("1. Stopping Ollama backend")
    if args.host_ollama:
        print("  Host Ollama detected: please stop it now.")
        print("  → macOS / Linux: pkill -f 'ollama serve'  or  killall ollama")
        input("  Press Enter when Ollama is stopped…")
    else:
        subprocess.run([*compose_cmd.split(), "stop", "ollama"], check=True)  # noqa: S603
        print("  Bundled Ollama container stopped.")
        time.sleep(1)

    # ── 2. Send failing requests ────────────────────────────────────────────
    _step(f"2. Sending {FAILURE_REQUESTS} requests (expect 503s: provider down)")
    print(
        f"  Each request tries {args.quality_model} → falls back to budget model → both fail.\n"
        f"  The circuit breaker records each failure."
    )
    _send_requests(client, FAILURE_REQUESTS, args.quality_model)

    print(
        f"\n  Circuit breaker is now OPEN for ollama / {args.quality_model}.\n"
        f"  → Refresh the 'Circuit breaker health' panel: it should show OPEN (red).\n"
        f"  → The 'Fallback rate' panel should show a non-zero rate."
    )

    # ── 3. Wait for open-duration ───────────────────────────────────────────
    _step(f"3. Waiting {args.open_duration}s for circuit to transition to HALF_OPEN")
    print("  (The gateway will automatically allow one probe request after this.)\n")
    for remaining in range(args.open_duration, 0, -1):
        print(f"  {remaining:>3}s remaining…", end="\r", flush=True)
        time.sleep(1)
    print()

    # ── 4. Restore Ollama ──────────────────────────────────────────────────
    _step("4. Restoring Ollama backend")
    if args.host_ollama:
        print("  Please start Ollama again now.")
        print("  → ollama serve")
        input("  Press Enter when Ollama is running…")
    else:
        subprocess.run([*compose_cmd.split(), "start", "ollama"], check=True)  # noqa: S603
        print("  Waiting for Ollama to be ready…")
        for i in range(15, 0, -1):
            print(f"  {i:>2}s…", end="\r", flush=True)
            time.sleep(1)
        print()

    # ── 5. Probe request ────────────────────────────────────────────────────
    _step("5. Sending probe request (circuit in HALF_OPEN → expect success)")
    try:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": args.quality_model,
                "messages": [{"role": "user", "content": "Say 'ok' in one word."}],
                "max_tokens": 4,
            },
            timeout=60.0,
        )
        model_used = r.headers.get("X-Gateway-Model-Used", "?")
        cost = r.headers.get("X-Gateway-Cost-USD", "?")
        print(f"  → {r.status_code}  model={model_used}  cost=${cost}")
        if r.is_success:
            print("  Circuit breaker transitioned: HALF_OPEN → CLOSED ✓")
        else:
            print(
                "  Request failed: Ollama may not be fully ready yet. "
                "Try: uv run python scripts/demo_rag.py"
            )
    except httpx.HTTPError as exc:
        print(f"  Network error: {exc}")
        print("  Ollama may not be ready. Try running: uv run python scripts/demo_rag.py")

    print(
        "\n  → Refresh the Operations dashboard to see the CLOSED (green) state:\n"
        "  http://localhost:3000/d/operations\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
