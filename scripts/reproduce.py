"""Regenerate every number the README shows and write ``results/manifest.json``.

    uv run python scripts/reproduce.py

Three runs, each by the script that owns the method:

1. ``scripts/bench_overhead.py --open-loop``: gateway overhead against the
   instant stub provider, sequential and k6 constant-arrival-rate runs.
2. ``scripts/calibrate_cache.py``: cosine similarity of labelled prompt pairs
   with the configured embedding model on a local Ollama, and the threshold sweep.
3. ``scripts/demo_rag.py`` against the stub provider, then
   ``GET /v1/rag/cost-report``: the per-stage cost split shown under "Try it".

Raw outputs are copied to ``results/raw/`` under fixed names, and the manifest
records the environment and each README metric with its sample size. Nothing
here needs a paid API key.

Needs: Redis on localhost:6379 (``docker compose up -d redis``), Docker for the
k6 runs, and Ollama with the configured models pulled
(``uv run python -m llm_gateway.pull_models``).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from statistics import median
from typing import Any, Final

import httpx

from llm_gateway.config import Settings
from llm_gateway.pull_models import configured_models

ROOT: Final = Path(__file__).resolve().parent.parent
RESULTS: Final = ROOT / "results"
RAW: Final = RESULTS / "raw"
CALIBRATION_DATASET: Final = ROOT / "bench" / "cache_calibration.jsonl"
# Ports distinct from the benchmark's (18000 / 18080) and from a dev gateway on 8000.
DEMO_STUB_PORT: Final = 18180
DEMO_GATEWAY_PORT: Final = 18100
DEMO_API_KEY: Final = "sk-gw-prod-search-replace-me"
PACKAGES: Final = ("fastapi", "starlette", "uvicorn", "httpx", "redis", "numpy", "pydantic")


def _run(*args: str) -> None:
    """Run a sibling script with this interpreter; its output streams through."""
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)  # noqa: S603


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)  # noqa: S603, S607
    return result.stdout.strip()


def _only_json(directory: Path, prefix: str) -> dict[str, Any]:
    (path,) = directory.glob(f"{prefix}-*.json")
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ollama_digests(ollama_url: str) -> dict[str, str]:
    """Digest of each model on the local Ollama, the version a name alone does not pin."""
    tags = httpx.get(f"{ollama_url}/api/tags", timeout=10).json()
    return {model["name"]: model["digest"] for model in tags["models"]}


def _ollama_tag(model: str, digests: dict[str, str]) -> str | None:
    """Ollama's full name for ``model``: a bare name is stored as ``<name>:latest``."""
    for candidate in (model, f"{model}:latest"):
        if candidate in digests:
            return candidate
    return None


@contextlib.contextmanager
def _serve(app: str, port: int, env: dict[str, str]) -> Generator[None]:
    """Run ``app`` under uvicorn until the block exits; wait for it to answer first."""
    command = [sys.executable, "-m", "uvicorn", app, "--port", str(port), "--log-level", "warning"]
    process = subprocess.Popen(command, cwd=ROOT, env=env)  # noqa: S603
    try:
        for _ in range(50):
            with contextlib.suppress(httpx.HTTPError):
                httpx.get(f"http://127.0.0.1:{port}/", timeout=1)
                break
            time.sleep(0.2)
        yield
    finally:
        process.terminate()
        process.wait(timeout=10)


def _rag_cost_split(tmp: Path) -> dict[str, Any]:
    """Run the RAG demo against the stub provider and read back the team's cost report."""
    env = {
        **os.environ,
        "GATEWAY_TEAMS_YAML": str(ROOT / "config" / "teams.example.yaml"),
        "GATEWAY_OLLAMA_BASE_URL": f"http://127.0.0.1:{DEMO_STUB_PORT}",
        "GATEWAY_DB_PATH": str(tmp / "demo.db"),
        "GATEWAY_REDIS_URL": "redis://localhost:6379/14",
    }
    base_url = f"http://127.0.0.1:{DEMO_GATEWAY_PORT}"
    with (
        _serve("bench.stub_ollama:app", DEMO_STUB_PORT, dict(os.environ)),
        _serve("llm_gateway.main:app", DEMO_GATEWAY_PORT, env),
    ):
        _run("scripts/demo_rag.py", "--base-url", base_url, "--api-key", DEMO_API_KEY)
        report = httpx.get(
            f"{base_url}/v1/rag/cost-report",
            headers={"Authorization": f"Bearer {DEMO_API_KEY}"},
            timeout=10,
        ).json()
    return report


def _overhead_metrics(bench: dict[str, Any]) -> dict[str, Any]:
    method = bench["method"]
    n = method["requests"] * method["repeats"]
    sequential = {
        name: {
            "gateway_self_ms_p50": scenario["median"]["gateway_self_ms"]["p50"]["median"],
            "gateway_self_ms_p99": scenario["median"]["gateway_self_ms"]["p99"]["median"],
            "n_requests": n,
        }
        for name, scenario in bench["sequential"].items()
        if "gateway_self_ms" in scenario["median"]
    }
    open_loop = [
        {
            "rate_per_s": run["rate_per_s"],
            "gateway_overhead_ms_p99": run["gateway_overhead_ms"]["p(99)"],
            "failed_rate": run["http_req_failed"]["rate"],
            "dropped_iterations": (run["dropped_iterations"] or {}).get("count", 0),
            "n_requests": run["iterations"]["count"],
        }
        for run in bench["open_loop"]
    ]
    return {
        "method": {**method, "statistic": "median over repeats of each run's percentile"},
        "sequential": sequential,
        "open_loop": open_loop,
    }


def _calibration_metrics(calibration: dict[str, Any]) -> dict[str, Any]:
    by_category: dict[str, list[float]] = {}
    for pair in calibration["pairs"]:
        by_category.setdefault(pair["category"], []).append(pair["similarity"])
    sweep = {
        f"{row['threshold']:.2f}": row
        for row in calibration["sweep"]
        if f"{row['threshold']:.2f}" in {"0.85", "0.88", "0.90", "0.92", "0.95", "0.97", "0.99"}
    }
    return {
        "embedding_model": calibration["embedding_model"],
        "prefix": calibration["prefix"],
        "median_cosine_by_category": {
            category: {"median": round(median(sims), 3), "n_pairs": len(sims)}
            for category, sims in by_category.items()
        },
        "sweep": sweep,
        "n_pairs": len(calibration["pairs"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the README numbers.")
    parser.parse_args(argv)
    if shutil.which("docker") is None:
        parser.error("docker is needed for the k6 open-loop runs")

    settings = Settings()
    digests = _ollama_digests(settings.ollama_base_url)
    configured = configured_models(settings)
    tags = {model: _ollama_tag(model, digests) for model in configured}
    missing = [model for model, tag in tags.items() if tag is None]
    if missing:
        parser.error(
            f"{', '.join(missing)} not in Ollama; run: uv run python -m llm_gateway.pull_models"
        )
    ollama_version = httpx.get(f"{settings.ollama_base_url}/api/version", timeout=10).json()

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        _run("scripts/bench_overhead.py", "--open-loop", "--out-dir", str(tmp))
        _run("scripts/calibrate_cache.py", "--out-dir", str(tmp))
        bench = _only_json(tmp, "overhead")
        calibration = _only_json(tmp, "calibration")
        rag_report = _rag_cost_split(tmp)

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "overhead.json").write_text(json.dumps(bench, indent=2) + "\n")
    (RAW / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    dirty = bool(_git("status", "--porcelain"))
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": "uncommitted" if dirty else _git("rev-parse", "HEAD"),
        "git_base_commit": _git("rev-parse", "HEAD"),
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "hardware": bench["machine"] | {"redis_ping_ms": bench["redis_ping_ms"]},
        "ollama_version": ollama_version["version"],
        "models": {
            "configured": {
                model: {"ollama_tag": tag, "digest": digests[tag] if tag else None}
                for model, tag in tags.items()
            },
            "used_by": {
                "cache_calibration": f"{settings.model_embeddings} on the local Ollama",
                "overhead_benchmark": "bench/stub_ollama.py (instant, fixed token counts)",
                "rag_demo_cost_split": "bench/stub_ollama.py (instant, fixed token counts)",
            },
        },
        "seed": None,
        "seed_note": "No sampling: the stub answers are fixed and embeddings are deterministic.",
        "datasets": {
            "cache_calibration": {
                "path": str(CALIBRATION_DATASET.relative_to(ROOT)),
                "sha256": _sha256(CALIBRATION_DATASET),
            }
        },
        "raw": {
            "overhead": "results/raw/overhead.json",
            "calibration": "results/raw/calibration.json",
        },
        "metrics": {
            "overhead": _overhead_metrics(bench),
            "cache_calibration": _calibration_metrics(calibration),
            "rag_demo_cost_split": {
                "total_cost_usd": rag_report["total_cost_usd"],
                "by_operation_usd": rag_report["by_operation"],
                "n_requests": 7,
                "prices": "simulated (providers/registry.py)",
            },
        },
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {RESULTS / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
