"""Pick the semantic-cache threshold from labelled prompt pairs.

Embeds both prompts of every pair in ``bench/cache_calibration.jsonl`` with
the gateway's own Ollama adapter and cache prefix, then sweeps thresholds and
reports, for each one, how many true paraphrases would hit and how many
near-misses would be served a wrong cached answer, with and without the
literal guard.

A wrong hit is the costly error (a confidently wrong answer), a missed hit only
costs one model call, so read the table for the lowest threshold whose false
hits are acceptable.

Usage::

    uv run python scripts/calibrate_cache.py
    uv run python scripts/calibrate_cache.py --model <embedding model> --prefix ""

The model, prefix and Ollama URL default to the gateway's settings
(``GATEWAY_MODEL_EMBEDDINGS``, ``GATEWAY_CACHE_EMBEDDING_PREFIX``,
``GATEWAY_OLLAMA_BASE_URL``), so the calibration measures what the cache would do.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

from llm_gateway.cache.semantic import extract_literals
from llm_gateway.config import Settings
from llm_gateway.providers.ollama import OllamaProvider

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "bench" / "cache_calibration.jsonl"
DEFAULT_OUT_DIR = ROOT / "results" / "raw"
THRESHOLDS = [round(0.80 + 0.01 * i, 2) for i in range(20)]


async def _embed_all(
    provider: OllamaProvider, model: str, texts: list[str]
) -> list[NDArray[np.float32]]:
    vectors: list[NDArray[np.float32]] = []
    for text in texts:
        result = await provider.embed(model, text)
        vector = np.asarray(result.embedding, dtype=np.float32)
        vectors.append((vector / np.linalg.norm(vector)).astype(np.float32))
    return vectors


def _sweep(pairs: list[dict[str, Any]], similarities: list[float]) -> list[dict[str, Any]]:
    positives = sum(1 for p in pairs if p["should_hit"])
    negatives = len(pairs) - positives
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        row: dict[str, Any] = {"threshold": threshold}
        for guarded in (False, True):
            hits = wrong = 0
            for pair, sim in zip(pairs, similarities, strict=True):
                accepted = sim >= threshold and (
                    not guarded or extract_literals(pair["a"]) == extract_literals(pair["b"])
                )
                if accepted and pair["should_hit"]:
                    hits += 1
                elif accepted:
                    wrong += 1
            suffix = "guarded" if guarded else "raw"
            row[f"hit_rate_{suffix}"] = hits / positives
            row[f"false_hits_{suffix}"] = wrong
            row[f"false_hit_rate_{suffix}"] = wrong / negatives
        rows.append(row)
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pick the semantic-cache threshold from labelled prompt pairs."
    )
    settings = Settings()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ollama-url", default=settings.ollama_base_url)
    parser.add_argument("--model", default=settings.model_embeddings)
    parser.add_argument("--prefix", default=settings.cache_embedding_prefix)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    pairs: list[dict[str, Any]] = [
        json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()
    ]
    provider = OllamaProvider(base_url=args.ollama_url, timeout_s=60.0)
    try:
        a_vectors = await _embed_all(provider, args.model, [args.prefix + p["a"] for p in pairs])
        b_vectors = await _embed_all(provider, args.model, [args.prefix + p["b"] for p in pairs])
    finally:
        await provider.aclose()
    similarities = [float(a @ b) for a, b in zip(a_vectors, b_vectors, strict=True)]
    rows = _sweep(pairs, similarities)

    by_category: dict[str, list[float]] = {}
    for pair, sim in zip(pairs, similarities, strict=True):
        by_category.setdefault(pair["category"], []).append(sim)

    console = Console()
    summary = Table(title="Cosine similarity by category")
    for column in ("category", "pairs", "min", "median", "max"):
        summary.add_column(column, justify="right")
    for category, sims in by_category.items():
        summary.add_row(
            category,
            str(len(sims)),
            f"{min(sims):.3f}",
            f"{float(np.median(sims)):.3f}",
            f"{max(sims):.3f}",
        )
    console.print(summary)

    table = Table(title="Threshold sweep (false hits out of all near-miss pairs)")
    for column in ("threshold", "hit rate", "false hits raw", "false hits guarded"):
        table.add_column(column, justify="right")
    for row in rows:
        table.add_row(
            f"{row['threshold']:.2f}",
            f"{row['hit_rate_guarded']:.0%}",
            f"{row['false_hits_raw']} ({row['false_hit_rate_raw']:.0%})",
            f"{row['false_hits_guarded']} ({row['false_hit_rate_guarded']:.0%})",
        )
    console.print(table)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out_dir / f"calibration-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "generated_at": stamp,
                "dataset": str(args.dataset.relative_to(ROOT)),
                "embedding_model": args.model,
                "prefix": args.prefix,
                "machine": platform.platform(),
                "pairs": [
                    {**pair, "similarity": round(sim, 4)}
                    for pair, sim in zip(pairs, similarities, strict=True)
                ],
                "sweep": rows,
            },
            indent=2,
        )
        + "\n"
    )
    console.print(f"Wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
