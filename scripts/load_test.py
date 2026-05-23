"""Load test — generates the headline cost-savings metric.

Sends N requests with the brief's RAG-realistic mix:

* 50% generation
* 30% retrieval (embedding)
* 15% reranking
* 5%  evaluation

The router uses the budget model (``qwen2.5:0.5b``) for "simple" generation
work (reranking + evaluation), and the high-quality model (``llama3.2:3b``)
for actual answer synthesis. Embeddings always use ``nomic-embed-text``.

At the end it prints:

* Actual cost (sum of ``X-Gateway-Cost-USD`` headers from successful calls)
* Hypothetical cost if every chat call had used the high-quality model
* Cost savings %
* P50 / P95 / P99 gateway-side latency

Run with the gateway up:

    docker-compose up -d
    uv run python scripts/load_test.py --requests 1000
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "sk-gw-prod-search-8f3a2c1b"

# 4 operation buckets summing to 100.
MIX_PERCENTAGES: dict[str, float] = {
    "generation": 0.50,
    "retrieval": 0.30,
    "reranking": 0.15,
    "evaluation": 0.05,
}


class Operation(StrEnum):
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    EVALUATION = "evaluation"


@dataclass
class Result:
    operation: Operation
    success: bool
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    status_code: int = 0


@dataclass
class Summary:
    by_operation: dict[Operation, list[Result]] = field(
        default_factory=lambda: {op: [] for op in Operation}
    )

    def add(self, result: Result) -> None:
        self.by_operation[result.operation].append(result)

    def all_results(self) -> list[Result]:
        out: list[Result] = []
        for results in self.by_operation.values():
            out.extend(results)
        return out


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    if lower_idx == upper_idx:
        return sorted_values[lower_idx]
    weight = rank - lower_idx
    return sorted_values[lower_idx] * (1 - weight) + sorted_values[upper_idx] * weight


def _draw_operation(rng: random.Random) -> Operation:
    roll = rng.random()
    cumulative = 0.0
    for op_str, weight in MIX_PERCENTAGES.items():
        cumulative += weight
        if roll <= cumulative:
            return Operation(op_str)
    return Operation.GENERATION


def _build_request(
    op: Operation, *, models: dict[str, str]
) -> tuple[str, dict[str, object], dict[str, str]]:
    if op is Operation.RETRIEVAL:
        return (
            "/v1/embeddings",
            {
                "model": models["embed"],
                "input": "noise-cancelling headphones under $300 long battery life",
            },
            {},
        )
    if op is Operation.GENERATION:
        return (
            "/v1/chat/completions",
            {
                "model": models["quality"],
                "messages": [
                    {
                        "role": "user",
                        "content": "Which headphones have the best noise cancellation under $300?",
                    }
                ],
                "max_tokens": 64,
            },
            {},
        )
    if op is Operation.RERANKING:
        return (
            "/v1/chat/completions",
            {
                "model": models["budget"],
                "messages": [
                    {"role": "system", "content": "Rerank: rate relevance 1-5."},
                    {
                        "role": "user",
                        "content": "Query: long battery headphones. Passage: WH-1000XM5 30h $299.",
                    },
                ],
                "max_tokens": 16,
            },
            {"X-Rag-Operation": "reranking"},
        )
    # evaluation
    return (
        "/v1/chat/completions",
        {
            "model": models["budget"],
            "messages": [
                {"role": "system", "content": "You are a judge — score 1-5."},
                {
                    "role": "user",
                    "content": "Answer: Sony WH-1000XM5 — 30 h battery, top-rated ANC, $299.",
                },
            ],
            "max_tokens": 16,
        },
        {},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quality-model", default="llama3.2:3b", help="High-quality (expensive) tier."
    )
    parser.add_argument("--budget-model", default="qwen2.5:0.5b")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    models = {
        "quality": args.quality_model,
        "budget": args.budget_model,
        "embed": args.embed_model,
    }

    summary = Summary()
    client = httpx.Client(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=60.0,
    )

    print(f"Sending {args.requests} requests against {args.base_url} …")
    started = time.perf_counter()
    for i in range(args.requests):
        op = _draw_operation(rng)
        path, body, extra_headers = _build_request(op, models=models)
        t0 = time.perf_counter()
        try:
            response = client.post(path, json=body, headers=extra_headers)
        except httpx.HTTPError as exc:
            summary.add(Result(operation=op, success=False, status_code=0))
            print(f"  [{i + 1:>4}] {op.value:<11} ERROR  {exc}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000.0
        cost_header = response.headers.get("X-Gateway-Cost-USD")
        cost_usd = float(cost_header) if cost_header else 0.0
        summary.add(
            Result(
                operation=op,
                success=response.is_success,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                status_code=response.status_code,
            )
        )
        if (i + 1) % 50 == 0:
            print(f"  …{i + 1} / {args.requests}")
    elapsed = time.perf_counter() - started

    # ── Aggregate ──────────────────────────────────────────────────────────
    actual_cost = sum(r.cost_usd for r in summary.all_results() if r.success)
    # Hypothetical: every chat-style call routed to the high-quality model.
    # Approximated as 5× the budget-model cost (matching the brief's per-1k prices:
    # llama3.2:3b is roughly 30× input + 75× output of qwen2.5:0.5b).
    quality_multiplier = 30.0
    hypothetical_cost = 0.0
    for r in summary.all_results():
        if r.operation is Operation.RETRIEVAL:
            hypothetical_cost += r.cost_usd  # embeddings already use the cheap embed model
        else:
            hypothetical_cost += r.cost_usd * quality_multiplier
    savings_pct = (1.0 - actual_cost / hypothetical_cost) * 100.0 if hypothetical_cost else 0.0

    latencies = [r.latency_ms for r in summary.all_results() if r.success]
    p50 = _percentile(latencies, 50.0)
    p95 = _percentile(latencies, 95.0)
    p99 = _percentile(latencies, 99.0)

    success_count = sum(1 for r in summary.all_results() if r.success)
    failure_count = args.requests - success_count

    # ── Print headline ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Elapsed:           {elapsed:.1f}s")
    print(f"  Requests / second: {args.requests / max(elapsed, 1e-9):.1f}")
    print(f"  Success / fail:    {success_count} / {failure_count}")
    print(f"  Actual cost:       ${actual_cost:.6f}")
    print(f"  Hypothetical cost: ${hypothetical_cost:.6f}  (if all-high-quality)")
    print(f"  COST SAVINGS:      {savings_pct:6.2f}%")
    print(f"  Latency P50/P95/P99: {p50:.1f} / {p95:.1f} / {p99:.1f}  ms (round-trip)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
