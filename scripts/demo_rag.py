"""Demo: a full RAG pipeline run with cost attribution.

Walks through the operations a real RAG system performs (retrieval with
embeddings, reranking, generation, and an LLM-as-Judge evaluation), then prints the
per-operation cost breakdown the gateway recorded.

Run with the gateway up:

    docker-compose up -d
    uv run python scripts/demo_rag.py [--api-key sk-...]

The default API key matches the example team in ``config/teams.example.yaml``.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Any

import httpx
from gateway_models import list_models, models_by_tier

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "sk-gw-prod-search-replace-me"

DOCUMENT_CHUNKS = [
    "Sony WH-1000XM5: 30-hour battery, noise cancellation, foldable, $299.",
    "Bose QuietComfort 45: 24-hour battery, adjustable ANC, lightweight, $279.",
    "Apple AirPods Pro (2nd gen): 6-hour battery (30 h with case), Adaptive Transparency, $249.",
]

QUERY = "noise-cancelling headphones with long battery life under $300"


def _step(name: str) -> None:
    print(f"\n── {name} {'─' * (60 - len(name))}")


def _post(
    client: httpx.Client, path: str, body: dict[str, Any], **extra_headers: str
) -> dict[str, Any]:
    headers: dict[str, str] = dict(extra_headers)
    response = client.post(path, json=body, headers=headers, timeout=60.0)
    response.raise_for_status()
    cost = response.headers.get("X-Gateway-Cost-USD", "?")
    op = response.headers.get("X-Gateway-Rag-Operation", "?")
    model = response.headers.get("X-Gateway-Model-Used", "?")
    print(f"  → {response.status_code}  model={model}  op={op}  cost=${cost}")
    return response.json()


def _get(client: httpx.Client, path: str) -> dict[str, Any]:
    response = client.get(path, timeout=10.0)
    response.raise_for_status()
    return response.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument(
        "--embedding-model", help="Default: what the gateway serves for this tier (GET /v1/models)."
    )
    parser.add_argument(
        "--budget-model", help="Default: what the gateway serves for this tier (GET /v1/models)."
    )
    parser.add_argument(
        "--quality-model", help="Default: what the gateway serves for this tier (GET /v1/models)."
    )
    args = parser.parse_args(argv)

    client = httpx.Client(
        base_url=args.base_url,
        headers={"Authorization": f"Bearer {args.api_key}"},
    )
    tiers = models_by_tier(list_models(client))
    args.embedding_model = args.embedding_model or tiers["embeddings"]
    args.budget_model = args.budget_model or tiers["budget"]
    args.quality_model = args.quality_model or tiers["high_quality"]

    print(
        textwrap.dedent(f"""
        LLM Gateway: RAG demo
        Base URL:        {args.base_url}
        Embedding model: {args.embedding_model}
        Budget model:    {args.budget_model}
        Quality model:   {args.quality_model}
    """).strip()
    )

    # 1. Retrieval: embed the query
    _step("1. Retrieval (embedding)")
    _post(client, "/v1/embeddings", {"model": args.embedding_model, "input": QUERY})

    # 2. Retrieval: embed each candidate chunk
    _step("2. Retrieval: chunk embeddings")
    for chunk in DOCUMENT_CHUNKS:
        _post(client, "/v1/embeddings", {"model": args.embedding_model, "input": chunk})

    # 3. Reranking: a cross-encoder style call (small model, signalled via header)
    _step("3. Reranking")
    _post(
        client,
        "/v1/chat/completions",
        {
            "model": args.budget_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Score the relevance of the passage to the query (1–5).",
                },
                {
                    "role": "user",
                    "content": f"Query: {QUERY}\nPassage: {DOCUMENT_CHUNKS[0]}",
                },
            ],
            "max_tokens": 64,
        },
        **{"X-Rag-Operation": "reranking"},
    )

    # 4. Generation: synthesise the answer with the high-quality model
    _step("4. Generation")
    _post(
        client,
        "/v1/chat/completions",
        {
            "model": args.quality_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Use the passages to answer the user's question concisely.",
                },
                {
                    "role": "user",
                    "content": (f"Question: {QUERY}\nPassages:\n- " + "\n- ".join(DOCUMENT_CHUNKS)),
                },
            ],
            "max_tokens": 256,
        },
    )

    # 5. Evaluation: LLM-as-Judge on the answer
    _step("5. Evaluation (LLM-as-Judge)")
    _post(
        client,
        "/v1/chat/completions",
        {
            "model": args.budget_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a judge. Score the answer's accuracy 1–5.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {QUERY}\n"
                        "Answer: The Sony WH-1000XM5 offers 30-hour battery "
                        "with top-rated noise cancellation at $299."
                    ),
                },
            ],
            "max_tokens": 64,
        },
    )

    # 6. Cost report
    _step("6. RAG cost breakdown for this team today")
    report = _get(client, "/v1/rag/cost-report")
    total = report["total_cost_usd"]
    print(f"  Total cost: ${total:.6f}")
    for op, cost in report["by_operation"].items():
        share = (cost / total * 100.0) if total > 0 else 0.0
        print(f"    {op:<12} ${cost:.6f}  ({share:5.1f}%)")

    print("\nView the live dashboard:  http://localhost:3000/d/rag-cost-attribution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
