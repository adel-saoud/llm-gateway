"""Model ids a running gateway serves, by tier, for the demo and load scripts.

The scripts ask the gateway (``GET /v1/models``) instead of carrying their own
defaults, so the models come from the gateway's settings alone and a model
change never leaves a script calling a model the gateway no longer serves.
"""

from __future__ import annotations

from typing import Any

import httpx


def list_models(client: httpx.Client) -> list[dict[str, Any]]:
    """The gateway's model catalogue: id, tier and simulated per-1k prices."""
    response = client.get("/v1/models")
    response.raise_for_status()
    return response.json()["data"]


def models_by_tier(catalogue: list[dict[str, Any]]) -> dict[str, str]:
    """First model id listed for each tier (``high_quality``, ``budget``, ``embeddings``)."""
    tiers: dict[str, str] = {}
    for model in catalogue:
        tiers.setdefault(model["tier"], model["id"])
    return tiers
