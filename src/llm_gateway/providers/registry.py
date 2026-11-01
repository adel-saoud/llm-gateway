"""Model registry: declarative inventory of routable models and their simulated costs.

Costs are synthetic. The gateway runs on free local Ollama models, so the
per-1k prices below only give the routing, the budget caps and the per-operation
split something realistic to work with: a high-quality model about 30 to 75
times dearer than a budget one, as with paid APIs. The token counts are real;
the dollar amounts are illustrative.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum


class ModelTier(StrEnum):
    """Routing tier: drives the high-quality vs budget decision in the router."""

    HIGH_QUALITY = "high_quality"
    BUDGET = "budget"
    EMBEDDINGS = "embeddings"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Static configuration for one routable model."""

    model_id: str
    tier: ModelTier
    cost_per_1k_input_usd: float
    cost_per_1k_output_usd: float
    is_embedding_model: bool = False

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Compute simulated USD cost for one completion."""
        input_cost = (prompt_tokens / 1000.0) * self.cost_per_1k_input_usd
        output_cost = (completion_tokens / 1000.0) * self.cost_per_1k_output_usd
        return input_cost + output_cost


class ProviderRegistry:
    """In-memory lookup of ``model_id → ModelConfig`` plus tier filters.

    Built once at startup from ``Settings``. Read-only thereafter so handlers can
    share the same instance across requests without locking.
    """

    def __init__(self, models: Iterable[ModelConfig]) -> None:
        by_id: dict[str, ModelConfig] = {}
        for m in models:
            if m.model_id in by_id:
                raise ValueError(f"duplicate model_id in registry: {m.model_id}")
            by_id[m.model_id] = m
        self._by_id: Mapping[str, ModelConfig] = by_id

    def get(self, model_id: str) -> ModelConfig:
        """Config of ``model_id``; raises ``KeyError`` for a model not in the registry."""
        try:
            return self._by_id[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model: {model_id}") from exc

    def tier_of(self, model_id: str) -> ModelTier | None:
        """Tier ``model_id`` serves, or None for a model not in the registry."""
        model = self._by_id.get(model_id)
        return model.tier if model is not None else None

    def has(self, model_id: str) -> bool:
        """True when ``model_id`` is routable."""
        return model_id in self._by_id

    def all(self) -> tuple[ModelConfig, ...]:
        """Every model, in registration order (the ``/v1/models`` order)."""
        return tuple(self._by_id.values())

    def by_tier(self, tier: ModelTier) -> tuple[ModelConfig, ...]:
        """Models of one tier, in registration order (the fallback order)."""
        return tuple(m for m in self._by_id.values() if m.tier == tier)


def default_registry(
    *,
    high_quality_model: str,
    budget_model: str,
    embedding_model: str,
) -> ProviderRegistry:
    """One model per tier, with the simulated prices listed in ``docs/configuration.md``."""
    return ProviderRegistry(
        [
            ModelConfig(
                model_id=high_quality_model,
                tier=ModelTier.HIGH_QUALITY,
                cost_per_1k_input_usd=0.003,
                cost_per_1k_output_usd=0.015,
            ),
            ModelConfig(
                model_id=budget_model,
                tier=ModelTier.BUDGET,
                cost_per_1k_input_usd=0.0001,
                cost_per_1k_output_usd=0.0002,
            ),
            ModelConfig(
                model_id=embedding_model,
                tier=ModelTier.EMBEDDINGS,
                cost_per_1k_input_usd=0.0001,
                cost_per_1k_output_usd=0.0,
                is_embedding_model=True,
            ),
        ]
    )
