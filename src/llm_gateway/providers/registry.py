"""Model registry — declarative inventory of routable models and their simulated costs.

Costs are intentionally synthetic — see ``context/decisions/004-simulated-costs.md``.
The routing logic, the savings %, and the per-operation attribution are real; the
dollar amounts are illustrative so the gateway can run free-tier on Ollama and still
demonstrate the production behaviour an OpenAI / Anthropic deployment would exhibit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum


class ModelTier(StrEnum):
    """Routing tier — drives the high-quality vs budget decision in the router."""

    HIGH_QUALITY = "high_quality"
    BUDGET = "budget"
    EMBEDDINGS = "embeddings"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Static configuration for one routable model."""

    model_id: str
    tier: ModelTier
    provider_name: str
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
        try:
            return self._by_id[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown model: {model_id}") from exc

    def has(self, model_id: str) -> bool:
        return model_id in self._by_id

    def all(self) -> tuple[ModelConfig, ...]:
        return tuple(self._by_id.values())

    def by_tier(self, tier: ModelTier) -> tuple[ModelConfig, ...]:
        return tuple(m for m in self._by_id.values() if m.tier == tier)


def default_registry(
    *,
    high_quality_model: str,
    budget_model: str,
    embedding_model: str,
) -> ProviderRegistry:
    """Build the registry matching the cost tiers in the brief.

    See ``context/decisions/004-simulated-costs.md`` for the prices.
    """
    return ProviderRegistry(
        [
            ModelConfig(
                model_id=high_quality_model,
                tier=ModelTier.HIGH_QUALITY,
                provider_name="ollama",
                cost_per_1k_input_usd=0.003,
                cost_per_1k_output_usd=0.015,
            ),
            ModelConfig(
                model_id=budget_model,
                tier=ModelTier.BUDGET,
                provider_name="ollama",
                cost_per_1k_input_usd=0.0001,
                cost_per_1k_output_usd=0.0002,
            ),
            ModelConfig(
                model_id=embedding_model,
                tier=ModelTier.EMBEDDINGS,
                provider_name="ollama",
                cost_per_1k_input_usd=0.0001,
                cost_per_1k_output_usd=0.0,
                is_embedding_model=True,
            ),
        ]
    )
