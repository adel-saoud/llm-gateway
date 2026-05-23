from __future__ import annotations

import pytest

from llm_gateway.providers.registry import (
    ModelConfig,
    ModelTier,
    ProviderRegistry,
    default_registry,
)


def test_default_registry_has_three_models() -> None:
    reg = default_registry(
        high_quality_model="llama3.2:3b",
        budget_model="qwen2.5:0.5b",
        embedding_model="nomic-embed-text",
    )
    assert len(reg.all()) == 3
    assert reg.get("llama3.2:3b").tier == ModelTier.HIGH_QUALITY
    assert reg.get("qwen2.5:0.5b").tier == ModelTier.BUDGET
    assert reg.get("nomic-embed-text").tier == ModelTier.EMBEDDINGS
    assert reg.get("nomic-embed-text").is_embedding_model is True


def test_by_tier_filters() -> None:
    reg = default_registry(
        high_quality_model="llama3.2:3b",
        budget_model="qwen2.5:0.5b",
        embedding_model="nomic-embed-text",
    )
    high = reg.by_tier(ModelTier.HIGH_QUALITY)
    assert len(high) == 1
    assert high[0].model_id == "llama3.2:3b"


def test_unknown_model_raises_keyerror() -> None:
    reg = default_registry(
        high_quality_model="llama3.2:3b",
        budget_model="qwen2.5:0.5b",
        embedding_model="nomic-embed-text",
    )
    assert reg.has("does-not-exist") is False
    with pytest.raises(KeyError, match="unknown model"):
        reg.get("does-not-exist")


def test_duplicate_model_id_rejected() -> None:
    dup = ModelConfig(
        model_id="x",
        tier=ModelTier.BUDGET,
        provider_name="ollama",
        cost_per_1k_input_usd=0.0,
        cost_per_1k_output_usd=0.0,
    )
    with pytest.raises(ValueError, match="duplicate model_id"):
        ProviderRegistry([dup, dup])


def test_cost_for_combines_input_and_output() -> None:
    m = ModelConfig(
        model_id="x",
        tier=ModelTier.HIGH_QUALITY,
        provider_name="ollama",
        cost_per_1k_input_usd=0.003,
        cost_per_1k_output_usd=0.015,
    )
    # 1000 input + 1000 output → 0.003 + 0.015 = 0.018
    expected_full = pytest.approx(0.018)  # pyright: ignore[reportUnknownMemberType]
    assert m.cost_for(1000, 1000) == expected_full
    # 500 input + 500 output → 0.0015 + 0.0075 = 0.009
    expected_half = pytest.approx(0.009)  # pyright: ignore[reportUnknownMemberType]
    assert m.cost_for(500, 500) == expected_half


def test_embedding_cost_is_input_only() -> None:
    m = ModelConfig(
        model_id="emb",
        tier=ModelTier.EMBEDDINGS,
        provider_name="ollama",
        cost_per_1k_input_usd=0.0001,
        cost_per_1k_output_usd=0.0,
        is_embedding_model=True,
    )
    expected = pytest.approx(0.0002)  # pyright: ignore[reportUnknownMemberType]
    assert m.cost_for(2000, 0) == expected
