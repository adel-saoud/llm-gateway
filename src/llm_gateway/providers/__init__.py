"""Provider layer — abstracts the wire contract behind a Protocol."""

from __future__ import annotations

from llm_gateway.providers.base import (
    ChatMessage,
    CompletionResult,
    EmbeddingResult,
    Provider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StreamChunk,
    TokenUsage,
)
from llm_gateway.providers.ollama import OllamaProvider
from llm_gateway.providers.registry import ModelConfig, ModelTier, ProviderRegistry
from llm_gateway.providers.router import (
    AllProvidersUnavailableError,
    ProviderRouter,
    RouteCandidate,
    RouteChain,
)

__all__ = [
    "AllProvidersUnavailableError",
    "ChatMessage",
    "CompletionResult",
    "EmbeddingResult",
    "ModelConfig",
    "ModelTier",
    "OllamaProvider",
    "Provider",
    "ProviderError",
    "ProviderRegistry",
    "ProviderRouter",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RouteCandidate",
    "RouteChain",
    "StreamChunk",
    "TokenUsage",
]
