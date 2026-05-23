"""Pydantic schemas for the OpenAI-compatible API surface.

Request schemas use ``extra="ignore"`` so unknown OpenAI fields (e.g.
``top_p``, ``frequency_penalty``) pass through silently — the gateway aims to
be a drop-in replacement, not a strict validator of OpenAI's full surface.

Admin schemas use ``extra="forbid"`` so a typo in a JSON body fails loudly
instead of silently dropping a field.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

# ── Chat completions ────────────────────────────────────────────────────────


class ChatMessageRequest(BaseModel):
    """One turn in the request's ``messages`` array."""

    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` body — OpenAI-compatible (tolerant)."""

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessageRequest] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False


class ChatMessageResponse(BaseModel):
    role: str
    content: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ── Embeddings ──────────────────────────────────────────────────────────────


class EmbeddingsRequest(BaseModel):
    """``POST /v1/embeddings`` body. We accept a single string for simplicity."""

    model_config = ConfigDict(extra="ignore")

    model: str
    input: str = Field(min_length=1)


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingsUsage


# ── RAG cost report ─────────────────────────────────────────────────────────


class CostReportResponse(BaseModel):
    team: str
    date: str
    total_cost_usd: float
    by_operation: Mapping[str, float]


# ── Models listing ──────────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    id: str
    tier: str
    cost_per_1k_input_usd: float
    cost_per_1k_output_usd: float
    is_embedding_model: bool


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ── Stats ───────────────────────────────────────────────────────────────────


class StatsResponse(BaseModel):
    team: str
    today_cost_usd: float
    budget_utilisation: float
    daily_cap_usd: float
    by_operation: Mapping[str, float]


# ── Admin ───────────────────────────────────────────────────────────────────


class RateLimitUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests_per_minute: int | None = Field(default=None, ge=1)
    burst_capacity: int | None = Field(default=None, ge=1)


class BudgetUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    daily_usd: float | None = Field(default=None, gt=0.0)


class TeamUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rate_limits: RateLimitUpdate | None = None
    budget: BudgetUpdate | None = None
    allowed_models: tuple[str, ...] | None = None


class TeamUpdateResponse(BaseModel):
    team_id: str
    allowed_models: tuple[str, ...]
    rate_limits: dict[str, int]
    budget: dict[str, float]
