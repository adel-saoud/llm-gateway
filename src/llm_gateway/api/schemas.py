"""Pydantic schemas for the OpenAI-compatible API surface.

Request schemas use ``extra="ignore"`` so unknown OpenAI fields (e.g.
``top_p``, ``frequency_penalty``) pass through silently: the gateway aims to
be a drop-in replacement, not a strict validator of OpenAI's full surface.

Admin schemas use ``extra="forbid"`` so a typo in a JSON body fails loudly
instead of silently dropping a field.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Chat completions ────────────────────────────────────────────────────────


class ChatMessageRequest(BaseModel):
    """One turn in the request's ``messages`` array."""

    role: str = Field(pattern="^(system|user|assistant|tool)$")
    content: str


# OpenAI's ``reasoning_effort`` values for chat completions.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` body: OpenAI-compatible (tolerant)."""

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessageRequest] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    reasoning_effort: ReasoningEffort | None = None
    """OpenAI's reasoning control. Absent or ``"none"`` turns thinking off.

    Thinking models (Qwen3.5) think by default, and a two-word answer then costs
    hundreds of hidden tokens, so the gateway turns it off unless the client asks.
    Any other value turns it on; Ollama's thinking-capable models other than
    gpt-oss take thinking as on or off, so the level itself is not forwarded.
    """
    # OpenAI's end-user id. Part of the cache partition, so one end user's
    # cached answer is never served to another user of the same team.
    user: str | None = None

    @property
    def think(self) -> bool:
        """Whether the model should think before answering."""
        return self.reasoning_effort not in {None, "none"}


class ChatMessageResponse(BaseModel):
    """The assistant turn in a non-streaming completion."""

    role: str
    content: str


class Usage(BaseModel):
    """OpenAI ``usage`` block; the same token counts the cost ledger is charged with."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """One completion choice; the gateway always returns exactly one (``n`` is ignored)."""

    index: int
    message: ChatMessageResponse
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """``POST /v1/chat/completions`` non-streaming body, in OpenAI's shape."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ── Embeddings ──────────────────────────────────────────────────────────────


class EmbeddingsRequest(BaseModel):
    """``POST /v1/embeddings`` body: a single string, the shape the cost model prices."""

    model_config = ConfigDict(extra="ignore")

    model: str
    input: str = Field(min_length=1)


class EmbeddingsUsage(BaseModel):
    """OpenAI embeddings ``usage`` block; embeddings have input tokens only."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingData(BaseModel):
    """One embedding vector; ``index`` is always 0 since the request holds one input."""

    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingsResponse(BaseModel):
    """``POST /v1/embeddings`` body, in OpenAI's shape."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingsUsage


# ── RAG cost report ─────────────────────────────────────────────────────────


class CostReportResponse(BaseModel):
    """``GET /v1/rag/cost-report`` body: one team, one UTC day."""

    team: str
    date: str
    total_cost_usd: float
    by_operation: Mapping[str, float]


# ── Models listing ──────────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    """One ``GET /v1/models`` entry; prices are the simulated ones the ledger uses."""

    id: str
    tier: str
    cost_per_1k_input_usd: float
    cost_per_1k_output_usd: float
    is_embedding_model: bool


class ModelsListResponse(BaseModel):
    """``GET /v1/models`` body, an OpenAI-style list."""

    object: str = "list"
    data: list[ModelInfo]


# ── Stats ───────────────────────────────────────────────────────────────────


class StatsResponse(BaseModel):
    """``GET /v1/stats`` body: the caller's budget position for today."""

    team: str
    today_cost_usd: float
    budget_utilisation: float
    daily_cap_usd: float
    by_operation: Mapping[str, float]


# ── Admin ───────────────────────────────────────────────────────────────────


class RateLimitUpdate(BaseModel):
    """Partial rate-limit change; an omitted field keeps its current value."""

    model_config = ConfigDict(extra="forbid")

    requests_per_minute: int | None = Field(default=None, ge=1)
    burst_capacity: int | None = Field(default=None, ge=1)


class BudgetUpdate(BaseModel):
    """Partial budget change; an omitted cap keeps the current one."""

    model_config = ConfigDict(extra="forbid")

    daily_usd: float | None = Field(default=None, gt=0.0)


class TeamUpdateRequest(BaseModel):
    """``PUT /v1/admin/teams/{team_id}`` body; only the fields sent are changed."""

    model_config = ConfigDict(extra="forbid")

    rate_limits: RateLimitUpdate | None = None
    budget: BudgetUpdate | None = None
    allowed_models: tuple[str, ...] | None = None


class TeamUpdateResponse(BaseModel):
    """The team's limits after the update (the API key is never echoed)."""

    team_id: str
    allowed_models: tuple[str, ...]
    rate_limits: dict[str, int]
    budget: dict[str, float]
