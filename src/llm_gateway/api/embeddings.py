"""``POST /v1/embeddings`` — OpenAI-compatible single-input embeddings.

Embedding requests are always tagged as :data:`RagOperationType.RETRIEVAL` by
the classifier (the embedding-model signal wins), so the per-operation cost
breakdown attributes them to retrieval.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, cast

import structlog
from fastapi import APIRouter, HTTPException, Request, Response

from llm_gateway.api.schemas import (
    EmbeddingData,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
)
from llm_gateway.providers.base import ProviderError
from llm_gateway.providers.router import AllProvidersUnavailableError, RouteCandidate
from llm_gateway.rag.models import RagOperationType
from llm_gateway.rag.tagger import X_RAG_OPERATION_HEADER, classify

if TYPE_CHECKING:
    from llm_gateway.auth.models import TeamConfig
    from llm_gateway.budget.enforcer import BudgetEnforcer
    from llm_gateway.budget.tracker import BudgetTracker
    from llm_gateway.observability.metrics import Metrics
    from llm_gateway.providers.base import Provider
    from llm_gateway.providers.registry import ProviderRegistry
    from llm_gateway.providers.router import ProviderRouter

_log = structlog.get_logger(__name__)

router = APIRouter(tags=["embeddings"])


@router.post("/v1/embeddings")
async def embeddings(
    body: EmbeddingsRequest,
    request: Request,
    response: Response,
) -> Response:
    team: TeamConfig = request.state.team
    state = request.app.state
    registry: ProviderRegistry = state.registry
    enforcer: BudgetEnforcer = state.budget_enforcer
    tracker: BudgetTracker = state.budget_tracker
    metrics: Metrics = state.metrics
    provider: Provider = state.ollama
    provider_router: ProviderRouter = state.provider_router

    if not team.is_model_allowed(body.model):
        raise HTTPException(
            status_code=403,
            detail=f"team '{team.team_id}' is not allowed to use model '{body.model}'",
        )
    if not registry.has(body.model):
        raise HTTPException(status_code=404, detail=f"unknown model: {body.model}")
    model_config = registry.get(body.model)

    decision = await enforcer.check(team)
    metrics.set_budget_utilisation(team=team.team_id, utilisation=decision.utilisation)
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "budget_exceeded",
                "today_usd": decision.today_usd,
                "daily_cap_usd": decision.daily_cap_usd,
            },
        )

    operation = classify(
        header_value=request.headers.get(X_RAG_OPERATION_HEADER),
        is_embedding_model=model_config.is_embedding_model,
        system_prompts=(),
    )
    # Embedding calls are RETRIEVAL by default; the header can still override.
    if operation is RagOperationType.GENERATION:
        operation = RagOperationType.RETRIEVAL

    chain = [RouteCandidate(provider=provider, model_id=body.model)]
    request_id = str(uuid.uuid4())

    try:
        result = await provider_router.embed(chain, body.input)
    except AllProvidersUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    event = await tracker.record(
        team_id=team.team_id,
        model_id=result.model,
        operation_type=operation,
        input_tokens=result.usage.prompt_tokens,
        output_tokens=0,
        request_id=request_id,
    )

    metrics.record_tokens(
        team=team.team_id,
        model=result.model,
        prompt_tokens=event.input_tokens,
        completion_tokens=0,
    )
    metrics.record_cost(team=team.team_id, model=result.model, cost_usd=event.cost_usd)
    metrics.record_rag_event(
        team=team.team_id,
        operation_type=event.operation_type,
        cost_usd=event.cost_usd,
        prompt_tokens=event.input_tokens,
        completion_tokens=0,
        duration_s=result.latency_ms / 1000.0,
    )

    response.headers["X-Gateway-Model-Used"] = result.model
    response.headers["X-Gateway-Cost-USD"] = f"{event.cost_usd:.6f}"
    response.headers["X-Gateway-Rag-Operation"] = event.operation_type.value
    response.headers["X-Gateway-Request-Id"] = request_id

    payload = EmbeddingsResponse(
        data=[EmbeddingData(index=0, embedding=list(result.embedding))],
        model=result.model,
        usage=EmbeddingsUsage(
            prompt_tokens=result.usage.prompt_tokens,
            total_tokens=result.usage.prompt_tokens,
        ),
    )
    return Response(
        content=payload.model_dump_json(),
        media_type="application/json",
        headers=cast(dict[str, str], response.headers),
    )
