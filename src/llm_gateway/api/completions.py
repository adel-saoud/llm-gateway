"""``POST /v1/chat/completions`` — OpenAI-compatible chat completions.

Both streaming (``stream=true``) and non-streaming modes are supported.
Streaming uses the OpenAI Server-Sent Events shape: ``data: {chunk_json}\\n\\n``
lines followed by a final ``data: [DONE]\\n\\n``.

The handler composes every gateway-level concern in a single place:

1. **Authorization** — model whitelist enforced from :class:`TeamConfig`.
2. **Budget enforcement** — :class:`BudgetEnforcer.check` before the call.
3. **RAG operation tagging** — :func:`classify` from request signals.
4. **Routing** — high-quality requests get a two-candidate chain (HQ → budget)
   so the router can fall back automatically if the primary model is unavailable.
5. **Cost recording** — :class:`BudgetTracker.record` after the call.
6. **Metrics** — tokens / cost / RAG event / budget gauge updates.
7. **Response headers** — ``X-Gateway-*`` set on the response so clients can
   observe routing decisions without parsing the body.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from llm_gateway.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    Choice,
    Usage,
)
from llm_gateway.providers.base import ChatMessage, ProviderError
from llm_gateway.providers.registry import ModelTier
from llm_gateway.providers.router import AllProvidersUnavailableError, RouteCandidate
from llm_gateway.rag.models import RagOperationType
from llm_gateway.rag.tagger import X_RAG_OPERATION_HEADER, classify

if TYPE_CHECKING:
    from llm_gateway.auth.models import TeamConfig
    from llm_gateway.budget.enforcer import BudgetEnforcer
    from llm_gateway.budget.tracker import BudgetTracker
    from llm_gateway.observability.metrics import Metrics
    from llm_gateway.providers.base import Provider
    from llm_gateway.providers.registry import ModelConfig, ProviderRegistry
    from llm_gateway.providers.router import ProviderRouter

_log = structlog.get_logger(__name__)

router = APIRouter(tags=["completions"])


def _gateway_headers(
    *, model: str, cost_usd: float, operation: RagOperationType, request_id: str
) -> dict[str, str]:
    return {
        "X-Gateway-Model-Used": model,
        "X-Gateway-Cost-USD": f"{cost_usd:.6f}",
        "X-Gateway-Rag-Operation": operation.value,
        "X-Gateway-Request-Id": request_id,
    }


def _validate_model(team: TeamConfig, registry: ProviderRegistry, model_id: str) -> ModelConfig:
    if not team.is_model_allowed(model_id):
        raise HTTPException(
            status_code=403,
            detail=f"team '{team.team_id}' is not allowed to use model '{model_id}'",
        )
    if not registry.has(model_id):
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    return registry.get(model_id)


def _classify_request(
    *, header_value: str | None, model_config: ModelConfig, messages: list[ChatMessage]
) -> RagOperationType:
    system_prompts = [m.content for m in messages if m.role == "system"]
    return classify(
        header_value=header_value,
        is_embedding_model=model_config.is_embedding_model,
        system_prompts=system_prompts,
    )


async def _check_budget(enforcer: BudgetEnforcer, team: TeamConfig, metrics: Metrics) -> None:
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


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
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

    model_config = _validate_model(team, registry, body.model)
    await _check_budget(enforcer, team, metrics)

    chat_messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    operation = _classify_request(
        header_value=request.headers.get(X_RAG_OPERATION_HEADER),
        model_config=model_config,
        messages=chat_messages,
    )

    chain: list[RouteCandidate] = [RouteCandidate(provider=provider, model_id=body.model)]
    if model_config.tier is ModelTier.HIGH_QUALITY:
        for fallback in registry.by_tier(ModelTier.BUDGET):
            chain.append(RouteCandidate(provider=provider, model_id=fallback.model_id))
    request_id = str(uuid.uuid4())

    if body.stream:
        return StreamingResponse(
            _stream_chat(
                provider_router=provider_router,
                chain=chain,
                messages=chat_messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                team=team,
                operation=operation,
                request_id=request_id,
                tracker=tracker,
                metrics=metrics,
            ),
            media_type="text/event-stream",
            headers={
                "X-Gateway-Rag-Operation": operation.value,
                "X-Gateway-Request-Id": request_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        result = await provider_router.chat(
            chain,
            chat_messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except AllProvidersUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    event = await tracker.record(
        team_id=team.team_id,
        model_id=result.model,
        operation_type=operation,
        input_tokens=result.usage.prompt_tokens,
        output_tokens=result.usage.completion_tokens,
        request_id=request_id,
    )

    metrics.record_tokens(
        team=team.team_id,
        model=result.model,
        prompt_tokens=event.input_tokens,
        completion_tokens=event.output_tokens,
    )
    metrics.record_cost(team=team.team_id, model=result.model, cost_usd=event.cost_usd)
    metrics.record_rag_event(
        team=team.team_id,
        operation_type=event.operation_type,
        cost_usd=event.cost_usd,
        prompt_tokens=event.input_tokens,
        completion_tokens=event.output_tokens,
        duration_s=result.latency_ms / 1000.0,
    )

    headers = _gateway_headers(
        model=result.model,
        cost_usd=event.cost_usd,
        operation=event.operation_type,
        request_id=request_id,
    )
    for key, value in headers.items():
        response.headers[key] = value

    payload = ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=int(time.time()),
        model=result.model,
        choices=[
            Choice(
                index=0,
                message=ChatMessageResponse(role="assistant", content=result.content),
                finish_reason=result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
    )
    return Response(
        content=payload.model_dump_json(),
        media_type="application/json",
        headers=cast(dict[str, str], response.headers),
    )


async def _stream_chat(
    *,
    provider_router: ProviderRouter,
    chain: list[RouteCandidate],
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int | None,
    team: TeamConfig,
    operation: RagOperationType,
    request_id: str,
    tracker: BudgetTracker,
    metrics: Metrics,
) -> AsyncGenerator[str]:
    """Yield OpenAI-format SSE chunks; record cost from the final chunk's usage."""
    completion_id = f"chatcmpl-{request_id}"
    created = int(time.time())
    start = time.perf_counter()
    final_model = chain[0].model_id
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = "stop"
    first_chunk = True
    try:
        async for chunk in provider_router.stream_chat(
            chain, messages, temperature=temperature, max_tokens=max_tokens
        ):
            final_model = chunk.model or final_model
            if first_chunk and chunk.delta:
                # OpenAI streams an initial role-only chunk; we fold the role
                # into the first content chunk to keep the wire format simple.
                first_chunk = False
                opening = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": final_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": chunk.delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(opening)}\n\n"
            elif chunk.delta:
                body = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": final_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(body)}\n\n"
            if chunk.done and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
    except AllProvidersUnavailableError as exc:
        error_payload = {"error": {"message": str(exc), "type": "providers_unavailable"}}
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
        return

    final: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": final_model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"

    duration_s = time.perf_counter() - start
    event = await tracker.record(
        team_id=team.team_id,
        model_id=final_model,
        operation_type=operation,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        request_id=request_id,
    )
    metrics.record_tokens(
        team=team.team_id,
        model=final_model,
        prompt_tokens=event.input_tokens,
        completion_tokens=event.output_tokens,
    )
    metrics.record_cost(team=team.team_id, model=final_model, cost_usd=event.cost_usd)
    metrics.record_rag_event(
        team=team.team_id,
        operation_type=event.operation_type,
        cost_usd=event.cost_usd,
        prompt_tokens=event.input_tokens,
        completion_tokens=event.output_tokens,
        duration_s=duration_s,
    )
