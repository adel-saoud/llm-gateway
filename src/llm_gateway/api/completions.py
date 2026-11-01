"""``POST /v1/chat/completions``: OpenAI-compatible chat completions.

Both streaming (``stream=true``) and non-streaming modes are supported.
Streaming uses the OpenAI Server-Sent Events shape: ``data: {chunk_json}\\n\\n``
lines followed by a final ``data: [DONE]\\n\\n``.

The handler composes every gateway-level concern in a single place:

1. **Authorization**: model whitelist enforced from :class:`TeamConfig`.
2. **Budget enforcement**: :class:`BudgetEnforcer.check` before the call.
3. **RAG operation tagging**: :func:`classify` from request signals.
4. **Semantic cache**: :class:`CompletionCache` lookup before routing, so a
   hit also skips the routing call. Hits are not recorded in the cost ledger:
   nothing was spent on a completion.
5. **Routing**: the configured :class:`RoutingPolicy` turns the requested
   model (or ``"auto"``) into a fallback chain, filtered by the team whitelist.
6. **Cost recording**: :class:`BudgetTracker.record` after the call.
7. **Metrics**: tokens / cost / RAG event / budget gauge / cache / routing.
8. **Tracing**: one OTel GenAI ``chat`` span around the upstream call (cache
   hits make no upstream call, so they emit none).
9. **Response headers**: ``X-Gateway-*`` set on the response so clients can
   observe routing and cache decisions without parsing the body.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from llm_gateway.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    Choice,
    Usage,
)
from llm_gateway.cache.completion_cache import (
    X_GATEWAY_CACHE_HEADER,
    X_GATEWAY_CACHE_SIMILARITY_HEADER,
    CacheLookup,
)
from llm_gateway.cache.semantic import CachedCompletion, CacheHit, build_cache_key
from llm_gateway.observability.middleware import UPSTREAM_SECONDS_STATE_KEY
from llm_gateway.observability.tracing import GenAiOperationName, GenAiSpan, start_genai_span
from llm_gateway.providers.base import ChatMessage, ProviderError
from llm_gateway.providers.router import AllProvidersUnavailableError, RouteCandidate
from llm_gateway.rag.models import RagOperationType
from llm_gateway.rag.tagger import X_RAG_OPERATION_HEADER, classify
from llm_gateway.routing.policy import (
    AUTO_MODEL,
    RoutingPlan,
    RoutingPolicy,
    RoutingRequest,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from llm_gateway.auth.models import TeamConfig
    from llm_gateway.budget.enforcer import BudgetEnforcer
    from llm_gateway.budget.tracker import BudgetTracker
    from llm_gateway.cache.completion_cache import CompletionCache
    from llm_gateway.config import Settings
    from llm_gateway.observability.metrics import Metrics
    from llm_gateway.providers.base import Provider
    from llm_gateway.providers.registry import ModelConfig, ProviderRegistry
    from llm_gateway.providers.router import ProviderRouter

router = APIRouter(tags=["completions"])

X_GATEWAY_ROUTING_REASON_HEADER = "X-Gateway-Routing-Reason"
"""Carries :attr:`RoutingPlan.reason`, so clients see why a model chain was chosen."""


def _gateway_headers(
    *, model: str, cost_usd: float, operation: RagOperationType, request_id: str
) -> dict[str, str]:
    """``X-Gateway-*`` headers of a completed request; ``request_id`` matches the ledger row."""
    return {
        "X-Gateway-Model-Used": model,
        "X-Gateway-Cost-USD": f"{cost_usd:.6f}",
        "X-Gateway-Rag-Operation": operation.value,
        "X-Gateway-Request-Id": request_id,
    }


def _validate_model(team: TeamConfig, registry: ProviderRegistry, model_id: str) -> ModelConfig:
    """Config of a model the team may call; 403 before 404 so the whitelist is checked first."""
    if not team.is_model_allowed(model_id, registry.tier_of(model_id)):
        raise HTTPException(
            status_code=403,
            detail=f"team '{team.team_id}' is not allowed to use model '{model_id}'",
        )
    if not registry.has(model_id):
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
    return registry.get(model_id)


def _classify_request(
    *, header_value: str | None, model_config: ModelConfig | None, messages: list[ChatMessage]
) -> RagOperationType:
    """Tag the request; ``model_config`` is None for ``"auto"``, which never targets an embedder."""
    system_prompts = [m.content for m in messages if m.role == "system"]
    return classify(
        header_value=header_value,
        is_embedding_model=model_config.is_embedding_model if model_config else False,
        system_prompts=system_prompts,
    )


def _cost_or_zero(
    registry: ProviderRegistry, model_id: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Simulated cost, or 0 for a model the registry no longer knows (e.g. a stale cache entry)."""
    if not registry.has(model_id):
        return 0.0
    return registry.get(model_id).cost_for(prompt_tokens, completion_tokens)


def _record_cache_lookup(
    *,
    metrics: Metrics,
    registry: ProviderRegistry,
    team: TeamConfig,
    model: str,
    lookup: CacheLookup,
) -> None:
    """Record the lookup's result, latency, similarity, embedding cost and cost saved."""
    embedding = lookup.embedding_result
    lookup_cost = (
        _cost_or_zero(registry, embedding.model, embedding.usage.prompt_tokens, 0)
        if embedding is not None
        else 0.0
    )
    cost_saved_usd = 0.0
    if lookup.hit is not None:
        cached = lookup.hit.completion
        cost_saved_usd = _cost_or_zero(
            registry, cached.model, cached.prompt_tokens, cached.completion_tokens
        )
    metrics.record_cache_lookup(
        team=team.team_id,
        model=model,
        result=lookup.result,
        duration_s=lookup.duration_s,
        top_similarity=lookup.top_similarity,
        lookup_cost_usd=lookup_cost,
        cost_saved_usd=cost_saved_usd,
    )


def _cached_response(
    hit: CacheHit, *, operation: RagOperationType, request_id: str, request: Request
) -> Response:
    """Replay a cached completion with the usage it originally cost."""
    setattr(request.state, UPSTREAM_SECONDS_STATE_KEY, 0.0)
    cached = hit.completion
    headers = _gateway_headers(
        model=cached.model, cost_usd=0.0, operation=operation, request_id=request_id
    )
    headers[X_GATEWAY_CACHE_HEADER] = "hit"
    headers[X_GATEWAY_CACHE_SIMILARITY_HEADER] = f"{hit.similarity:.4f}"
    payload = ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=int(time.time()),
        model=cached.model,
        choices=[
            Choice(
                index=0,
                message=ChatMessageResponse(role="assistant", content=cached.content),
                finish_reason=cached.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=cached.prompt_tokens,
            completion_tokens=cached.completion_tokens,
            total_tokens=cached.prompt_tokens + cached.completion_tokens,
        ),
    )
    return Response(
        content=payload.model_dump_json(), media_type="application/json", headers=headers
    )


def _route_chain(
    plan: RoutingPlan, *, team: TeamConfig, registry: ProviderRegistry, provider: Provider
) -> list[RouteCandidate]:
    """The plan's models the team may use, in order; 403 if none are left."""
    allowed = [
        m
        for m in plan.model_ids
        if registry.has(m) and team.is_model_allowed(m, registry.tier_of(m))
    ]
    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=f"team '{team.team_id}' is not allowed any model the router chose",
        )
    return [RouteCandidate(provider=provider, model_id=m) for m in allowed]


async def _check_budget(enforcer: BudgetEnforcer, team: TeamConfig, metrics: Metrics) -> None:
    """Refresh the utilisation gauge and raise 429 once today's spend reaches the cap."""
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
    """OpenAI-compatible chat completion, streamed as SSE when ``stream`` is true."""
    team: TeamConfig = request.state.team
    state = request.app.state
    registry: ProviderRegistry = state.registry
    enforcer: BudgetEnforcer = state.budget_enforcer
    tracker: BudgetTracker = state.budget_tracker
    metrics: Metrics = state.metrics
    provider: Provider = state.provider
    provider_router: ProviderRouter = state.provider_router
    completion_cache: CompletionCache = state.completion_cache
    routing_policy: RoutingPolicy = state.routing_policy
    settings: Settings = state.settings
    tracer: Tracer = state.tracer

    model_config = None if body.model == AUTO_MODEL else _validate_model(team, registry, body.model)
    await _check_budget(enforcer, team, metrics)

    chat_messages = [ChatMessage(role=m.role, content=m.content) for m in body.messages]
    operation = _classify_request(
        header_value=request.headers.get(X_RAG_OPERATION_HEADER),
        model_config=model_config,
        messages=chat_messages,
    )
    request_id = str(uuid.uuid4())

    bypass = completion_cache.bypass_reason(
        team=team,
        stream=body.stream,
        temperature=body.temperature,
        think=body.think,
        cache_control=request.headers.get("cache-control"),
    )
    cache_key = None
    lookup = None
    if bypass is not None:
        metrics.record_cache_bypass(team=team.team_id, model=body.model, reason=bypass)
    else:
        cache_key = build_cache_key(
            team_id=team.team_id, user=body.user, model=body.model, messages=chat_messages
        )
        lookup = await completion_cache.lookup(*cache_key)
        _record_cache_lookup(
            metrics=metrics, registry=registry, team=team, model=body.model, lookup=lookup
        )
        if lookup.hit is not None:
            return _cached_response(
                lookup.hit, operation=operation, request_id=request_id, request=request
            )

    plan = await routing_policy.plan(
        RoutingRequest(
            request_id=request_id,
            team_id=team.team_id,
            requested_model=body.model,
            messages=chat_messages,
        )
    )
    metrics.record_routing_decision(team=team.team_id, policy=plan.policy.value, reason=plan.reason)
    chain = _route_chain(plan, team=team, registry=registry, provider=provider)
    open_genai_span = partial(
        start_genai_span,
        tracer,
        operation=GenAiOperationName.CHAT,
        provider_name=provider.name,
        request_model=body.model,
        request_headers=request.headers,
        input_messages=chat_messages,
        capture_message_content=settings.tracing_capture_message_content,
    )

    if body.stream:
        return StreamingResponse(
            _stream_chat(
                open_span=open_genai_span,
                provider_router=provider_router,
                chain=chain,
                messages=chat_messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                think=body.think,
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
                X_GATEWAY_ROUTING_REASON_HEADER: plan.reason,
                X_GATEWAY_CACHE_HEADER: "bypass",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    try:
        with open_genai_span() as genai_span:
            result = await provider_router.chat(
                chain,
                chat_messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                think=body.think,
            )
            genai_span.record_chat_response(
                response_model=result.model,
                input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens,
                content=result.content,
                finish_reason=result.finish_reason,
            )
    except AllProvidersUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    setattr(request.state, UPSTREAM_SECONDS_STATE_KEY, result.latency_ms / 1000.0)

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
    headers[X_GATEWAY_ROUTING_REASON_HEADER] = plan.reason
    headers[X_GATEWAY_CACHE_HEADER] = "bypass" if lookup is None else "miss"
    for key, value in headers.items():
        response.headers[key] = value

    if cache_key is not None and lookup is not None:
        completion_cache.store(
            *cache_key,
            lookup,
            CachedCompletion(
                model=result.model,
                content=result.content,
                finish_reason=result.finish_reason,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
            ),
        )

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
    open_span: Callable[[], GenAiSpan],
    provider_router: ProviderRouter,
    chain: list[RouteCandidate],
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int | None,
    think: bool,
    team: TeamConfig,
    operation: RagOperationType,
    request_id: str,
    tracker: BudgetTracker,
    metrics: Metrics,
) -> AsyncGenerator[str]:
    """Stream :func:`_stream_chat_events` inside the request's GenAI span.

    The span is opened here, on the first iteration, not in the handler: a
    client that disconnects before the body starts closes this generator
    without running it, and a span started earlier would never be ended or
    exported. Once open, the span covers the whole stream and is ended when
    the generator exits, so a mid-stream disconnect (``GeneratorExit``) still
    closes it.
    """
    with open_span() as genai_span:
        async for sse_line in _stream_chat_events(
            provider_router=provider_router,
            chain=chain,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            team=team,
            operation=operation,
            request_id=request_id,
            tracker=tracker,
            metrics=metrics,
            genai_span=genai_span,
        ):
            yield sse_line


async def _stream_chat_events(
    *,
    provider_router: ProviderRouter,
    chain: list[RouteCandidate],
    messages: list[ChatMessage],
    temperature: float,
    max_tokens: int | None,
    think: bool,
    team: TeamConfig,
    operation: RagOperationType,
    request_id: str,
    tracker: BudgetTracker,
    metrics: Metrics,
    genai_span: GenAiSpan,
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
    content_parts: list[str] = []
    try:
        async for chunk in provider_router.stream_chat(
            chain, messages, temperature=temperature, max_tokens=max_tokens, think=think
        ):
            final_model = chunk.model or final_model
            if genai_span.capture_message_content and chunk.delta:
                content_parts.append(chunk.delta)
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
            if chunk.done and chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason
    except AllProvidersUnavailableError as exc:
        genai_span.record_error(exc)
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
    genai_span.record_chat_response(
        response_model=final_model,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        content="".join(content_parts),
        finish_reason=finish_reason,
    )
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
