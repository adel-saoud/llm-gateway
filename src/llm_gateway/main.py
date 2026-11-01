"""FastAPI application factory.

:func:`create_app` takes the few components that reach outside the process
(provider, Redis bucket, SQLite ledger, team file, routing sidecar, span
exporter) so tests can inject hermetic substitutes. Everything else is built
from :class:`~llm_gateway.config.Settings` when the lifespan starts.

Lifecycle is managed via :class:`~contextlib.AsyncExitStack`: each component
that owns an external resource (Ollama HTTP client, Redis connection pool,
autopilot client, span exporter) registers its cleanup there, so shutdown is
exception-safe and runs in reverse order of startup.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

import structlog
from fastapi import FastAPI
from opentelemetry.sdk.trace import TracerProvider
from redis.asyncio import Redis as AsyncRedis

from llm_gateway import __version__
from llm_gateway.api.admin import router as admin_router
from llm_gateway.api.completions import router as completions_router
from llm_gateway.api.embeddings import router as embeddings_router
from llm_gateway.api.health import router as health_router
from llm_gateway.api.info import router as info_router
from llm_gateway.api.metrics import router as metrics_router
from llm_gateway.api.rag import router as rag_router
from llm_gateway.auth.middleware import AuthMiddleware
from llm_gateway.auth.resolver import TeamResolver
from llm_gateway.budget.enforcer import BudgetEnforcer
from llm_gateway.budget.tracker import BudgetTracker
from llm_gateway.cache.completion_cache import CompletionCache
from llm_gateway.cache.semantic import SemanticCache
from llm_gateway.circuit_breaker.monitor import FailureRateMonitor
from llm_gateway.circuit_breaker.state_machine import CircuitBreaker
from llm_gateway.config import LogLevel, Settings
from llm_gateway.observability.metrics import Metrics
from llm_gateway.observability.middleware import MetricsMiddleware
from llm_gateway.observability.tracing import build_tracer_provider, get_tracer
from llm_gateway.providers.base import Provider
from llm_gateway.providers.ollama import OllamaProvider
from llm_gateway.providers.registry import ProviderRegistry, default_registry
from llm_gateway.providers.router import ProviderRouter, RouteCandidate
from llm_gateway.ratelimit.middleware import RateLimitMiddleware
from llm_gateway.ratelimit.token_bucket import TokenBucket
from llm_gateway.routing.policy import RoutingPolicy, RoutingPolicyName, StaticTierPolicy
from llm_gateway.storage.db import CircuitStateRepository, CostLedger


def _configure_logging(level: LogLevel) -> None:
    """Route stdlib and structlog output to stdout as one JSON object per line."""
    logging.basicConfig(level=level.value, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level.value),
        cache_logger_on_first_use=True,
    )


# ── Per-component builders ──────────────────────────────────────────────────


@asynccontextmanager
async def _open_provider(settings: Settings, injected: Provider | None) -> AsyncGenerator[Provider]:
    """Yield the LLM provider; close the HTTP client only if it was built here."""
    if injected is not None:
        yield injected
        return
    provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        timeout_s=settings.ollama_timeout_s,
    )
    try:
        yield provider
    finally:
        await provider.aclose()


@asynccontextmanager
async def _open_redis(settings: Settings) -> AsyncGenerator[AsyncRedis]:
    """Yield a redis.asyncio client; close its connection pool on exit."""
    client = AsyncRedis.from_url(settings.redis_url)  # pyright: ignore[reportUnknownMemberType]
    try:
        yield client
    finally:
        await client.aclose()


async def _build_ledger(settings: Settings, injected: CostLedger | None) -> CostLedger:
    """Initialise the schema whether the ledger was injected or built from settings.

    ``initialize()`` is idempotent (``CREATE TABLE IF NOT EXISTS``), so calling
    it on a ledger a test pre-seeded with rows is safe.
    """
    ledger = injected or CostLedger(settings.db_path)
    await ledger.initialize()
    return ledger


def _build_circuit_breaker(
    settings: Settings,
    ledger: CostLedger,
    metrics: Metrics,
) -> CircuitBreaker:
    """Circuit breaker persisting its state in the ledger's SQLite file.

    The rolling window holds four times ``min_requests`` (at least 40 outcomes)
    so one slow minute does not dominate the failure rate once the minimum
    sample count is reached.
    """
    state_repository = CircuitStateRepository(ledger.db_path)
    monitor = FailureRateMonitor(window_size=max(40, settings.circuit_breaker_min_requests * 4))
    return CircuitBreaker(
        state_repository,
        monitor,
        failure_rate_threshold=settings.circuit_breaker_failure_rate_threshold,
        min_requests=settings.circuit_breaker_min_requests,
        open_duration_s=settings.circuit_breaker_open_duration_s,
        on_state_change=lambda provider_name, model_id, state: metrics.set_circuit_state(
            provider=provider_name, model=model_id, state=state
        ),
    )


def _build_completion_cache(
    settings: Settings, provider: Provider, provider_router: ProviderRouter
) -> CompletionCache:
    """Cache whose lookups embed through the router, so they get circuit breaking too."""
    store = SemanticCache(
        similarity_threshold=settings.cache_similarity_threshold,
        ttl_s=settings.cache_ttl_s,
        max_entries_per_partition=settings.cache_max_entries_per_partition,
        max_partitions=settings.cache_max_partitions,
    )
    embedding_chain = [RouteCandidate(provider=provider, model_id=settings.model_embeddings)]
    return CompletionCache(
        store,
        embed=lambda text: provider_router.embed(embedding_chain, text),
        enabled=settings.cache_enabled,
        semantic_enabled=settings.cache_semantic_enabled,
        embedding_prefix=settings.cache_embedding_prefix,
        max_temperature=settings.cache_max_temperature,
    )


def _build_routing_policy(
    settings: Settings, registry: ProviderRegistry, stack: AsyncExitStack
) -> RoutingPolicy:
    """Static by default; the autopilot adapter is imported only when selected."""
    static_policy = StaticTierPolicy(registry)
    if settings.routing_policy is RoutingPolicyName.STATIC:
        return static_policy
    from llm_gateway.routing.autopilot import AutopilotClient, AutopilotPolicy

    api_token = settings.autopilot_api_token
    client = AutopilotClient(
        base_url=settings.autopilot_base_url,
        timeout_s=settings.autopilot_timeout_s,
        api_token=api_token.get_secret_value() if api_token is not None else None,
    )
    stack.push_async_callback(client.aclose)
    return AutopilotPolicy(
        client=client,
        registry=registry,
        fallback=static_policy,
        tier_map={
            "simple": settings.autopilot_tier_simple,
            "moderate": settings.autopilot_tier_moderate,
        },
        timeout_s=settings.autopilot_timeout_s,
    )


def _build_tracer_provider(
    settings: Settings, injected: TracerProvider | None, stack: AsyncExitStack
) -> TracerProvider | None:
    """Injected provider (tests) as is; else an OTLP one when tracing is on.

    A provider built here is shut down with the app so batched spans are
    flushed; an injected one belongs to its caller.
    """
    if injected is not None:
        return injected
    if not settings.tracing_enabled:
        return None
    tracer_provider = build_tracer_provider()
    stack.callback(tracer_provider.shutdown)
    return tracer_provider


# ── App factory ─────────────────────────────────────────────────────────────


def create_app(
    settings: Settings | None = None,
    *,
    provider: Provider | None = None,
    team_resolver: TeamResolver | None = None,
    token_bucket: TokenBucket | None = None,
    cost_ledger: CostLedger | None = None,
    routing_policy: RoutingPolicy | None = None,
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    """Build a FastAPI app with middleware, routes and a lifespan that wires ``app.state``.

    Every argument left as ``None`` is built from ``settings`` (or the
    environment) at startup. Handlers and middleware read their collaborators
    from ``app.state``, never from module globals, so two apps in one process
    (tests) never share state.
    """
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
        _configure_logging(resolved_settings.log_level)
        log = structlog.get_logger(__name__)
        state = app.state

        async with AsyncExitStack() as stack:
            state.settings = resolved_settings
            state.registry = default_registry(
                high_quality_model=resolved_settings.model_high_quality,
                budget_model=resolved_settings.model_budget,
                embedding_model=resolved_settings.model_embeddings,
            )
            state.provider = await stack.enter_async_context(
                _open_provider(resolved_settings, provider)
            )
            state.team_resolver = team_resolver or TeamResolver.from_yaml(
                resolved_settings.teams_yaml
            )
            state.cost_ledger = await _build_ledger(resolved_settings, cost_ledger)
            await stack.enter_async_context(state.cost_ledger.keep_wal_open())

            state.budget_tracker = BudgetTracker(state.cost_ledger, state.registry)
            state.budget_enforcer = BudgetEnforcer(state.budget_tracker)

            if token_bucket is not None:
                state.token_bucket = token_bucket
            else:
                redis_client = await stack.enter_async_context(_open_redis(resolved_settings))
                state.token_bucket = TokenBucket(redis_client)

            state.circuit_breaker = _build_circuit_breaker(
                resolved_settings, state.cost_ledger, state.metrics
            )
            state.provider_router = ProviderRouter(
                state.circuit_breaker,
                on_fallback=lambda from_model, to_model, reason: state.metrics.record_fallback(
                    from_model=from_model, to_model=to_model, reason=reason
                ),
            )
            state.completion_cache = _build_completion_cache(
                resolved_settings, state.provider, state.provider_router
            )
            state.routing_policy = routing_policy or _build_routing_policy(
                resolved_settings, state.registry, stack
            )
            state.tracer = get_tracer(
                _build_tracer_provider(resolved_settings, tracer_provider, stack)
            )

            log.info(
                "gateway_starting",
                ollama_base_url=resolved_settings.ollama_base_url,
                routing_policy=resolved_settings.routing_policy.value,
                cache_enabled=resolved_settings.cache_enabled,
                tracing_enabled=resolved_settings.tracing_enabled,
            )
            try:
                yield
            finally:
                log.info("gateway_stopped")

    application = FastAPI(
        title="LLM Gateway",
        description="Open-source LLM gateway with RAG-aware cost attribution.",
        version=__version__,
        lifespan=_lifespan,
    )

    # Built here rather than in the lifespan: the middleware reads it on every
    # request and the /metrics endpoint renders the same registry.
    application.state.metrics = Metrics()

    # Starlette runs middleware in reverse order of registration, so requests
    # go metrics -> auth -> rate limit. Metrics is outermost so it also counts
    # the 401 and 429 rejections.
    application.add_middleware(RateLimitMiddleware)
    application.add_middleware(AuthMiddleware)
    application.add_middleware(MetricsMiddleware)

    application.include_router(health_router)
    application.include_router(metrics_router)
    application.include_router(completions_router)
    application.include_router(embeddings_router)
    application.include_router(rag_router)
    application.include_router(info_router)
    application.include_router(admin_router)
    return application


app = create_app()
"""ASGI entry point for ``uvicorn llm_gateway.main:app``."""
