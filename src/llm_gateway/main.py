"""FastAPI application factory.

The factory takes optional pre-built components so tests can inject hermetic
substitutes (fake Redis via fakeredis, in-memory ledger, mock provider).
Anything omitted is constructed from :class:`~llm_gateway.config.Settings`
at lifespan-start time.

Lifecycle is managed via :class:`~contextlib.AsyncExitStack`: each component
that owns an external resource (Ollama HTTP client, Redis connection pool) is
entered as an async context manager so cleanup is exception-safe and ordered.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Final

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis as AsyncRedis

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
from llm_gateway.circuit_breaker.monitor import FailureRateMonitor
from llm_gateway.circuit_breaker.state_machine import CircuitBreaker
from llm_gateway.config import LogLevel, Settings
from llm_gateway.observability.metrics import Metrics
from llm_gateway.observability.middleware import MetricsMiddleware
from llm_gateway.providers.base import Provider
from llm_gateway.providers.ollama import OllamaProvider
from llm_gateway.providers.registry import ProviderRegistry, default_registry
from llm_gateway.providers.router import ProviderRouter
from llm_gateway.ratelimit.middleware import RateLimitMiddleware
from llm_gateway.ratelimit.token_bucket import TokenBucket
from llm_gateway.storage.db import CircuitStateRepository, CostLedger

_LEVEL_MAP: Final[dict[LogLevel, int]] = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
}


def _configure_logging(level: LogLevel) -> None:
    numeric = _LEVEL_MAP[level]
    logging.basicConfig(level=numeric, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        cache_logger_on_first_use=True,
    )


# ── Per-component builders ──────────────────────────────────────────────────


@asynccontextmanager
async def _provider_cm(settings: Settings, override: Provider | None) -> AsyncGenerator[Provider]:
    """Yield the LLM provider; close any client we created on exit."""
    if override is not None:
        yield override
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
async def _redis_cm(settings: Settings) -> AsyncGenerator[AsyncRedis]:
    """Yield a redis.asyncio client; close it on exit."""
    client = AsyncRedis.from_url(settings.redis_url)  # pyright: ignore[reportUnknownMemberType]
    try:
        yield client
    finally:
        await client.aclose()


async def _build_ledger(settings: Settings, override: CostLedger | None) -> CostLedger:
    """Initialise the schema whether the ledger was supplied or built from settings.

    ``initialize()`` is idempotent (``CREATE TABLE IF NOT EXISTS``), so calling
    it on a ledger a test pre-seeded with rows is safe.
    """
    ledger = override or CostLedger(settings.db_path)
    await ledger.initialize()
    return ledger


def _build_registry(settings: Settings, override: ProviderRegistry | None) -> ProviderRegistry:
    if override is not None:
        return override
    return default_registry(
        high_quality_model=settings.model_high_quality,
        budget_model=settings.model_budget,
        embedding_model=settings.model_embeddings,
    )


def _build_team_resolver(settings: Settings, override: TeamResolver | None) -> TeamResolver:
    if override is not None:
        return override
    return TeamResolver.from_yaml(settings.teams_yaml)


def _build_circuit_breaker(
    settings: Settings,
    ledger: CostLedger,
    metrics: Metrics,
) -> CircuitBreaker:
    """Default ``CircuitBreaker`` sharing the ledger's SQLite file."""
    repo = CircuitStateRepository(ledger.db_path)
    monitor = FailureRateMonitor(window_size=max(40, settings.cb_min_requests * 4))
    return CircuitBreaker(
        repo,
        monitor,
        failure_rate_threshold=settings.cb_failure_rate_threshold,
        min_requests=settings.cb_min_requests,
        open_duration_s=settings.cb_open_duration_s,
        on_state_change=lambda p, m, s: metrics.set_circuit_state(provider=p, model=m, state=s),
    )


# ── App factory ─────────────────────────────────────────────────────────────


def create_app(
    settings: Settings | None = None,
    *,
    provider: Provider | None = None,
    team_resolver: TeamResolver | None = None,
    token_bucket: TokenBucket | None = None,
    cost_ledger: CostLedger | None = None,
    budget_enforcer: BudgetEnforcer | None = None,
    registry: ProviderRegistry | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    provider_router: ProviderRouter | None = None,
    metrics: Metrics | None = None,
) -> FastAPI:
    """Build a fresh ``FastAPI`` instance with all middleware wired up."""
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        _configure_logging(resolved_settings.log_level)
        log = structlog.get_logger(__name__)

        async with AsyncExitStack() as stack:
            _app.state.settings = resolved_settings
            _app.state.registry = _build_registry(resolved_settings, registry)
            _app.state.ollama = await stack.enter_async_context(
                _provider_cm(resolved_settings, provider)
            )
            _app.state.team_resolver = _build_team_resolver(resolved_settings, team_resolver)
            _app.state.cost_ledger = await _build_ledger(resolved_settings, cost_ledger)

            tracker = BudgetTracker(_app.state.cost_ledger, _app.state.registry)
            _app.state.budget_tracker = tracker
            _app.state.budget_enforcer = budget_enforcer or BudgetEnforcer(tracker)

            if token_bucket is not None:
                _app.state.token_bucket = token_bucket
            else:
                redis_client = await stack.enter_async_context(_redis_cm(resolved_settings))
                _app.state.redis = redis_client
                _app.state.token_bucket = TokenBucket(redis_client)

            _app.state.circuit_breaker = circuit_breaker or _build_circuit_breaker(
                resolved_settings, _app.state.cost_ledger, _app.state.metrics
            )
            _app.state.provider_router = provider_router or ProviderRouter(
                _app.state.circuit_breaker,
                on_fallback=lambda f, t, r: _app.state.metrics.record_fallback(
                    from_model=f, to_model=t, reason=r
                ),
            )

            log.info(
                "gateway_starting",
                ollama_base_url=resolved_settings.ollama_base_url,
                port=resolved_settings.port,
            )
            try:
                yield
            finally:
                log.info("gateway_stopped")

    application = FastAPI(
        title="LLM Gateway",
        description="Open-source LLM gateway with RAG-aware cost attribution.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    # ``Metrics`` is built once at create-app time so the middleware and the
    # ``/metrics`` endpoint share the same registry. Set it on app.state
    # immediately — middleware reads from there at request time.
    application.state.metrics = metrics or Metrics()

    # Middleware execution order is reverse of registration. Inside-out:
    # rate-limit → auth → metrics. Metrics runs outermost so it observes
    # rejections (401/429) as well as successful requests.
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
