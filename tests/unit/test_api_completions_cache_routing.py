"""Handler-level tests: semantic cache, ``model="auto"`` routing, Server-Timing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from llm_gateway.config import Settings
from llm_gateway.main import create_app
from llm_gateway.providers.base import ProviderUnavailableError
from llm_gateway.routing.autopilot import AutopilotPolicy
from llm_gateway.routing.policy import (
    AUTO_MODEL,
    RoutingPlan,
    RoutingPolicyName,
    RoutingRequest,
)
from tests.conftest import ScriptedProvider, gateway_client_factory, make_team

AUTH = {"Authorization": "Bearer sk-test"}


def _cache_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"cache_enabled": True, "cache_semantic_enabled": True})


def _chat(
    client: TestClient,
    content: str,
    *,
    model: str = "qwen3.5:4b",
    headers: dict[str, str] | None = None,
    **extra: Any,
) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        headers={**AUTH, **(headers or {})},
        json={"model": model, "messages": [{"role": "user", "content": content}], **extra},
    )


def _metric_value(client: TestClient, line_prefix: str) -> float:
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(line_prefix):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


# ── Semantic cache ────────────────────────────────────────────────────────


def test_paraphrase_is_served_from_cache(settings: Settings, tmp_path: Path) -> None:
    # ScriptedProvider returns the same embedding for every input, so any
    # single-turn prompt with the same literals is a semantic match.
    provider = ScriptedProvider(provider_name="ollama")
    team = make_team(semantic_cache=True)
    with gateway_client_factory(
        _cache_settings(settings), tmp_path, team=team, provider=provider
    ) as client:
        first = _chat(client, "what is the return policy?")
        second = _chat(client, "how do returns work?")
        saved = _metric_value(client, "llm_gateway_semantic_cache_cost_saved_usd_total{")
        hits = _metric_value(
            client,
            'llm_gateway_semantic_cache_requests_total{model="qwen3.5:4b",'
            'result="hit_semantic",team="t1"}',
        )
    assert first.headers["X-Gateway-Cache"] == "miss"
    assert second.status_code == 200
    assert second.headers["X-Gateway-Cache"] == "hit"
    assert float(second.headers["X-Gateway-Cache-Similarity"]) > 0.99
    assert second.headers["X-Gateway-Cost-USD"] == "0.000000"
    assert second.json()["choices"][0]["message"]["content"] == "scripted-response"
    assert provider.chat_calls == 1
    assert hits == 1.0
    assert saved > 0.0
    assert "gateway;dur=" in second.headers["server-timing"]
    assert "upstream;dur=0.000" in second.headers["server-timing"]


def test_identical_request_is_exact_hit_without_embedding(
    settings: Settings, tmp_path: Path
) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(
        _cache_settings(settings),
        tmp_path,
        team=make_team(semantic_cache=True),
        provider=provider,
    ) as client:
        _chat(client, "hello")
        embeds_after_first = provider.embed_calls
        second = _chat(client, "hello")
    assert second.headers["X-Gateway-Cache"] == "hit"
    assert second.headers["X-Gateway-Cache-Similarity"] == "1.0000"
    assert provider.embed_calls == embeds_after_first


def test_different_order_number_is_not_a_hit(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(
        _cache_settings(settings),
        tmp_path,
        team=make_team(semantic_cache=True),
        provider=provider,
    ) as client:
        _chat(client, "cancel order 123")
        second = _chat(client, "cancel order 124")
    assert second.headers["X-Gateway-Cache"] == "miss"
    assert provider.chat_calls == 2


def test_end_users_do_not_share_cache(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(
        _cache_settings(settings),
        tmp_path,
        team=make_team(semantic_cache=True),
        provider=provider,
    ) as client:
        _chat(client, "what is my order status?", user="alice")
        second = _chat(client, "what is my order status?", user="bob")
    assert second.headers["X-Gateway-Cache"] == "miss"


def test_no_cache_header_bypasses(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(
        _cache_settings(settings),
        tmp_path,
        team=make_team(semantic_cache=True),
        provider=provider,
    ) as client:
        _chat(client, "hello")
        second = _chat(client, "hello", headers={"Cache-Control": "no-cache"})
        bypassed = _metric_value(
            client,
            'llm_gateway_semantic_cache_bypass_total{reason="no_cache_header",team="t1"}',
        )
    assert second.headers["X-Gateway-Cache"] == "bypass"
    assert provider.chat_calls == 2
    assert bypassed == 1.0


def test_team_without_opt_in_bypasses(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(_cache_settings(settings), tmp_path, provider=provider) as client:
        _chat(client, "hello")
        second = _chat(client, "hello")
    assert second.headers["X-Gateway-Cache"] == "bypass"
    assert provider.chat_calls == 2


def test_embedding_outage_still_serves_and_caches_exact(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(
        provider_name="ollama",
        embed_responses=[ProviderUnavailableError("down")] * 4,
    )
    with gateway_client_factory(
        _cache_settings(settings),
        tmp_path,
        team=make_team(semantic_cache=True),
        provider=provider,
    ) as client:
        first = _chat(client, "hello")
        second = _chat(client, "hello")
    assert first.status_code == 200
    assert first.headers["X-Gateway-Cache"] == "miss"
    assert second.headers["X-Gateway-Cache"] == "hit"


def test_streaming_bypasses_cache(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(
        _cache_settings(settings),
        tmp_path,
        team=make_team(semantic_cache=True),
        provider=provider,
    ) as client:
        response = _chat(client, "hello", stream=True)
    assert response.headers["X-Gateway-Cache"] == "bypass"
    assert "server-timing" not in response.headers


# ── Server-Timing on a fresh completion ───────────────────────────────────


def test_fresh_completion_reports_upstream_and_overhead(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = _chat(client, "hello")
        overhead_count = _metric_value(client, 'llm_gateway_overhead_seconds_count{team="t1"}')
    assert "upstream;dur=1.000" in response.headers["server-timing"]
    assert overhead_count == 1.0


# ── model="auto" routing ──────────────────────────────────────────────────


def test_auto_with_static_policy_serves_high_quality(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    with gateway_client_factory(settings, tmp_path, provider=provider) as client:
        response = _chat(client, "hello", model=AUTO_MODEL)
    assert response.status_code == 200
    assert response.headers["X-Gateway-Model-Used"] == "qwen3.5:4b"
    assert response.headers["X-Gateway-Routing-Reason"] == "auto_default_high_quality"


def test_auto_filtered_by_team_whitelist(settings: Settings, tmp_path: Path) -> None:
    provider = ScriptedProvider(provider_name="ollama")
    team = make_team(allowed_models=("qwen3.5:0.8b",))
    with gateway_client_factory(settings, tmp_path, team=team, provider=provider) as client:
        response = _chat(client, "hello", model=AUTO_MODEL)
    assert response.status_code == 200
    assert response.headers["X-Gateway-Model-Used"] == "qwen3.5:0.8b"


def test_auto_with_no_allowed_model_is_403(settings: Settings, tmp_path: Path) -> None:
    team = make_team(allowed_models=("qwen3-embedding:0.6b",))
    with gateway_client_factory(
        settings, tmp_path, team=team, provider=ScriptedProvider()
    ) as client:
        response = _chat(client, "hello", model=AUTO_MODEL)
    assert response.status_code == 403


class FakeBudgetPolicy:
    """Routes everything to the budget model."""

    async def plan(self, request: RoutingRequest) -> RoutingPlan:
        return RoutingPlan(
            model_ids=("qwen3.5:0.8b",),
            policy=RoutingPolicyName.AUTOPILOT,
            reason=f"fake_{request.team_id}",
        )


def test_injected_policy_decides_the_model(settings: Settings, tmp_path: Path) -> None:
    with gateway_client_factory(
        settings,
        tmp_path,
        provider=ScriptedProvider(provider_name="ollama"),
        routing_policy=FakeBudgetPolicy(),
    ) as client:
        response = _chat(client, "hello", model=AUTO_MODEL)
    assert response.headers["X-Gateway-Routing-Reason"] == "fake_t1"
    assert response.headers["X-Gateway-Model-Used"] == "qwen3.5:0.8b"


def test_autopilot_setting_builds_autopilot_policy(settings: Settings, tmp_path: Path) -> None:
    from fakeredis import FakeAsyncRedis

    from llm_gateway.auth.resolver import TeamResolver
    from llm_gateway.ratelimit.token_bucket import TokenBucket
    from llm_gateway.storage.db import CostLedger

    app = create_app(
        settings.model_copy(update={"routing_policy": RoutingPolicyName.AUTOPILOT}),
        provider=ScriptedProvider(),  # type: ignore[arg-type]
        team_resolver=TeamResolver([make_team()]),
        token_bucket=TokenBucket(FakeAsyncRedis()),  # type: ignore[arg-type]
        cost_ledger=CostLedger(tmp_path / "test.db"),
    )
    with TestClient(app):
        assert isinstance(app.state.routing_policy, AutopilotPolicy)
