"""Unit tests for the llm-cost-autopilot HTTP adapter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import respx
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from llm_gateway.providers.base import ChatMessage
from llm_gateway.providers.registry import ModelTier, default_registry
from llm_gateway.routing.autopilot import (
    ROUTE_PATH,
    AutopilotClient,
    AutopilotPolicy,
    RouteMessage,
    RouteRequest,
    RouteResponse,
)
from llm_gateway.routing.policy import (
    AUTO_MODEL,
    RoutingPolicy,
    RoutingPolicyName,
    RoutingRequest,
    StaticTierPolicy,
)

BASE_URL = "http://autopilot-test:8000"
SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "contracts" / "autopilot-route-api.schema.json"
)
REGISTRY = default_registry(
    high_quality_model="qwen3.5:4b",
    budget_model="qwen3.5:0.8b",
    embedding_model="qwen3-embedding:0.6b",
)


def _policy(timeout_s: float = 0.5, api_token: str | None = None) -> AutopilotPolicy:
    return AutopilotPolicy(
        client=AutopilotClient(base_url=BASE_URL, timeout_s=timeout_s, api_token=api_token),
        registry=REGISTRY,
        fallback=StaticTierPolicy(REGISTRY),
        tier_map={
            "simple": ModelTier.BUDGET,
            "moderate": ModelTier.HIGH_QUALITY,
        },
        timeout_s=timeout_s,
    )


def _request(model: str = AUTO_MODEL, messages: list[ChatMessage] | None = None) -> RoutingRequest:
    return RoutingRequest(
        request_id="r1",
        team_id="team-a",
        requested_model=model,
        messages=messages
        or [
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="what is 2+2?"),
        ],
    )


def _decision(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "route_id": "rt-1",
        "tier": "simple",
        "confidence": 0.9,
        "reason": "stage_1_embedding",
    }
    body.update(overrides)
    return body


def test_policy_satisfies_protocol() -> None:
    assert isinstance(_policy(), RoutingPolicy)


@respx.mock
async def test_simple_tier_routes_to_budget_and_sends_tenant() -> None:
    route = respx.post(f"{BASE_URL}{ROUTE_PATH}").mock(
        return_value=httpx.Response(200, json=_decision())
    )
    plan = await _policy().plan(_request())
    assert plan.model_ids == ("qwen3.5:0.8b",)
    assert plan.policy is RoutingPolicyName.AUTOPILOT
    assert plan.reason == "autopilot_simple_stage_1_embedding"
    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "tenant_id": "team-a",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "what is 2+2?"},
        ],
    }
    assert "authorization" not in route.calls.last.request.headers


@respx.mock
async def test_bearer_token_sent_when_configured() -> None:
    route = respx.post(f"{BASE_URL}{ROUTE_PATH}").mock(
        return_value=httpx.Response(200, json=_decision())
    )
    await _policy(api_token="secret").plan(_request())
    assert route.calls.last.request.headers["authorization"] == "Bearer secret"


@respx.mock
async def test_tool_turns_are_not_sent() -> None:
    route = respx.post(f"{BASE_URL}{ROUTE_PATH}").mock(
        return_value=httpx.Response(200, json=_decision())
    )
    await _policy().plan(
        _request(
            messages=[
                ChatMessage(role="user", content="weather?"),
                ChatMessage(role="tool", content='{"temp": 20}'),
            ]
        )
    )
    sent = json.loads(route.calls.last.request.content)
    assert [m["role"] for m in sent["messages"]] == ["user"]


async def test_only_tool_turns_fall_back_without_calling() -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE_URL}{ROUTE_PATH}")
        plan = await _policy().plan(_request(messages=[ChatMessage(role="tool", content="x")]))
    assert not route.called
    assert plan.reason == "autopilot_no_routable_message"


@respx.mock
async def test_moderate_tier_gets_high_quality_chain_with_fallback() -> None:
    respx.post(f"{BASE_URL}{ROUTE_PATH}").mock(
        return_value=httpx.Response(
            200, json=_decision(tier="moderate", reason="stage_2_zero_shot")
        )
    )
    plan = await _policy().plan(_request())
    assert plan.model_ids == ("qwen3.5:4b", "qwen3.5:0.8b")
    assert plan.reason == "autopilot_moderate_stage_2_zero_shot"


@respx.mock
async def test_unknown_reason_is_bucketed_and_extra_fields_ignored() -> None:
    respx.post(f"{BASE_URL}{ROUTE_PATH}").mock(
        return_value=httpx.Response(
            200, json=_decision(reason="stage_3_new_thing", new_additive_field=True)
        )
    )
    plan = await _policy().plan(_request())
    assert plan.reason == "autopilot_simple_other"


async def test_named_model_bypasses_autopilot() -> None:
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{BASE_URL}{ROUTE_PATH}")
        plan = await _policy().plan(_request("qwen3.5:0.8b"))
    assert not route.called
    assert plan.policy is RoutingPolicyName.STATIC
    assert plan.model_ids == ("qwen3.5:0.8b",)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"detail": "boom"}),
        httpx.Response(401, json={"detail": "bad token"}),
        httpx.Response(200, json={"not": "a decision"}),
        httpx.Response(200, json=_decision(tier="extreme")),
        httpx.Response(200, json=_decision(confidence=1.5)),
    ],
)
@respx.mock
async def test_bad_responses_fall_back_to_static(response: httpx.Response) -> None:
    respx.post(f"{BASE_URL}{ROUTE_PATH}").mock(return_value=response)
    plan = await _policy().plan(_request())
    assert plan.policy is RoutingPolicyName.STATIC
    assert plan.reason == "autopilot_unavailable"
    assert plan.model_ids == ("qwen3.5:4b", "qwen3.5:0.8b")


@respx.mock
async def test_connection_error_falls_back_to_static() -> None:
    respx.post(f"{BASE_URL}{ROUTE_PATH}").mock(side_effect=httpx.ConnectError("refused"))
    plan = await _policy().plan(_request())
    assert plan.reason == "autopilot_unavailable"


@respx.mock
async def test_slow_autopilot_times_out_and_falls_back() -> None:
    async def slow(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.0)
        return httpx.Response(200, json=_decision())  # pragma: no cover

    respx.post(f"{BASE_URL}{ROUTE_PATH}").mock(side_effect=slow)
    plan = await _policy(timeout_s=0.05).plan(_request())
    assert plan.reason == "autopilot_unavailable"


async def test_client_closes_only_what_it_owns() -> None:
    shared = httpx.AsyncClient(base_url=BASE_URL)
    borrowed = AutopilotClient(base_url=BASE_URL, timeout_s=1.0, client=shared)
    await borrowed.aclose()
    assert not shared.is_closed
    await shared.aclose()

    owned = AutopilotClient(base_url=BASE_URL, timeout_s=1.0)
    await owned.aclose()


# ── Contract: our wire shapes against autopilot's published schema ─────────


def _schema() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def test_schema_is_route_api_v2() -> None:
    schema = _schema()
    assert schema["api_version"] == "2"
    assert schema["endpoint"] == f"POST {ROUTE_PATH}"


def test_request_fields_match_schema() -> None:
    request_schema = cast(dict[str, Any], _schema()["request"])
    assert set(RouteRequest.model_fields) == set(request_schema["properties"])
    assert set(request_schema["required"]) <= set(RouteRequest.model_fields)
    message_schema = cast(dict[str, Any], request_schema["$defs"]["ChatCompletionMessage"])
    assert set(RouteMessage.model_fields) == set(message_schema["properties"])
    role_enum = set(message_schema["properties"]["role"]["enum"])
    assert role_enum == {"system", "user", "assistant"}


def test_response_fields_match_schema() -> None:
    response_schema = cast(dict[str, Any], _schema()["response"])
    assert set(RouteResponse.model_fields) == set(response_schema["properties"])
    assert set(response_schema["required"]) == set(RouteResponse.model_fields)
    tiers = set(response_schema["$defs"]["ComplexityTier"]["enum"])
    assert tiers == {"simple", "moderate"}


def _validator(part: str) -> Draft202012Validator:
    part_schema = cast(dict[str, Any], _schema()[part])
    Draft202012Validator.check_schema(part_schema)
    return Draft202012Validator(part_schema)


@respx.mock
async def test_request_the_client_sends_is_valid_against_the_schema() -> None:
    """The body actually posted, not the model definition, is what autopilot validates."""
    route = respx.post(f"{BASE_URL}{ROUTE_PATH}").mock(
        return_value=httpx.Response(200, json=_decision())
    )
    messages = [
        ChatMessage(role="system", content="be brief"),
        ChatMessage(role="user", content="what is 2+2?"),
        ChatMessage(role="assistant", content="4"),
        ChatMessage(role="tool", content="dropped before sending"),
        ChatMessage(role="user", content="and 3+3?"),
    ]
    await _policy().plan(_request(messages=messages))
    sent = json.loads(route.calls.last.request.content)
    _validator("request").validate(sent)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("tier", ["simple", "moderate"])
def test_responses_the_gateway_accepts_are_valid_against_the_schema(tier: str) -> None:
    body = _decision(tier=tier)
    _validator("response").validate(body)  # pyright: ignore[reportUnknownMemberType]
    assert RouteResponse.model_validate(body).tier == tier


@pytest.mark.parametrize(
    "body",
    [
        _decision(tier="complex"),
        _decision(tier="trivial"),
        _decision(confidence=1.5),
        {"tier": "simple"},
    ],
    ids=["removed-v1-tier", "unknown-tier", "confidence-out-of-range", "missing-fields"],
)
def test_gateway_rejects_what_the_schema_rejects(body: dict[str, Any]) -> None:
    assert not _validator("response").is_valid(body)  # pyright: ignore[reportUnknownMemberType]
    with pytest.raises(ValidationError):
        RouteResponse.model_validate(body)
