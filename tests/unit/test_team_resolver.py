from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from llm_gateway.auth.models import BudgetConfig, RateLimitConfig, TeamConfig
from llm_gateway.auth.resolver import TeamResolver, TeamResolverError


def _team(team_id: str, api_key: str) -> TeamConfig:
    return TeamConfig(
        team_id=team_id,
        api_key=SecretStr(api_key),
        allowed_models=("m",),
        rate_limits=RateLimitConfig(requests_per_minute=60, burst_capacity=10),
        budget=BudgetConfig(daily_usd=1.0),
    )


def test_resolve_returns_matching_team() -> None:
    resolver = TeamResolver([_team("a", "sk-a"), _team("b", "sk-b")])
    assert resolver.resolve("sk-a") is not None
    assert resolver.resolve("sk-a").team_id == "a"  # type: ignore[union-attr]
    assert resolver.resolve("sk-b").team_id == "b"  # type: ignore[union-attr]


def test_resolve_returns_none_for_unknown_key() -> None:
    resolver = TeamResolver([_team("a", "sk-a")])
    assert resolver.resolve("nope") is None


def test_duplicate_api_keys_rejected() -> None:
    with pytest.raises(TeamResolverError, match="duplicate api_key"):
        TeamResolver([_team("a", "sk-x"), _team("b", "sk-x")])


def test_duplicate_team_ids_rejected() -> None:
    with pytest.raises(TeamResolverError, match="duplicate team_id"):
        TeamResolver([_team("a", "sk-1"), _team("a", "sk-2")])


def test_from_yaml_loads_valid_file(tmp_path: Path) -> None:
    yaml_path = tmp_path / "teams.yaml"
    yaml_path.write_text(
        """
teams:
  - team_id: search-team
    api_key: sk-search
    allowed_models: [llama3.2:3b]
    rate_limits:
      requests_per_minute: 60
      burst_capacity: 100
    budget:
      daily_usd: 5.0
""".strip(),
        encoding="utf-8",
    )
    resolver = TeamResolver.from_yaml(yaml_path)
    assert resolver.resolve("sk-search") is not None
    assert resolver.resolve("sk-search").team_id == "search-team"  # type: ignore[union-attr]


def test_from_yaml_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TeamResolverError, match="not found"):
        TeamResolver.from_yaml(tmp_path / "missing.yaml")


def test_from_yaml_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("teams: [unclosed", encoding="utf-8")
    with pytest.raises(TeamResolverError, match="invalid YAML"):
        TeamResolver.from_yaml(path)


def test_from_yaml_missing_teams_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "weird.yaml"
    path.write_text("something_else: []", encoding="utf-8")
    with pytest.raises(TeamResolverError, match="'teams' list"):
        TeamResolver.from_yaml(path)


def test_from_yaml_invalid_team_entry_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text(
        """
teams:
  - team_id: x
    api_key: sk
    rate_limits: {requests_per_minute: 0, burst_capacity: 1}
    budget: {daily_usd: 1.0}
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(TeamResolverError, match="invalid"):
        TeamResolver.from_yaml(path)
