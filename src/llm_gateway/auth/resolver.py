"""Team resolver: loads the YAML config and looks up teams by API key.

The resolver is read-only after construction. Live config updates replace the
whole resolver atomically on ``app.state.team_resolver`` rather than mutating
the in-memory map: that keeps lookups lock-free.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from secrets import compare_digest
from typing import Any, cast

import yaml
from pydantic import ValidationError

from llm_gateway.auth.models import TeamConfig


class TeamResolverError(Exception):
    """Raised on malformed teams YAML or duplicate API keys."""


class TeamResolver:
    """In-memory lookup of ``api_key → TeamConfig``.

    Duplicates on ``team_id`` or ``api_key`` are rejected at construction time:
    two teams silently sharing a key is a security smell we want to fail loud
    on.
    """

    def __init__(self, teams: Iterable[TeamConfig]) -> None:
        by_key: dict[str, TeamConfig] = {}
        by_id: dict[str, TeamConfig] = {}
        for team in teams:
            api_key = team.api_key.get_secret_value()
            if api_key in by_key:
                raise TeamResolverError(
                    f"duplicate api_key for teams {by_key[api_key].team_id!r} and {team.team_id!r}"
                )
            if team.team_id in by_id:
                raise TeamResolverError(f"duplicate team_id: {team.team_id!r}")
            by_key[api_key] = team
            by_id[team.team_id] = team
        self._by_key = by_key
        self._by_id = by_id

    def resolve(self, api_key: str) -> TeamConfig | None:
        """Return the team matching ``api_key``, or ``None`` if no match.

        Constant-time comparison avoids leaking key length / prefix via timing.
        """
        for key, team in self._by_key.items():
            if compare_digest(key, api_key):
                return team
        return None

    def get(self, team_id: str) -> TeamConfig | None:
        """The team with this id, for admin updates (never for authentication)."""
        return self._by_id.get(team_id)

    def all_teams(self) -> tuple[TeamConfig, ...]:
        """Every team, so an admin update can rebuild the resolver with one team replaced."""
        return tuple(self._by_id.values())

    @classmethod
    def from_yaml(cls, path: Path) -> TeamResolver:
        """Parse a teams YAML file. See ``config/teams.example.yaml``."""
        if not path.exists():
            raise TeamResolverError(f"teams YAML not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TeamResolverError(f"invalid YAML in {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise TeamResolverError(f"{path} must contain a top-level mapping")
        raw_map = cast(dict[str, Any], raw)
        teams_raw = raw_map.get("teams")
        if not isinstance(teams_raw, list):
            raise TeamResolverError(f"{path} must have a 'teams' list")

        parsed: list[TeamConfig] = []
        for index, entry in enumerate(cast(list[Any], teams_raw)):
            try:
                parsed.append(TeamConfig.model_validate(entry))
            except ValidationError as exc:
                raise TeamResolverError(f"team at index {index} invalid: {exc}") from exc
        return cls(parsed)
