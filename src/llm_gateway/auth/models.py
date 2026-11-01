"""Team-scoped configuration models.

A team is the unit of authentication, rate limiting, and budget enforcement.
Each team's config lives in a YAML file (see ``config/teams.example.yaml``) and
is loaded once at startup by :class:`~llm_gateway.auth.resolver.TeamResolver`.

The models are immutable (``frozen=True``) and reject unknown keys
(``extra="forbid"``): a typo in the YAML fails loudly at parse time rather
than silently dropping configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class RateLimitConfig(BaseModel):
    """Token-bucket parameters for one team."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requests_per_minute: int = Field(ge=1, description="Steady-state refill rate")
    burst_capacity: int = Field(
        ge=1,
        description="Max tokens the bucket can hold: caps short-window bursts",
    )

    @property
    def refill_per_second(self) -> float:
        """Refill rate in the unit the Redis token bucket takes."""
        return self.requests_per_minute / 60.0


class BudgetConfig(BaseModel):
    """Per-team daily spend cap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    daily_usd: float = Field(gt=0.0, description="Hard daily cap: gateway blocks at 100%")


class TeamConfig(BaseModel):
    """One row of the teams YAML file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    team_id: str = Field(min_length=1, description="Stable identifier: logged with every event")
    api_key: SecretStr = Field(description="Bearer token; matched against Authorization header")
    allowed_models: tuple[str, ...] = Field(
        default=(),
        description=(
            "Whitelist of tiers (high_quality, budget, embeddings) and model ids; "
            "empty means no model restriction"
        ),
    )
    rate_limits: RateLimitConfig
    budget: BudgetConfig
    is_admin: bool = Field(
        default=False,
        description="When True, the team can call /v1/admin/* endpoints",
    )
    semantic_cache: bool = Field(
        default=False,
        description="Opt in to the semantic cache (also needs GATEWAY_CACHE_ENABLED=true)",
    )

    def is_model_allowed(self, model_id: str, tier: str | None = None) -> bool:
        """Whether the team may call ``model_id``, which serves ``tier`` (None if unknown).

        An entry may name a tier, so a team config survives a model change made
        in the settings; it may also name a model id, for a team pinned to one
        model. An empty whitelist means unrestricted.
        """
        if not self.allowed_models:
            return True
        return model_id in self.allowed_models or (tier is not None and tier in self.allowed_models)
