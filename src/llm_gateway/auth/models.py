"""Team-scoped configuration models.

A team is the unit of authentication, rate limiting, and budget enforcement.
Each team's config lives in a YAML file (see ``config/teams.example.yaml``) and
is loaded once at startup by :class:`~llm_gateway.auth.resolver.TeamResolver`.

The models are immutable (``frozen=True``) and reject unknown keys
(``extra="forbid"``) — a typo in the YAML fails loudly at parse time rather
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
        description="Max tokens the bucket can hold — caps short-window bursts",
    )

    @property
    def refill_per_second(self) -> float:
        return self.requests_per_minute / 60.0


class BudgetConfig(BaseModel):
    """Per-team daily spend cap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    daily_usd: float = Field(gt=0.0, description="Hard daily cap — gateway blocks at 100%")


class TeamConfig(BaseModel):
    """One row of the teams YAML file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    team_id: str = Field(min_length=1, description="Stable identifier — logged with every event")
    api_key: SecretStr = Field(description="Bearer token; matched against Authorization header")
    allowed_models: tuple[str, ...] = Field(
        default=(),
        description="Whitelist of model ids; empty means no model restriction",
    )
    rate_limits: RateLimitConfig
    budget: BudgetConfig
    is_admin: bool = Field(
        default=False,
        description="When True, the team can call /v1/admin/* endpoints",
    )

    def is_model_allowed(self, model_id: str) -> bool:
        """Empty whitelist = unrestricted. Otherwise must match exactly."""
        return not self.allowed_models or model_id in self.allowed_models
