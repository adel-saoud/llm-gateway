"""Runtime configuration: env-driven, validated by pydantic-settings.

Resolution order: explicit kwargs, then ``GATEWAY_*`` environment variables,
then the ``.env`` file, then the defaults below. The attribute docstrings are
the reference for each setting (``use_attribute_docstrings`` also turns them
into field descriptions). Host and port are not settings: uvicorn owns them
(``--host`` / ``--port``).

Unknown ``GATEWAY_*`` names, in the environment or in ``.env``, stop startup
with :class:`UnknownSettingError`. pydantic-settings would otherwise ignore
them, so a misspelled or renamed variable would silently keep its default.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final

from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from llm_gateway.providers.registry import ModelTier
from llm_gateway.routing.policy import RoutingPolicyName


class LogLevel(StrEnum):
    """Levels accepted by ``GATEWAY_LOG_LEVEL``; the values are stdlib logging level names."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


ENV_PREFIX: Final = "GATEWAY_"

# difflib ratio below which a suggestion would mislead more than help
# (PORT is not "close" to DB_PATH; CB_MIN_REQUESTS is to CIRCUIT_BREAKER_MIN_REQUESTS).
_CLOSE_NAME_CUTOFF: Final = 0.6

# Settings removed without a successor. A close-name hint would point at a
# different setting (TIER_COMPLEX is "close" to TIER_SIMPLE), so these say why
# the name is gone instead.
REMOVED_SETTINGS: Final[Mapping[str, str]] = {
    "GATEWAY_AUTOPILOT_TIER_COMPLEX": (
        "removed: autopilot route API v2 has no complex tier; delete the variable"
    ),
}


class UnknownSettingError(ValueError):
    """A ``GATEWAY_*`` variable that matches no setting, with the closest valid name."""


def reject_unknown_env_names(names: Iterable[str], known: Iterable[str]) -> None:
    """Raise :class:`UnknownSettingError` if a ``GATEWAY_*`` name is not a known setting.

    Names are compared upper-cased because settings are case-insensitive.
    Variables without the prefix (``OTEL_*``, ``PATH``) are not the gateway's
    business and are skipped.
    """
    valid = sorted({name.upper() for name in known})
    unknown = sorted(
        {
            upper
            for name in names
            if (upper := name.upper()).startswith(ENV_PREFIX) and upper not in valid
        }
    )
    if not unknown:
        return
    # Compare the part after the prefix: the shared prefix alone would make
    # every pair look similar.
    suffixes = [name.removeprefix(ENV_PREFIX) for name in valid]
    hints: list[str] = []
    for name in unknown:
        close = difflib.get_close_matches(
            name.removeprefix(ENV_PREFIX), suffixes, n=1, cutoff=_CLOSE_NAME_CUTOFF
        )
        if name in REMOVED_SETTINGS:
            hint = REMOVED_SETTINGS[name]
        elif close:
            hint = f"did you mean {ENV_PREFIX}{close[0]}?"
        else:
            hint = "no setting with a close name"
        hints.append(f"{name} ({hint})")
    raise UnknownSettingError(
        f"unknown {ENV_PREFIX}* variable(s): {', '.join(hints)}. "
        "Valid names are listed in .env.example."
    )


class Settings(BaseSettings):
    """Process-wide settings, built once by :func:`llm_gateway.main.create_app`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix=ENV_PREFIX,
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
        use_attribute_docstrings=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keep the default sources; first reject unknown names in the environment and ``.env``.

        This hook sees the same ``.env`` file pydantic-settings loads, including
        an ``_env_file`` passed at construction.
        """
        seen = [
            name
            for source in (env_settings, dotenv_settings)
            if isinstance(source, EnvSettingsSource)
            for name in source.env_vars
        ]
        reject_unknown_env_names(
            seen, (f"{ENV_PREFIX}{field}" for field in settings_cls.model_fields)
        )
        return init_settings, env_settings, dotenv_settings, file_secret_settings

    log_level: LogLevel = Field(default=LogLevel.INFO)
    """Minimum level for the JSON logs written to stdout."""

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(default="http://localhost:11434")
    """Ollama server root; inside docker-compose this is ``http://ollama:11434``."""
    ollama_timeout_s: float = Field(default=120.0, ge=1.0)
    """Read timeout per Ollama call. CPU-only generation on a 4B model can take a minute."""

    model_high_quality: str = Field(default="qwen3.5:4b")
    """Model id of the high-quality tier; the budget model is its fallback."""
    model_budget: str = Field(default="qwen3.5:0.8b")
    """Model id of the budget tier."""
    model_embeddings: str = Field(default="qwen3-embedding:0.6b")
    """Model id used for ``/v1/embeddings`` and for semantic cache lookups."""

    # ── State ───────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")
    """Redis holding the per-team token buckets."""
    db_path: Path = Field(default=Path("data/gateway.db"))
    """SQLite file shared by the cost ledger and the circuit-breaker state."""
    teams_yaml: Path = Field(default=Path("config/teams.yaml"))
    """Teams file read at startup (see ``config/teams.example.yaml``)."""

    # ── Circuit breaker ─────────────────────────────────────────────────────
    circuit_breaker_failure_rate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Failure rate over the rolling window that opens a ``(provider, model)`` circuit."""
    circuit_breaker_min_requests: int = Field(default=10, ge=1)
    """Samples needed before the failure rate can open a circuit: smaller windows are noise."""
    circuit_breaker_open_duration_s: float = Field(default=30.0, ge=1.0)
    """How long an open circuit is skipped before one probe request is let through."""

    # ── Response cache ──────────────────────────────────────────────────────
    cache_enabled: bool = Field(default=False)
    """Global switch; each team must also opt in with ``semantic_cache: true``."""
    cache_semantic_enabled: bool = Field(default=False)
    """Embedding-similarity matching on top of exact matching.

    Off by default: on the 48 labelled pairs of ``scripts/calibrate_cache.py``,
    the threshold with no wrong hit catches only about a third of paraphrases
    (``docs/cache.md``). Enable only after calibrating on your own traffic.
    """
    cache_similarity_threshold: float = Field(default=0.92, gt=0.0, le=1.0)
    """Cosine similarity a semantic match must reach.

    0.92 is the lowest threshold with no wrong hit on the calibration pairs with
    the default model and prefix; recalibrate when either changes.
    """
    cache_ttl_s: float = Field(default=3600.0, gt=0.0)
    """Lifetime of one cache entry."""
    cache_max_entries_per_partition: int = Field(default=1000, ge=1)
    """LRU cap per partition; also bounds the size of one similarity search."""
    cache_max_partitions: int = Field(default=10_000, ge=1)
    """LRU cap on partitions (team, user, model, system prompt) held in memory."""
    cache_max_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    """Requests above this temperature ask for varied answers, so they bypass the cache."""
    cache_embedding_prefix: str = Field(
        default=(
            "Instruct: Given a question, retrieve questions that are semantically "
            "equivalent to the given question\nQuery:"
        )
    )
    """Text prepended before embedding a prompt for a cache lookup.

    Qwen3-Embedding takes a task instruction in ``Instruct: ...\nQuery:`` form. Its
    model card puts it on queries and not on documents; a cache lookup compares a
    query with earlier queries, so the same instruction goes on both sides. That
    symmetric use is not covered by the card: it was chosen by calibration, where
    this duplicate-question task separated near-misses better than no prefix or a
    generic similarity task. Set to ``""`` for models without instructions.
    """

    # ── Routing ─────────────────────────────────────────────────────────────
    routing_policy: RoutingPolicyName = Field(default=RoutingPolicyName.STATIC)
    """Policy for ``model="auto"`` requests; a named model is never overridden."""
    autopilot_base_url: str = Field(default="http://localhost:8100")
    """Host port of the sidecar in ``docker-compose.autopilot.yml`` (container port 8000)."""
    autopilot_api_token: SecretStr | None = Field(default=None)
    """Sent as a Bearer token when set; must match ``AUTOPILOT_API_TOKEN`` on the sidecar."""
    autopilot_timeout_s: float = Field(default=0.25, gt=0.0)
    """Hard ceiling on the routing call; past it the static policy is used."""
    autopilot_tier_simple: ModelTier = Field(default=ModelTier.BUDGET)
    """Gateway tier serving prompts autopilot rates ``simple``."""
    autopilot_tier_moderate: ModelTier = Field(default=ModelTier.HIGH_QUALITY)
    """Gateway tier serving prompts autopilot rates ``moderate``."""

    # ── Tracing ─────────────────────────────────────────────────────────────
    tracing_enabled: bool = Field(default=False)
    """Export one OTel GenAI span per LLM call over OTLP/HTTP.

    The endpoint and service name come from the standard ``OTEL_*``
    environment variables, not from this class.
    """
    tracing_capture_message_content: bool = Field(default=False)
    """Add ``gen_ai.input.messages`` / ``gen_ai.output.messages`` to those spans.

    Prompts and answers may hold personal data, so this is opt-in, as the
    GenAI semantic conventions recommend.
    """
