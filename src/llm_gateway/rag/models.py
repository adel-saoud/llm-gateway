"""RAG operation types and the structured cost-event/report shapes.

These three types are the canonical "what kind of LLM work was this?" vocabulary
used across the gateway:

* :class:`RagOperationType`: the enum the tagger emits.
* :class:`RagCostEvent`: one strongly-typed cost row in memory; what the
  budget tracker returns and what downstream metrics/log emitters consume.
* :class:`RagCostReport`: the aggregated per-day, per-operation view exposed
  by ``GET /v1/rag/cost-report``.

The storage layer persists ``operation_type`` as a plain string, so adding an
operation type needs no migration and the report reads the column back as is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class RagOperationType(StrEnum):
    """The RAG stage a request belongs to.

    The classifier never returns ``UNKNOWN``; it is the ``cost_ledger`` column
    default, so it only shows up for rows written without a tag.
    """

    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RagCostEvent:
    """One tagged completion's cost event: emitted after the LLM call succeeds."""

    team_id: str
    model_id: str
    operation_type: RagOperationType
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_id: str
    ts: datetime


@dataclass(frozen=True, slots=True)
class RagCostReport:
    """Per-operation breakdown of one team's spend on one UTC calendar date.

    The ``by_operation`` mapping always contains the four named operations
    (``retrieval``, ``reranking``, ``generation``, ``evaluation``), with
    ``0.0`` for any that had no traffic. ``unknown`` is included only when
    non-zero so opaque traffic stays visible in the total.
    """

    team_id: str
    date: str
    total_cost_usd: float
    by_operation: Mapping[str, float]
