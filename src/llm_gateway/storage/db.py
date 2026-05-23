"""SQLite-backed persistence — cost ledger and circuit-breaker state.

Each completion (or embedding) is recorded with token counts, computed cost,
and the RAG operation tag. The budget enforcer queries the cost ledger for the
team's running daily total; the RAG cost-report endpoint groups the same rows
by operation type.

Circuit-breaker state for each ``(provider, model)`` pair is persisted in
``circuit_state`` so the breaker survives process restarts — a process that
comes back up while a downstream is OPEN still skips it until the open
duration elapses.

The ``operation_type`` column defaults to ``"unknown"`` — the RAG tagger
populates it for traffic that flows through the tagger.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import aiosqlite

CURRENT_SCHEMA_VERSION: Final[int] = 2

# ``CREATE TABLE IF NOT EXISTS`` makes the same script work for fresh
# installs and idempotent upgrades from v1 → v2 (we only added a new table).
_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cost_ledger (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    team_id         TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    operation_type  TEXT NOT NULL DEFAULT 'unknown',
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    cost_usd        REAL NOT NULL,
    request_id      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cost_ledger_team_day
    ON cost_ledger (team_id, substr(ts, 1, 10));

CREATE INDEX IF NOT EXISTS idx_cost_ledger_op_day
    ON cost_ledger (team_id, operation_type, substr(ts, 1, 10));

CREATE TABLE IF NOT EXISTS circuit_state (
    provider_name TEXT NOT NULL,
    model_id      TEXT NOT NULL,
    state         TEXT NOT NULL,
    opened_at     TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (provider_name, model_id)
);
"""


class StorageError(Exception):
    """Raised on storage-layer failures."""


# ── Cost ledger ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CostLedgerEntry:
    """One row of the ``cost_ledger`` table."""

    ts: datetime
    team_id: str
    model_id: str
    operation_type: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_id: str


class CostLedger:
    """Append-only ledger and aggregation queries.

    Connections are opened per-call — SQLite is fine with this pattern under
    ``aiosqlite``'s single-writer semantics, and it sidesteps the
    "transaction held open across await" footgun.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        """File path the ledger writes to — sibling stores (e.g. the
        circuit-state repository) share the same file."""
        return self._db_path

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(CURRENT_SCHEMA_VERSION),),
            )
            await db.commit()

    async def schema_version(self) -> int:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def record(self, entry: CostLedgerEntry) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO cost_ledger
                    (ts, team_id, model_id, operation_type,
                     input_tokens, output_tokens, cost_usd, request_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.ts.isoformat(),
                    entry.team_id,
                    entry.model_id,
                    entry.operation_type,
                    entry.input_tokens,
                    entry.output_tokens,
                    entry.cost_usd,
                    entry.request_id,
                ),
            )
            await db.commit()

    async def total_for_day(self, team_id: str, *, day: datetime) -> float:
        """Sum of ``cost_usd`` for ``team_id`` on the calendar date of ``day`` (UTC)."""
        date_str = day.astimezone(UTC).date().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0.0)
                FROM cost_ledger
                WHERE team_id = ? AND substr(ts, 1, 10) = ?
                """,
                (team_id, date_str),
            )
            row = await cursor.fetchone()
        return float(row[0]) if row is not None else 0.0

    async def totals_by_operation_for_day(self, team_id: str, *, day: datetime) -> dict[str, float]:
        """Per-operation-type spend for ``team_id`` on a given UTC date."""
        date_str = day.astimezone(UTC).date().isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT operation_type, COALESCE(SUM(cost_usd), 0.0)
                FROM cost_ledger
                WHERE team_id = ? AND substr(ts, 1, 10) = ?
                GROUP BY operation_type
                """,
                (team_id, date_str),
            )
            rows = await cursor.fetchall()
        return {str(row[0]): float(row[1]) for row in rows}


# ── Circuit-breaker state ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CircuitStateRecord:
    """One row of the ``circuit_state`` table.

    ``state`` is a string at this layer so the storage module stays
    independent of the higher-level enum; the
    :class:`~llm_gateway.circuit_breaker.state_machine.CircuitBreaker` does
    the enum conversion at its boundary.
    """

    provider_name: str
    model_id: str
    state: str
    opened_at: datetime | None
    updated_at: datetime


class CircuitStateRepository:
    """Read/write circuit-breaker state for ``(provider, model)`` pairs."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    async def get(self, provider_name: str, model_id: str) -> CircuitStateRecord | None:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT state, opened_at, updated_at
                FROM circuit_state
                WHERE provider_name = ? AND model_id = ?
                """,
                (provider_name, model_id),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return CircuitStateRecord(
            provider_name=provider_name,
            model_id=model_id,
            state=str(row[0]),
            opened_at=_parse_iso(row[1]),
            updated_at=_parse_iso_required(row[2]),
        )

    async def upsert(self, record: CircuitStateRecord) -> None:
        opened_at_iso = record.opened_at.isoformat() if record.opened_at is not None else None
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO circuit_state
                    (provider_name, model_id, state, opened_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (provider_name, model_id) DO UPDATE SET
                    state = excluded.state,
                    opened_at = excluded.opened_at,
                    updated_at = excluded.updated_at
                """,
                (
                    record.provider_name,
                    record.model_id,
                    record.state,
                    opened_at_iso,
                    record.updated_at.isoformat(),
                ),
            )
            await db.commit()

    async def list_all(self) -> tuple[CircuitStateRecord, ...]:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT provider_name, model_id, state, opened_at, updated_at
                FROM circuit_state
                ORDER BY provider_name, model_id
                """
            )
            rows = await cursor.fetchall()
        return tuple(
            CircuitStateRecord(
                provider_name=str(r[0]),
                model_id=str(r[1]),
                state=str(r[2]),
                opened_at=_parse_iso(r[3]),
                updated_at=_parse_iso_required(r[4]),
            )
            for r in rows
        )


def _parse_iso(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise StorageError(f"expected ISO datetime string, got {type(value).__name__}")
    return datetime.fromisoformat(value)


def _parse_iso_required(value: object) -> datetime:
    parsed = _parse_iso(value)
    if parsed is None:
        raise StorageError("required ISO datetime is NULL")
    return parsed
