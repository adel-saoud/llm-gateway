"""SQLite-backed persistence: cost ledger and circuit-breaker state.

Each completion (or embedding) is recorded with token counts, computed cost,
and the RAG operation tag. The budget enforcer queries the cost ledger for the
team's running daily total; the RAG cost-report endpoint groups the same rows
by operation type.

Circuit-breaker state for each ``(provider, model)`` pair is persisted in
``circuit_state`` so the breaker survives process restarts: a process that
comes back up while a downstream is OPEN still skips it until the open
duration elapses.

The ``operation_type`` column defaults to ``"unknown"``: the RAG tagger
populates it for traffic that flows through the tagger.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import aiosqlite

CURRENT_SCHEMA_VERSION: Final[int] = 2
"""Written to ``schema_meta`` on every start; bump it with any non-additive schema change."""

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
"""DDL run on every start: ``IF NOT EXISTS`` makes it serve fresh installs and the v1 to v2 upgrade.

The ``substr(ts, 1, 10)`` indexes match the per-day queries, which compare the
date prefix of the ISO timestamp.
"""

BUSY_TIMEOUT_S: Final[float] = 5.0
"""How long a connection waits for a lock before ``database is locked``.

Every request reads today's spend and writes a ledger row, each on its own
connection, so writers queue behind one another under load. The wait is stated
here rather than left to the ``sqlite3`` default so it is visible and tunable.
"""


def _connect(db_path: Path) -> aiosqlite.Connection:
    """Every connection the gateway opens to its SQLite file, with the same lock wait."""
    return aiosqlite.connect(db_path, timeout=BUSY_TIMEOUT_S)


class StorageError(Exception):
    """A row holds a value the schema does not allow (for example a NULL required timestamp)."""


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

    Connections are opened per-call: SQLite is fine with this pattern under
    ``aiosqlite``'s single-writer semantics, and it sidesteps the
    "transaction held open across await" footgun. The file is in WAL mode, so
    the budget reads on every request do not block the ledger writes.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        """SQLite file of the ledger; the circuit-state repository shares it."""
        return self._db_path

    async def initialize(self) -> None:
        """Create the parent directory and the tables, then stamp the schema version.

        WAL mode is persistent in the database file, so setting it here once
        covers every later connection, from this process or another.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with _connect(self._db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(_SCHEMA)
            await db.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(CURRENT_SCHEMA_VERSION),),
            )
            await db.commit()

    @contextlib.asynccontextmanager
    async def keep_wal_open(self) -> AsyncGenerator[None]:
        """Hold one connection on the file for as long as the gateway runs.

        When the last connection to a WAL database closes, SQLite checkpoints
        the WAL into the main file and deletes it. With a connection per call,
        a request that is alone in the process pays that on every write: 1.7 ms
        per budget read plus ledger write against 0.6 ms with the WAL held open.
        The held connection runs no transaction, so automatic checkpoints still
        happen.
        """
        async with _connect(self._db_path) as keeper:
            # A connection attaches to the WAL on its first read, not when it opens.
            await keeper.execute_fetchall("SELECT 1 FROM schema_meta LIMIT 1")
            yield

    async def schema_version(self) -> int:
        """Version stamped by :meth:`initialize`; 0 before the first start."""
        async with _connect(self._db_path) as db:
            cursor = await db.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def record(self, entry: CostLedgerEntry) -> None:
        """Append one cost row."""
        async with _connect(self._db_path) as db:
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
        async with _connect(self._db_path) as db:
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
        async with _connect(self._db_path) as db:
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
        """Persisted state of one endpoint, or None if it never transitioned."""
        async with _connect(self._db_path) as db:
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
        """Insert or replace the state of one endpoint."""
        opened_at_iso = record.opened_at.isoformat() if record.opened_at is not None else None
        async with _connect(self._db_path) as db:
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


def _parse_iso(value: object) -> datetime | None:
    """A nullable ISO-8601 column as a datetime."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise StorageError(f"expected ISO datetime string, got {type(value).__name__}")
    return datetime.fromisoformat(value)


def _parse_iso_required(value: object) -> datetime:
    """A NOT NULL ISO-8601 column as a datetime; a NULL means the row is corrupt."""
    parsed = _parse_iso(value)
    if parsed is None:
        raise StorageError("required ISO datetime is NULL")
    return parsed
