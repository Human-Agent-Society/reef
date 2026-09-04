"""SQLite-backed record storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from weakref import WeakValueDictionary

from reef.core.artifact_ref import decode_artifact_ref, encode_artifact_ref
from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RequestType


class RecordConflict(ReefError):
    """Raised when append content conflicts with an existing agent_record_id."""


@dataclass(frozen=True)
class AppendResult:
    item: AgentRecord
    inserted: bool


class RecordStore:
    """Store scenario records in append order.

    The SQLite schema uses an ``agent_record`` table with an
    ``agent_record_id`` column, mirroring the wire id
    (``x-reef-agent-record-id``).

    Passing a filesystem path makes the store durable. The default in-memory
    database keeps standalone/test construction lightweight; production callers
    should always pass a path.
    """

    _SQLITE_ID_CHUNK_SIZE = 900

    def __init__(self, database: str | Path | None = None) -> None:
        self._database = ":memory:" if database is None else str(database)
        if self._database != ":memory:":
            Path(self._database).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._live_records: WeakValueDictionary[str, AgentRecord] = WeakValueDictionary()
        self._connection = sqlite3.connect(
            self._database,
            timeout=30,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    @property
    def database(self) -> str:
        return self._database

    def _initialize(self) -> None:
        with self._lock, self._connection:
            if self._database != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_record (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_record_id TEXT NOT NULL UNIQUE,
                    scenario TEXT NOT NULL,
                    request_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    references_json TEXT NOT NULL,
                    artifact_json TEXT
                )
            """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS consumed_agent_record (
                    agent_record_id TEXT PRIMARY KEY,
                    content_sha256 TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS agent_record_scenario_sequence ON agent_record (scenario, sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS agent_record_scenario_type_sequence "
                "ON agent_record (scenario, request_type, sequence)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS compaction_receipts (
                    scenario TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    compacted_ids_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    recorded_at REAL NOT NULL,
                    PRIMARY KEY (scenario, receipt_id, compacted_ids_json)
                )
                """
            )

    #: Column order of the tuples produced by :meth:`_encode` and
    #: :meth:`_row_content`; :meth:`_content` uses it to name the fields that
    #: participate in retry-content comparison.
    _ENCODED_FIELDS = (
        "agent_record_id",
        "scenario",
        "request_type",
        "created_at",
        "payload_json",
        "references_json",
        "artifact_json",
    )

    @staticmethod
    def _json(value: object) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("a record must contain JSON-serializable values") from exc

    @classmethod
    def _encode(cls, item: AgentRecord) -> tuple[str, str, str, float, str, str, str | None]:
        artifact = item.artifact_ref
        artifact_json = None
        if artifact is not None:
            artifact_json = cls._json(encode_artifact_ref(artifact))
        return (
            item.agent_record_id,
            item.scenario,
            item.request_type.value,
            item.created_at,
            cls._json(dict(item.payload)),
            cls._json(item.references),
            artifact_json,
        )

    @staticmethod
    def _decode(row: sqlite3.Row) -> AgentRecord:
        raw_artifact = json.loads(row["artifact_json"]) if row["artifact_json"] is not None else None
        artifact = None
        if raw_artifact is not None:
            artifact = decode_artifact_ref(raw_artifact)
        return AgentRecord(
            agent_record_id=row["agent_record_id"],
            scenario=row["scenario"],
            request_type=RequestType(row["request_type"]),
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            references=tuple(json.loads(row["references_json"])),
            artifact_ref=artifact,
        )

    @staticmethod
    def _row_content(row: sqlite3.Row) -> tuple[str, str, str, float, str, str, str | None]:
        return (
            row["agent_record_id"],
            row["scenario"],
            row["request_type"],
            row["created_at"],
            row["payload_json"],
            row["references_json"],
            row["artifact_json"],
        )

    @classmethod
    def _content(cls, encoded: tuple[str, str, str, float, str, str, str | None]) -> dict[str, object]:
        """The encoded fields that define row content, excluding ``created_at``.

        A client retrying with its own agent_record_id regenerates the
        timestamp, so a timestamp difference alone must dedup, not conflict.
        """
        return {name: value for name, value in zip(cls._ENCODED_FIELDS, encoded, strict=True) if name != "created_at"}

    @classmethod
    def _content_sha256(cls, encoded: tuple[str, str, str, float, str, str, str | None]) -> str:
        canonical = cls._json(cls._content(encoded)).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def append(self, item: AgentRecord) -> AgentRecord:
        return self.append_result(item).item

    def append_result(self, item: AgentRecord) -> AppendResult:
        encoded = self._encode(item)
        with self._lock, self._connection:
            consumed = self._connection.execute(
                "SELECT content_sha256 FROM consumed_agent_record WHERE agent_record_id = ?",
                (item.agent_record_id,),
            ).fetchone()
            if consumed is not None:
                if consumed["content_sha256"] != self._content_sha256(encoded):
                    raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
                return AppendResult(item, False)
            if item.request_type is RequestType.REPORT and item.references:
                placeholders = ",".join("?" for _ in item.references)
                consumed = self._connection.execute(
                    f"SELECT 1 FROM consumed_agent_record WHERE agent_record_id IN ({placeholders}) LIMIT 1",
                    item.references,
                ).fetchone()
                if consumed is not None:
                    # A report is discarded once its references are gone, but a row
                    # already stored under this id stays canonical: check the discard
                    # against that row so a divergent retry cannot register its own
                    # content as the receipt and reject the honest retry that follows.
                    existing = self._connection.execute(
                        "SELECT * FROM agent_record WHERE agent_record_id = ?",
                        (item.agent_record_id,),
                    ).fetchone()
                    if existing is not None and self._content(self._row_content(existing)) != self._content(encoded):
                        raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
                    self._connection.execute(
                        "INSERT INTO consumed_agent_record (agent_record_id, content_sha256) VALUES (?, ?)",
                        (item.agent_record_id, self._content_sha256(encoded)),
                    )
                    return AppendResult(item, False)
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO agent_record (
                    agent_record_id, scenario, request_type, created_at,
                    payload_json, references_json, artifact_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                encoded,
            )
            if cursor.rowcount == 1:
                self._live_records[item.agent_record_id] = item
                return AppendResult(item, True)
            existing = self._connection.execute(
                "SELECT * FROM agent_record WHERE agent_record_id = ?",
                (item.agent_record_id,),
            ).fetchone()
            existing_content = None if existing is None else self._row_content(existing)
            if existing_content is None or self._content(existing_content) != self._content(encoded):
                raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
            live = self._live_records.get(item.agent_record_id)
            if live is not None:
                return AppendResult(live, False)
            stored = self._decode(existing)
            self._live_records[item.agent_record_id] = stored
            return AppendResult(stored, False)

    def get(self, scenario: str, agent_record_id: str) -> AgentRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agent_record WHERE scenario = ? AND agent_record_id = ?",
                (scenario, agent_record_id),
            ).fetchone()
        return None if row is None else self._decode(row)

    def replay(
        self,
        scenario: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[AgentRecord, ...]:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return ()
        sql = "SELECT * FROM agent_record WHERE scenario = ? ORDER BY sequence LIMIT ? OFFSET ?"
        size = -1 if limit is None else limit
        with self._lock:
            rows = self._connection.execute(sql, (scenario, size, offset)).fetchall()
        return tuple(self._decode(row) for row in rows)

    def replay_page(
        self,
        scenario: str,
        *,
        after_sequence: int = 0,
        limit: int = 256,
    ) -> tuple[tuple[int, AgentRecord], ...]:
        """Read a bounded keyset page for internal streaming consumers."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM agent_record
                WHERE scenario = ? AND sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (scenario, after_sequence, limit),
            ).fetchall()
        return tuple((int(row["sequence"]), self._decode(row)) for row in rows)

    def count(self, scenario: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM agent_record WHERE scenario = ?",
                (scenario,),
            ).fetchone()
        if row is None:
            raise RuntimeError("record count query returned no row")
        return int(row["count"])

    def compact(
        self,
        scenario: str,
        agent_record_ids: frozenset[str],
        *,
        receipt_id: str | None = None,
        receipt_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Delete payloads while retaining retry-safe idempotency receipts.

        Compaction is irreversible: deleted rows no longer participate in
        replay, lookup, or reference availability. Durable receipts preserve
        append deduplication and reject reports that reference consumed data.
        Callers must ensure the payloads are no longer needed.
        """
        if (receipt_id is None) != (receipt_metadata is None):
            raise ValueError("compaction receipt_id and receipt_metadata must be provided together")
        if receipt_id is not None and not receipt_id:
            raise ValueError("compaction receipt_id must be non-empty")
        if not agent_record_ids and receipt_id is None:
            return
        compacted_ids_json = self._json(sorted(agent_record_ids))
        metadata_json = self._json(dict(receipt_metadata or {}))
        with self._lock, self._connection:
            if receipt_id is not None:
                existing = self._connection.execute(
                    """
                    SELECT compacted_ids_json, metadata_json
                    FROM compaction_receipts
                    WHERE scenario = ? AND receipt_id = ? AND compacted_ids_json = ?
                    """,
                    (scenario, receipt_id, compacted_ids_json),
                ).fetchone()
                if existing is not None and (
                    existing["compacted_ids_json"] != compacted_ids_json or existing["metadata_json"] != metadata_json
                ):
                    raise RecordConflict(
                        f"compaction receipt {receipt_id!r} for scenario {scenario!r} has different content"
                    )
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO compaction_receipts (
                        scenario, receipt_id, compacted_ids_json, metadata_json, recorded_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (scenario, receipt_id, compacted_ids_json, metadata_json, time.time()),
                )
            if agent_record_ids:
                sorted_ids = sorted(agent_record_ids)
                for start in range(0, len(sorted_ids), self._SQLITE_ID_CHUNK_SIZE):
                    chunk = sorted_ids[start : start + self._SQLITE_ID_CHUNK_SIZE]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = self._connection.execute(
                        f"SELECT * FROM agent_record WHERE scenario = ? AND agent_record_id IN ({placeholders})",
                        (scenario, *chunk),
                    ).fetchall()
                    self._connection.executemany(
                        """
                        INSERT OR IGNORE INTO consumed_agent_record (agent_record_id, content_sha256)
                        VALUES (?, ?)
                        """,
                        ((row["agent_record_id"], self._content_sha256(self._row_content(row))) for row in rows),
                    )
                    self._connection.execute(
                        f"DELETE FROM agent_record WHERE scenario = ? AND agent_record_id IN ({placeholders})",
                        (scenario, *chunk),
                    )
        for agent_record_id in agent_record_ids:
            self._live_records.pop(agent_record_id, None)

    def compaction_receipts(self, scenario: str) -> tuple[dict[str, object], ...]:
        """Return durable, ordered metadata for explicitly recorded compactions."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT receipt_id, compacted_ids_json, metadata_json, recorded_at
                FROM compaction_receipts
                WHERE scenario = ?
                ORDER BY recorded_at, receipt_id
                """,
                (scenario,),
            ).fetchall()
        return tuple(
            {
                "receipt_id": row["receipt_id"],
                "compacted_ids": tuple(json.loads(row["compacted_ids_json"])),
                "metadata": json.loads(row["metadata_json"]),
                "recorded_at": float(row["recorded_at"]),
            }
            for row in rows
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> RecordStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
