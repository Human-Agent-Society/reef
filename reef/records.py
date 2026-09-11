"""SQLite-backed record storage using SQLAlchemy Core."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import NamedTuple
from weakref import WeakValueDictionary

from sqlalchemy import (
    REAL,
    URL,
    Column,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Table,
    Text,
    cast,
    create_engine,
    func,
    inspect,
    select,
    tuple_,
)
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from reef.core.artifact_ref import decode_artifact_ref, encode_artifact_ref
from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RequestType

_METADATA = MetaData()
_AGENT_RECORD = Table(
    "agent_record",
    _METADATA,
    Column("sequence", Integer, primary_key=True, nullable=True),
    Column("agent_record_id", Text, nullable=False, unique=True),
    Column("scenario", Text, nullable=False),
    Column("request_type", Text, nullable=False),
    Column("payload_json", Text, nullable=False),
    Column("created_at", REAL, nullable=False),
    Column("references_json", Text, nullable=False),
    Column("artifact_json", Text),
    Column("compacted_at", REAL),
    Column("body_bytes", Integer, nullable=False, server_default="0"),
    # A purged sequence must never be reused by a later append.
    sqlite_autoincrement=True,
)
_CONSUMED_RECORD = Table(
    "consumed_agent_record",
    _METADATA,
    Column("agent_record_id", Text, primary_key=True, nullable=True),
    Column("content_sha256", Text, nullable=False),
)
_COMPACTION_RECEIPTS = Table(
    "compaction_receipts",
    _METADATA,
    Column("scenario", Text, primary_key=True),
    Column("receipt_id", Text, primary_key=True),
    Column("compacted_ids_json", Text, primary_key=True),
    Column("metadata_json", Text, nullable=False),
    Column("recorded_at", REAL, nullable=False),
)
Index("agent_record_scenario_sequence", _AGENT_RECORD.c.scenario, _AGENT_RECORD.c.sequence)
Index(
    "agent_record_scenario_type_sequence",
    _AGENT_RECORD.c.scenario,
    _AGENT_RECORD.c.request_type,
    _AGENT_RECORD.c.sequence,
)
Index(
    "agent_record_active_sequence",
    _AGENT_RECORD.c.scenario,
    _AGENT_RECORD.c.sequence,
    sqlite_where=_AGENT_RECORD.c.compacted_at.is_(None),
)
Index(
    "agent_record_active_type_sequence",
    _AGENT_RECORD.c.scenario,
    _AGENT_RECORD.c.request_type,
    _AGENT_RECORD.c.sequence,
    sqlite_where=_AGENT_RECORD.c.compacted_at.is_(None),
)
Index(
    "agent_record_compacted_at",
    _AGENT_RECORD.c.scenario,
    _AGENT_RECORD.c.compacted_at,
    _AGENT_RECORD.c.sequence,
    sqlite_where=_AGENT_RECORD.c.compacted_at.is_not(None),
)
Index(
    "agent_record_retention",
    _AGENT_RECORD.c.compacted_at,
    _AGENT_RECORD.c.sequence,
    _AGENT_RECORD.c.body_bytes,
    sqlite_where=_AGENT_RECORD.c.compacted_at.is_not(None),
)


class EncodedRecord(NamedTuple):
    """One record in its stored column order, as the ``agent_record`` row.

    The field names bind INSERT parameters by name and identify the fields
    that participate in retry-content comparison.
    """

    agent_record_id: str
    scenario: str
    request_type: str
    created_at: float
    payload_json: str
    references_json: str
    artifact_json: str | None


class RecordConflict(ReefError):
    """Raised when append content conflicts with an existing agent_record_id."""


@dataclass(frozen=True)
class AppendResult:
    item: AgentRecord
    inserted: bool


@dataclass(frozen=True)
class StoredRecord:
    """A retained record and its storage state, for audit reads only.

    ``compacted_at`` marks retirement from training, not proof of learning.
    The commit log's ``consumed_ids`` identifies which records a step consumed.
    """

    sequence: int
    item: AgentRecord
    compacted_at: float | None


class RecordStore:
    """Store scenario records in append order.

    The SQLite schema uses an ``agent_record`` table with an
    ``agent_record_id`` column, mirroring the wire id
    (``x-reef-agent-record-id``).

    Passing a filesystem path makes the store durable. The default in-memory
    database keeps standalone/test construction lightweight; production callers
    should always pass a path.

    Training reads hide compacted rows. Explicit audit reads retain access to
    their bodies until :meth:`purge_compacted` physically removes them.
    """

    _SQLITE_ID_CHUNK_SIZE = 900
    #: How long to keep retrying the WAL switch, matching the connection timeout.
    _WAL_SWITCH_TIMEOUT = 30.0
    _WAL_RETRY_INTERVAL = 0.01

    def __init__(self, database: str | Path | None = None) -> None:
        self._database = ":memory:" if database is None else str(database)
        if self._database != ":memory:":
            Path(self._database).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._live_records: WeakValueDictionary[str, AgentRecord] = WeakValueDictionary()
        self._engine = create_engine(
            URL.create("sqlite+pysqlite", database=self._database),
            connect_args={"timeout": 30, "check_same_thread": False},
            poolclass=NullPool,
            hide_parameters=True,
        )
        # Hold one connection for the store's lifetime, including for :memory:.
        # The lock serializes all use across HTTP and training threads.
        self._connection = self._engine.connect()
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    @property
    def database(self) -> str:
        return self._database

    def _enable_wal(self) -> None:
        """Switch the database to WAL, waiting out a concurrent opener.

        SQLite takes an exclusive lock to change the journal mode and answers
        SQLITE_BUSY immediately instead of running the busy handler, so the
        connection timeout does not cover this statement. Openers that race to
        upgrade a database still in rollback mode therefore have to retry: the
        first one converts it and the rest read back ``wal`` on a later attempt.
        A database already in WAL answers on the first try, so the steady state
        costs nothing.
        """
        deadline = time.monotonic() + self._WAL_SWITCH_TIMEOUT
        observed = None
        while True:
            # A contended switch shows up either way: as SQLITE_BUSY, or as the
            # unchanged mode read back, which SQLite returns instead of raising.
            try:
                row = self._connection.exec_driver_sql("PRAGMA journal_mode = WAL").fetchone()
            except OperationalError as exc:
                if "locked" not in str(exc.orig) and "busy" not in str(exc.orig):
                    raise
            else:
                observed = None if row is None else str(row[0]).lower()
                if observed == "wal":
                    return
            if time.monotonic() >= deadline:
                raise ReefError(
                    f"could not switch {self._database} to WAL within {self._WAL_SWITCH_TIMEOUT:g}s; "
                    f"another connection held the database (journal mode is {observed or 'unknown'})"
                )
            time.sleep(self._WAL_RETRY_INTERVAL)

    def _initialize(self) -> None:
        with self._lock:
            if self._database != ":memory:":
                self._enable_wal()
                self._connection.exec_driver_sql("PRAGMA synchronous = FULL")
                self._connection.commit()
            with self._connection.begin():
                # SQLite-specific transaction control and legacy ALTERs stay at
                # this boundary; schema creation and all data queries use Core.
                self._connection.exec_driver_sql("BEGIN IMMEDIATE")
                if inspect(self._connection).has_table(_AGENT_RECORD.name):
                    columns = {column["name"] for column in inspect(self._connection).get_columns(_AGENT_RECORD.name)}
                    if "compacted_at" not in columns:
                        self._connection.exec_driver_sql("ALTER TABLE agent_record ADD COLUMN compacted_at REAL")
                    if "body_bytes" not in columns:
                        self._connection.exec_driver_sql(
                            "ALTER TABLE agent_record ADD COLUMN body_bytes INTEGER NOT NULL DEFAULT 0"
                        )
                        self._connection.execute(
                            _AGENT_RECORD.update().values(
                                body_bytes=func.length(cast(_AGENT_RECORD.c.payload_json, LargeBinary))
                                + func.length(cast(_AGENT_RECORD.c.references_json, LargeBinary))
                                + func.coalesce(func.length(cast(_AGENT_RECORD.c.artifact_json, LargeBinary)), 0)
                            )
                        )
                _METADATA.create_all(self._connection)
                # create_all skips indexes on tables that already existed.
                for index in _AGENT_RECORD.indexes:
                    index.create(self._connection, checkfirst=True)

    @staticmethod
    def _json(value: object) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("a record must contain JSON-serializable values") from exc

    @classmethod
    def _encode(cls, item: AgentRecord) -> EncodedRecord:
        artifact = item.artifact_ref
        artifact_json = None
        if artifact is not None:
            artifact_json = cls._json(encode_artifact_ref(artifact))
        return EncodedRecord(
            agent_record_id=item.agent_record_id,
            scenario=item.scenario,
            request_type=item.request_type.value,
            created_at=item.created_at,
            payload_json=cls._json(dict(item.payload)),
            references_json=cls._json(item.references),
            artifact_json=artifact_json,
        )

    @staticmethod
    def _decode(row: RowMapping) -> AgentRecord:
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
    def _row_content(row: RowMapping) -> EncodedRecord:
        return EncodedRecord(*(row[name] for name in EncodedRecord._fields))

    @classmethod
    def _content(cls, encoded: EncodedRecord) -> dict[str, object]:
        """The encoded fields that define row content, excluding ``created_at``.

        A client retrying with its own agent_record_id regenerates the
        timestamp, so a timestamp difference alone must dedup, not conflict.
        """
        return {name: value for name, value in encoded._asdict().items() if name != "created_at"}

    @classmethod
    def _content_sha256(cls, encoded: EncodedRecord) -> str:
        canonical = cls._json(cls._content(encoded)).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def append(self, item: AgentRecord) -> AgentRecord:
        return self.append_result(item).item

    def existing_receipt(self, item: AgentRecord) -> AgentRecord | None:
        """Validate a retry before applying admission rules for new records.

        Compacted records retain content hashes, so an already accepted
        instruction can still be retried after the training mode changes.
        This lookup never appends data or changes its receipt.
        """
        encoded = self._encode(item)
        with self._lock, self._connection.begin():
            consumed = (
                self._connection.execute(
                    select(_CONSUMED_RECORD.c.content_sha256).where(
                        _CONSUMED_RECORD.c.agent_record_id == item.agent_record_id
                    )
                )
                .mappings()
                .first()
            )
            if consumed is not None:
                if consumed["content_sha256"] != self._content_sha256(encoded):
                    raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
                return item
            row = (
                self._connection.execute(
                    select(_AGENT_RECORD).where(_AGENT_RECORD.c.agent_record_id == item.agent_record_id)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            if self._content(self._row_content(row)) != self._content(encoded):
                raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
            return self._decode(row)

    def append_result(self, item: AgentRecord) -> AppendResult:
        encoded = self._encode(item)
        with self._lock, self._connection.begin():
            consumed = (
                self._connection.execute(
                    select(_CONSUMED_RECORD.c.content_sha256).where(
                        _CONSUMED_RECORD.c.agent_record_id == item.agent_record_id
                    )
                )
                .mappings()
                .first()
            )
            if consumed is not None:
                if consumed["content_sha256"] != self._content_sha256(encoded):
                    raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
                return AppendResult(item, False)
            if item.request_type is RequestType.REPORT and item.references:
                retired_reference = self._connection.execute(
                    select(_CONSUMED_RECORD.c.agent_record_id)
                    .where(_CONSUMED_RECORD.c.agent_record_id.in_(item.references))
                    .limit(1)
                ).first()
                if retired_reference is not None:
                    # A report is discarded once its references are gone, but a row
                    # already stored under this id stays canonical: check the discard
                    # against that row so a divergent retry cannot register its own
                    # content as the receipt and reject the honest retry that follows.
                    existing = (
                        self._connection.execute(
                            select(_AGENT_RECORD).where(_AGENT_RECORD.c.agent_record_id == item.agent_record_id)
                        )
                        .mappings()
                        .first()
                    )
                    if existing is not None and self._content(self._row_content(existing)) != self._content(encoded):
                        raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
                    self._connection.execute(
                        _CONSUMED_RECORD.insert().values(
                            agent_record_id=item.agent_record_id, content_sha256=self._content_sha256(encoded)
                        )
                    )
                    return AppendResult(item, False)
            cursor = self._connection.execute(
                _AGENT_RECORD.insert().prefix_with("OR IGNORE"),
                {
                    **encoded._asdict(),
                    "body_bytes": sum(len(value.encode("utf-8")) for value in encoded[4:] if value is not None),
                },
            )
            if cursor.rowcount == 1:
                self._live_records[item.agent_record_id] = item
                return AppendResult(item, True)
            existing = (
                self._connection.execute(
                    select(_AGENT_RECORD).where(_AGENT_RECORD.c.agent_record_id == item.agent_record_id)
                )
                .mappings()
                .first()
            )
            if existing is None or self._content(self._row_content(existing)) != self._content(encoded):
                raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
            live = self._live_records.get(item.agent_record_id)
            if live is not None:
                return AppendResult(live, False)
            stored = self._decode(existing)
            self._live_records[item.agent_record_id] = stored
            return AppendResult(stored, False)

    def get(self, scenario: str, agent_record_id: str) -> AgentRecord | None:
        """Read a record still visible to training, scoped to its scenario."""
        with self._lock, self._connection.begin():
            row = (
                self._connection.execute(
                    select(_AGENT_RECORD).where(
                        _AGENT_RECORD.c.scenario == scenario,
                        _AGENT_RECORD.c.agent_record_id == agent_record_id,
                        _AGENT_RECORD.c.compacted_at.is_(None),
                    )
                )
                .mappings()
                .first()
            )
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
        statement = (
            select(_AGENT_RECORD)
            .where(_AGENT_RECORD.c.scenario == scenario, _AGENT_RECORD.c.compacted_at.is_(None))
            .order_by(_AGENT_RECORD.c.sequence)
            .limit(limit)
            .offset(offset)
        )
        with self._lock, self._connection.begin():
            rows = self._connection.execute(statement).mappings().all()
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
        with self._lock, self._connection.begin():
            rows = (
                self._connection.execute(
                    select(_AGENT_RECORD)
                    .where(
                        _AGENT_RECORD.c.scenario == scenario,
                        _AGENT_RECORD.c.sequence > after_sequence,
                        _AGENT_RECORD.c.compacted_at.is_(None),
                    )
                    .order_by(_AGENT_RECORD.c.sequence)
                    .limit(limit)
                )
                .mappings()
                .all()
            )
        return tuple((int(row["sequence"]), self._decode(row)) for row in rows)

    def count(self, scenario: str, *, request_type: RequestType | None = None, after_sequence: int = 0) -> int:
        """Count training-visible records, optionally by type and after an append sequence."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        statement = (
            select(func.count())
            .select_from(_AGENT_RECORD)
            .where(
                _AGENT_RECORD.c.scenario == scenario,
                _AGENT_RECORD.c.sequence > after_sequence,
                _AGENT_RECORD.c.compacted_at.is_(None),
            )
        )
        if request_type is not None:
            statement = statement.where(_AGENT_RECORD.c.request_type == request_type.value)
        with self._lock, self._connection.begin():
            return int(self._connection.execute(statement).scalar_one())

    def get_for_audit(self, scenario: str, agent_record_id: str) -> StoredRecord | None:
        """Read a retained record including its compaction state; never reactivate it."""
        with self._lock, self._connection.begin():
            row = (
                self._connection.execute(
                    select(_AGENT_RECORD).where(
                        _AGENT_RECORD.c.scenario == scenario, _AGENT_RECORD.c.agent_record_id == agent_record_id
                    )
                )
                .mappings()
                .first()
            )
        return None if row is None else self._audit_record(row)

    def audit_page(
        self,
        scenario: str,
        *,
        after_sequence: int = 0,
        limit: int = 256,
    ) -> tuple[StoredRecord, ...]:
        """Read a bounded append-order page including compacted bodies, scoped to one scenario."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._lock, self._connection.begin():
            rows = (
                self._connection.execute(
                    select(_AGENT_RECORD)
                    .where(_AGENT_RECORD.c.scenario == scenario, _AGENT_RECORD.c.sequence > after_sequence)
                    .order_by(_AGENT_RECORD.c.sequence)
                    .limit(limit)
                )
                .mappings()
                .all()
            )
        return tuple(self._audit_record(row) for row in rows)

    @classmethod
    def _audit_record(cls, row: RowMapping) -> StoredRecord:
        return StoredRecord(sequence=int(row["sequence"]), item=cls._decode(row), compacted_at=row["compacted_at"])

    def compact(
        self,
        scenario: str,
        agent_record_ids: frozenset[str],
        *,
        receipt_id: str | None = None,
        receipt_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Retire records from training while retaining their bodies for audit.

        Compacted rows no longer participate in training replay, lookup, or
        reference availability. Durable receipts preserve append deduplication
        and refuse reports that reference retired data. Repeated compaction
        preserves the first retirement time. Physical deletion is separate.
        """
        if (receipt_id is None) != (receipt_metadata is None):
            raise ValueError("compaction receipt_id and receipt_metadata must be provided together")
        if receipt_id is not None and not receipt_id:
            raise ValueError("compaction receipt_id must be non-empty")
        if not agent_record_ids and receipt_id is None:
            return
        compacted_ids_json = self._json(sorted(agent_record_ids))
        metadata_json = self._json(dict(receipt_metadata or {}))
        compacted_at = time.time()
        with self._lock, self._connection.begin():
            if receipt_id is not None:
                # A receipt is identified by (scenario, receipt_id, compacted ids), the
                # primary key of compaction_receipts. One receipt_id may therefore cover
                # several distinct id sets, so only the metadata can conflict.
                existing = (
                    self._connection.execute(
                        select(_COMPACTION_RECEIPTS.c.metadata_json).where(
                            _COMPACTION_RECEIPTS.c.scenario == scenario,
                            _COMPACTION_RECEIPTS.c.receipt_id == receipt_id,
                            _COMPACTION_RECEIPTS.c.compacted_ids_json == compacted_ids_json,
                        )
                    )
                    .mappings()
                    .first()
                )
                if existing is not None and existing["metadata_json"] != metadata_json:
                    raise RecordConflict(
                        f"compaction receipt {receipt_id!r} for scenario {scenario!r} has different content"
                    )
                self._connection.execute(
                    _COMPACTION_RECEIPTS.insert()
                    .prefix_with("OR IGNORE")
                    .values(
                        scenario=scenario,
                        receipt_id=receipt_id,
                        compacted_ids_json=compacted_ids_json,
                        metadata_json=metadata_json,
                        recorded_at=time.time(),
                    )
                )
            if agent_record_ids:
                sorted_ids = sorted(agent_record_ids)
                for start in range(0, len(sorted_ids), self._SQLITE_ID_CHUNK_SIZE):
                    chunk = sorted_ids[start : start + self._SQLITE_ID_CHUNK_SIZE]
                    selected = (
                        _AGENT_RECORD.c.scenario == scenario,
                        _AGENT_RECORD.c.compacted_at.is_(None),
                        _AGENT_RECORD.c.agent_record_id.in_(chunk),
                    )
                    rows = self._connection.execute(select(_AGENT_RECORD).where(*selected)).mappings().all()
                    if rows:
                        self._connection.execute(
                            _CONSUMED_RECORD.insert().prefix_with("OR IGNORE"),
                            [
                                {
                                    "agent_record_id": row["agent_record_id"],
                                    "content_sha256": self._content_sha256(self._row_content(row)),
                                }
                                for row in rows
                            ],
                        )
                    self._connection.execute(_AGENT_RECORD.update().where(*selected).values(compacted_at=compacted_at))
        for agent_record_id in agent_record_ids:
            self._live_records.pop(agent_record_id, None)

    def purge_compacted(self, scenario: str, *, before: float, limit: int = 256) -> int:
        """Delete at most ``limit`` bodies retired before a finite Unix timestamp.

        This explicit operation is irreversible. Active records, retry hashes,
        and compaction receipts are retained. This call schedules no further maintenance.
        """
        if not math.isfinite(before):
            raise ValueError("before must be a finite Unix timestamp")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        selected = (
            select(_AGENT_RECORD.c.sequence)
            .where(_AGENT_RECORD.c.scenario == scenario, _AGENT_RECORD.c.compacted_at < before)
            .order_by(_AGENT_RECORD.c.compacted_at, _AGENT_RECORD.c.sequence)
            .limit(limit)
        )
        with self._lock, self._connection.begin():
            cursor = self._connection.execute(_AGENT_RECORD.delete().where(_AGENT_RECORD.c.sequence.in_(selected)))
            return cursor.rowcount

    def compaction_receipts(self, scenario: str) -> tuple[dict[str, object], ...]:
        """Return durable, ordered metadata for explicitly recorded compactions."""
        with self._lock, self._connection.begin():
            rows = (
                self._connection.execute(
                    select(_COMPACTION_RECEIPTS)
                    .where(_COMPACTION_RECEIPTS.c.scenario == scenario)
                    .order_by(_COMPACTION_RECEIPTS.c.recorded_at, _COMPACTION_RECEIPTS.c.receipt_id)
                )
                .mappings()
                .all()
            )
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
            self._engine.dispose()

    def __enter__(self) -> RecordStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


@dataclass(frozen=True)
class RecordRetention:
    """Deployment-wide limits on compacted JSON bodies, applied by service maintenance.

    The byte budget excludes active records, indexes, tombstones, and WAL pages.
    Cleanup reuses SQLite pages; it does not impose a physical file-size limit.
    """

    days: float = 7.0
    max_bytes: int = 20 * 1024**3

    def __post_init__(self) -> None:
        if (
            isinstance(self.days, bool)
            or not isinstance(self.days, (int, float))
            or not math.isfinite(self.days)
            or self.days <= 0
        ):
            raise ValueError("agent_record_retention_days must be positive and finite")
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or self.max_bytes <= 0:
            raise ValueError("agent_record_retention_max_bytes must be a positive integer")

    def prune(self, directory: Path) -> int:
        """Purge expired bodies, then the oldest bodies across this directory to meet the budget.

        The caller must serialize this sweep with scenario file moves. Deletes
        commit in batches of 256 and never touch active records or retry metadata.
        Concurrent compaction may exceed the budget until the next sweep.
        """
        cutoff = time.time() - self.days * 86400
        paths = sorted((*directory.glob("*.sqlite3"), *(directory / "archived").rglob("*.sqlite3")))
        purged = 0
        total = 0
        retained_paths: list[str] = []
        for database_path in paths:
            with self._connect(str(database_path)) as connection:
                with connection.begin():
                    inspector = inspect(connection)
                    columns = (
                        {column["name"] for column in inspector.get_columns(_AGENT_RECORD.name)}
                        if inspector.has_table(_AGENT_RECORD.name)
                        else set()
                    )
                if not {"compacted_at", "body_bytes"} <= columns:
                    # Old stores have no retained compacted bodies; migration belongs to RecordStore.
                    continue
                retained_paths.append(str(database_path))
                expired = (
                    select(_AGENT_RECORD.c.sequence)
                    .where(_AGENT_RECORD.c.compacted_at < cutoff)
                    .order_by(_AGENT_RECORD.c.compacted_at, _AGENT_RECORD.c.sequence)
                    .limit(256)
                )
                while True:
                    with connection.begin():
                        count = connection.execute(
                            _AGENT_RECORD.delete().where(_AGENT_RECORD.c.sequence.in_(expired))
                        ).rowcount
                    purged += count
                    if count < 256:
                        break

                with connection.begin():
                    total += connection.execute(
                        select(func.coalesce(func.sum(_AGENT_RECORD.c.body_bytes), 0)).where(
                            _AGENT_RECORD.c.compacted_at.is_not(None)
                        )
                    ).scalar_one()
        if total <= self.max_bytes:
            return purged
        pending: dict[str, list[int]] = {path: [] for path in retained_paths}
        rows = heapq.merge(*(self._rows(path) for path in retained_paths))
        for _, path, sequence, size in rows:
            pending[path].append(sequence)
            total -= size
            if len(pending[path]) == 256:
                purged += self._delete(path, pending[path])
                pending[path].clear()
            if total <= self.max_bytes:
                break
        for path, sequences in pending.items():
            if sequences:
                purged += self._delete(path, sequences)
        return purged

    @staticmethod
    @contextmanager
    def _connect(path: str) -> Iterator[Connection]:
        # URI mode=rw prevents maintenance from recreating a moved database.
        engine = create_engine(
            URL.create("sqlite+pysqlite", database=Path(path).resolve().as_uri(), query={"mode": "rw", "uri": "true"}),
            connect_args={"timeout": 30},
            poolclass=NullPool,
            hide_parameters=True,
        )
        try:
            with engine.connect() as connection:
                yield connection
        finally:
            engine.dispose()

    @classmethod
    def _rows(cls, path: str) -> Iterator[tuple[float, str, int, int]]:
        after_time: float = -math.inf
        after_sequence = 0
        while True:
            with cls._connect(path) as connection, connection.begin():
                rows = connection.execute(
                    select(_AGENT_RECORD.c.compacted_at, _AGENT_RECORD.c.sequence, _AGENT_RECORD.c.body_bytes)
                    .where(
                        _AGENT_RECORD.c.compacted_at.is_not(None),
                        tuple_(_AGENT_RECORD.c.compacted_at, _AGENT_RECORD.c.sequence) > (after_time, after_sequence),
                    )
                    .order_by(_AGENT_RECORD.c.compacted_at, _AGENT_RECORD.c.sequence)
                    .limit(256)
                ).all()
            if not rows:
                return
            for compacted_at, sequence, size in rows:
                yield compacted_at, path, sequence, size
            after_time, after_sequence = rows[-1][:2]

    @classmethod
    def _delete(cls, path: str, sequences: list[int]) -> int:
        with cls._connect(path) as connection, connection.begin():
            return connection.execute(
                _AGENT_RECORD.delete().where(
                    _AGENT_RECORD.c.compacted_at.is_not(None), _AGENT_RECORD.c.sequence.in_(sequences)
                )
            ).rowcount
