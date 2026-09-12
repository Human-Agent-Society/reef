"""SQLite schema, connections, and retained-file maintenance for SQL record storage."""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from alembic.migration import MigrationContext
from alembic.operations import Operations
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
)
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from reef.core.errors import ReefError
from reef.records import RecordRetention
from reef.storage.sql_records import RecordTables, SQLRecordRetention, SQLRecordStore

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

_TABLES = RecordTables(_AGENT_RECORD, _CONSUMED_RECORD, _COMPACTION_RECEIPTS)


class SQLiteRecordStore(SQLRecordStore):
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
    _WAL_SWITCH_TIMEOUT = 30.0
    _WAL_RETRY_INTERVAL = 0.01

    def __init__(self, database: str | Path | None = None) -> None:
        self._database = ":memory:" if database is None else str(database)
        if self._database != ":memory:":
            Path(self._database).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        super().__init__(_TABLES, id_chunk_size=self._SQLITE_ID_CHUNK_SIZE)
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
                # Keep Alembic's column upgrades and the Core backfill in one
                # SQLite transaction, serialized across concurrent openers.
                self._connection.exec_driver_sql("BEGIN IMMEDIATE")
                if inspect(self._connection).has_table(_AGENT_RECORD.name):
                    columns = {column["name"] for column in inspect(self._connection).get_columns(_AGENT_RECORD.name)}
                    if not {"compacted_at", "body_bytes"} <= columns:
                        operations = Operations(MigrationContext.configure(self._connection))
                        if "compacted_at" not in columns:
                            operations.add_column(_AGENT_RECORD.name, Column("compacted_at", REAL))
                        if "body_bytes" not in columns:
                            operations.add_column(
                                _AGENT_RECORD.name, Column("body_bytes", Integer, nullable=False, server_default="0")
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

    @contextmanager
    def _transaction(self, scenario: str, *, write: bool) -> Iterator[Connection]:
        with self._lock, self._connection.begin():
            # The driver's legacy mode does not begin a transaction for SELECT.
            # Writes also lock before reading retry state across connections.
            self._connection.exec_driver_sql("BEGIN IMMEDIATE" if write else "BEGIN")
            yield self._connection

    def _insert_if_absent(
        self,
        connection: Connection,
        table: Table,
        values: Mapping[str, object] | Sequence[Mapping[str, object]],
    ) -> bool:
        return connection.execute(table.insert().prefix_with("OR IGNORE"), values).rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._connection.close()
            self._engine.dispose()

    def __enter__(self) -> SQLiteRecordStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class SQLiteRecordRetention:
    """Apply a deployment's body-retention limits to its SQLite record files.

    The scenario storage factory serializes cleanup with file archival.
    """

    def __init__(self, retention: RecordRetention) -> None:
        self._retention = retention
        self._queries = SQLRecordRetention(_TABLES)

    def prune(self, directory: Path) -> int:
        """Purge expired bodies, then the oldest bodies across this directory to meet the budget.

        The caller must serialize this sweep with scenario file moves. Deletes
        commit in batches of 256 and never touch active records or retry metadata.
        Concurrent compaction may exceed the budget until the next sweep.
        """
        cutoff = time.time() - self._retention.days * 86400
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
                    # Old stores have no retained compacted bodies; migration belongs to SQLiteRecordStore.
                    continue
                retained_paths.append(str(database_path))
                while True:
                    with connection.begin():
                        count = self._queries.purge_expired(connection, before=cutoff)
                    purged += count
                    if count < 256:
                        break

                with connection.begin():
                    total += self._queries.retained_bytes(connection)
        if total <= self._retention.max_bytes:
            return purged
        pending: dict[str, list[int]] = {path: [] for path in retained_paths}
        rows = heapq.merge(*(self._rows(path) for path in retained_paths))
        for _, path, sequence, size in rows:
            pending[path].append(sequence)
            total -= size
            if len(pending[path]) == 256:
                purged += self._delete(path, pending[path])
                pending[path].clear()
            if total <= self._retention.max_bytes:
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

    def _rows(self, path: str) -> Iterator[tuple[float, str, int, int]]:
        after_time: float = -math.inf
        after_sequence = 0
        while True:
            with self._connect(path) as connection, connection.begin():
                rows = self._queries.page(connection, after_time=after_time, after_sequence=after_sequence)
            if not rows:
                return
            for compacted_at, sequence, size in rows:
                yield compacted_at, path, sequence, size
            after_time, after_sequence = rows[-1][:2]

    def _delete(self, path: str, sequences: list[int]) -> int:
        with self._connect(path) as connection, connection.begin():
            return self._queries.delete(connection, sequences)
