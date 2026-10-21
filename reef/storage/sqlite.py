"""SQLite records, scenario storage, and retained-file maintenance."""

from __future__ import annotations

import hashlib
import heapq
import logging
import math
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Self

from sqlalchemy import REAL, URL, Column, Index, Integer, MetaData, Table, Text, create_engine, inspect
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from reef.core.errors import ReefError
from reef.core.plugins import NoSettings
from reef.storage.commit_log import CommitHistoryScenarioStore, CommitLog
from reef.storage.migrations import archive_imported_files, legacy_model_configs, migrate_record_storage
from reef.storage.records import RecordRetention
from reef.storage.scenario import ScenarioSettings, ScenarioStorage
from reef.storage.sql_records import RecordTables, SQLRecordRetention, SQLRecordStore
from reef.storage.sql_tasks import SQLTaskStorage, task_tables
from reef.storage.sql_trajectories import SQLTrajectoryStore, TrajectoryTable, trajectory_table
from reef.storage.tasks import LocalTaskFileStore, TaskFileStore

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
CONSUMPTION_RECEIPTS = Table(
    "record_consumption",
    _METADATA,
    Column("scenario", Text, primary_key=True),
    Column("receipt_id", Text, primary_key=True),
    Column("consumed_ids_json", Text, nullable=False),
    Column("consumed_ids_sha256", Text, primary_key=True),
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
RECORD_EVICTION = Table(
    "record_eviction",
    _METADATA,
    Column("scenario", Text, primary_key=True),
    Column("record_count", Integer, nullable=False),
    Column("body_bytes", Integer, nullable=False),
    Column("first_sequence", Integer, nullable=False),
    Column("last_sequence", Integer, nullable=False),
)
Index("agent_record_capacity", _AGENT_RECORD.c.created_at, _AGENT_RECORD.c.sequence)
# One database file holds one scenario, so its settings need no scenario column.
_SCENARIO_SETTING = Table(
    "scenario_setting",
    _METADATA,
    Column("name", Text, primary_key=True),
    Column("value_json", Text, nullable=False),
)
TASK_TABLES = task_tables(_METADATA)
TRAJECTORY_TABLE = TrajectoryTable(trajectory_table(_METADATA))
logger = logging.getLogger(__name__)

_TABLES = RecordTables(_AGENT_RECORD, _CONSUMED_RECORD, CONSUMPTION_RECEIPTS, RECORD_EVICTION, _SCENARIO_SETTING)


class SQLiteRecordStore(SQLRecordStore):
    """Store scenario records in append order.

    The SQLite schema uses an ``agent_record`` table with an
    ``agent_record_id`` column, mirroring the wire id
    (``x-reef-agent-record-id``).

    Passing a filesystem path makes the store durable. The default in-memory
    database keeps standalone/test construction lightweight; production callers
    should always pass a path.

    Consumption does not change visibility. Capacity eviction is the only
    operation that deletes record bodies.
    """

    _WAL_SWITCH_TIMEOUT = 30.0
    _WAL_RETRY_INTERVAL = 0.01

    def __init__(self, database: str | Path | None = None) -> None:
        self._database = ":memory:" if database is None else str(database)
        if self._database != ":memory:":
            Path(self._database).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        super().__init__(_TABLES)
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
                _METADATA.create_all(self._connection)
                migrate_record_storage(self._connection, _AGENT_RECORD, CONSUMPTION_RECEIPTS)
                # create_all skips indexes on tables that already existed.
                for index in _AGENT_RECORD.indexes:
                    index.create(self._connection, checkfirst=True)

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[Connection]:
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

    def task_storage(self, files: TaskFileStore, scenario: str) -> SQLTaskStorage:
        """Open the task stores of ``scenario`` in this database, with their task files in ``files``."""
        return SQLTaskStorage(self, TASK_TABLES, files, scenario)

    def trajectory_store(self) -> SQLTrajectoryStore:
        """Open the uploaded trajectories in this database."""
        return SQLTrajectoryStore(self, TRAJECTORY_TABLE)

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

    The scenario storage service serializes cleanup with file archival.
    """

    def __init__(self, retention: RecordRetention) -> None:
        self._retention = retention
        self._queries = SQLRecordRetention(_TABLES)

    def prune(self, directory: Path) -> int:
        """Evict oldest bodies under capacity pressure, regardless of training state.

        Deletes commit in bounded batches. Concurrent appends can exceed the
        configured body budget until the next sweep. Freed database pages are
        reusable; the budget does not measure filesystem allocation.
        """
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
                if "body_bytes" not in columns:
                    # Incomplete stores must be upgraded by SQLiteRecordStore before maintenance.
                    continue
                retained_paths.append(str(database_path))
                with connection.begin():
                    connection.exec_driver_sql("BEGIN IMMEDIATE")
                    RECORD_EVICTION.create(connection, checkfirst=True)
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
            for created_at, sequence, size in rows:
                yield created_at, path, sequence, size
            after_time, after_sequence = rows[-1][:2]

    def _delete(self, path: str, sequences: list[int]) -> int:
        with self._connect(path) as connection, connection.begin():
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            losses = self._queries.evict(connection, sequences)
        for scenario, loss in losses:
            logger.warning(
                "Record capacity exceeded: evicted %d records (%d body bytes), scenario=%s, sequence=%d..%d; "
                "training data may be incomplete",
                loss.record_count,
                loss.body_bytes,
                scenario,
                loss.first_sequence,
                loss.last_sequence,
            )
        return sum(loss.record_count for _, loss in losses)


class SQLiteScenarioStorage(ScenarioStorage):
    """Open existing hashed SQLite and JSONL paths and manage their lifecycle.

    Each scenario has one database file for its records, settings, generated
    tasks and uploaded trajectories, and one JSONL commit log. Without a directory, all state is in memory.
    Records, commits and task states then last for one session, and settings
    last for the storage. ``task_files`` keeps the files of published tasks.
    The default is a ``LocalTaskFileStore`` in ``directory/task-files``. If
    the storage has no directory, the default store uses a temporary directory.
    """

    def __init__(self, directory: Path | None = None, *, task_files: TaskFileStore | None = None) -> None:
        self._directory = None if directory is None else Path(directory)
        self._lock = RLock()
        self._closed = False
        self._memory_settings: dict[str, ScenarioSettings] = {}
        if task_files is None:
            task_files = LocalTaskFileStore(None if self._directory is None else self._directory / "task-files")
        self.task_files = task_files
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._import_model_config_files(self._directory)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(directory, task_files=task_files)

    @property
    def durable(self) -> bool:
        return self._directory is not None

    def read_settings(self, scenario: str) -> ScenarioSettings:
        with self._lock:
            self._ensure_open()
            key = self._scenario_key(scenario)
            if self._directory is None:
                return self._memory_settings.get(scenario, ScenarioSettings())
            database = self._directory / f"{key}.sqlite3"
            if not database.exists():
                return ScenarioSettings()
            with closing(SQLiteRecordStore(database)) as records:
                return records.read_settings()

    def write_settings(self, scenario: str, settings: ScenarioSettings) -> None:
        with self._lock:
            self._ensure_open()
            key = self._scenario_key(scenario)
            if self._directory is None:
                self._memory_settings[scenario] = settings
                return
            self._write_database_settings(self._directory / f"{key}.sqlite3", settings)

    def open(self, scenario: str) -> CommitHistoryScenarioStore:
        with self._lock:
            self._ensure_open()
            key = self._scenario_key(scenario)
            database = None if self._directory is None else self._directory / f"{key}.sqlite3"
            commit_log = None if self._directory is None else CommitLog(self._directory / f"{key}.commits.jsonl")
            records = SQLiteRecordStore(database)
            try:
                return CommitHistoryScenarioStore(
                    scenario,
                    records,
                    commit_log,
                    tasks=records.task_storage(self.task_files, scenario),
                    trajectories=records.trajectory_store(),
                )
            except BaseException:
                records.close()
                raise

    def archive(self, scenario: str) -> tuple[str, ...]:
        with self._lock:
            self._ensure_open()
            paths = self.state_paths(scenario)
            archived_files = self.task_files.archive(scenario)
            if self._directory is None:
                self._memory_settings.pop(scenario, None)
                return archived_files
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            destination = self._directory / "archived" / f"{self._scenario_key(scenario)}-{stamp}-{uuid.uuid4().hex}"
            moved: list[str] = []
            for path in paths:
                if path.exists():
                    destination.mkdir(parents=True, exist_ok=True)
                    target = destination / path.name
                    shutil.move(str(path), str(target))
                    moved.append(str(target))
            return (*moved, *archived_files)

    def prune(self, *, max_bytes: int) -> int:
        with self._lock:
            self._ensure_open()
            retention = RecordRetention(max_bytes=max_bytes)
            return 0 if self._directory is None else SQLiteRecordRetention(retention).prune(self._directory)

    def state_paths(self, scenario: str) -> tuple[Path, ...]:
        """Existing local state paths; the stable process lock is never moved."""
        with self._lock:
            self._ensure_open()
            key = self._scenario_key(scenario)
            if self._directory is None:
                return ()
            return tuple(
                self._directory / name
                for name in (f"{key}.sqlite3", f"{key}.sqlite3-wal", f"{key}.sqlite3-shm", f"{key}.commits.jsonl")
            )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self.task_files.close()

    @staticmethod
    def _import_model_config_files(directory: Path) -> None:
        """Move model configuration files of older versions into their scenario databases."""
        imported = []
        for key, (value, path) in legacy_model_configs(directory).items():
            # Older versions stored no other setting, so the import replaces all settings.
            SQLiteScenarioStorage._write_database_settings(
                directory / f"{key}.sqlite3", ScenarioSettings(model_config=value)
            )
            imported.append(path)
        for location in archive_imported_files(directory, imported):
            logger.info("Imported a model configuration into its scenario database; the file moved to %s", location)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scenario storage is closed")

    @staticmethod
    def _write_database_settings(database: Path, settings: ScenarioSettings) -> None:
        """Write settings into a scenario database and keep its files owner-only.

        A model configuration can hold an upstream credential. Thus the database
        and its WAL files get the owner-only mode of the old model file.
        """
        with closing(SQLiteRecordStore(database)) as records:
            records.write_settings(settings)
        for path in (database, database.with_name(f"{database.name}-wal"), database.with_name(f"{database.name}-shm")):
            if path.exists():
                path.chmod(0o600)

    @staticmethod
    def _scenario_key(scenario: str) -> str:
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        return hashlib.sha256(scenario.encode("utf-8")).hexdigest()
