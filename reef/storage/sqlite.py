"""SQLite scenario databases and scenario storage.

Each scenario has one database file for its commits, settings, generated tasks, training state and evaluations. Its
records are record items in the storage's data store (RFC #1219).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Self
from weakref import WeakValueDictionary

from sqlalchemy import URL, Column, MetaData, Table, Text, create_engine, inspect
from sqlalchemy.engine import Connection
from sqlalchemy.pool import NullPool

from reef.core.plugins import NoSettings
from reef.storage.commit_history import CommitHistoryScenarioStore
from reef.storage.data import DataStore
from reef.storage.migrations import (
    migrate_manual_buffers,
    migrate_published_tasks,
    migrate_scenario_settings,
    migrate_training_samples,
)
from reef.storage.record_items import ItemRecordStore, old_record_copied
from reef.storage.scenario import ScenarioSettings, ScenarioStorage
from reef.storage.sql_commits import SQLCommitHistory, commit_table
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sql_evaluations import SQLEvaluationStore, evaluation_table
from reef.storage.sql_records import ScopedTable, SQLScenarioDatabase
from reef.storage.sql_samples import SQLTrainingState, sample_train_count_table, stage_record_table
from reef.storage.sql_setup import enable_wal
from reef.storage.sql_tasks import SQLTaskStorage, task_tables
from reef.storage.tasks import LocalTaskFileStore, TaskFileStore

# Named memory databases share table locks; weak references release locks after their last store is gone.
memory_locks: WeakValueDictionary[str, RLock] = WeakValueDictionary()
memory_locks_lock = RLock()
_METADATA = MetaData()
# One database file holds one scenario, so its settings need no scenario column.
_SCENARIO_SETTING = Table(
    "scenario_setting",
    _METADATA,
    Column("name", Text, primary_key=True),
    Column("value_json", Text, nullable=False),
)
TASK_TABLES = task_tables(_METADATA)
SAMPLE_TRAIN_COUNT_TABLE = ScopedTable(sample_train_count_table(_METADATA))
STAGE_RECORD_TABLE = ScopedTable(stage_record_table(_METADATA))
EVALUATION_TABLE = ScopedTable(evaluation_table(_METADATA))
COMMIT_TABLE = ScopedTable(commit_table(_METADATA))
#: The record tables of a version before the record items (RFC #1219); the newest of them is ``agent_record``.
OLD_RECORD_TABLES = ("record_node", "consumed_agent_record", "record_consumption", "record_eviction", "agent_record")
logger = logging.getLogger(__name__)


class SQLiteScenarioDatabase(SQLScenarioDatabase):
    """The SQLite database of one scenario's state.

    A filesystem path makes the database durable. Without one it is in memory, for standalone and test
    construction; production callers pass a path.
    """

    _WAL_SWITCH_TIMEOUT = 30.0
    _WAL_RETRY_INTERVAL = 0.01

    def __init__(self, database: str | Path | None = None, *, memory_name: str | None = None) -> None:
        """``database`` is the file; without it the database is in memory. ``memory_name`` names a shared in-memory
        database: every database of one name in this process reads and writes the same state while one of them is
        open. Transactions on one named memory database share a process-local lock across all instances."""
        if memory_name is not None and database is not None:
            raise ValueError("a database is in a file or in a named memory database, not both")
        self._database = ":memory:" if database is None else str(database)
        if self._database != ":memory:":
            Path(self._database).parent.mkdir(parents=True, exist_ok=True)
        if memory_name is None:
            self.transaction_lock = RLock()
        else:
            with memory_locks_lock:
                shared_lock = memory_locks.get(memory_name)
                if shared_lock is None:
                    shared_lock = RLock()
                    memory_locks[memory_name] = shared_lock
                self.transaction_lock = shared_lock
        super().__init__(_SCENARIO_SETTING)
        url = (
            URL.create("sqlite+pysqlite", database=self._database)
            if memory_name is None
            else URL.create(
                "sqlite+pysqlite",
                database=f"file:{memory_name}",
                query={"mode": "memory", "cache": "shared", "uri": "true"},
            )
        )
        self._engine = create_engine(
            url,
            connect_args={"timeout": 30, "check_same_thread": False},
            poolclass=NullPool,
            hide_parameters=True,
        )
        # Hold one connection for the database's lifetime, including for :memory:.
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
        """Switch the database to WAL, waiting out a concurrent opener (``enable_wal``)."""
        enable_wal(
            self._connection,
            self._database,
            timeout_seconds=self._WAL_SWITCH_TIMEOUT,
            retry_seconds=self._WAL_RETRY_INTERVAL,
        )

    def _initialize(self) -> None:
        with self.transaction_lock:
            if self._database != ":memory:":
                self._enable_wal()
                self._connection.exec_driver_sql("PRAGMA synchronous = FULL")
                self._connection.commit()
            with self._connection.begin():
                # Keep Alembic's column upgrades and the Core backfill in one
                # SQLite transaction, serialized across concurrent openers.
                self._connection.exec_driver_sql("BEGIN IMMEDIATE")
                _METADATA.create_all(self._connection)
                migrate_training_samples(self._connection, SAMPLE_TRAIN_COUNT_TABLE.table)
                migrate_scenario_settings(self._connection, _SCENARIO_SETTING)
                migrate_manual_buffers(self._connection, None)
                migrate_published_tasks(self._connection, None)

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[Connection]:
        with self.transaction_lock, self._connection.begin():
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

    def training_state(self) -> SQLTrainingState:
        """Open the training state of the scenario in this database."""
        return SQLTrainingState(self, SAMPLE_TRAIN_COUNT_TABLE, STAGE_RECORD_TABLE)

    def evaluation_store(self) -> SQLEvaluationStore:
        """Open the artifact evaluations in this database."""
        return SQLEvaluationStore(self, EVALUATION_TABLE)

    def commit_history(self) -> SQLCommitHistory:
        """Open the commit history in this database; it is durable unless the database is in memory."""
        return SQLCommitHistory(
            self,
            COMMIT_TABLE,
            SAMPLE_TRAIN_COUNT_TABLE,
            label=f"{self._database}#scenario_commit",
            durable=self._database != ":memory:",
        )

    def close(self) -> None:
        with self.transaction_lock:
            self._connection.close()
            self._engine.dispose()

    def drop_copied_records(self, data: DataStore, owner: str, scenario: str) -> bool:
        """Drop the ``OLD_RECORD_TABLES`` when the record item of their newest record is in ``data``; return whether
        the database has none of them then. A version from before the record items copied each record at the first
        open of its scenario, in record order, so the newest record shows that the copy is complete."""
        with self._transaction(write=True) as connection:
            if not inspect(connection).has_table("agent_record"):
                return True
            newest = connection.exec_driver_sql(
                "SELECT agent_record_id FROM agent_record ORDER BY sequence DESC LIMIT 1"
            ).scalar_one_or_none()
            if newest is not None and not old_record_copied(data, owner, scenario, newest):
                return False
            for table in OLD_RECORD_TABLES:
                connection.exec_driver_sql(f"DROP TABLE IF EXISTS {table}")
            return True

    def __enter__(self) -> SQLiteScenarioDatabase:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class SQLiteScenarioStorage(ScenarioStorage):
    """Open hashed SQLite database paths and manage their lifecycle.

    Each scenario has one database file for its commits, settings, generated tasks, training state and evaluations.
    Its records are record items in ``data`` (``ItemRecordStore``). Without a directory, all state is in memory and
    lasts for the storage: every session of one scenario reads and writes the same memory database, as the services
    of one process share the store.
    ``task_files`` keeps the files of published tasks. The default is a ``LocalTaskFileStore`` in
    ``directory/task-files``; without a directory, it uses a temporary directory. ``data`` is the data store. The
    default is a ``SQLiteDataStore`` in ``directory/data``, or in memory without a directory; the storage closes a
    default data store with itself.

    The first open of a scenario whose database still has the record tables of an older version drops them when its
    records are record items (``SQLiteScenarioDatabase.drop_copied_records``). Else it keeps them and logs a warning.
    """

    def __init__(
        self,
        directory: Path | None = None,
        *,
        task_files: TaskFileStore | None = None,
        data: DataStore | None = None,
    ) -> None:
        self._directory = None if directory is None else Path(directory)
        self._lock = RLock()
        self._closed = False
        self._memory_settings: dict[str, ScenarioSettings] = {}
        #: Without a directory: per scenario, the name of its memory database and one open database that keeps it
        #: alive between sessions. An archive drops both, so the next session starts a new database.
        self._memory_anchors: dict[str, tuple[str, SQLiteScenarioDatabase]] = {}
        #: The scenarios whose databases have no record tables of an older version.
        self._checked: set[str] = set()
        if task_files is None:
            task_files = LocalTaskFileStore(None if self._directory is None else self._directory / "task-files")
        self.task_files = task_files
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)
        self._owns_data = data is None
        self.data = (
            SQLiteDataStore(None if self._directory is None else self._directory / "data", task_files)
            if data is None
            else data
        )

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore, data: DataStore | None = None
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(directory, task_files=task_files, data=data)

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
            with closing(SQLiteScenarioDatabase(database)) as opened:
                return opened.read_settings()

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
            if self._directory is None:
                if key not in self._memory_anchors:
                    name = f"reef-{key}-{uuid.uuid4().hex}"
                    self._memory_anchors[key] = (name, SQLiteScenarioDatabase(memory_name=name))
                database = SQLiteScenarioDatabase(memory_name=self._memory_anchors[key][0])
                owner = self._memory_settings.get(scenario, ScenarioSettings()).owner or scenario
            else:
                database = SQLiteScenarioDatabase(self._directory / f"{key}.sqlite3")
            try:
                if self._directory is not None:
                    owner = database.read_settings().owner or scenario
                    if scenario not in self._checked:
                        if database.drop_copied_records(self.data, owner, scenario):
                            self._checked.add(scenario)
                        else:
                            logger.warning(
                                "the records of scenario %r are not all record items: its database keeps them in "
                                "the record tables. Run Reef 83a6f3ab once to copy them.",
                                scenario,
                            )
                records = ItemRecordStore(self.data, owner, scenario, self.data.scenario_generation(owner, scenario))
                return CommitHistoryScenarioStore(
                    scenario,
                    records,
                    database.commit_history(),
                    tasks=database.task_storage(self.task_files, scenario),
                    training_state=database.training_state(),
                    evaluations=database.evaluation_store(),
                    database=database,
                )
            except BaseException:
                database.close()
                raise

    def archive(self, scenario: str) -> tuple[str, ...]:
        with self._lock:
            self._ensure_open()
            self.data.remove_scenario(self.read_settings(scenario).owner or scenario, scenario)
            paths = self.state_paths(scenario)
            archived_files = self.task_files.archive(scenario)
            if self._directory is None:
                self._memory_settings.pop(scenario, None)
                anchor = self._memory_anchors.pop(self._scenario_key(scenario), None)
                if anchor is not None:
                    anchor[1].close()
                return archived_files
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            destination = self._directory / "archived" / f"{self._scenario_key(scenario)}-{stamp}-{uuid.uuid4().hex}"
            moved: list[str] = []
            database = paths[0]
            for path in paths:
                if not path.exists():
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                target = destination / path.name
                try:
                    shutil.move(str(path), str(target))
                except FileNotFoundError:
                    # SQLite removes a WAL file only after it folded its frames into the database, which moved
                    # first; a reader of another service that closes meanwhile makes it do so. Nothing is lost.
                    if path == database:
                        raise
                    continue
                moved.append(str(target))
            return (*moved, *archived_files)

    def state_paths(self, scenario: str) -> tuple[Path, ...]:
        """The local state paths of a scenario: its database and the WAL files of SQLite."""
        with self._lock:
            self._ensure_open()
            key = self._scenario_key(scenario)
            if self._directory is None:
                return ()
            return tuple(
                self._directory / name for name in (f"{key}.sqlite3", f"{key}.sqlite3-wal", f"{key}.sqlite3-shm")
            )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                for _, anchor in self._memory_anchors.values():
                    anchor.close()
                self._memory_anchors.clear()
                if self._owns_data:
                    # The data store shares the task file store, so it closes first.
                    self.data.close()
                self.task_files.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scenario storage is closed")

    @staticmethod
    def _write_database_settings(database: Path, settings: ScenarioSettings) -> None:
        """Write settings into a scenario database, and give the database and its WAL files the owner-only mode."""
        with closing(SQLiteScenarioDatabase(database)) as opened:
            opened.write_settings(settings)
        for path in (database, database.with_name(f"{database.name}-wal"), database.with_name(f"{database.name}-shm")):
            if path.exists():
                path.chmod(0o600)

    @staticmethod
    def _scenario_key(scenario: str) -> str:
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        return hashlib.sha256(scenario.encode("utf-8")).hexdigest()
