"""PostgreSQL commits, settings, samples, tasks, evaluations, pooled transactions, and storage lifecycle.

Every per-scenario table is keyed by a store generation, so archiving a scenario retires its commits, settings,
samples and tasks together. Its records are record items in the storage's data store (RFC #1219). The database
alone holds a scenario's state; the local directory holds no scenario state.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any, Self

from sqlalchemy import Column, Float, ForeignKey, Integer, MetaData, String, Table, Text, create_engine, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection
from sqlalchemy.schema import CreateSchema

from reef.core.config import config_metadata, config_option
from reef.core.errors import ReefError
from reef.core.plugins import PluginSettings
from reef.storage.commit_history import CommitHistoryScenarioStore
from reef.storage.data import DataStore
from reef.storage.record_items import ItemRecordStore
from reef.storage.scenario import ScenarioSettings, ScenarioStorage
from reef.storage.sql_commits import SQLCommitHistory, commit_table
from reef.storage.sql_data import PostgresDataStore
from reef.storage.sql_evaluations import SQLEvaluationStore, evaluation_table
from reef.storage.sql_records import ScopedTable, SQLScenarioDatabase, batches
from reef.storage.sql_samples import SQLTrainingState, stage_record_table
from reef.storage.sql_setup import postgres_url, validate_postgres_schema
from reef.storage.sql_tasks import SQLTaskStorage, task_tables
from reef.storage.tasks import LocalTaskFileStore, TaskFileStore

#: The most parameters that one PostgreSQL statement takes.
MAX_PARAMETERS = 65535

#: The version of the record schema that this adapter writes and reads; it opens no other version.
POSTGRES_SCHEMA_VERSION = 23


class PostgresRecordDatabase:
    """Own a connection pool and shared tables in one deployment's SQL schema.

    Store generations isolate archived data from a newly created scenario with
    the same name. Sessions share this pool and return connections after each
    operation. Closing the database does not delete data.
    """

    def __init__(self, database_url: str, *, schema: str = "reef_records") -> None:
        validate_postgres_schema(schema)
        self._closed = False
        self._schema = schema
        metadata = MetaData(schema=schema)
        self._stores = Table(
            "record_store",
            metadata,
            Column("storage_id", String(32), primary_key=True),
            Column("active_key", String(64), unique=True),
            Column("name", Text, nullable=False),
            Column("archived_at", Float(53)),
        )
        version = Table("schema_version", metadata, Column("version", Integer, primary_key=True))
        self.settings_table = Table(
            "scenario_setting",
            metadata,
            Column("storage_id", String(32), ForeignKey("record_store.storage_id"), primary_key=True),
            Column("name", Text, primary_key=True),
            Column("value_json", Text, nullable=False),
        )
        self.task_tables = task_tables(metadata, foreign_key="record_store.storage_id")
        self.stage_record_table = stage_record_table(metadata, foreign_key="record_store.storage_id")
        self.evaluation_table = evaluation_table(metadata, foreign_key="record_store.storage_id")
        self.commit_table = commit_table(metadata, foreign_key="record_store.storage_id")
        try:
            self._engine = create_engine(postgres_url(database_url), pool_pre_ping=True, hide_parameters=True)
        except ModuleNotFoundError:
            raise ReefError("PostgreSQL records require the reef-infra[postgres] extra") from None
        try:
            with self._engine.begin() as connection:
                # DDL check-then-create also needs serialization across service starts.
                key = int.from_bytes(
                    hashlib.sha256(f"reef-record-schema:{schema}".encode()).digest()[:8], "big", signed=True
                )
                connection.execute(select(func.pg_advisory_xact_lock(key)))
                connection.execute(CreateSchema(schema, if_not_exists=True))
                version.create(connection, checkfirst=True)
                versions: list[int] = list(connection.execute(select(version.c.version)).scalars().all())
                if versions and versions != [POSTGRES_SCHEMA_VERSION]:
                    raise ReefError("unsupported PostgreSQL record schema version")
                metadata.create_all(connection)
                if not versions:
                    connection.execute(version.insert().values(version=POSTGRES_SCHEMA_VERSION))
        except BaseException:
            self._engine.dispose()
            raise

    @property
    def schema(self) -> str:
        return self._schema

    def active_storage_id(self, name: str) -> str | None:
        """The active generation of a store name, or ``None`` when the name has none."""
        key = hashlib.sha256(name.encode("utf-8")).hexdigest()
        with self.transaction(read_only=True) as connection:
            storage_id = connection.execute(
                select(self._stores.c.storage_id).where(self._stores.c.active_key == key)
            ).scalar_one_or_none()
        return None if storage_id is None else str(storage_id)

    def storage_id(self, name: str) -> str:
        """Open or create a named store generation, safely across connections."""
        if not isinstance(name, str) or not name:
            raise ValueError("record store name must be a non-empty string")
        key = hashlib.sha256(name.encode("utf-8")).hexdigest()
        with self.transaction() as connection:
            connection.execute(
                insert(self._stores)
                .values(storage_id=uuid.uuid4().hex, active_key=key, name=name)
                .on_conflict_do_nothing(index_elements=[self._stores.c.active_key])
            )
            row = connection.execute(select(self._stores).where(self._stores.c.active_key == key)).mappings().one()
            if row["name"] != name:
                raise ReefError("record store name hash collision")
            return str(row["storage_id"])

    @contextmanager
    def transaction(self, *, read_only: bool = False) -> Iterator[Connection]:
        """Borrow a connection; readers get a consistent multi-query snapshot."""
        if self._closed:
            raise RuntimeError("PostgreSQL record database is closed")
        isolation = "REPEATABLE READ" if read_only else "READ COMMITTED"
        with (
            self._engine.connect().execution_options(isolation_level=isolation) as connection,
            connection.begin(),
        ):
            yield connection

    def check_store(self, connection: Connection, storage_id: str, *, write: bool) -> None:
        """Serialize writers before sequence allocation and reject archived sessions."""
        statement = select(self._stores.c.active_key).where(self._stores.c.storage_id == storage_id)
        if write:
            statement = statement.with_for_update()
        if connection.execute(statement).scalar_one_or_none() is None:
            raise RuntimeError("PostgreSQL record store has been archived")

    def archive(self, name: str) -> str | None:
        """Retire a generation while preserving its records and retry receipts."""
        key = hashlib.sha256(name.encode("utf-8")).hexdigest()
        with self.transaction() as connection:
            storage_id = connection.execute(
                select(self._stores.c.storage_id).where(self._stores.c.active_key == key).with_for_update()
            ).scalar_one_or_none()
            if storage_id is not None:
                connection.execute(
                    self._stores.update()
                    .where(self._stores.c.storage_id == storage_id)
                    .values(active_key=None, archived_at=time.time())
                )
            return storage_id

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._engine.dispose()


class PostgresScenarioDatabase(SQLScenarioDatabase):
    """The state of one scenario generation in the PostgreSQL record database, with its transactions and conflict
    insertion.

    A URL creates an owned pool; a PostgresRecordDatabase shares its pool with
    other sessions. ``name`` identifies the scenario across restarts, and ``schema``
    isolates deployments when constructing from a URL.
    """

    def __init__(
        self, database: str | PostgresRecordDatabase, *, name: str = "default", schema: str = "reef_records"
    ) -> None:
        self._owns_database = isinstance(database, str)
        self._database = PostgresRecordDatabase(database, schema=schema) if isinstance(database, str) else database
        self._lock = RLock()
        self._closed = False
        try:
            self.storage_id = self._database.storage_id(name)
            super().__init__(self._database.settings_table, {"storage_id": self.storage_id})
        except BaseException:
            if self._owns_database:
                self._database.close()
            raise

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("PostgreSQL scenario database is closed")
            with self._database.transaction(read_only=not write) as connection:
                self._database.check_store(connection, self.storage_id, write=write)
                yield connection

    def _write_if_absent(
        self, connection: Connection, table: Table, values: Mapping[str, object] | Sequence[Mapping[str, object]]
    ) -> bool:
        rows = [dict(values)] if isinstance(values, Mapping) else [dict(row) for row in values]
        inserted = False
        # One statement takes at most MAX_PARAMETERS parameters, and each row gives one for each of its columns.
        for batch in batches(rows, MAX_PARAMETERS // len(rows[0]) if rows else 1):
            statement = insert(table).values(batch).on_conflict_do_nothing().returning(next(iter(table.primary_key)))
            if connection.execute(statement).first() is not None:
                inserted = True
        return inserted

    def task_storage(self, files: TaskFileStore, scenario: str) -> SQLTaskStorage:
        """Open the task stores of this store generation, with their task files in ``files``."""
        tables = replace(self._database.task_tables, scope={"storage_id": self.storage_id})
        return SQLTaskStorage(self, tables, files, scenario)

    def training_state(self) -> SQLTrainingState:
        """Open the training state of this store generation."""
        scope = {"storage_id": self.storage_id}
        return SQLTrainingState(
            self,
            ScopedTable(self._database.commit_table, scope=scope),
            ScopedTable(self._database.stage_record_table, scope=scope),
        )

    def evaluation_store(self) -> SQLEvaluationStore:
        """Open the artifact evaluations of this store generation."""
        return SQLEvaluationStore(
            self, ScopedTable(self._database.evaluation_table, scope={"storage_id": self.storage_id})
        )

    def commit_history(self) -> SQLCommitHistory:
        """Open the commit history of this store generation."""
        return SQLCommitHistory(
            self,
            ScopedTable(self._database.commit_table, scope={"storage_id": self.storage_id}),
            label=f"postgres://{self._database.schema}/record-store/{self.storage_id}/commits",
            durable=True,
        )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._owns_database:
                    self._database.close()


@dataclass(frozen=True)
class PostgresStorageSettings(PluginSettings):
    """The config of :class:`PostgresScenarioStorage` when a deployment names it as a plugin."""

    database_url: str = field(repr=False, metadata=config_metadata("PostgreSQL connection URL."))
    schema: str = config_option("reef_records", help="Dedicated SQL schema of this deployment.")


class PostgresScenarioStorage(ScenarioStorage):
    """Keep every scenario's commits, settings and tasks in one PostgreSQL schema, and its records as record items in
    ``data`` (``ItemRecordStore``).

    It keeps no scenario state in files under ``directory``. ``task_files`` keeps the files of published tasks. The
    default is a ``LocalTaskFileStore`` in ``directory/task-files``. ``data`` is the data store. The default is a
    ``PostgresDataStore`` in the schema ``<schema>_data`` of the same database, so that two record schemas do not
    share records; the storage closes a default data store with itself.
    Deployment assembly gives its data store.
    """

    def __init__(
        self,
        database_url: str,
        directory: Path,
        *,
        schema: str = "reef_records",
        task_files: TaskFileStore | None = None,
        data: DataStore | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._database = PostgresRecordDatabase(database_url, schema=schema)
        self._schema = schema
        self._lock = RLock()
        self._closed = False
        self.task_files = LocalTaskFileStore(self._directory / "task-files") if task_files is None else task_files
        self._owns_data = data is None
        try:
            self.data = (
                PostgresDataStore(database_url, self.task_files, schema=f"{schema}_data") if data is None else data
            )
        except BaseException:
            self._database.close()
            raise

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore, data: DataStore | None = None
    ) -> Self:
        settings = PostgresStorageSettings.parse(config, plugin_class=cls)
        return cls(settings.database_url, directory, schema=settings.schema, task_files=task_files, data=data)

    @property
    def durable(self) -> bool:
        return True

    def open(self, scenario: str) -> CommitHistoryScenarioStore:
        with self._lock:
            self._ensure_open()
            database = PostgresScenarioDatabase(self._database, name=scenario)
            try:
                owner = database.read_settings().owner or scenario
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

    def read_settings(self, scenario: str) -> ScenarioSettings:
        with self._lock:
            self._ensure_open()
            if self._database.active_storage_id(scenario) is None:
                return ScenarioSettings()
            with closing(PostgresScenarioDatabase(self._database, name=scenario)) as database:
                return database.read_settings()

    def write_settings(self, scenario: str, settings: ScenarioSettings) -> None:
        with self._lock:
            self._ensure_open()
            with closing(PostgresScenarioDatabase(self._database, name=scenario)) as database:
                database.write_settings(settings)

    def archive(self, scenario: str) -> tuple[str, ...]:
        with self._lock:
            self._ensure_open()
            self.data.delete_scenario(self.read_settings(scenario).owner or scenario, scenario)
            storage_id = self._database.archive(scenario)
            archived_files = self.task_files.archive(scenario)
            if storage_id is None:
                return archived_files
            return (f"postgres://{self._schema}/record-store/{storage_id}", *archived_files)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._database.close()
                    if self._owns_data:
                        # The data store shares the task file store, so it closes first.
                        self.data.close()
                finally:
                    self.task_files.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scenario storage is closed")
