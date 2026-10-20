"""PostgreSQL records, commits, settings, generated tasks, pooled transactions, and scenario storage lifecycle.

Every per-scenario table is keyed by a store generation, so archiving a
scenario retires its records, commits, settings and tasks together. The database
alone holds a scenario's state; the local directory holds no scenario state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Any, Self

from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import URL, Connection, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.schema import CreateSchema

from reef.core.config import config_metadata, config_option
from reef.core.errors import ReefError
from reef.core.plugins import PluginSettings
from reef.storage.commit_log import (
    CommitHistory,
    CommitHistoryScenarioStore,
    CommitLog,
    same_commit,
    training_run_position,
)
from reef.storage.commits import CommitLogError, CommitRecord
from reef.storage.migrations import archive_imported_files, legacy_model_configs, migrate_record_storage
from reef.storage.records import RecordLoss, RecordRetention
from reef.storage.scenario import ScenarioSettings, ScenarioStorage
from reef.storage.sql_records import RecordTables, SQLRecordRetention, SQLRecordStore
from reef.storage.sql_tasks import SQLTaskStorage, task_tables
from reef.storage.tasks import LocalTaskFileStore, TaskFileStore

logger = logging.getLogger(__name__)


def postgres_url(database_url: str) -> URL:
    """Validate a PostgreSQL URL and select psycopg 3 without exposing credentials."""
    try:
        url = make_url(database_url)
        if url.drivername not in {"postgres", "postgresql", "postgresql+psycopg"}:
            raise ValueError
        # Force validation here; malformed ports otherwise fail during engine setup.
        if url.port is not None and not 0 < url.port < 65536:
            raise ValueError
    except (TypeError, ValueError, AttributeError, ArgumentError):
        raise ValueError("record_database_url must be a PostgreSQL URL using psycopg 3") from None
    return url.set(drivername="postgresql+psycopg")


def validate_postgres_schema(schema: str) -> None:
    """Require a dedicated, portable SQL schema name for deployment isolation."""
    if not isinstance(schema, str) or re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema) is None:
        raise ValueError("record_database_schema must be a lowercase SQL identifier of at most 63 characters")
    if schema == "public" or schema == "information_schema" or schema.startswith("pg_"):
        raise ValueError("record_database_schema must be a dedicated Reef schema")


def _record_tables(metadata: MetaData) -> RecordTables:
    records = Table(
        "agent_record",
        metadata,
        Column("sequence", BigInteger, Identity(), primary_key=True),
        Column("storage_id", String(32), ForeignKey("record_store.storage_id"), nullable=False),
        Column("agent_record_id", Text, nullable=False),
        Column("scenario", Text, nullable=False),
        Column("request_type", Text, nullable=False),
        Column("created_at", Float(53), nullable=False),
        Column("payload_json", Text, nullable=False),
        Column("references_json", Text, nullable=False),
        Column("artifact_json", Text),
        Column("body_bytes", BigInteger, nullable=False),
        UniqueConstraint("storage_id", "agent_record_id"),
    )
    consumed = Table(
        "consumed_agent_record",
        metadata,
        Column("storage_id", String(32), ForeignKey("record_store.storage_id"), primary_key=True),
        Column("agent_record_id", Text, primary_key=True),
        Column("content_sha256", String(64), nullable=False),
    )
    receipts = Table(
        "record_consumption",
        metadata,
        Column("storage_id", String(32), ForeignKey("record_store.storage_id"), primary_key=True),
        Column("scenario", Text, primary_key=True),
        Column("receipt_id", Text, primary_key=True),
        # Large id sets cannot be PostgreSQL btree keys. Shared SQL operations
        # still compare the complete canonical ids and metadata after insertion.
        Column("consumed_ids_sha256", String(64), primary_key=True),
        Column("consumed_ids_json", Text, nullable=False),
        Column("metadata_json", Text, nullable=False),
        Column("recorded_at", Float(53), nullable=False),
    )
    Index("agent_record_scenario_sequence", records.c.storage_id, records.c.scenario, records.c.sequence)
    Index(
        "agent_record_type_sequence",
        records.c.storage_id,
        records.c.scenario,
        records.c.request_type,
        records.c.sequence,
    )
    eviction = Table(
        "record_eviction",
        metadata,
        Column("storage_id", String(32), ForeignKey("record_store.storage_id"), primary_key=True),
        Column("scenario", Text, primary_key=True),
        Column("record_count", BigInteger, nullable=False),
        Column("body_bytes", BigInteger, nullable=False),
        Column("first_sequence", BigInteger, nullable=False),
        Column("last_sequence", BigInteger, nullable=False),
    )
    Index("agent_record_capacity", records.c.created_at, records.c.sequence)
    settings = Table(
        "scenario_setting",
        metadata,
        Column("storage_id", String(32), ForeignKey("record_store.storage_id"), primary_key=True),
        Column("name", Text, primary_key=True),
        Column("value_json", Text, nullable=False),
    )
    return RecordTables(records, consumed, receipts, eviction, settings)


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
        self.tables = _record_tables(metadata)
        self.task_tables = task_tables(metadata, foreign_key="record_store.storage_id")
        #: One row per committed step, as the JSON object a commit log line holds.
        self.commits = Table(
            "scenario_commit",
            metadata,
            Column("storage_id", String(32), ForeignKey("record_store.storage_id"), primary_key=True),
            Column("step", BigInteger, primary_key=True),
            Column("commit_json", Text, nullable=False),
        )
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
                if versions and versions not in ([1], [2], [3], [4], [5]):
                    raise ReefError("unsupported PostgreSQL record schema version")
                # Version 4 adds the commit and settings tables. Version 5 adds the task tables.
                # create_all adds them to an older schema.
                metadata.create_all(connection)
                migrate_record_storage(connection, self.tables.records, self.tables.consumption)
                for index in self.tables.records.indexes:
                    index.create(connection, checkfirst=True)
                if versions and versions != [5]:
                    connection.execute(version.update().values(version=5))
                elif not versions:
                    connection.execute(version.insert().values(version=5))
        except BaseException:
            self._engine.dispose()
            raise

    @property
    def schema(self) -> str:
        return self._schema

    def active_stores(self) -> tuple[tuple[str, str], ...]:
        """``(storage_id, name)`` of every store generation that is not archived."""
        with self.transaction(read_only=True) as connection:
            rows = connection.execute(
                select(self._stores.c.storage_id, self._stores.c.name).where(self._stores.c.active_key.is_not(None))
            ).all()
        return tuple((str(storage_id), str(name)) for storage_id, name in rows)

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

    def prune(self, retention: RecordRetention) -> int:
        """Evict oldest bodies across generations when the capacity budget is exceeded."""
        queries = SQLRecordRetention(self.tables)
        losses: list[tuple[str, RecordLoss]] = []
        removed = 0
        with self.transaction() as connection:
            # Concurrent maintenance must not count the same deletion twice.
            key = int.from_bytes(
                hashlib.sha256(f"reef-record-retention:{self._schema}".encode()).digest()[:8], "big", signed=True
            )
            connection.execute(select(func.pg_advisory_xact_lock(key)))
            # Use the same namespace locks as append/eviction to protect retry hashes.
            connection.execute(
                select(self._stores.c.storage_id).order_by(self._stores.c.storage_id).with_for_update()
            ).all()
            retained = queries.retained_bytes(connection)
            after_time, after_sequence = float("-inf"), 0
            while retained > retention.max_bytes:
                page = queries.page(connection, after_time=after_time, after_sequence=after_sequence)
                if not page:
                    break
                selected: list[int] = []
                for created_at, sequence, size in page:
                    selected.append(sequence)
                    retained -= size
                    after_time, after_sequence = created_at, sequence
                    if retained <= retention.max_bytes:
                        break
                batch_losses = queries.evict(connection, selected)
                losses.extend(batch_losses)
                removed += sum(loss.record_count for _, loss in batch_losses)
        for scenario, loss in losses:
            logging.getLogger(__name__).warning(
                "Record capacity exceeded: evicted %d records (%d body bytes), scenario=%s, sequence=%d..%d; "
                "training data may be incomplete",
                loss.record_count,
                loss.body_bytes,
                scenario,
                loss.first_sequence,
                loss.last_sequence,
            )
        return removed

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._engine.dispose()


class PostgresRecordStore(SQLRecordStore):
    """Reuse SQL record operations with PostgreSQL transactions and conflict insertion.

    A URL creates an owned pool; a PostgresRecordDatabase shares its pool with
    other sessions. ``name`` identifies the store across restarts, and ``schema``
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
            super().__init__(replace(self._database.tables, scope={"storage_id": self.storage_id}))
        except BaseException:
            if self._owns_database:
                self._database.close()
            raise

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("PostgreSQL record store is closed")
            with self._database.transaction(read_only=not write) as connection:
                self._database.check_store(connection, self.storage_id, write=write)
                yield connection

    def _insert_if_absent(
        self, connection: Connection, table: Table, values: Mapping[str, object] | Sequence[Mapping[str, object]]
    ) -> bool:
        rows = [dict(values)] if isinstance(values, Mapping) else [dict(row) for row in values]
        statement = insert(table).values(rows).on_conflict_do_nothing().returning(next(iter(table.primary_key)))
        return connection.execute(statement).first() is not None

    def task_storage(self, files: TaskFileStore, scenario: str) -> SQLTaskStorage:
        """Open the task stores of this store generation, with their task files in ``files``."""
        tables = replace(self._database.task_tables, scope={"storage_id": self.storage_id})
        return SQLTaskStorage(self, tables, files, scenario)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._live_records.clear()
                if self._owns_database:
                    self._database.close()


class PostgresCommitTable(CommitHistory):
    """The commits of one store generation, in the ``scenario_commit`` table.

    ``locked`` holds an advisory lock in one transaction. The reads and appends
    of the locking thread use that transaction, so an append is durable when
    the locked block ends. Outside a locked block, each call uses its own
    transaction. Commits are append-only, so a read gets only the steps after
    the cached ones.
    """

    def __init__(self, database: PostgresRecordDatabase, storage_id: str) -> None:
        self._database = database
        self._storage_id = storage_id
        self._lock = RLock()
        self._connection: Connection | None = None
        self._cached: tuple[CommitRecord, ...] = ()
        self._lock_key = int.from_bytes(
            hashlib.sha256(f"reef-commit:{database.schema}:{storage_id}".encode()).digest()[:8], "big", signed=True
        )

    @property
    def label(self) -> str:
        return f"postgres://{self._database.schema}/record-store/{self._storage_id}/commits"

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._lock:
            try:
                with self._database.transaction() as connection:
                    connection.execute(select(func.pg_advisory_xact_lock(self._lock_key)))
                    self._connection = connection
                    try:
                        yield
                    finally:
                        self._connection = None
            except BaseException:
                # The transaction rolled back, so the cache may hold a step that was never stored.
                self._cached = ()
                raise

    def records(self) -> tuple[CommitRecord, ...]:
        with self._lock, self._transaction(write=False) as connection:
            return self._read(connection)

    def append(self, record: CommitRecord) -> None:
        with self._lock, self._transaction(write=True) as connection:
            self._database.check_store(connection, self._storage_id, write=True)
            records = self._read(connection)
            existing = next((item for item in records if item.step == record.step), None)
            if existing is not None:
                if same_commit(existing, record):
                    return
                raise CommitLogError(f"commit step {record.step} conflicts with its existing record")
            if records and record.step < records[-1].step:
                raise CommitLogError(f"commit step {record.step} precedes existing step {records[-1].step}")
            encoded = json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            connection.execute(
                self._database.commits.insert().values(
                    storage_id=self._storage_id, step=record.step, commit_json=encoded
                )
            )

    def training_run_position(self) -> tuple[int, int]:
        return training_run_position(self.records())

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[Connection]:
        if self._connection is not None:
            yield self._connection
            return
        with self._database.transaction(read_only=not write) as connection:
            yield connection

    def _read(self, connection: Connection) -> tuple[CommitRecord, ...]:
        commits = self._database.commits
        last_step = self._cached[-1].step if self._cached else 0
        rows = connection.execute(
            select(commits.c.commit_json)
            .where(commits.c.storage_id == self._storage_id, commits.c.step > last_step)
            .order_by(commits.c.step)
        ).scalars()
        added = tuple(CommitRecord.from_dict(json.loads(row)) for row in rows)
        if added:
            self._cached = (*self._cached, *added)
        return self._cached


@dataclass(frozen=True)
class PostgresStorageSettings(PluginSettings):
    """The config of :class:`PostgresScenarioStorage` when a deployment names it as a plugin."""

    database_url: str = field(repr=False, metadata=config_metadata("PostgreSQL connection URL."))
    schema: str = config_option("reef_records", help="Dedicated SQL schema of this deployment.")


class PostgresScenarioStorage(ScenarioStorage):
    """Keep every scenario's records, commits, settings and tasks in one PostgreSQL schema.

    On start, it imports the commit logs and model configuration files that
    older versions kept under ``directory`` and moves them under
    ``directory/archived``. After that it keeps no scenario state in files.
    ``task_files`` keeps the files of published tasks. The default is a
    ``LocalTaskFileStore`` in ``directory/task-files``.
    """

    def __init__(
        self,
        database_url: str,
        directory: Path,
        *,
        schema: str = "reef_records",
        task_files: TaskFileStore | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._database = PostgresRecordDatabase(database_url, schema=schema)
        self._schema = schema
        self._lock = RLock()
        self._closed = False
        self.task_files = LocalTaskFileStore(self._directory / "task-files") if task_files is None else task_files
        try:
            self._import_local_files()
        except BaseException:
            self.close()
            raise

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore) -> Self:
        settings = PostgresStorageSettings.parse(config, plugin_class=cls)
        return cls(settings.database_url, directory, schema=settings.schema, task_files=task_files)

    @property
    def durable(self) -> bool:
        return True

    def open(self, scenario: str) -> CommitHistoryScenarioStore:
        with self._lock:
            self._ensure_open()
            records = PostgresRecordStore(self._database, name=scenario)
            try:
                commits = PostgresCommitTable(self._database, records.storage_id)
                return CommitHistoryScenarioStore(
                    scenario, records, commits, tasks=records.task_storage(self.task_files, scenario)
                )
            except BaseException:
                records.close()
                raise

    def read_settings(self, scenario: str) -> ScenarioSettings:
        with self._lock:
            self._ensure_open()
            if self._database.active_storage_id(scenario) is None:
                return ScenarioSettings()
            with closing(PostgresRecordStore(self._database, name=scenario)) as records:
                return records.read_settings()

    def write_settings(self, scenario: str, settings: ScenarioSettings) -> None:
        with self._lock:
            self._ensure_open()
            with closing(PostgresRecordStore(self._database, name=scenario)) as records:
                records.write_settings(settings)

    def archive(self, scenario: str) -> tuple[str, ...]:
        with self._lock:
            self._ensure_open()
            storage_id = self._database.archive(scenario)
            archived_files = self.task_files.archive(scenario)
            if storage_id is None:
                return archived_files
            return (f"postgres://{self._schema}/record-store/{storage_id}", *archived_files)

    def prune(self, *, max_bytes: int) -> int:
        with self._lock:
            self._ensure_open()
            return self._database.prune(RecordRetention(max_bytes=max_bytes))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._database.close()
                finally:
                    self.task_files.close()

    def _import_local_files(self) -> None:
        """Import the commit logs and model configuration files of older versions, then move them aside.

        Only files of this schema's active store generations are imported;
        files of another schema that shares the directory stay in place.
        """
        model_configs = legacy_model_configs(self._directory)
        imported = []
        for storage_id, name in self._database.active_stores():
            log_path = self._directory / f"postgres-{storage_id}.commits.jsonl"
            if log_path.exists():
                commits = PostgresCommitTable(self._database, storage_id)
                for record in CommitLog(log_path).records():
                    commits.append(record)
                imported.append(log_path)
            model_config = model_configs.get(hashlib.sha256(name.encode("utf-8")).hexdigest())
            if model_config is not None:
                value, model_path = model_config
                # Older versions stored no other setting, so the import replaces all settings.
                with closing(PostgresRecordStore(self._database, name=name)) as records:
                    records.write_settings(ScenarioSettings(model_config=value))
                imported.append(model_path)
        for location in archive_imported_files(self._directory, imported):
            logger.info("Imported a scenario file into PostgreSQL; the file moved to %s", location)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scenario storage is closed")
