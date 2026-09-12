"""PostgreSQL records, pooled transactions, and scenario storage lifecycle."""

from __future__ import annotations

import hashlib
import re
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import RLock

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

from reef.core.errors import ReefError
from reef.storage.commit_log import CommitLog, CommitLogScenarioStore
from reef.storage.records import RecordRetention
from reef.storage.scenario import ScenarioStorage
from reef.storage.sql_records import RecordTables, SQLRecordRetention, SQLRecordStore


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
        Column("compacted_at", Float(53)),
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
        "compaction_receipts",
        metadata,
        Column("storage_id", String(32), ForeignKey("record_store.storage_id"), primary_key=True),
        Column("scenario", Text, primary_key=True),
        Column("receipt_id", Text, primary_key=True),
        # Large id sets cannot be PostgreSQL btree keys. Shared SQL operations
        # still compare the complete canonical ids and metadata after insertion.
        Column("compacted_ids_sha256", String(64), primary_key=True),
        Column("compacted_ids_json", Text, nullable=False),
        Column("metadata_json", Text, nullable=False),
        Column("recorded_at", Float(53), nullable=False),
    )
    Index("agent_record_scenario_sequence", records.c.storage_id, records.c.scenario, records.c.sequence)
    Index(
        "agent_record_active_type_sequence",
        records.c.storage_id,
        records.c.scenario,
        records.c.request_type,
        records.c.sequence,
        postgresql_where=records.c.compacted_at.is_(None),
    )
    Index(
        "agent_record_retention",
        records.c.compacted_at,
        records.c.sequence,
        postgresql_where=records.c.compacted_at.is_not(None),
    )
    return RecordTables(records, consumed, receipts)


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
                versions = connection.execute(select(version.c.version)).scalars().all()
                if versions and versions != [1]:
                    raise ReefError("unsupported PostgreSQL record schema version")
                metadata.create_all(connection)
                if not versions:
                    connection.execute(version.insert().values(version=1))
        except BaseException:
            self._engine.dispose()
            raise

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
        """Apply age and body-byte limits across active and archived generations."""
        queries = SQLRecordRetention(self.tables)
        cutoff = time.time() - retention.days * 86400
        removed = 0
        with self.transaction() as connection:
            # Concurrent maintenance must not count the same deletion twice.
            key = int.from_bytes(
                hashlib.sha256(f"reef-record-retention:{self._schema}".encode()).digest()[:8], "big", signed=True
            )
            connection.execute(select(func.pg_advisory_xact_lock(key)))
            while True:
                count = queries.purge_expired(connection, before=cutoff)
                removed += count
                if count == 0:
                    break
            retained = queries.retained_bytes(connection)
            after_time, after_sequence = float("-inf"), 0
            while retained > retention.max_bytes:
                page = queries.page(connection, after_time=after_time, after_sequence=after_sequence)
                if not page:
                    break
                selected: list[int] = []
                for compacted_at, sequence, size in page:
                    selected.append(sequence)
                    retained -= size
                    after_time, after_sequence = compacted_at, sequence
                    if retained <= retention.max_bytes:
                        break
                removed += queries.delete(connection, selected)
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
    def _transaction(self, scenario: str, *, write: bool) -> Iterator[Connection]:
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
        if table is self._tables.compaction_receipts:
            for row in rows:
                row["compacted_ids_sha256"] = hashlib.sha256(
                    str(row["compacted_ids_json"]).encode("utf-8")
                ).hexdigest()
        statement = insert(table).values(rows).on_conflict_do_nothing().returning(next(iter(table.primary_key)))
        return connection.execute(statement).first() is not None

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._live_records.clear()
                if self._owns_database:
                    self._database.close()


class PostgresScenarioStorage(ScenarioStorage):
    """Combine pooled PostgreSQL records with generation-specific local commit logs."""

    def __init__(self, database_url: str, directory: Path, *, schema: str = "reef_records") -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._database = PostgresRecordDatabase(database_url, schema=schema)
        self._schema = schema
        self._lock = RLock()
        self._closed = False

    @property
    def durable(self) -> bool:
        return True

    def open(self, scenario: str) -> CommitLogScenarioStore:
        with self._lock:
            self._ensure_open()
            records = PostgresRecordStore(self._database, name=scenario)
            try:
                commit_log = CommitLog(self._log_path(records.storage_id))
                return CommitLogScenarioStore(scenario, records, commit_log)
            except BaseException:
                records.close()
                raise

    def archive(self, scenario: str) -> tuple[str, ...]:
        with self._lock:
            self._ensure_open()
            storage_id = self._database.archive(scenario)
            if storage_id is None:
                return ()
            archived = [f"postgres://{self._schema}/record-store/{storage_id}"]
            # The next generation always gets another log name. If moving the
            # old log fails, reopening cannot apply it to the new empty store.
            path = self._log_path(storage_id)
            if path.exists():
                destination = self._directory / "archived" / f"postgres-{storage_id}"
                destination.mkdir(parents=True, exist_ok=True)
                target = destination / path.name
                shutil.move(str(path), str(target))
                archived.append(str(target))
            return tuple(archived)

    def prune(self, *, days: float, max_bytes: int) -> int:
        with self._lock:
            self._ensure_open()
            return self._database.prune(RecordRetention(days, max_bytes))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._database.close()

    def _log_path(self, storage_id: str) -> Path:
        return self._directory / f"postgres-{storage_id}.commits.jsonl"

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scenario storage is closed")
