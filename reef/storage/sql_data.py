"""The data store (RFC #1028) in a SQL database, for SQLite and PostgreSQL.

``data_item`` holds one row per item: its owner, ID, kind, digest, metadata as JSON, and the JSON content of a
trajectory or a record. ``data_item_scenario`` repeats the ``scenarios`` metadata, one row per scenario that uses
an item, so that a query by scenario uses an index. Both are written in one transaction. Task directories are in
the task file store, in each owner's partition.
"""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Self

from sqlalchemy import (
    URL,
    BigInteger,
    Column,
    Identity,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    exists,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Connection, Engine, RowMapping
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema

from reef.core.config import config_metadata, config_option
from reef.core.errors import ReefError
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.storage.data import (
    MAX_QUERY_LIMIT,
    DataStore,
    ItemConflict,
    ItemKind,
    ItemNotFound,
    NewItem,
    StoredItem,
    check_owner,
    check_query,
    patched_metadata,
    prepared_items,
    task_partition,
)
from reef.storage.postgres import postgres_url, validate_postgres_schema
from reef.storage.tasks import TaskFileStore


def data_tables(metadata: MetaData, *, sqlite: bool) -> tuple[Table, Table]:
    """Define ``data_item`` and ``data_item_scenario`` in ``metadata``.

    ``sequence`` orders the items as they arrived. ``content_json`` is the canonical JSON of a trajectory's or a
    record's content, and NULL for a task. ``metadata_json`` is the item's metadata.
    """
    sequence = (
        Column("sequence", Integer, primary_key=True)
        if sqlite
        else Column("sequence", BigInteger, Identity(), primary_key=True)
    )
    items = Table(
        "data_item",
        metadata,
        sequence,
        Column("owner", Text, nullable=False),
        Column("item_id", Text, nullable=False),
        Column("kind", String(16), nullable=False),
        Column("digest", String(64), nullable=False),
        Column("metadata_json", Text, nullable=False),
        Column("content_json", Text, nullable=True),
        UniqueConstraint("owner", "item_id"),
        # SQLite reuses the largest rowid after a delete without it; other dialects ignore the option.
        sqlite_autoincrement=sqlite,
    )
    scenarios = Table(
        "data_item_scenario",
        metadata,
        Column("owner", Text, nullable=False),
        Column("item_id", Text, nullable=False),
        Column("scenario", Text, nullable=False),
        PrimaryKeyConstraint("owner", "item_id", "scenario"),
        Index("data_item_scenario_by_scenario", "owner", "scenario"),
    )
    return items, scenarios


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class SQLDataStore(DataStore):
    """A data store in the tables of ``data_tables``.

    A subclass gives the tables and ``transaction``, one database transaction on one connection. Every operation
    runs in one transaction, so an operation that refuses changes nothing in the database.
    """

    def __init__(self, items: Table, scenarios: Table, task_files: TaskFileStore) -> None:
        self.items_table = items
        self.scenarios_table = scenarios
        self.task_files = task_files

    @abstractmethod
    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """One transaction: commit when the block ends, roll back when it raises."""

    def stored_item(self, row: RowMapping) -> StoredItem:
        content = None if row["content_json"] is None else json.loads(row["content_json"])
        return StoredItem(
            row["sequence"], row["owner"], row["item_id"], json.loads(row["metadata_json"]), row["digest"], content
        )

    def rows(self, connection: Connection, owner: str, item_ids: Sequence[str]) -> dict[str, RowMapping]:
        found = connection.execute(
            select(self.items_table).where(
                self.items_table.c.owner == owner, self.items_table.c.item_id.in_(list(item_ids))
            )
        ).mappings()
        return {row["item_id"]: row for row in found}

    def write_scenarios(self, connection: Connection, owner: str, item_id: str, scenarios: Mapping[str, str]) -> None:
        connection.execute(
            delete(self.scenarios_table).where(
                self.scenarios_table.c.owner == owner, self.scenarios_table.c.item_id == item_id
            )
        )
        if scenarios:
            connection.execute(
                insert(self.scenarios_table),
                [{"owner": owner, "item_id": item_id, "scenario": scenario} for scenario in scenarios],
            )

    def create(self, owner: str, items: Sequence[NewItem]) -> tuple[StoredItem, ...]:
        prepared = prepared_items(owner, items, self.task_files)
        if not prepared:
            return ()
        with self.transaction() as connection:
            existing = self.rows(connection, owner, [entry[0].item_id for entry in prepared])
            for entry in prepared:
                row = existing.get(entry[0].item_id)
                if row is not None and row["digest"] != entry[2]:
                    raise ItemConflict(f"item {entry[0].item_id!r} of {owner!r} is stored with other content")
            for item, metadata, digest in prepared:
                if item.item_id in existing:
                    continue
                connection.execute(
                    insert(self.items_table).values(
                        owner=owner,
                        item_id=item.item_id,
                        kind=metadata["kind"],
                        digest=digest,
                        metadata_json=canonical_json(metadata),
                        content_json=canonical_json(item.content) if isinstance(item.content, Mapping) else None,
                    )
                )
                self.write_scenarios(connection, owner, item.item_id, metadata["scenarios"])
            stored = self.rows(connection, owner, [entry[0].item_id for entry in prepared])
        return tuple(self.stored_item(stored[entry[0].item_id]) for entry in prepared)

    def read(self, owner: str, item_id: str) -> StoredItem | None:
        check_owner(owner)
        with self.transaction() as connection:
            row = self.rows(connection, owner, [item_id]).get(item_id)
        return None if row is None else self.stored_item(row)

    def query(
        self,
        owner: str,
        *,
        kind: ItemKind | None = None,
        scenario: str | None = None,
        after: int = 0,
        limit: int = MAX_QUERY_LIMIT,
    ) -> tuple[StoredItem, ...]:
        check_query(owner, kind, scenario, after, limit)
        items = self.items_table
        statement = select(items).where(items.c.owner == owner, items.c.sequence > after)
        if kind is not None:
            statement = statement.where(items.c.kind == kind)
        if scenario is not None:
            uses = self.scenarios_table
            statement = statement.where(
                exists().where(uses.c.owner == owner, uses.c.item_id == items.c.item_id, uses.c.scenario == scenario)
            )
        with self.transaction() as connection:
            found = connection.execute(statement.order_by(items.c.sequence).limit(limit)).mappings().all()
        return tuple(self.stored_item(row) for row in found)

    def update(self, owner: str, item_ids: Sequence[str], patch: Mapping[str, Any]) -> tuple[StoredItem, ...]:
        check_owner(owner)
        unique_ids = list(dict.fromkeys(item_ids))
        with self.transaction() as connection:
            existing = self.rows(connection, owner, unique_ids)
            missing = [item_id for item_id in unique_ids if item_id not in existing]
            if missing:
                raise ItemNotFound(f"{owner!r} stores no item {missing[0]!r}")
            patched = {
                item_id: patched_metadata(json.loads(existing[item_id]["metadata_json"]), patch)
                for item_id in unique_ids
            }
            for item_id, metadata in patched.items():
                connection.execute(
                    update(self.items_table)
                    .where(self.items_table.c.owner == owner, self.items_table.c.item_id == item_id)
                    .values(metadata_json=canonical_json(metadata))
                )
                self.write_scenarios(connection, owner, item_id, metadata["scenarios"])
            stored = self.rows(connection, owner, unique_ids)
        return tuple(self.stored_item(stored[item_id]) for item_id in unique_ids)

    def delete(self, owner: str, item_id: str) -> bool:
        check_owner(owner)
        with self.transaction() as connection:
            connection.execute(
                delete(self.scenarios_table).where(
                    self.scenarios_table.c.owner == owner, self.scenarios_table.c.item_id == item_id
                )
            )
            removed = connection.execute(
                delete(self.items_table).where(
                    self.items_table.c.owner == owner, self.items_table.c.item_id == item_id
                )
            )
        return removed.rowcount > 0

    def task_path(self, item: StoredItem) -> Path:
        if item.kind != "task":
            raise ValueError(f"item {item.item_id!r} is a {item.kind}, not a task")
        return self.task_files.path(task_partition(item.owner), item.digest)


class SQLiteDataStore(SQLDataStore):
    """The data store in a SQLite file, ``<directory>/data-store.sqlite3``, or in memory without a directory.

    It holds one connection for its lifetime and serializes its use with a lock, as the SQLite record store does.
    """

    def __init__(self, directory: Path | None, task_files: TaskFileStore) -> None:
        metadata = MetaData()
        items, scenarios = data_tables(metadata, sqlite=True)
        super().__init__(items, scenarios, task_files)
        if directory is None:
            database = ":memory:"
        else:
            directory.mkdir(parents=True, exist_ok=True)
            database = str(directory / "data-store.sqlite3")
        self.engine = create_engine(
            URL.create("sqlite+pysqlite", database=database),
            connect_args={"timeout": 30, "check_same_thread": False},
            poolclass=NullPool,
            hide_parameters=True,
        )
        self.lock = RLock()
        self.connection: Connection | None = self.engine.connect()
        if database != ":memory:":
            self.connection.exec_driver_sql("PRAGMA journal_mode = WAL")
        metadata.create_all(self.connection)
        self.connection.commit()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore) -> Self:
        if config:
            raise PluginConfigError(
                f"the sqlite data store takes no config, got {', '.join(sorted(map(str, config)))}"
            )
        return cls(directory, task_files)

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        with self.lock:
            if self.connection is None:
                raise ReefError("the data store is closed")
            with self.connection.begin():
                yield self.connection

    def close(self) -> None:
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
                self.engine.dispose()


@dataclass(frozen=True)
class PostgresDataStoreSettings(PluginSettings):
    """The config of the postgres data store: the database, and the schema of its tables."""

    database_url: str = field(repr=False, metadata=config_metadata("PostgreSQL connection URL."))
    schema: str = config_option("reef_data", help="SQL schema of the data store's tables.")


class PostgresDataStore(SQLDataStore):
    """The data store in a PostgreSQL schema. Deployments that name one database and schema share their items."""

    def __init__(self, database_url: str, task_files: TaskFileStore, *, schema: str = "reef_data") -> None:
        validate_postgres_schema(schema)
        metadata = MetaData(schema=schema)
        items, scenarios = data_tables(metadata, sqlite=False)
        super().__init__(items, scenarios, task_files)
        try:
            self.engine: Engine | None = create_engine(
                postgres_url(database_url), pool_pre_ping=True, hide_parameters=True
            )
        except ModuleNotFoundError:
            raise ReefError("the PostgreSQL data store requires the reef-infra[postgres] extra") from None
        key = int.from_bytes(hashlib.sha256(f"reef-data-schema:{schema}".encode()).digest()[:8], "big", signed=True)
        with self.engine.begin() as connection:
            # Serialize the check-then-create of the schema across deployments that start together.
            connection.execute(select(func.pg_advisory_xact_lock(key)))
            connection.execute(CreateSchema(schema, if_not_exists=True))
            metadata.create_all(connection)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore) -> Self:
        settings = PostgresDataStoreSettings.parse(config, plugin_class=cls)
        return cls(settings.database_url, task_files, schema=settings.schema)

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        if self.engine is None:
            raise ReefError("the data store is closed")
        with self.engine.begin() as connection:
            yield connection

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None


__all__ = [
    "PostgresDataStore",
    "PostgresDataStoreSettings",
    "SQLDataStore",
    "SQLiteDataStore",
    "data_tables",
]
