"""The data store (RFC #1028) in a SQL database, for SQLite and PostgreSQL.

``data_item`` holds one row per item: its owner, ID, kind, digest, metadata as JSON, and the JSON content of a
trajectory or a record. ``data_item_scenario`` repeats the ``scenarios`` metadata, one row per scenario that uses
an item, so that a query by scenario uses an index. ``data_record_node`` keeps the trajectory node of each record
item with a message list, per owner (RFC #1054): the node keeps the payload as ATIF, with only the entries after
its parent's, and the item's content keeps an empty payload. The tables are written in one transaction. Task directories are
in the task file store, in each owner's partition.

With a segment store, the content of a trajectory item, and the content of a trajectory node, can live in a segment
line instead (``reef.storage.segments``). ``create`` writes the content of new trajectory items as one segment
when it is at least ``CREATE_SEGMENT_BYTES``. ``maintain`` moves the rest into segments once an owner has
``PACK_BYTES`` of it, merges the segments of small writes (``merge_small_segments``), and deletes the segments
that no row uses. ``data_segment`` lists each segment that the store wrote, so that ``maintain`` finds the unused
ones.

A delete moves the item's row from ``data_item`` into ``data_deleted_item``, and a restore moves it back. A deleted
item keeps its trajectory node and its segment line until a purge removes its row.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import uuid
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import batched
from pathlib import Path
from threading import Event, Lock, RLock
from typing import Any, Self

from sqlalchemy import (
    URL,
    BigInteger,
    Column,
    Float,
    FromClause,
    Identity,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    bindparam,
    case,
    cast,
    create_engine,
    delete,
    exists,
    func,
    insert,
    inspect,
    literal,
    literal_column,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, Engine, RowMapping
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema
from sqlalchemy.sql.elements import ColumnElement

from reef.core.atif_payloads import call_shape, provider_payload
from reef.core.config import config_metadata, config_option
from reef.core.errors import ReefError
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.core.records_types import RecordLoss
from reef.core.trajectory_index import call_key
from reef.storage.data import (
    HOLD_ITEM_KINDS,
    JOB_DIRECTORY_TTL_SECONDS,
    MAX_QUERY_LIMIT,
    SET_KEYS,
    DataStore,
    DeletedItem,
    ItemConflictError,
    ItemGroup,
    ItemKind,
    ItemNotFoundError,
    ItemUse,
    NewItem,
    RecordLink,
    ScenarioArchivedError,
    ScenarioHold,
    StoredItem,
    check_deleted_listing,
    check_filters,
    check_groups,
    check_hold,
    check_owner,
    check_query,
    check_sets,
    generation_origin,
    held_sets,
    patched_metadata,
    prepared_items,
    record_link_scope,
    replaced_task_digest,
    replaced_trajectory_digest,
)
from reef.storage.item_kinds import TaskFilesReference, task_partition
from reef.storage.schema_upgrades import SchemaStep, upgrade_schema
from reef.storage.segments import LocalSegmentStore, SegmentLine, SegmentPartition, SegmentStore, segment_lines
from reef.storage.sql_records import (
    SEGMENT_COLUMNS,
    EvictionSizes,
    NewNode,
    ScopedTable,
    TrajectoryNodes,
    batches,
    node_call_json,
    node_line,
    row_line,
    trajectory_node_table,
)
from reef.storage.sql_setup import enable_wal, postgres_url, prefix_range, validate_postgres_schema
from reef.storage.task_collections import TaskCollection
from reef.storage.tasks import TaskFileStore

#: The content bytes that an owner's rows hold before ``maintain`` moves them into one segment.
PACK_BYTES = 64 * 2**20
#: The content bytes of the new trajectory items of one ``create`` call from which the call writes them as a segment.
CREATE_SEGMENT_BYTES = 2**20
#: The age in seconds after which ``maintain`` deletes a segment that no row uses. A new segment has no rows
#: until the transaction that writes them commits.
UNUSED_SEGMENT_SECONDS = 24 * 3600
#: The record items that one page of ``prune_records`` reads.
PRUNE_PAGE_SIZE = 256
#: The most items that one transaction of a group commit writes.
GROUP_COMMIT_ITEMS = 200
#: The table of the uses of items by scenarios.
USES_TABLE = "data_item_use"
#: The table of the version of the data tables (``reef.storage.schema_upgrades``).
VERSION_TABLE = "data_schema_version"
#: The name of the lock that serializes the writes of the holds of an owner and of the benchmarks of its items. No
#: scenario has this name.
HOLDS_LOCK = "\x00benchmark-holds"


def item_evaluation_id(items: Table, *, sqlite: bool) -> ColumnElement[Any]:
    """The ``evaluation_id`` of the metadata of an item row, in the form of the index
    ``data_item_by_evaluation_sequence``.

    The JSON path is a literal, not a parameter, so that a query with the same expression can use the index.
    """
    if sqlite:
        return func.json_extract(items.c.metadata_json, literal_column("'$.evaluation_id'"))
    return cast(items.c.metadata_json, JSONB).op("->>")(literal_column("'evaluation_id'"))


def item_id_prefix(items: Table, prefix: str, *, sqlite: bool) -> ColumnElement[bool]:
    """The condition that the ``item_id`` of an item row starts with ``prefix``, in the form of an index: on SQLite
    the unique index of (``owner``, ``item_id``), on PostgreSQL ``data_item_by_item_id``.

    The condition is the range of the IDs that start with ``prefix`` in code point order (``prefix_range``), not
    ``LIKE``: PostgreSQL uses an index for ``LIKE`` only under the "C" collation, and SQLite only when ``LIKE`` is
    case-sensitive. Without an index, a query of one prefix reads every row of the table.
    """
    return prefix_range(items.c.item_id if sqlite else items.c.item_id.collate("C"), prefix)


def use_id_prefix(uses: Table, prefix: str, *, sqlite: bool) -> ColumnElement[bool]:
    """The condition that the ``item_id`` of a row of ``data_item_use`` starts with ``prefix``, in a form that no
    index of the table serves.

    A query of the uses of one scenario then reads them from ``data_item_use_by_item`` in item order and checks the
    ID of each (``SQLDataStore.filtered``). On PostgreSQL the condition is the range of ``prefix_range`` in the "C"
    collation, which no index of the table has. On SQLite that range would use the unique index of the uses, so the
    condition compares the first ``len(prefix)`` characters of the ID with ``prefix`` instead. SQLite counts these
    characters in code points, as Python does, and compares strings by their bytes, so it keeps the same rows.
    """
    if sqlite:
        return func.substr(uses.c.item_id, 1, len(prefix)) == prefix
    return prefix_range(uses.c.item_id.collate("C"), prefix)


def item_is_inference(items: Table, *, sqlite: bool) -> ColumnElement[bool]:
    """The condition that an item row is a record item of the source ``inference``
    (``reef.storage.data.record_source``), in the form of the predicate of the partial index ``data_item_of_inference``.

    The condition holds only literals, not parameters, so that the database can prove that a query with the same
    condition reads only rows of the index, also in a prepared statement.
    """
    source: ColumnElement[str] = (
        func.json_extract(items.c.metadata_json, literal_column("'$.origin.source'"))
        if sqlite
        else cast(items.c.metadata_json, JSONB).op("#>>", return_type=Text)(literal_column("'{origin,source}'"))
    )
    return and_(
        items.c.kind == literal_column("'record'"),
        func.coalesce(source, literal_column("'inference'")) == literal_column("'inference'"),
    )


def node_of_scenario_call(nodes: Table) -> ColumnElement[bool]:
    """The condition that a row of ``data_record_node`` is the node of a scenario's own call, not of an evaluation
    call, in the form of the predicate of the partial index ``data_record_node_scenario_bytes``.

    The scenario of a node is the link scope of its record (``record_link_scope``): the origin scenario of a
    scenario's own call, and ``<scenario>/<rollout>`` for an evaluation call. A scenario name has no ``/``, so only
    the node of an evaluation call has one. The pattern is a literal for the same reason as in
    ``item_is_inference``.
    """
    return nodes.c.scenario.not_like(literal_column("'%/%'"))


def item_columns() -> list[Column[Any]]:
    """New columns of an item row but ``sequence``, which ``data_item`` and ``data_deleted_item`` both have."""
    return [
        Column("owner", Text, nullable=False),
        Column("item_id", Text, nullable=False),
        Column("kind", String(16), nullable=False),
        Column("digest", String(64), nullable=False),
        Column("metadata_json", Text, nullable=False),
        Column("content_json", Text, nullable=True),
        Column("message_field", Text, nullable=True),
        Column("request_type", String(16), nullable=True),
        Column("segment_id", Text, nullable=True),
        Column("segment_offset", BigInteger, nullable=True),
        Column("segment_length", BigInteger, nullable=True),
    ]


def data_tables(
    metadata: MetaData, *, sqlite: bool
) -> tuple[Table, Table, Table, Table, Table, Table, Table, Table, Table, Table, Table]:
    """Define ``data_item``, ``data_item_use``, ``data_record_node``, ``data_segment``, ``data_job_directory``,
    ``data_scenario``, ``data_record_loss``, ``data_collection``, ``data_scenario_hold``, ``data_item_set`` and
    ``data_deleted_item`` in ``metadata``. ``data_deleted_item`` has the columns of ``data_item`` and
    ``deleted_at``, one row per deleted item (``DataStore.delete``): the row that the item had, and the time of the
    delete in seconds since the epoch. Its ``sequence`` is the item's, not a new one. ``data_item_set`` has one row
    for each set that an item is in (``held_sets``): the owner, the
    set's kind and name, and the item's sequence and ID. It changes with the item's metadata in one transaction.
    ``data_scenario_hold`` has one row per hold of a set by a scenario of an owner, with the set's kind and name and
    the hold's split (``DataStore.hold``); a scenario can hold one benchmark in both splits. ``data_collection`` has one row
    per ``TaskCollection`` of an owner, with its JSON. ``data_scenario`` has the current generation of each
    scenario of an owner (``DataStore.scenario_generation``). ``data_record_loss`` has the ``RecordLoss`` of each
    origin scenario of an owner whose record items ``prune_records`` deleted. ``data_job_directory`` has the key of each job data directory and the time that a job last used it.

    ``sequence`` orders the items as they arrived. ``data_item_use`` has one row per use of an item by a
    scenario, with its split and its held set: the name of the held benchmark or trajectory set that gives the use,
    or "" for the use of the item's own ``scenarios`` metadata. Its own ``sequence`` orders the uses as they began, and a use whose split
    changes is a new row (``DataStore.scenario_items``). ``item_sequence`` is the ``sequence`` of the used item, so
    that a query of the items that a scenario uses reads them in item order from the index ``data_item_use_by_item``.
    ``content_json`` is the canonical JSON of a trajectory's or a
    record's content, and NULL for a task. ``message_field`` names the message list of a record payload that the
    item's trajectory node keeps (``call_shape``); the content then holds an empty payload. It is NULL for an
    item without one. ``request_type`` is a record's request type, from its content, and NULL for another kind.
    ``metadata_json`` is the item's metadata. ``data_record_node`` is the node table of ``TrajectoryNodes``,
    scoped by ``owner``; a node's sequence is its item's, and its scenario the item's link scope
    (``record_link_scope``).

    The ``SEGMENT_COLUMNS`` of ``data_item`` and ``data_record_node`` are NULL, or the segment line that holds the
    content of the row: then ``content_json`` of the item, or the JSON columns of the node, are NULL.
    ``data_segment`` has one row per segment that the store wrote: its owner, ID, creation time in seconds since the
    epoch and size.
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
        *item_columns(),
        UniqueConstraint("owner", "item_id"),
        Index("data_item_by_segment", "owner", "segment_id"),
        Index("data_item_by_kind", "owner", "kind", "sequence"),
        # SQLite reuses the largest rowid after a delete without it; other dialects ignore the option.
        sqlite_autoincrement=sqlite,
    )
    uses = Table(
        USES_TABLE,
        metadata,
        (
            Column("sequence", Integer, primary_key=True)
            if sqlite
            else Column("sequence", BigInteger, Identity(), primary_key=True)
        ),
        Column("owner", Text, nullable=False),
        Column("item_id", Text, nullable=False),
        Column("scenario", Text, nullable=False),
        Column("held_set", Text, nullable=False),
        Column("split", String(8), nullable=False),
        Column("item_sequence", BigInteger, nullable=False),
        UniqueConstraint("owner", "item_id", "scenario", "held_set", "split"),
        Index("data_item_use_by_scenario", "owner", "scenario", "sequence"),
        Index("data_item_use_by_item", "owner", "scenario", "item_sequence"),
        sqlite_autoincrement=sqlite,
    )
    holds = Table(
        "data_scenario_hold",
        metadata,
        Column("owner", Text, nullable=False),
        Column("scenario", Text, nullable=False),
        Column("kind", String(16), nullable=False),
        Column("name", Text, nullable=False),
        Column("split", String(8), nullable=False),
        PrimaryKeyConstraint("owner", "scenario", "kind", "name", "split"),
        Index("data_scenario_hold_by_set", "owner", "kind", "name"),
    )
    nodes = trajectory_node_table(metadata, "data_record_node", Column("owner", Text, primary_key=True), segments=True)
    segments = Table(
        "data_segment",
        metadata,
        Column("owner", Text, nullable=False),
        Column("segment_id", Text, nullable=False),
        Column("created_at", Float, nullable=False),
        Column("byte_count", BigInteger, nullable=False),
        PrimaryKeyConstraint("owner", "segment_id"),
    )
    job_directories = Table(
        "data_job_directory",
        metadata,
        Column("key", String(64), primary_key=True),
        Column("last_used_at", Float, nullable=False),
    )
    generations = Table(
        "data_scenario",
        metadata,
        Column("owner", Text, nullable=False),
        Column("scenario", Text, nullable=False),
        Column("generation", String(32), nullable=False),
        PrimaryKeyConstraint("owner", "scenario"),
    )
    losses = Table(
        "data_record_loss",
        metadata,
        Column("owner", Text, nullable=False),
        Column("scenario", Text, nullable=False),
        Column("record_count", BigInteger, nullable=False),
        Column("body_bytes", BigInteger, nullable=False),
        Column("first_sequence", BigInteger, nullable=False),
        Column("last_sequence", BigInteger, nullable=False),
        PrimaryKeyConstraint("owner", "scenario"),
    )
    # The calls of one evaluation are read by their evaluation_id, a page at a time in sequence order
    # (``SQLDataStore.query``). With ``sequence`` in the index, a page reads only its own rows, not every row of the
    # evaluation.
    Index(
        "data_item_by_evaluation_sequence",
        items.c.owner,
        item_evaluation_id(items, sqlite=sqlite),
        items.c.sequence,
    )
    if not sqlite:
        # A query of an ID prefix reads the range of its IDs in code point order (``item_id_prefix``). The unique
        # index of (owner, item_id) has the database's collation, whose order is another one.
        Index("data_item_by_item_id", items.c.owner, items.c.item_id.collate("C"))
    # Record retention (``SQLDataStore.prune_records``) counts the bytes of the record items of the source inference
    # and of the nodes of the scenarios' own calls, then reads those items in sequence order. The partial indexes
    # hold only these rows, so that the counts parse no metadata and read no row of an evaluation call.
    inference = item_is_inference(items, sqlite=sqlite)
    Index("data_item_of_inference", items.c.sequence, sqlite_where=inference, postgresql_where=inference)
    own_calls = node_of_scenario_call(nodes)
    Index("data_record_node_scenario_bytes", nodes.c.body_bytes, sqlite_where=own_calls, postgresql_where=own_calls)
    collections = Table(
        "data_collection",
        metadata,
        Column("owner", Text, nullable=False),
        Column("name", Text, nullable=False),
        Column("collection_json", Text, nullable=False),
        PrimaryKeyConstraint("owner", "name"),
    )
    # A page of a set reads its rows in item order, and a delete of an item finds the rows of its sets.
    item_sets = Table(
        "data_item_set",
        metadata,
        Column("owner", Text, nullable=False),
        Column("kind", String(16), nullable=False),
        Column("name", Text, nullable=False),
        Column("sequence", BigInteger, nullable=False),
        Column("item_id", Text, nullable=False),
        PrimaryKeyConstraint("owner", "kind", "name", "sequence"),
        Index("data_item_set_by_item", "owner", "item_id"),
    )
    deleted = Table(
        "data_deleted_item",
        metadata,
        Column("sequence", BigInteger, nullable=False),
        *item_columns(),
        Column("deleted_at", Float, nullable=False),
        PrimaryKeyConstraint("owner", "item_id"),
        Index("data_deleted_item_by_sequence", "owner", "sequence"),
        Index("data_deleted_item_by_segment", "owner", "segment_id"),
        Index("data_deleted_item_by_time", "deleted_at"),
    )
    return items, uses, nodes, segments, job_directories, generations, losses, collections, holds, item_sets, deleted


class UsesWithHeldSets(SchemaStep):
    """Version 1 to 2 of the data tables (RFC #1420): the uses of items by scenarios move from
    ``data_item_scenario`` to ``data_item_use``, which also has the held set of a use.

    Version 1 had one use for each item and scenario, of the item's own ``scenarios`` metadata, so each copied use
    has no held set. The sequences are new, and their order is the earlier one, so a trainer reads its samples in
    the same order. Each copied use has the sequence of its item (``UseItemSequences``); a use of no item gives no
    row, as no query found its item.
    """

    def apply(self, connection: Connection, schema: str | None) -> None:
        tables = MetaData(schema=schema)
        earlier = Table(
            "data_item_scenario",
            tables,
            Column("sequence", BigInteger),
            Column("owner", Text),
            Column("item_id", Text),
            Column("scenario", Text),
            Column("split", String(8)),
        )
        uses = Table(
            USES_TABLE,
            tables,
            Column("owner", Text),
            Column("item_id", Text),
            Column("scenario", Text),
            Column("held_set", Text),
            Column("split", String(8)),
            Column("item_sequence", BigInteger),
        )
        items = Table(
            "data_item", tables, Column("sequence", BigInteger), Column("owner", Text), Column("item_id", Text)
        )
        rows = select(
            earlier.c.owner, earlier.c.item_id, earlier.c.scenario, literal(""), earlier.c.split, items.c.sequence
        ).join(items, (items.c.owner == earlier.c.owner) & (items.c.item_id == earlier.c.item_id))
        connection.execute(
            insert(uses).from_select(
                ["owner", "item_id", "scenario", "held_set", "split", "item_sequence"],
                rows.order_by(earlier.c.sequence),
            )
        )
        earlier.drop(connection)


class ItemIdIndex(SchemaStep):
    """Version 2 to 3 of the data tables: on PostgreSQL, ``data_item`` gets the index ``data_item_by_item_id`` of
    (``owner``, ``item_id`` in the "C" collation), so that a query of an ID prefix reads only the rows of its range
    (``item_id_prefix``). On SQLite nothing changes: the unique index of (``owner``, ``item_id``) is in that order.
    """

    def apply(self, connection: Connection, schema: str | None) -> None:
        if connection.dialect.name == "sqlite":
            return
        items = Table("data_item", MetaData(schema=schema), Column("owner", Text), Column("item_id", Text))
        Index("data_item_by_item_id", items.c.owner, items.c.item_id.collate("C")).create(connection)


class ItemSets(SchemaStep):
    """Version 3 to 4 of the data tables: a trajectory can be in many trajectory sets, as a task in many benchmarks,
    and ``data_item_set`` holds the sets of each item.

    The ``set_id`` of a trajectory's metadata, when it is a non-empty string, becomes ``trajectory_sets``, a list of
    that one name, and leaves the metadata. A trajectory sample that a scenario's stage made from its records
    (``origin`` with a ``component``) joins the set ``samples:<scenario>`` (``record_samples_set``).
    ``data_item_set`` then gets one row for each set of each task and trajectory: the names of its ``benchmarks`` or
    ``trajectory_sets``. The step reads the items a page at a time, and parses only the metadata whose text has one
    of these keys.
    """

    def apply(self, connection: Connection, schema: str | None) -> None:
        tables = MetaData(schema=schema)
        items = Table(
            "data_item",
            tables,
            Column("sequence", BigInteger),
            Column("owner", Text),
            Column("item_id", Text),
            Column("kind", String(16)),
            Column("metadata_json", Text),
        )
        item_sets = Table(
            "data_item_set",
            tables,
            Column("owner", Text),
            Column("kind", String(16)),
            Column("name", Text),
            Column("sequence", BigInteger),
            Column("item_id", Text),
        )
        set_keys = {"task": ("benchmark", "benchmarks"), "trajectory": ("trajectory_set", "trajectory_sets")}
        keyed = or_(
            *(
                items.c.metadata_json.contains(f'"{key}"')
                for key in ("set_id", "benchmarks", "trajectory_sets", "component")
            )
        )
        after = 0
        while rows := connection.execute(
            select(items.c.sequence, items.c.owner, items.c.item_id, items.c.kind, items.c.metadata_json)
            .where(items.c.sequence > after, items.c.kind.in_(list(set_keys)), keyed)
            .order_by(items.c.sequence)
            .limit(MAX_QUERY_LIMIT)
        ).all():
            rewritten: list[dict[str, Any]] = []
            members: list[dict[str, Any]] = []
            for row in rows:
                metadata = json.loads(row.metadata_json)
                joined: list[str] = []
                if row.kind == "trajectory":
                    name, origin = metadata.get("set_id"), metadata.get("origin")
                    if isinstance(name, str) and name:
                        metadata.pop("set_id")
                        joined.append(name)
                    if isinstance(origin, dict) and "component" in origin:
                        joined.append(f"samples:{origin['scenario']}")
                if joined:
                    metadata["trajectory_sets"] = joined
                    rewritten.append({"row_sequence": row.sequence, "metadata_json": canonical_json(metadata)})
                set_kind, key = set_keys[row.kind]
                names = metadata.get(key)
                members.extend(
                    {
                        "owner": row.owner,
                        "kind": set_kind,
                        "name": name,
                        "sequence": row.sequence,
                        "item_id": row.item_id,
                    }
                    for name in dict.fromkeys(names if isinstance(names, list) else [])
                    if isinstance(name, str) and name
                )
            if rewritten:
                connection.execute(
                    update(items)
                    .where(items.c.sequence == bindparam("row_sequence"))
                    .values(metadata_json=bindparam("metadata_json")),
                    rewritten,
                )
            if members:
                connection.execute(insert(item_sets), members)
            after = rows[-1].sequence


class UseItemSequences(SchemaStep):
    """Version 4 to 5 of the data tables: each row of ``data_item_use`` gets ``item_sequence``, the sequence of its
    item, and the index ``data_item_use_by_item`` of (``owner``, ``scenario``, ``item_sequence``). A query of the
    items that a scenario uses then reads them in item order from that index (``SQLDataStore.filtered``), and no
    longer reads every item of the owner that the other filters give.

    The step deletes the uses of no item first: no query found their items. A table that this upgrade made at the
    current layout, from version 1, has the column and the index already, and the step changes nothing in it.
    SQLite adds a ``NOT NULL`` column only with a default, so there the column has the default 0, which no write
    uses.
    """

    def apply(self, connection: Connection, schema: str | None) -> None:
        if any(column["name"] == "item_sequence" for column in inspect(connection).get_columns(USES_TABLE, schema)):
            return
        tables = MetaData(schema=schema)
        uses = Table(
            USES_TABLE,
            tables,
            Column("owner", Text),
            Column("item_id", Text),
            Column("scenario", Text),
            Column("item_sequence", BigInteger),
        )
        items = Table(
            "data_item", tables, Column("sequence", BigInteger), Column("owner", Text), Column("item_id", Text)
        )
        item_of_use = (items.c.owner == uses.c.owner) & (items.c.item_id == uses.c.item_id)
        connection.execute(delete(uses).where(~exists().where(item_of_use)))
        name = connection.dialect.identifier_preparer.format_table(uses)
        sqlite = connection.dialect.name == "sqlite"
        connection.exec_driver_sql(
            f"ALTER TABLE {name} ADD COLUMN item_sequence BIGINT" + (" NOT NULL DEFAULT 0" if sqlite else "")
        )
        connection.execute(
            update(uses).values(item_sequence=select(items.c.sequence).where(item_of_use).scalar_subquery())
        )
        if not sqlite:
            connection.exec_driver_sql(f"ALTER TABLE {name} ALTER COLUMN item_sequence SET NOT NULL")
        Index("data_item_use_by_item", uses.c.owner, uses.c.scenario, uses.c.item_sequence).create(connection)


class RetentionIndexes(SchemaStep):
    """Version 5 to 6 of the data tables: the partial indexes ``data_item_of_inference`` of the record items of the
    source inference (``item_is_inference``) and ``data_record_node_scenario_bytes`` of the nodes of the scenarios'
    own calls (``node_of_scenario_call``). Record retention counts and reads these rows from them. ``data_item``
    is in every version. A database of version 1 can lack ``data_record_node``: this upgrade then made it at the
    current layout, with its index, and the step leaves that index.
    """

    def apply(self, connection: Connection, schema: str | None) -> None:
        sqlite = connection.dialect.name == "sqlite"
        tables = MetaData(schema=schema)
        items = Table(
            "data_item",
            tables,
            Column("sequence", BigInteger),
            Column("kind", String(16)),
            Column("metadata_json", Text),
        )
        nodes = Table("data_record_node", tables, Column("scenario", Text), Column("body_bytes", BigInteger))
        inference = item_is_inference(items, sqlite=sqlite)
        own_calls = node_of_scenario_call(nodes)
        Index("data_item_of_inference", items.c.sequence, sqlite_where=inference, postgresql_where=inference).create(
            connection
        )
        Index(
            "data_record_node_scenario_bytes", nodes.c.body_bytes, sqlite_where=own_calls, postgresql_where=own_calls
        ).create(connection, checkfirst=True)


class DeletedItems(SchemaStep):
    """Version 6 to 7 of the data tables: a delete keeps the item in ``data_deleted_item``, which ``upgrade_schema``
    makes, empty. The step moves no row. The version changes all the same, so that an earlier Reef refuses the
    tables: it would delete the segments that only deleted items use."""

    def apply(self, connection: Connection, schema: str | None) -> None:
        return None


#: The steps of the data tables, in the order of their versions: step ``n`` upgrades version ``n``.
DATA_SCHEMA_STEPS: tuple[SchemaStep, ...] = (
    UsesWithHeldSets(),
    ItemIdIndex(),
    ItemSets(),
    UseItemSequences(),
    RetentionIndexes(),
    DeletedItems(),
)


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deleted_item(row: RowMapping) -> DeletedItem:
    """The ``DeletedItem`` of a row of ``data_deleted_item``, without content."""
    return DeletedItem(
        StoredItem(row["sequence"], row["owner"], row["item_id"], json.loads(row["metadata_json"]), row["digest"]),
        float(row["deleted_at"]),
    )


def scenarios_of(metadata: Iterable[Mapping[str, Any]]) -> set[str]:
    """The scenarios whose order a write of items with ``metadata`` changes: those that use them, and the origin of a
    record item, whose trajectory nodes it changes."""
    names: set[str] = set()
    for value in metadata:
        names.update(value.get("scenarios", {}))
        if value["kind"] == "record":
            names.add(value["origin"]["scenario"])
    return names


@dataclass
class PendingCreate:
    """A ``create`` call whose rows are ready, and that waits for a transaction that writes them."""

    prepared: list[tuple[NewItem, dict[str, Any], str]]
    values: dict[str, dict[str, Any]]
    calls: dict[str, NewNode]
    generation: tuple[str, str] | None
    done: Event = field(default_factory=Event)
    lead: bool = False
    outcome: tuple[StoredItem, ...] | Exception | None = None


class SQLDataStore(DataStore):
    """A data store in the tables of ``data_tables``.

    A subclass gives the tables, ``transaction`` (one database transaction on one connection), ``write_new_items``
    (an insert that leaves an existing ID alone) and, where transactions run at once, ``lock_scenarios``. Every
    operation runs in one transaction, so an operation that refuses changes nothing in the database. ``create``
    inserts first and compares digests after, so two creates of one ID at once end in a no-op or an
    ``ItemConflictError``, never in a failed insert. ``create``, ``update`` and ``delete`` take ``lock_scenarios`` for
    the scenarios that they give a use or whose trajectory nodes they change. Concurrent creates of one owner and one
    set of scenarios in this process share one transaction (a group commit), because only one of them can hold
    the lock at a time. ``segments`` is None, or the segment store that keeps the content of
    trajectory items and nodes in the partition of each owner; the store closes it with itself. ``job_directories`` keeps
    the job data directories: ``job_directories`` when given, else ``segments``, else a local segment store in a new
    temporary directory, which only this process reads. ``job_directory_ttl_seconds`` is how long the store keeps a
    job data directory that no job used.
    """

    def __init__(
        self,
        tables: tuple[Table, Table, Table, Table, Table, Table, Table, Table, Table, Table, Table],
        task_files: TaskFileStore,
        segments: SegmentStore | None,
        job_directories: SegmentStore | None = None,
        job_directory_ttl_seconds: float = JOB_DIRECTORY_TTL_SECONDS,
    ) -> None:
        (
            self.items_table,
            self.uses_table,
            self.nodes_table,
            self.segment_table,
            self.job_directory_table,
            self.generations_table,
            self.losses_table,
            self.collections_table,
            self.holds_table,
            self.item_sets_table,
            self.deleted_table,
        ) = tables
        self.task_files = task_files
        self.segments = segments
        self.job_directories = (
            job_directories or segments or LocalSegmentStore(Path(tempfile.mkdtemp(prefix="reef-job-directories-")))
        )
        self.job_directory_ttl_seconds = job_directory_ttl_seconds
        self.group_lock = Lock()
        self.waiting: dict[tuple[str, frozenset[str]], list[PendingCreate]] = {}

    def write_job_directory_use(self, key: str, time: float) -> None:
        table = self.job_directory_table
        with self.transaction() as connection:
            connection.execute(delete(table).where(table.c.key == key))
            connection.execute(insert(table).values(key=key, last_used_at=time))

    def job_directories_unused_since(self, time: float) -> tuple[str, ...]:
        table = self.job_directory_table
        with self.transaction(read_only=True) as connection:
            return tuple(connection.execute(select(table.c.key).where(table.c.last_used_at < time)).scalars())

    def delete_job_directory_record(self, key: str) -> None:
        with self.transaction() as connection:
            connection.execute(delete(self.job_directory_table).where(self.job_directory_table.c.key == key))

    @abstractmethod
    @contextmanager
    def transaction(self, *, read_only: bool = False) -> Iterator[Connection]:
        """One transaction: commit when the block ends, roll back when it raises.

        A ``read_only`` transaction reads one snapshot in all its statements, so a concurrent ``delete`` cannot
        remove a row between two of them."""

    @abstractmethod
    def write_new_items(self, connection: Connection, values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        """Insert the ``data_item`` rows ``values`` in one statement, each unless the owner has its ID already; return
        the sequence of each row that it inserted, by item ID.

        The insert never fails on the unique ``(owner, item_id)``: a row that another transaction inserts first wins,
        and the caller compares digests after the insert.
        """

    def lock_scenarios(self, connection: Connection, owner: str, scenarios: Iterable[str]) -> None:
        """Serialize the writes that name each of ``scenarios`` of ``owner``, until the transaction ends. Take the locks
        in sorted order, so that writers do not deadlock.

        Sequence order is then commit order, so a reader that pages with ``after`` misses no item. The default
        does nothing, for a store whose transactions already run one at a time.
        """

    @abstractmethod
    def metadata_string(self, key: str) -> ColumnElement[str]:
        """The string under ``key`` in the ``metadata_json`` of an item row, or NULL when the value there is not a
        string. ``key`` passes ``GROUP_KEY``."""

    @abstractmethod
    def evaluation_id(self) -> ColumnElement[Any]:
        """The ``evaluation_id`` of the metadata of an item row, as the index ``data_item_by_evaluation_sequence``
        holds it (``item_evaluation_id``)."""

    @abstractmethod
    def has_id_prefix(self, prefix: str) -> ColumnElement[bool]:
        """The condition that the ``item_id`` of an item row starts with ``prefix``, in the form of an index
        (``item_id_prefix``)."""

    @abstractmethod
    def use_has_id_prefix(self, prefix: str) -> ColumnElement[bool]:
        """The condition that the ``item_id`` of a use row starts with ``prefix``, in a form that no index of the
        uses serves (``use_id_prefix``)."""

    @abstractmethod
    def is_inference(self) -> ColumnElement[bool]:
        """The condition that an item row is a record item of the source ``inference``, which ``prune_records``
        counts and deletes (``item_is_inference``)."""

    def groups(
        self,
        owner: str,
        *,
        kind: ItemKind,
        key: str,
        scenario: str | None = None,
        before: int | None = None,
        limit: int = MAX_QUERY_LIMIT,
    ) -> tuple[ItemGroup, ...]:
        check_groups(owner, kind, key, scenario, before, limit)
        items = self.items_table
        values = (
            select(items.c.sequence, self.metadata_string(key).label("value"))
            .where(items.c.owner == owner, items.c.kind == kind)
            .where(*self.used_by(owner, items, scenario))
            .subquery()
        )
        first = func.min(values.c.sequence)
        statement = select(values.c.value, func.count(), first, func.max(values.c.sequence)).group_by(values.c.value)
        if before is not None:
            statement = statement.having(first < before)
        with self.transaction(read_only=True) as connection:
            found = connection.execute(statement.order_by(first.desc()).limit(limit)).all()
            metadata = self.metadata_of(connection, owner, [int(row[2]) for row in found])
        return tuple(
            ItemGroup(value, int(count), int(low), int(high), metadata[int(low)]) for value, count, low, high in found
        )

    def sets(
        self,
        owner: str,
        kind: str,
        *,
        scenario: str | None = None,
        before: int | None = None,
        limit: int = MAX_QUERY_LIMIT,
    ) -> tuple[ItemGroup, ...]:
        check_sets(owner, kind, scenario, before, limit)
        item_sets, uses = self.item_sets_table, self.uses_table
        first = func.min(item_sets.c.sequence)
        statement = (
            select(item_sets.c.name, func.count(), first, func.max(item_sets.c.sequence))
            .where(item_sets.c.owner == owner, item_sets.c.kind == kind)
            .group_by(item_sets.c.name)
        )
        if scenario is not None:
            # An item that the scenario uses through several sets or splits counts once.
            used = exists().where(
                uses.c.owner == owner, uses.c.item_id == item_sets.c.item_id, uses.c.scenario == scenario
            )
            statement = statement.where(used)
        if before is not None:
            statement = statement.having(first < before)
        with self.transaction(read_only=True) as connection:
            found = connection.execute(statement.order_by(first.desc()).limit(limit)).all()
            metadata = self.metadata_of(connection, owner, [int(row[2]) for row in found])
        return tuple(
            ItemGroup(name, int(count), int(low), int(high), metadata[int(low)]) for name, count, low, high in found
        )

    def metadata_of(self, connection: Connection, owner: str, sequences: Sequence[int]) -> dict[int, dict[str, Any]]:
        """The metadata of the items of ``owner`` with ``sequences``, by sequence."""
        items = self.items_table
        return {
            int(sequence): json.loads(text)
            for sequence, text in connection.execute(
                select(items.c.sequence, items.c.metadata_json).where(
                    items.c.owner == owner, items.c.sequence.in_(list(sequences))
                )
            )
        }

    def insert_sets(
        self, connection: Connection, owner: str, items: Mapping[str, tuple[int, Mapping[str, Any]]]
    ) -> None:
        """Insert the rows of ``data_item_set`` of each item ID in ``items``, which maps it to the item's sequence and
        metadata: one for each set of the item (``held_sets``)."""
        rows: list[dict[str, Any]] = []
        for item_id, (sequence, metadata) in items.items():
            set_kind, names = held_sets(metadata) or ("", ())
            rows.extend(
                {"owner": owner, "kind": set_kind, "name": name, "sequence": sequence, "item_id": item_id}
                for name in names
            )
        if rows:
            connection.execute(insert(self.item_sets_table), rows)

    @contextmanager
    def pack_lock(self, owner: str) -> Iterator[bool]:
        """Try to take the lock that lets one caller at a time ``pack`` the content of ``owner``; yield whether it
        took it. The lock holds until the block ends. The default always takes it, for a store whose transactions
        already run one at a time."""
        yield True

    def partition(self, owner: str) -> SegmentPartition | None:
        """The segment partition of ``owner``, or None without a segment store."""
        return None if self.segments is None else SegmentPartition(self.segments, owner)

    def trajectory_nodes(self, owner: str) -> TrajectoryNodes:
        """The trajectory nodes of the record items of ``owner``."""
        return TrajectoryNodes(ScopedTable(self.nodes_table, {"owner": owner}), self.partition(owner))

    def write_segment(self, partition: SegmentPartition, lines: Sequence[str]) -> tuple[SegmentLine, ...]:
        """Write ``lines`` as a new segment of ``partition``, and return the place of each line.

        The ``data_segment`` row commits before the segment is written, so ``maintain`` deletes the segment later
        when no row comes to use it.
        """
        segment_id = uuid.uuid4().hex
        data, places = segment_lines(segment_id, lines)
        with self.transaction() as connection:
            connection.execute(
                insert(self.segment_table).values(
                    owner=partition.partition, segment_id=segment_id, created_at=time.time(), byte_count=len(data)
                )
            )
        partition.store.write(partition.partition, segment_id, data)
        return places

    def stored_items(
        self, connection: Connection, owner: str, rows: Sequence[RowMapping], *, content: bool = True
    ) -> tuple[StoredItem, ...]:
        """The items of the ``data_item`` rows ``rows`` of ``owner``. With ``content``, each has its content: from
        its segment line, or from the row, with a record payload rendered from its trajectory node. Without, each
        item's content is None."""
        if not content:
            return tuple(
                StoredItem(
                    row["sequence"], row["owner"], row["item_id"], json.loads(row["metadata_json"]), row["digest"]
                )
                for row in rows
            )
        contents = self.trajectory_nodes(owner).contents(
            connection, (int(row["sequence"]) for row in rows if row["message_field"] is not None)
        )
        packed = {int(row["sequence"]): line for row in rows if (line := row_line(row)) is not None}
        partition = self.partition(owner)
        if packed and partition is None:
            raise ReefError("an item's content is in a segment, but the data store has no segment store")
        texts = (
            dict(zip(packed, partition.read_text(list(packed.values())), strict=True))
            if packed and partition is not None
            else {}
        )
        items: list[StoredItem] = []
        for row in rows:
            text = texts.get(int(row["sequence"]), row["content_json"])
            item_content = None if text is None else json.loads(text)
            if item_content is not None and row["message_field"] is not None:
                item_content["payload"] = provider_payload(contents[int(row["sequence"])].atif_call())
            items.append(
                StoredItem(
                    row["sequence"],
                    row["owner"],
                    row["item_id"],
                    json.loads(row["metadata_json"]),
                    row["digest"],
                    item_content,
                )
            )
        return tuple(items)

    def rows(
        self, connection: Connection, owner: str, item_ids: Sequence[str], *, lock: bool = False
    ) -> dict[str, RowMapping]:
        """The rows of ``item_ids`` by item ID. With ``lock``, the transaction holds them until it ends, in item ID
        order, so that two transactions that lock the same rows do not deadlock."""
        statement = select(self.items_table).where(
            self.items_table.c.owner == owner, self.items_table.c.item_id.in_(list(item_ids))
        )
        if lock:
            statement = statement.order_by(self.items_table.c.item_id).with_for_update()
        return {row["item_id"]: row for row in connection.execute(statement).mappings()}

    def holds_of(
        self, connection: Connection, owner: str, sets: Mapping[str, Iterable[str]]
    ) -> dict[tuple[str, str], set[tuple[str, str]]]:
        """The holds of each set of ``sets`` (kind, names) by the scenarios of ``owner``: for the kind and the name
        of a held set, the scenario and the split of each hold of it."""
        holds = self.holds_table
        found: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for kind, names in sorted(sets.items()):
            wanted = sorted(set(names))
            if not wanted:
                continue
            for row in connection.execute(
                select(holds.c.name, holds.c.scenario, holds.c.split).where(
                    holds.c.owner == owner, holds.c.kind == kind, holds.c.name.in_(wanted)
                )
            ):
                found.setdefault((kind, row.name), set()).add((row.scenario, row.split))
        return found

    def write_uses(
        self,
        connection: Connection,
        owner: str,
        items: Mapping[str, tuple[int, Mapping[str, Any]]],
        holds: Mapping[tuple[str, str], set[tuple[str, str]]] | None,
    ) -> None:
        """Make the uses of each item ID in ``items``, which maps it to the item's sequence and metadata, those that
        its metadata gives: one for each scenario of its ``scenarios``, and one for each hold in ``holds``
        (``holds_of``) of a set of the item (``held_sets``). With None for ``holds``, the write leaves the uses that
        holds give: the caller did not change the sets of the items, and does not hold ``HOLDS_LOCK``. An unchanged
        use keeps its row and sequence, and a new use gets a new row. At most three statements read and write the
        uses."""
        uses = self.uses_table
        metadata = {item_id: value for item_id, (_, value) in items.items()}
        wanted: dict[str, set[tuple[str, str, str]]] = {}
        for item_id, value in metadata.items():
            wanted[item_id] = {(scenario, "", split) for scenario, split in value["scenarios"].items()}
            kind, names = held_sets(value) or ("", ())
            if holds is not None:
                for name in names:
                    wanted[item_id].update((scenario, name, split) for scenario, split in holds.get((kind, name), ()))
        current: dict[str, set[tuple[str, str, str]]] = {item_id: set() for item_id in metadata}
        own = () if holds is not None else (uses.c.held_set == "",)
        for row in connection.execute(
            select(uses.c.item_id, uses.c.scenario, uses.c.held_set, uses.c.split).where(
                uses.c.owner == owner, uses.c.item_id.in_(list(metadata)), *own
            )
        ):
            current[row.item_id].add((row.scenario, row.held_set, row.split))
        ended = [
            {"row_item_id": item_id, "row_scenario": scenario, "row_held_set": held_set, "row_split": split}
            for item_id, kept in wanted.items()
            for scenario, held_set, split in sorted(current[item_id] - kept)
        ]
        if ended:
            connection.execute(
                delete(uses).where(
                    uses.c.owner == owner,
                    uses.c.item_id == bindparam("row_item_id"),
                    uses.c.scenario == bindparam("row_scenario"),
                    uses.c.held_set == bindparam("row_held_set"),
                    uses.c.split == bindparam("row_split"),
                ),
                ended,
            )
        begun = [
            {
                "owner": owner,
                "item_id": item_id,
                "scenario": scenario,
                "held_set": held_set,
                "split": split,
                "item_sequence": items[item_id][0],
            }
            for item_id, kept in wanted.items()
            for scenario, held_set, split in sorted(kept - current[item_id])
        ]
        if begun:
            connection.execute(insert(uses), begun)

    def lock_uses(
        self, connection: Connection, owner: str, metadata: Iterable[Mapping[str, Any]]
    ) -> dict[tuple[str, str], set[tuple[str, str]]]:
        """Take the locks of a write of items with ``metadata``, and return the holds of their sets (``holds_of``).

        A write of an item of a set takes ``HOLDS_LOCK`` first, so that no hold of the set begins or ends until the
        write commits. It then locks, in one call, the scenarios of the items (``scenarios_of``) and the scenarios
        that hold their sets: each gets a use, and a use gets its sequence under the scenario's lock.
        """
        values = list(metadata)
        sets: dict[str, set[str]] = {}
        for value in values:
            kind, names = held_sets(value) or ("", ())
            sets.setdefault(kind, set()).update(names)
        holds: dict[tuple[str, str], set[tuple[str, str]]] = {}
        if any(sets.values()):
            self.lock_scenarios(connection, owner, [HOLDS_LOCK])
            holds = self.holds_of(connection, owner, sets)
        holding = {scenario for held in holds.values() for scenario, _ in held}
        self.lock_scenarios(connection, owner, scenarios_of(values) | holding)
        return holds

    def create(self, owner: str, items: Sequence[NewItem], *, generation: str | None = None) -> tuple[StoredItem, ...]:
        prepared = prepared_items(owner, items, self.task_files)
        if not prepared:
            return ()
        origin = generation_origin(prepared, generation)
        item_ids = [entry[0].item_id for entry in prepared]
        places: dict[str, SegmentLine] = {}
        partition = self.partition(owner)
        if partition is not None:
            with self.transaction() as connection:
                existing = self.rows(connection, owner, item_ids)
            new = [
                (item.item_id, canonical_json(item.content))
                for item, metadata, _ in prepared
                if metadata["kind"] == "trajectory"
                and isinstance(item.content, Mapping)
                and item.item_id not in existing
            ]
            # Less content waits in the rows for ``maintain``: a segment for each small write, such as one sample that
            # a stage made, costs a reader one request for each item.
            if new and sum(len(text.encode("utf-8")) for _, text in new) >= CREATE_SEGMENT_BYTES:
                lines = self.write_segment(partition, [text for _, text in new])
                places = {item_id: line for (item_id, _), line in zip(new, lines, strict=True)}
        # The rows and the nodes, but for their sequences, need no lock: they depend on the items alone.
        values: dict[str, dict[str, Any]] = {}
        calls: dict[str, NewNode] = {}
        for item, metadata, digest in prepared:
            if item.item_id in values:
                continue
            content = dict(item.content) if isinstance(item.content, Mapping) else None
            place = places.get(item.item_id)
            payload = content["payload"] if content is not None and metadata["kind"] == "record" else None
            key = None if payload is None else call_key(payload)
            shape = None if payload is None else call_shape(payload)
            if content is not None and shape is not None:
                content["payload"] = {}
            if payload is not None and key is not None and shape is not None:
                calls[item.item_id] = NewNode(
                    0,
                    record_link_scope(metadata),
                    item.item_id,
                    payload,
                    key,
                    shape,
                    node_call_json(payload, shape),
                )
            values[item.item_id] = {
                "owner": owner,
                "item_id": item.item_id,
                "kind": metadata["kind"],
                "digest": digest,
                "metadata_json": canonical_json(metadata),
                "content_json": None if content is None or place is not None else canonical_json(content),
                "message_field": None if shape is None else shape.message_field,
                "request_type": (
                    content["request_type"] if content is not None and metadata["kind"] == "record" else None
                ),
                "segment_id": None if place is None else place.segment_id,
                "segment_offset": None if place is None else place.offset,
                "segment_length": None if place is None else place.length,
            }
        pending = PendingCreate(prepared, values, calls, None if origin is None else (origin, str(generation)))
        group = (owner, frozenset(scenarios_of(metadata for _, metadata, _ in prepared)))
        with self.group_lock:
            pending.lead = group not in self.waiting
            self.waiting.setdefault(group, []).append(pending)
        if not pending.lead:
            pending.done.wait()
        if pending.lead:
            self.commit_group(group)
        if isinstance(pending.outcome, Exception):
            raise pending.outcome
        if pending.outcome is None:
            raise ReefError("the transaction of this create stopped before it wrote the items")
        return pending.outcome

    def commit_group(self, group: tuple[str, frozenset[str]]) -> None:
        """Write the first waiting creates of ``group``, up to ``GROUP_COMMIT_ITEMS`` items, in one transaction, and
        give the lead to the next waiting create. When the transaction fails, write each create alone, so that one
        create's ``ItemConflictError`` does not refuse the others."""
        owner = group[0]
        with self.group_lock:
            waiting = self.waiting[group]
            batch = [waiting.pop(0)]
            count = len(batch[0].prepared)
            while waiting and count + len(waiting[0].prepared) <= GROUP_COMMIT_ITEMS:
                count += len(waiting[0].prepared)
                batch.append(waiting.pop(0))
        try:
            try:
                for pending, outcome in zip(batch, self.commit_creates(owner, batch), strict=True):
                    pending.outcome = outcome
            except Exception as error:
                if len(batch) == 1:
                    batch[0].outcome = error
                else:
                    for pending in batch:
                        try:
                            pending.outcome = self.commit_creates(owner, [pending])[0]
                        except Exception as alone:
                            pending.outcome = alone
        finally:
            with self.group_lock:
                if waiting:
                    waiting[0].lead = True
                    waiting[0].done.set()
                else:
                    self.waiting.pop(group)
            for pending in batch:
                pending.done.set()

    def commit_creates(self, owner: str, batch: Sequence[PendingCreate]) -> list[tuple[StoredItem, ...]]:
        """Write the rows of the creates ``batch`` of ``owner`` in one transaction, and return the items of each.

        An ID in two creates gets the row of the first; another digest in the later one is an ``ItemConflictError``.
        """
        prepared = [entry for pending in batch for entry in pending.prepared]
        values: dict[str, dict[str, Any]] = {}
        calls: dict[str, NewNode] = {}
        first: dict[str, PendingCreate] = {}
        for pending in batch:
            for item_id, value in pending.values.items():
                values.setdefault(item_id, value)
                first.setdefault(item_id, pending)
            for item_id, call in pending.calls.items():
                calls.setdefault(item_id, call)
        with self.transaction() as connection:
            holds = self.lock_uses(connection, owner, [metadata for _, metadata, _ in prepared])
            # The archive ends a generation under the same lock, so no write of an ended generation commits.
            checked = {pending.generation for pending in batch if pending.generation is not None}
            current = {origin: self.current_generation(connection, owner, origin) for origin, _ in checked}
            for origin, generation in checked:
                if current[origin] != generation:
                    raise ScenarioArchivedError(f"generation {generation!r} of {origin!r} is no longer current")
            sequences = self.write_new_items(connection, list(values.values()))
            metadata_of = {entry[0].item_id: entry[1] for entry in reversed(prepared)}
            inserted = {item_id: (sequence, metadata_of[item_id]) for item_id, sequence in sequences.items()}
            self.write_uses(connection, owner, inserted, holds)
            self.insert_sets(connection, owner, inserted)
            self.trajectory_nodes(owner).append_many(
                connection,
                [
                    replace(calls[item_id], node_sequence=sequence)
                    for item_id, sequence in sorted(sequences.items(), key=lambda pair: pair[1])
                    if item_id in calls
                ],
            )
            # An ID that was stored before, or that a concurrent create stored first, keeps its row; another digest
            # there is a conflict, and the rollback removes the rows that this call inserted. A row of this call has
            # the digest of its first item with that ID.
            kept = [item_id for item_id in values if item_id not in sequences]
            stored_rows = self.rows(connection, owner, kept) if kept else {}
            for item_id, row in stored_rows.items():
                if row["digest"] != values[item_id]["digest"]:
                    raise ItemConflictError(f"item {item_id!r} of {owner!r} is stored with other content")
            stored = {
                item.item_id: item
                for item in self.stored_items(connection, owner, list(stored_rows.values()), content=False)
            }
            for item, _, digest in prepared:
                if item.item_id in sequences and digest != values[item.item_id]["digest"]:
                    raise ItemConflictError(f"item {item.item_id!r} of {owner!r} is stored with other content")
            return [
                tuple(
                    (
                        stored[item.item_id]
                        if item.item_id in stored
                        else StoredItem(
                            sequences[item.item_id],
                            owner,
                            item.item_id,
                            metadata_of[item.item_id],
                            values[item.item_id]["digest"],
                            None,
                            inserted=first[item.item_id] is pending,
                        )
                    )
                    for item, _, _ in pending.prepared
                )
                for pending in batch
            ]

    def read(self, owner: str, item_id: str) -> StoredItem | None:
        found = self.read_many(owner, [item_id])
        return found[0] if found else None

    def read_many(self, owner: str, item_ids: Sequence[str], *, content: bool = True) -> tuple[StoredItem, ...]:
        check_owner(owner)
        unique_ids = list(dict.fromkeys(item_ids))
        with self.transaction(read_only=True) as connection:
            # One query for the rows, and one segment read for the content of all of them.
            rows = self.rows(connection, owner, unique_ids)
            found = [rows[item_id] for item_id in unique_ids if item_id in rows]
            return self.stored_items(connection, owner, found, content=content)

    def query(
        self,
        owner: str,
        *,
        kind: ItemKind | None = None,
        scenario: str | None = None,
        origin: str | None = None,
        request_type: str | None = None,
        after: int = 0,
        limit: int = MAX_QUERY_LIMIT,
        content: bool = True,
        group: tuple[str, str | None] | None = None,
        benchmark: str | None = None,
        trajectory_set: str | None = None,
        id_prefix: str | None = None,
        before: int | None = None,
    ) -> tuple[StoredItem, ...]:
        check_query(
            owner,
            kind,
            scenario,
            after,
            limit,
            origin=origin,
            request_type=request_type,
            group=group,
            benchmark=benchmark,
            trajectory_set=trajectory_set,
            id_prefix=id_prefix,
            before=before,
        )
        items = self.items_table
        # Without content, read only the columns of the metadata: content_json can be large.
        columns = (
            (items,)
            if content
            else (items.c.sequence, items.c.owner, items.c.item_id, items.c.metadata_json, items.c.digest)
        )
        source, sequence, conditions = self.filtered(owner, kind, scenario, origin, request_type, after)
        statement = select(*columns).select_from(source).where(*conditions)
        if group is not None:
            value = self.metadata_string(group[0])
            statement = statement.where(value.is_(None) if group[1] is None else value == group[1])
            if group[0] == "evaluation_id" and group[1] is not None:
                # The same condition in the form of the index data_item_by_evaluation_sequence, which finds the rows.
                statement = statement.where(self.evaluation_id() == group[1])
        for set_kind, name in (("benchmark", benchmark), ("trajectory_set", trajectory_set)):
            if name is not None:
                member = self.item_sets_table.alias(f"{set_kind}_member")
                statement = statement.join(
                    member,
                    and_(
                        member.c.owner == items.c.owner,
                        member.c.sequence == items.c.sequence,
                        member.c.kind == set_kind,
                        member.c.name == name,
                    ),
                )
        if id_prefix is not None:
            statement = statement.where(self.has_id_prefix(id_prefix))
        # A page before a sequence reads backward from it, newest first. ``sequence`` is the column of the item
        # sequence that ``filtered`` reads in order: the use's for a scenario, else the item's.
        order = sequence if before is None else sequence.desc()
        if before is not None:
            statement = statement.where(sequence < before)
        with self.transaction(read_only=True) as connection:
            found = connection.execute(statement.order_by(order).limit(limit)).mappings().all()
            return self.stored_items(connection, owner, found, content=content)

    def count(
        self,
        owner: str,
        *,
        kind: ItemKind | None = None,
        scenario: str | None = None,
        origin: str | None = None,
        request_type: str | None = None,
        after: int = 0,
    ) -> int:
        check_filters(owner, kind, scenario, origin, request_type, after)
        source, _, conditions = self.filtered(owner, kind, scenario, origin, request_type, after)
        statement = select(func.count()).select_from(source).where(*conditions)
        with self.transaction(read_only=True) as connection:
            return int(connection.execute(statement).scalar_one())

    def last_sequence(
        self,
        owner: str,
        *,
        kind: ItemKind | None = None,
        scenario: str | None = None,
        origin: str | None = None,
        request_type: str | None = None,
    ) -> int:
        check_filters(owner, kind, scenario, origin, request_type, 0)
        source, sequence, conditions = self.filtered(owner, kind, scenario, origin, request_type, 0)
        # The largest sequence as the first row in descending order: the database reads an index backwards from the
        # newest row and stops at the first row that the conditions keep. A max() over a join reads every row.
        statement = select(sequence).select_from(source).where(*conditions).order_by(sequence.desc()).limit(1)
        with self.transaction(read_only=True) as connection:
            return int(connection.execute(statement).scalar() or 0)

    def benchmark_item_ids(self, owner: str, benchmarks: Sequence[str]) -> dict[str, set[str]]:
        check_owner(owner)
        found: dict[str, set[str]] = {benchmark: set() for benchmark in benchmarks}
        if not found:
            return found
        items, members = self.items_table, self.item_sets_table
        statement = (
            select(members.c.name, items.c.item_id)
            .join(items, and_(items.c.owner == members.c.owner, items.c.sequence == members.c.sequence))
            .where(
                members.c.owner == owner,
                members.c.kind == "benchmark",
                members.c.name.in_(found),
                items.c.kind == "task",
            )
        )
        with self.transaction(read_only=True) as connection:
            for benchmark, item_id in connection.execute(statement):
                found[benchmark].add(item_id)
        return found

    def collections(self, owner: str) -> tuple[TaskCollection, ...]:
        check_owner(owner)
        table = self.collections_table
        statement = select(table.c.collection_json).where(table.c.owner == owner).order_by(table.c.name)
        with self.transaction(read_only=True) as connection:
            return tuple(
                TaskCollection.from_json(json.loads(value)) for value in connection.execute(statement).scalars()
            )

    def write_collection(self, owner: str, collection: TaskCollection) -> None:
        check_owner(owner)
        table = self.collections_table
        with self.transaction() as connection:
            connection.execute(delete(table).where(table.c.owner == owner, table.c.name == collection.name))
            connection.execute(
                insert(table).values(
                    owner=owner, name=collection.name, collection_json=canonical_json(collection.to_json())
                )
            )

    def delete_collection(self, owner: str, name: str) -> bool:
        check_owner(owner)
        table = self.collections_table
        with self.transaction() as connection:
            removed = connection.execute(delete(table).where(table.c.owner == owner, table.c.name == name))
            return removed.rowcount > 0

    def filtered(
        self,
        owner: str,
        kind: ItemKind | None,
        scenario: str | None,
        origin: str | None,
        request_type: str | None,
        after: int,
    ) -> tuple[FromClause, Column[int], list[ColumnElement[bool]]]:
        """The rows that the filters of ``query`` give: a FROM clause, its column of the item sequence, and the
        conditions on its rows. The ID of a record item starts with the scenario that made it (``record_item``).

        Without a scenario, the FROM clause is ``data_item``. With a scenario, it is ``data_item`` joined with the
        first use of each item by the scenario on ``item_sequence``. The conditions on the sequence and the ID are
        on the use, and the ID condition has a form that no index serves (``use_has_id_prefix``). So the database
        reads the uses of the scenario in item order from the index ``data_item_use_by_item``, also in a prepared
        statement, and stops at the limit of a page. It never reads the items of the owner that the scenario does not use, such as
        the evaluation calls. Each item gives one row: an item that the scenario uses through several sets or
        splits gives the row of its first use.
        """
        items, uses = self.items_table, self.uses_table
        conditions = [items.c.owner == owner]
        if kind is not None:
            conditions.append(items.c.kind == kind)
        if request_type is not None:
            conditions.append(items.c.request_type == request_type)
        if scenario is None:
            source: FromClause = items
            sequence = items.c.sequence
            if origin is not None:
                conditions.append(self.has_id_prefix(f"{origin}/"))
        else:
            source = uses.join(items, items.c.sequence == uses.c.item_sequence)
            sequence = uses.c.item_sequence
            earlier = uses.alias("earlier_use")
            first_use = ~exists().where(
                earlier.c.owner == uses.c.owner,
                earlier.c.scenario == uses.c.scenario,
                earlier.c.item_sequence == uses.c.item_sequence,
                earlier.c.sequence < uses.c.sequence,
            )
            conditions += [uses.c.owner == owner, uses.c.scenario == scenario, first_use]
            if origin is not None:
                conditions.append(self.use_has_id_prefix(f"{origin}/"))
        conditions.append(sequence > after)
        return source, sequence, conditions

    def used_by(self, owner: str, items: Table, scenario: str | None) -> tuple[ColumnElement[bool], ...]:
        """The condition that ``scenario`` uses the item of the row of ``items``, or none without a scenario."""
        if scenario is None:
            return ()
        uses = self.uses_table
        return (exists().where(uses.c.owner == owner, uses.c.item_id == items.c.item_id, uses.c.scenario == scenario),)

    def scenario_items(
        self,
        owner: str,
        scenario: str,
        *,
        kinds: Sequence[str],
        after: int = 0,
        limit: int = MAX_QUERY_LIMIT,
        content: bool = True,
        excluding_origin: str | None = None,
    ) -> tuple[ItemUse, ...]:
        check_query(owner, None, scenario, after, limit)
        items, uses = self.items_table, self.uses_table
        statement = (
            select(
                items,
                uses.c.sequence.label("use_sequence"),
                uses.c.split.label("use_split"),
                uses.c.held_set.label("use_held_set"),
            )
            .join(uses, (uses.c.owner == items.c.owner) & (uses.c.item_id == items.c.item_id))
            .where(
                uses.c.owner == owner, uses.c.scenario == scenario, uses.c.sequence > after, items.c.kind.in_(kinds)
            )
            .order_by(uses.c.sequence)
            .limit(limit)
        )
        if excluding_origin is not None:
            statement = statement.where(~uses.c.item_id.startswith(f"{excluding_origin}/", autoescape=True))
        with self.transaction(read_only=True) as connection:
            found = connection.execute(statement).mappings().all()
            stored = self.stored_items(connection, owner, found, content=content)
        return tuple(
            ItemUse(int(row["use_sequence"]), item, row["use_split"], row["use_held_set"] or None)
            for row, item in zip(found, stored, strict=True)
        )

    def item_splits(self, owner: str, scenario: str, item_ids: Sequence[str]) -> dict[str, frozenset[str]]:
        check_owner(owner)
        uses = self.uses_table
        found: dict[str, set[str]] = {}
        with self.transaction(read_only=True) as connection:
            for batch in batched(dict.fromkeys(item_ids), MAX_QUERY_LIMIT):
                statement = select(uses.c.item_id, uses.c.split).where(
                    uses.c.owner == owner, uses.c.scenario == scenario, uses.c.item_id.in_(batch)
                )
                for row in connection.execute(statement):
                    found.setdefault(row.item_id, set()).add(row.split)
        return {item_id: frozenset(splits) for item_id, splits in found.items()}

    def set_count(self, connection: Connection, owner: str, kind: str, name: str) -> int:
        """The number of items of ``owner`` in the set ``name`` of ``kind``."""
        return self.set_counts(connection, owner, kind, [name]).get(name, 0)

    def set_counts(self, connection: Connection, owner: str, kind: str, names: Iterable[str]) -> dict[str, int]:
        """The number of items of ``owner`` in each set of ``names`` of ``kind``, by name. A set with no item has no
        entry. One query reads the rows of ``data_item_set`` of the sets, not the items."""
        wanted = sorted(set(names))
        if not wanted:
            return {}
        item_sets = self.item_sets_table
        statement = (
            select(item_sets.c.name, func.count())
            .where(item_sets.c.owner == owner, item_sets.c.kind == kind, item_sets.c.name.in_(wanted))
            .group_by(item_sets.c.name)
        )
        return {str(row[0]): int(row[1]) for row in connection.execute(statement)}

    def hold(self, owner: str, scenario: str, kind: str, name: str, split: str) -> ScenarioHold:
        check_owner(owner)
        check_hold(scenario, kind, name, split)
        uses, holds = self.uses_table, self.holds_table
        with self.transaction() as connection:
            # No item joins or leaves the set until this commits, and each new use gets its sequence under the
            # scenario's lock (RFC #1219).
            self.lock_scenarios(connection, owner, [HOLDS_LOCK])
            self.lock_scenarios(connection, owner, [scenario])
            held = connection.execute(
                select(holds.c.split).where(
                    holds.c.owner == owner,
                    holds.c.scenario == scenario,
                    holds.c.kind == kind,
                    holds.c.name == name,
                    holds.c.split == split,
                )
            ).first()
            if held is None:
                connection.execute(
                    insert(holds).values(owner=owner, scenario=scenario, kind=kind, name=name, split=split)
                )
                # One statement writes the use of each member, in the order of the items.
                item_sets = self.item_sets_table
                begun = (
                    select(
                        literal(owner),
                        item_sets.c.item_id,
                        literal(scenario),
                        literal(name),
                        literal(split),
                        item_sets.c.sequence,
                    )
                    .where(item_sets.c.owner == owner, item_sets.c.kind == kind, item_sets.c.name == name)
                    .order_by(item_sets.c.sequence)
                )
                connection.execute(
                    insert(uses).from_select(
                        ["owner", "item_id", "scenario", "held_set", "split", "item_sequence"], begun
                    )
                )
            return ScenarioHold(scenario, kind, name, split, self.set_count(connection, owner, kind, name))

    def release(self, owner: str, scenario: str, kind: str, name: str, split: str) -> bool:
        check_owner(owner)
        check_hold(scenario, kind, name, split)
        items, uses, holds = self.items_table, self.uses_table, self.holds_table
        with self.transaction() as connection:
            self.lock_scenarios(connection, owner, [HOLDS_LOCK])
            self.lock_scenarios(connection, owner, [scenario])
            ended = connection.execute(
                delete(holds).where(
                    holds.c.owner == owner,
                    holds.c.scenario == scenario,
                    holds.c.kind == kind,
                    holds.c.name == name,
                    holds.c.split == split,
                )
            ).rowcount
            # A benchmark and a trajectory set can have one name: only the items of this kind lose their use.
            members = select(items.c.item_id).where(items.c.owner == owner, items.c.kind == HOLD_ITEM_KINDS[kind])
            connection.execute(
                delete(uses).where(
                    uses.c.owner == owner,
                    uses.c.scenario == scenario,
                    uses.c.held_set == name,
                    uses.c.split == split,
                    uses.c.item_id.in_(members),
                )
            )
            return ended > 0

    def holds(self, owner: str, scenario: str) -> tuple[ScenarioHold, ...]:
        check_owner(owner)
        holds = self.holds_table
        statement = (
            select(holds.c.kind, holds.c.name, holds.c.split)
            .where(holds.c.owner == owner, holds.c.scenario == scenario)
            .order_by(holds.c.kind, holds.c.name, holds.c.split)
        )
        with self.transaction(read_only=True) as connection:
            rows = connection.execute(statement).all()
            counts = {
                kind: self.set_counts(connection, owner, kind, [row.name for row in rows if row.kind == kind])
                for kind in {row.kind for row in rows}
            }
        return tuple(
            ScenarioHold(scenario, row.kind, row.name, row.split, counts[row.kind].get(row.name, 0)) for row in rows
        )

    def holding_scenarios(self, owner: str, kind: str, names: Sequence[str]) -> tuple[str, ...]:
        check_owner(owner)
        with self.transaction(read_only=True) as connection:
            holds = self.holds_of(connection, owner, {kind: names})
        return tuple(sorted({scenario for held in holds.values() for scenario, _ in held}))

    def scenarios_with_records(self, owner: str) -> tuple[str, ...]:
        check_owner(owner)
        items, uses = self.items_table, self.uses_table
        statement = (
            select(uses.c.scenario)
            .join(items, (uses.c.owner == items.c.owner) & (uses.c.item_id == items.c.item_id))
            .where(uses.c.owner == owner, items.c.kind == "record")
            .distinct()
            .order_by(uses.c.scenario)
        )
        with self.transaction(read_only=True) as connection:
            return tuple(connection.execute(statement).scalars())

    def last_use_sequence(self, owner: str, scenario: str) -> int:
        check_owner(owner)
        uses = self.uses_table
        statement = select(func.max(uses.c.sequence)).where(uses.c.owner == owner, uses.c.scenario == scenario)
        with self.transaction(read_only=True) as connection:
            return int(connection.execute(statement).scalar() or 0)

    def update(self, owner: str, item_ids: Sequence[str], patch: Mapping[str, Any]) -> tuple[StoredItem, ...]:
        check_owner(owner)
        unique_ids = list(dict.fromkeys(item_ids))
        with self.transaction() as connection:
            # A new use gets its sequence here, so that a reader's cursor of use sequences misses none (RFC #1219).
            scenarios = patch.get("scenarios") if isinstance(patch, Mapping) else None
            marked = set(scenarios) if isinstance(scenarios, Mapping) else set()
            holds: dict[tuple[str, str], set[tuple[str, str]]] | None = None
            if isinstance(patch, Mapping) and any(key in patch for key in SET_KEYS.values()):
                # The uses that the holds give change with the sets of the items. Under ``HOLDS_LOCK`` no other
                # write changes a set of these items or a hold, so the unlocked read below names the scenarios to
                # lock, and one call locks them all.
                self.lock_scenarios(connection, owner, [HOLDS_LOCK])
                before = self.rows(connection, owner, unique_ids)
                sets: dict[str, set[str]] = {}
                for set_kind, key in SET_KEYS.items():
                    named = patch.get(key)
                    sets[set_kind] = {
                        name for name in (named if isinstance(named, list) else []) if isinstance(name, str)
                    }
                for row in before.values():
                    kind, names = held_sets(json.loads(row["metadata_json"])) or ("", ())
                    sets.setdefault(kind, set()).update(names)
                holds = self.holds_of(connection, owner, sets)
                marked.update(scenario for held in holds.values() for scenario, _ in held)
            self.lock_scenarios(connection, owner, marked)
            # A patch merges into the whole metadata, so a concurrent patch must read after this one commits.
            existing = self.rows(connection, owner, unique_ids, lock=True)
            missing = [item_id for item_id in unique_ids if item_id not in existing]
            if missing:
                raise ItemNotFoundError(f"{owner!r} stores no item {missing[0]!r}")
            patched = {
                item_id: patched_metadata(json.loads(existing[item_id]["metadata_json"]), patch)
                for item_id in unique_ids
            }
            items = self.items_table
            connection.execute(
                update(items)
                .where(items.c.owner == owner, items.c.item_id == bindparam("row_item_id"))
                .values(metadata_json=bindparam("metadata_json")),
                [
                    {"row_item_id": item_id, "metadata_json": canonical_json(metadata)}
                    for item_id, metadata in patched.items()
                ],
            )
            changed = {
                item_id: (int(existing[item_id]["sequence"]), metadata) for item_id, metadata in patched.items()
            }
            self.write_uses(connection, owner, changed, holds)
            if holds is not None:
                # The patch named a key of the sets: the rows of the items' sets follow their new metadata.
                item_sets = self.item_sets_table
                connection.execute(
                    delete(item_sets).where(item_sets.c.owner == owner, item_sets.c.item_id.in_(unique_ids))
                )
                self.insert_sets(connection, owner, changed)
            return tuple(
                StoredItem(existing[item_id]["sequence"], owner, item_id, metadata, existing[item_id]["digest"])
                for item_id, metadata in patched.items()
            )

    def scenario_generation(self, owner: str, scenario: str) -> str:
        check_owner(owner)
        check_owner(scenario)
        with self.transaction() as connection:
            self.lock_scenarios(connection, owner, [scenario])
            current = self.current_generation(connection, owner, scenario)
            if current is not None:
                return current
            generation = uuid.uuid4().hex
            connection.execute(
                insert(self.generations_table).values(owner=owner, scenario=scenario, generation=generation)
            )
            return generation

    def prune_records(self, max_bytes: int) -> int:
        items, nodes = self.items_table, self.nodes_table
        # An evaluation call is not pruned, so neither its row nor its node counts. Its node is in a scope of its own
        # (``record_link_scope``), so no node of an inference record needs it. Both counts read the rows of a partial
        # index (``data_item_of_inference``, ``data_record_node_scenario_bytes``).
        with self.transaction(read_only=True) as connection:
            held = connection.execute(
                select(func.coalesce(func.sum(func.length(items.c.content_json)), 0)).where(self.is_inference())
            ).scalar_one()
            held += connection.execute(
                select(func.coalesce(func.sum(nodes.c.body_bytes), 0)).where(node_of_scenario_call(nodes))
            ).scalar_one()
        excess = int(held) - max_bytes
        sizes = EvictionSizes(nodes, ("owner",))
        evicted, after = 0, 0
        while excess > 0:
            with self.transaction(read_only=True) as connection:
                page = (
                    connection.execute(
                        select(
                            items.c.sequence,
                            items.c.owner,
                            items.c.item_id,
                            items.c.message_field,
                            items.c.metadata_json,
                            func.coalesce(func.length(items.c.content_json), 0).label("body_bytes"),
                        )
                        .where(self.is_inference(), items.c.sequence > after)
                        .order_by(items.c.sequence)
                        .limit(PRUNE_PAGE_SIZE)
                    )
                    .mappings()
                    .all()
                )
                sizes.read(connection, page)
            if not page:
                break
            chosen: list[tuple[RowMapping, int]] = []
            for row in page:
                size = sizes.count(row)
                chosen.append((row, size))
                excess -= size
                if excess <= 0:
                    break
            after = int(chosen[-1][0]["sequence"])
            by_owner: dict[str, list[tuple[RowMapping, int]]] = {}
            for row, size in chosen:
                by_owner.setdefault(row["owner"], []).append((row, size))
            for owner, rows in by_owner.items():
                evicted += self.evict_records(owner, rows)
        return evicted

    def evict_records(self, owner: str, rows: Sequence[tuple[RowMapping, int]]) -> int:
        """Delete the record items of ``owner`` in ``rows``, each with the bytes that deleting it frees, and add them
        to the losses of their origins. Return the number that it deleted: another writer can delete one first."""
        items, uses, losses = self.items_table, self.uses_table, self.losses_table
        metadata = {row["item_id"]: json.loads(row["metadata_json"]) for row, _ in rows}
        with self.transaction() as connection:
            self.lock_scenarios(connection, owner, scenarios_of(metadata.values()))
            present = set(
                connection.execute(
                    select(items.c.item_id).where(items.c.owner == owner, items.c.item_id.in_(list(metadata)))
                ).scalars()
            )
            kept = [(row, size) for row, size in rows if row["item_id"] in present]
            if not kept:
                return 0
            ids = [row["item_id"] for row, _ in kept]
            connection.execute(delete(uses).where(uses.c.owner == owner, uses.c.item_id.in_(ids)))
            connection.execute(delete(items).where(items.c.owner == owner, items.c.item_id.in_(ids)))
            self.trajectory_nodes(owner).release(
                connection, [int(row["sequence"]) for row, _ in kept if row["message_field"] is not None]
            )
            by_origin: dict[str, list[tuple[RowMapping, int]]] = {}
            for row, size in kept:
                by_origin.setdefault(metadata[row["item_id"]]["origin"]["scenario"], []).append((row, size))
            for scenario, members in by_origin.items():
                sequences = [int(row["sequence"]) for row, _ in members]
                added = RecordLoss(len(members), sum(size for _, size in members), min(sequences), max(sequences))
                mine = (losses.c.owner == owner, losses.c.scenario == scenario)
                previous = connection.execute(select(losses).where(*mine)).mappings().first()
                if previous is None:
                    connection.execute(
                        insert(losses).values(
                            owner=owner,
                            scenario=scenario,
                            record_count=added.record_count,
                            body_bytes=added.body_bytes,
                            first_sequence=added.first_sequence,
                            last_sequence=added.last_sequence,
                        )
                    )
                else:
                    connection.execute(
                        update(losses)
                        .where(*mine)
                        .values(
                            record_count=losses.c.record_count + added.record_count,
                            body_bytes=losses.c.body_bytes + added.body_bytes,
                            first_sequence=min(int(previous["first_sequence"]), added.first_sequence),
                            last_sequence=max(int(previous["last_sequence"]), added.last_sequence),
                        )
                    )
            return len(kept)

    def record_loss(self, owner: str, scenario: str) -> RecordLoss:
        check_owner(owner)
        losses = self.losses_table
        with self.transaction(read_only=True) as connection:
            row = (
                connection.execute(select(losses).where(losses.c.owner == owner, losses.c.scenario == scenario))
                .mappings()
                .first()
            )
        if row is None:
            return RecordLoss()
        return RecordLoss(
            *(int(row[name]) for name in ("record_count", "body_bytes", "first_sequence", "last_sequence"))
        )

    def current_generation(self, connection: Connection, owner: str, scenario: str) -> str | None:
        """The current generation of ``scenario`` of ``owner``, or None when it has none."""
        generations = self.generations_table
        return connection.execute(
            select(generations.c.generation).where(generations.c.owner == owner, generations.c.scenario == scenario)
        ).scalar_one_or_none()

    def delete_scenario(self, owner: str, scenario: str) -> int:
        check_owner(owner)
        items, uses = self.items_table, self.uses_table
        with self.transaction() as connection:
            # No write of an item's benchmarks gives the scenario a use while its holds end.
            self.lock_scenarios(connection, owner, [HOLDS_LOCK])
            self.lock_scenarios(connection, owner, [scenario])
            generations = self.generations_table
            connection.execute(
                delete(generations).where(generations.c.owner == owner, generations.c.scenario == scenario)
            )
            holds = self.holds_table
            connection.execute(delete(holds).where(holds.c.owner == owner, holds.c.scenario == scenario))
            # The rows stay locked, so that a concurrent patch of another scenario reads after this one commits.
            rows = connection.execute(
                select(items.c.item_id, items.c.metadata_json)
                .where(
                    items.c.owner == owner,
                    exists().where(
                        uses.c.owner == owner,
                        uses.c.item_id == items.c.item_id,
                        uses.c.scenario == scenario,
                        uses.c.held_set == "",
                    ),
                )
                .order_by(items.c.item_id)
                .with_for_update(of=items)
            ).all()
            connection.execute(delete(uses).where(uses.c.owner == owner, uses.c.scenario == scenario))
            deleted = self.deleted_table
            # A scenario name is in the text of each metadata that names it, so the condition reads fewer rows.
            named = connection.execute(
                select(deleted.c.item_id, deleted.c.metadata_json).where(
                    deleted.c.owner == owner, deleted.c.metadata_json.contains(json.dumps(scenario), autoescape=True)
                )
            ).all()
            rewritten = []
            for item_id, text in named:
                metadata = json.loads(text)
                if metadata["scenarios"].pop(scenario, None) is not None:
                    rewritten.append({"row_item_id": item_id, "metadata_json": canonical_json(metadata)})
            if rewritten:
                connection.execute(
                    update(deleted)
                    .where(deleted.c.owner == owner, deleted.c.item_id == bindparam("row_item_id"))
                    .values(metadata_json=bindparam("metadata_json")),
                    rewritten,
                )
            if not rows:
                return 0
            values = []
            for item_id, text in rows:
                metadata = json.loads(text)
                metadata["scenarios"].pop(scenario, None)
                values.append({"row_owner": owner, "row_item_id": item_id, "metadata_json": canonical_json(metadata)})
            connection.execute(
                update(items)
                .where(items.c.owner == bindparam("row_owner"), items.c.item_id == bindparam("row_item_id"))
                .values(metadata_json=bindparam("metadata_json")),
                values,
            )
            return len(rows)

    def replace_task(self, owner: str, item_id: str, files: Path | TaskFilesReference) -> StoredItem:
        check_owner(owner)
        with self.transaction() as connection:
            row = self.rows(connection, owner, [item_id]).get(item_id)
            if row is None:
                raise ItemNotFoundError(f"{owner!r} stores no item {item_id!r}")
            metadata = json.loads(row["metadata_json"])
            # The locks order this write with the other writes of the item's scenarios, as an update does.
            self.lock_uses(connection, owner, [metadata])
            row = self.rows(connection, owner, [item_id], lock=True)[item_id]
            stored = StoredItem(row["sequence"], owner, item_id, metadata, row["digest"])
            digest = replaced_task_digest(owner, stored, files, self.task_files)
            if digest == stored.digest:
                return stored
            items = self.items_table
            connection.execute(
                update(items).where(items.c.owner == owner, items.c.item_id == item_id).values(digest=digest)
            )
            self.begin_uses_again(connection, owner, stored)
            return replace(stored, digest=digest)

    def replace_trajectory(self, owner: str, item_id: str, content: Mapping[str, Any]) -> StoredItem:
        check_owner(owner)
        with self.transaction() as connection:
            row = self.rows(connection, owner, [item_id]).get(item_id)
        if row is None:
            raise ItemNotFoundError(f"{owner!r} stores no item {item_id!r}")
        found = StoredItem(row["sequence"], owner, item_id, json.loads(row["metadata_json"]), row["digest"])
        digest = replaced_trajectory_digest(found, content, self.task_files)
        if digest == found.digest:
            return found
        text = canonical_json(content)
        # Large content goes to a new segment before the transaction, as a create's does; ``maintain`` deletes the
        # segment when no row comes to use it, and the old segment line when no row uses its segment any more.
        partition = self.partition(owner)
        place = None
        if partition is not None and len(text.encode("utf-8")) >= CREATE_SEGMENT_BYTES:
            (place,) = self.write_segment(partition, [text])
        with self.transaction() as connection:
            # The locks order this write with the other writes of the item's scenarios, as an update does.
            self.lock_uses(connection, owner, [found.metadata])
            row = self.rows(connection, owner, [item_id], lock=True).get(item_id)
            if row is None:
                raise ItemNotFoundError(f"{owner!r} stores no item {item_id!r}")
            stored = StoredItem(row["sequence"], owner, item_id, json.loads(row["metadata_json"]), row["digest"])
            if digest == stored.digest:
                return stored
            items = self.items_table
            connection.execute(
                update(items)
                .where(items.c.owner == owner, items.c.item_id == item_id)
                .values(
                    digest=digest,
                    content_json=text if place is None else None,
                    segment_id=None if place is None else place.segment_id,
                    segment_offset=None if place is None else place.offset,
                    segment_length=None if place is None else place.length,
                )
            )
            self.begin_uses_again(connection, owner, stored)
            return replace(stored, digest=digest)

    def begin_uses_again(self, connection: Connection, owner: str, stored: StoredItem) -> None:
        """Give each use of the item ``stored`` a new use sequence, in the order of the uses, so that a trainer that
        pages by use sequence reads the item again after a replace."""
        uses = self.uses_table
        where = (uses.c.owner == owner, uses.c.item_id == stored.item_id)
        begun = [
            {
                "owner": owner,
                "item_id": stored.item_id,
                "scenario": use.scenario,
                "held_set": use.held_set,
                "split": use.split,
                "item_sequence": stored.sequence,
            }
            for use in connection.execute(
                select(uses.c.scenario, uses.c.held_set, uses.c.split).where(*where).order_by(uses.c.sequence)
            )
        ]
        connection.execute(delete(uses).where(*where))
        if begun:
            connection.execute(insert(uses), begun)

    def delete(self, owner: str, item_id: str) -> bool:
        check_owner(owner)
        items, deleted = self.items_table, self.deleted_table
        with self.transaction() as connection:
            row = self.rows(connection, owner, [item_id]).get(item_id)
            if row is None:
                return False
            # A deleted item with the same ID goes for good, and its record's node is released under its lock too.
            earlier = list(self.deleted_rows(connection, owner, [item_id]).values())
            self.lock_uses(connection, owner, [json.loads(found["metadata_json"]) for found in [row, *earlier]])
            self.purge_rows(connection, owner, earlier)
            connection.execute(
                delete(self.uses_table).where(self.uses_table.c.owner == owner, self.uses_table.c.item_id == item_id)
            )
            connection.execute(
                delete(self.item_sets_table).where(
                    self.item_sets_table.c.owner == owner, self.item_sets_table.c.item_id == item_id
                )
            )
            names = [column.name for column in items.columns]
            connection.execute(
                insert(deleted).from_select(
                    [*names, "deleted_at"],
                    select(*items.columns, literal(time.time(), Float)).where(
                        items.c.owner == owner, items.c.item_id == item_id
                    ),
                )
            )
            connection.execute(delete(items).where(items.c.owner == owner, items.c.item_id == item_id))
        return True

    def deleted_rows(self, connection: Connection, owner: str, item_ids: Sequence[str]) -> dict[str, RowMapping]:
        """The rows of ``data_deleted_item`` of ``item_ids`` by item ID."""
        deleted = self.deleted_table
        found: dict[str, RowMapping] = {}
        for batch in batches(list(dict.fromkeys(item_ids))):
            statement = select(deleted).where(deleted.c.owner == owner, deleted.c.item_id.in_(batch))
            found.update({row["item_id"]: row for row in connection.execute(statement).mappings()})
        return found

    def purge_rows(self, connection: Connection, owner: str, rows: Sequence[RowMapping]) -> None:
        """Delete the rows ``rows`` of ``data_deleted_item`` of ``owner`` and release their trajectory nodes. The
        caller holds the locks of the scenarios of their metadata (``scenarios_of``)."""
        deleted = self.deleted_table
        for batch in batches([row["item_id"] for row in rows]):
            connection.execute(delete(deleted).where(deleted.c.owner == owner, deleted.c.item_id.in_(batch)))
        self.trajectory_nodes(owner).release(
            connection, [int(row["sequence"]) for row in rows if row["message_field"] is not None]
        )

    def deleted_items(
        self, owner: str, *, kind: ItemKind | None = None, before: int | None = None, limit: int = MAX_QUERY_LIMIT
    ) -> tuple[DeletedItem, ...]:
        check_deleted_listing(owner, kind, before, limit)
        deleted = self.deleted_table
        statement = select(
            deleted.c.sequence,
            deleted.c.owner,
            deleted.c.item_id,
            deleted.c.metadata_json,
            deleted.c.digest,
            deleted.c.deleted_at,
        ).where(deleted.c.owner == owner)
        if kind is not None:
            statement = statement.where(deleted.c.kind == kind)
        if before is not None:
            statement = statement.where(deleted.c.sequence < before)
        with self.transaction(read_only=True) as connection:
            rows = connection.execute(statement.order_by(deleted.c.sequence.desc()).limit(limit)).mappings().all()
        return tuple(deleted_item(row) for row in rows)

    def read_deleted(self, owner: str, item_ids: Sequence[str]) -> tuple[DeletedItem, ...]:
        check_owner(owner)
        with self.transaction(read_only=True) as connection:
            rows = self.deleted_rows(connection, owner, item_ids)
        return tuple(deleted_item(rows[item_id]) for item_id in dict.fromkeys(item_ids) if item_id in rows)

    def restore(self, owner: str, item_ids: Sequence[str]) -> tuple[StoredItem, ...]:
        check_owner(owner)
        wanted = list(dict.fromkeys(item_ids))
        if not wanted:
            return ()
        items, deleted = self.items_table, self.deleted_table
        with self.transaction() as connection:
            rows = self.deleted_rows(connection, owner, wanted)
            missing = [item_id for item_id in wanted if item_id not in rows]
            if missing:
                raise ItemNotFoundError(f"{owner!r} has no deleted item {missing[0]!r}")
            metadata = {item_id: json.loads(rows[item_id]["metadata_json"]) for item_id in wanted}
            holds = self.lock_uses(connection, owner, metadata.values())
            # The archive of a scenario rewrites the metadata of deleted items under the scenario's lock.
            if {
                item_id: json.loads(row["metadata_json"])
                for item_id, row in self.deleted_rows(connection, owner, wanted).items()
            } != metadata:
                raise ItemConflictError("a deleted item changed while it was being restored; try again")
            taken = [item_id for batch in batches(wanted) for item_id in self.rows(connection, owner, batch)]
            if taken:
                raise ItemConflictError(f"{owner!r} stores an item {taken[0]!r}, so its deleted item cannot come back")
            names = [column.name for column in items.columns]
            for batch in batches(wanted):
                chosen = (deleted.c.owner == owner, deleted.c.item_id.in_(batch))
                connection.execute(
                    insert(items).from_select(names, select(*(deleted.c[name] for name in names)).where(*chosen))
                )
                connection.execute(delete(deleted).where(*chosen))
            restored = {item_id: (int(rows[item_id]["sequence"]), metadata[item_id]) for item_id in wanted}
            self.write_uses(connection, owner, restored, holds)
            self.insert_sets(connection, owner, restored)
        return tuple(deleted_item(rows[item_id]).item for item_id in wanted)

    def purge(self, owner: str, item_id: str) -> bool:
        check_owner(owner)
        with self.transaction() as connection:
            row = self.deleted_rows(connection, owner, [item_id]).get(item_id)
            if row is None:
                return False
            self.lock_scenarios(connection, owner, scenarios_of([json.loads(row["metadata_json"])]))
            self.purge_rows(connection, owner, [row])
        return True

    def purge_deleted(self, time: float) -> int:
        deleted = self.deleted_table
        with self.transaction(read_only=True) as connection:
            found = connection.execute(
                select(deleted.c.owner, deleted.c.item_id).where(deleted.c.deleted_at < time).order_by(deleted.c.owner)
            ).all()
        by_owner: dict[str, list[str]] = {}
        for owner, item_id in found:
            by_owner.setdefault(owner, []).append(item_id)
        purged = 0
        for owner, item_ids in by_owner.items():
            for batch in batches(item_ids):
                with self.transaction() as connection:
                    # A restore or a purge can take a row first.
                    rows = [
                        row for row in self.deleted_rows(connection, owner, batch).values() if row["deleted_at"] < time
                    ]
                    self.lock_scenarios(
                        connection, owner, scenarios_of(json.loads(row["metadata_json"]) for row in rows)
                    )
                    self.purge_rows(connection, owner, rows)
                    purged += len(rows)
        return purged

    def waiting_content(self) -> tuple[tuple[Table, ColumnElement[int], ColumnElement[bool]], ...]:
        """For the items and for the nodes: the table, the content bytes of a row, and the condition of the rows
        whose content the database holds and a segment can hold."""
        items, nodes = self.items_table, self.nodes_table
        return (
            (
                items,
                func.length(items.c.content_json),
                and_(items.c.kind == "trajectory", items.c.content_json.is_not(None)),
            ),
            (nodes, nodes.c.body_bytes, nodes.c.entries_json.is_not(None)),
        )

    def pack(self, owner: str) -> int:
        """Move content of ``owner`` from the database into segments, oldest first, while at least ``PACK_BYTES`` of
        it waits, and return the number of rows moved. Each segment holds at least ``PACK_BYTES``.

        The caller holds ``pack_lock(owner)``, so no other call moves the same rows at once.
        """
        partition = self.partition(owner)
        if partition is None:
            return 0
        moved = 0
        for table, size, waiting in self.waiting_content():
            is_items = table is self.items_table
            key = table.c.sequence if is_items else table.c.node_sequence
            texts = (
                (table.c.content_json,) if is_items else (table.c.entries_json, table.c.shape_json, table.c.call_json)
            )
            condition = and_(table.c.owner == owner, waiting)
            while True:
                rows: list[RowMapping] = []
                total = 0
                with self.transaction() as connection:
                    if (connection.scalar(select(func.sum(size)).where(condition)) or 0) < PACK_BYTES:
                        break
                    result = connection.execute(
                        select(key.label("key"), size.label("size"), *texts).where(condition).order_by(key)
                    ).mappings()
                    for row in result:
                        rows.append(row)
                        total += int(row["size"])
                        if total >= PACK_BYTES:
                            break
                    result.close()
                lines = [
                    (
                        row["content_json"]
                        if is_items
                        else node_line(row["entries_json"], row["shape_json"], row["call_json"])
                    )
                    for row in rows
                ]
                places = self.write_segment(partition, lines)
                cleared = (
                    {"content_json": None}
                    if is_items
                    else {"entries_json": None, "shape_json": None, "call_json": None}
                )
                with self.transaction() as connection:
                    connection.execute(
                        update(table)
                        .where(condition, key == bindparam("row_key"))
                        .values(
                            **cleared,
                            **{name: bindparam(f"new_{name}") for name in SEGMENT_COLUMNS},
                        ),
                        [
                            {
                                "row_key": row["key"],
                                "new_segment_id": place.segment_id,
                                "new_segment_offset": place.offset,
                                "new_segment_length": place.length,
                            }
                            for row, place in zip(rows, places, strict=True)
                        ],
                    )
                moved += len(rows)
        return moved

    def small_segment_items(self) -> tuple[ColumnElement[bool], ColumnElement[bool]]:
        """The join condition of ``data_item`` and ``data_segment``, and the condition of the item rows whose content
        is in a segment that a small write made: one of less than ``CREATE_SEGMENT_BYTES``."""
        items, segment = self.items_table, self.segment_table
        return (
            and_(items.c.owner == segment.c.owner, items.c.segment_id == segment.c.segment_id),
            segment.c.byte_count < min(CREATE_SEGMENT_BYTES, PACK_BYTES),
        )

    def merge_small_segments(self, owner: str) -> int:
        """Move the trajectory content of ``owner`` that is in small segments (``small_segment_items``) into segments
        of at least ``PACK_BYTES``, oldest first, while at least ``PACK_BYTES`` of it is there; return the number of
        rows moved. A read of many items then sends fewer requests to the segment store.

        A reader that holds the old place of a line still reads it: ``delete_unused_segments`` keeps the old
        segment for ``UNUSED_SEGMENT_SECONDS``. The caller holds ``pack_lock(owner)``.
        """
        partition = self.partition(owner)
        if partition is None:
            return 0
        items, segment = self.items_table, self.segment_table
        joined, small = self.small_segment_items()
        source = items.join(segment, joined)
        condition = and_(items.c.owner == owner, small)
        moved = 0
        while True:
            rows: list[RowMapping] = []
            total = 0
            with self.transaction() as connection:
                waiting = select(func.sum(items.c.segment_length)).select_from(source).where(condition)
                if (connection.scalar(waiting) or 0) < PACK_BYTES:
                    break
                result = connection.execute(
                    select(items.c.sequence, items.c.segment_id, items.c.segment_offset, items.c.segment_length)
                    .select_from(source)
                    .where(condition)
                    .order_by(items.c.sequence)
                ).mappings()
                for row in result:
                    rows.append(row)
                    total += int(row["segment_length"])
                    if total >= PACK_BYTES:
                        break
                result.close()
            lines = [
                SegmentLine(row["segment_id"], int(row["segment_offset"]), int(row["segment_length"])) for row in rows
            ]
            places = self.write_segment(partition, partition.read_text(lines))
            with self.transaction() as connection:
                # A row that another write moved meanwhile keeps its new place.
                connection.execute(
                    update(items)
                    .where(
                        items.c.owner == owner,
                        items.c.sequence == bindparam("row_sequence"),
                        items.c.segment_id == bindparam("old_segment_id"),
                    )
                    .values(**{name: bindparam(f"new_{name}") for name in SEGMENT_COLUMNS}),
                    [
                        {
                            "row_sequence": row["sequence"],
                            "old_segment_id": row["segment_id"],
                            "new_segment_id": place.segment_id,
                            "new_segment_offset": place.offset,
                            "new_segment_length": place.length,
                        }
                        for row, place in zip(rows, places, strict=True)
                    ],
                )
            moved += len(rows)
        return moved

    def maintain(self) -> None:
        super().maintain()
        if self.segments is None:
            return
        owners: set[str] = set()
        with self.transaction() as connection:
            for table, size, waiting in self.waiting_content():
                owners.update(
                    connection.scalars(
                        select(table.c.owner)
                        .where(waiting)
                        .group_by(table.c.owner)
                        .having(func.sum(size) >= PACK_BYTES)
                    )
                )
        items, segment = self.items_table, self.segment_table
        joined, small = self.small_segment_items()
        with self.transaction() as connection:
            scattered = set(
                connection.scalars(
                    select(items.c.owner)
                    .select_from(items.join(segment, joined))
                    .where(small)
                    .group_by(items.c.owner)
                    .having(func.sum(items.c.segment_length) >= PACK_BYTES)
                )
            )
        for owner in sorted(owners | scattered):
            with self.pack_lock(owner) as locked:
                if locked:
                    self.pack(owner)
                    self.merge_small_segments(owner)
        self.delete_unused_segments()

    def delete_unused_segments(self) -> int:
        """Delete each segment older than ``UNUSED_SEGMENT_SECONDS`` that no item, deleted item or node uses; return
        the count.

        New rows only use new segments, so a segment that is old and unused stays unused. A restore moves a row
        with its segment line, so a segment that a deleted item uses stays used until the purge.
        """
        if self.segments is None:
            return 0
        segment, items, nodes, deleted = self.segment_table, self.items_table, self.nodes_table, self.deleted_table
        with self.transaction() as connection:
            unused = connection.execute(
                select(segment.c.owner, segment.c.segment_id).where(
                    segment.c.created_at < time.time() - UNUSED_SEGMENT_SECONDS,
                    ~exists().where(items.c.owner == segment.c.owner, items.c.segment_id == segment.c.segment_id),
                    ~exists().where(deleted.c.owner == segment.c.owner, deleted.c.segment_id == segment.c.segment_id),
                    ~exists().where(nodes.c.owner == segment.c.owner, nodes.c.segment_id == segment.c.segment_id),
                )
            ).all()
        for owner, segment_id in unused:
            self.segments.delete(owner, segment_id)
            with self.transaction() as connection:
                connection.execute(delete(segment).where(segment.c.owner == owner, segment.c.segment_id == segment_id))
        return len(unused)

    def task_path(self, item: StoredItem) -> Path:
        if item.kind != "task":
            raise ValueError(f"item {item.item_id!r} is a {item.kind}, not a task")
        return self.task_files.path(task_partition(item.owner), item.digest)

    def record_links(self, owner: str, item_ids: Sequence[str]) -> dict[str, RecordLink]:
        check_owner(owner)
        items = self.items_table
        with self.transaction(read_only=True) as connection:
            linked: dict[int, str] = {}
            for batch in batches(dict.fromkeys(item_ids)):
                for item_id, row in self.rows(connection, owner, batch).items():
                    if row["kind"] == "record" and row["message_field"] is not None:
                        linked[int(row["sequence"])] = item_id
            nodes = {
                int(row["node_sequence"]): row
                for row in self.trajectory_nodes(owner).paths(connection, linked, "agent_record_id", "entry_count")
            }
            # A node stays while a child needs it, after its item is deleted; the link skips such ancestors.
            stored = set(linked)
            for ancestors in batches(sorted(set(nodes) - stored)):
                stored.update(
                    int(sequence)
                    for sequence in connection.scalars(
                        select(items.c.sequence).where(items.c.owner == owner, items.c.sequence.in_(ancestors))
                    )
                )
        links: dict[str, RecordLink] = {}
        for sequence, item_id in linked.items():
            parent = nodes[sequence]["parent_sequence"]
            while parent is not None and int(parent) not in stored:
                parent = nodes[int(parent)]["parent_sequence"]
            parent_id = None if parent is None else nodes[int(parent)]["agent_record_id"]
            links[item_id] = RecordLink(parent_id, int(nodes[sequence]["entry_count"]))
        return links


class SQLiteDataStore(SQLDataStore):
    """The data store in a SQLite file, ``<directory>/data-store.sqlite3``, or in memory without a directory.

    It holds one connection for its lifetime and serializes its use with a lock, as the SQLite record store does.
    """

    def __init__(
        self,
        directory: Path | None,
        task_files: TaskFileStore,
        segments: SegmentStore | None = None,
        job_directories: SegmentStore | None = None,
        job_directory_ttl_seconds: float = JOB_DIRECTORY_TTL_SECONDS,
    ) -> None:
        metadata = MetaData()
        tables = data_tables(metadata, sqlite=True)
        super().__init__(tables, task_files, segments, job_directories, job_directory_ttl_seconds)
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
            # The services of one host open the same file at once; the first switches it to WAL.
            enable_wal(self.connection, database, timeout_seconds=30.0, retry_seconds=0.01)
            self.connection.commit()
        with self.connection.begin():
            # Serialize the table creation of concurrent openers.
            self.connection.exec_driver_sql("BEGIN IMMEDIATE")
            upgrade_schema(
                self.connection, metadata, DATA_SCHEMA_STEPS, version_table=VERSION_TABLE, first_table="data_item"
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        directory: Path,
        task_files: TaskFileStore,
        segments: SegmentStore | None,
        job_directory_ttl_seconds: float = JOB_DIRECTORY_TTL_SECONDS,
    ) -> Self:
        if config:
            raise PluginConfigError(
                f"the sqlite data store takes no config, got {', '.join(sorted(map(str, config)))}"
            )
        return cls(directory, task_files, segments, job_directory_ttl_seconds=job_directory_ttl_seconds)

    @contextmanager
    def transaction(self, *, read_only: bool = False) -> Iterator[Connection]:
        # The lock and the write lock already run each transaction alone, so a read sees one state.
        with self.lock:
            if self.connection is None:
                raise ReefError("the data store is closed")
            with self.connection.begin():
                # Another process can hold the file: take the write lock now, within the busy timeout, so a read
                # never has to upgrade to a write, which SQLite refuses at once under contention.
                self.connection.exec_driver_sql("BEGIN IMMEDIATE")
                yield self.connection

    def write_new_items(self, connection: Connection, values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        statement = (
            sqlite_insert(self.items_table)
            .values(list(values))
            .on_conflict_do_nothing(index_elements=["owner", "item_id"])
            .returning(self.items_table.c.item_id, self.items_table.c.sequence)
        )
        return {item_id: int(sequence) for item_id, sequence in connection.execute(statement)}

    def metadata_string(self, key: str) -> ColumnElement[str]:
        path = f"$.{key}"
        metadata = self.items_table.c.metadata_json
        return case((func.json_type(metadata, path) == "text", func.json_extract(metadata, path)))

    def is_inference(self) -> ColumnElement[bool]:
        return item_is_inference(self.items_table, sqlite=True)

    def evaluation_id(self) -> ColumnElement[Any]:
        return item_evaluation_id(self.items_table, sqlite=True)

    def has_id_prefix(self, prefix: str) -> ColumnElement[bool]:
        return item_id_prefix(self.items_table, prefix, sqlite=True)

    def use_has_id_prefix(self, prefix: str) -> ColumnElement[bool]:
        return use_id_prefix(self.uses_table, prefix, sqlite=True)

    def close(self) -> None:
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
                self.engine.dispose()
                if self.segments is not None:
                    self.segments.close()


@dataclass(frozen=True)
class PostgresDataStoreSettings(PluginSettings):
    """The config of the postgres data store: the database, and the schema of its tables."""

    database_url: str = field(repr=False, metadata=config_metadata("PostgreSQL connection URL."))
    schema: str = config_option("reef_data", help="SQL schema of the data store's tables.")


class PostgresDataStore(SQLDataStore):
    """The data store in a PostgreSQL schema. Deployments that name one database and schema share their items."""

    def __init__(
        self,
        database_url: str,
        task_files: TaskFileStore,
        *,
        schema: str = "reef_data",
        segments: SegmentStore | None = None,
        job_directories: SegmentStore | None = None,
        job_directory_ttl_seconds: float = JOB_DIRECTORY_TTL_SECONDS,
    ) -> None:
        validate_postgres_schema(schema)
        metadata = MetaData(schema=schema)
        tables = data_tables(metadata, sqlite=False)
        super().__init__(tables, task_files, segments, job_directories, job_directory_ttl_seconds)
        try:
            self.engine: Engine | None = create_engine(
                postgres_url(database_url), pool_pre_ping=True, hide_parameters=True
            )
        except ModuleNotFoundError:
            raise ReefError("the PostgreSQL data store requires the reef-infra[postgres] extra") from None
        key = int.from_bytes(hashlib.sha256(f"reef-data-schema:{schema}".encode()).digest()[:8], "big", signed=True)
        self.schema = schema
        with self.engine.begin() as connection:
            # Serialize the check-then-create of the schema across deployments that start together.
            connection.execute(select(func.pg_advisory_xact_lock(key)))
            connection.execute(CreateSchema(schema, if_not_exists=True))
            upgrade_schema(
                connection, metadata, DATA_SCHEMA_STEPS, version_table=VERSION_TABLE, first_table="data_item"
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        directory: Path,
        task_files: TaskFileStore,
        segments: SegmentStore | None,
        job_directory_ttl_seconds: float = JOB_DIRECTORY_TTL_SECONDS,
    ) -> Self:
        settings = PostgresDataStoreSettings.parse(config, plugin_class=cls)
        return cls(
            settings.database_url,
            task_files,
            schema=settings.schema,
            segments=segments,
            job_directory_ttl_seconds=job_directory_ttl_seconds,
        )

    @contextmanager
    def transaction(self, *, read_only: bool = False) -> Iterator[Connection]:
        if self.engine is None:
            raise ReefError("the data store is closed")
        isolation = "REPEATABLE READ" if read_only else "READ COMMITTED"
        with self.engine.connect().execution_options(isolation_level=isolation) as connection, connection.begin():
            yield connection

    def write_new_items(self, connection: Connection, values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        statement = (
            postgresql_insert(self.items_table)
            .values(list(values))
            .on_conflict_do_nothing(index_elements=["owner", "item_id"])
            .returning(self.items_table.c.item_id, self.items_table.c.sequence)
        )
        return {item_id: int(sequence) for item_id, sequence in connection.execute(statement)}

    def metadata_string(self, key: str) -> ColumnElement[str]:
        metadata = cast(self.items_table.c.metadata_json, JSONB)
        return case((func.jsonb_typeof(metadata[key]) == "string", metadata[key].astext))

    def is_inference(self) -> ColumnElement[bool]:
        return item_is_inference(self.items_table, sqlite=False)

    def evaluation_id(self) -> ColumnElement[Any]:
        return item_evaluation_id(self.items_table, sqlite=False)

    def has_id_prefix(self, prefix: str) -> ColumnElement[bool]:
        return item_id_prefix(self.items_table, prefix, sqlite=False)

    def use_has_id_prefix(self, prefix: str) -> ColumnElement[bool]:
        return use_id_prefix(self.uses_table, prefix, sqlite=False)

    @contextmanager
    def pack_lock(self, owner: str) -> Iterator[bool]:
        # A session lock on its own connection: a pack runs several transactions and writes segments between them.
        if self.engine is None:
            raise ReefError("the data store is closed")
        key = int.from_bytes(
            hashlib.sha256(f"reef-data-pack:{self.schema}:{owner}".encode()).digest()[:8], "big", signed=True
        )
        with self.engine.connect() as connection:
            locked = bool(connection.scalar(select(func.pg_try_advisory_lock(key))))
            connection.commit()
            try:
                yield locked
            finally:
                if locked:
                    connection.execute(select(func.pg_advisory_unlock(key)))
                    connection.commit()

    def lock_scenarios(self, connection: Connection, owner: str, scenarios: Iterable[str]) -> None:
        # Identity values are given at insert time, not at commit time. A lock per scenario makes the two orders the
        # same within the scenario, also across deployments that share the schema. A commit makes its rows visible
        # before it releases the lock, so a reader of one scenario misses none. A sequence means nothing across
        # scenarios, so no reader may keep one cursor over several (RFC #1219).
        for scenario in sorted(set(scenarios)):
            name = f"reef-data-use:{self.schema}:{owner}:{scenario}"
            key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big", signed=True)
            connection.execute(select(func.pg_advisory_xact_lock(key)))

    def close(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
            if self.segments is not None:
                self.segments.close()


__all__ = [
    "PostgresDataStore",
    "PostgresDataStoreSettings",
    "SQLDataStore",
    "SQLiteDataStore",
    "data_tables",
]
