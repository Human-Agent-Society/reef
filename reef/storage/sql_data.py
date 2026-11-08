"""The data store (RFC #1028) in a SQL database, for SQLite and PostgreSQL.

``data_item`` holds one row per item: its owner, ID, kind, digest, metadata as JSON, and the JSON content of a
trajectory or a record. ``data_item_scenario`` repeats the ``scenarios`` metadata, one row per scenario that uses
an item, so that a query by scenario uses an index. ``data_record_node`` keeps the trajectory node of each record
item with a message list, per owner (RFC #1054): the node keeps the payload as ATIF, with only the entries after
its parent's, and the item's content keeps an empty payload. The tables are written in one transaction. Task directories are
in the task file store, in each owner's partition.

With a segment store, the content of a trajectory item, and the content of a trajectory node, can live in a segment
line instead (``reef.storage.segments``). ``create`` writes the content of new trajectory items as one segment.
``maintain`` moves the rest into segments once an owner has ``PACK_BYTES`` of it, and deletes the segments that no
row uses. ``data_segment`` lists each segment that the store wrote, so that ``maintain`` finds the unused ones.
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
from pathlib import Path
from threading import Event, Lock, RLock
from typing import Any, Self

from sqlalchemy import (
    URL,
    BigInteger,
    Column,
    Float,
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
    literal_column,
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
    JOB_DIRECTORY_TTL_SECONDS,
    MAX_QUERY_LIMIT,
    DataStore,
    ItemConflict,
    ItemGroup,
    ItemKind,
    ItemNotFound,
    NewItem,
    RecordLink,
    ScenarioArchived,
    StoredItem,
    check_filters,
    check_groups,
    check_owner,
    check_query,
    generation_origin,
    item_digest_matches,
    patched_metadata,
    prepared_items,
    record_link_scope,
)
from reef.storage.item_kinds import task_partition
from reef.storage.migrations import (
    migrate_data_indexes,
    migrate_data_request_types,
    migrate_data_segments,
    migrate_data_trajectories,
    migrate_data_uses,
)
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
from reef.storage.sql_setup import enable_wal, postgres_url, validate_postgres_schema
from reef.storage.tasks import TaskFileStore

#: The content bytes that an owner's rows hold before ``maintain`` moves them into one segment.
PACK_BYTES = 64 * 2**20
#: The age in seconds after which ``maintain`` deletes a segment that no row uses. A new segment has no rows
#: until the transaction that writes them commits.
UNUSED_SEGMENT_SECONDS = 24 * 3600
#: The record items that one page of ``prune_records`` reads.
PRUNE_PAGE_SIZE = 256
#: The most items that one transaction of a group commit writes.
GROUP_COMMIT_ITEMS = 200


def item_evaluation_id(items: Table, *, sqlite: bool) -> ColumnElement[Any]:
    """The ``evaluation_id`` of the metadata of an item row, in the form of the index ``data_item_by_evaluation``.

    The JSON path is a literal, not a parameter, so that a query with the same expression can use the index.
    """
    if sqlite:
        return func.json_extract(items.c.metadata_json, literal_column("'$.evaluation_id'"))
    return cast(items.c.metadata_json, JSONB).op("->>")(literal_column("'evaluation_id'"))


def data_tables(metadata: MetaData, *, sqlite: bool) -> tuple[Table, Table, Table, Table, Table, Table, Table]:
    """Define ``data_item``, ``data_item_scenario``, ``data_record_node``, ``data_segment``, ``data_job_directory``,
    ``data_scenario`` and ``data_record_loss`` in ``metadata``. ``data_scenario`` has the current generation of each
    scenario of an owner (``DataStore.scenario_generation``). ``data_record_loss`` has the ``RecordLoss`` of each
    origin scenario of an owner whose record items ``prune_records`` deleted. ``data_job_directory`` has the key of each job data directory and the time that a job last used it.

    ``sequence`` orders the items as they arrived. ``data_item_scenario`` has one row per use of an item by a
    scenario, with its split; its own ``sequence`` orders the uses as they began, and a use whose split changes is a
    new row (``DataStore.scenario_items``). ``content_json`` is the canonical JSON of a trajectory's or a
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
        UniqueConstraint("owner", "item_id"),
        Index("data_item_by_segment", "owner", "segment_id"),
        Index("data_item_by_kind", "owner", "kind", "sequence"),
        # SQLite reuses the largest rowid after a delete without it; other dialects ignore the option.
        sqlite_autoincrement=sqlite,
    )
    scenarios = Table(
        "data_item_scenario",
        metadata,
        (
            Column("sequence", Integer, primary_key=True)
            if sqlite
            else Column("sequence", BigInteger, Identity(), primary_key=True)
        ),
        Column("owner", Text, nullable=False),
        Column("item_id", Text, nullable=False),
        Column("scenario", Text, nullable=False),
        Column("split", String(8), nullable=False),
        UniqueConstraint("owner", "item_id", "scenario"),
        Index("data_item_scenario_by_scenario", "owner", "scenario", "sequence"),
        sqlite_autoincrement=sqlite,
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
    # The calls of one evaluation are read by their evaluation_id (``SQLDataStore.query``), so a read does not scan
    # all the record items of an owner. ``migrate_data_indexes`` adds the index to an older table.
    Index("data_item_by_evaluation", items.c.owner, item_evaluation_id(items, sqlite=sqlite))
    return items, scenarios, nodes, segments, job_directories, generations, losses


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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

    A subclass gives the tables, ``transaction`` (one database transaction on one connection), ``insert_new_items``
    (an insert that leaves an existing ID alone) and, where transactions run at once, ``lock_scenarios``. Every
    operation runs in one transaction, so an operation that refuses changes nothing in the database. ``create``
    inserts first and compares digests after, so two creates of one ID at once end in a no-op or an
    ``ItemConflict``, never in a failed insert. ``create``, ``update`` and ``delete`` take ``lock_scenarios`` for the
    scenarios that they give a use or whose trajectory nodes they change. Concurrent creates of one owner and one
    set of scenarios in this process share one transaction (a group commit), because only one of them can hold
    the lock at a time. ``segments`` is None, or the segment store that keeps the content of
    trajectory items and nodes in the partition of each owner; the store closes it with itself. ``job_directories`` keeps
    the job data directories: ``job_directories`` when given, else ``segments``, else a local segment store in a new
    temporary directory, which only this process reads. ``job_directory_ttl_seconds`` is how long the store keeps a
    job data directory that no job used.
    """

    def __init__(
        self,
        tables: tuple[Table, Table, Table, Table, Table, Table, Table],
        task_files: TaskFileStore,
        segments: SegmentStore | None,
        job_directories: SegmentStore | None = None,
        job_directory_ttl_seconds: float = JOB_DIRECTORY_TTL_SECONDS,
    ) -> None:
        (
            self.items_table,
            self.scenarios_table,
            self.nodes_table,
            self.segment_table,
            self.job_directory_table,
            self.generations_table,
            self.losses_table,
        ) = tables
        self.task_files = task_files
        self.segments = segments
        self.job_directories = (
            job_directories or segments or LocalSegmentStore(Path(tempfile.mkdtemp(prefix="reef-job-directories-")))
        )
        self.job_directory_ttl_seconds = job_directory_ttl_seconds
        self.group_lock = Lock()
        self.waiting: dict[tuple[str, frozenset[str]], list[PendingCreate]] = {}

    def record_job_directory_use(self, key: str, time: float) -> None:
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
    def insert_new_items(self, connection: Connection, values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
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
    def record_source(self) -> ColumnElement[str]:
        """The source of the record item of an item row (``reef.storage.data.record_source``): its
        ``origin.source``, or ``inference`` when it has none."""

    @abstractmethod
    def evaluation_id(self) -> ColumnElement[Any]:
        """The ``evaluation_id`` of the metadata of an item row, as the index ``data_item_by_evaluation`` holds it
        (``item_evaluation_id``)."""

    def is_inference(self) -> ColumnElement[bool]:
        """The condition that an item row is a record item of the source ``inference``, which ``prune_records``
        counts and deletes."""
        return and_(self.items_table.c.kind == "record", self.record_source() == "inference")

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
            metadata = {
                int(sequence): json.loads(text)
                for sequence, text in connection.execute(
                    select(items.c.sequence, items.c.metadata_json).where(
                        items.c.owner == owner, items.c.sequence.in_([int(row[2]) for row in found])
                    )
                )
            }
        return tuple(
            ItemGroup(value, int(count), int(low), int(high), metadata[int(low)]) for value, count, low, high in found
        )

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
        partition.store.put(partition.partition, segment_id, data)
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

    def write_scenarios(self, connection: Connection, owner: str, scenarios: Mapping[str, Mapping[str, str]]) -> None:
        """Make the uses of each item ID in ``scenarios`` match its scenarios: an unchanged use keeps its row and
        sequence, and a new use or a new split gets a new row. At most three statements read and write the uses."""
        uses = self.scenarios_table
        current: dict[str, dict[str, str]] = {item_id: {} for item_id in scenarios}
        for row in connection.execute(
            select(uses.c.item_id, uses.c.scenario, uses.c.split).where(
                uses.c.owner == owner, uses.c.item_id.in_(list(scenarios))
            )
        ):
            current[row.item_id][row.scenario] = row.split
        ended = [
            {"row_item_id": item_id, "row_scenario": scenario}
            for item_id, wanted in scenarios.items()
            for scenario, split in current[item_id].items()
            if wanted.get(scenario) != split
        ]
        if ended:
            connection.execute(
                delete(uses).where(
                    uses.c.owner == owner,
                    uses.c.item_id == bindparam("row_item_id"),
                    uses.c.scenario == bindparam("row_scenario"),
                ),
                ended,
            )
        begun = [
            {"owner": owner, "item_id": item_id, "scenario": scenario, "split": split}
            for item_id, wanted in scenarios.items()
            for scenario, split in wanted.items()
            if current[item_id].get(scenario) != split
        ]
        if begun:
            connection.execute(insert(uses), begun)

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
            if new:
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
        create's ``ItemConflict`` does not refuse the others."""
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

        An ID in two creates gets the row of the first; another digest in the later one is an ``ItemConflict``.
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
            self.lock_scenarios(connection, owner, scenarios_of(metadata for _, metadata, _ in prepared))
            # The archive ends a generation under the same lock, so no write of an ended generation commits.
            checked = {pending.generation for pending in batch if pending.generation is not None}
            current = {origin: self.current_generation(connection, owner, origin) for origin, _ in checked}
            for origin, generation in checked:
                if current[origin] != generation:
                    raise ScenarioArchived(f"generation {generation!r} of {origin!r} is no longer current")
            sequences = self.insert_new_items(connection, list(values.values()))
            metadata_of = {entry[0].item_id: entry[1] for entry in reversed(prepared)}
            uses = [
                {"owner": owner, "item_id": item_id, "scenario": scenario, "split": split}
                for item_id in sequences
                for scenario, split in metadata_of[item_id]["scenarios"].items()
            ]
            if uses:
                connection.execute(self.scenarios_table.insert().values(uses))
            self.trajectory_nodes(owner).add_many(
                connection,
                [
                    replace(calls[item_id], node_sequence=sequence)
                    for item_id, sequence in sorted(sequences.items(), key=lambda pair: pair[1])
                    if item_id in calls
                ],
            )
            # An ID that was stored before, or that a concurrent create stored first, keeps its row; another digest
            # there is a conflict, and the rollback removes the rows that this call inserted. A row of this call has
            # the digest of its first item with that ID. Only another digest needs the stored content, which can be
            # in S3, and this transaction holds its locks.
            kept = [item_id for item_id in values if item_id not in sequences]
            stored_rows = self.rows(connection, owner, kept) if kept else {}
            for row in self.stored_items(
                connection,
                owner,
                [row for item_id, row in stored_rows.items() if row["digest"] != values[item_id]["digest"]],
            ):
                if not item_digest_matches(row, values[row.item_id]["digest"]):
                    raise ItemConflict(f"item {row.item_id!r} of {owner!r} is stored with other content")
            stored = {
                item.item_id: item
                for item in self.stored_items(connection, owner, list(stored_rows.values()), content=False)
            }
            for item, _, digest in prepared:
                if item.item_id in sequences and digest != values[item.item_id]["digest"]:
                    raise ItemConflict(f"item {item.item_id!r} of {owner!r} is stored with other content")
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
    ) -> tuple[StoredItem, ...]:
        check_query(owner, kind, scenario, after, limit, origin=origin, request_type=request_type, group=group)
        items = self.items_table
        # Without content, read only the columns of the metadata: content_json can be large.
        columns = (
            (items,)
            if content
            else (items.c.sequence, items.c.owner, items.c.item_id, items.c.metadata_json, items.c.digest)
        )
        statement = select(*columns).where(*self.filters(owner, kind, scenario, origin, request_type, after))
        if group is not None:
            value = self.metadata_string(group[0])
            statement = statement.where(value.is_(None) if group[1] is None else value == group[1])
            if group[0] == "evaluation_id" and group[1] is not None:
                # The same condition in the form of the index data_item_by_evaluation, which finds the rows.
                statement = statement.where(self.evaluation_id() == group[1])
        with self.transaction(read_only=True) as connection:
            found = connection.execute(statement.order_by(items.c.sequence).limit(limit)).mappings().all()
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
        statement = select(func.count()).where(*self.filters(owner, kind, scenario, origin, request_type, after))
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
        newest = func.coalesce(func.max(self.items_table.c.sequence), 0)
        statement = select(newest).where(*self.filters(owner, kind, scenario, origin, request_type, 0))
        with self.transaction(read_only=True) as connection:
            return int(connection.execute(statement).scalar_one())

    def filters(
        self,
        owner: str,
        kind: ItemKind | None,
        scenario: str | None,
        origin: str | None,
        request_type: str | None,
        after: int,
    ) -> list[ColumnElement[bool]]:
        """The conditions on the rows of ``data_item`` that the filters of ``query`` give. The ID of a record item
        starts with the scenario that made it (``record_item``)."""
        items = self.items_table
        conditions = [items.c.owner == owner, items.c.sequence > after, *self.used_by(owner, items, scenario)]
        if kind is not None:
            conditions.append(items.c.kind == kind)
        if origin is not None:
            conditions.append(items.c.item_id.startswith(f"{origin}/", autoescape=True))
        if request_type is not None:
            conditions.append(items.c.request_type == request_type)
        return conditions

    def used_by(self, owner: str, items: Table, scenario: str | None) -> tuple[ColumnElement[bool], ...]:
        """The condition that ``scenario`` uses the item of the row of ``items``, or none without a scenario."""
        if scenario is None:
            return ()
        uses = self.scenarios_table
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
    ) -> tuple[tuple[int, StoredItem], ...]:
        check_query(owner, None, scenario, after, limit)
        items, uses = self.items_table, self.scenarios_table
        statement = (
            select(items, uses.c.sequence.label("use_sequence"))
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
        return tuple((int(row["use_sequence"]), item) for row, item in zip(found, stored, strict=True))

    def scenarios_with_records(self, owner: str) -> tuple[str, ...]:
        check_owner(owner)
        items, uses = self.items_table, self.scenarios_table
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
        uses = self.scenarios_table
        statement = select(func.max(uses.c.sequence)).where(uses.c.owner == owner, uses.c.scenario == scenario)
        with self.transaction(read_only=True) as connection:
            return int(connection.execute(statement).scalar() or 0)

    def update(self, owner: str, item_ids: Sequence[str], patch: Mapping[str, Any]) -> tuple[StoredItem, ...]:
        check_owner(owner)
        unique_ids = list(dict.fromkeys(item_ids))
        with self.transaction() as connection:
            # A new use gets its sequence here, so that a reader's cursor of use sequences misses none (RFC #1219).
            scenarios = patch.get("scenarios") if isinstance(patch, Mapping) else None
            self.lock_scenarios(connection, owner, scenarios if isinstance(scenarios, Mapping) else ())
            # A patch merges into the whole metadata, so a concurrent patch must read after this one commits.
            existing = self.rows(connection, owner, unique_ids, lock=True)
            missing = [item_id for item_id in unique_ids if item_id not in existing]
            if missing:
                raise ItemNotFound(f"{owner!r} stores no item {missing[0]!r}")
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
            self.write_scenarios(
                connection, owner, {item_id: metadata["scenarios"] for item_id, metadata in patched.items()}
            )
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
        # (``record_link_scope``), so no node of an inference record needs it.
        evaluation_node = exists().where(
            items.c.owner == nodes.c.owner,
            items.c.sequence == nodes.c.node_sequence,
            items.c.kind == "record",
            self.record_source() == "evaluation",
        )
        with self.transaction(read_only=True) as connection:
            held = connection.execute(
                select(func.coalesce(func.sum(func.length(items.c.content_json)), 0)).where(self.is_inference())
            ).scalar_one()
            held += connection.execute(
                select(func.coalesce(func.sum(nodes.c.body_bytes), 0)).where(~evaluation_node)
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
        items, uses, losses = self.items_table, self.scenarios_table, self.losses_table
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

    def remove_scenario(self, owner: str, scenario: str) -> int:
        check_owner(owner)
        items, uses = self.items_table, self.scenarios_table
        with self.transaction() as connection:
            self.lock_scenarios(connection, owner, [scenario])
            generations = self.generations_table
            connection.execute(
                delete(generations).where(generations.c.owner == owner, generations.c.scenario == scenario)
            )
            # The rows stay locked, so that a concurrent patch of another scenario reads after this one commits.
            rows = connection.execute(
                select(items.c.item_id, items.c.metadata_json)
                .where(
                    items.c.owner == owner,
                    exists().where(
                        uses.c.owner == owner, uses.c.item_id == items.c.item_id, uses.c.scenario == scenario
                    ),
                )
                .order_by(items.c.item_id)
                .with_for_update(of=items)
            ).all()
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
            connection.execute(delete(uses).where(uses.c.owner == owner, uses.c.scenario == scenario))
            return len(rows)

    def delete(self, owner: str, item_id: str) -> bool:
        check_owner(owner)
        with self.transaction() as connection:
            row = self.rows(connection, owner, [item_id]).get(item_id)
            if row is None:
                return False
            self.lock_scenarios(connection, owner, scenarios_of([json.loads(row["metadata_json"])]))
            connection.execute(
                delete(self.scenarios_table).where(
                    self.scenarios_table.c.owner == owner, self.scenarios_table.c.item_id == item_id
                )
            )
            connection.execute(
                delete(self.items_table).where(
                    self.items_table.c.owner == owner, self.items_table.c.item_id == item_id
                )
            )
            if row["message_field"] is not None:
                self.trajectory_nodes(owner).release(connection, [int(row["sequence"])])
        return True

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
        for owner in sorted(owners):
            with self.pack_lock(owner) as locked:
                if locked:
                    self.pack(owner)
        self.delete_unused_segments()

    def delete_unused_segments(self) -> int:
        """Delete each segment older than ``UNUSED_SEGMENT_SECONDS`` that no item or node uses; return the count.

        New rows only use new segments, so a segment that is old and unused stays unused.
        """
        if self.segments is None:
            return 0
        segment, items, nodes = self.segment_table, self.items_table, self.nodes_table
        with self.transaction() as connection:
            unused = connection.execute(
                select(segment.c.owner, segment.c.segment_id).where(
                    segment.c.created_at < time.time() - UNUSED_SEGMENT_SECONDS,
                    ~exists().where(items.c.owner == segment.c.owner, items.c.segment_id == segment.c.segment_id),
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
            metadata.create_all(self.connection)
            migrate_data_segments(self.connection, self.items_table, self.nodes_table)
            migrate_data_request_types(self.connection, self.items_table)
            migrate_data_uses(self.connection, self.items_table, self.scenarios_table)
            migrate_data_indexes(self.connection, self.items_table)
            migrate_data_trajectories(self.connection, self.items_table, self.nodes_table)

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

    def insert_new_items(self, connection: Connection, values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
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

    def record_source(self) -> ColumnElement[str]:
        return func.coalesce(func.json_extract(self.items_table.c.metadata_json, "$.origin.source"), "inference")

    def evaluation_id(self) -> ColumnElement[Any]:
        return item_evaluation_id(self.items_table, sqlite=True)

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
            metadata.create_all(connection)
            migrate_data_segments(connection, self.items_table, self.nodes_table)
            migrate_data_request_types(connection, self.items_table)
            migrate_data_uses(connection, self.items_table, self.scenarios_table)
            migrate_data_indexes(connection, self.items_table)
            migrate_data_trajectories(connection, self.items_table, self.nodes_table)

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

    def insert_new_items(self, connection: Connection, values: Sequence[Mapping[str, Any]]) -> dict[str, int]:
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

    def record_source(self) -> ColumnElement[str]:
        metadata = cast(self.items_table.c.metadata_json, JSONB)
        return func.coalesce(metadata[("origin", "source")].astext, "inference")

    def evaluation_id(self) -> ColumnElement[Any]:
        return item_evaluation_id(self.items_table, sqlite=False)

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
