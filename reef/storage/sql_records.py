"""Shared SQLAlchemy operations, independent of database setup and dialect: the trajectory nodes of record items
(``TrajectoryNodes``), their eviction sizes, and the database of one scenario's state (``SQLScenarioDatabase``).

This module has no SQLite, filesystem, or scenario lifecycle dependencies.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from sqlalchemy import BigInteger, Column, Index, MetaData, String, Table, Text, and_, select, true, update
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.sql.elements import ColumnElement

from reef.core.atif_payloads import (
    AtifCall,
    CallShape,
    atif_agent,
    atif_entries,
    atif_reply,
    call_shape,
    canonical_json,
)
from reef.core.errors import ReefError
from reef.core.serving_config import ServingConfig
from reef.core.trajectory_index import CallKey, ParentCandidate, best_parent
from reef.storage.commit_history import Closeable
from reef.storage.scenario import ScenarioSettings
from reef.storage.segments import SegmentLine, SegmentPartition

#: The most values in one SQL ``IN`` list, below the variable limit of every supported SQLite build.
IN_LIST_SIZE = 500


#: The columns that place a row's content in a segment, NULL for content in the row itself.
SEGMENT_COLUMNS = ("segment_id", "segment_offset", "segment_length")


def row_line(row: RowMapping) -> SegmentLine | None:
    """The segment line of a row with the ``SEGMENT_COLUMNS``, or None for a row that holds its own content."""
    if row["segment_id"] is None:
        return None
    return SegmentLine(row["segment_id"], int(row["segment_offset"]), int(row["segment_length"]))


def batches[T](values: Iterable[T], size: int = IN_LIST_SIZE) -> Iterable[list[T]]:
    """``values`` in order, in lists of at most ``size``."""
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass(frozen=True)
class ScopedTable:
    """One table of a database, and the namespace of one store in it.

    Values: ``table`` is a SQLAlchemy table. ``scope`` maps column names of
    ``table`` to values; every name in ``scope`` is a column of ``table``, and
    the constructor checks it. ``scope`` is empty for a table without a
    namespace, and it names ``storage_id`` for a table that has that column.

    Purpose: a SQL adapter makes it for one store. The store restricts each
    operation to the rows whose columns equal ``scope`` (``condition``) and
    writes ``scope`` into each row it inserts.
    """

    table: Table
    scope: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.scope) - set(self.table.c.keys())
        if unknown:
            raise ValueError(f"the scope names columns that {self.table.name} does not have: {sorted(unknown)!r}")

    def condition(self) -> ColumnElement[bool]:
        """Select the rows of this store: every column in ``scope`` equals its value."""
        return and_(true(), *(self.table.c[key] == value for key, value in self.scope.items()))


def trajectory_node_table(metadata: MetaData, name: str, *scope_columns: Column[str], segments: bool = False) -> Table:
    """Define the node table of ``TrajectoryNodes`` in ``metadata``, named ``name``.

    ``scope_columns`` are the columns of a store's namespace: none in a database of one store, ``storage_id`` in the
    PostgreSQL record database, ``owner`` in the data store. Each node is unique by them and ``node_sequence``, and
    an index finds the nodes of a scenario by ``list_sha256``. With ``segments``, the table also has the
    ``SEGMENT_COLUMNS``, and ``entries_json`` and ``call_json`` are NULL in a node that a segment holds.
    """
    scope_names = [column.name for column in scope_columns]
    segment_columns = (
        (
            Column("segment_id", Text),
            Column("segment_offset", BigInteger),
            Column("segment_length", BigInteger),
            Index(f"{name}_by_segment", *scope_names, "segment_id"),
        )
        if segments
        else ()
    )
    return Table(
        name,
        metadata,
        *scope_columns,
        Column("node_sequence", BigInteger, primary_key=True, autoincrement=False),
        Column("parent_sequence", BigInteger),
        Column("scenario", Text, nullable=False),
        Column("agent_record_id", Text, nullable=False),
        Column("entry_count", BigInteger, nullable=False),
        Column("list_sha256", String(64), nullable=False),
        Column("reply_count", BigInteger, nullable=False),
        Column("reply_sha256", String(64)),
        Column("entries_json", Text, nullable=segments),
        Column("shape_json", Text),
        Column("call_json", Text, nullable=segments),
        Column("body_bytes", BigInteger, nullable=False),
        Column("record_count", BigInteger, nullable=False),
        *segment_columns,
        Index(f"{name}_by_prefix", *scope_names, "scenario", "list_sha256"),
        Index(f"{name}_by_record", *scope_names, "scenario", "agent_record_id"),
    )


class NodeContent(NamedTuple):
    """What the trajectory nodes hold for one record's payload, as JSON: the ``entries_json`` of the nodes on its
    path, root first, the ``shape_json`` of its root and the ``call_json`` of its own node."""

    entry_parts: tuple[str, ...]
    shape_json: str
    call_json: str

    def atif_call(self) -> AtifCall:
        """The record's ATIF call, parsed anew, so that no two records share an object."""
        entries = tuple(entry for part in self.entry_parts for entry in json.loads(part))
        shape = json.loads(self.shape_json)
        call = json.loads(self.call_json)
        return AtifCall(
            CallShape(**shape["shape"]),
            entries,
            shape["agent"],
            tuple(call["reply"]),
            call["call"],
        )


def node_line(entries_json: str, shape_json: str | None, call_json: str) -> str:
    """The segment line of a trajectory node: one JSON object of its entries, its root shape or null, and its call."""
    return f'{{"call":{call_json},"entries":{entries_json},"shape":{shape_json or "null"}}}'


def node_texts(line: str) -> tuple[str, str | None, str]:
    """The ``entries_json``, ``shape_json`` and ``call_json`` of a node from its segment line (``node_line``)."""
    node = json.loads(line)
    return (
        canonical_json(node["entries"]),
        None if node["shape"] is None else canonical_json(node["shape"]),
        canonical_json(node["call"]),
    )


@dataclass(frozen=True)
class NewNode:
    """The node of one record to add: its sequence, scenario and record ID, and its payload with that payload's key,
    shape and ``call_json`` (``node_call_json``).

    Purpose: a store gives these to ``TrajectoryNodes.add_many`` when it writes records. All but the sequence
    depend on the record alone, so a store can make them before it takes a lock.
    """

    node_sequence: int
    scenario: str
    agent_record_id: str
    payload: Mapping[str, Any]
    key: CallKey
    shape: CallShape
    call_json: str


def node_call_json(payload: Mapping[str, Any], shape: CallShape) -> str:
    """The ``call_json`` of the node of ``payload``: its reply entries and call fields."""
    reply, call = atif_reply(payload, shape)
    return canonical_json({"reply": reply, "call": call})


@dataclass(frozen=True)
class TrajectoryNodes:
    """The trajectory nodes of one store's records, in one scoped node table (RFC #1054).

    Values: ``nodes`` is a ``ScopedTable`` whose table has the columns ``node_sequence``, ``parent_sequence``,
    ``scenario``, ``agent_record_id``, ``entry_count``, ``list_sha256``, ``reply_count``, ``reply_sha256``,
    ``entries_json``, ``shape_json``, ``call_json``, ``body_bytes`` and ``record_count``. Each row of the scope is
    the node of one record whose payload has a message list, keyed by the record's sequence, which no later record
    reuses. The node holds the record's ATIF call (``reef.core.atif_payloads.atif_call``):

    * ``parent_sequence`` is ``NULL`` or a node of the scope with the same ``scenario`` and the same trajectory
      hash, a smaller sequence and an ``entry_count`` at most this node's. The entries of a node are the entries of
      its parent followed by the JSON list ``entries_json``; ``entry_count`` counts them and ``list_sha256`` is
      the call's ``CallKey.list_sha256`` (``reef.core.trajectory_index``).
    * The shape and the tools of a node are those of its root: ``shape_json`` of a root is the JSON of the call's
      ``shape`` and ``agent``, and ``shape_json`` of any other node is ``NULL``.
    * ``call_json`` is the JSON of the call's ``reply`` and ``call``: its reply and its other request and response
      fields.
    * ``reply_count`` and ``reply_sha256`` are those of the call's ``CallKey``.
    * ``body_bytes`` is the UTF-8 length of ``entries_json``, ``shape_json`` and ``call_json``.
    * ``record_count`` is 1 while the node's record is stored, plus the number of child nodes. A node whose count
      reaches 0 is deleted, so a node stays while a stored record needs its entries.

    ``segments`` is None, or the segment partition of a table with the ``SEGMENT_COLUMNS``. There, a node whose
    ``entries_json`` is NULL keeps its ``entries_json``, ``shape_json`` and ``call_json`` in a segment line
    (``node_line``).

    Purpose: a SQL store makes it for its scope. The store adds a node when it writes a record (``add``), reads the
    ATIF calls of records back (``contents``) and releases the nodes of records that it deletes (``release``), in
    the same transaction. Writers of one scope must be serialized: ``add`` reads the candidate parents before it
    inserts.
    """

    nodes: ScopedTable
    segments: SegmentPartition | None = None

    def add(
        self,
        connection: Connection,
        node_sequence: int,
        scenario: str,
        agent_record_id: str,
        payload: Mapping[str, Any],
        key: CallKey,
    ) -> None:
        """Add the node of the record at ``node_sequence``, whose payload ``payload`` has the key ``key``.

        The parent is the node of ``scenario`` that ``best_parent`` chooses among the nodes whose ``list_sha256``
        is one of the key's prefix hashes: same shape and tools, longest prefix, a call whose reply the request
        resends, then the newest. The node converts only the request values after its parent's to ATIF. A node
        without a parent keeps the shape and the tools.
        """
        shape = call_shape(payload)
        if shape is None:
            raise ValueError(f"record {agent_record_id!r} has no message list, so it has no trajectory node")
        self.add_many(
            connection,
            [NewNode(node_sequence, scenario, agent_record_id, payload, key, shape, node_call_json(payload, shape))],
        )

    def add_many(self, connection: Connection, new_nodes: Sequence[NewNode]) -> None:
        """Add ``new_nodes``, in sequence order, with a number of statements that does not grow with their count.

        The parent of each is as ``add`` gives, also among the earlier ones of ``new_nodes``.
        """
        if not new_nodes:
            return
        table = self.nodes.table
        candidates: dict[tuple[str, str], list[ParentCandidate]] = {}
        hashes = sorted({prefix for node in new_nodes for prefix in node.key.prefix_sha256})
        for batch in batches(hashes):
            rows = connection.execute(
                select(
                    table.c.scenario,
                    table.c.list_sha256,
                    table.c.node_sequence,
                    table.c.entry_count,
                    table.c.reply_count,
                    table.c.reply_sha256,
                ).where(
                    self.nodes.condition(),
                    table.c.scenario.in_({node.scenario for node in new_nodes}),
                    table.c.list_sha256.in_(batch),
                )
            )
            for scenario, list_sha256, sequence, count, reply_count, reply in rows:
                candidates.setdefault((scenario, list_sha256), []).append(
                    ParentCandidate(int(sequence), int(count), int(reply_count), reply)
                )
        values: list[dict[str, Any]] = []
        children: dict[int, int] = {}
        for node in new_nodes:
            shape = node.shape
            parent = best_parent(
                node.key,
                (found for prefix in node.key.prefix_sha256 for found in candidates.get((node.scenario, prefix), ())),
            )
            start = 0 if parent is None else parent.entry_count
            entries_json = canonical_json(atif_entries(node.payload, shape, start))
            shape_json = None
            if parent is None:
                agent = atif_agent(node.payload["tools"], shape.provider) if shape.has_tools else None
                shape_json = canonical_json({"shape": shape.as_json(), "agent": agent})
            values.append(
                {
                    **self.nodes.scope,
                    "node_sequence": node.node_sequence,
                    "parent_sequence": None if parent is None else parent.sequence,
                    "scenario": node.scenario,
                    "agent_record_id": node.agent_record_id,
                    "entry_count": node.key.entry_count,
                    "list_sha256": node.key.list_sha256,
                    "reply_count": node.key.reply_count,
                    "reply_sha256": node.key.reply_sha256,
                    "entries_json": entries_json,
                    "shape_json": shape_json,
                    "call_json": node.call_json,
                    "body_bytes": len((entries_json + (shape_json or "") + node.call_json).encode("utf-8")),
                    "record_count": 1,
                }
            )
            if parent is not None:
                children[parent.sequence] = children.get(parent.sequence, 0) + 1
            candidates.setdefault((node.scenario, node.key.list_sha256), []).append(
                ParentCandidate(node.node_sequence, node.key.entry_count, node.key.reply_count, node.key.reply_sha256)
            )
        connection.execute(table.insert().values(values))
        # A parent that is new in this call gets its children here too. Most parents have one child, so this is
        # usually one statement.
        by_count: dict[int, list[int]] = {}
        for sequence, count in children.items():
            by_count.setdefault(count, []).append(sequence)
        for count, sequences in by_count.items():
            connection.execute(
                update(table)
                .where(self.nodes.condition(), table.c.node_sequence.in_(sequences))
                .values(record_count=table.c.record_count + count)
            )

    def paths(self, connection: Connection, node_sequences: Iterable[int], *columns: str) -> list[RowMapping]:
        """The nodes on the paths from ``node_sequences`` to their roots, each once, with ``node_sequence``,
        ``parent_sequence`` and ``columns``."""
        table = self.nodes.table
        found: dict[int, RowMapping] = {}
        for batch in batches(sorted(set(node_sequences))):
            path = (
                select(table.c.node_sequence, table.c.parent_sequence)
                .where(self.nodes.condition(), table.c.node_sequence.in_(batch))
                .cte("trajectory_path", recursive=True)
            )
            path = path.union(
                select(table.c.node_sequence, table.c.parent_sequence).where(
                    self.nodes.condition(), table.c.node_sequence == path.c.parent_sequence
                )
            )
            rows = connection.execute(
                select(table.c.node_sequence, table.c.parent_sequence, *(table.c[name] for name in columns)).where(
                    self.nodes.condition(), table.c.node_sequence.in_(select(path.c.node_sequence))
                )
            ).mappings()
            found.update((int(row["node_sequence"]), row) for row in rows)
        return sorted(found.values(), key=lambda row: int(row["node_sequence"]))

    def contents(self, connection: Connection, node_sequences: Iterable[int]) -> dict[int, NodeContent]:
        """The content of each node of ``node_sequences``: the entries of its path, its root's shape and its call."""
        wanted = set(node_sequences)
        columns = ("entries_json", "shape_json", "call_json")
        rows = self.paths(connection, wanted, *columns, *(SEGMENT_COLUMNS if self.segments is not None else ()))
        texts: dict[int, tuple[str, str | None, str]] = {}
        packed: dict[int, SegmentLine] = {}
        for row in rows:
            sequence = int(row["node_sequence"])
            if row["entries_json"] is not None:
                texts[sequence] = (row["entries_json"], row["shape_json"], row["call_json"])
                continue
            line = None if self.segments is None else row_line(row)
            if line is None:
                raise ReefError(f"trajectory node {sequence} is in a segment, but the store has no segment store")
            packed[sequence] = line
        if packed and self.segments is not None:
            for sequence, text in zip(packed, self.segments.read_text(list(packed.values())), strict=True):
                texts[sequence] = node_texts(text)
        paths: dict[int, tuple[tuple[str, ...], str]] = {}
        contents: dict[int, NodeContent] = {}
        # A parent has a smaller sequence than its children, so it comes first.
        for row in rows:
            sequence = int(row["node_sequence"])
            parent = row["parent_sequence"]
            entries_json, shape_json, call_json = texts[sequence]
            if parent is None:
                if shape_json is None:
                    raise ValueError(f"the root node {sequence} has no shape")
                paths[sequence] = ((entries_json,), shape_json)
            else:
                above, root_shape_json = paths[int(parent)]
                paths[sequence] = ((*above, entries_json), root_shape_json)
            if sequence in wanted:
                contents[sequence] = NodeContent(*paths[sequence], call_json)
        return contents

    def release(self, connection: Connection, node_sequences: Iterable[int]) -> None:
        """Release the nodes of deleted records: each loses its record, and each node whose count reaches 0 is
        deleted and releases its parent in turn."""
        table = self.nodes.table
        pending: Counter[int] = Counter(node_sequences)
        while pending:
            by_change: dict[int, list[int]] = {}
            for sequence, change in pending.items():
                by_change.setdefault(change, []).append(sequence)
            for change, sequences in by_change.items():
                for batch in batches(sequences):
                    connection.execute(
                        update(table)
                        .where(self.nodes.condition(), table.c.node_sequence.in_(batch))
                        .values(record_count=table.c.record_count - change)
                    )
            released: list[tuple[int, int | None]] = []
            for batch in batches(sorted(pending)):
                rows = connection.execute(
                    select(table.c.node_sequence, table.c.parent_sequence).where(
                        self.nodes.condition(), table.c.node_sequence.in_(batch), table.c.record_count <= 0
                    )
                )
                released.extend((int(sequence), parent) for sequence, parent in rows)
            for batch in batches([sequence for sequence, _ in released]):
                connection.execute(table.delete().where(self.nodes.condition(), table.c.node_sequence.in_(batch)))
            pending = Counter(int(parent) for _, parent in released if parent is not None)


class EvictionSizes:
    """The bytes that evicting records in a fixed order frees, with their trajectory nodes.

    Values: ``scope_names`` are the columns of a record row that scope its node in ``nodes``, the node table: empty
    in a database of one store, ``storage_id`` in a database of several. For each scope ``s`` (the values of
    ``scope_names`` in a row) and node ``n`` on the path of a counted record, ``seen[(s, n)]`` is ``(parent_sequence,
    record_count, body_bytes)`` of the node when ``read`` read it, and ``counted[(s, n)]`` the number of its counts
    that the counted records released. Both start empty.

    Purpose: a retention scan and ``SQLRecordRetention.evict`` count records with it in eviction order. The size of
    a record is the bytes of its row plus the bytes of each node that evicting it deletes. The sizes hold while
    nothing writes the store between the counts and the eviction.
    """

    def __init__(self, nodes: Table, scope_names: Sequence[str]) -> None:
        self.nodes = nodes
        self.scope_names = tuple(scope_names)
        self.seen: dict[tuple[tuple[str, ...], int], tuple[int | None, int, int]] = {}
        self.counted: Counter[tuple[tuple[str, ...], int]] = Counter()

    def scope(self, row: RowMapping) -> tuple[str, ...]:
        """The scope of the node of the record row ``row``."""
        return tuple(str(row[name]) for name in self.scope_names)

    def trajectory_nodes(self, scope: tuple[str, ...]) -> TrajectoryNodes:
        """The trajectory nodes of one scope."""
        return TrajectoryNodes(ScopedTable(self.nodes, dict(zip(self.scope_names, scope, strict=True))))

    def read(self, connection: Connection, rows: Iterable[RowMapping]) -> None:
        """Read the nodes on the paths of ``rows`` that are not seen yet.

        Each row has the scope columns, ``sequence`` and ``message_field``.
        """
        unseen: dict[tuple[str, ...], list[int]] = {}
        for row in rows:
            scope = self.scope(row)
            if row["message_field"] is not None and (scope, int(row["sequence"])) not in self.seen:
                unseen.setdefault(scope, []).append(int(row["sequence"]))
        for scope, sequences in unseen.items():
            for node in self.trajectory_nodes(scope).paths(connection, sequences, "record_count", "body_bytes"):
                parent = node["parent_sequence"]
                self.seen[(scope, int(node["node_sequence"]))] = (
                    None if parent is None else int(parent),
                    int(node["record_count"]),
                    int(node["body_bytes"]),
                )

    def count(self, row: RowMapping) -> int:
        """Count the record row ``row`` as evicted after the records counted before; return its size.

        ``read`` has read the row's path.
        """
        scope = self.scope(row)
        size = int(row["body_bytes"])
        node: int | None = int(row["sequence"]) if row["message_field"] is not None else None
        while node is not None and (scope, node) in self.seen:
            self.counted[(scope, node)] += 1
            parent, record_count, body_bytes = self.seen[(scope, node)]
            if self.counted[(scope, node)] < record_count:
                break
            size += body_bytes
            node = parent
        return size


class SQLScenarioDatabase(Closeable):
    """The database of one scenario's state: its settings, and the tables of the stores that share its transactions.

    Values: ``settings`` is the settings table, and ``scope`` maps its namespace columns to their values (empty in a
    database of one scenario, ``storage_id`` in the PostgreSQL record database). A subclass gives ``_transaction``
    and ``_insert_if_absent``.

    Purpose: the task, sample, evaluation and commit stores of a session run their SQL in its transactions.
    """

    def __init__(self, settings: Table, scope: Mapping[str, str] | None = None) -> None:
        self._settings = ScopedTable(settings, dict(scope or {}))

    @abstractmethod
    def _transaction(self, *, write: bool) -> AbstractContextManager[Connection]:
        """Supply one connection with the isolation and local serialization needed. A write scope serializes the
        writers of the scenario across connections. An adapter that joins an outer transaction leaves its end to
        that transaction's owner."""

    @abstractmethod
    def _insert_if_absent(
        self,
        connection: Connection,
        table: Table,
        values: Mapping[str, object] | Sequence[Mapping[str, object]],
    ) -> bool:
        """Insert one or more rows, ignoring duplicate keys; report any insertion. Adapters must return an accurate
        result without assuming that every driver gives the INSERT rowcount."""

    def read_settings(self) -> ScenarioSettings:
        """The scenario's settings; defaults when none are stored."""
        table = self._settings.table
        with self._transaction(write=False) as connection:
            rows = connection.execute(select(table.c.name, table.c.value_json).where(self._settings.condition()))
            values = {name: json.loads(value_json) for name, value_json in rows}
        return ScenarioSettings(
            training_mode=values.get("training_mode"),
            artifact_evaluation=values.get("artifact_evaluation"),
            activation=values.get("activation"),
            owner=values.get("owner"),
            serving_config=ServingConfig.from_json(values.get("serving_config", {})),
        )

    def write_settings(self, settings: ScenarioSettings) -> None:
        """Replace the stored settings in one transaction; a ``None`` setting is not stored."""
        table = self._settings.table
        values = {
            "training_mode": settings.training_mode,
            "artifact_evaluation": (
                None if settings.artifact_evaluation is None else dict(settings.artifact_evaluation)
            ),
            "activation": None if settings.activation is None else dict(settings.activation),
            "owner": settings.owner,
            "serving_config": settings.serving_config.to_json() or None,
        }
        rows = [
            {**self._settings.scope, "name": name, "value_json": json.dumps(value, ensure_ascii=False)}
            for name, value in values.items()
            if value is not None
        ]
        with self._transaction(write=True) as connection:
            connection.execute(table.delete().where(self._settings.condition()))
            if rows:
                connection.execute(table.insert(), rows)
