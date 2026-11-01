"""Shared SQLAlchemy record operations, independent of database setup and dialect.

Adapters supply tables, transaction-scoped connections, conflict insertion, and
resource cleanup. The domain behavior is implemented here once; this module
has no SQLite, filesystem, or scenario lifecycle dependencies.
"""

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from weakref import WeakValueDictionary

from sqlalchemy import (
    BigInteger,
    Column,
    Index,
    MetaData,
    String,
    Table,
    Text,
    and_,
    exists,
    func,
    select,
    true,
    tuple_,
    update,
)
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.sql.elements import ColumnElement

from reef.core.artifact_ref import decode_artifact_ref, encode_artifact_ref
from reef.core.atif_payloads import (
    AtifCall,
    CallShape,
    atif_agent,
    atif_entries,
    atif_reply,
    call_shape,
    canonical_json,
    provider_payload,
)
from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectory_index import CallKey, ParentCandidate, best_parent, call_key, sha256_text, trajectory_calls
from reef.storage.records import (
    AppendResult,
    ConsumptionReceipt,
    RecordConflict,
    RecordLoss,
    RecordStore,
    StoredRecord,
)
from reef.storage.scenario import ScenarioSettings

#: The most values in one SQL ``IN`` list, below the variable limit of every supported SQLite build.
IN_LIST_SIZE = 500


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
class RecordTables:
    """Tables for record bodies, trajectory nodes, retry hashes, consumption, capacity losses, scenario settings, and
    the records that stored samples hold.

    The adapter owns schema types, constraints, indexes, and migrations. Table
    columns retain the names consumed by SQLRecordStore; backend-specific
    generated keys or indexes may be added without replacing record operations.
    ``nodes`` is the node table of ``TrajectoryNodes``, and ``sample_records`` the table of
    ``sample_record_table``.
    """

    records: Table
    consumed: Table
    consumption: Table
    eviction: Table
    settings: Table
    nodes: Table
    sample_records: Table
    scope: Mapping[str, str] = field(default_factory=dict)

    def condition(self, table: Table) -> ColumnElement[bool]:
        """Restrict operations to the adapter's storage namespace, if any."""
        return and_(true(), *(table.c[key] == value for key, value in self.scope.items()))


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


def trajectory_node_table(metadata: MetaData, name: str, *scope_columns: Column[str]) -> Table:
    """Define the node table of ``TrajectoryNodes`` in ``metadata``, named ``name``.

    ``scope_columns`` are the columns of a store's namespace: none in a database of one store, ``storage_id`` in the
    PostgreSQL record database, ``owner`` in the data store. Each node is unique by them and ``node_sequence``, and
    an index finds the nodes of a scenario by ``list_sha256``.
    """
    scope_names = [column.name for column in scope_columns]
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
        Column("entries_json", Text, nullable=False),
        Column("shape_json", Text),
        Column("call_json", Text, nullable=False),
        Column("body_bytes", BigInteger, nullable=False),
        Column("record_count", BigInteger, nullable=False),
        Index(f"{name}_by_prefix", *scope_names, "scenario", "list_sha256"),
        Index(f"{name}_by_record", *scope_names, "scenario", "agent_record_id"),
    )


def sample_record_table(metadata: MetaData, *scope_columns: Column[str]) -> Table:
    """Define the ``sample_record`` table in ``metadata``: the records that a stored sample holds.

    A row ``(sample_id, agent_record_id)`` says that the stored sample ``sample_id`` renders its trajectory from the
    record ``agent_record_id`` (``SQLSampleStore``), so capacity eviction keeps the record. ``scope_columns`` are the
    columns of a store's namespace, as for ``trajectory_node_table``. An index finds the samples of a record.
    """
    scope_names = [column.name for column in scope_columns]
    return Table(
        "sample_record",
        metadata,
        *scope_columns,
        Column("sample_id", Text, primary_key=True),
        Column("agent_record_id", Text, primary_key=True),
        Index("sample_record_by_record", *scope_names, "agent_record_id"),
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

    Purpose: a SQL store makes it for its scope. The store adds a node when it writes a record (``add``), reads the
    ATIF calls of records back (``contents``) and releases the nodes of records that it deletes (``release``), in
    the same transaction. Writers of one scope must be serialized: ``add`` reads the candidate parents before it
    inserts.
    """

    nodes: ScopedTable

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
        table = self.nodes.table
        candidates: list[ParentCandidate] = []
        for batch in batches(key.prefix_sha256):
            rows = connection.execute(
                select(table.c.node_sequence, table.c.entry_count, table.c.reply_count, table.c.reply_sha256).where(
                    self.nodes.condition(), table.c.scenario == scenario, table.c.list_sha256.in_(batch)
                )
            )
            candidates.extend(
                ParentCandidate(int(sequence), int(count), int(reply_count), reply)
                for sequence, count, reply_count, reply in rows
            )
        parent = best_parent(key, candidates)
        entries_json = canonical_json(atif_entries(payload, shape, 0 if parent is None else parent.entry_count))
        shape_json = None
        if parent is None:
            agent = atif_agent(payload["tools"], shape.provider) if shape.has_tools else None
            shape_json = canonical_json({"shape": shape.as_json(), "agent": agent})
        reply, call = atif_reply(payload, shape)
        call_json = canonical_json({"reply": reply, "call": call})
        connection.execute(
            table.insert().values(
                **self.nodes.scope,
                node_sequence=node_sequence,
                parent_sequence=None if parent is None else parent.sequence,
                scenario=scenario,
                agent_record_id=agent_record_id,
                entry_count=key.entry_count,
                list_sha256=key.list_sha256,
                reply_count=key.reply_count,
                reply_sha256=key.reply_sha256,
                entries_json=entries_json,
                shape_json=shape_json,
                call_json=call_json,
                body_bytes=len((entries_json + (shape_json or "") + call_json).encode("utf-8")),
                record_count=1,
            )
        )
        if parent is not None:
            connection.execute(
                update(table)
                .where(self.nodes.condition(), table.c.node_sequence == parent.sequence)
                .values(record_count=table.c.record_count + 1)
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
        paths: dict[int, tuple[tuple[str, ...], str]] = {}
        contents: dict[int, NodeContent] = {}
        # A parent has a smaller sequence than its children, so it comes first.
        for row in self.paths(connection, wanted, "entries_json", "shape_json", "call_json"):
            sequence = int(row["node_sequence"])
            parent = row["parent_sequence"]
            if parent is None:
                paths[sequence] = ((row["entries_json"],), row["shape_json"])
            else:
                above, shape_json = paths[int(parent)]
                paths[sequence] = ((*above, row["entries_json"]), shape_json)
            if sequence in wanted:
                contents[sequence] = NodeContent(*paths[sequence], row["call_json"])
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


class EncodedRecord(NamedTuple):
    """One record in its stored column order, as the ``agent_record`` row.

    A payload with a message list lives in the record's trajectory node as ATIF (``TrajectoryNodes``): then
    ``payload_json`` is ``{}``, ``message_field`` names the list, and ``payload_sha256`` is the SHA-256 of the
    payload's canonical JSON. A payload without one stays in ``payload_json``, and the other two are ``None``. The
    field names bind INSERT parameters by name and identify the fields that participate in retry-content
    comparison: two records have the same content if and only if their encoded fields other than ``created_at``
    are equal.
    """

    agent_record_id: str
    scenario: str
    request_type: str
    created_at: float
    payload_json: str
    message_field: str | None
    payload_sha256: str | None
    references_json: str
    artifact_json: str | None

    @property
    def body_bytes(self) -> int:
        """The UTF-8 bytes of the row's stored bodies: payload, references and artifact."""
        values = (self.payload_json, self.references_json, self.artifact_json)
        return sum(len(value.encode("utf-8")) for value in values if value is not None)


class SQLRecordRetention:
    """Shared retention queries executed inside the caller's transaction.

    The caller selects databases or namespaces and owns the byte-budget
    policy. Adapters must provide accurate row counts for single DELETEs.
    """

    def __init__(self, tables: RecordTables) -> None:
        self._records = tables.records
        self._tables = tables

    def retained_bytes(self, connection: Connection) -> int:
        """Count all stored bodies, record rows and trajectory nodes, independent of training consumption."""
        total = 0
        for table in (self._records, self._tables.nodes):
            total += int(
                connection.execute(
                    select(func.coalesce(func.sum(table.c.body_bytes), 0)).where(self._tables.condition(table))
                ).scalar_one()
            )
        return total

    def unheld(self) -> ColumnElement[bool]:
        """The condition of the record rows that no stored sample holds (``sample_record_table``); eviction takes only
        those, so a stored sample can always render its trajectory."""
        records, held = self._records, self._tables.sample_records
        return ~exists(
            select(held.c.agent_record_id).where(
                held.c.agent_record_id == records.c.agent_record_id,
                *(held.c[name] == records.c[name] for name in self.scope_names()),
            )
        )

    def eviction_sizes(self) -> EvictionSizes:
        """New eviction sizes for this database's records and trajectory nodes."""
        return EvictionSizes(self._tables.nodes, self.scope_names())

    def scope_names(self) -> tuple[str, ...]:
        """The columns of a record row that scope its trajectory node and its loss totals."""
        return tuple(column.name for column in self._tables.eviction.primary_key if column.name != "scenario")

    def page(
        self, connection: Connection, sizes: EvictionSizes, *, after_time: float, after_sequence: int, limit: int = 256
    ) -> tuple[tuple[float, int, int], ...]:
        """Read the oldest records after a position that no stored sample holds, across scenarios, using their stored
        timestamps.

        Return ``(created_at, sequence, size)`` of each record, where ``size`` is the bytes that evicting it frees
        after the records that ``sizes`` counted before; ``sizes`` counts each returned record.
        """
        records = self._records
        rows = (
            connection.execute(
                select(
                    records.c.created_at,
                    records.c.sequence,
                    records.c.body_bytes,
                    records.c.message_field,
                    *(records.c[name] for name in sizes.scope_names),
                )
                .where(
                    self._tables.condition(records),
                    tuple_(records.c.created_at, records.c.sequence) > (after_time, after_sequence),
                    self.unheld(),
                )
                .order_by(records.c.created_at, records.c.sequence)
                .limit(limit)
            )
            .mappings()
            .all()
        )
        sizes.read(connection, rows)
        return tuple((float(row["created_at"]), int(row["sequence"]), sizes.count(row)) for row in rows)

    def evict(self, connection: Connection, sequences: Sequence[int]) -> tuple[tuple[str, RecordLoss], ...]:
        """Delete bodies and persist retry hashes and loss totals in the same transaction.

        The trajectory nodes that no remaining record needs are deleted with the records. A record that a stored sample
        holds stays. The adapter serializes this operation with record writes. Loss totals are per storage generation and scenario; their bytes are the
        bytes that the eviction frees. Metadata never includes payloads.
        """
        tables = self._tables
        rows = (
            connection.execute(
                select(self._records)
                .where(tables.condition(self._records), self._records.c.sequence.in_(sequences), self.unheld())
                .order_by(self._records.c.created_at, self._records.c.sequence)
            )
            .mappings()
            .all()
        )
        if not rows:
            return ()
        sizes = self.eviction_sizes()
        sizes.read(connection, rows)
        freed = {int(row["sequence"]): sizes.count(row) for row in rows}
        groups: dict[tuple[str, ...], list[RowMapping]] = {}
        for row in rows:
            groups.setdefault((*sizes.scope(row), str(row["scenario"])), []).append(row)
        losses: list[tuple[str, RecordLoss]] = []
        for key, members in groups.items():
            scope = dict(zip(sizes.scope_names, key[:-1], strict=True))
            scenario = key[-1]
            nodes = sizes.trajectory_nodes(key[:-1])
            ids = [row["agent_record_id"] for row in members]
            hashes = tables.consumed
            present: set[str] = set(
                connection.execute(
                    select(hashes.c.agent_record_id).where(
                        *(hashes.c[name] == value for name, value in scope.items()),
                        hashes.c.agent_record_id.in_(ids),
                    )
                ).scalars()
            )
            absent = [row for row in members if row["agent_record_id"] not in present]
            if absent:
                records = SQLRecordStore.decode_rows(connection, nodes, absent)
                connection.execute(
                    hashes.insert(),
                    [
                        {**scope, "agent_record_id": record.agent_record_id, "content_sha256": content_sha256(record)}
                        for record in records
                    ],
                )
            nodes.release(connection, (int(row["sequence"]) for row in members if row["message_field"] is not None))
            loss = RecordLoss(
                len(members),
                sum(freed[int(row["sequence"])] for row in members),
                min(int(row["sequence"]) for row in members),
                max(int(row["sequence"]) for row in members),
            )
            state = tables.eviction
            condition = and_(state.c.scenario == scenario, *(state.c[name] == value for name, value in scope.items()))
            previous = connection.execute(select(state).where(condition)).mappings().first()
            if previous is None:
                connection.execute(
                    state.insert().values(
                        **scope,
                        scenario=scenario,
                        record_count=loss.record_count,
                        body_bytes=loss.body_bytes,
                        first_sequence=loss.first_sequence,
                        last_sequence=loss.last_sequence,
                    )
                )
            else:
                connection.execute(
                    state.update()
                    .where(condition)
                    .values(
                        record_count=state.c.record_count + loss.record_count,
                        body_bytes=state.c.body_bytes + loss.body_bytes,
                        first_sequence=min(int(previous["first_sequence"]), loss.first_sequence),
                        last_sequence=max(int(previous["last_sequence"]), loss.last_sequence),
                    )
                )
            losses.append((scenario, loss))
        connection.execute(
            self._records.delete().where(
                tables.condition(self._records), self._records.c.sequence.in_([row["sequence"] for row in rows])
            )
        )
        return tuple(losses)


def content_sha256(item: AgentRecord) -> str:
    """The retry fingerprint of a record: the SHA-256 of its canonical content without ``created_at``.

    The content holds the whole payload, so the fingerprint does not depend on how a store splits the payload,
    and the fingerprints in older stores stay valid. A client retrying with its own agent_record_id regenerates
    the timestamp, so a timestamp difference alone must dedup, not conflict.
    """
    artifact = item.artifact_ref
    content = {
        "agent_record_id": item.agent_record_id,
        "scenario": item.scenario,
        "request_type": item.request_type.value,
        "payload_json": SQLRecordStore._json(dict(item.payload)),
        "references_json": SQLRecordStore._json(item.references),
        "artifact_json": None if artifact is None else SQLRecordStore._json(encode_artifact_ref(artifact)),
    }
    return hashlib.sha256(SQLRecordStore._json(content).encode("utf-8")).hexdigest()


class SQLRecordStore(RecordStore):
    """Implement record behavior using an adapter's SQLAlchemy connections.

    A transaction hook may open a transaction or join one already owned by a
    surrounding store operation. Its connection covers the complete operation;
    the hook must serialize conflicting writers and preserve visible append
    order. In particular, sequence allocation alone is not commit ordering.
    """

    def __init__(self, tables: RecordTables) -> None:
        self._tables = tables
        self._nodes = TrajectoryNodes(ScopedTable(tables.nodes, tables.scope))
        self._live_records: WeakValueDictionary[str, AgentRecord] = WeakValueDictionary()

    @abstractmethod
    def _transaction(self, *, write: bool) -> AbstractContextManager[Connection]:
        """Supply one connection with the isolation and local serialization needed.

        Multi-query receipt reads must observe a consistent state. Write scopes
        must serialize conflicting appends and evictions across connections,
        including sequence allocation through commit. An adapter joining an
        outer transaction leaves its completion to that transaction's owner.
        """

    @abstractmethod
    def _insert_if_absent(
        self,
        connection: Connection,
        table: Table,
        values: Mapping[str, object] | Sequence[Mapping[str, object]],
    ) -> bool:
        """Insert one or more rows, ignoring duplicate keys; report any insertion.

        Only conflicts covered by the table's record identity may be ignored.
        The caller compares canonical content after a duplicate. Adapters must
        return an accurate insertion result without assuming INSERT rowcount is
        available from every driver.
        """

    def _insert(
        self,
        connection: Connection,
        table: Table,
        values: Mapping[str, object] | Sequence[Mapping[str, object]],
    ) -> bool:
        rows = [values] if isinstance(values, Mapping) else values
        return self._insert_if_absent(connection, table, [{**row, **self._tables.scope} for row in rows])

    @staticmethod
    def _json(value: object) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("a record must contain JSON-serializable values") from exc

    @classmethod
    def _encode(cls, item: AgentRecord) -> tuple[EncodedRecord, CallKey | None]:
        """The row of ``item``, and its payload as an ATIF call."""
        artifact = item.artifact_ref
        artifact_json = None
        if artifact is not None:
            artifact_json = cls._json(encode_artifact_ref(artifact))
        key = call_key(item.payload)
        shape = None if key is None else call_shape(item.payload)
        encoded = EncodedRecord(
            agent_record_id=item.agent_record_id,
            scenario=item.scenario,
            request_type=item.request_type.value,
            created_at=item.created_at,
            payload_json="{}" if shape is not None else cls._json(dict(item.payload)),
            message_field=None if shape is None else shape.message_field,
            payload_sha256=None if shape is None else sha256_text(cls._json(dict(item.payload))),
            references_json=cls._json(item.references),
            artifact_json=artifact_json,
        )
        return encoded, key

    @staticmethod
    def decode_rows(
        connection: Connection, nodes: TrajectoryNodes, rows: Sequence[RowMapping]
    ) -> tuple[AgentRecord, ...]:
        """The records of the ``agent_record`` rows ``rows``, with their payloads rendered from ``nodes``."""
        contents = nodes.contents(
            connection, (int(row["sequence"]) for row in rows if row["message_field"] is not None)
        )
        records: list[AgentRecord] = []
        for row in rows:
            raw_artifact = json.loads(row["artifact_json"]) if row["artifact_json"] is not None else None
            artifact = None
            if raw_artifact is not None:
                artifact = decode_artifact_ref(raw_artifact)
            if row["message_field"] is not None:
                payload = provider_payload(contents[int(row["sequence"])].atif_call())
            else:
                payload = json.loads(row["payload_json"])
            records.append(
                AgentRecord(
                    agent_record_id=row["agent_record_id"],
                    scenario=row["scenario"],
                    request_type=RequestType(row["request_type"]),
                    payload=payload,
                    created_at=row["created_at"],
                    references=tuple(json.loads(row["references_json"])),
                    artifact_ref=artifact,
                )
            )
        return tuple(records)

    def _decode(self, connection: Connection, rows: Sequence[RowMapping]) -> tuple[AgentRecord, ...]:
        return self.decode_rows(connection, self._nodes, rows)

    @staticmethod
    def row_content(row: RowMapping) -> EncodedRecord:
        return EncodedRecord(*(row[name] for name in EncodedRecord._fields))

    @classmethod
    def _content(cls, encoded: EncodedRecord) -> dict[str, object]:
        """The encoded fields that define row content, excluding ``created_at`` (``content_sha256``)."""
        return {name: value for name, value in encoded._asdict().items() if name != "created_at"}

    def append(self, item: AgentRecord) -> AgentRecord:
        return self.append_result(item).item

    def existing_receipt(self, item: AgentRecord) -> AgentRecord | None:
        """Validate a retry before applying admission rules for new records.

        Compacted records retain content hashes, so an already accepted
        instruction can still be retried after the training mode changes.
        This lookup never appends data or changes its receipt.
        """
        encoded, _ = self._encode(item)
        with self._transaction(write=False) as connection:
            consumed = (
                connection.execute(
                    select(self._tables.consumed.c.content_sha256).where(
                        self._tables.condition(self._tables.consumed),
                        self._tables.consumed.c.agent_record_id == item.agent_record_id,
                    )
                )
                .mappings()
                .first()
            )
            if consumed is not None:
                if consumed["content_sha256"] != content_sha256(item):
                    raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
                return item
            row = (
                connection.execute(
                    select(self._tables.records).where(
                        self._tables.condition(self._tables.records),
                        self._tables.records.c.agent_record_id == item.agent_record_id,
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            if self._content(self.row_content(row)) != self._content(encoded):
                raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
            return self._decode(connection, [row])[0]

    def append_result(self, item: AgentRecord) -> AppendResult:
        return self.append_many((item,))[0]

    def append_many(self, items: Sequence[AgentRecord]) -> tuple[AppendResult, ...]:
        if not items:
            return ()
        scenario = items[0].scenario
        if any(item.scenario != scenario for item in items):
            raise ValueError("a record batch must belong to one scenario")
        with self._transaction(write=True) as connection:
            return tuple(self.append_in_transaction(connection, item) for item in items)

    def append_in_transaction(self, connection: Connection, item: AgentRecord) -> AppendResult:
        """Apply the same retry and capacity-eviction rules to single and batch writes."""
        encoded, key = self._encode(item)
        consumed = (
            connection.execute(
                select(self._tables.consumed.c.content_sha256).where(
                    self._tables.condition(self._tables.consumed),
                    self._tables.consumed.c.agent_record_id == item.agent_record_id,
                )
            )
            .mappings()
            .first()
        )
        if consumed is not None:
            if consumed["content_sha256"] != content_sha256(item):
                raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
            return AppendResult(item, False)
        if item.request_type is RequestType.REPORT and item.references:
            evicted_reference = connection.execute(
                select(self._tables.consumed.c.agent_record_id)
                .where(
                    self._tables.condition(self._tables.consumed),
                    self._tables.consumed.c.agent_record_id.in_(item.references),
                    ~select(self._tables.records.c.agent_record_id)
                    .where(
                        self._tables.condition(self._tables.records),
                        self._tables.records.c.agent_record_id == self._tables.consumed.c.agent_record_id,
                    )
                    .exists(),
                )
                .limit(1)
            ).first()
            if evicted_reference is not None:
                # A report is discarded once its references are gone, but a row
                # already stored under this id stays canonical: check the discard
                # against that row so a divergent retry cannot register its own
                # content as the receipt and reject the honest retry that follows.
                existing = (
                    connection.execute(
                        select(self._tables.records).where(
                            self._tables.condition(self._tables.records),
                            self._tables.records.c.agent_record_id == item.agent_record_id,
                        )
                    )
                    .mappings()
                    .first()
                )
                if existing is not None and self._content(self.row_content(existing)) != self._content(encoded):
                    raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
                self._insert(
                    connection,
                    self._tables.consumed,
                    {"agent_record_id": item.agent_record_id, "content_sha256": content_sha256(item)},
                )
                return AppendResult(item, False)
        inserted = self._insert(
            connection, self._tables.records, {**encoded._asdict(), "body_bytes": encoded.body_bytes}
        )
        if inserted:
            if key is not None:
                sequence = connection.execute(
                    select(self._tables.records.c.sequence).where(
                        self._tables.condition(self._tables.records),
                        self._tables.records.c.agent_record_id == item.agent_record_id,
                    )
                ).scalar_one()
                self._nodes.add(connection, int(sequence), item.scenario, item.agent_record_id, item.payload, key)
            self._live_records[item.agent_record_id] = item
            return AppendResult(item, True)
        existing = (
            connection.execute(
                select(self._tables.records).where(
                    self._tables.condition(self._tables.records),
                    self._tables.records.c.agent_record_id == item.agent_record_id,
                )
            )
            .mappings()
            .first()
        )
        if existing is None or self._content(self.row_content(existing)) != self._content(encoded):
            raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
        stored = self._decode(connection, [existing])[0]
        live = self._live_records.get(item.agent_record_id)
        # An outer transaction can roll back after caching the inserted
        # object. Reuse it only while the persisted record still matches.
        if live is not None and live == stored:
            return AppendResult(live, False)
        self._live_records[item.agent_record_id] = stored
        return AppendResult(stored, False)

    def get(self, scenario: str, agent_record_id: str) -> AgentRecord | None:
        """Read a record still visible to training, scoped to its scenario."""
        with self._transaction(write=False) as connection:
            row = (
                connection.execute(
                    select(self._tables.records).where(
                        self._tables.condition(self._tables.records),
                        self._tables.records.c.scenario == scenario,
                        self._tables.records.c.agent_record_id == agent_record_id,
                    )
                )
                .mappings()
                .first()
            )
            return None if row is None else self._decode(connection, [row])[0]

    def replay(
        self,
        scenario: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[AgentRecord, ...]:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return ()
        statement = (
            select(self._tables.records)
            .where(
                self._tables.condition(self._tables.records),
                self._tables.records.c.scenario == scenario,
            )
            .order_by(self._tables.records.c.sequence)
            .limit(limit)
            .offset(offset)
        )
        with self._transaction(write=False) as connection:
            rows = connection.execute(statement).mappings().all()
            return self._decode(connection, rows)

    def replay_page(
        self,
        scenario: str,
        *,
        after_sequence: int = 0,
        limit: int = 256,
    ) -> tuple[tuple[int, AgentRecord], ...]:
        """Read a bounded keyset page for internal streaming consumers."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._transaction(write=False) as connection:
            rows = (
                connection.execute(
                    select(self._tables.records)
                    .where(
                        self._tables.condition(self._tables.records),
                        self._tables.records.c.scenario == scenario,
                        self._tables.records.c.sequence > after_sequence,
                    )
                    .order_by(self._tables.records.c.sequence)
                    .limit(limit)
                )
                .mappings()
                .all()
            )
            records = self._decode(connection, rows)
        return tuple((int(row["sequence"]), record) for row, record in zip(rows, records, strict=True))

    def count(self, scenario: str, *, request_type: RequestType | None = None, after_sequence: int = 0) -> int:
        """Count training-visible records, optionally by type and after an append sequence."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        statement = (
            select(func.count())
            .select_from(self._tables.records)
            .where(
                self._tables.condition(self._tables.records),
                self._tables.records.c.scenario == scenario,
                self._tables.records.c.sequence > after_sequence,
            )
        )
        if request_type is not None:
            statement = statement.where(self._tables.records.c.request_type == request_type.value)
        with self._transaction(write=False) as connection:
            return int(connection.execute(statement).scalar_one())

    def last_sequence(self, scenario: str) -> int:
        statement = select(func.max(self._tables.records.c.sequence)).where(
            self._tables.condition(self._tables.records), self._tables.records.c.scenario == scenario
        )
        with self._transaction(write=False) as connection:
            return int(connection.execute(statement).scalar_one() or 0)

    def get_for_audit(self, scenario: str, agent_record_id: str) -> StoredRecord | None:
        """Read a retained record with its append sequence."""
        with self._transaction(write=False) as connection:
            row = (
                connection.execute(
                    select(self._tables.records).where(
                        self._tables.condition(self._tables.records),
                        self._tables.records.c.scenario == scenario,
                        self._tables.records.c.agent_record_id == agent_record_id,
                    )
                )
                .mappings()
                .first()
            )
            return None if row is None else self._audit_records(connection, [row])[0]

    def audit_page(
        self,
        scenario: str,
        *,
        after_sequence: int = 0,
        limit: int = 256,
        request_type: RequestType | None = None,
    ) -> tuple[StoredRecord, ...]:
        """Read a bounded append-order page including consumed bodies, scoped to one scenario and optional type."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        statement = select(self._tables.records).where(
            self._tables.condition(self._tables.records),
            self._tables.records.c.scenario == scenario,
            self._tables.records.c.sequence > after_sequence,
        )
        if request_type is not None:
            statement = statement.where(self._tables.records.c.request_type == request_type.value)
        with self._transaction(write=False) as connection:
            rows = (
                connection.execute(statement.order_by(self._tables.records.c.sequence).limit(limit)).mappings().all()
            )
            return self._audit_records(connection, rows)

    def _audit_records(self, connection: Connection, rows: Sequence[RowMapping]) -> tuple[StoredRecord, ...]:
        records = self._decode(connection, rows)
        return tuple(
            StoredRecord(sequence=int(row["sequence"]), item=record) for row, record in zip(rows, records, strict=True)
        )

    def trajectory(self, scenario: str, agent_record_id: str) -> tuple[str, ...]:
        """The record IDs of the trajectory that the record ``agent_record_id`` of ``scenario`` ends, oldest first.

        The calls of the trajectory are the record and the records on its node path whose messages are a strict
        prefix of the next call's messages; a record whose messages equal the next call's (a retry) is left out
        (``trajectory_calls``). An ID can name a record that capacity eviction deleted while a later record needs
        its messages. A stored record without a message list is a trajectory of one call; return ``()`` for a
        record that the store keeps neither as a record nor as a node.
        """
        nodes = self._tables.nodes
        with self._transaction(write=False) as connection:
            sequence = connection.execute(
                select(nodes.c.node_sequence).where(
                    self._tables.condition(nodes),
                    nodes.c.scenario == scenario,
                    nodes.c.agent_record_id == agent_record_id,
                )
            ).scalar_one_or_none()
            if sequence is None:
                stored = connection.execute(
                    select(self._tables.records.c.sequence).where(
                        self._tables.condition(self._tables.records),
                        self._tables.records.c.scenario == scenario,
                        self._tables.records.c.agent_record_id == agent_record_id,
                    )
                ).first()
                return () if stored is None else (agent_record_id,)
            rows = self._nodes.paths(connection, [int(sequence)], "agent_record_id", "entry_count")
        by_sequence = {int(row["node_sequence"]): row for row in rows}
        path: list[tuple[str, int]] = []
        node: int | None = int(sequence)
        while node is not None:
            row = by_sequence[node]
            path.append((str(row["agent_record_id"]), int(row["entry_count"])))
            node = None if row["parent_sequence"] is None else int(row["parent_sequence"])
        return trajectory_calls(path[::-1])

    def records_by_id(
        self, connection: Connection, scenario: str, agent_record_ids: Iterable[str]
    ) -> dict[str, AgentRecord]:
        """The stored records of ``scenario`` among ``agent_record_ids``, by ID, read in ``connection``; an ID that the
        store does not keep is absent."""
        ids = list(dict.fromkeys(agent_record_ids))
        if not ids:
            return {}
        records = self._tables.records
        rows = (
            connection.execute(
                select(records).where(
                    self._tables.condition(records), records.c.scenario == scenario, records.c.agent_record_id.in_(ids)
                )
            )
            .mappings()
            .all()
        )
        return {record.agent_record_id: record for record in self._decode(connection, rows)}

    def hold_records(self, connection: Connection, sample_id: str, agent_record_ids: Iterable[str]) -> None:
        """Write in ``connection`` that the stored sample ``sample_id`` holds the records ``agent_record_ids``:
        capacity eviction keeps them (``sample_record_table``)."""
        rows = [
            {**self._tables.scope, "sample_id": sample_id, "agent_record_id": record_id}
            for record_id in dict.fromkeys(agent_record_ids)
        ]
        if rows:
            self._insert_if_absent(connection, self._tables.sample_records, rows)

    def loss(self, scenario: str) -> RecordLoss:
        table = self._tables.eviction
        with self._transaction(write=False) as connection:
            row = (
                connection.execute(select(table).where(self._tables.condition(table), table.c.scenario == scenario))
                .mappings()
                .first()
            )
        if row is None:
            return RecordLoss()
        return RecordLoss(
            *(int(row[name]) for name in ("record_count", "body_bytes", "first_sequence", "last_sequence"))
        )

    def read_settings(self) -> ScenarioSettings:
        """The scenario settings kept with these records; defaults when none are stored."""
        table = self._tables.settings
        with self._transaction(write=False) as connection:
            rows = connection.execute(select(table.c.name, table.c.value_json).where(self._tables.condition(table)))
            values = {name: json.loads(value_json) for name, value_json in rows}
        return ScenarioSettings(
            model_config=values.get("model_config"),
            training_mode=values.get("training_mode"),
            artifact_evaluation=values.get("artifact_evaluation"),
            activation=values.get("activation"),
            owner=values.get("owner"),
        )

    def write_settings(self, settings: ScenarioSettings) -> None:
        """Replace the stored settings in one transaction; a ``None`` setting is not stored."""
        table = self._tables.settings
        values = {
            "model_config": None if settings.model_config is None else dict(settings.model_config),
            "training_mode": settings.training_mode,
            "artifact_evaluation": (
                None if settings.artifact_evaluation is None else dict(settings.artifact_evaluation)
            ),
            "activation": None if settings.activation is None else dict(settings.activation),
            "owner": settings.owner,
        }
        rows = [
            {**self._tables.scope, "name": name, "value_json": json.dumps(value, ensure_ascii=False)}
            for name, value in values.items()
            if value is not None
        ]
        with self._transaction(write=True) as connection:
            connection.execute(table.delete().where(self._tables.condition(table)))
            if rows:
                connection.execute(table.insert(), rows)

    def consumption_receipts(self, scenario: str) -> tuple[ConsumptionReceipt, ...]:
        """Read ordered consumption metadata for this scenario."""
        with self._transaction(write=False) as connection:
            rows = (
                connection.execute(
                    select(self._tables.consumption)
                    .where(
                        self._tables.condition(self._tables.consumption),
                        self._tables.consumption.c.scenario == scenario,
                    )
                    .order_by(self._tables.consumption.c.recorded_at, self._tables.consumption.c.receipt_id)
                )
                .mappings()
                .all()
            )
        receipts: list[ConsumptionReceipt] = []
        for row in rows:
            consumed_ids = json.loads(row["consumed_ids_json"])
            metadata = json.loads(row["metadata_json"])
            if not isinstance(metadata, dict):
                raise ValueError("consumption receipt metadata must be an object")
            if not isinstance(consumed_ids, list) or any(not isinstance(record_id, str) for record_id in consumed_ids):
                raise ValueError("consumption receipt IDs must be a list of strings")
            receipts.append(
                {
                    "receipt_id": row["receipt_id"],
                    "consumed_ids": tuple(consumed_ids),
                    "metadata": metadata,
                    "recorded_at": float(row["recorded_at"]),
                }
            )
        return tuple(receipts)
