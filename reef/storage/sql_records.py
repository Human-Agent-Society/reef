"""Shared SQLAlchemy record operations, independent of database setup and dialect.

Adapters supply tables, transaction-scoped connections, conflict insertion, and
resource cleanup. The domain behavior is implemented here once; this module
has no SQLite, filesystem, or scenario lifecycle dependencies.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import NamedTuple
from weakref import WeakValueDictionary

from sqlalchemy import Table, and_, func, select, true, tuple_
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.sql.elements import ColumnElement

from reef.core.artifact_ref import decode_artifact_ref, encode_artifact_ref
from reef.core.records_types import AgentRecord, RequestType
from reef.storage.records import (
    AppendResult,
    ConsumptionReceipt,
    RecordConflict,
    RecordLoss,
    RecordStore,
    StoredRecord,
)


@dataclass(frozen=True)
class RecordTables:
    """Tables for record bodies, retry hashes, consumption, and capacity losses.

    The adapter owns schema types, constraints, indexes, and migrations. Table
    columns retain the names consumed by SQLRecordStore; backend-specific
    generated keys or indexes may be added without replacing record operations.
    """

    records: Table
    consumed: Table
    consumption: Table
    eviction: Table
    scope: Mapping[str, str] = field(default_factory=dict)

    def condition(self, table: Table) -> ColumnElement[bool]:
        """Restrict operations to the adapter's storage namespace, if any."""
        return and_(true(), *(table.c[key] == value for key, value in self.scope.items()))


class EncodedRecord(NamedTuple):
    """One record in its stored column order, as the ``agent_record`` row.

    The field names bind INSERT parameters by name and identify the fields
    that participate in retry-content comparison.
    """

    agent_record_id: str
    scenario: str
    request_type: str
    created_at: float
    payload_json: str
    references_json: str
    artifact_json: str | None


class SQLRecordRetention:
    """Shared retention queries executed inside the caller's transaction.

    The caller selects databases or namespaces and owns the byte-budget
    policy. Adapters must provide accurate row counts for single DELETEs.
    """

    def __init__(self, tables: RecordTables) -> None:
        self._records = tables.records
        self._tables = tables

    def retained_bytes(self, connection: Connection) -> int:
        """Count all stored JSON bodies, independent of training consumption."""
        return int(
            connection.execute(
                select(func.coalesce(func.sum(self._records.c.body_bytes), 0)).where(
                    self._tables.condition(self._records)
                )
            ).scalar_one()
        )

    def page(
        self, connection: Connection, *, after_time: float, after_sequence: int, limit: int = 256
    ) -> tuple[tuple[float, int, int], ...]:
        """Read oldest records across scenarios using their stored timestamps."""
        rows = connection.execute(
            select(self._records.c.created_at, self._records.c.sequence, self._records.c.body_bytes)
            .where(
                self._tables.condition(self._records),
                tuple_(self._records.c.created_at, self._records.c.sequence) > (after_time, after_sequence),
            )
            .order_by(self._records.c.created_at, self._records.c.sequence)
            .limit(limit)
        ).all()
        return tuple((float(created_at), int(sequence), int(size)) for created_at, sequence, size in rows)

    def evict(self, connection: Connection, sequences: Sequence[int]) -> tuple[tuple[str, RecordLoss], ...]:
        """Delete bodies and persist retry hashes and loss totals in the same transaction.

        The adapter serializes this operation with record writes. Loss totals are
        per storage generation and scenario; metadata never includes payloads.
        """
        tables = self._tables
        rows = (
            connection.execute(
                select(self._records).where(tables.condition(self._records), self._records.c.sequence.in_(sequences))
            )
            .mappings()
            .all()
        )
        if not rows:
            return ()
        groups: dict[tuple[str, ...], list[RowMapping]] = {}
        scope_names = tuple(column.name for column in tables.eviction.primary_key if column.name != "scenario")
        for row in rows:
            key = tuple(str(row[name]) for name in (*scope_names, "scenario"))
            groups.setdefault(key, []).append(row)
        losses: list[tuple[str, RecordLoss]] = []
        for key, members in groups.items():
            scope = dict(zip(scope_names, key[:-1], strict=True))
            scenario = key[-1]
            ids = [row["agent_record_id"] for row in members]
            hashes = tables.consumed
            present = set(
                connection.execute(
                    select(hashes.c.agent_record_id).where(
                        *(hashes.c[name] == value for name, value in scope.items()),
                        hashes.c.agent_record_id.in_(ids),
                    )
                ).scalars()
            )
            missing = [
                {
                    **scope,
                    "agent_record_id": row["agent_record_id"],
                    "content_sha256": SQLRecordStore._content_sha256(SQLRecordStore._row_content(row)),
                }
                for row in members
                if row["agent_record_id"] not in present
            ]
            if missing:
                connection.execute(hashes.insert(), missing)
            loss = RecordLoss(
                len(members),
                sum(int(row["body_bytes"]) for row in members),
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


class SQLRecordStore(RecordStore):
    """Implement record behavior using an adapter's SQLAlchemy connections.

    A transaction hook may open a transaction or join one already owned by a
    surrounding store operation. Its connection covers the complete operation;
    the hook must serialize conflicting writers and preserve visible append
    order. In particular, sequence allocation alone is not commit ordering.
    """

    def __init__(self, tables: RecordTables) -> None:
        self._tables = tables
        self._live_records: WeakValueDictionary[str, AgentRecord] = WeakValueDictionary()

    @abstractmethod
    def _transaction(self, scenario: str, *, write: bool) -> AbstractContextManager[Connection]:
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
    def _encode(cls, item: AgentRecord) -> EncodedRecord:
        artifact = item.artifact_ref
        artifact_json = None
        if artifact is not None:
            artifact_json = cls._json(encode_artifact_ref(artifact))
        return EncodedRecord(
            agent_record_id=item.agent_record_id,
            scenario=item.scenario,
            request_type=item.request_type.value,
            created_at=item.created_at,
            payload_json=cls._json(dict(item.payload)),
            references_json=cls._json(item.references),
            artifact_json=artifact_json,
        )

    @staticmethod
    def _decode(row: RowMapping) -> AgentRecord:
        raw_artifact = json.loads(row["artifact_json"]) if row["artifact_json"] is not None else None
        artifact = None
        if raw_artifact is not None:
            artifact = decode_artifact_ref(raw_artifact)
        return AgentRecord(
            agent_record_id=row["agent_record_id"],
            scenario=row["scenario"],
            request_type=RequestType(row["request_type"]),
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            references=tuple(json.loads(row["references_json"])),
            artifact_ref=artifact,
        )

    @staticmethod
    def _row_content(row: RowMapping) -> EncodedRecord:
        return EncodedRecord(*(row[name] for name in EncodedRecord._fields))

    @classmethod
    def _content(cls, encoded: EncodedRecord) -> dict[str, object]:
        """The encoded fields that define row content, excluding ``created_at``.

        A client retrying with its own agent_record_id regenerates the
        timestamp, so a timestamp difference alone must dedup, not conflict.
        """
        return {name: value for name, value in encoded._asdict().items() if name != "created_at"}

    @classmethod
    def _content_sha256(cls, encoded: EncodedRecord) -> str:
        canonical = cls._json(cls._content(encoded)).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def append(self, item: AgentRecord) -> AgentRecord:
        return self.append_result(item).item

    def existing_receipt(self, item: AgentRecord) -> AgentRecord | None:
        """Validate a retry before applying admission rules for new records.

        Compacted records retain content hashes, so an already accepted
        instruction can still be retried after the training mode changes.
        This lookup never appends data or changes its receipt.
        """
        encoded = self._encode(item)
        with self._transaction(item.scenario, write=False) as connection:
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
                if consumed["content_sha256"] != self._content_sha256(encoded):
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
            if self._content(self._row_content(row)) != self._content(encoded):
                raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
            return self._decode(row)

    def append_result(self, item: AgentRecord) -> AppendResult:
        return self.append_many((item,))[0]

    def append_many(self, items: Sequence[AgentRecord]) -> tuple[AppendResult, ...]:
        if not items:
            return ()
        scenario = items[0].scenario
        if any(item.scenario != scenario for item in items):
            raise ValueError("a record batch must belong to one scenario")
        with self._transaction(scenario, write=True) as connection:
            return tuple(self.append_in_transaction(connection, item) for item in items)

    def append_in_transaction(self, connection: Connection, item: AgentRecord) -> AppendResult:
        """Apply the same retry and capacity-eviction rules to single and batch writes."""
        encoded = self._encode(item)
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
            if consumed["content_sha256"] != self._content_sha256(encoded):
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
                if existing is not None and self._content(self._row_content(existing)) != self._content(encoded):
                    raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
                self._insert(
                    connection,
                    self._tables.consumed,
                    {"agent_record_id": item.agent_record_id, "content_sha256": self._content_sha256(encoded)},
                )
                return AppendResult(item, False)
        inserted = self._insert(
            connection,
            self._tables.records,
            {
                **encoded._asdict(),
                "body_bytes": sum(len(value.encode("utf-8")) for value in encoded[4:] if value is not None),
            },
        )
        if inserted:
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
        if existing is None or self._content(self._row_content(existing)) != self._content(encoded):
            raise RecordConflict(f"agent_record_id {item.agent_record_id!r} already has different content")
        stored = self._decode(existing)
        live = self._live_records.get(item.agent_record_id)
        # An outer transaction can roll back after caching the inserted
        # object. Reuse it only while the persisted record still matches.
        if live is not None and live == stored:
            return AppendResult(live, False)
        self._live_records[item.agent_record_id] = stored
        return AppendResult(stored, False)

    def get(self, scenario: str, agent_record_id: str) -> AgentRecord | None:
        """Read a record still visible to training, scoped to its scenario."""
        with self._transaction(scenario, write=False) as connection:
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
        return None if row is None else self._decode(row)

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
        with self._transaction(scenario, write=False) as connection:
            rows = connection.execute(statement).mappings().all()
        return tuple(self._decode(row) for row in rows)

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
        with self._transaction(scenario, write=False) as connection:
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
        return tuple((int(row["sequence"]), self._decode(row)) for row in rows)

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
        with self._transaction(scenario, write=False) as connection:
            return int(connection.execute(statement).scalar_one())

    def get_for_audit(self, scenario: str, agent_record_id: str) -> StoredRecord | None:
        """Read a retained record with its append sequence."""
        with self._transaction(scenario, write=False) as connection:
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
        return None if row is None else self._audit_record(row)

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
        with self._transaction(scenario, write=False) as connection:
            rows = (
                connection.execute(statement.order_by(self._tables.records.c.sequence).limit(limit)).mappings().all()
            )
        return tuple(self._audit_record(row) for row in rows)

    @classmethod
    def _audit_record(cls, row: RowMapping) -> StoredRecord:
        return StoredRecord(sequence=int(row["sequence"]), item=cls._decode(row))

    def record_consumption(
        self,
        scenario: str,
        agent_record_ids: frozenset[str],
        *,
        receipt_id: str,
        metadata: Mapping[str, object],
    ) -> None:
        """Persist a skipped batch's consumption without modifying its records."""
        if not receipt_id:
            raise ValueError("consumption receipt_id must be non-empty")
        consumed_ids_json = self._json(sorted(agent_record_ids))
        metadata_json = self._json(dict(metadata))
        table = self._tables.consumption
        with self._transaction(scenario, write=True) as connection:
            self._insert(
                connection,
                table,
                {
                    "scenario": scenario,
                    "receipt_id": receipt_id,
                    "consumed_ids_json": consumed_ids_json,
                    "metadata_json": metadata_json,
                    "recorded_at": time.time(),
                    "consumed_ids_sha256": hashlib.sha256(consumed_ids_json.encode()).hexdigest(),
                },
            )
            existing = connection.execute(
                select(table.c.metadata_json).where(
                    self._tables.condition(table),
                    table.c.scenario == scenario,
                    table.c.receipt_id == receipt_id,
                    table.c.consumed_ids_json == consumed_ids_json,
                )
            ).scalar_one_or_none()
            if existing != metadata_json:
                raise RecordConflict(
                    f"consumption receipt {receipt_id!r} for scenario {scenario!r} has different content"
                )

    def loss(self, scenario: str) -> RecordLoss:
        table = self._tables.eviction
        with self._transaction(scenario, write=False) as connection:
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

    def consumption_receipts(self, scenario: str) -> tuple[ConsumptionReceipt, ...]:
        """Read ordered consumption metadata for this scenario."""
        with self._transaction(scenario, write=False) as connection:
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
