"""Storage-neutral record contracts, reusable by future RecordStore adapters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import Column, Float, Integer, MetaData, Table, Text, create_engine
from sqlalchemy.engine import Connection

from reef.core.records_types import AgentRecord, RequestType
from reef.storage.postgres import PostgresRecordStore
from reef.storage.records import AppendResult, RecordConflict, RecordStore
from reef.storage.sql_records import RecordTables, SQLRecordStore
from reef.storage.sqlite import SQLiteRecordStore


@pytest.fixture(params=("memory", "file", "postgres"))
def records(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[RecordStore]:
    store: RecordStore
    if request.param == "postgres":
        store = PostgresRecordStore(request.getfixturevalue("postgres_database"))
    else:
        store = SQLiteRecordStore(None if request.param == "memory" else tmp_path / "records.sqlite3")
    try:
        yield store
    finally:
        store.close()
        store.close()


def record(record_id: str, scenario: str = "math", *, references: tuple[str, ...] = ()) -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=record_id,
        scenario=scenario,
        request_type=RequestType.REPORT if references else RequestType.INFERENCE,
        payload={"value": record_id},
        references=references,
        created_at=123.0,
    )


def test_append_retry_and_scenario_reads(records: RecordStore) -> None:
    original = record("first")
    assert records.existing_receipt(original) is None
    assert records.append_result(original) == AppendResult(original, True)
    assert records.append_result(replace(original, created_at=456.0)).inserted is False
    assert records.existing_receipt(original) == original
    with pytest.raises(RecordConflict):
        records.append(replace(original, scenario="code"))
    with pytest.raises(RecordConflict):
        records.existing_receipt(replace(original, payload={"value": "changed"}))

    other = record("other", "code")
    report = record("report", references=(original.agent_record_id,))
    records.append(other)
    records.append(report)
    page = records.replay_page("math", limit=1)
    assert len(page) == 1
    sequence, stored = page[0]
    assert stored == original
    assert records.replay_page("math", after_sequence=sequence)[0][1] == report
    assert records.replay("math", offset=1, limit=1) == (report,)
    assert records.count("math") == 2
    assert records.count("math", request_type=RequestType.REPORT) == 1
    assert records.count("math", after_sequence=sequence) == 1
    assert records.get("code", original.agent_record_id) is None
    assert records.get_for_audit("code", original.agent_record_id) is None
    assert records.replay("code") == (other,)


class TransactionRecordStore(SQLRecordStore):
    """Exercise shared SQL behavior with caller-owned tables and transactions."""

    def __init__(self, tables: RecordTables, connection: Connection) -> None:
        super().__init__(tables)
        self.connection = connection
        self.closed = False

    def _transaction(self, scenario: str, *, write: bool) -> AbstractContextManager[Connection]:
        if self.closed or not self.connection.in_transaction():
            raise RuntimeError("an open caller-owned transaction is required")
        return nullcontext(self.connection)

    def _insert_if_absent(
        self,
        connection: Connection,
        table: Table,
        values: Mapping[str, object] | Sequence[Mapping[str, object]],
    ) -> bool:
        return connection.execute(table.insert().prefix_with("OR IGNORE"), values).rowcount > 0

    def close(self) -> None:
        self.closed = True


def custom_record_tables(metadata: MetaData) -> RecordTables:
    return RecordTables(
        Table(
            "custom_records",
            metadata,
            Column("sequence", Integer, primary_key=True),
            Column("agent_record_id", Text, unique=True, nullable=False),
            Column("scenario", Text, nullable=False),
            Column("request_type", Text, nullable=False),
            Column("created_at", Float, nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("references_json", Text, nullable=False),
            Column("artifact_json", Text),
            Column("body_bytes", Integer, nullable=False),
            sqlite_autoincrement=True,
        ),
        Table(
            "custom_consumed",
            metadata,
            Column("agent_record_id", Text, primary_key=True),
            Column("content_sha256", Text, nullable=False),
        ),
        Table(
            "custom_receipts",
            metadata,
            Column("scenario", Text, primary_key=True),
            Column("receipt_id", Text, primary_key=True),
            Column("consumed_ids_json", Text, nullable=False),
            Column("consumed_ids_sha256", Text, primary_key=True),
            Column("metadata_json", Text, nullable=False),
            Column("recorded_at", Float, nullable=False),
        ),
        Table(
            "custom_eviction",
            metadata,
            Column("scenario", Text, primary_key=True),
            Column("record_count", Integer, nullable=False),
            Column("body_bytes", Integer, nullable=False),
            Column("first_sequence", Integer, nullable=False),
            Column("last_sequence", Integer, nullable=False),
        ),
    )


def test_shared_sql_operations_use_supplied_tables_and_join_the_outer_transaction() -> None:
    metadata = MetaData()
    tables = custom_record_tables(metadata)
    # SQLite executes this test adapter; this is SQL reuse coverage, not PostgreSQL certification.
    engine = create_engine("sqlite://")
    try:
        with engine.connect() as connection:
            metadata.create_all(connection)
            connection.commit()
            records = TransactionRecordStore(tables, connection)
            original = record("first")
            with pytest.raises(RuntimeError, match="abort outer transaction"), connection.begin():
                records.append(original)
                records.record_consumption("math", frozenset({"first"}), receipt_id="step", metadata={"step": 1})
                assert len(records.consumption_receipts("math")) == 1
                raise RuntimeError("abort outer transaction")

            committed = replace(original, payload={"value": "replacement"})
            with connection.begin():
                assert records.audit_page("math") == ()
                assert records.consumption_receipts("math") == ()
                assert records.existing_receipt(original) is None
                assert records.append_result(committed).inserted
                records.record_consumption("math", frozenset({"first"}), receipt_id="step", metadata={"step": 2})
                records.append(record("second"))

            with connection.begin():
                assert records.append_result(committed).inserted is False
                assert [entry.agent_record_id for entry in records.replay("math")] == ["first", "second"]
                assert [entry.item.agent_record_id for entry in records.audit_page("math")] == ["first", "second"]
                assert records.consumption_receipts("math")[0]["metadata"] == {"step": 2}
            records.close()
    finally:
        engine.dispose()


@pytest.mark.parametrize("changed_payload", (False, True), ids=("same-payload", "changed-payload"))
def test_retry_after_outer_rollback_returns_the_other_stores_canonical_record(changed_payload: bool) -> None:
    metadata = MetaData()
    tables = custom_record_tables(metadata)
    engine = create_engine("sqlite://")
    try:
        with engine.connect() as connection:
            metadata.create_all(connection)
            connection.commit()
            first = TransactionRecordStore(tables, connection)
            other = TransactionRecordStore(tables, connection)
            original = record("reused")
            transaction = connection.begin()
            assert first.append_result(original) == AppendResult(original, True)
            transaction.rollback()

            canonical = replace(
                original,
                payload={"value": "replacement"} if changed_payload else original.payload,
                created_at=456.0,
            )
            with connection.begin():
                assert other.append_result(canonical) == AppendResult(canonical, True)

            with connection.begin():
                retry = first.append_result(canonical)
                assert retry.inserted is False
                assert retry.item.payload == canonical.payload
                assert retry.item.created_at == 456.0
                assert retry.item == canonical
            first.close()
            other.close()
    finally:
        engine.dispose()


def test_batch_commit_rollback_and_retired_retries(records: RecordStore) -> None:
    first, second = record("first"), record("second")
    with pytest.raises(RecordConflict):
        records.append_many([first, second, replace(first, payload={"changed": True})])
    assert records.count("math") == 0
    assert records.existing_receipt(first) is None
    assert records.append_many([first, second, first]) == (
        AppendResult(first, True),
        AppendResult(second, True),
        AppendResult(first, False),
    )
    assert [item.agent_record_id for _, item in records.replay_page("math")] == ["first", "second"]
    with pytest.raises(ValueError, match="one scenario"):
        records.append_many([record("third"), record("other", "other")])
    assert records.count("math") == 2
    records.record_consumption("math", frozenset({"first"}), receipt_id="skip", metadata={})
    assert records.append_many([first])[0].inserted is False
    with pytest.raises(RecordConflict):
        records.append_many([record("third"), replace(first, payload={"changed": True})])
    assert records.get("math", "third") is None


def test_batch_uses_one_write_transaction_and_observes_only_committed_records(tmp_path):
    from sqlalchemy import event

    from reef.storage.observer import ObservedRecordStore, RecordObserver

    class Observer(RecordObserver):
        def __init__(self):
            self.accepted = []

        def record_accepted(self, item):
            self.accepted.append(item.agent_record_id)

    inner = SQLiteRecordStore(tmp_path / "batch.sqlite3")
    observer = Observer()
    store = ObservedRecordStore(inner, observer)
    transactions = []

    def committed(connection):
        transactions.append("commit")

    event.listen(inner._engine, "commit", committed)
    try:
        store.append_many([record(str(index)) for index in range(100)])
        assert transactions == ["commit"]
        assert observer.accepted == [str(index) for index in range(100)]
        with pytest.raises(RecordConflict):
            store.append_many([record("new"), replace(record("0"), payload={"changed": True})])
        assert "new" not in observer.accepted
        assert transactions == ["commit"]
    finally:
        store.close()


def test_consumption_receipts_preserve_bodies_and_reject_conflicting_retries(records: RecordStore) -> None:
    original = record("source")
    records.append(original)
    ids = frozenset({"source"})
    metadata = {"outcome": "stale", "metrics": {"dropped": 1}}
    records.record_consumption("math", ids, receipt_id="batch-1", metadata=metadata)
    [receipt] = records.consumption_receipts("math")
    assert receipt["consumed_ids"] == ("source",)
    assert receipt["metadata"] == metadata
    records.record_consumption("math", ids, receipt_id="batch-1", metadata=metadata)
    assert records.consumption_receipts("math") == (receipt,)
    with pytest.raises(RecordConflict, match="different content"):
        records.record_consumption("math", ids, receipt_id="batch-1", metadata={"outcome": "changed"})
    assert records.consumption_receipts("math") == (receipt,)
    assert records.consumption_receipts("code") == ()
    assert records.get("math", "source") == original
    assert records.replay("math") == (original,)
    assert records.append_result(original).inserted is False
    # Storage accepts feedback independently of whether a consumer processed the source.
    assert records.append_result(record("late", references=("source",))).inserted
    with pytest.raises(ValueError, match="non-empty"):
        records.record_consumption("math", ids, receipt_id="", metadata={})
