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


def test_retirement_purge_and_retry_receipts(records: RecordStore) -> None:
    original = record("first")
    other = record("other", "code")
    records.append(original)
    records.append(other)
    before_compaction = records.audit_page("math")[0]
    assert before_compaction.compacted_at is None
    compacted_ids = frozenset({original.agent_record_id, other.agent_record_id})
    metadata = {"step": 1}
    records.compact("math", compacted_ids, receipt_id="step-1", receipt_metadata=metadata)
    retired = records.get_for_audit("math", original.agent_record_id)
    assert retired is not None
    assert retired.item == original
    assert retired.compacted_at is not None
    assert records.count("math") == 0
    assert records.replay("math") == ()
    assert records.replay_page("math") == ()
    assert records.get("math", original.agent_record_id) is None
    assert records.get("code", other.agent_record_id) == other
    assert records.audit_page("math") == (retired,)

    records.compact("math", compacted_ids, receipt_id="step-1", receipt_metadata=metadata)
    assert records.get_for_audit("math", original.agent_record_id) == retired
    with pytest.raises(RecordConflict):
        records.compact("math", compacted_ids, receipt_id="step-1", receipt_metadata={"step": 2})
    receipts = records.compaction_receipts("math")
    assert len(receipts) == 1
    assert receipts[0]["receipt_id"] == "step-1"
    assert receipts[0]["compacted_ids"] == tuple(sorted(compacted_ids))
    assert receipts[0]["metadata"] == metadata
    assert records.compaction_receipts("code") == ()

    assert records.purge_compacted("math", before=retired.compacted_at + 1, limit=1) == 1
    assert records.get_for_audit("math", original.agent_record_id) is None
    assert records.compaction_receipts("math") == receipts
    assert records.append_result(original).inserted is False
    assert records.existing_receipt(original) is not None
    with pytest.raises(RecordConflict):
        records.append(replace(original, payload={"value": "changed"}))
    assert records.count("math") == 0
    records.append(record("later"))
    assert records.replay_page("math")[0][0] > before_compaction.sequence


def test_reports_referencing_retired_records_keep_retry_protection(records: RecordStore) -> None:
    original = record("first")
    records.append(original)
    records.compact("math", frozenset({original.agent_record_id}))
    report = record("late", references=(original.agent_record_id,))
    assert records.append_result(report).inserted is False
    assert records.existing_receipt(report) is not None
    assert records.count("math") == 0
    with pytest.raises(RecordConflict):
        records.append(replace(report, payload={"value": "changed"}))


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
            Column("compacted_at", Float),
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
            Column("compacted_ids_json", Text, primary_key=True),
            Column("metadata_json", Text, nullable=False),
            Column("recorded_at", Float, nullable=False),
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
                records.compact("math", frozenset({"first"}), receipt_id="step", receipt_metadata={"step": 1})
                assert len(records.compaction_receipts("math")) == 1
                raise RuntimeError("abort outer transaction")

            committed = replace(original, payload={"value": "replacement"})
            with connection.begin():
                assert records.audit_page("math") == ()
                assert records.compaction_receipts("math") == ()
                assert records.existing_receipt(original) is None
                assert records.append_result(committed).inserted
                records.compact("math", frozenset({"first"}), receipt_id="step", receipt_metadata={"step": 2})
                records.append(record("second"))

            with connection.begin():
                assert records.append_result(committed).inserted is False
                assert [entry.agent_record_id for entry in records.replay("math")] == ["second"]
                assert [entry.item.agent_record_id for entry in records.audit_page("math")] == ["first", "second"]
                assert records.compaction_receipts("math")[0]["metadata"] == {"step": 2}
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
