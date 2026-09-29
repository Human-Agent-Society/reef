from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from reef.artifact import ArtifactRef, LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.errors import ReefError
from reef.storage.records import RecordConflict
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage


def item(
    agent_record_id: str,
    scenario: str,
    request_type: RequestType = RequestType.INFERENCE,
    *,
    references: tuple[str, ...] = (),
) -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=agent_record_id,
        scenario=scenario,
        request_type=request_type,
        payload={"value": agent_record_id},
        created_at=float(len(agent_record_id)),
        references=references,
    )


def test_read_transaction_keeps_one_snapshot_during_another_connection_commit(tmp_path):
    database = tmp_path / "records.sqlite3"
    with SQLiteRecordStore(database) as reader, SQLiteRecordStore(database) as writer:
        with reader._transaction("math", write=False) as connection:
            assert connection.exec_driver_sql("SELECT COUNT(*) FROM agent_record").scalar_one() == 0
            writer.append(item("first", "math"))
            assert connection.exec_driver_sql("SELECT COUNT(*) FROM agent_record").scalar_one() == 0
        assert reader.count("math") == 1


@pytest.mark.unit
def test_agent_record_replays_in_append_order_per_scenario() -> None:
    records = SQLiteRecordStore()
    records.append(item("a", "math"))
    records.append(item("b", "code"))
    records.append(item("c", "math", RequestType.REPORT, references=("a",)))

    assert [item.agent_record_id for item in records.replay("math")] == ["a", "c"]
    assert [item.agent_record_id for item in records.replay("code")] == ["b"]
    assert records.get("math", "c").references == ("a",)


@pytest.mark.unit
def test_append_is_idempotent_for_identical_data() -> None:
    records = SQLiteRecordStore()
    original = item("a", "math")
    retry = item("a", "math")

    assert records.append(original) is original
    assert records.append(retry) is original
    assert records.replay("math") == (original,)


@pytest.mark.unit
def test_duplicate_agent_record_id_rejects_different_content() -> None:
    records = SQLiteRecordStore()
    records.append(item("same", "math"))

    with pytest.raises(RecordConflict, match="same"):
        records.append(item("same", "code"))


@pytest.mark.unit
def test_lookup_does_not_cross_scenario_boundaries() -> None:
    records = SQLiteRecordStore()
    records.append(item("a", "math"))

    assert records.get("code", "a") is None


@pytest.mark.unit
def test_inference_data_can_record_release_id() -> None:
    artifact = ArtifactRef("artifact-1", "version-1", "initial")

    inference = AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"model": "reef"},
        artifact_ref=artifact,
    )

    assert inference.artifact_ref == artifact


@pytest.mark.unit
def test_agent_record_persists_across_store_restarts(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    artifact = LiveWeightArtifactRef(
        "artifact-1",
        "version-1",
        "initial",
        runtime_load_id="weights-1",
    )
    original = AgentRecord.create(
        agent_record_id="persisted",
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"messages": [{"role": "user", "content": "你好"}]},
        created_at=123.5,
        references=("parent",),
        artifact_ref=artifact,
    )

    with SQLiteRecordStore(database) as first:
        first.append(original)

    with SQLiteRecordStore(database) as second:
        assert second.get("math", "persisted") == original
        assert second.replay("math") == (original,)
        assert second.count("math") == 1


@pytest.mark.unit
def test_in_memory_store_shares_records_across_threads() -> None:
    with SQLiteRecordStore() as records:
        records.append(item("seed", "math"))

        def append_record(index: int) -> None:
            assert records.get("math", "seed") == item("seed", "math")
            assert records.append_result(item(f"worker-{index}", "math")).inserted

        with ThreadPoolExecutor(max_workers=4) as executor:
            tuple(executor.map(append_record, range(32)))

        assert {record.agent_record_id for record in records.replay("math")} == {
            "seed",
            *(f"worker-{index}" for index in range(32)),
        }


@pytest.mark.unit
def test_reads_and_retries_leave_store_ready_for_external_and_local_writes(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    original = item("a", "math")
    with SQLiteRecordStore(database) as first, SQLiteRecordStore(database) as second:
        first.append(original)
        assert first.get("math", "a") == original
        assert first.count("math") == 1

        second.append(item("b", "math"))
        assert [record.agent_record_id for record in first.replay("math")] == ["a", "b"]
        assert first.existing_receipt(original) == original
        assert first.append_result(original).inserted is False
        with pytest.raises(RecordConflict):
            first.append(replace(original, payload={"value": "changed"}))

        second.record_consumption("math", frozenset({"a"}), receipt_id="skip", metadata={})
        assert first.get("math", "a") == original
        first.append(item("c", "math"))
        assert [record.agent_record_id for record in second.replay("math")] == ["a", "b", "c"]

    with SQLiteRecordStore(database) as recovered:
        assert [record.agent_record_id for record in recovered.replay("math")] == ["a", "b", "c"]


@pytest.mark.unit
def test_agent_record_replay_supports_bounded_pages() -> None:
    records = SQLiteRecordStore()
    for agent_record_id in ("a", "b", "c", "d"):
        records.append(item(agent_record_id, "math"))

    assert [record.agent_record_id for record in records.replay("math", offset=1, limit=2)] == ["b", "c"]
    assert records.replay("math", offset=4, limit=2) == ()


@pytest.mark.unit
def test_agent_record_keyset_pages_skip_other_scenarios() -> None:
    records = SQLiteRecordStore()
    records.append(item("a", "math"))
    records.append(item("other", "code"))
    records.append(item("b", "math"))

    first = records.replay_page("math", limit=1)
    second = records.replay_page("math", after_sequence=first[-1][0], limit=1)

    assert [record.agent_record_id for _, record in first] == ["a"]
    assert [record.agent_record_id for _, record in second] == ["b"]


@pytest.mark.unit
def test_conflicting_retry_is_rejected_after_store_restart(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    with SQLiteRecordStore(database) as first:
        first.append(item("same", "math"))

    with SQLiteRecordStore(database) as second:
        assert second.append(item("same", "math")).agent_record_id == "same"
        with pytest.raises(RecordConflict, match="same"):
            second.append(item("same", "code"))


@pytest.mark.unit
def test_old_schema_migrates_without_losing_live_records_or_reviving_deleted_bodies(tmp_path) -> None:
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE agent_record (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_record_id TEXT NOT NULL UNIQUE, scenario TEXT NOT NULL,
                request_type TEXT NOT NULL, payload_json TEXT NOT NULL,
                created_at REAL NOT NULL, references_json TEXT NOT NULL, artifact_json TEXT
            );
            CREATE TABLE consumed_agent_record (
                agent_record_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL
            );
            INSERT INTO agent_record VALUES (7, 'live', 'math', 'inference', '{"value":"live"}', 4, '[]', NULL);
            INSERT INTO consumed_agent_record VALUES ('deleted', 'legacy-fingerprint');
            """
        )

    def open_store(_: int) -> None:
        with SQLiteRecordStore(database) as records:
            assert records.get("math", "live") == item("live", "math")
            entry = records.get_for_audit("math", "live")
            assert entry.sequence == 7
            assert records.get_for_audit("math", "deleted") is None

    # Concurrent openers must see one atomic, idempotent schema upgrade.
    with ThreadPoolExecutor(max_workers=3) as executor:
        tuple(executor.map(open_store, range(3)))
    with SQLiteRecordStore(database) as records:
        late = item("late", "math", RequestType.REPORT, references=("deleted",))
        assert records.append_result(late).inserted is False
        records.append(item("next", "math"))
        assert records.get_for_audit("math", "next").sequence > 7

    with SQLiteRecordStore(database) as records:
        assert records.get("math", "live") == item("live", "math")
        assert records.get_for_audit("math", "live").item == item("live", "math")
        with closing(SQLiteScenarioStorage(tmp_path)) as factory:
            assert factory.prune(days=7, max_bytes=1) == 2
        assert records.get_for_audit("math", "live") is None
        assert records.get("math", "next") is None
        assert records.loss("math").record_count == 2


@pytest.mark.unit
def test_legacy_schema_migrates_consumption_and_backfills_utf8_body_sizes(tmp_path, monkeypatch) -> None:
    database = tmp_path / "legacy.sqlite3"
    legacy_database(database)
    payload_json = '{"text":"海"}'
    references_json = '["起点"]'
    artifact_json = '{"kind":"artifact","content_id":"内容","release_id":"发布","parent_release_id":null}'
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE agent_record ADD COLUMN compacted_at REAL")
        connection.execute(
            "INSERT INTO agent_record VALUES (7, 'retired', 'math', 'inference', ?, 4, ?, ?, 100)",
            (payload_json, references_json, artifact_json),
        )
        connection.execute(
            "INSERT INTO agent_record VALUES (8, 'live', 'math', 'inference', '{\"value\":\"live\"}', 4, '[]', NULL, NULL)"
        )

    with SQLiteRecordStore(database) as records:
        retired = records.get_for_audit("math", "retired")
        assert retired.sequence == 7
        assert records.consumption_receipts("math")[0]["consumed_ids"] == ("retired",)
        assert retired.item.payload == {"text": "海"}
        assert retired.item.references == ("起点",)
        assert retired.item.artifact_ref == ArtifactRef("内容", "发布", None)
        assert records.replay("math") == (retired.item, item("live", "math"))

    with SQLiteRecordStore(database) as records:
        monkeypatch.setattr("reef.storage.sql_records.time.time", lambda: 200.0)
        expected_bytes = len((payload_json + references_json + artifact_json).encode("utf-8"))
        expected_bytes += len(b'{"value":"live"}[]')
        with closing(SQLiteScenarioStorage(tmp_path)) as factory:
            assert factory.prune(days=7, max_bytes=expected_bytes) == 0
            assert factory.prune(days=7, max_bytes=expected_bytes - 1) == 1
        assert records.get_for_audit("math", "retired") is None
        assert records.get("math", "live") == item("live", "math")


@pytest.mark.unit
def test_failed_legacy_backfill_rolls_back_added_columns_and_allows_retry(tmp_path) -> None:
    database = tmp_path / "legacy.sqlite3"
    legacy_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO agent_record VALUES (7, 'live', 'math', 'inference', '{\"value\":\"live\"}', 4, '[]', NULL)"
        )
        connection.execute(
            "CREATE TRIGGER reject_backfill BEFORE UPDATE ON agent_record "
            "BEGIN SELECT RAISE(ABORT, 'injected backfill failure'); END"
        )
        original = connection.execute("SELECT * FROM agent_record").fetchone()

    with pytest.raises(IntegrityError, match="injected backfill failure"):
        SQLiteRecordStore(database)

    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(agent_record)")}
        assert {"compacted_at", "body_bytes"}.isdisjoint(columns)
        assert connection.execute("SELECT * FROM agent_record").fetchone() == original
        connection.execute("DROP TRIGGER reject_backfill")

    with SQLiteRecordStore(database) as records:
        assert records.replay("math") == (item("live", "math"),)
        assert records.get_for_audit("math", "live").sequence == 7
        records.append(item("next", "math"))
        assert [record.agent_record_id for record in records.replay("math")] == ["live", "next"]


@pytest.mark.unit
def test_failed_append_does_not_include_record_payload_in_error_text(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    private_text = "synthetic-private-transcript-marker"
    with SQLiteRecordStore(database) as records:
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TRIGGER reject_append BEFORE INSERT ON agent_record "
                "BEGIN SELECT RAISE(ABORT, 'injected append failure'); END"
            )
        with pytest.raises(IntegrityError, match="injected append failure") as failure:
            records.append(replace(item("private", "math"), payload={"text": private_text}))

        assert private_text not in str(failure.value)
        assert records.count("math") == 0


@pytest.mark.unit
@pytest.mark.parametrize("options", [{"after_sequence": -1}, {"limit": 0}, {"limit": -1}])
def test_audit_page_rejects_invalid_bounds(options) -> None:
    with SQLiteRecordStore() as records, pytest.raises(ValueError):
        records.audit_page("math", **options)


def legacy_database(path) -> None:
    """A record database still in SQLite's rollback journal mode."""
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """
            CREATE TABLE agent_record (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_record_id TEXT NOT NULL UNIQUE, scenario TEXT NOT NULL,
                request_type TEXT NOT NULL, payload_json TEXT NOT NULL,
                created_at REAL NOT NULL, references_json TEXT NOT NULL, artifact_json TEXT
            )
            """
        )
        connection.commit()


class SwitchCursor:
    def __init__(self, row) -> None:
        self._row = row

    def fetchone(self):
        return self._row


class SwitchReplies:
    """A connection whose journal-mode switch answers the way SQLite would.

    Replies are played in order and the last one repeats: ``"busy"`` raises
    SQLITE_BUSY, the way a contended switch does; any other value is returned as
    the resulting journal mode, which is how SQLite reports a switch it declined
    to make.
    """

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls = 0

    def exec_driver_sql(self, statement: str) -> SwitchCursor:
        self.calls += 1
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if reply == "busy":
            raise OperationalError(statement, None, sqlite3.OperationalError("database is locked"))
        return SwitchCursor((reply,))


@pytest.mark.unit
def test_a_contended_journal_mode_switch_is_retried_until_the_database_is_wal(tmp_path, monkeypatch) -> None:
    """SQLite answers a contended journal-mode switch with SQLITE_BUSY without
    running the busy handler, so the connection timeout does not cover it. Both
    that and a declined switch (the unchanged mode, returned rather than raised)
    have to be retried, or a racing opener fails outright."""
    monkeypatch.setattr(SQLiteRecordStore, "_WAL_RETRY_INTERVAL", 0.0)
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as store:
        real, replies = store._connection, SwitchReplies("busy", "delete", "busy", "wal")
        try:
            store._connection = replies
            store._enable_wal()
        finally:
            store._connection = real
    assert replies.calls == 4  # three contended answers, then the switch lands


@pytest.mark.unit
@pytest.mark.parametrize("reply, reported", [("busy", "unknown"), ("delete", "delete")])
def test_a_switch_that_never_lands_names_the_database_and_the_wait(tmp_path, monkeypatch, reply, reported) -> None:
    monkeypatch.setattr(SQLiteRecordStore, "_WAL_SWITCH_TIMEOUT", 0.02)
    monkeypatch.setattr(SQLiteRecordStore, "_WAL_RETRY_INTERVAL", 0.0)
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as store:
        real = store._connection
        try:
            store._connection = SwitchReplies(reply)
            with pytest.raises(ReefError, match=f"could not switch .* to WAL within .*journal mode is {reported}"):
                store._enable_wal()
        finally:
            store._connection = real


@pytest.mark.unit
def test_concurrent_openers_all_upgrade_a_rollback_mode_database(tmp_path) -> None:
    """The race as it reaches CI: several stores opening one legacy database at
    once, each trying to convert it to WAL. Repeated because the loser of the
    race is timing-dependent; a single round misses the regression most times."""

    def open_store(_: int) -> None:
        with SQLiteRecordStore(database):
            pass

    for attempt in range(60):
        database = tmp_path / f"legacy-{attempt}.sqlite3"
        legacy_database(database)
        with ThreadPoolExecutor(max_workers=12) as executor:
            tuple(executor.map(open_store, range(12)))  # raises if any opener saw "database is locked"


@pytest.mark.parametrize(
    "ids_json, metadata_json",
    [('["source", "report"]', '{"outcome":"stale"}'), ("[]", '{"consumed_ids":["report","source"]}')],
)
def test_legacy_consumption_receipts_are_migrated_once(tmp_path, ids_json, metadata_json):
    database = tmp_path / "legacy.sqlite3"
    with SQLiteRecordStore(database):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE compaction_receipts (scenario TEXT, receipt_id TEXT, compacted_ids_json TEXT, "
            "metadata_json TEXT, recorded_at REAL, PRIMARY KEY (scenario, receipt_id, compacted_ids_json))"
        )
        connection.execute(
            "INSERT INTO compaction_receipts VALUES ('math', 'batch-1', ?, ?, 123)",
            (ids_json, metadata_json),
        )
    with SQLiteRecordStore(database) as records:
        [receipt] = records.consumption_receipts("math")
        assert receipt["consumed_ids"] == ("report", "source")
        assert receipt["receipt_id"] == "legacy:batch-1"
        assert receipt["recorded_at"] == 123


def test_legacy_retirement_migration_pages_ids_and_rolls_back_as_one_upgrade(tmp_path):
    database = tmp_path / "records.sqlite3"
    with SQLiteRecordStore(database) as records:
        records.append_many([item(str(index), "math") for index in range(600)])
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "ALTER TABLE agent_record ADD COLUMN compacted_at REAL;"
            "UPDATE agent_record SET compacted_at=1;"
            "CREATE INDEX agent_record_active_sequence ON agent_record(sequence) WHERE compacted_at IS NULL;"
            "CREATE TRIGGER reject_migration BEFORE INSERT ON record_consumption "
            "BEGIN SELECT RAISE(ABORT, 'injected migration failure'); END;"
        )
    with pytest.raises(IntegrityError, match="injected migration failure"):
        SQLiteRecordStore(database)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM agent_record WHERE compacted_at=1").fetchone() == (600,)
        assert connection.execute("SELECT COUNT(*) FROM record_consumption").fetchone() == (0,)
        connection.execute("DROP TRIGGER reject_migration")
    for _ in range(2):
        with SQLiteRecordStore(database) as records:
            assert records.count("math") == 600
            receipts = records.consumption_receipts("math")
            assert len(receipts) == 3
            assert {key for receipt in receipts for key in receipt["consumed_ids"]} == {
                str(index) for index in range(600)
            }
    with sqlite3.connect(database) as connection:
        assert "compacted_at" not in {row[1] for row in connection.execute("PRAGMA table_info(agent_record)")}
        assert "agent_record_active_sequence" not in {
            row[1] for row in connection.execute("PRAGMA index_list(agent_record)")
        }
