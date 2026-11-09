"""The SQLite database of a scenario's state, and the scenario storage's data store and old record tables."""

from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError

from reef.artifact import SnapshotRef
from reef.core import AgentRecord, RequestType
from reef.core.errors import ReefError
from reef.storage.commits import CommitRecord
from reef.storage.record_items import record_item
from reef.storage.scenario import ScenarioSettings
from reef.storage.sqlite import OLD_RECORD_TABLES, SQLiteScenarioDatabase, SQLiteScenarioStorage


def report(record_id: str, scenario: str = "math") -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=record_id, scenario=scenario, request_type=RequestType.REPORT, payload={"score": 1.0}
    )


def commit(step: int = 1) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=step,
        snapshot=SnapshotRef(f"release:{step}", "base"),
        checkpoint=True,
        algorithm_state={"step": step},
        high_water_sequence=step,
        high_water_offset=step,
        consumed_ids=frozenset({f"record-{step}"}),
        recorded_at=1000.0 + step,
    )


def test_a_read_transaction_keeps_one_snapshot_while_another_connection_commits(tmp_path):
    database = tmp_path / "math.sqlite3"
    with closing(SQLiteScenarioDatabase(database)) as reader, closing(SQLiteScenarioDatabase(database)) as writer:
        with reader._transaction(write=False) as connection:
            assert connection.exec_driver_sql("SELECT COUNT(*) FROM scenario_commit").scalar_one() == 0
            writer.commit_history().append(commit())
            assert connection.exec_driver_sql("SELECT COUNT(*) FROM scenario_commit").scalar_one() == 0
        assert len(reader.commit_history().records()) == 1


def test_a_storage_keeps_the_records_in_its_own_data_store_across_a_restart(tmp_path):
    first = report("r1")
    with closing(SQLiteScenarioStorage(tmp_path)) as storage, closing(storage.open("math")) as session:
        session.records.append(first)
    with closing(SQLiteScenarioStorage(tmp_path)) as storage, closing(storage.open("math")) as session:
        assert session.records.get("math", "r1") == first
        assert storage.data.read("math", "math/r1") is not None


def scenario_file(directory, scenario: str = "math"):
    return directory / f"{hashlib.sha256(scenario.encode()).hexdigest()}.sqlite3"


@pytest.mark.parametrize(
    ("newest", "stored", "copied"),
    [("old-2", "math", True), ("old-2", None, False), ("code/r9", "code", True)],
    ids=["copied", "not-copied", "copy-of-another-scenario"],
)
def test_the_first_open_drops_the_old_record_tables_only_when_their_records_are_record_items(
    tmp_path, newest, stored, copied
):
    with closing(SQLiteScenarioStorage(tmp_path)) as storage:
        storage.write_settings("math", ScenarioSettings(owner="acme"))
        if stored is not None:
            stored_id = newest.split("/")[-1]
            storage.data.create("acme", [record_item(replace(report(stored_id), scenario=stored))])
    legacy_database(scenario_file(tmp_path))
    with closing(sqlite3.connect(scenario_file(tmp_path))) as connection, connection:
        connection.executemany(
            "INSERT INTO agent_record (agent_record_id, scenario, request_type, payload_json, created_at, "
            "references_json) VALUES (?, 'math', 'report', '{}', 1.0, '[]')",
            [("old-1",), (newest,)],
        )
    with closing(SQLiteScenarioStorage(tmp_path)) as storage:
        storage.open("math").close()
        with (
            closing(SQLiteScenarioDatabase(scenario_file(tmp_path))) as database,
            database._transaction(write=False) as connection,
        ):
            kept = set(inspect(connection).get_table_names()) & set(OLD_RECORD_TABLES)
        assert kept == (set() if copied else {"agent_record"})
        assert storage.read_settings("math").owner == "acme"


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
    monkeypatch.setattr(SQLiteScenarioDatabase, "_WAL_RETRY_INTERVAL", 0.0)
    with SQLiteScenarioDatabase(tmp_path / "records.sqlite3") as store:
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
    monkeypatch.setattr(SQLiteScenarioDatabase, "_WAL_SWITCH_TIMEOUT", 0.02)
    monkeypatch.setattr(SQLiteScenarioDatabase, "_WAL_RETRY_INTERVAL", 0.0)
    with SQLiteScenarioDatabase(tmp_path / "records.sqlite3") as store:
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
        with SQLiteScenarioDatabase(database):
            pass

    for attempt in range(60):
        database = tmp_path / f"legacy-{attempt}.sqlite3"
        legacy_database(database)
        with ThreadPoolExecutor(max_workers=12) as executor:
            tuple(executor.map(open_store, range(12)))  # raises if any opener saw "database is locked"
