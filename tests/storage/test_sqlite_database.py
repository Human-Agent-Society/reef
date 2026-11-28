"""The SQLite database of a scenario's state and its data store."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier
from time import monotonic
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError
from tests.support.records import report

from reef.artifact import SnapshotRef
from reef.core.errors import ReefError
from reef.storage.scenario.commits import CommitRecord
from reef.storage.scenario.store import ScenarioSettings
from reef.storage.sql import setup as sql_setup
from reef.storage.sql.sqlite import SQLiteScenarioDatabase, SQLiteScenarioStorage


def commit(step: int = 1) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        commit_number=step,
        snapshot=SnapshotRef(f"snapshot:{step}", "base"),
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
        assert session.records.read("math", "r1") == first
        assert storage.data.read("math", "math/r1") is not None


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
def test_a_declined_journal_mode_switch_is_retried_until_the_database_is_wal() -> None:
    # SQLite can return the unchanged mode when it cannot make the switch.
    replies = SwitchReplies("delete", "wal")
    sql_setup.enable_wal(replies, "records.sqlite3", timeout_seconds=1, retry_seconds=0)
    assert replies.calls >= 2


@pytest.mark.unit
@pytest.mark.parametrize("reply, reported", [("busy", "unknown"), ("delete", "delete")])
def test_a_switch_that_never_lands_names_the_database_and_the_wait(reply: str, reported: str) -> None:
    with pytest.raises(
        ReefError, match=rf"could not switch records\.sqlite3 to WAL within 0\.01s;.*mode is {reported}"
    ):
        sql_setup.enable_wal(SwitchReplies(reply), "records.sqlite3", timeout_seconds=0.01, retry_seconds=0.001)


@pytest.mark.unit
def test_concurrent_openers_all_upgrade_a_rollback_mode_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "records.sqlite3"

    def open_store(_: int) -> ScenarioSettings:
        with SQLiteScenarioDatabase(database) as store:
            return store.read_settings()

    with closing(sqlite3.connect(database, check_same_thread=False)) as holder:
        holder.execute("CREATE TABLE old_table (value TEXT)")
        holder.commit()
        # A real write transaction prevents the switch until both openers have tried.
        holder.execute("BEGIN IMMEDIATE")
        retrying = Barrier(2, action=holder.rollback)

        def wait_for_unlock(_: float) -> None:
            if holder.in_transaction:
                retrying.wait(timeout=10)

        monkeypatch.setattr(sql_setup, "time", SimpleNamespace(monotonic=monotonic, sleep=wait_for_unlock))
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert tuple(executor.map(open_store, range(2))) == (ScenarioSettings(),) * 2
    with closing(sqlite3.connect(database)) as reopened:
        assert reopened.execute("PRAGMA journal_mode").fetchone() == ("wal",)
