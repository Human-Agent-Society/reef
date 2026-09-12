from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Event

import pytest
from aiohttp import web
from sqlalchemy.exc import OperationalError

from reef.core import AgentRecord, RequestType
from reef.dispatcher import build_default_dispatcher
from reef.service import assembly
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.storage.records import RecordConflict, RecordRetention
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage


def trace(record_id: str, scenario: str = "math") -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=record_id, scenario=scenario, request_type=RequestType.INFERENCE, payload={"text": "海"}
    )


BODY_BYTES = len('{"text":"海"}[]'.encode())


@pytest.fixture
def store_factory(tmp_path):
    with closing(SQLiteScenarioStorage(tmp_path)) as factory:
        yield factory


def test_retention_uses_one_budget_across_scenarios_and_archived_databases(tmp_path, monkeypatch, store_factory):
    archive = tmp_path / "archived" / "removed" / "archived.sqlite3"
    oldest = trace("old")
    with SQLiteRecordStore(tmp_path / "a.sqlite3") as first, SQLiteRecordStore(tmp_path / "b.sqlite3") as second:
        with SQLiteRecordStore(archive) as removed:
            for clock, store, record in (
                (10.0, first, oldest),
                (15.0, removed, trace("archived")),
                (20.0, second, trace("middle", "code")),
                (30.0, first, trace("new")),
            ):
                monkeypatch.setattr("reef.storage.sql_records.time.time", lambda clock=clock: clock)
                store.append(record)
                store.compact(record.scenario, frozenset({record.agent_record_id}))
        first.append(replace(trace("active"), payload={"text": "x" * 4096}))
        monkeypatch.setattr("reef.storage.sql_records.time.time", lambda: 40.0)
        assert store_factory.prune(days=7, max_bytes=2 * BODY_BYTES) == 2
        assert first.get_for_audit("math", "old") is None
        assert first.get_for_audit("math", "new") is not None
        assert second.get_for_audit("code", "middle") is not None
        assert first.get("math", "active") is not None
        assert first.append_result(oldest).inserted is False
        with pytest.raises(RecordConflict):
            first.append(replace(oldest, payload={"text": "changed"}))
        assert (
            first.append_result(replace(trace("late"), request_type=RequestType.REPORT, references=("old",))).inserted
            is False
        )
    with SQLiteRecordStore(archive) as removed:
        assert removed.get_for_audit("math", "archived") is None


def test_retention_expires_bodies_in_batches_and_preserves_boundary_and_receipts(tmp_path, monkeypatch, store_factory):
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        for index in range(257):
            records.append(trace(f"old-{index}"))
        monkeypatch.setattr("reef.storage.sql_records.time.time", lambda: 99.0)
        records.compact(
            "math",
            frozenset(f"old-{index}" for index in range(257)),
            receipt_id="batch",
            receipt_metadata={"outcome": "stale"},
        )
        records.append(trace("boundary"))
        monkeypatch.setattr("reef.storage.sql_records.time.time", lambda: 100.0)
        records.compact("math", frozenset({"boundary"}))
        monkeypatch.setattr("reef.storage.sql_records.time.time", lambda: 7 * 86400 + 100.0)
        retention = RecordRetention()
        assert store_factory.prune(days=retention.days, max_bytes=retention.max_bytes) == 257
        assert [entry.item.agent_record_id for entry in records.audit_page("math")] == ["boundary"]
        assert records.compaction_receipts("math")[0]["receipt_id"] == "batch"
        assert store_factory.prune(days=retention.days, max_bytes=retention.max_bytes) == 0


def test_budget_purge_pages_across_equal_timestamps_without_skipping_rows(tmp_path, monkeypatch, store_factory):
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        for index in range(600):
            records.append(trace(str(index)))
        monkeypatch.setattr("reef.storage.sql_records.time.time", lambda: 100.0)
        records.compact("math", frozenset(str(index) for index in range(600)))
        assert store_factory.prune(days=7, max_bytes=3 * BODY_BYTES) == 597
        assert [entry.item.agent_record_id for entry in records.audit_page("math")] == ["597", "598", "599"]


def test_large_finite_retention_days_still_enforce_the_byte_budget(tmp_path, monkeypatch, store_factory):
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        for timestamp, record_id in ((1.0, "old"), (2.0, "new")):
            records.append(trace(record_id))
            monkeypatch.setattr("reef.storage.sql_records.time.time", lambda timestamp=timestamp: timestamp)
            records.compact("math", frozenset({record_id}))
        monkeypatch.setattr("reef.storage.sql_records.time.time", lambda: 3.0)

        # A finite number of days can produce an infinite cutoff in seconds.
        assert store_factory.prune(days=1e308, max_bytes=BODY_BYTES) == 1
        assert [entry.item.agent_record_id for entry in records.audit_page("math")] == ["new"]


def test_retention_skips_unmigrated_stores_and_empty_directories(tmp_path, store_factory):
    retention = RecordRetention()
    assert store_factory.prune(days=retention.days, max_bytes=retention.max_bytes) == 0
    with sqlite3.connect(tmp_path / "legacy.sqlite3") as connection:
        connection.execute("CREATE TABLE agent_record (sequence INTEGER PRIMARY KEY, payload_json TEXT)")
        connection.execute("INSERT INTO agent_record VALUES (1, 'original')")
    assert store_factory.prune(days=retention.days, max_bytes=retention.max_bytes) == 0
    with sqlite3.connect(tmp_path / "legacy.sqlite3") as connection:
        assert connection.execute("SELECT payload_json FROM agent_record").fetchone() == ("original",)


@pytest.mark.parametrize("days", [0, -1, float("nan"), float("inf"), True])
def test_retention_rejects_invalid_days(days):
    with pytest.raises(ValueError, match="retention_days"):
        RecordRetention(days=days)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_retention_rejects_invalid_budgets(size):
    with pytest.raises(ValueError, match="retention_max_bytes"):
        RecordRetention(max_bytes=size)


def test_service_config_defaults_to_seven_days_and_twenty_gib_and_accepts_overrides():
    defaults = service_config_from_mapping({"reef": {"recipe": "recipe"}})
    assert defaults.agent_record_retention_days == 7
    assert defaults.agent_record_retention_max_bytes == 20 * 1024**3
    settings = service_config_from_mapping(
        {"reef": {"recipe": "recipe", "agent_record_retention_days": 3, "agent_record_retention_max_bytes": 1024}}
    )
    assert settings.agent_record_retention_days == 3
    assert settings.agent_record_retention_max_bytes == 1024
    assert "agent_record_retention_days" not in assembly._recipe_owned_settings(settings)
    with pytest.raises(ValueError, match="retention_max_bytes"):
        service_config_from_mapping({"reef": {"recipe": "recipe", "agent_record_retention_max_bytes": 0}})


def test_service_runs_retention_retries_failure_and_stops_on_cleanup(tmp_path, monkeypatch, caplog):
    dispatcher = build_default_dispatcher(agent_record_dir=tmp_path, scenario_storage=SQLiteScenarioStorage(tmp_path))
    monkeypatch.setattr(assembly, "build_dispatcher", lambda *args, **kwargs: dispatcher)
    monkeypatch.setattr("reef.service.app._RECORD_RETENTION_INTERVAL_SECONDS", 0.005)
    original = dispatcher.prune_record_archives
    attempts = []

    def flaky(retention):
        attempts.append(retention)
        if len(attempts) == 1:
            raise OperationalError(
                "DELETE FROM agent_record", None, sqlite3.OperationalError("temporary storage failure")
            )
        return original(retention)

    monkeypatch.setattr(dispatcher, "prune_record_archives", flaky)
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        records.append(trace("expired"))
        with monkeypatch.context() as clock:
            clock.setattr("reef.storage.sql_records.time.time", lambda: 1.0)
            records.compact("math", frozenset({"expired"}))

        async def run():
            app = assembly.build_app(ServiceConfig(recipe="recipe", agent_record_dir=str(tmp_path)))
            runner = web.AppRunner(app)
            await runner.setup()
            try:

                async def wait_for_purge():
                    while records.get_for_audit("math", "expired") is not None:
                        await asyncio.sleep(0.005)

                await asyncio.wait_for(wait_for_purge(), timeout=2)
            finally:
                await runner.cleanup()
            count = len(attempts)
            await asyncio.sleep(0.02)
            assert len(attempts) == count

        asyncio.run(run())
    assert len(attempts) >= 2
    assert attempts[0] == RecordRetention(days=7, max_bytes=20 * 1024**3)
    assert "record retention failed" in caplog.text


def test_retention_serializes_with_scenario_file_archival(tmp_path, monkeypatch):
    dispatcher = build_default_dispatcher(agent_record_dir=tmp_path, scenario_storage=SQLiteScenarioStorage(tmp_path))
    dispatcher.get_or_create_scenario("math")
    started, release, deleting = Event(), Event(), Event()
    original = SQLiteScenarioStorage.prune

    def held(factory, *, days, max_bytes):
        started.set()
        assert release.wait(3)
        return original(factory, days=days, max_bytes=max_bytes)

    def remove():
        deleting.set()
        return dispatcher.delete_scenario("math")

    monkeypatch.setattr(SQLiteScenarioStorage, "prune", held)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pruning = executor.submit(dispatcher.prune_record_archives, RecordRetention())
            assert started.wait(3)
            deletion = executor.submit(remove)
            try:
                assert deleting.wait(3)
                assert not deletion.done()
                assert list(tmp_path.glob("*.sqlite3"))
            finally:
                release.set()
            assert pruning.result(timeout=3) == 0
            assert deletion.result(timeout=3)["archived"]
    finally:
        dispatcher.close()
