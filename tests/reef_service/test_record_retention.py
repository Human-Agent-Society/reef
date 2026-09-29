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


def test_capacity_evicts_unconsumed_and_consumed_records_across_archives(tmp_path, store_factory, caplog):
    archive = tmp_path / "archived" / "removed" / "archived.sqlite3"
    oldest = replace(trace("old"), created_at=10.0)
    with SQLiteRecordStore(tmp_path / "a.sqlite3") as first, SQLiteRecordStore(tmp_path / "b.sqlite3") as second:
        with SQLiteRecordStore(archive) as removed:
            first.append(oldest)
            removed.append(replace(trace("archived"), created_at=15.0))
            second.append(replace(trace("middle", "code"), created_at=20.0))
            first.append(replace(trace("new"), created_at=30.0))
            first.record_consumption("math", frozenset({"old"}), receipt_id="skip", metadata={})
        assert store_factory.prune(days=7, max_bytes=2 * BODY_BYTES) == 2
        assert first.get_for_audit("math", "old") is None
        assert first.get("math", "new") is not None
        assert second.get("code", "middle") is not None
        assert first.loss("math").record_count == 1
        assert first.loss("other").record_count == 0
        assert first.append_result(oldest).inserted is False
        with pytest.raises(RecordConflict):
            first.append(replace(oldest, payload={"text": "changed"}))
    with SQLiteRecordStore(archive) as removed:
        assert removed.get("math", "archived") is None
        assert removed.loss("math").record_count == 1
    with SQLiteRecordStore(tmp_path / "a.sqlite3") as reopened:
        assert reopened.loss("math").body_bytes == BODY_BYTES
        assert reopened.loss("math").first_sequence == reopened.loss("math").last_sequence == 1
    assert "Record capacity exceeded" in caplog.text
    assert "scenario=math" in caplog.text
    assert "sequence=1..1" in caplog.text


def test_record_age_and_training_completion_do_not_trigger_deletion(tmp_path, store_factory):
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        records.append(replace(trace("old"), created_at=1.0))
        records.record_consumption("math", frozenset({"old"}), receipt_id="skip", metadata={})
        assert store_factory.prune(days=0.01, max_bytes=BODY_BYTES) == 0
        assert records.get_for_audit("math", "old") is not None
        assert records.loss("math").record_count == 0


def test_budget_purge_pages_across_equal_timestamps_without_skipping_rows(tmp_path, monkeypatch, store_factory):
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        for index in range(600):
            records.append(trace(str(index)))
        monkeypatch.setattr("reef.storage.sql_records.time.time", lambda: 100.0)
        assert store_factory.prune(days=7, max_bytes=3 * BODY_BYTES) == 597
        assert [entry.item.agent_record_id for entry in records.audit_page("math")] == ["597", "598", "599"]


def test_large_finite_retention_days_still_enforce_the_byte_budget(tmp_path, monkeypatch, store_factory):
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        for timestamp, record_id in ((1.0, "old"), (2.0, "new")):
            records.append(replace(trace(record_id), created_at=timestamp))
            monkeypatch.setattr("reef.storage.sql_records.time.time", lambda timestamp=timestamp: timestamp)
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

        async def run():
            app = assembly.build_app(
                ServiceConfig(recipe="recipe", agent_record_dir=str(tmp_path), agent_record_retention_max_bytes=1)
            )
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
    assert attempts[0] == RecordRetention(days=7, max_bytes=1)
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


def test_sqlite_upgrade_preserves_records_and_adds_capacity_metadata(tmp_path):
    database = tmp_path / "records.sqlite3"
    original = trace("original")
    with SQLiteRecordStore(database) as records:
        records.append(original)
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE record_eviction")
        connection.execute("DROP INDEX agent_record_capacity")
    with SQLiteRecordStore(database) as records:
        assert records.get("math", "original") == original
        assert records.loss("math").record_count == 0
    with closing(SQLiteScenarioStorage(tmp_path)) as storage:
        assert storage.prune(days=7, max_bytes=1) == 1
    with SQLiteRecordStore(database) as records:
        assert records.loss("math").record_count == 1
        assert not records.append_result(original).inserted


def test_capacity_eviction_rolls_back_bodies_hashes_and_loss_totals(tmp_path):
    from reef.storage.sql_records import SQLRecordRetention

    original = trace("original")
    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        records.append(original)
        with pytest.raises(RuntimeError, match="rollback"), records._transaction("math", write=True) as connection:
            SQLRecordRetention(records._tables).evict(connection, [1])
            raise RuntimeError("rollback")
        assert records.get("math", "original") == original
        assert records.loss("math").record_count == 0
        assert not records.append_result(original).inserted
        with records._transaction("math", write=False) as connection:
            assert connection.exec_driver_sql("SELECT count(*) FROM consumed_agent_record").scalar_one() == 0
