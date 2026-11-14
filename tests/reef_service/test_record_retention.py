from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from aiohttp import web
from sqlalchemy.exc import OperationalError

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.dispatcher.scenario_deletes import DELETES_QUEUE
from reef.recipe import Recipe
from reef.scenario.live_objects import LiveObjects
from reef.service import assembly
from reef.service.deploy.service_config import ServiceSettings, service_config_from_mapping
from reef.storage.data import DataStore
from reef.storage.record_items import record_item
from reef.storage.records import RecordRetention
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore, TaskFileStore


def trace(record_id: str, scenario: str = "math") -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=record_id, scenario=scenario, request_type=RequestType.INFERENCE, payload={"text": "海"}
    )


def retention_dispatcher(tmp_path, data: DataStore, task_files: TaskFileStore) -> Dispatcher:
    """A dispatcher whose scenario storage keeps its records in ``data``, which the dispatcher prunes."""
    return Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(),
        scenario_storage=SQLiteScenarioStorage(tmp_path, task_files=task_files, data=data),
        live=LiveObjects(data_store=data),
        coordination=memory_coordination(),
    )


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_retention_rejects_invalid_budgets(size):
    with pytest.raises(ValueError, match="retention_max_bytes"):
        RecordRetention(max_bytes=size)


def test_service_config_defaults_to_twenty_gib_and_accepts_overrides():
    defaults = service_config_from_mapping({"reef": {"recipe": "recipe"}})
    assert defaults.agent_record_retention_max_bytes == 20 * 1024**3
    settings = service_config_from_mapping({"reef": {"recipe": "recipe", "agent_record_retention_max_bytes": 1024}})
    assert settings.agent_record_retention_max_bytes == 1024
    assert "agent_record_retention_max_bytes" not in assembly._recipe_owned_settings(settings)
    with pytest.raises(ValueError, match="retention_max_bytes"):
        service_config_from_mapping({"reef": {"recipe": "recipe", "agent_record_retention_max_bytes": 0}})


def test_service_runs_retention_retries_failure_and_stops_on_cleanup(tmp_path, monkeypatch, caplog):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = SQLiteDataStore(tmp_path / "data", task_files)
    dispatcher = retention_dispatcher(tmp_path, data, task_files)
    monkeypatch.setattr(assembly, "build_dispatcher", lambda *args, **kwargs: dispatcher)
    monkeypatch.setattr("reef.service.app._RECORD_RETENTION_INTERVAL_SECONDS", 0.005)
    original = dispatcher.prune_record_archives
    attempts = []

    def flaky(retention):
        attempts.append(retention)
        if len(attempts) == 1:
            raise OperationalError(
                "DELETE FROM data_item", None, sqlite3.OperationalError("temporary storage failure")
            )
        return original(retention)

    monkeypatch.setattr(dispatcher, "prune_record_archives", flaky)
    data.create("math", [record_item(trace("expired"))])

    async def run():
        app = assembly.build_app(
            ServiceSettings(recipe="recipe", agent_record_dir=str(tmp_path), agent_record_retention_max_bytes=1)
        )
        runner = web.AppRunner(app)
        await runner.setup()
        try:

            async def wait_for_purge():
                while data.read("math", "math/expired") is not None:
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(wait_for_purge(), timeout=2)
        finally:
            await runner.cleanup()
        count = len(attempts)
        await asyncio.sleep(0.02)
        assert len(attempts) == count

    asyncio.run(run())
    assert len(attempts) >= 2
    assert attempts[0] == RecordRetention(max_bytes=1)
    assert "record retention failed" in caplog.text


def test_retention_serializes_with_scenario_file_archival(tmp_path, monkeypatch):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = SQLiteDataStore(tmp_path / "data", task_files)
    dispatcher = retention_dispatcher(tmp_path, data, task_files)
    dispatcher.get_or_create_scenario("math")
    started, release, deleting = Event(), Event(), Event()
    original = SQLiteDataStore.prune_records

    def held(store, max_bytes):
        started.set()
        assert release.wait(3)
        return original(store, max_bytes)

    def remove():
        deleting.set()
        dispatcher.delete_scenario("math")
        # The sweep archives the scenario's storage, and waits for the retention that runs.
        dispatcher.sweeper.sweep()
        (cleanup,) = dispatcher.queue.items(DELETES_QUEUE, scenario="math")
        return cleanup.result

    monkeypatch.setattr(SQLiteDataStore, "prune_records", held)
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
