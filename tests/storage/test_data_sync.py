"""The data sync (RFC #1028, RFC #1140): a scenario reads the items of its owner that it uses."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, replace

import pytest
from tests.support.samples import sample_store

from reef.service.task_import import DataScenarioStorage
from reef.storage.data import NewItem
from reef.storage.data_queue import DATA_SYNC_KIND, DATA_SYNC_QUEUE
from reef.storage.data_sync import data_sync_key, request_data_sync
from reef.storage.sql_coordination import SQLiteCoordinationDatabase, SQLWorkQueue
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore


def trajectory(item_id: str, **scenarios: str) -> NewItem:
    content = {"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1, "message": "hi"}]}
    return NewItem(item_id, content, {"kind": "trajectory", "scenarios": scenarios})


@dataclass
class Stores:
    """A scenario storage and a data store over one task file store, which the scenario storage owns."""

    storage: SQLiteScenarioStorage
    data: SQLiteDataStore


@pytest.fixture
def stores(tmp_path) -> Iterator[Stores]:
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = SQLiteDataStore(tmp_path / "data", task_files)
    storage = SQLiteScenarioStorage(tmp_path / "records", task_files=task_files, data=data)
    try:
        yield Stores(storage, data)
    finally:
        data.close()
        storage.close()


def sample_ids(storage, session) -> list[str]:
    return [stored.sample_id for stored in sample_store(storage, "math", session).page()]


def test_the_first_open_sets_the_owner_and_each_open_reads_the_owners_items(stores):
    stores.data.create("math", [trajectory("own", math="train")])
    stores.data.create("acme", [trajectory("t1", math="train")])
    opened = DataScenarioStorage(stores.storage, stores.data)
    with closing(opened.open("math")) as session:
        assert opened.read_settings("math").owner == "math"
        assert sample_ids(opened, session) == ["own"]
    # A scenario of another owner uses that owner's items only.
    stores.storage.write_settings("math", replace(stores.storage.read_settings("math"), owner="acme"))
    with closing(opened.open("math")) as session:
        assert sample_ids(opened, session) == ["t1"]


def test_an_archive_releases_the_items_so_a_new_scenario_of_the_name_uses_none(stores):
    stores.data.create("math", [trajectory("t1", math="train")])
    opened = DataScenarioStorage(stores.storage, stores.data)
    with closing(opened.open("math")) as session:
        assert sample_ids(opened, session) == ["t1"]
    opened.archive("math")
    assert stores.data.read("math", "t1").scenarios == {}
    with closing(opened.open("math")) as session:
        assert sample_ids(opened, session) == []


def test_one_queued_sync_covers_each_request_of_its_scenario():
    with closing(SQLiteCoordinationDatabase(None)) as database:
        queue = SQLWorkQueue(database)
        first = request_data_sync(queue, "acme", "math")
        assert request_data_sync(queue, "acme", "math").id == first.id
        other = request_data_sync(queue, "acme", "code")
        assert other.id != first.id and other.serial_key == data_sync_key("code") != data_sync_key("math")
        assert (first.queue, first.kind, first.scenario, dict(first.payload)) == (
            DATA_SYNC_QUEUE,
            DATA_SYNC_KIND,
            "math",
            {"owner": "acme"},
        )
