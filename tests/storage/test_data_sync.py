"""The data sync (RFC #1028): a scenario's uploads are copies of the data items that it uses."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from tests.support.task_service_stubs import task_for

from reef.core.job_data import TrajectorySample
from reef.core.tasks import write_harbor_task
from reef.service.task_import import DataScenarioStorage
from reef.storage.data import NewItem
from reef.storage.data_sync import adopt_uploads, sync_scenario_samples
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore


def atif(text: str) -> dict:
    return {"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1, "message": text}]}


def trajectory(item_id: str, text: str = "", **scenarios: str) -> NewItem:
    return NewItem(item_id, atif(text or item_id), {"kind": "trajectory", "scenarios": scenarios})


@dataclass
class Stores:
    """A scenario storage and a data store over one task file store, which the scenario storage owns."""

    storage: SQLiteScenarioStorage
    data: SQLiteDataStore
    root: Path

    def task(self, name: str, port: int = 9000) -> Path:
        return write_harbor_task(task_for(name, "record-1", port), self.root / "tasks" / str(port))


@pytest.fixture
def stores(tmp_path) -> Iterator[Stores]:
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    storage = SQLiteScenarioStorage(tmp_path / "records", task_files=task_files)
    data = SQLiteDataStore(tmp_path / "data", task_files)
    try:
        yield Stores(storage, data, tmp_path)
    finally:
        data.close()
        storage.close()


def uploads_of(samples) -> list[tuple[str, str]]:
    return [(stored.sample_id, stored.split) for stored in samples.page() if stored.component is None]


def test_a_sync_copies_each_item_that_the_scenario_uses_and_removes_the_rest(stores):
    stores.data.create("acme", [trajectory("t1", math="train"), trajectory("t2", math="train"), trajectory("t3")])
    stores.data.create(
        "acme", [NewItem("task-a", stores.task("task-a"), {"kind": "task", "scenarios": {"math": "eval"}})]
    )
    with closing(stores.storage.open("math")) as session:
        result = sync_scenario_samples(session.samples, stores.data, "acme", "math")
        assert (result.added, result.removed) == (("t1", "t2", "task-a"), ())
        assert uploads_of(session.samples) == [("t1", "train"), ("t2", "train"), ("task-a", "eval")]
        (task,) = session.samples.imported_tasks()
        assert (task.sample.task_path / "instruction.md").is_file()
        assert (
            sync_scenario_samples(session.samples, stores.data, "acme", "math") == type(result)()
        ), "a sync again changes nothing"

        # An item that leaves, or moves to the other split, leaves its copy; a moved task gets a new one.
        stores.data.update("acme", ["t1"], {"scenarios": {"math": None}})
        stores.data.update("acme", ["task-a"], {"scenarios": {"math": "train"}})
        stores.data.delete("acme", "t2")
        result = sync_scenario_samples(session.samples, stores.data, "acme", "math")
        assert (result.added, result.removed) == (("task-a",), ("t1", "t2", "task-a"))
        assert uploads_of(session.samples) == [("task-a", "train")]


def test_a_sync_replaces_a_copy_whose_item_came_back_with_other_content_and_keeps_stage_samples(stores):
    stores.data.create("acme", [trajectory("t1", "old", math="train")])
    with closing(stores.storage.open("math")) as session:
        session.samples.add_record_samples("", None, ("record-1",), (("record-1:0", TrajectorySample(atif("made"))),))
        sync_scenario_samples(session.samples, stores.data, "acme", "math")
        stores.data.delete("acme", "t1")
        stores.data.create("acme", [trajectory("t1", "new", math="train")])
        result = sync_scenario_samples(session.samples, stores.data, "acme", "math")
        assert (result.added, result.removed) == (("t1",), ("t1",))
        (copy,) = session.samples.get(("t1",))
        assert copy.sample.trajectory == atif("new")
        assert [stored.sample_id for stored in session.samples.page() if stored.component == ""] == ["record-1:0"]


def test_old_uploads_become_items_that_the_next_sync_keeps(stores):
    with closing(stores.storage.open("math")) as session:
        session.samples.add_trajectories((("t1", TrajectorySample(atif("t1"))),))
        session.samples.add_tasks((stores.task("task-a"),), "eval")
        adopt_uploads(session.samples, stores.data, "math", "math")
        assert [(item.item_id, item.kind, item.scenarios) for item in stores.data.query("math")] == [
            ("t1", "trajectory", {"math": "train"}),
            ("task-a", "task", {"math": "eval"}),
        ]
        assert sync_scenario_samples(session.samples, stores.data, "math", "math").removed == ()


def test_the_first_open_sets_the_owner_and_each_open_syncs_its_items(stores):
    with closing(stores.storage.open("math")) as session:
        session.samples.add_trajectories((("old", TrajectorySample(atif("old"))),))
    opened = DataScenarioStorage(stores.storage, stores.data)
    with closing(opened.open("math")) as session:
        assert opened.read_settings("math").owner == "math"
        assert uploads_of(session.samples) == [("old", "train")]
    # A scenario of another owner uses that owner's items only.
    stores.storage.write_settings("math", replace(stores.storage.read_settings("math"), owner="acme"))
    stores.data.create("acme", [trajectory("t1", math="train")])
    with closing(opened.open("math")) as session:
        assert uploads_of(session.samples) == [("t1", "train")]
