"""Record items (RFC #1028): each inference and report of a scenario is an item of the scenario's owner."""

from __future__ import annotations

from contextlib import closing

import pytest

from reef.core.artifact_ref import ArtifactRef
from reef.core.records_types import AgentRecord, RequestType
from reef.storage import record_items
from reef.storage.data import ItemConflict, MemoryDataStore, NewItem
from reef.storage.data_sync import release_items
from reef.storage.record_items import copied_record, copy_records, record_item, sync_scenario_records
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore


def record(record_id: str, request_type: RequestType = RequestType.REPORT, **payload) -> AgentRecord:
    return AgentRecord.create(
        scenario="math", request_type=request_type, payload=payload, agent_record_id=record_id, created_at=1.0
    )


def test_a_record_item_keeps_the_record_and_says_where_it_came_from():
    served = AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "runtime_load_id": "load-1"},
        agent_record_id="i1",
        created_at=1.0,
        artifact_ref=ArtifactRef(content_id="c1", release_id="rel-1", parent_release_id=None),
    )
    item = record_item(served)
    assert item.item_id == "math/i1"
    assert item.content == {
        "agent_record_id": "i1",
        "request_type": "inference",
        "payload": {"tokens": [1, 2], "runtime_load_id": "load-1"},
        "created_at": 1.0,
        "references": [],
        "artifact_ref": {"kind": "artifact", "content_id": "c1", "release_id": "rel-1", "parent_release_id": None},
    }
    assert item.metadata == {
        "kind": "record",
        "scenarios": {"math": "train"},
        "origin": {"scenario": "math", "release_id": "rel-1", "load_id": "load-1"},
    }
    assert record_item(record("r1", score=1.0)).metadata["origin"] == {
        "scenario": "math",
        "release_id": None,
        "load_id": None,
    }
    with pytest.raises(ValueError, match="training instruction"):
        record_item(record("t1", RequestType.TRAIN))


def test_a_copy_makes_one_page_of_items_skips_instructions_and_is_safe_to_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(record_items, "RECORD_PAGE_SIZE", 2)
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = MemoryDataStore(task_files)
    with closing(SQLiteScenarioStorage(tmp_path / "records")) as storage, closing(storage.open("math")) as session:
        for appended in (record("r1", score=1.0), record("t1", RequestType.TRAIN), record("r2", score=0.0)):
            session.records.append(appended)
        assert copy_records(session.records, data, "acme", "math", 0) == 2
        assert copy_records(session.records, data, "acme", "math", 2) == 3
        assert copy_records(session.records, data, "acme", "math", 3) == 3
        assert [item.item_id for item in data.query("acme")] == ["math/r1", "math/r2"]
        assert copy_records(session.records, data, "acme", "math", 0) == 2, "a copy again changes nothing"
        assert len(data.query("acme")) == 2

        # A client item with a record item's ID and other content stops the copy.
        data.delete("acme", "math/r1")
        data.create("acme", [NewItem("math/r1", {"other": True}, record_item(record("r1")).metadata)])
        with pytest.raises(ItemConflict):
            copy_records(session.records, data, "acme", "math", 0)


def test_a_copy_in_another_scenario_names_the_copies_of_its_references():
    inference = record_item(record("i1", RequestType.INFERENCE, tokens=[1]))
    report = record_item(
        AgentRecord.create(
            scenario="math",
            request_type=RequestType.REPORT,
            payload={"score": 1.0, "references": ["i1"]},
            agent_record_id="r1",
            created_at=2.0,
        )
    )
    assert copied_record(inference.item_id, inference.content, "math", "code").agent_record_id == "math/i1"
    copy = copied_record(report.item_id, report.content, "math", "code")
    assert (copy.scenario, copy.references, copy.payload["references"]) == ("code", ("math/i1",), ["math/i1"])
    for item_id, content, reason in [
        ("other/r1", report.content, "has the ID"),
        ("math/r1", {**report.content, "request_type": "train"}, "'inference' or 'report'"),
        ("math/r1", {**report.content, "extra": 1}, "exactly the keys"),
        ("math/r1", {**report.content, "created_at": "now"}, "created_at"),
    ]:
        with pytest.raises(ValueError, match=reason):
            copied_record(item_id, content, "math", "code")


def test_another_scenario_of_the_owner_holds_copies_that_get_no_items_and_stay(tmp_path):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = MemoryDataStore(task_files)
    with closing(SQLiteScenarioStorage(tmp_path / "records")) as storage:
        with closing(storage.open("math")) as session:
            session.records.append(record("i1", RequestType.INFERENCE, tokens=[1]))
            session.records.append(record("r1", score=1.0, references=["i1"]))
            copy_records(session.records, data, "acme", "math", 0)
        data.update("acme", ["math/i1", "math/r1"], {"scenarios": {"code": "train"}})
        with closing(storage.open("code")) as session:
            assert sync_scenario_records(session.records, data, "acme", "code") == ("math/i1", "math/r1")
            assert sync_scenario_records(session.records, data, "acme", "code") == (), "a sync again adds nothing"
            assert session.records.get("code", "math/r1").references == ("math/i1",)
            assert copy_records(session.records, data, "acme", "code", 0) == 2
            assert [item.item_id for item in data.query("acme")] == ["math/i1", "math/r1"]

            # The origin keeps its record items; another scenario can drop them, and its copies stay.
            with pytest.raises(ValueError, match="stays in 'math'"):
                data.update("acme", ["math/r1"], {"scenarios": {"math": None}})
            data.update("acme", ["math/i1", "math/r1"], {"scenarios": {"code": None}})
            assert sync_scenario_records(session.records, data, "acme", "code") == ()
            assert session.records.get("code", "math/i1") is not None


def test_an_archive_releases_the_items_of_the_scenario_but_not_the_records_it_made(tmp_path):
    data = MemoryDataStore(LocalTaskFileStore(tmp_path / "task-files"))
    data.create("acme", [record_item(record("r1", score=1.0))])
    data.create("acme", [NewItem("t1", {"steps": []}, {"kind": "trajectory", "scenarios": {"math": "train"}})])
    release_items(data, "acme", "math")
    assert [(item.item_id, dict(item.scenarios)) for item in data.query("acme")] == [
        ("math/r1", {"math": "train"}),
        ("t1", {}),
    ]
