"""Record items (RFC #1028): each inference and report of a scenario is an item of the scenario's owner."""

from __future__ import annotations

from contextlib import closing

import pytest

from reef.core.records_types import AgentRecord, RequestType
from reef.storage import record_items
from reef.storage.data import ItemConflict, MemoryDataStore, NewItem
from reef.storage.item_kinds import copied_record
from reef.storage.record_items import copy_records, record_item, used_record, used_records
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
        payload={"tokens": [1, 2], "artifact_id": "artifact-1"},
        agent_record_id="i1",
        created_at=1.0,
        artifact_id="artifact-1",
    )
    item = record_item(served)
    assert item.item_id == "math/i1"
    assert item.content == {
        "agent_record_id": "i1",
        "request_type": "inference",
        "payload": {"tokens": [1, 2], "artifact_id": "artifact-1"},
        "created_at": 1.0,
        "references": [],
        "artifact_id": "artifact-1",
    }
    assert item.metadata == {
        "kind": "record",
        "scenarios": {"math": "train"},
        "origin": {"scenario": "math", "artifact_id": "artifact-1"},
    }
    assert record_item(record("r1", score=1.0)).metadata["origin"] == {
        "scenario": "math",
        "artifact_id": None,
    }


def test_a_copy_makes_one_page_of_items_and_is_safe_to_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(record_items, "RECORD_PAGE_SIZE", 2)
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = MemoryDataStore(task_files)
    with closing(SQLiteScenarioStorage(tmp_path / "records")) as storage, closing(storage.open("math")) as session:
        for appended in (record("r1", score=1.0), record("r2", score=0.0), record("r3", score=0.5)):
            session.records.append(appended)
        assert copy_records(session.records, data, "acme", "math", 0) == 2
        assert copy_records(session.records, data, "acme", "math", 2) == 3
        assert copy_records(session.records, data, "acme", "math", 3) == 3
        assert [item.item_id for item in data.query("acme")] == ["math/r1", "math/r2", "math/r3"]
        assert copy_records(session.records, data, "acme", "math", 0) == 2, "a copy again changes nothing"
        assert len(data.query("acme")) == 3

        # A client item with a record item's ID and other content stops the copy.
        data.delete("acme", "math/r1")
        other = record_item(record("r1", score=0.5))
        data.create("acme", [NewItem("math/r1", other.content, other.metadata)])
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


def test_another_scenario_of_the_owner_reads_the_records_it_uses_and_stores_no_copy(tmp_path):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = MemoryDataStore(task_files)
    with closing(SQLiteScenarioStorage(tmp_path / "records")) as storage:
        with closing(storage.open("math")) as session:
            session.records.append(record("i1", RequestType.INFERENCE, tokens=[1]))
            session.records.append(record("r1", score=1.0, references=["i1"]))
            copy_records(session.records, data, "acme", "math", 0)
        data.update("acme", ["math/i1", "math/r1"], {"scenarios": {"code": "train"}})
        with closing(storage.open("code")) as session:
            cursor, records = used_records(data, "acme", "code")
            assert [item.agent_record_id for item in records] == ["math/i1", "math/r1"]
            assert records[1].references == ("math/i1",) and records[1].scenario == "code"
            assert used_records(data, "acme", "code", after_sequence=cursor) == (cursor, ()), "nothing new"
            assert used_record(data, "acme", "code", "math/r1").references == ("math/i1",)
            assert session.records.get("code", "math/r1") is None, "the scenario stores no copy"
            assert copy_records(session.records, data, "acme", "code", 0) == 0
            assert [item.item_id for item in data.query("acme")] == ["math/i1", "math/r1"]

            # A scenario does not read its own items as records of another scenario.
            assert used_records(data, "acme", "math")[1] == ()
            assert used_record(data, "acme", "math", "math/i1") is None

            # The origin keeps its record items. Another scenario can drop them, and the read stops at once.
            with pytest.raises(ValueError, match="stays in 'math'"):
                data.update("acme", ["math/r1"], {"scenarios": {"math": None}})
            data.update("acme", ["math/i1", "math/r1"], {"scenarios": {"code": None}})
            assert used_records(data, "acme", "code")[1] == ()
            assert used_record(data, "acme", "code", "math/i1") is None
