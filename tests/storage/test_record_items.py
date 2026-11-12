"""Record items (RFC #1028): each inference and report of a scenario is an item of the scenario's owner."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace

import pytest
from tests.support.conversations import conversation

from reef.core.evaluation import EvaluationCall, EvaluationCallSummary
from reef.core.records_types import AgentRecord, RequestType
from reef.storage.data import MemoryDataStore, ScenarioArchived
from reef.storage.item_kinds import copied_record
from reef.storage.record_items import ItemRecordStore, evaluation_call_item, record_item, used_record, used_records
from reef.storage.records import AppendResult, RecordConflict
from reef.storage.scenario import ScenarioSettings
from reef.storage.sql_data import SQLiteDataStore
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
    with closing(SQLiteScenarioStorage(tmp_path / "records", task_files=task_files, data=data)) as storage:
        for scenario in ("math", "code"):
            storage.write_settings(scenario, ScenarioSettings(owner="acme"))
        with closing(storage.open("math")) as session:
            session.records.append(record("i1", RequestType.INFERENCE, tokens=[1]))
            session.records.append(record("r1", score=1.0, references=["i1"]))
        data.update("acme", ["math/i1", "math/r1"], {"scenarios": {"code": "train"}})
        with closing(storage.open("code")) as session:
            cursor, records = used_records(data, "acme", "code")
            assert [item.agent_record_id for item in records] == ["math/i1", "math/r1"]
            assert records[1].references == ("math/i1",) and records[1].scenario == "code"
            assert used_records(data, "acme", "code", after_sequence=cursor) == (cursor, ()), "nothing new"
            assert used_record(data, "acme", "code", "math/r1").references == ("math/i1",)
            assert session.records.get("code", "math/r1") is None, "the scenario stores no copy"
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


@pytest.fixture(params=["memory", "sqlite"])
def item_records(request, tmp_path):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = MemoryDataStore(task_files) if request.param == "memory" else SQLiteDataStore(None, task_files)
    with closing(task_files), closing(data):
        yield ItemRecordStore(data, "acme", "math", data.scenario_generation("acme", "math"))


def plain(record_id: str, *references: str) -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=record_id,
        scenario="math",
        request_type=RequestType.REPORT if references else RequestType.INFERENCE,
        payload={"value": record_id},
        references=references,
        created_at=123.0,
    )


def test_a_scenario_reads_and_retries_its_records_as_record_items(item_records):
    original = plain("first")
    assert item_records.existing_receipt(original) is None
    assert item_records.append_result(original) == AppendResult(original, True)
    retry = replace(original, created_at=456.0)
    assert item_records.append_result(retry) == AppendResult(original, False)
    assert item_records.existing_receipt(retry) == original
    with pytest.raises(RecordConflict):
        item_records.existing_receipt(replace(original, payload={"value": "changed"}))
    with pytest.raises(RecordConflict):
        item_records.append(replace(original, payload={"value": "changed"}))
    with pytest.raises(ValueError, match="scenario 'math'"):
        item_records.append(replace(original, scenario="code"))

    calls = [conversation(f"call-{turns}", turns) for turns in range(1, 4)]
    report = plain("report", "first")
    for item in (*calls, report):
        item_records.append(item)
    assert item_records.replay("math") == (original, *calls, report)
    page = item_records.replay_page("math", limit=1)
    sequence = page[0][0]
    assert page[0][1] == original
    assert item_records.replay_page("math", after_sequence=sequence, limit=1)[0][1] == calls[0]
    assert item_records.replay("math", offset=4, limit=1) == (report,)
    assert item_records.count("math") == 5
    assert item_records.count("math", request_type=RequestType.REPORT) == 1
    assert item_records.count("math", after_sequence=sequence) == 4
    assert [stored.item for stored in item_records.audit_page("math", request_type=RequestType.REPORT)] == [report]
    assert item_records.get_for_audit("math", "report").item == report
    assert item_records.get("math", "missing") is None
    assert item_records.get("code", "first") is None and item_records.replay("code") == ()
    assert item_records.last_sequence("math") == item_records.get_for_audit("math", "report").sequence


def test_a_batch_of_records_stores_all_or_none_and_a_repeat_is_a_retry(item_records):
    first, second = plain("first"), plain("second")
    with pytest.raises(RecordConflict):
        item_records.append_many([first, second, replace(first, payload={"changed": True})])
    assert item_records.count("math") == 0
    assert item_records.append_many([first, second, first]) == (
        AppendResult(first, True),
        AppendResult(second, True),
        AppendResult(first, False),
    )
    assert item_records.append_many([replace(first, created_at=9.0)]) == (AppendResult(first, False),)
    with pytest.raises(RecordConflict):
        item_records.append_many([plain("third"), replace(first, payload={"changed": True})])
    assert item_records.get("math", "third") is None


def test_an_archive_ends_the_writes_of_the_open_records_and_the_loss_is_the_data_stores(item_records):
    data = item_records.data
    item_records.append(plain("first"))
    assert data.prune_records(0) == 1
    assert item_records.loss("math").record_count == 1
    data.remove_scenario("acme", "math")
    with pytest.raises(ScenarioArchived):
        item_records.append(plain("late"))
    assert item_records.replay("math") == ()


def test_a_copy_in_its_origin_has_the_item_id_and_the_item_references():
    item = record_item(plain("report", "first"))
    copied = copied_record(item.item_id, item.content, "math", "math")
    assert (copied.agent_record_id, copied.references) == ("math/report", ("math/first",))


def test_a_scenarios_records_and_its_trainers_never_read_its_evaluation_calls(item_records):
    data = item_records.data
    own = conversation("call-1", 1)
    item_records.append(own)
    request = dict(conversation("unused", 2).payload)
    response = request.pop("response")
    summary = EvaluationCallSummary(0, 0, 1, 0, "/v1/chat/completions", 200, 5.0, 0.1, 3, 1, "stop", 0)
    call = evaluation_call_item("math", "a1", "e1", EvaluationCall(summary, request, response))
    data.create("acme", [call], generation=item_records.generation)
    assert data.read("acme", call.item_id) is not None
    assert item_records.replay("math") == (own,)
    assert item_records.count("math") == 1 and item_records.count("math", request_type=RequestType.INFERENCE) == 1
    assert item_records.get("math", "e1:0:0:1:0") is None
    assert item_records.last_sequence("math") == item_records.get_for_audit("math", "call-1").sequence
    assert item_records.trajectory("math", "e1:0:0:1:0") == ()
    assert item_records.trajectory("math", "call-1") == ("call-1",)
    assert used_records(data, "acme", "code")[1] == () and used_record(data, "acme", "code", call.item_id) is None
    assert data.scenarios_with_records("acme") == ("math",)
