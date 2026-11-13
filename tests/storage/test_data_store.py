"""The data store contract (RFC #1028), for every store: items of content and metadata, each in one owner."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import closing
from dataclasses import replace

import pytest
import sqlalchemy
import sqlalchemy.event
from sqlalchemy import func, select
from tests.support.conversations import conversation
from tests.support.trajectories import policy_trajectory

from reef.core.atif_payloads import canonical_json
from reef.core.errors import ReefError
from reef.core.evaluation import EvaluationCall, EvaluationCallSummary, EvaluationTrajectory
from reef.core.job_data import JobData
from reef.core.records_types import AgentRecord, RecordLoss, RequestType
from reef.core.trajectory_index import call_key, trajectory_calls
from reef.service.assembly import build_dispatcher
from reef.service.deploy.service_config import ServiceSettings
from reef.storage.data import (
    ItemConflictError,
    ItemNotFoundError,
    MemoryDataStore,
    NewItem,
    RecordLink,
    ScenarioArchivedError,
    ScenarioHold,
)
from reef.storage.item_kinds import json_digest
from reef.storage.job_data import job_data_key
from reef.storage.record_items import (
    evaluation_call_item,
    evaluation_call_items,
    evaluation_trajectory_item,
    evaluation_trajectory_items,
    record_item,
    stored_evaluation_call,
)
from reef.storage.sql_data import PostgresDataStore, SQLiteDataStore
from reef.storage.sql_records import TrajectoryNodes
from reef.storage.task_collections import TaskCollection, collection_benchmarks
from reef.storage.tasks import LocalTaskFileStore, files_digest

ATIF = {"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1, "message": "hi"}]}
ORIGIN = {"scenario": "coding", "artifact_id": "a1"}


def record_content(record_id: str) -> dict:
    """The content of the record item of the inference ``record_id`` (``reef.storage.record_items``)."""
    return {
        "agent_record_id": record_id,
        "request_type": "inference",
        "payload": {},
        "created_at": 1.0,
        "references": [],
        "artifact_id": None,
    }


def artifact_content(source_uri: str | None, component: str | None) -> dict:
    return {"source_uri": source_uri, "kind": None, "component": component}


def trajectory(item_id: str, text: str = "hi", **metadata) -> NewItem:
    content = {**ATIF, "steps": [{"step_id": 1, "message": text}]}
    return NewItem(item_id, content, {"kind": "trajectory", **metadata})


@pytest.fixture(params=["memory", "sqlite-memory", "sqlite", "postgres"])
def store(request, tmp_path):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    if request.param == "memory":
        opened = MemoryDataStore(task_files)
    elif request.param == "sqlite-memory":
        opened = SQLiteDataStore(None, task_files)
    elif request.param == "sqlite":
        opened = SQLiteDataStore(tmp_path / "data", task_files)
    else:
        url, schema = request.getfixturevalue("postgres_config")
        opened = PostgresDataStore(url, task_files, schema=schema)
    with closing(task_files), closing(opened):
        yield opened


@pytest.fixture(params=["sqlite", "postgres"])
def open_durable_store(request, tmp_path):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        yield lambda: PostgresDataStore(url, task_files, schema=schema)
    else:
        yield lambda: SQLiteDataStore(tmp_path / "data", task_files)
    task_files.close()


def test_created_items_keep_their_content_and_arrival_order(store):
    first, second = store.create("acme", [trajectory("t1", "one"), trajectory("t2", "two", note="kept")])
    assert first.sequence < second.sequence
    assert first.content is None and second.content is None, "a create reads no content back"
    assert store.read("acme", "t1").content["steps"][0]["message"] == "one"
    assert first.digest == json_digest(store.read("acme", "t1").content)
    assert second.metadata == {"kind": "trajectory", "owner": "acme", "scenarios": {}, "note": "kept"}
    assert replace(store.read("acme", "t2"), content=None) == second
    assert [item.item_id for item in store.query("acme")] == ["t1", "t2"]
    assert [item.item_id for item in store.query("acme", after=first.sequence)] == ["t2"]
    assert [item.item_id for item in store.query("acme", limit=1)] == ["t1"]


def test_a_read_of_many_items_gives_each_known_one_once_in_the_order_asked(store):
    store.create("acme", [trajectory("t1", "one"), trajectory("t2", "two"), trajectory("t3", "three")])
    store.create("globex", [trajectory("t4", "globex")])
    found = store.read_many("acme", ["t3", "unknown", "t1", "t3", "t4"])
    assert [item.item_id for item in found] == ["t3", "t1"]
    assert [item.content for item in found] == [store.read("acme", "t3").content, store.read("acme", "t1").content]
    assert store.read_many("acme", []) == ()


def test_only_the_create_that_stores_an_item_marks_it_inserted(store):
    (first,) = store.create("acme", [trajectory("t1")])
    assert first.inserted
    again, second = store.create("acme", [trajectory("t1"), trajectory("t2")])
    assert (again.inserted, second.inserted) == (False, True)
    assert again == first
    assert not store.read("acme", "t1").inserted


def test_the_same_content_again_changes_nothing_and_other_content_is_a_conflict_that_stores_nothing(store):
    (first,) = store.create("acme", [trajectory("t1", "one")])
    assert store.create("acme", [trajectory("t1", "one")]) == (first,)
    with pytest.raises(ItemConflictError):
        store.create("acme", [trajectory("t3", "three"), trajectory("t1", "changed")])
    assert [item.item_id for item in store.query("acme")] == ["t1"]


def test_each_owner_sees_only_its_own_items(store):
    store.create("acme", [trajectory("t1", "acme")])
    store.create("globex", [trajectory("t1", "globex")])
    assert store.read("acme", "t1").content["steps"][0]["message"] == "acme"
    assert store.read("globex", "t1").content["steps"][0]["message"] == "globex"
    assert store.read("initech", "t1") is None
    assert [item.owner for item in store.query("globex")] == ["globex"]
    assert store.delete("globex", "t1") is True
    assert store.read("acme", "t1") is not None


def test_metadata_says_what_an_item_is_and_which_scenarios_use_it(store):
    (record,) = store.create(
        "acme",
        [
            NewItem(
                "coding/r1",
                record_content("r1"),
                {"kind": "record", "origin": ORIGIN, "scenarios": {"coding": "train"}},
            )
        ],
    )
    assert record.kind == "record"
    assert record.metadata["origin"] == ORIGIN
    # An imported record names no snapshot and no artifact.
    unserved = {**ORIGIN, "artifact_id": None}
    (imported,) = store.create(
        "acme", [NewItem("coding/r2", record_content("r2"), {"kind": "record", "origin": unserved})]
    )
    assert imported.metadata["origin"] == unserved
    # An artifact item has an origin when a scenario made it, and no scenarios: a scenario imports it.
    made, uploaded = store.create(
        "acme",
        [
            NewItem("coding/a1", artifact_content(None, "weights"), {"kind": "artifact", "origin": ORIGIN}),
            NewItem("upload-1", artifact_content("s3://bucket/u/", None), {"kind": "artifact"}),
        ],
    )
    assert (made.metadata["origin"], made.metadata["scenarios"]) == (ORIGIN, {})
    assert "origin" not in uploaded.metadata
    with pytest.raises(ValueError, match="importing it"):
        store.update("acme", ["upload-1"], {"scenarios": {"coding": "train"}})
    bad = [
        (trajectory("x", kind="dataset"), "metadata.kind"),
        (NewItem("x", ATIF, {"kind": "trajectory", "owner": "globex"}), "metadata.owner"),
        (NewItem("x", {}, {"kind": "record"}), "metadata.origin"),
        (NewItem("x", {}, {"kind": "record", "origin": {**ORIGIN, "artifact_id": ""}}), "origin.artifact_id"),
        (trajectory("x", origin=ORIGIN), "stage sample.s metadata.origin"),
        (NewItem("x", {}, {"kind": "artifact", "scenarios": {"coding": "train"}}), "importing it"),
        (NewItem("x", {}, {"kind": "artifact", "origin": unserved}), "names its artifact_id"),
        (trajectory("x", scenarios={"coding": "eval"}), "always train"),
        (trajectory("x", scenarios={"coding": "dev"}), "train or eval"),
        (NewItem("x", ATIF, {"kind": "task"}), "task directory"),
        # The content of each kind follows its rules (reef.storage.item_kinds).
        (NewItem("x", {"steps": []}, {"kind": "trajectory"}), "ATIF"),
        (NewItem("coding/x", {"request": {}}, {"kind": "record", "origin": ORIGIN}), "exactly the keys"),
        (NewItem("other/r1", record_content("r1"), {"kind": "record", "origin": ORIGIN}), "has the ID"),
        (NewItem("x", artifact_content(None, None), {"kind": "artifact"}), "names its source_uri"),
    ]
    for item, reason in bad:
        with pytest.raises(ValueError, match=reason):
            store.create("acme", [item])
    with pytest.raises(ValueError, match="given twice"):
        store.create("acme", [trajectory("y"), trajectory("y")])
    with pytest.raises(ValueError, match="owner"):
        store.create("a/b", [trajectory("y")])


def test_an_update_patches_the_metadata_and_a_scenario_query_follows_it(store):
    store.create("acme", [trajectory("t1"), trajectory("t2"), trajectory("t3", scenarios={"other": "train"})])
    store.update("acme", ["t1", "t2"], {"scenarios": {"coding": "train"}, "label": "week 43"})
    assert [item.item_id for item in store.query("acme", scenario="coding")] == ["t1", "t2"]
    (patched,) = store.update("acme", ["t2"], {"scenarios": {"coding": None}, "label": None})
    assert patched.metadata == {"kind": "trajectory", "owner": "acme", "scenarios": {}}
    assert [item.item_id for item in store.query("acme", scenario="coding")] == ["t1"]
    assert [item.item_id for item in store.query("acme", scenario="other")] == ["t3"]
    assert store.read("acme", "t2").content == store.read("acme", "t1").content


def test_a_scenario_reads_its_items_in_the_order_that_their_uses_began(store, tmp_path):
    store.create("acme", [trajectory("t1", scenarios={"coding": "train"}), trajectory("t2")])
    store.create("acme", [trajectory("t3", "three", scenarios={"coding": "train"})])

    def used(after=0):
        return [
            (item.item_id, sequence)
            for sequence, item in (
                (use.sequence, use.item)
                for use in store.scenario_items("acme", "coding", kinds=("trajectory", "task"), after=after)
            )
        ]

    first = used()
    assert [item_id for item_id, _ in first] == ["t1", "t3"]
    assert store.scenario_items("acme", "coding", kinds=("task",)) == ()
    assert store.scenario_items("acme", "coding", kinds=("trajectory",))[0].item.content == ATIF
    # An older item that joins later is read after the others; a metadata change keeps a use's sequence.
    store.update("acme", ["t2"], {"scenarios": {"coding": "train"}})
    store.update("acme", ["t1"], {"label": "week 43"})
    assert [item_id for item_id, _ in used(after=first[-1][1])] == ["t2"]
    assert used()[:2] == first
    # A use that ends and a use whose split changes: the task's new split is a new use.
    task = tmp_path / "fix-bug"
    task.mkdir()
    (task / "instruction.md").write_text("Fix the bug.\n")
    store.create("acme", [NewItem("fix-bug", task, {"kind": "task", "scenarios": {"coding": "eval"}})])
    before = used()[-1][1]
    store.update("acme", ["fix-bug", "t3"], {"scenarios": {"coding": "train"}})
    store.update("acme", ["t1"], {"scenarios": {"coding": None}})
    assert [item_id for item_id, _ in used(after=before)] == ["fix-bug"]
    assert [item_id for item_id, _ in used()] == ["t3", "t2", "fix-bug"]
    store.delete("acme", "t2")
    assert [item_id for item_id, _ in used()] == ["t3", "fix-bug"]


def index_names(store) -> set[str]:
    """The names of the indexes of the ``data_item`` table of the SQL data store ``store``."""
    items = store.items_table
    with store.transaction(read_only=True) as connection:
        if connection.dialect.name == "sqlite":
            statement = sqlalchemy.text("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = :table")
            return set(connection.execute(statement, {"table": items.name}).scalars())
        statement = sqlalchemy.text(
            "SELECT indexname FROM pg_indexes WHERE schemaname = :schema AND tablename = :table"
        )
        return set(connection.execute(statement, {"schema": items.schema, "table": items.name}).scalars())


def test_an_update_never_changes_the_fixed_metadata_and_refuses_as_a_whole(store):
    store.create("acme", [trajectory("t1"), trajectory("t2")])
    for patch, reason in [
        ({"kind": "task"}, "metadata.kind"),
        ({"owner": "globex"}, "metadata.owner"),
        ({"origin": ORIGIN}, "metadata.origin"),
        ({"scenarios": {"coding": "eval"}}, "always train"),
    ]:
        with pytest.raises(ValueError, match=reason):
            store.update("acme", ["t1"], patch)
    with pytest.raises(ItemNotFoundError):
        store.update("acme", ["t1", "missing"], {"scenarios": {"coding": "train"}})
    assert store.query("acme", scenario="coding") == ()


def test_a_task_items_files_are_replaced_in_place_and_its_uses_are_read_again(store, tmp_path):
    first, second = tmp_path / "first" / "fix-bug", tmp_path / "second" / "fix-bug"
    for directory, text in ((first, "Fix the bug.\n"), (second, "Fix the bug in parse.\n")):
        directory.mkdir(parents=True)
        (directory / "instruction.md").write_text(text)
    (item,) = store.create("acme", [NewItem("fix-bug", first, {"kind": "task", "scenarios": {"coding": "train"}})])
    old_path = store.task_path(store.read("acme", "fix-bug"))
    used_before = store.scenario_items("acme", "coding", kinds=("task",))[0].sequence

    replaced = store.replace_task("acme", "fix-bug", second)
    assert (replaced.sequence, replaced.metadata, replaced.digest) == (
        item.sequence,
        item.metadata,
        files_digest(second),
    )
    assert (store.task_path(store.read("acme", "fix-bug")) / "instruction.md").read_text() == "Fix the bug in parse.\n"
    # A reader that opened the old files can go on, and a trainer reads the task again after its cursor.
    assert (old_path / "instruction.md").read_text() == "Fix the bug.\n"
    assert [
        use.item.item_id for use in store.scenario_items("acme", "coding", kinds=("task",), after=used_before)
    ] == ["fix-bug"]
    # The same files again change nothing, not even the use.
    (used_after,) = [use.sequence for use in store.scenario_items("acme", "coding", kinds=("task",))]
    assert store.replace_task("acme", "fix-bug", second) == replaced
    assert store.scenario_items("acme", "coding", kinds=("task",))[0].sequence == used_after

    with pytest.raises(ItemNotFoundError):
        store.replace_task("acme", "missing", second)
    with pytest.raises(ValueError, match="same name"):
        store.replace_task("acme", "fix-bug", first.parent)
    store.create("acme", [trajectory("t1")])
    trajectory_task = tmp_path / "t1"
    trajectory_task.mkdir()
    with pytest.raises(ValueError, match="only a task's files"):
        store.replace_task("acme", "t1", trajectory_task)
    assert store.read("acme", "fix-bug").digest == replaced.digest


def test_a_task_item_keeps_its_directory_in_the_owners_partition(store, tmp_path):
    task = tmp_path / "fix-bug"
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text("Fix the bug.\n")
    (task / "tests" / "data.bin").write_bytes(bytes(range(256)))
    (item,) = store.create("acme", [NewItem("fix-bug", task, {"kind": "task", "scenarios": {"coding": "eval"}})])
    assert item.content is None
    assert item.digest == files_digest(task)
    path = store.task_path(item)
    assert (path / "instruction.md").read_text() == "Fix the bug.\n"
    assert (path / "tests" / "data.bin").read_bytes() == bytes(range(256))
    assert [found.item_id for found in store.query("acme", kind="task", scenario="coding")] == ["fix-bug"]
    with pytest.raises(ValueError, match="not a task"):
        store.task_path(store.create("acme", [trajectory("t1")])[0])


def test_a_task_names_its_benchmarks_and_a_query_keeps_the_tasks_of_one(store, tmp_path):
    def task(name: str, **metadata) -> NewItem:
        (tmp_path / name).mkdir()
        (tmp_path / name / "instruction.md").write_text(f"Do {name}.\n")
        return NewItem(name, tmp_path / name, {"kind": "task", **metadata})

    store.create(
        "acme",
        [task("add", benchmarks=["lite", "full"]), task("sort", benchmarks=["full"]), task("plain"), trajectory("t1")],
    )
    store.create("globex", [task("other", benchmarks=["full"])])
    assert [item.item_id for item in store.query("acme", benchmark="full")] == ["add", "sort"]
    assert [item.item_id for item in store.query("acme", kind="task", benchmark="lite")] == ["add"]
    assert store.query("acme", benchmark="unknown") == ()
    assert store.query("acme", kind="trajectory", benchmark="full") == ()
    # A patch replaces the list, and null removes it.
    store.update("acme", ["sort"], {"benchmarks": ["lite"]})
    store.update("acme", ["add"], {"benchmarks": None})
    assert [item.item_id for item in store.query("acme", benchmark="lite")] == ["sort"]
    assert store.query("acme", benchmark="full") == ()
    for benchmarks in ("lite", ["lite", "lite"], [""], [1]):
        with pytest.raises(ValueError, match="benchmarks"):
            store.update("acme", ["sort"], {"benchmarks": benchmarks})
    with pytest.raises(ValueError, match="benchmarks"):
        store.create("acme", [trajectory("t2", benchmarks=["lite"])])
    with pytest.raises(ValueError, match="benchmark"):
        store.query("acme", benchmark="")


def test_an_owner_stores_replaces_and_removes_its_collections(store):
    coding = TaskCollection("coding", benchmarks=("lite", "full"))
    store.write_collection("acme", coding)
    store.write_collection("acme", TaskCollection("all", collections=("coding",), average="benchmark"))
    assert store.collections("acme") == (TaskCollection("all", (), ("coding",), "benchmark"), coding)
    assert store.collections("globex") == ()
    store.write_collection("acme", TaskCollection("coding", benchmarks=("lite",)))
    assert [collection.benchmarks for collection in store.collections("acme")] == [(), ("lite",)]
    assert store.delete_collection("acme", "coding") and not store.delete_collection("acme", "coding")
    assert [collection.name for collection in store.collections("acme")] == ["all"]


def test_a_collection_has_its_benchmarks_and_those_of_the_collections_it_names():
    stored = {
        collection.name: collection
        for collection in (
            TaskCollection("coding", benchmarks=("lite", "full")),
            TaskCollection("math", benchmarks=("algebra", "lite")),
            TaskCollection("all", benchmarks=("extra",), collections=("coding", "math")),
            TaskCollection("first", collections=("second",)),
            TaskCollection("second", collections=("first",)),
            TaskCollection("open", collections=("missing",)),
        )
    }
    assert collection_benchmarks(stored, "all") == ("extra", "lite", "full", "algebra")
    with pytest.raises(ValueError, match="first -> second -> first make a cycle"):
        collection_benchmarks(stored, "first")
    with pytest.raises(ValueError, match="no collection 'missing'"):
        collection_benchmarks(stored, "open")
    assert TaskCollection.from_json(stored["all"].to_json()) == stored["all"]
    empty = TaskCollection.from_json({"name": "empty"})
    assert (empty.benchmarks, empty.collections) == ((), ())
    assert collection_benchmarks({"empty": empty}, "empty") == ()
    for value in (
        {"name": "a/b", "benchmarks": ["lite"]},
        {"name": "self", "collections": ["self"]},
        {"name": "twice", "benchmarks": ["lite", "lite"]},
        {"name": "mean", "benchmarks": ["lite"], "average": "macro"},
        {"name": "more", "benchmarks": ["lite"], "tasks": []},
        {"benchmarks": ["lite"]},
    ):
        with pytest.raises(ValueError):
            TaskCollection.from_json(value)


def test_groups_count_the_items_of_each_metadata_value_newest_first(store):
    store.create("acme", [trajectory("a1", set_id="a", set_name="A"), trajectory("a2", set_id="a")])
    store.create("acme", [trajectory("n1"), trajectory("b1", set_id="b", set_name="B"), trajectory("x", set_id=7)])
    store.create("acme", calls(1))
    store.create("other", [trajectory("o1", set_id="a")])
    groups = store.groups("acme", kind="trajectory", key="set_id")
    assert [(group.value, group.item_count, group.metadata.get("set_name")) for group in groups] == [
        ("b", 1, "B"),
        (None, 2, None),
        ("a", 2, "A"),
    ]
    assert groups[2].last_sequence == groups[2].first_sequence + 1
    assert store.groups("acme", kind="trajectory", key="set_id", limit=2) == groups[:2]
    assert store.groups("acme", kind="trajectory", key="set_id", before=groups[1].first_sequence) == groups[2:]
    with pytest.raises(ValueError, match="key"):
        store.groups("acme", kind="trajectory", key="set.id")
    # A query reads the items of one group, also of the group without a string value.
    for value, item_ids in [("a", ["a1", "a2"]), (None, ["n1", "x"])]:
        found = store.query("acme", kind="trajectory", group=("set_id", value), content=False)
        assert [item.item_id for item in found] == item_ids
    with pytest.raises(ValueError, match="key"):
        store.query("acme", group=("set.id", "a"))


def test_groups_of_a_scenario_count_only_the_items_it_uses(store):
    store.create("acme", [trajectory("a1", set_id="a"), trajectory("a2", set_id="a", scenarios={"coding": "train"})])
    store.create("acme", [trajectory("b1", set_id="b"), trajectory("a3", set_id="a", scenarios={"coding": "train"})])
    store.create("acme", [trajectory("c1", set_id="c", scenarios={"writing": "train"})])
    (group,) = store.groups("acme", kind="trajectory", key="set_id", scenario="coding")
    assert (group.value, group.item_count) == ("a", 2)
    assert [group.first_sequence, group.last_sequence] == [
        item.sequence for item in store.query("acme", scenario="coding")
    ]
    assert store.groups("acme", kind="trajectory", key="set_id", scenario="unused") == ()


def test_query_arguments_are_checked(store):
    for arguments, reason in [
        ({"kind": "dataset"}, "kind"),
        ({"after": -1}, "after"),
        ({"limit": 0}, "limit"),
        ({"id_prefix": ""}, "id_prefix"),
    ]:
        with pytest.raises(ValueError, match=reason):
            store.query("acme", **arguments)


def test_a_query_keeps_the_items_whose_id_starts_with_a_prefix(store):
    store.create("acme", [trajectory(item_id) for item_id in ("e1:1:1:0", "e1:1:10:0", "e1_1:1:0", "e1%1:1:0")])
    assert [item.item_id for item in store.query("acme", id_prefix="e1:1:1:")] == ["e1:1:1:0"]
    # A SQL pattern gives _ and % a meaning: the prefix matches them as characters only.
    assert [item.item_id for item in store.query("acme", id_prefix="e1_")] == ["e1_1:1:0"]
    assert [item.item_id for item in store.query("acme", id_prefix="e1%")] == ["e1%1:1:0"]


def calls(*turns: int) -> list[NewItem]:
    """The record items of calls of one conversation of the scenario ``coding``, with ``turns`` turns each."""
    return [record_item(conversation(f"call-{count}", count, scenario="coding")) for count in turns]


def test_record_items_that_share_messages_read_back_exactly(store):
    items = calls(1, 2, 3)
    store.create("acme", items)
    assert [canonical_json(item.content) for item in store.query("acme")] == [
        canonical_json(item.content) for item in items
    ]
    assert [item.content for item in store.query("acme", kind="record")] == [item.content for item in items]
    assert store.read("acme", "coding/call-2").content == items[1].content
    assert store.delete("acme", "coding/call-2")
    assert [item.content for item in store.query("acme")] == [items[0].content, items[2].content]
    store.create("acme", [items[1]])
    assert store.read("acme", "coding/call-2").content == items[1].content


def test_record_links_place_each_call_in_its_trajectory_and_skip_deleted_ancestors(store):
    retry = conversation("retry-2", 2, scenario="coding")
    review = conversation("call-3", 3, scenario="review")
    report = AgentRecord.create(
        scenario="coding",
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": ["call-3"]},
        agent_record_id="report-1",
        references=("call-3",),
    )
    store.create("acme", [*calls(1, 2, 3), record_item(retry), record_item(review), record_item(report)])
    entries = {turns: call_key(conversation("c", turns).payload).entry_count for turns in (1, 2, 3)}
    ids = [
        "coding/call-1",
        "coding/call-2",
        "coding/call-3",
        "coding/retry-2",
        "review/call-3",
        "coding/report-1",
        "x",
    ]
    links = store.record_links("acme", ids)
    # A retry has its parent's entries; another scenario starts its own tree; a report and an unknown ID have none.
    assert links == {
        "coding/call-1": RecordLink(None, entries[1]),
        "coding/call-2": RecordLink("coding/call-1", entries[2]),
        "coding/call-3": RecordLink("coding/call-2", entries[3]),
        "coding/retry-2": RecordLink("coding/call-2", entries[2]),
        "review/call-3": RecordLink(None, entries[3]),
    }
    path = ["coding/call-1", "coding/call-2", "coding/retry-2"]
    assert trajectory_calls([(item_id, links[item_id].entry_count) for item_id in path]) == (
        "coding/call-1",
        "coding/retry-2",
    )
    assert store.record_links("globex", ids) == {}
    # A deleted ancestor is skipped: the link names the nearest ancestor whose item is stored.
    assert store.delete("acme", "coding/call-2")
    assert store.record_links("acme", ["coding/call-3", "coding/retry-2", "coding/call-2"]) == {
        "coding/call-3": RecordLink("coding/call-1", entries[3]),
        "coding/retry-2": RecordLink("coding/call-1", entries[2]),
    }


def test_writes_of_one_scenario_wait_and_writes_of_another_do_not(postgres_config, tmp_path):
    url, schema = postgres_config
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    with closing(task_files), closing(PostgresDataStore(url, task_files, schema=schema)) as store:
        new = {"kind": "trajectory", "scenarios": {"s1": "train"}}
        with ThreadPoolExecutor(2) as pool, store.transaction() as holder:
            store.lock_scenarios(holder, "acme", ["s1"])
            waits = pool.submit(store.create, "acme", [NewItem("a", ATIF, new)])
            other = {**new, "scenarios": {"s2": "train"}}
            assert pool.submit(store.create, "acme", [NewItem("b", ATIF, other)]).result(timeout=10)
            assert pool.submit(store.create, "globex", [NewItem("c", ATIF, new)]).result(timeout=10)
            with pytest.raises(FutureTimeoutError):
                waits.result(timeout=0.5)
        assert waits.result(timeout=10)


def test_creates_that_wait_for_one_scenario_share_a_transaction(postgres_config, tmp_path):
    url, schema = postgres_config
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    with closing(task_files), closing(PostgresDataStore(url, task_files, schema=schema)) as store:
        new = {"kind": "trajectory", "scenarios": {"s1": "train"}}
        inserts = []

        def listener(conn, cursor, statement, *_):
            if statement.startswith("INSERT INTO") and "data_item " in statement:
                inserts.append(statement)

        def until(waiting_count):
            deadline = time.monotonic() + 10
            while len(store.waiting.get(("acme", frozenset({"s1"})), [None])) != waiting_count:
                assert time.monotonic() < deadline
                time.sleep(0.01)

        def held(*item_ids):
            # The first create takes itself from the queue and waits for the lock; the others queue behind it.
            with ThreadPoolExecutor(len(item_ids)) as pool, store.transaction() as holder:
                store.lock_scenarios(holder, "acme", ["s1"])
                futures = [pool.submit(store.create, "acme", [NewItem(item_ids[0], ATIF, new)])]
                until(0)
                for item_id in item_ids[1:]:
                    content = {**ATIF, "agent": {"name": "other"}} if item_id == "a" else ATIF
                    futures.append(pool.submit(store.create, "acme", [NewItem(item_id, content, new)]))
                until(len(item_ids) - 1)
            return futures

        sqlalchemy.event.listen(store.engine, "before_cursor_execute", listener)
        futures = held("a", "b", "c", "d")
        assert [future.result(timeout=10)[0].item_id for future in futures] == ["a", "b", "c", "d"]
        assert len(inserts) == 2
        # A conflict in the shared transaction refuses only its own create.
        futures = held("e", "f", "a", "g")
        assert [future.exception(timeout=10) is None for future in futures] == [True, True, False, True]
        assert isinstance(futures[2].exception(), ItemConflictError)
        assert {item.item_id for item in store.query("acme")} == set("abcdefg")
        # Two creates of one ID in one transaction: only the first stores it.
        futures = held("h", "i", "i")
        assert sorted(future.result(timeout=10)[0].inserted for future in futures[1:]) == [False, True]
        assert not store.waiting


def test_removing_a_scenario_drops_all_its_uses_and_keeps_the_items(store):
    (made,) = calls(1)
    store.create("acme", [made, trajectory("t1", scenarios={"coding": "train", "review": "train"})])
    store.update("acme", [made.item_id], {"scenarios": {"review": "train"}})
    with pytest.raises(ValueError, match="stays in 'coding'"):
        store.update("acme", [made.item_id], {"scenarios": {"coding": None}})

    assert store.delete_scenario("acme", "coding") == 2
    assert store.query("acme", scenario="coding") == ()
    assert store.scenario_items("acme", "coding", kinds=("record", "trajectory")) == ()
    assert [(item.item_id, dict(item.scenarios)) for item in store.query("acme")] == [
        (made.item_id, {"review": "train"}),
        ("t1", {"review": "train"}),
    ]
    assert store.scenarios_with_records("acme") == ("review",)
    assert store.delete_scenario("acme", "coding") == 0
    assert store.delete_scenario("globex", "review") == 0


def test_a_write_of_an_ended_generation_stores_nothing(store):
    generation = store.scenario_generation("acme", "coding")
    assert store.scenario_generation("acme", "coding") == generation
    assert store.scenario_generation("globex", "coding") != generation
    first, second = calls(1, 2)
    assert store.create("acme", [first], generation=generation)[0].inserted
    with pytest.raises(ScenarioArchivedError, match="no longer current"):
        store.create("acme", [second], generation="other")

    store.delete_scenario("acme", "coding")
    with pytest.raises(ScenarioArchivedError):
        store.create("acme", [second], generation=generation)
    assert store.read("acme", second.item_id) is None
    renewed = store.scenario_generation("acme", "coding")
    assert renewed != generation
    assert store.create("acme", [second], generation=renewed)[0].inserted

    review = record_item(conversation("other", 1, scenario="review"))
    for items in ([trajectory("t1")], [first, review]):
        with pytest.raises(ValueError, match="one origin scenario"):
            store.create("acme", items, generation=renewed)


def test_pruning_deletes_the_oldest_record_items_and_counts_the_loss(store):
    reports = [
        record_item(
            AgentRecord.create(
                scenario="coding",
                request_type=RequestType.REPORT,
                payload={"score": 1.0},
                agent_record_id=f"r{index}",
                created_at=1.0,
            )
        )
        for index in range(3)
    ]
    created = [store.create(owner, [item])[0] for owner, item in zip(("acme", "globex", "acme"), reports, strict=True)]
    store.create("acme", [trajectory("t1")])
    size = len(canonical_json(reports[0].content))
    assert store.prune_records(3 * size) == 0
    assert store.prune_records(2 * size) == 1
    assert store.read("acme", "coding/r0") is None
    assert store.record_loss("acme", "coding") == RecordLoss(1, size, created[0].sequence, created[0].sequence)
    assert store.record_loss("globex", "coding") == RecordLoss()

    store.create("acme", calls(1, 2, 3))
    assert store.prune_records(0) == 5
    assert store.query("acme", kind="record") == () and store.query("globex") == ()
    assert [item.item_id for item in store.query("acme")] == ["t1"]
    assert store.record_loss("acme", "coding").record_count == 5
    if not isinstance(store, MemoryDataStore):
        assert node_counts(store)[0] == 0


def test_the_store_lists_the_scenarios_with_record_items(store):
    assert store.scenarios_with_records("acme") == ()
    store.create("acme", calls(1))
    store.create("acme", [NewItem("t", ATIF, {"kind": "trajectory", "scenarios": {"other": "train"}})])
    assert store.scenarios_with_records("acme") == ("coding",)
    store.update("acme", ["coding/call-1"], {"scenarios": {"reader": "train"}})
    assert store.scenarios_with_records("acme") == ("coding", "reader")
    assert store.scenarios_with_records("globex") == ()


def test_a_create_of_many_records_takes_as_many_statements_as_a_create_of_one(open_durable_store):
    statements = []

    def counted(store, items):
        statements.clear()

        def listener(conn, cursor, statement, *_):
            statements.append(statement)

        sqlalchemy.event.listen(store.engine, "before_cursor_execute", listener)
        try:
            return store.create("acme", items)
        finally:
            sqlalchemy.event.remove(store.engine, "before_cursor_execute", listener)

    with closing(open_durable_store()) as store:
        store.create("acme", calls(1))
        counted(store, calls(2))
        one = len(statements)
        # Each call continues the one before it, in this create too.
        created = counted(store, calls(*range(3, 13)))
        assert len(statements) == one
        assert [item.sequence for item in created] == sorted(item.sequence for item in created)
        links = store.record_links("acme", [item.item_id for item in created])
        assert [links[item.item_id].parent for item in created] == [f"coding/call-{turn}" for turn in range(2, 12)]
        assert [item.content for item in store.query("acme")] == [item.content for item in calls(*range(1, 13))]


def test_concurrent_updates_of_one_item_keep_every_use(open_durable_store):
    writers = 8
    with closing(open_durable_store()) as store:
        for round_number in range(10):
            item_id = f"trajectory-{round_number}"
            store.create("acme", [NewItem(item_id, ATIF, {"kind": "trajectory"})])
            barrier = threading.Barrier(writers)

            def use(index: int, item_id: str = item_id, barrier: threading.Barrier = barrier) -> None:
                barrier.wait()
                store.update("acme", [item_id], {"scenarios": {f"scenario-{index}": "train"}})

            with ThreadPoolExecutor(writers) as pool:
                list(pool.map(use, range(writers)))
            assert len(store.read("acme", item_id).metadata["scenarios"]) == writers


@pytest.mark.parametrize(
    "read",
    [
        lambda store: store.read("acme", "coding/call-2").item_id,
        lambda store: [item.item_id for item in store.query("acme", kind="record")],
        lambda store: list(store.record_links("acme", ["coding/call-1", "coding/call-2"])),
    ],
    ids=["read", "query", "record_links"],
)
def test_a_read_sees_one_snapshot_while_a_delete_commits(postgres_config, tmp_path, monkeypatch, read):
    url, schema = postgres_config
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    with (
        closing(task_files),
        closing(PostgresDataStore(url, task_files, schema=schema)) as reader,
        closing(PostgresDataStore(url, task_files, schema=schema)) as writer,
    ):
        writer.create("acme", calls(1, 2))
        paths = TrajectoryNodes.paths

        def paths_after_delete(self, connection, node_sequences, *columns):
            # The leaf call-2 and its node go between the read's item rows and its node rows.
            writer.delete("acme", "coding/call-2")
            return paths(self, connection, node_sequences, *columns)

        monkeypatch.setattr(TrajectoryNodes, "paths", paths_after_delete)
        before = read(reader)
        monkeypatch.undo()
        assert before in ("coding/call-2", ["coding/call-1", "coding/call-2"])
        assert reader.read("acme", "coding/call-2") is None


def test_a_record_link_is_checked_at_construction():
    for parent, entry_count in (("", 1), ("a", -1), ("a", True), (3, 1)):
        with pytest.raises(ValueError, match="record link"):
            RecordLink(parent, entry_count)


def node_counts(store) -> tuple[int, int]:
    """The number of trajectory nodes that the SQL data store ``store`` keeps, and the sum of their counts."""
    nodes = store.nodes_table
    with store.transaction() as connection:
        count, uses = connection.execute(select(func.count(), func.sum(nodes.c.record_count))).one()
    return count, uses or 0


def test_an_owner_keeps_a_node_while_a_record_item_needs_its_messages(open_durable_store):
    with closing(open_durable_store()) as store:
        store.create("acme", calls(1, 2, 3))
        store.create("other", calls(1))
        assert node_counts(store) == (3 + 1, 2 + 2 + 1 + 1)
        store.delete("acme", "coding/call-3")
        assert node_counts(store) == (2 + 1, 2 + 1 + 1)
        # call-2 continues call-1, so call-1's node stays while call-2 is stored.
        store.delete("acme", "coding/call-1")
        assert store.read("acme", "coding/call-2").content == calls(2)[0].content
        store.delete("acme", "coding/call-2")
        assert node_counts(store) == (1, 1)


def test_query_and_count_keep_the_records_of_one_origin_and_request_type(store):
    report = AgentRecord.create(
        scenario="coding", request_type=RequestType.REPORT, payload={"score": 1.0}, agent_record_id="rep"
    )
    other = record_item(conversation("other", 1, scenario="review"))
    store.create("acme", [*calls(1, 2), record_item(report), other, trajectory("t1", scenarios={"coding": "train"})])
    store.update("acme", [other.item_id], {"scenarios": {"coding": "train"}})

    def ids(**filters):
        found = [item.item_id for item in store.query("acme", kind="record", scenario="coding", **filters)]
        assert store.count("acme", kind="record", scenario="coding", **filters) == len(found)
        return found

    assert ids() == ["coding/call-1", "coding/call-2", "coding/rep", "review/other"]
    assert ids(origin="coding") == ["coding/call-1", "coding/call-2", "coding/rep"]
    assert ids(origin="coding", request_type="report") == ["coding/rep"]
    assert ids(request_type="inference") == ["coding/call-1", "coding/call-2", "review/other"]
    first = store.read("acme", "coding/call-1").sequence
    assert ids(origin="coding", after=first) == ["coding/call-2", "coding/rep"]
    assert store.count("acme") == 5
    newest = store.read("acme", "coding/rep").sequence
    assert store.last_sequence("acme", kind="record", scenario="coding", origin="coding") == newest
    assert store.last_sequence("acme", kind="record", origin="review") == store.read("acme", "review/other").sequence
    assert store.last_sequence("globex") == 0
    with pytest.raises(ValueError, match="kind='record'"):
        store.query("acme", origin="coding")
    with pytest.raises(ValueError, match="request_type"):
        store.count("acme", kind="record", request_type="training")


def test_items_survive_a_restart(open_durable_store):
    with closing(open_durable_store()) as first:
        first.create("acme", [trajectory("t1", scenarios={"coding": "train"})])
    with closing(open_durable_store()) as second:
        assert [item.item_id for item in second.query("acme", scenario="coding")] == ["t1"]
        assert second.delete("acme", "t1") is True
        assert second.delete("acme", "t1") is False


def deployment(tmp_path, **storage) -> ServiceSettings:
    return ServiceSettings(
        recipe="recipe",
        agent_record_dir=str(tmp_path / "records"),
        artifact_repository=str(tmp_path / "artifacts.git"),
        artifact_work_dir=str(tmp_path / "work"),
        artifact_cache_dir=str(tmp_path / "cache"),
        task_file_dir=str(tmp_path / "files"),
        **storage,
    )


@pytest.mark.parametrize(
    ("data_store", "store_class"),
    [
        (None, SQLiteDataStore),
        ("sqlite", SQLiteDataStore),
        ({"module": "reef.storage.data:MemoryDataStore"}, MemoryDataStore),
    ],
)
def test_the_deployment_builds_the_data_store_it_names_and_closes_it(tmp_path, data_store, store_class):
    dispatcher = build_dispatcher(deployment(tmp_path, data_store=data_store))
    try:
        store = dispatcher.data_store
        assert type(store) is store_class
        store.create("acme", [trajectory("t1")])
        assert [item.item_id for item in store.query("acme")] == ["t1"]
    finally:
        dispatcher.close()
    if store_class is SQLiteDataStore:
        assert (tmp_path / "records" / "data" / "data-store.sqlite3").is_file()
        assert store.connection is None


@pytest.mark.parametrize(
    ("data_store", "reason"),
    [
        ("unknown", "storage.data_store"),
        ("reef.storage.data:DataStore", "abstract"),
        ({"module": "reef.storage.sqlite:SQLiteScenarioStorage"}, "DataStore"),
        ("postgres", "record_backend: postgres"),
    ],
)
def test_the_deployment_refuses_an_invalid_data_store(data_store, reason):
    with pytest.raises(ValueError, match=reason):
        ServiceSettings(recipe="recipe", data_store=data_store)


def test_archiving_a_scenario_never_moves_an_owners_task_files(tmp_path):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    with closing(task_files), closing(MemoryDataStore(task_files)) as store:
        task = tmp_path / "fix-bug"
        task.mkdir()
        (task / "instruction.md").write_text("Fix the bug.\n")
        (item,) = store.create("acme", [NewItem("fix-bug", task, {"kind": "task"})])
        # Scenarios named after the owner, or after its partition without the separator, have other partitions.
        for scenario in ("acme", "data-acme", "data:acme", "data-owner:acme"):
            task_files.archive(scenario)
        assert (store.task_path(item) / "instruction.md").read_text() == "Fix the bug.\n"


def test_concurrent_creates_of_one_id_end_in_a_no_op_or_a_conflict(open_durable_store):
    from concurrent.futures import ThreadPoolExecutor

    stores = [open_durable_store() for _ in range(4)]
    try:

        def create(index: int) -> str:
            text = "changed" if index == 7 else "same"
            try:
                stores[index % len(stores)].create("acme", [trajectory(f"t{n}", text) for n in range(10)])
            except ItemConflictError:
                return "conflict"
            return "created"

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(create, range(8)))
        reader = stores[0]
        assert len(reader.query("acme")) == 10
        stored_text = reader.read("acme", "t0").content["steps"][0]["message"]
        # Whichever content came first is stored; every other call with that content changed nothing.
        assert outcomes.count("conflict") == (7 if stored_text == "changed" else 1)
    finally:
        for store in stores:
            store.close()


def test_a_record_retry_with_a_new_time_matches(store):
    (item,) = calls(1)
    (created,) = store.create("acme", [item])
    retry = NewItem(item.item_id, {**item.content, "created_at": item.content["created_at"] + 5}, item.metadata)
    assert store.create("acme", [retry]) == (created,)
    stored = store.read("acme", item.item_id)
    changed = {**item.content, "references": ["other"]}
    with pytest.raises(ItemConflictError, match="other content"):
        store.create("acme", [NewItem(item.item_id, changed, item.metadata)])
    assert store.read("acme", item.item_id) == stored


def test_the_store_keeps_one_job_data_directory_for_the_same_samples_until_no_job_uses_it(store):
    sample = policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0)
    first = store.write_job_directory(JobData("s:data:1", (sample,)))
    # A job of the same samples, under another data ID, uses the same directory.
    again = store.write_job_directory(JobData("s:request:r1", (sample,)))
    assert (again.uri, again.data_id, again.sample_count, again.kind) == (first.uri, "s:request:r1", 1, "trajectory")
    assert store.open_job_directory(again.to_dict()).read().samples == (sample,)
    other = store.write_job_directory(JobData("s:data:2", (sample, sample)))
    assert other.uri != first.uri
    # A directory that no job used since the time goes; a used one stays.
    store.write_job_directory_use(job_data_key(JobData("x", (sample,))), 10.0)
    assert store.delete_unused_job_directories(100.0) == 1
    with pytest.raises(FileNotFoundError):
        store.open_job_directory(first.to_dict()).read()
    assert store.open_job_directory(other.to_dict()).read().samples == (sample, sample)
    # The periodic work keeps a directory that a job used within the time to live.
    store.maintain()
    assert store.open_job_directory(other.to_dict()).read().samples == (sample, sample)


def evaluation_call(turns: int, attempt: int, index: int, status: int = 200, response: object = "reply"):
    """The record item of call ``index`` of attempt ``attempt`` of rollout (0, 0) of evaluation ``e1`` of ``coding``:
    the request of a conversation of ``turns`` turns, with its reply, or ``response`` when it is not ``"reply"``."""
    payload = dict(conversation("unused", turns, scenario="coding").payload)
    reply = payload.pop("response")
    summary = EvaluationCallSummary(
        0, 0, attempt, index, "/v1/chat/completions", status, 10.0 + index, 0.5, None, None, None, 0
    )
    call = EvaluationCall(summary, payload, reply if response == "reply" else response)
    return call, evaluation_call_item("coding", "a1", "e1", call)


def test_the_calls_of_one_rollout_leave_out_the_other_rollouts_of_the_evaluation(store):
    def call_item(task_index: int, repeat: int, caller: str = "agent") -> NewItem:
        summary = EvaluationCallSummary(
            task_index, repeat, 1, 0, "/v1/chat/completions", 200, 10.0, 0.5, None, None, None, 0, caller
        )
        return evaluation_call_item("coding", "a1", "e1", EvaluationCall(summary, None, None))

    store.create("acme", [call_item(1, 1), call_item(1, 10), call_item(11, 1), call_item(1, 1, "verifier")])
    kept = evaluation_call_items(store, "acme", "coding", "e1", content=False, rollout=(1, 1))
    assert [item.item_id for item in kept] == ["coding/e1:1:1:1:0", "coding/e1:1:1:1:verifier:0"]
    assert len(list(evaluation_call_items(store, "acme", "coding", "e1", content=False))) == 4


def test_an_evaluation_call_links_only_within_its_attempt_and_is_no_scenarios_data(store):
    first, second = evaluation_call(1, 1, 0), evaluation_call(2, 1, 1)
    # A failed call with no reply, and one with an error body, take part in the links as calls without a reply.
    no_reply = evaluation_call(3, 1, 2, status=502, response=None)
    error = evaluation_call(3, 1, 3, status=500, response={"error": {"message": "the engine stopped"}})
    retried = evaluation_call(2, 2, 0)
    store.create("acme", [item for _, item in (first, second, no_reply, error, retried)])
    # The scenario's own calls come later and have the same messages, but they never link to an evaluation call.
    store.create("acme", calls(1, 2))
    entries = {turns: call_key(conversation("c", turns).payload).entry_count for turns in (1, 2, 3)}
    ids = [item.item_id for _, item in (first, second, no_reply, error, retried)]
    assert store.record_links("acme", [*ids, "coding/call-1", "coding/call-2"]) == {
        "coding/e1:0:0:1:0": RecordLink(None, entries[1]),
        "coding/e1:0:0:1:1": RecordLink("coding/e1:0:0:1:0", entries[2]),
        "coding/e1:0:0:1:2": RecordLink("coding/e1:0:0:1:1", entries[3]),
        # The same request again retries the failed call.
        "coding/e1:0:0:1:3": RecordLink("coding/e1:0:0:1:2", entries[3]),
        "coding/e1:0:0:2:0": RecordLink(None, entries[2]),
        "coding/call-1": RecordLink(None, entries[1]),
        "coding/call-2": RecordLink("coding/call-1", entries[2]),
    }
    for call, item in (first, second, no_reply, error, retried):
        assert stored_evaluation_call(store.read("acme", item.item_id)) == call
    if not isinstance(store, MemoryDataStore):
        # The node of a continued call keeps only the messages after its parent's.
        nodes, items = store.nodes_table, store.items_table
        with store.transaction() as connection:
            kept = connection.execute(
                select(nodes.c.entries_json)
                .join(items, (items.c.owner == nodes.c.owner) & (items.c.sequence == nodes.c.node_sequence))
                .where(items.c.item_id == "coding/e1:0:0:1:1")
            ).scalar_one()
        assert len(json.loads(kept)) == entries[2] - entries[1]

    found = store.query("acme", kind="record", origin="coding", group=("evaluation_id", "e1"), content=False)
    assert [item.item_id for item in found] == ids
    assert all(item.scenarios == {} for item in found)
    assert [item.item_id for item in store.query("acme", scenario="coding")] == ["coding/call-1", "coding/call-2"]
    assert [use.item.item_id for use in store.scenario_items("acme", "coding", kinds=("record",))] == [
        "coding/call-1",
        "coding/call-2",
    ]
    with pytest.raises(ValueError, match="no scenario"):
        store.update("acme", [ids[0]], {"scenarios": {"coding": "train"}})
    with pytest.raises(ValueError, match="never changes"):
        store.update("acme", [ids[0]], {"evaluation_id": "e2"})
    # Pruning deletes and counts the scenario's own calls only; evaluation calls stay.
    assert store.prune_records(0) == 2
    assert store.record_loss("acme", "coding").record_count == 2
    assert [item.item_id for item in store.query("acme", kind="record")] == ids


def test_a_verifier_call_links_apart_from_the_agent_calls_and_a_call_kept_before_reads_as_an_agent_call(store):
    agent, _ = evaluation_call(1, 1, 0)
    summary = replace(agent.summary, caller="verifier", label="judge:quality")
    verifier = EvaluationCall(summary, agent.request, agent.response)
    verifier_item = evaluation_call_item("coding", "a1", "e1", verifier)
    assert verifier_item.item_id == "coding/e1:0:0:1:verifier:0"
    # A call kept before RFC #1293 has no caller and no label in its metadata.
    old = evaluation_call_item("coding", "a1", "e1", agent)
    location = {
        key: value for key, value in old.metadata["origin"]["evaluation"].items() if key not in ("caller", "label")
    }
    old = NewItem(
        old.item_id, old.content, {**old.metadata, "origin": {**old.metadata["origin"], "evaluation": location}}
    )
    store.create("acme", [old, verifier_item])
    # The same messages in the same attempt, but a verifier call never links to an agent call.
    assert store.record_links("acme", [verifier_item.item_id])[verifier_item.item_id].parent is None
    assert stored_evaluation_call(store.read("acme", old.item_id)) == agent
    assert stored_evaluation_call(store.read("acme", verifier_item.item_id)) == verifier
    with pytest.raises(ValueError, match="evaluation call has the ID"):
        content = {**verifier_item.content, "agent_record_id": "e1:0:0:1:0"}
        store.create("acme", [NewItem("coding/e1:0:0:1:0", content, verifier_item.metadata)])


def test_a_trajectory_that_a_verifier_wrote_is_no_scenarios_data(store):
    trajectory = EvaluationTrajectory(0, 0, 1, 0, "trajectory.json", ATIF)
    item = evaluation_trajectory_item("coding", "e1", trajectory)
    assert item.item_id == "coding/e1:0:0:1:verifier:trajectory:0"
    store.create("acme", [item])
    [found] = evaluation_trajectory_items(store, "acme", "coding", "e1", content=True)
    assert (found.content, found.scenarios, found.metadata["evaluation"]["name"]) == (ATIF, {}, "trajectory.json")
    assert list(evaluation_trajectory_items(store, "acme", "other", "e1", content=False)) == []
    for metadata, message in (
        ({**item.metadata, "scenarios": {"coding": "train"}}, "empty scenarios"),
        ({**item.metadata, "origin": {"scenario": "coding", "component": "", "records": []}}, "no origin"),
        ({**item.metadata, "evaluation_id": "e2"}, "evaluation_id"),
        ({**item.metadata, "evaluation": {**item.metadata["evaluation"], "attempt": 0}}, "attempt"),
        ({"kind": "task", "evaluation_id": "e1"}, "only an evaluation call"),
    ):
        with pytest.raises(ValueError, match=message):
            store.create("acme", [NewItem("coding/other", ATIF, metadata)])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda metadata: {**metadata, "scenarios": {"coding": "train"}}, "no scenario"),
        (lambda metadata: {**metadata, "evaluation_id": "e2"}, "evaluation_id"),
        (lambda metadata: {**metadata, "origin": {**metadata["origin"], "source": "replay"}}, "source"),
        (lambda metadata: {**metadata, "origin": {**metadata["origin"], "source": "inference"}}, "evaluation"),
        (
            lambda metadata: {
                **metadata,
                "origin": {**metadata["origin"], "evaluation": {**metadata["origin"]["evaluation"], "attempt": 0}},
            },
            "attempt",
        ),
        (
            lambda metadata: {
                **metadata,
                "origin": {**metadata["origin"], "evaluation": {**metadata["origin"]["evaluation"], "index": 7}},
            },
            "evaluation call has the ID",
        ),
        (
            lambda metadata: {
                **metadata,
                "origin": {
                    **metadata["origin"],
                    "evaluation": {**metadata["origin"]["evaluation"], "caller": "judge"},
                },
            },
            "caller",
        ),
        (
            lambda metadata: {
                **metadata,
                "origin": {**metadata["origin"], "evaluation": {**metadata["origin"]["evaluation"], "label": "x"}},
            },
            "an agent call has no label",
        ),
    ],
)
def test_the_metadata_of_an_evaluation_call_is_checked(store, change, message):
    _, item = evaluation_call(1, 1, 0)
    with pytest.raises(ValueError, match=message):
        store.create("acme", [NewItem(item.item_id, item.content, change(item.metadata))])
    plain = record_item(conversation("call-1", 1, scenario="coding"))
    with pytest.raises(ValueError, match="evaluation_id"):
        store.create("acme", [NewItem(plain.item_id, plain.content, {**plain.metadata, "evaluation_id": "e1"})])
    explicit = {**plain.metadata, "origin": {**plain.metadata["origin"], "source": "inference"}}
    assert store.create("acme", [NewItem(plain.item_id, plain.content, explicit)])[0].inserted


def benchmark_task(tmp_path, name: str, *benchmarks: str, **metadata) -> NewItem:
    """The task item ``name`` in ``benchmarks``, with a directory of one file."""
    directory = tmp_path / "tasks" / name
    directory.mkdir(parents=True)
    (directory / "instruction.md").write_text(f"Do {name}.\n")
    return NewItem(name, directory, {"kind": "task", "benchmarks": list(benchmarks), **metadata})


def uses_of(store, scenario: str, after: int = 0, kind: str = "task") -> list[tuple[str, str, str | None]]:
    """Each use by ``scenario`` of an item of ``kind`` of acme after ``after``, in use order: item, split, held set."""
    uses = store.scenario_items("acme", scenario, kinds=(kind,), after=after, content=False)
    assert [use.sequence for use in uses] == sorted({use.sequence for use in uses}), "each use has its own sequence"
    return [(use.item.item_id, use.split, use.held_set) for use in uses]


def test_a_scenario_that_holds_a_benchmark_uses_its_tasks_without_a_write_to_them(store, tmp_path):
    store.create("acme", [benchmark_task(tmp_path, "add", "math"), benchmark_task(tmp_path, "sub", "math", "old")])
    store.create("acme", [benchmark_task(tmp_path, "sort", "lists")])
    before = {item.item_id: item.metadata for item in store.query("acme", kind="task")}

    held = ScenarioHold("coding", "benchmark", "math", "train", 2)
    assert store.hold("acme", "coding", "benchmark", "math", "train") == held

    assert uses_of(store, "coding") == [("add", "train", "math"), ("sub", "train", "math")]
    assert {item.item_id: item.metadata for item in store.query("acme", kind="task")} == before
    assert [item.item_id for item in store.query("acme", scenario="coding")] == ["add", "sub"]
    assert store.item_splits("acme", "coding", ["add", "sort", "none"]) == {"add": frozenset({"train"})}
    assert store.holds("acme", "coding") == (held,)
    assert store.holds("acme", "other") == () and uses_of(store, "other") == []
    assert store.holding_scenarios("acme", "benchmark", ["math", "lists"]) == ("coding",)
    assert store.holding_scenarios("acme", "trajectory_set", ["math"]) == ()
    # The same hold again changes nothing, and another owner's scenario holds nothing of acme.
    last = store.last_use_sequence("acme", "coding")
    store.hold("acme", "coding", "benchmark", "math", "train")
    assert store.last_use_sequence("acme", "coding") == last
    assert store.hold("other", "coding", "benchmark", "math", "train").item_count == 0
    for arguments in (
        ("coding", "benchmark", "", "train"),
        ("", "benchmark", "math", "train"),
        ("coding", "benchmark", "math", "test"),
        ("coding", "collection", "math", "train"),
        ("coding", "trajectory_set", "day-1", "eval"),
    ):
        with pytest.raises(ValueError, match="a hold"):
            store.hold("acme", *arguments)


def test_a_task_has_one_use_for_its_own_mark_and_one_for_each_held_benchmark_of_it(store, tmp_path):
    store.create("acme", [benchmark_task(tmp_path, "add", "math", "easy", scenarios={"coding": "eval"})])
    store.hold("acme", "coding", "benchmark", "math", "train")
    store.hold("acme", "coding", "benchmark", "easy", "train")
    # A task in two held benchmarks is there twice, and it can train and be evaluated at the same time.
    assert uses_of(store, "coding") == [("add", "eval", None), ("add", "train", "math"), ("add", "train", "easy")]
    assert store.item_splits("acme", "coding", ["add"]) == {"add": frozenset({"train", "eval"})}
    assert [item.item_id for item in store.query("acme", scenario="coding")] == ["add"]

    # A scenario can hold one benchmark in both splits: the holds are apart, and each gives its own use.
    last = store.last_use_sequence("acme", "coding")
    assert store.hold("acme", "coding", "benchmark", "easy", "eval").split == "eval"
    assert uses_of(store, "coding", after=last) == [("add", "eval", "easy")]
    assert [(hold.name, hold.split) for hold in store.holds("acme", "coding")] == [
        ("easy", "eval"),
        ("easy", "train"),
        ("math", "train"),
    ]

    assert store.release("acme", "coding", "benchmark", "math", "train")
    assert not store.release("acme", "coding", "benchmark", "math", "train")
    assert not store.release("acme", "coding", "benchmark", "math", "eval")
    assert store.release("acme", "coding", "benchmark", "easy", "train")
    assert uses_of(store, "coding") == [("add", "eval", None), ("add", "eval", "easy")]
    # The end of a scenario ends its holds, so a new scenario of that name uses nothing.
    assert store.delete_scenario("acme", "coding") == 1
    assert uses_of(store, "coding") == [] and store.holds("acme", "coding") == ()
    assert store.holding_scenarios("acme", "benchmark", ["easy"]) == ()


def test_a_task_that_joins_or_leaves_a_held_benchmark_joins_or_leaves_the_scenario(store, tmp_path):
    store.create("acme", [benchmark_task(tmp_path, "add", "math"), benchmark_task(tmp_path, "sort")])
    store.hold("acme", "coding", "benchmark", "math", "train")
    store.hold("acme", "review", "benchmark", "math", "eval")
    last = store.last_use_sequence("acme", "coding")

    # A patch of the benchmarks and a new item both give the holding scenarios a use, after the earlier uses.
    store.update("acme", ["sort"], {"benchmarks": ["math", "lists"]})
    store.create("acme", [benchmark_task(tmp_path, "mul", "math")])
    assert uses_of(store, "coding", after=last) == [("sort", "train", "math"), ("mul", "train", "math")]
    assert uses_of(store, "review") == [("add", "eval", "math"), ("sort", "eval", "math"), ("mul", "eval", "math")]
    assert store.holds("acme", "coding")[0].item_count == 3
    # One listing counts each held set: a task of two held benchmarks counts in each, and a set with no task has 0.
    # A task without a benchmarks list is in no benchmark.
    store.create("acme", [NewItem("loose", benchmark_task(tmp_path, "loose").content, {"kind": "task"})])
    store.hold("acme", "coding", "benchmark", "lists", "eval")
    store.hold("acme", "coding", "benchmark", "empty", "train")
    assert [(hold.name, hold.split, hold.item_count) for hold in store.holds("acme", "coding")] == [
        ("empty", "train", 0),
        ("lists", "eval", 1),
        ("math", "train", 3),
    ]
    store.release("acme", "coding", "benchmark", "lists", "eval")
    store.release("acme", "coding", "benchmark", "empty", "train")

    # A patch that leaves the benchmarks keeps the uses of the holds and their sequences.
    kept = store.scenario_items("acme", "coding", kinds=("task",), content=False)
    store.update("acme", ["add"], {"scenarios": {"coding": "eval"}, "note": "x"})
    after = store.scenario_items("acme", "coding", kinds=("task",), content=False)
    assert [(use.sequence, use.held_set) for use in after if use.held_set] == [
        (use.sequence, use.held_set) for use in kept
    ]
    assert ("add", "eval", None) in uses_of(store, "coding")

    store.update("acme", ["add"], {"benchmarks": []})
    assert ("add", "train", "math") not in uses_of(store, "coding")
    assert [use[0] for use in uses_of(store, "review")] == ["sort", "mul"]
    # A deleted item and replaced files: the uses end, or begin again.
    last = store.last_use_sequence("acme", "coding")
    changed = tmp_path / "changed" / "mul"
    changed.mkdir(parents=True)
    (changed / "instruction.md").write_text("Multiply.\n")
    store.replace_task("acme", "mul", changed)
    assert uses_of(store, "coding", after=last) == [("mul", "train", "math")]
    assert store.delete("acme", "sort")
    assert [use[0] for use in uses_of(store, "review")] == ["mul"]


def test_a_scenario_that_holds_a_trajectory_set_uses_its_trajectories(store, tmp_path):
    store.create("acme", [trajectory("t1", set_id="day-1"), trajectory("t2", set_id="day-2"), trajectory("t3")])
    # A benchmark with the name of the set is another set: its task is no trajectory.
    store.create("acme", [benchmark_task(tmp_path, "add", "day-1")])
    held = store.hold("acme", "coding", "trajectory_set", "day-1", "train")
    assert held == ScenarioHold("coding", "trajectory_set", "day-1", "train", 1)
    assert uses_of(store, "coding", kind="trajectory") == [("t1", "train", "day-1")]
    assert uses_of(store, "coding") == [] and store.read("acme", "t1").scenarios == {}

    # A trajectory joins the set by a patch or by its create, and leaves it with another set.
    last = store.last_use_sequence("acme", "coding")
    store.update("acme", ["t3"], {"set_id": "day-1"})
    store.create("acme", [trajectory("t4", "four", set_id="day-1")])
    assert uses_of(store, "coding", after=last, kind="trajectory") == [
        ("t3", "train", "day-1"),
        ("t4", "train", "day-1"),
    ]
    store.update("acme", ["t1"], {"set_id": "day-2"})
    assert [use[0] for use in uses_of(store, "coding", kind="trajectory")] == ["t3", "t4"]
    assert store.holding_scenarios("acme", "trajectory_set", ["day-1", "day-2"]) == ("coding",)

    # The scenario holds the benchmark of that name too: the release of the set keeps the task's use.
    store.hold("acme", "coding", "benchmark", "day-1", "eval")
    assert [(hold.kind, hold.item_count) for hold in store.holds("acme", "coding")] == [
        ("benchmark", 1),
        ("trajectory_set", 2),
    ]
    assert store.release("acme", "coding", "trajectory_set", "day-1", "train")
    assert uses_of(store, "coding", kind="trajectory") == [] and uses_of(store, "coding") == [("add", "eval", "day-1")]


def test_a_store_upgrades_the_tables_of_the_version_before_holds(open_durable_store):
    with closing(open_durable_store()) as store:
        store.create("acme", [trajectory("t1"), trajectory("t2"), trajectory("t3")])
        schema = store.uses_table.schema
        prefix = "" if schema is None else f"{schema}."
        with store.transaction() as connection:
            # A database of version 1: the uses are in data_item_scenario, and no table records the version.
            connection.exec_driver_sql(f"DROP TABLE {prefix}data_schema_version")
            connection.exec_driver_sql(
                f"CREATE TABLE {prefix}data_item_scenario (sequence INTEGER PRIMARY KEY, owner TEXT, item_id TEXT,"
                " scenario TEXT, split VARCHAR(8))"
            )
            connection.exec_driver_sql(
                f"INSERT INTO {prefix}data_item_scenario (sequence, owner, item_id, scenario, split) VALUES"
                " (7, 'acme', 't3', 'coding', 'train'), (9, 'acme', 't1', 'coding', 'train')"
            )
    for _ in range(2):
        # The second open finds version 2 and changes nothing.
        with closing(open_durable_store()) as store:
            assert uses_of(store, "coding", kind="trajectory") == [("t3", "train", None), ("t1", "train", None)]
            with store.transaction(read_only=True) as connection:
                assert not sqlalchemy.inspect(connection).has_table("data_item_scenario", schema=schema)
                versions = connection.exec_driver_sql(f"SELECT version FROM {prefix}data_schema_version")
                assert versions.fetchall() == [(2,)]
    with closing(open_durable_store()) as store, store.transaction() as connection:
        connection.exec_driver_sql(f"UPDATE {prefix}data_schema_version SET version = 3")
    with pytest.raises(ReefError, match="a later Reef wrote them"):
        open_durable_store()
