"""The data store contract (RFC #1028), for every store: items of content and metadata, each in one owner."""

from __future__ import annotations

from contextlib import closing

import pytest
from sqlalchemy import func, select
from tests.support.conversations import conversation

from reef.core.atif_payloads import canonical_json
from reef.service.assembly import build_dispatcher
from reef.service.deploy.service_config import ServiceConfig
from reef.storage.data import ItemConflict, ItemNotFound, MemoryDataStore, NewItem
from reef.storage.item_kinds import json_digest
from reef.storage.record_items import record_item
from reef.storage.sql_data import PostgresDataStore, SQLiteDataStore
from reef.storage.tasks import LocalTaskFileStore, files_digest

ATIF = {"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1, "message": "hi"}]}
ORIGIN = {"scenario": "coding", "release_id": "r1", "artifact_id": "a1"}


def record_content(record_id: str) -> dict:
    """The content of the record item of the inference ``record_id`` (``reef.storage.record_items``)."""
    return {
        "agent_record_id": record_id,
        "request_type": "inference",
        "payload": {},
        "created_at": 1.0,
        "references": [],
        "artifact_ref": None,
    }


def artifact_content(source_uri: str | None, component: str | None) -> dict:
    return {"source_uri": source_uri, "kind": None, "component": component, "content_id": None}


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
    assert first.content["steps"][0]["message"] == "one"
    assert first.digest == json_digest(first.content)
    assert second.metadata == {"kind": "trajectory", "owner": "acme", "scenarios": {}, "note": "kept"}
    assert store.read("acme", "t2") == second
    assert [item.item_id for item in store.query("acme")] == ["t1", "t2"]
    assert [item.item_id for item in store.query("acme", after=first.sequence)] == ["t2"]
    assert [item.item_id for item in store.query("acme", limit=1)] == ["t1"]


def test_the_same_content_again_changes_nothing_and_other_content_is_a_conflict_that_stores_nothing(store):
    (first,) = store.create("acme", [trajectory("t1", "one")])
    assert store.create("acme", [trajectory("t1", "one")]) == (first,)
    with pytest.raises(ItemConflict):
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
    # An imported record names no release and no artifact.
    unserved = {**ORIGIN, "release_id": None, "artifact_id": None}
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
        (trajectory("x", origin=ORIGIN), "only a record or an artifact"),
        (NewItem("x", {}, {"kind": "artifact", "scenarios": {"coding": "train"}}), "importing it"),
        (NewItem("x", {}, {"kind": "artifact", "origin": unserved}), "names its release_id and artifact_id"),
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
    with pytest.raises(ItemNotFound):
        store.update("acme", ["t1", "missing"], {"scenarios": {"coding": "train"}})
    assert store.query("acme", scenario="coding") == ()


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


def test_query_arguments_are_checked(store):
    for arguments, reason in [({"kind": "dataset"}, "kind"), ({"after": -1}, "after"), ({"limit": 0}, "limit")]:
        with pytest.raises(ValueError, match=reason):
            store.query("acme", **arguments)


def calls(*turns: int) -> list[NewItem]:
    """The record items of calls of one conversation of the scenario ``coding``, with ``turns`` turns each."""
    return [record_item(conversation(f"call-{count}", count, scenario="coding")) for count in turns]


def test_record_items_that_share_messages_read_back_exactly(store):
    items = calls(1, 2, 3)
    created = store.create("acme", items)
    assert [canonical_json(item.content) for item in created] == [canonical_json(item.content) for item in items]
    assert [item.content for item in store.query("acme", kind="record")] == [item.content for item in items]
    assert store.read("acme", "coding/call-2").content == items[1].content
    assert store.delete("acme", "coding/call-2")
    assert [item.content for item in store.query("acme")] == [items[0].content, items[2].content]
    assert store.create("acme", [items[1]])[0].content == items[1].content


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


def test_an_upgrade_links_the_stored_record_items(open_durable_store):
    items = calls(1, 2, 3)
    with closing(open_durable_store()) as store:
        store.create("acme", [*items, trajectory("t1")])
        table = store.items_table
        with store.transaction() as connection:
            # Write the rows back as the earlier version kept them: the whole content, and no node table.
            for item in items:
                connection.execute(
                    table.update()
                    .where(table.c.item_id == item.item_id)
                    .values(content_json=canonical_json(item.content))
                )
            connection.exec_driver_sql(f"ALTER TABLE {table.fullname} DROP COLUMN message_field")
            store.nodes_table.drop(connection)
    with closing(open_durable_store()) as store:
        assert [item.content for item in store.query("acme", kind="record")] == [item.content for item in items]
        assert node_counts(store) == (3, 5)


def test_items_survive_a_restart(open_durable_store):
    with closing(open_durable_store()) as first:
        first.create("acme", [trajectory("t1", scenarios={"coding": "train"})])
    with closing(open_durable_store()) as second:
        assert [item.item_id for item in second.query("acme", scenario="coding")] == ["t1"]
        assert second.delete("acme", "t1") is True
        assert second.delete("acme", "t1") is False


def deployment(tmp_path, **storage) -> ServiceConfig:
    return ServiceConfig(
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
        ServiceConfig(recipe="recipe", data_store=data_store)


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
            except ItemConflict:
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
