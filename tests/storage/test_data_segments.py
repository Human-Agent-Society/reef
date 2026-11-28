"""The data store with a segment store: content of trajectory items and record nodes in JSON Lines segments."""

from __future__ import annotations

import json
from contextlib import closing

import pytest
from sqlalchemy import event, func, select
from tests.support.conversations import conversation

from reef.api.assembly import build_dispatcher
from reef.api.settings import ServiceSettings
from reef.storage.data.segments import (
    LocalSegmentStore,
    SegmentLine,
    SegmentNotFoundError,
    SegmentPartition,
    segment_lines,
)
from reef.storage.data.store import NewItem
from reef.storage.data.tasks import LocalTaskFileStore
from reef.storage.scenario.record_items import record_item
from reef.storage.sql import data as sql_data
from reef.storage.sql.data import PostgresDataStore, SQLiteDataStore

ATIF = {"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}}


def trajectory(item_id: str, text: str) -> NewItem:
    return NewItem(item_id, {**ATIF, "steps": [{"step_id": 1, "message": text}]}, {"kind": "trajectory"})


def calls(*turns: int) -> list[NewItem]:
    return [record_item(conversation(f"call-{count}", count, scenario="coding")) for count in turns]


@pytest.fixture(params=["sqlite", "postgres"])
def open_store(request, tmp_path):
    """Open the durable data store of the test; ``segments=False`` opens it without a segment store."""
    task_files = LocalTaskFileStore(tmp_path / "task-files")

    def opened(*, segments: bool = True):
        segment_store = LocalSegmentStore(tmp_path / "segments") if segments else None
        if request.param == "postgres":
            url, schema = request.getfixturevalue("postgres_config")
            return PostgresDataStore(url, task_files, schema=schema, segments=segment_store)
        return SQLiteDataStore(tmp_path / "data", task_files, segment_store)

    yield opened
    task_files.close()


@pytest.fixture(autouse=True)
def create_segments(monkeypatch):
    """Each ``create`` call of these tests writes its trajectory items as a segment, whatever their size."""
    monkeypatch.setattr(sql_data, "CREATE_SEGMENT_BYTES", 0)


def waiting_rows(store) -> tuple[int, int]:
    """The trajectory items and the record nodes whose content the database of ``store`` still holds."""
    items, nodes = store.items_table, store.nodes_table
    with store.transaction() as connection:
        return (
            connection.scalar(
                select(func.count()).where(items.c.kind == "trajectory", items.c.content_json.is_not(None))
            ),
            connection.scalar(select(func.count()).where(nodes.c.entries_json.is_not(None))),
        )


def segment_files(tmp_path) -> list:
    return sorted((tmp_path / "segments").glob("*/*.jsonl"))


def test_a_segment_is_json_lines_and_a_reader_reads_only_its_lines(tmp_path):
    store = LocalSegmentStore(tmp_path)
    data, places = segment_lines("s1", ['{"a":1}', '{"b":"二"}'])
    store.write("acme", "s1", data)
    assert [json.loads(line) for line in data.decode().splitlines()] == [{"a": 1}, {"b": "二"}]
    assert store.read("acme", list(reversed(places))) == ('{"b":"二"}'.encode(), b'{"a":1}')
    with pytest.raises(SegmentNotFoundError):
        store.read("other", places)
    with pytest.raises(SegmentNotFoundError):
        store.read("acme", [SegmentLine("s1", places[1].offset, 100)])
    store.delete("acme", "s1")
    store.delete("acme", "s1")
    with pytest.raises(SegmentNotFoundError):
        store.read("acme", places)
    with pytest.raises(ValueError, match="newline"):
        segment_lines("s2", ["{}\n{}"])


def test_new_trajectory_items_go_to_one_segment_and_read_back(open_store, tmp_path):
    with closing(open_store()) as store:
        created = [trajectory("t1", "one"), trajectory("t2", "two")]
        store.create("acme", created)
        assert waiting_rows(store) == (0, 0)
        assert [len(path.read_text().splitlines()) for path in segment_files(tmp_path)] == [2]
        assert store.read("acme", "t2").content == created[1].content
        assert [item.content for item in store.query("acme")] == [item.content for item in created]
        assert [item.content for item in store.query("acme", content=False)] == [None, None]
        # The same content again writes no new segment.
        store.create("acme", [trajectory("t1", "one")])
        assert len(segment_files(tmp_path)) == 1


def test_small_new_trajectory_items_wait_in_the_database_for_one_segment(open_store, monkeypatch, tmp_path):
    created = [trajectory(f"t{number}", str(number)) for number in range(3)]
    size = len(json.dumps(created[0].content, separators=(",", ":")))
    monkeypatch.setattr(sql_data, "CREATE_SEGMENT_BYTES", 2 * size)
    monkeypatch.setattr(sql_data, "PACK_BYTES", 3 * size)
    with closing(open_store()) as store:
        # One item for each call, as a stage stores its samples.
        for item in created[:2]:
            store.create("acme", [item])
        store.maintain()
        assert waiting_rows(store) == (2, 0) and segment_files(tmp_path) == []
        store.create("acme", [created[2]])
        store.maintain()
        assert waiting_rows(store) == (0, 0)
        assert [len(path.read_text().splitlines()) for path in segment_files(tmp_path)] == [3]
        assert [item.content for item in store.query("acme")] == [item.content for item in created]
        # A call with enough content writes its own segment.
        store.create("acme", [trajectory("t3", "3"), trajectory("t4", "4")])
        assert waiting_rows(store) == (0, 0) and len(segment_files(tmp_path)) == 2


def test_a_failed_revision_releases_only_its_unpublished_segments(open_store, monkeypatch, tmp_path):
    old, new = trajectory("old", "first"), trajectory("new", "second")
    with closing(open_store()) as store:
        store.create("acme", [old])
        committed = segment_files(tmp_path)

        def interrupted():
            yield new
            assert len(segment_files(tmp_path)) == 2
            assert [item.item_id for item in store.query("acme")] == ["old"]
            raise RuntimeError("source interrupted")

        with pytest.raises(RuntimeError, match="source interrupted"):
            store.replace_trajectories("acme", interrupted(), ["old"])
        monkeypatch.setattr(sql_data, "UNUSED_SEGMENT_SECONDS", -1)
        store.maintain()
        assert segment_files(tmp_path) == committed
        assert store.read("acme", "old").content == old.content
        assert store.read("acme", "new") is None


def test_a_read_of_many_items_reads_their_segment_lines_together(open_store, monkeypatch):
    with closing(open_store()) as store:
        created = [trajectory(f"t{number}", str(number)) for number in range(5)]
        store.create("acme", created)
        reads = []
        read_text = SegmentPartition.read_text
        monkeypatch.setattr(
            SegmentPartition, "read_text", lambda self, lines: reads.append(lines) or read_text(self, lines)
        )
        found = store.read_many("acme", [item.item_id for item in reversed(created)])
        assert [item.content for item in found] == [item.content for item in reversed(created)]
        assert [len(lines) for lines in reads] == [5]


def test_maintain_moves_waiting_content_into_segments_and_reads_stay_exact(open_store, monkeypatch):
    items = calls(1, 2, 3)
    with closing(open_store(segments=False)) as store:
        created = [*items, trajectory("t1", "kept in the database")]
        store.create("acme", created)
        assert waiting_rows(store) == (1, 3)
    monkeypatch.setattr(sql_data, "PACK_BYTES", 1)
    with closing(open_store()) as store:
        store.maintain()
        assert waiting_rows(store) == (0, 0)
        assert [item.content for item in store.query("acme")] == [item.content for item in created]
        assert store.read("acme", "coding/call-3").content == items[2].content
        # A record that continues a moved node still links to it, and reads back whole.
        later = calls(4)
        read_text = SegmentPartition.read_text

        def refuse(*_):
            raise AssertionError("a create reads no content from a segment, under the scenario's lock")

        monkeypatch.setattr(SegmentPartition, "read_text", refuse)
        store.create("acme", later)
        monkeypatch.setattr(SegmentPartition, "read_text", read_text)
        assert store.read("acme", "coding/call-4").content == later[0].content
        assert store.record_links("acme", ["coding/call-4"])["coding/call-4"].parent == "coding/call-3"


def test_maintain_reads_the_waiting_content_of_postgres_a_few_rows_at_a_time(postgres_config, monkeypatch, tmp_path):
    url, schema = postgres_config
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    items = calls(1, 2, 3)
    with closing(PostgresDataStore(url, task_files, schema=schema)) as store:
        store.create("acme", items)
    monkeypatch.setattr(sql_data, "PACK_BYTES", 1)
    segments = LocalSegmentStore(tmp_path / "segments")
    with closing(PostgresDataStore(url, task_files, schema=schema, segments=segments)) as store:
        content_cursors = []

        @event.listens_for(store.engine, "before_cursor_execute")
        def keep_cursor(connection, cursor, statement, parameters, context, executemany):
            if statement.startswith("SELECT") and "call_json" in statement:
                content_cursors.append(type(cursor).__name__)

        store.maintain()
        # A client-side cursor holds every waiting row of the owner at once, which can be many segments.
        assert content_cursors and set(content_cursors) == {"ServerCursor"}
        assert waiting_rows(store) == (0, 0)
        assert [item.content for item in store.query("acme")] == [item.content for item in items]
    task_files.close()


def test_maintain_merges_the_segments_of_small_writes_and_reads_stay_exact(open_store, monkeypatch, tmp_path):
    created = [trajectory(f"t{number}", str(number)) for number in range(3)]
    size = len(json.dumps(created[0].content, separators=(",", ":")))
    with closing(open_store()) as store:
        # One segment for each item, as a store of an earlier version wrote them.
        for item in created:
            store.create("acme", [item])
        assert [len(path.read_text().splitlines()) for path in segment_files(tmp_path)] == [1, 1, 1]
        monkeypatch.setattr(sql_data, "CREATE_SEGMENT_BYTES", 2**20)
        monkeypatch.setattr(sql_data, "PACK_BYTES", 2 * size)
        store.maintain()
        # The two oldest fill one segment; the third waits for more. The old segments stay for a reader.
        assert sorted(len(path.read_text().splitlines()) for path in segment_files(tmp_path)) == [1, 1, 1, 2]
        assert [item.content for item in store.query("acme")] == [item.content for item in created]
        monkeypatch.setattr(sql_data, "UNUSED_SEGMENT_SECONDS", -1)
        store.maintain()
        assert sorted(len(path.read_text().splitlines()) for path in segment_files(tmp_path)) == [1, 2]
        assert [item.content for item in store.read_many("acme", ["t2", "t0"])] == [
            created[2].content,
            created[0].content,
        ]


def test_maintain_deletes_only_segments_that_no_row_uses(open_store, monkeypatch, tmp_path):
    monkeypatch.setattr(sql_data, "UNUSED_SEGMENT_SECONDS", -1)
    with closing(open_store()) as store:
        store.create("acme", [trajectory("t1", "one")])
        store.create("acme", [trajectory("t2", "two"), trajectory("t3", "three")])
        store.maintain()
        assert len(segment_files(tmp_path)) == 2
        store.delete("acme", "t2")
        store.maintain()
        assert len(segment_files(tmp_path)) == 2
        store.delete("acme", "t1")
        store.maintain()
        # A deleted item keeps its segment until the purge.
        assert len(segment_files(tmp_path)) == 2
        assert store.purge("acme", "t1")
        store.maintain()
        assert len(segment_files(tmp_path)) == 1
        assert [item.item_id for item in store.query("acme")] == ["t3"]
        with store.transaction() as connection:
            assert connection.scalar(select(func.count()).select_from(store.segment_table)) == 1


def test_a_restored_item_reads_its_content_from_the_segment_that_its_delete_kept(open_store, monkeypatch, tmp_path):
    monkeypatch.setattr(sql_data, "UNUSED_SEGMENT_SECONDS", -1)
    with closing(open_store()) as store:
        (created,) = store.create("acme", [trajectory("t1", "one")])
        store.delete("acme", "t1")
        store.maintain()
        assert len(segment_files(tmp_path)) == 1
        store.restore("acme", ["t1"])
        assert store.read("acme", "t1").content == {**ATIF, "steps": [{"step_id": 1, "message": "one"}]}
        assert store.read("acme", "t1").digest == created.digest


def test_a_store_without_segments_refuses_to_guess_moved_content(open_store):
    with closing(open_store()) as store:
        store.create("acme", [trajectory("t1", "one")])
    with closing(open_store(segments=False)) as store:
        assert store.query("acme", content=False)[0].item_id == "t1"
        with pytest.raises(sql_data.ReefError, match="segment"):
            store.read("acme", "t1")


def test_the_deployment_gives_its_segment_store_to_the_data_store(tmp_path):
    settings = ServiceSettings(
        recipe="recipe",
        agent_record_dir=str(tmp_path / "records"),
        artifact_repository=str(tmp_path / "artifacts.git"),
        artifact_work_dir=str(tmp_path / "work"),
        artifact_cache_dir=str(tmp_path / "cache"),
        task_file_dir=str(tmp_path / "files"),
        segment_backend="local",
    )
    dispatcher = build_dispatcher(settings)
    try:
        dispatcher.data_store.create("acme", [trajectory("t1", "one")])
    finally:
        dispatcher.close()
    assert len(list((tmp_path / "records" / "segments").glob("*/*.jsonl"))) == 1
    with pytest.raises(ValueError, match="segment_backend"):
        ServiceSettings(recipe="recipe", segment_backend="unknown")


def test_a_replaced_trajectory_goes_to_a_new_segment_and_maintain_deletes_the_old_one(
    open_store, monkeypatch, tmp_path
):
    monkeypatch.setattr(sql_data, "UNUSED_SEGMENT_SECONDS", -1)
    with closing(open_store()) as store:
        store.create("acme", [trajectory("t1", "a long copy")])
        (old_segment,) = segment_files(tmp_path)
        replaced = {**ATIF, "steps": [{"step_id": 1, "message": "short"}]}
        store.replace_trajectory("acme", "t1", replaced)
        assert store.read("acme", "t1").content == replaced
        assert waiting_rows(store) == (0, 0)
        store.maintain()
        # No row uses the old segment, so maintain deletes it; the new one stays.
        assert old_segment not in segment_files(tmp_path) and len(segment_files(tmp_path)) == 1
        assert store.read("acme", "t1").content == replaced


def test_pack_preserves_a_concurrent_trajectory_replacement(open_store, monkeypatch):
    original = trajectory("t", "A" * 1024)
    replacement = trajectory("t", "B")
    monkeypatch.setattr(sql_data, "CREATE_SEGMENT_BYTES", 8192)
    monkeypatch.setattr(sql_data, "PACK_BYTES", 1024)
    with closing(open_store()) as store:
        store.create("acme", [original])
        write = store.write_segment

        def replace_before_publish(partition, lines):
            places = write(partition, lines)
            store.replace_trajectory("acme", "t", replacement.content)
            return places

        monkeypatch.setattr(store, "write_segment", replace_before_publish)
        assert store.pack("acme") == 0
        assert store.read("acme", "t").content == replacement.content


def test_failed_segment_sync_leaves_no_staged_file(tmp_path, monkeypatch):
    import os

    store = LocalSegmentStore(tmp_path)
    target = store.path("acme", "s")
    target.parent.mkdir()

    def fail_sync(descriptor):
        raise OSError("disk error")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="disk error"):
        store.write("acme", "s", b"payload")
    assert list(target.parent.iterdir()) == []


def test_create_refuses_a_staged_segment_that_cleanup_reclaimed(open_store, monkeypatch):
    with closing(open_store()) as store:
        write = store.write_segment

        def reclaim_before_publish(partition, lines):
            places = write(partition, lines)
            assert store.delete_unused_segments() == 1
            return places

        monkeypatch.setattr(sql_data, "UNUSED_SEGMENT_SECONDS", -1)
        monkeypatch.setattr(store, "write_segment", reclaim_before_publish)
        with pytest.raises(SegmentNotFoundError, match="reclaimed"):
            store.create("acme", [trajectory("t", "A")])
        assert store.read("acme", "t") is None


def test_cleanup_waits_for_a_postgres_reader_of_a_replaced_segment(postgres_config, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event

    url, schema = postgres_config
    files = LocalTaskFileStore(tmp_path / "tasks")
    segments = LocalSegmentStore(tmp_path / "segments")
    with (
        closing(PostgresDataStore(url, files, schema=schema, segments=segments)) as store,
        ThreadPoolExecutor(2) as pool,
    ):
        selected, fetch = Event(), Event()
        read = segments.read

        def paused_read(partition, lines):
            selected.set()
            assert fetch.wait(10)
            return read(partition, lines)

        a, b = trajectory("t", "A"), trajectory("t", "B")
        store.create("acme", [a])
        monkeypatch.setattr(segments, "read", paused_read)
        reader = pool.submit(store.read, "acme", "t")
        try:
            assert selected.wait(10)
            store.replace_trajectory("acme", "t", b.content)
            monkeypatch.setattr(sql_data, "UNUSED_SEGMENT_SECONDS", -1)
            cleanup = pool.submit(store.delete_unused_segments)
            with pytest.raises(TimeoutError):
                cleanup.result(timeout=0.15)
        finally:
            fetch.set()
        assert reader.result(timeout=10).content == a.content
        assert cleanup.result(timeout=10) == 1
        assert store.read("acme", "t").content == b.content
