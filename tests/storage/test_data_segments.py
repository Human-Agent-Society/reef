"""The data store with a segment store: content of trajectory items and record nodes in JSON Lines segments."""

from __future__ import annotations

import json
from contextlib import closing

import pytest
from sqlalchemy import func, select
from tests.support.conversations import conversation

from reef.service.assembly import build_dispatcher
from reef.service.deploy.service_config import ServiceConfig
from reef.storage import sql_data
from reef.storage.data import NewItem
from reef.storage.record_items import record_item
from reef.storage.segments import LocalSegmentStore, SegmentLine, SegmentNotFound, SegmentPartition, segment_lines
from reef.storage.sql_data import PostgresDataStore, SQLiteDataStore
from reef.storage.tasks import LocalTaskFileStore

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
    store.put("acme", "s1", data)
    assert [json.loads(line) for line in data.decode().splitlines()] == [{"a": 1}, {"b": "二"}]
    assert store.read("acme", list(reversed(places))) == ('{"b":"二"}'.encode(), b'{"a":1}')
    with pytest.raises(SegmentNotFound):
        store.read("other", places)
    with pytest.raises(SegmentNotFound):
        store.read("acme", [SegmentLine("s1", places[1].offset, 100)])
    store.delete("acme", "s1")
    store.delete("acme", "s1")
    with pytest.raises(SegmentNotFound):
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
        assert len(segment_files(tmp_path)) == 1
        assert [item.item_id for item in store.query("acme")] == ["t3"]
        with store.transaction() as connection:
            assert connection.scalar(select(func.count()).select_from(store.segment_table)) == 1


def test_a_store_without_segments_refuses_to_guess_moved_content(open_store):
    with closing(open_store()) as store:
        store.create("acme", [trajectory("t1", "one")])
    with closing(open_store(segments=False)) as store:
        assert store.query("acme", content=False)[0].item_id == "t1"
        with pytest.raises(sql_data.ReefError, match="segment"):
            store.read("acme", "t1")


def test_an_upgrade_adds_the_segment_columns(open_store):
    with closing(open_store(segments=False)) as store:
        store.create("acme", [*calls(1, 2), trajectory("t1", "one")])
        with store.transaction() as connection:
            # Remove the columns, as the earlier version kept the tables.
            for table in (store.items_table, store.nodes_table):
                index = next(index for index in table.indexes if "segment_id" in index.columns)
                index.drop(connection)
                for name in ("segment_id", "segment_offset", "segment_length"):
                    connection.exec_driver_sql(f"ALTER TABLE {table.fullname} DROP COLUMN {name}")
    with closing(open_store()) as store:
        assert store.read("acme", "coding/call-2").content == calls(2)[0].content
        store.create("acme", [trajectory("t2", "two")])
        assert store.read("acme", "t2").content["steps"][0]["message"] == "two"


def test_the_deployment_gives_its_segment_store_to_the_data_store(tmp_path):
    settings = ServiceConfig(
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
        ServiceConfig(recipe="recipe", segment_backend="unknown")
