from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from reef.core.batches import TaskItem, TrajectoryItem
from reef.storage.data_items import DataItemConflict
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import TaskStoreError


def trajectory(text: str) -> TrajectoryItem:
    return TrajectoryItem(
        {"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1, "message": text}]}
    )


def task_directory(root: Path, instruction: str = "Print hello.") -> Path:
    task = root / "demo"
    (task / "tests").mkdir(parents=True)
    (task / "instruction.md").write_text(instruction)
    (task / "tests" / "test.sh").write_bytes(b"exit 0\n\x00")
    return task


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        opened = PostgresScenarioStorage(url, tmp_path, schema=schema)
    else:
        opened = SQLiteScenarioStorage(None if request.param == "memory" else tmp_path)
    with closing(opened):
        yield opened


@pytest.fixture(params=["sqlite", "postgres"])
def open_durable_storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        return lambda: PostgresScenarioStorage(url, tmp_path / "files", schema=schema)
    return lambda: SQLiteScenarioStorage(tmp_path / "files")


def test_data_items_keep_upload_order_and_their_task_files(storage, tmp_path):
    source = task_directory(tmp_path / "upload")
    task = TaskItem(source, metadata={"origin": "fixture"})
    with closing(storage.open("math")) as session, closing(storage.open("other")) as other:
        assert session.data_items.add((("t1", trajectory("a")), ("k1", task), ("t2", trajectory("b"))))
        stored = session.data_items.page()
        assert [item.item_id for item in stored] == ["t1", "k1", "t2"]
        assert stored[0].sequence < stored[1].sequence < stored[2].sequence
        assert stored[0].item == trajectory("a") and stored[2].item == trajectory("b")
        stored_task = stored[1].item
        assert isinstance(stored_task, TaskItem) and stored_task.metadata == {"origin": "fixture"}
        assert stored_task.task_path != source and stored_task.task_path.name == "demo"
        assert (stored_task.task_path / "tests" / "test.sh").read_bytes() == b"exit 0\n\x00"
        assert [item.item_id for item in session.data_items.page(after_sequence=stored[0].sequence, limit=1)] == ["k1"]
        assert session.data_items.count() == 3
        assert session.data_items.count(after_sequence=stored[2].sequence) == 0
        assert other.data_items.count() == 0


def test_a_retry_changes_nothing_and_a_conflict_saves_no_item(storage, tmp_path):
    source = task_directory(tmp_path / "upload")
    changed = task_directory(tmp_path / "changed", instruction="Print goodbye.")
    with closing(storage.open("math")) as session:
        assert session.data_items.add((("t1", trajectory("a")), ("k1", TaskItem(source))))
        assert not session.data_items.add((("t1", trajectory("a")), ("k1", TaskItem(source))))
        for conflict in ((("t2", trajectory("c")), ("t1", trajectory("b"))), (("k1", TaskItem(changed)),)):
            with pytest.raises(DataItemConflict, match="different content"):
                session.data_items.add(conflict)
        assert [item.item_id for item in session.data_items.page()] == ["t1", "k1"]


def test_data_items_survive_a_restart_and_archive_retires_them(open_durable_storage, tmp_path):
    source = task_directory(tmp_path / "upload")
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        session.data_items.add((("t1", trajectory("a")), ("k1", TaskItem(source))))

    with closing(open_durable_storage()) as storage:
        with closing(storage.open("math")) as session:
            stored = session.data_items.page()
            assert [item.item_id for item in stored] == ["t1", "k1"]
            task_path = stored[1].item.task_path
            assert (task_path / "instruction.md").read_text() == "Print hello."
        storage.archive("math")
        with closing(storage.open("math")) as session:
            assert session.data_items.count() == 0
            assert not task_path.exists()


def test_a_task_directory_with_a_link_is_refused(storage, tmp_path):
    source = task_directory(tmp_path / "upload")
    (source / "link").symlink_to(source / "instruction.md")
    with closing(storage.open("math")) as session:
        with pytest.raises(TaskStoreError, match="not a file"):
            session.data_items.add((("k1", TaskItem(source)),))
        assert session.data_items.count() == 0
