"""The import of a deployment's Harbor tasks as task items that each scenario uses (RFC #1028)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.task_service_stubs import task_for

from reef.core.job_data import TaskSample
from reef.core.tasks import write_harbor_task
from reef.core.tasks.split import TaskSplit, TaskSplitError, write_split_manifest
from reef.service.assembly import build_dispatcher
from reef.service.deploy.deployment_config import translate_layout
from reef.service.deploy.service_config import ServiceConfig, normalize_service_config, service_config_from_mapping
from reef.service.task_import import DataScenarioStorage, TaskImport, TaskImportError
from reef.storage.data import ItemConflict
from reef.storage.sql_data import SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore

IMAGE = bytes(range(256))


def harbor_task(root: Path, name: str, port: int, *, instruction: str = "") -> Path:
    """A Harbor task directory with a binary file, which Reef's own task reader would refuse."""
    task = task_for(name, "record-1", port)
    directory = write_harbor_task(replace(task, instruction=instruction) if instruction else task, root)
    (directory / "environment" / "image.png").write_bytes(IMAGE)
    return directory


def manifest(path: Path, train: tuple[str, ...], eval_: tuple[str, ...]) -> Path:
    write_split_manifest(path, TaskSplit(train=train, eval=eval_, seed=0, eval_fraction=0.5))
    return path


@contextmanager
def data_storage(tmp_path: Path, task_import: TaskImport | None) -> Iterator[DataScenarioStorage]:
    """Scenario storage with the data sync over SQLite stores; the scenario storage owns the task files."""
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    data = SQLiteDataStore(tmp_path / "data", task_files)
    storage = DataScenarioStorage(
        SQLiteScenarioStorage(tmp_path / "records", task_files=task_files), data, task_import
    )
    try:
        yield storage
    finally:
        data.close()
        storage.close()


def imported(storage: DataScenarioStorage) -> list[tuple[str, str]]:
    with closing(storage.open("math")) as session:
        return [(task.sample_id, task.split) for task in session.samples.imported_tasks()]


def test_each_open_imports_the_manifest_into_the_sample_store(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    for port, name in enumerate(("b-task", "a-task", "c-task")):
        harbor_task(root, name, 9000 + port)
    path = manifest(tmp_path / "split.json", ("b-task", "a-task"), ("c-task",))
    task_import = TaskImport(path, root)
    with data_storage(tmp_path, task_import) as storage:
        assert imported(storage) == [("a-task", "train"), ("b-task", "train"), ("c-task", "eval")]
        items = storage.data.query("math", kind="task")
        assert [(item.item_id, item.scenarios) for item in items] == [
            ("a-task", {"math": "train"}),
            ("b-task", {"math": "train"}),
            ("c-task", {"math": "eval"}),
        ]
        with closing(storage.open("math")) as session:
            (evaluated,) = session.samples.imported_tasks(split="eval")
            assert isinstance(evaluated.sample, TaskSample)
            assert (evaluated.sample.task_path / "environment" / "image.png").read_bytes() == IMAGE

        # The next open imports what the operator added and leaves the published tasks as they are.
        harbor_task(root, "d-task", 9010)
        manifest(path, ("b-task", "a-task"), ("c-task", "d-task"))
        assert imported(storage)[-1] == ("d-task", "eval") and len(imported(storage)) == 4

    with data_storage(tmp_path, task_import) as storage:
        assert len(imported(storage)) == 4, "a restart imports nothing new"


def test_a_task_that_changed_or_cannot_load_stops_the_open(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    harbor_task(root, "a-task", 9000)
    path = manifest(tmp_path / "split.json", ("a-task",), ())
    with data_storage(tmp_path, TaskImport(path, root)) as storage:
        assert imported(storage) == [("a-task", "train")]

        instruction = root / "a-task" / "instruction.md"
        original = instruction.read_text()
        instruction.write_text("Solve another problem.\n")
        with pytest.raises(ItemConflict, match="other content"):
            storage.open("math")
        instruction.write_text(original)
        # The split is the item's metadata, so a task that the manifest moves moves in the scenario.
        manifest(path, (), ("a-task",))
        assert imported(storage) == [("a-task", "eval")]

        manifest(path, ("a-task", "missing-task"), ())
        with pytest.raises(TaskImportError, match="missing-task is not a Harbor task"):
            storage.open("math")

        path.write_text("{}")
        with pytest.raises(TaskSplitError, match="split manifest"):
            storage.open("math")


def test_the_deployment_imports_its_tasks_into_each_scenario(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    harbor_task(root, "a-task", 9000)
    path = manifest(tmp_path / "split.json", (), ("a-task",))
    config = translate_layout(
        {
            "storage": {"agent-record-dir": str(tmp_path / "records")},
            "task-import": {"manifest": str(path), "tasks-root": str(root)},
        }
    )
    settings = service_config_from_mapping(normalize_service_config(config))
    assert (settings.task_import_manifest, settings.task_import_tasks_root) == (str(path), str(root))
    # An in-memory artifact backend: this test checks the task import, not the Git LFS repository.
    settings = replace(
        settings,
        recipe="recipe",
        artifact_backend={
            "module": "tests.support.artifact_backend_plugin:MemoryArtifactBackend",
            "config": {"label": "tasks"},
        },
        artifact_work_dir=str(tmp_path / "work"),
        artifact_cache_dir=str(tmp_path / "cache"),
        task_file_dir=str(tmp_path / "files"),
    )
    dispatcher = build_dispatcher(settings)
    try:
        samples = dispatcher.get_or_create_scenario("math").store.samples
        assert [(task.sample_id, task.split) for task in samples.imported_tasks()] == [("a-task", "eval")]
    finally:
        dispatcher.close()

    with pytest.raises(ValueError, match="task-import needs both manifest and tasks-root"):
        ServiceConfig(recipe="recipe", task_import_manifest=str(path))
