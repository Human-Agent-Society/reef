"""The import of a deployment's Harbor tasks into each scenario's task store."""

from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.task_service_stubs import task_for

from reef.core.tasks import write_harbor_task
from reef.core.tasks.split import TaskSplit, TaskSplitError, write_split_manifest
from reef.service.assembly import build_dispatcher
from reef.service.deploy.deployment_config import translate_layout
from reef.service.deploy.service_config import ServiceConfig, normalize_service_config, service_config_from_mapping
from reef.service.task_import import TaskImport, TaskImportError, TaskImportingScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import IMPORTED_TASK_STORE, TaskConflict

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


def imported(storage: TaskImportingScenarioStorage) -> list[tuple[str, str]]:
    with closing(storage.open("math")) as session:
        return [(task.name, task.split) for task in session.tasks.open(IMPORTED_TASK_STORE).tasks()]


def test_each_open_imports_the_manifest_into_imported_tasks(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    for port, name in enumerate(("b-task", "a-task", "c-task")):
        harbor_task(root, name, 9000 + port)
    path = manifest(tmp_path / "split.json", ("b-task", "a-task"), ("c-task",))
    task_import = TaskImport(path, root)
    with closing(TaskImportingScenarioStorage(SQLiteScenarioStorage(tmp_path / "records"), task_import)) as storage:
        assert imported(storage) == [("a-task", "train"), ("b-task", "train"), ("c-task", "eval")]
        with closing(storage.open("math")) as session:
            store = session.tasks.open(IMPORTED_TASK_STORE)
            (evaluated,) = store.tasks(split="eval")
            assert (store.path(evaluated.files_digest) / "environment" / "image.png").read_bytes() == IMAGE

        # The next open imports what the operator added and leaves the published tasks as they are.
        harbor_task(root, "d-task", 9010)
        manifest(path, ("b-task", "a-task"), ("c-task", "d-task"))
        assert imported(storage)[-1] == ("d-task", "eval") and len(imported(storage)) == 4

    with closing(TaskImportingScenarioStorage(SQLiteScenarioStorage(tmp_path / "records"), task_import)) as storage:
        assert len(imported(storage)) == 4, "a restart imports nothing new"


def test_a_task_that_changed_or_cannot_load_stops_the_open(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    harbor_task(root, "a-task", 9000)
    path = manifest(tmp_path / "split.json", ("a-task",), ())
    storage = TaskImportingScenarioStorage(SQLiteScenarioStorage(tmp_path / "records"), TaskImport(path, root))
    with closing(storage):
        assert imported(storage) == [("a-task", "train")]

        instruction = root / "a-task" / "instruction.md"
        original = instruction.read_text()
        instruction.write_text("Solve another problem.\n")
        with pytest.raises(TaskConflict, match="other files"):
            storage.open("math")
        instruction.write_text(original)
        manifest(path, (), ("a-task",))
        with pytest.raises(TaskConflict, match="published to train, not eval"):
            storage.open("math")

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
    settings = replace(
        settings,
        recipe="recipe",
        artifact_repository=str(tmp_path / "artifacts.git"),
        artifact_work_dir=str(tmp_path / "work"),
        artifact_cache_dir=str(tmp_path / "cache"),
        task_file_dir=str(tmp_path / "files"),
    )
    dispatcher = build_dispatcher(settings)
    try:
        store = dispatcher.get_or_create_scenario("math").store.tasks.open(IMPORTED_TASK_STORE)
        assert [(task.name, task.split) for task in store.tasks()] == [("a-task", "eval")]
    finally:
        dispatcher.close()

    with pytest.raises(ValueError, match="task-import needs both manifest and tasks-root"):
        ServiceConfig(recipe="recipe", task_import_manifest=str(path))
