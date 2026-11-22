"""The ``task-import`` section of a deployment: its settings and the import into each scenario (RFC #1028)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.task_service_stubs import task_for

from reef.core.tasks import write_harbor_task
from reef.core.tasks.split import TaskSplit, write_split_manifest
from reef.service.assembly import build_dispatcher
from reef.service.deploy.deployment_config import translate_layout
from reef.service.deploy.service_config import ServiceSettings, normalize_service_config, service_config_from_mapping


def test_the_deployment_imports_its_tasks_into_each_scenario(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    write_harbor_task(task_for("a-task", "record-1", 9000), root)
    path = tmp_path / "split.json"
    write_split_manifest(path, TaskSplit(train=(), eval=("a-task",), seed=0, eval_fraction=0.5))
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
        samples = dispatcher.storage.sample_store("math", dispatcher.get_or_create_scenario("math").store)
        assert [(task.sample_id, task.split) for task in samples.imported_tasks()] == [("a-task", "eval")]
    finally:
        dispatcher.close()

    with pytest.raises(ValueError, match="task-import needs both manifest and tasks-root"):
        ServiceSettings(recipe="recipe", task_import_manifest=str(path))
