"""The import of a deployment's Harbor tasks into each scenario's task store.

A deployment names a split manifest and the directory of its tasks in its
``task-import`` section. When a scenario opens, :class:`TaskImportingScenarioStorage`
adds each task of the manifest to the scenario's task store ``imported-tasks``,
in the split that the manifest gives. An identical import changes nothing, so
the import runs again at each open and picks up the tasks that the operator
added to the manifest. A task that conflicts with a published one stops the open.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from harbor.models.task.task import Task as HarborTaskLoader

from reef.core.errors import ReefError
from reef.core.plugins import PluginConfigError
from reef.core.tasks.split import read_split_manifest
from reef.storage.scenario import ScenarioSettings, ScenarioStorage, ScenarioStore
from reef.storage.tasks import IMPORTED_TASK_STORE, SplitName, TaskFileStore, TaskStorage


class TaskImportError(ReefError):
    """A task of the manifest is missing, or Harbor cannot load it."""


@dataclass(frozen=True)
class TaskImport:
    """The tasks that a deployment imports into each scenario.

    Values: ``(manifest, tasks_root)`` is a task import if and only if
    ``manifest`` is the path of a split manifest (:mod:`reef.core.tasks.split`)
    and ``tasks_root`` is the directory that holds a Harbor task directory for
    each task name of the manifest. The files can change between imports.

    Purpose: deployment assembly makes it from the ``task-import`` section.
    :class:`TaskImportingScenarioStorage` runs it when a scenario opens.
    """

    manifest: Path
    tasks_root: Path

    def run(self, tasks: TaskStorage) -> None:
        """Publish each task of the manifest to its split in the store ``imported-tasks``.

        The train tasks come first, then the eval tasks, each in name order.
        Raises ``TaskSplitError`` if the manifest cannot be read,
        ``TaskImportError`` if a task directory is missing or is not a Harbor
        task, and ``TaskConflict`` if a task is published with other files or
        in the other split.
        """
        manifest = read_split_manifest(self.manifest)
        store = tasks.open(IMPORTED_TASK_STORE)
        splits: tuple[tuple[SplitName, tuple[str, ...]], ...] = (("train", manifest.train), ("eval", manifest.eval))
        for split, names in splits:
            for name in names:
                directory = self.tasks_root / name
                try:
                    HarborTaskLoader(directory)
                except (ValueError, OSError) as exc:
                    raise TaskImportError(f"{directory} is not a Harbor task: {exc}") from exc
                store.add_task(directory=directory, split=split)


class TaskImportingScenarioStorage(ScenarioStorage):
    """Scenario storage whose sessions hold the deployment's imported tasks when they open.

    Values: ``(inner, task_import)`` is such a storage if and only if ``inner``
    is a scenario storage and ``task_import`` is a task import. Each operation
    is the operation of ``inner``, except ``open``: it opens the session of
    ``inner``, runs ``task_import`` on its task storage, and returns the
    session. If the import raises, ``open`` closes the session and raises.

    Purpose: deployment assembly wraps the scenario storage in it when the
    deployment has a ``task-import`` section. The scenario factory opens each
    scenario's session through it.
    """

    def __init__(self, inner: ScenarioStorage, task_import: TaskImport) -> None:
        self.inner = inner
        self.task_import = task_import

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore) -> Self:
        raise PluginConfigError(f"{cls.__name__} wraps another storage; deployment assembly applies it")

    @property
    def durable(self) -> bool:
        return self.inner.durable

    def read_settings(self, scenario: str) -> ScenarioSettings:
        return self.inner.read_settings(scenario)

    def write_settings(self, scenario: str, settings: ScenarioSettings) -> None:
        self.inner.write_settings(scenario, settings)

    def open(self, scenario: str) -> ScenarioStore:
        session = self.inner.open(scenario)
        try:
            self.task_import.run(session.tasks)
        except BaseException:
            session.close()
            raise
        return session

    def archive(self, scenario: str) -> tuple[str, ...]:
        return self.inner.archive(scenario)

    def prune(self, *, max_bytes: int) -> int:
        return self.inner.prune(max_bytes=max_bytes)

    def close(self) -> None:
        self.inner.close()
