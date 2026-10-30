"""The data of each scenario when it opens: the data sync and the deployment's task import (RFC #1028).

:class:`DataScenarioStorage` opens each scenario's session in three steps.
A scenario that the data sync has not opened before has no owner: its
uploads become items of the owner with the scenario's name, and that name
becomes its owner. Then the deployment's ``task-import`` section, when it
is set, adds each task of its split manifest as an item of the owner that
the scenario uses in the manifest's split. Last, the data sync makes the
scenario's uploads match its items (:mod:`reef.storage.data_sync`).

An identical import changes nothing, so the import runs again at each open
and picks up the tasks that the operator added to the manifest. A task that
conflicts with a stored item or sample stops the open.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from harbor.models.task.task import Task as HarborTaskLoader

from reef.core.errors import ReefError
from reef.core.plugins import PluginConfigError
from reef.core.tasks.split import read_split_manifest
from reef.storage.data import DataStore, NewItem
from reef.storage.data_sync import adopt_uploads, release_items, sync_scenario_samples
from reef.storage.record_items import sync_scenario_records
from reef.storage.scenario import ScenarioSettings, ScenarioStorage, ScenarioStore
from reef.storage.tasks import SplitName, TaskFileStore


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
    :class:`DataScenarioStorage` runs it when a scenario opens.
    """

    manifest: Path
    tasks_root: Path

    def run(self, data: DataStore, owner: str, scenario: str) -> None:
        """Save each task of the manifest as a task item of ``owner`` that ``scenario`` uses in its split.

        The train tasks come first, then the eval tasks, each in name order.
        Raises ``TaskSplitError`` if the manifest cannot be read,
        ``TaskImportError`` if a task directory is missing or is not a Harbor
        task, and ``ItemConflict`` if ``owner`` stores a task of the same name
        with other files.
        """
        manifest = read_split_manifest(self.manifest)
        splits: tuple[tuple[SplitName, tuple[str, ...]], ...] = (("train", manifest.train), ("eval", manifest.eval))
        for split, names in splits:
            directories = tuple(self.tasks_root / name for name in names)
            for directory in directories:
                try:
                    HarborTaskLoader(directory)
                except (ValueError, OSError) as exc:
                    raise TaskImportError(f"{directory} is not a Harbor task: {exc}") from exc
            data.create(owner, [NewItem(directory.name, directory, {"kind": "task"}) for directory in directories])
            if directories:
                data.update(owner, [directory.name for directory in directories], {"scenarios": {scenario: split}})


class DataScenarioStorage(ScenarioStorage):
    """Scenario storage whose sessions hold copies of their scenario's data items when they open.

    Values: ``(inner, data, task_import)`` is such a storage if and only if
    ``inner`` is a scenario storage, ``data`` is the deployment's data store,
    and ``task_import`` is a task import or ``None``. Each operation is the
    operation of ``inner``, except ``open``, which opens the session of
    ``inner`` and then, in order:

    1. sets the owner of a scenario that has none to the scenario's name,
       after its uploads become items of that owner (``adopt_uploads``);
    2. runs ``task_import`` for the owner, when it is set;
    3. syncs the session's uploads with the owner's items
       (``sync_scenario_samples``), and appends to its record store the copy
       of each record item of the owner that it uses and that another
       scenario made (``sync_scenario_records``).

    If a step raises, ``open`` closes the session and raises. ``archive`` also
    removes the scenario from the ``scenarios`` of its owner's items, except
    the record items that it made, before the archive of ``inner``.

    Purpose: deployment assembly wraps the scenario storage in it. The
    scenario factory opens each scenario's session through it.
    """

    def __init__(self, inner: ScenarioStorage, data: DataStore, task_import: TaskImport | None = None) -> None:
        self.inner = inner
        self.data = data
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
            settings = self.inner.read_settings(scenario)
            owner = settings.owner
            if owner is None:
                owner = scenario
                adopt_uploads(session.samples, self.data, owner, scenario)
                self.inner.write_settings(scenario, replace(settings, owner=owner))
            if self.task_import is not None:
                self.task_import.run(self.data, owner, scenario)
            sync_scenario_samples(session.samples, self.data, owner, scenario)
            sync_scenario_records(session.records, self.data, owner, scenario)
        except BaseException:
            session.close()
            raise
        return session

    def archive(self, scenario: str) -> tuple[str, ...]:
        owner = self.inner.read_settings(scenario).owner
        if owner is not None:
            release_items(self.data, owner, scenario)
        return self.inner.archive(scenario)

    def prune(self, *, max_bytes: int) -> int:
        return self.inner.prune(max_bytes=max_bytes)

    def close(self) -> None:
        self.inner.close()
