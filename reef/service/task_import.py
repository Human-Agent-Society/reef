"""The data of each scenario when it opens: its owner, its training samples and the deployment's task import
(RFC #1028, RFC #1140).

:class:`DataScenarioStorage` opens each scenario's session in three steps. A scenario without an owner gets the
owner with its name. Then the deployment's ``task-import`` section, when it is set, adds each task of its split
manifest as an item of the owner that the scenario uses in the manifest's split. Last, the record items of another
scenario that it uses are appended to its record store (``sync_scenario_records``). The session's training samples
are the items of the owner that the scenario uses (``SampleStore``).

An identical import changes nothing, so the import runs again at each open and picks up the tasks that the operator
added to the manifest. A task that conflicts with a stored item stops the open.
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
from reef.storage.commits import CommitRecord
from reef.storage.data import DataStore, NewItem
from reef.storage.data_sync import release_items
from reef.storage.evaluations import EvaluationStore
from reef.storage.record_items import sync_scenario_records
from reef.storage.records import RecordStore
from reef.storage.samples import SampleStore, TrainingState
from reef.storage.scenario import ScenarioSettings, ScenarioStorage, ScenarioStore
from reef.storage.tasks import SplitName, TaskFileStore, TaskStorage


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


class DataScenarioStore(ScenarioStore):
    """A scenario session whose training samples are the items of the data store that its scenario uses.

    Values: ``inner`` is the session of the scenario storage, and ``samples`` the scenario's ``SampleStore``,
    whose training state is ``inner.training_state``. Each operation is the operation of ``inner``, except
    ``samples``.

    Purpose: ``DataScenarioStorage.open`` makes it.
    """

    def __init__(self, inner: ScenarioStore, samples: SampleStore) -> None:
        self.inner = inner
        self._samples = samples

    @property
    def records(self) -> RecordStore:
        return self.inner.records

    @property
    def durable(self) -> bool:
        return self.inner.durable

    @property
    def tasks(self) -> TaskStorage:
        return self.inner.tasks

    @property
    def training_state(self) -> TrainingState:
        return self.inner.training_state

    @property
    def samples(self) -> SampleStore:
        return self._samples

    @property
    def evaluations(self) -> EvaluationStore:
        return self.inner.evaluations

    def history(self) -> tuple[CommitRecord, ...]:
        return self.inner.history()

    def training_run_position(self) -> tuple[int, int]:
        return self.inner.training_run_position()

    def commit_step(self, *, expected_step: int, commit: CommitRecord) -> CommitRecord:
        return self.inner.commit_step(expected_step=expected_step, commit=commit)

    def recover(self, *, checkpoint: CommitRecord | None) -> CommitRecord | None:
        return self.inner.recover(checkpoint=checkpoint)

    def close(self) -> None:
        self.inner.close()


class DataScenarioStorage(ScenarioStorage):
    """Scenario storage whose sessions read their training samples from the deployment's data store.

    Values: ``(inner, data, task_import)`` is such a storage if and only if
    ``inner`` is a scenario storage, ``data`` is the deployment's data store,
    and ``task_import`` is a task import or ``None``. Each operation is the
    operation of ``inner``, except ``open``, which opens the session of
    ``inner`` and then, in order:

    1. sets the owner of a scenario that has none to the scenario's name;
    2. runs ``task_import`` for the owner, when it is set;
    3. appends to its record store the copy of each record item of the owner
       that it uses and that another scenario made (``sync_scenario_records``);
    4. returns the session as a ``DataScenarioStore``, whose samples are the
       items of the owner that the scenario uses.

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
                self.inner.write_settings(scenario, replace(settings, owner=owner))
            if self.task_import is not None:
                self.task_import.run(self.data, owner, scenario)
            sync_scenario_records(session.records, self.data, owner, scenario)
        except BaseException:
            session.close()
            raise
        return DataScenarioStore(session, SampleStore(self.data, owner, scenario, session.training_state))

    def archive(self, scenario: str) -> tuple[str, ...]:
        owner = self.inner.read_settings(scenario).owner
        if owner is not None:
            release_items(self.data, owner, scenario)
        return self.inner.archive(scenario)

    def prune(self, *, max_bytes: int) -> int:
        return self.inner.prune(max_bytes=max_bytes)

    def close(self) -> None:
        self.inner.close()
