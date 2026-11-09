"""Storage boundary for a scenario's records, commits, settings, tasks, training state, evaluations, and recovery.

Artifact publication remains the responsibility of the committer. A
store settles record consumption and the committed scenario state together;
implementations may use a database transaction or a recoverable journal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from reef.core.errors import ReefError
from reef.core.plugins import Plugin
from reef.core.serving_config import ServingConfig
from reef.storage.commits import CommitRecord, CommitSnapshots
from reef.storage.data import DataStore, check_owner
from reef.storage.evaluations import EvaluationStore
from reef.storage.records import RecordStore
from reef.storage.samples import SampleStore, TrainingState
from reef.storage.tasks import TaskFileStore, TaskStorage


class ScenarioStoreConflict(ReefError):
    """A commit's expected step or content conflicts with the stored history."""


@dataclass(frozen=True)
class ScenarioSettings:
    """The settings of one scenario, kept with its records and commits.

    ``training_mode`` is the selected mode (``None`` selects the recipe's
    default). ``artifact_evaluation`` is the
    ``ArtifactEvaluationSettings`` as JSON that Reef evaluates each new
    artifact with (``None`` evaluates none). ``activation`` is the scenario's
    own activation rule as JSON: ``{"module": <built-in rule>, "config": {...}}``.
    ``None`` follows the deployment's ``activation`` section. The scenario
    registry checks the rule before it stores it. ``owner`` is the data store
    owner whose items the scenario uses (RFC #1028), a name that passes
    ``check_owner``. It is ``None`` only for a scenario that the data sync has
    not opened yet; the first open sets it. ``serving_config`` is the serving fields that the scenario sets over
    the deployment's values (RFC #1097); the scenario registry checks it with the runtime before it stores it.
    Construction checks these types.

    Purpose: the scenario registry and the dispatcher write the settings
    through ``ScenarioStorage.write_settings`` and read them back when a
    scenario loads.
    """

    training_mode: str | None = None
    artifact_evaluation: Mapping[str, object] | None = None
    activation: Mapping[str, object] | None = None
    owner: str | None = None
    serving_config: ServingConfig = field(default_factory=ServingConfig)

    def __post_init__(self) -> None:
        if self.training_mode is not None and not isinstance(self.training_mode, str):
            raise ValueError("scenario settings training_mode must be a string or null")
        if self.artifact_evaluation is not None and not isinstance(self.artifact_evaluation, Mapping):
            raise ValueError("scenario settings artifact_evaluation must be an object or null")
        if self.activation is not None and not isinstance(self.activation, Mapping):
            raise ValueError("scenario settings activation must be an object or null")
        if self.owner is not None:
            check_owner(self.owner)
        if not isinstance(self.serving_config, ServingConfig):
            raise ValueError("scenario settings serving_config must be a ServingConfig")


class ScenarioStore(ABC):
    """One scenario's storage session, including its record connection.

    The caller closes the session when the scenario is unloaded. Closing is
    idempotent; subsequent reads and writes fail. Records from other scenarios
    must never be committed through this session.
    """

    @property
    @abstractmethod
    def records(self) -> RecordStore:
        """The records connection owned by this session."""

    @property
    @abstractmethod
    def durable(self) -> bool:
        """Whether committed state survives a process restart."""

    @property
    @abstractmethod
    def tasks(self) -> TaskStorage:
        """Opens the generated task stores of this scenario.

        Each record-to-sample stage opens its own store name. The stores close
        with this session. They survive a restart if ``durable`` is true.
        """

    @property
    @abstractmethod
    def training_state(self) -> TrainingState:
        """The training state of this scenario: the train counts of its samples and the records that its stages
        consumed. It closes with this session, and survives a restart if ``durable`` is true."""

    @property
    @abstractmethod
    def evaluations(self) -> EvaluationStore:
        """The artifact evaluations that clients asked this scenario for.

        The store closes with this session. Its evaluations survive a restart if ``durable`` is true.
        """

    def use_snapshots(self, snapshots: CommitSnapshots) -> None:
        """Locate the snapshot of each record that ``history`` reads. A store that keeps its records in memory holds
        their snapshots, so it does nothing."""
        return

    @abstractmethod
    def history(self) -> tuple[CommitRecord, ...]:
        """Read committed steps in increasing order: training steps, activations and new training starts."""

    @abstractmethod
    def training_run_position(self) -> tuple[int, int]:
        """Return the step of the last new training start and the training count after it."""

    @abstractmethod
    def commit_step(self, *, expected_step: int, commit: CommitRecord) -> CommitRecord:
        """Persist exactly ``expected_step + 1`` with its consumption progress.

        Conflicting writers are serialized across sessions. An identical retry
        (ignoring only ``recorded_at``) returns the original record, even after
        later steps. Different content or
        a stale expected step raises ``ScenarioStoreConflict``.

        An exception can occur after the commit becomes durable. Callers
        inspect history or retry the same commit before repeating training.
        """

    @abstractmethod
    def recover(self, *, checkpoint: CommitRecord | None) -> CommitRecord | None:
        """Reconcile a checkpoint commit with history.

        ``None`` denotes initial registration before any committed step.
        A checkpoint ahead of history is adopted durably.
        Validate scenario identity and continuity after the checkpoint. Return
        the committed head, or ``None`` at creation, without publishing artifacts.
        """

    @abstractmethod
    def close(self) -> None:
        """Release this session's records and other resources exactly once."""


class ScenarioStorage(Plugin, ABC):
    """Open sessions and maintain their storage without exposing file paths.

    The owner serializes archive and retention with lifecycle changes and
    closes a scenario's sessions before archiving it. Closing storage releases
    shared resources only; sessions remain owned by their scenarios.

    Every piece of per-scenario state that Reef owns goes through this
    interface: records, commits, generated tasks, training state and evaluations through sessions, settings
    through ``read_settings`` and ``write_settings``. Archiving a scenario
    retires all of it. A deployment names a subclass in ``storage.record_backend``.
    """

    @classmethod
    @abstractmethod
    def from_config(
        cls, config: Mapping[str, Any], *, directory: Path, task_files: TaskFileStore, data: DataStore | None = None
    ) -> Self:
        """Build the storage a deployment names.

        ``config`` is the plugin's own configuration. ``directory`` is the
        deployment's ``storage.agent_record_dir``, which the storage may use
        for local files. ``task_files`` keeps the files of published tasks.
        The storage owns it: ``archive`` retires the files of a scenario, and
        ``close`` closes it. ``data`` is the deployment's data store, which keeps
        each scenario's records as record items (RFC #1219); the storage shares
        it and does not close it.
        """

    @property
    @abstractmethod
    def durable(self) -> bool:
        """Whether newly opened sessions retain commits and settings across restarts."""

    @abstractmethod
    def read_settings(self, scenario: str) -> ScenarioSettings:
        """Read the scenario's stored settings; a scenario with none reads as defaults."""

    @abstractmethod
    def write_settings(self, scenario: str, settings: ScenarioSettings) -> None:
        """Replace the scenario's settings durably; the caller serializes writers of one scenario."""

    @abstractmethod
    def open(self, scenario: str) -> ScenarioStore:
        """Open an independently owned session for this scenario."""

    def sample_store(self, scenario: str, session: ScenarioStore) -> SampleStore | None:
        """The data items that ``scenario`` trains on, with the training state of its open ``session`` (RFC #1140).
        A storage without a data store has none: its trainers keep their samples in memory."""
        return None

    @abstractmethod
    def archive(self, scenario: str) -> tuple[str, ...]:
        """Archive a closed scenario's records, commits, tasks and settings; return archive locations."""

    @abstractmethod
    def close(self) -> None:
        """Release shared storage resources; repeated closes are harmless."""


__all__ = ["ScenarioSettings", "ScenarioStorage", "ScenarioStore", "ScenarioStoreConflict"]
