"""Storage boundary for a scenario's records, commits, settings, and recovery.

Artifact publication remains the responsibility of the committer. A
store settles record consumption and the committed scenario state together;
implementations may use a database transaction or a recoverable journal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.core.errors import ReefError
from reef.core.plugins import Plugin
from reef.storage.commits import CommitRecord
from reef.storage.records import RecordStore


class ScenarioStoreConflict(ReefError):
    """A commit's expected step or content conflicts with the stored history."""


@dataclass(frozen=True)
class ScenarioSettings:
    """The settings a person selects for one scenario, kept with its records and commits.

    ``model_config`` is the validated model override as JSON (``None`` selects
    the recipe's model). ``training_mode`` is the selected mode (``None``
    selects the recipe's default).
    """

    model_config: Mapping[str, object] | None = None
    training_mode: str | None = None

    def __post_init__(self) -> None:
        if self.model_config is not None and not isinstance(self.model_config, Mapping):
            raise ValueError("scenario settings model_config must be an object or null")
        if self.training_mode is not None and not isinstance(self.training_mode, str):
            raise ValueError("scenario settings training_mode must be a string or null")


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

    @abstractmethod
    def history(self) -> tuple[CommitRecord, ...]:
        """Read committed steps in increasing order, including rollbacks."""

    @abstractmethod
    def training_run_position(self) -> tuple[int, int]:
        """Return the last rollback step and subsequent training count."""

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
    interface: records and commits through sessions, settings through
    ``read_settings`` and ``write_settings``. Archiving a scenario retires all
    of it. A deployment names a subclass in ``storage.record_backend``.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path) -> Self:
        """Build the storage a deployment names.

        ``config`` is the plugin's own configuration. ``directory`` is the
        deployment's ``storage.agent_record_dir``, which the storage may use
        for local files.
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

    @abstractmethod
    def archive(self, scenario: str) -> tuple[str, ...]:
        """Archive a closed scenario's records, commits and settings; return archive locations."""

    @abstractmethod
    def prune(self, *, max_bytes: int) -> int:
        """Evict oldest bodies over max_bytes."""

    @abstractmethod
    def close(self) -> None:
        """Release shared storage resources; repeated closes are harmless."""


__all__ = ["ScenarioSettings", "ScenarioStorage", "ScenarioStore", "ScenarioStoreConflict"]
