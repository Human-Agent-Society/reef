"""Storage boundary for a scenario's records, commits, and recovery.

Artifact publication remains the responsibility of the commit protocol. A
store settles record consumption and the committed scenario state together;
implementations may use a database transaction or a recoverable journal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


from reef.core.artifact_ref import ArtifactRef
from reef.core.errors import ReefError
from reef.records import RecordStore
from reef.scenario.state import CommitRecord, ScenarioSnapshot


class ScenarioStoreConflict(ReefError):
    """A commit's expected step or content conflicts with the stored history."""


class ScenarioStore(ABC):
    """One scenario's storage session, including its record connection.

    The caller closes the session when the scenario is unloaded. Closing is
    idempotent; subsequent reads and writes fail. Records from other scenarios
    must never be committed or compacted through this session.
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
        """Commit exactly ``expected_step + 1`` and settle its record compaction.

        Conflicting writers are serialized across sessions. An identical retry
        (ignoring only ``recorded_at``) returns the original record, even after
        later steps, and repairs interrupted compaction. Different content or
        a stale expected step raises ``ScenarioStoreConflict``.

        An exception can occur after durability but before compaction. Callers
        inspect history or retry the same commit before repeating training.
        """

    @abstractmethod
    def recover(self, *, snapshot: ScenarioSnapshot, checkpoint_head: ArtifactRef) -> CommitRecord | None:
        """Reconcile the checkpoint snapshot and history and replay compaction.

        A checkpoint ahead of history is adopted durably before compaction.
        Validate scenario identity and continuity after the checkpoint. Return
        the committed head, or ``None`` at creation, without publishing artifacts.
        """

    @abstractmethod
    def close(self) -> None:
        """Release this session's records and other resources exactly once."""


class ScenarioStoreFactory(ABC):
    """Open sessions and maintain their storage without exposing file paths.

    The owner serializes archive and retention with lifecycle changes and
    closes a scenario's sessions before archiving it. Factory closure releases
    factory resources only; sessions remain owned by their scenarios.
    """

    @property
    @abstractmethod
    def durable(self) -> bool:
        """Whether newly opened sessions retain commits across restarts."""

    @abstractmethod
    def open(self, scenario: str) -> ScenarioStore:
        """Open an independently owned session for this scenario."""

    @abstractmethod
    def archive(self, scenario: str) -> tuple[str, ...]:
        """Archive a closed scenario's state and return archive locations."""

    @abstractmethod
    def prune(self, *, days: float, max_bytes: int) -> int:
        """Purge retained compacted bodies across active and archived storage."""

    @abstractmethod
    def close(self) -> None:
        """Release factory resources; repeated closes are harmless."""


__all__ = ["ScenarioStore", "ScenarioStoreConflict", "ScenarioStoreFactory"]
