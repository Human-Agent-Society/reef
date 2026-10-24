"""Storage contract for the ATIF trajectories that clients upload to a scenario.

A client can upload an ATIF trajectory to train on directly. An uploaded
trajectory is not a record. The trajectory store keeps the uploaded trajectories
of one scenario in upload order. A trainer reads them after its cursor and gives
them to the task-to-batch stage. A commit lists the uploaded trajectories that
its batch consumed, by :func:`consumed_trajectory_id`, so a rebuilt trainer
skips them. Archiving the scenario retires its uploaded trajectories with its
records. Record retention does not delete them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.core.batches import TrajectoryItem
from reef.core.errors import ReefError

#: Marks the ID of a consumed uploaded trajectory in a commit, so that it cannot be a record ID.
CONSUMED_TRAJECTORY_PREFIX = "trajectory:"


class TrajectoryConflict(ReefError):
    """A trajectory ID is already stored with a different trajectory."""


def consumed_trajectory_id(trajectory_id: str) -> str:
    """Return the ID that a commit lists for a consumed uploaded trajectory."""
    return f"{CONSUMED_TRAJECTORY_PREFIX}{trajectory_id}"


@dataclass(frozen=True)
class StoredTrajectory:
    """One uploaded trajectory in a trajectory store.

    Values: ``(sequence, trajectory_id, trajectory)`` is a stored trajectory if
    and only if ``sequence`` is a positive integer, ``trajectory_id`` is a
    non-empty string, and ``trajectory`` is a ``TrajectoryItem``. ``sequence`` is
    the position of the trajectory in upload order. ``trajectory_id`` is the ID
    that the client gave it.

    Purpose: ``TrajectoryStore.page`` makes it. The trainer reads it:
    ``sequence`` moves the trainer's cursor,
    ``consumed_trajectory_id(trajectory_id)`` tells whether a commit consumed
    it, and the pipeline batches ``trajectory``.
    """

    sequence: int
    trajectory_id: str
    trajectory: TrajectoryItem

    def __post_init__(self) -> None:
        if self.sequence < 1 or not self.trajectory_id:
            raise ValueError("a stored trajectory needs a positive sequence and a non-empty trajectory_id")


class TrajectoryStore(ABC):
    """The uploaded trajectories of one scenario.

    Values: the state of a store is a finite sequence of stored trajectories,
    in increasing ``sequence``. No two of them have the same ``trajectory_id``.
    A stored trajectory does not change, and record retention does not remove
    it. Archiving the scenario retires the whole store.

    Purpose: the dispatcher adds the trajectories of each upload (``add``). The
    trainer reads them in upload order, after its cursor (``page`` and ``count``).
    """

    @abstractmethod
    def add(self, trajectories: tuple[tuple[str, TrajectoryItem], ...]) -> bool:
        """Save each ``(trajectory_id, trajectory)`` pair in one transaction, in order. Return True if one or more is new.

        A pair whose ID is stored with the same trajectory changes nothing.
        Raises ``ValueError`` if an ID is empty, and ``TrajectoryConflict`` if
        an ID is stored with a different trajectory. Then the store saves no
        trajectory of the call.
        """

    @abstractmethod
    def page(self, *, after_sequence: int = 0, limit: int = 256) -> tuple[StoredTrajectory, ...]:
        """Read at most ``limit`` stored trajectories after ``after_sequence``, in upload order."""

    @abstractmethod
    def count(self, *, after_sequence: int = 0) -> int:
        """Return the number of stored trajectories after ``after_sequence``."""
