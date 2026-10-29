"""Storage contract for the training samples of one scenario.

A client can upload an ATIF trajectory to train on directly. The sample store
keeps the uploaded trajectories of one scenario in upload order, each with the
number of committed jobs that trained it. A commit lists the samples that its
job data consumed, by :func:`consumed_sample_id`, and the commit history counts
them in the same transaction as the commit (``SQLCommitHistory``). A rebuilt
trainer reads only the samples that no committed job trained. Archiving the
scenario retires its samples with its records. Record retention does not
delete them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.core.errors import ReefError
from reef.core.job_data import TrajectorySample

#: Marks the ID of a consumed sample in a commit, so that it cannot be a record ID.
CONSUMED_SAMPLE_PREFIX = "sample:"


class SampleConflict(ReefError):
    """A sample ID is already stored with a different sample."""


def consumed_sample_id(sample_id: str) -> str:
    """Return the ID that a commit lists for a consumed sample."""
    return f"{CONSUMED_SAMPLE_PREFIX}{sample_id}"


@dataclass(frozen=True)
class StoredSample:
    """One sample in a sample store, and the number of committed jobs that trained it.

    Values: ``(sequence, sample_id, sample, train_count)`` is a stored sample
    if and only if ``sequence`` is a positive int, ``sample_id`` is a non-empty
    string, ``sample`` is a ``TrajectorySample``, and ``train_count`` is a
    non-negative int. ``sequence`` is the position of the sample in upload
    order. ``sample_id`` is the ID that the client gave it. ``train_count`` is
    the number of committed jobs whose commit lists
    ``consumed_sample_id(sample_id)``.

    Purpose: ``SampleStore.page`` makes it. The trainer reads it:
    ``sequence`` moves the trainer's cursor, ``train_count`` tells whether a
    committed job trained it, and the pipeline puts ``sample`` into job data.
    """

    sequence: int
    sample_id: str
    sample: TrajectorySample
    train_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("a stored sample needs a positive int sequence")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("a stored sample needs a non-empty sample_id")
        if not isinstance(self.sample, TrajectorySample):
            raise TypeError("a stored sample holds a TrajectorySample")
        if isinstance(self.train_count, bool) or not isinstance(self.train_count, int) or self.train_count < 0:
            raise ValueError("a stored sample needs a non-negative int train_count")


class SampleStore(ABC):
    """The training samples of one scenario.

    Values: the state of a store is a finite sequence of stored samples, in
    increasing ``sequence``. No two of them have the same ``sample_id``. A
    stored sample's ``sample`` does not change, and record retention does not
    remove it. Only a commit changes ``train_count``: the commit history adds
    one for each consumed sample ID that the commit lists. Archiving the
    scenario retires the whole store.

    Purpose: the dispatcher adds the trajectories of each upload (``add``). The
    trainer reads them in upload order, after its cursor (``page`` and ``count``).
    """

    @abstractmethod
    def add(self, samples: tuple[tuple[str, TrajectorySample], ...]) -> bool:
        """Save each ``(sample_id, sample)`` pair in one transaction, in order. Return True if one or more is new.

        A new sample has ``train_count`` 0. A pair whose ID is stored with the
        same sample changes nothing. Raises ``ValueError`` if an ID is empty,
        and ``SampleConflict`` if an ID is stored with a different sample.
        Then the store saves no sample of the call.
        """

    @abstractmethod
    def page(self, *, after_sequence: int = 0, limit: int = 256) -> tuple[StoredSample, ...]:
        """Read at most ``limit`` stored samples after ``after_sequence``, in upload order."""

    @abstractmethod
    def count(self, *, after_sequence: int = 0) -> int:
        """Return the number of stored samples after ``after_sequence``."""


__all__ = ["CONSUMED_SAMPLE_PREFIX", "SampleConflict", "SampleStore", "StoredSample", "consumed_sample_id"]
