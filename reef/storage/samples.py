"""Storage contract for the training samples of one scenario.

A client can upload an ATIF trajectory to train on directly. The sample store
keeps the uploaded trajectories of one scenario in upload order, each with the
number of committed jobs of each trainer that trained it. A commit lists the
samples that its job data consumed, by :func:`consumed_sample_id`, and the
commit history counts them for the commit's trainer in the same transaction as
the commit (``SQLCommitHistory``). A rebuilt trainer reads only the samples
that no committed job of its own trained, so in a scenario of several
components each trainer trains each sample. Archiving the
scenario retires its samples with its records. Record retention does not
delete them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

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
    """One sample in a sample store, and the number of committed jobs of each trainer that trained it.

    Values: ``(sequence, sample_id, sample, train_counts)`` is a stored sample
    if and only if ``sequence`` is a positive int, ``sample_id`` is a non-empty
    string, ``sample`` is a ``TrajectorySample``, and ``train_counts`` maps
    strings to positive ints. ``sequence`` is the position of the sample in
    upload order. ``sample_id`` is the ID that the client gave it.
    ``train_counts[c]`` is the number of committed jobs of the trainer ``c``
    whose commit lists ``consumed_sample_id(sample_id)``. ``c`` is the
    component that the commits name, or ``""`` for the only trainer of a
    scenario, whose commits name none. A trainer that is not a key has trained
    the sample 0 times. The value keeps its own copy of ``train_counts``.

    Operation: ``train_count(component)`` is ``train_counts`` of the key of
    ``component`` (``""`` for ``None``), or 0 when it is not a key.

    Purpose: ``SampleStore.page`` makes it. The trainer reads it:
    ``sequence`` moves the trainer's cursor, ``train_count`` of its own
    component tells whether one of its committed jobs trained it, and the
    pipeline puts ``sample`` into job data.
    """

    sequence: int
    sample_id: str
    sample: TrajectorySample
    train_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("a stored sample needs a positive int sequence")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("a stored sample needs a non-empty sample_id")
        if not isinstance(self.sample, TrajectorySample):
            raise TypeError("a stored sample holds a TrajectorySample")
        if not isinstance(self.train_counts, Mapping) or any(
            not isinstance(component, str) or isinstance(count, bool) or not isinstance(count, int) or count < 1
            for component, count in self.train_counts.items()
        ):
            raise ValueError("a stored sample's train_counts map component names to positive ints")
        object.__setattr__(self, "train_counts", dict(self.train_counts))

    def train_count(self, component: str | None) -> int:
        """The number of committed jobs of the trainer ``component`` (``None`` for a scenario's only one) that held it."""
        return self.train_counts.get(component or "", 0)


class SampleStore(ABC):
    """The training samples of one scenario.

    Values: the state of a store is a finite sequence of stored samples, in
    increasing ``sequence``. No two of them have the same ``sample_id``. A
    stored sample's ``sample`` does not change, and record retention does not
    remove it. Only a commit changes ``train_counts``: the commit history adds
    one, for the commit's trainer, for each consumed sample ID that the commit
    lists. Archiving the
    scenario retires the whole store.

    Purpose: the dispatcher adds the trajectories of each upload (``add``). The
    trainer reads them in upload order, after its cursor (``page`` and ``count``).
    """

    @abstractmethod
    def add(self, samples: tuple[tuple[str, TrajectorySample], ...]) -> bool:
        """Save each ``(sample_id, sample)`` pair in one transaction, in order. Return True if one or more is new.

        A new sample has no train count. A pair whose ID is stored with the
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
