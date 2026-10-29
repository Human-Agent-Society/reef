"""Storage contract for the training samples of one scenario.

The sample store keeps the samples of one scenario in the order they arrived:
the ATIF trajectories that clients upload to train on directly, and the Harbor
tasks that clients or the deployment import, each in the ``train`` or the
``eval`` split. Each sample has the number of committed jobs of each trainer
that trained it. A commit lists the samples that its job data consumed, by
:func:`consumed_sample_id`, and the commit history counts them for the commit's
trainer in the same transaction as the commit (``SQLCommitHistory``). A rebuilt
trainer reads only the samples that no committed job of its own trained, so in
a scenario of several components each trainer trains each sample. An ``eval``
task never trains: an evaluation runs it. Archiving the scenario retires its
samples with its records. Record retention does not delete them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from reef.core.errors import ReefError
from reef.core.job_data import TaskSample, TrajectorySample
from reef.storage.tasks import SPLIT_NAMES, SplitName

#: Marks the ID of a consumed sample in a commit, so that it cannot be a record ID.
CONSUMED_SAMPLE_PREFIX = "sample:"

SampleKind = Literal["trajectory", "task"]
SAMPLE_KINDS: tuple[SampleKind, ...] = ("trajectory", "task")


class SampleConflict(ReefError):
    """A sample ID is already stored with a different sample, kind or split."""


def consumed_sample_id(sample_id: str) -> str:
    """Return the ID that a commit lists for a consumed sample."""
    return f"{CONSUMED_SAMPLE_PREFIX}{sample_id}"


@dataclass(frozen=True)
class StoredSample:
    """One sample in a sample store, and the number of committed jobs of each trainer that trained it.

    Values: ``(sequence, sample_id, split, sample, files_digest, train_counts,
    component, group, record_ids)`` is a stored sample if and only if all of
    these hold:

    * ``sequence`` is a positive int, the position of the sample in arrival order.
    * ``sample_id`` is a non-empty string: the ID that the client gave a
      trajectory, or the name of a task.
    * ``sample`` is a ``TrajectorySample`` or a ``TaskSample``.
    * A trajectory has ``split`` ``train`` and ``files_digest`` ``None``.
    * A task has ``split`` ``train`` or ``eval``, and ``files_digest`` is the
      non-empty digest of its directory in the task file store. Its
      ``sample.task_path`` is that directory, and ``sample.metadata["task"]``
      is ``{"name": sample_id, "files_digest": files_digest, "split": split}``.
    * ``train_counts`` maps strings to positive ints. ``train_counts[c]`` is
      the number of committed jobs of the trainer ``c`` whose commit lists
      ``consumed_sample_id(sample_id)``. ``c`` is the component that the
      commits name, or ``""`` for the only trainer of a scenario, whose
      commits name none. A trainer that is not a key has trained the sample
      0 times. The value keeps its own copy of ``train_counts``.
    * ``component`` is the trainer that made the sample from records, as a
      ``train_counts`` key, or ``None`` for a sample that every trainer reads:
      an upload or an imported task. Only a trajectory has a ``component``.
    * ``group`` is ``None`` or a non-empty string that the samples of one unit
      share, so that they train together. ``record_ids`` is a tuple of
      distinct non-empty strings: the records that the sample's unit consumed,
      empty exactly when ``component`` is ``None``.

    Operations: ``kind`` is ``trajectory`` for a ``TrajectorySample`` and
    ``task`` for a ``TaskSample``. ``train_count(component)`` is
    ``train_counts`` of the key of ``component`` (``""`` for ``None``), or 0
    when it is not a key.

    Purpose: ``SampleStore`` makes it. The trainer reads the trajectories:
    ``sequence`` moves its cursor, and ``train_count`` of its own component
    tells whether one of its committed jobs trained the sample. A recipe's
    stage reads the untrained ``train`` tasks, an evaluation the ``eval``
    tasks, and the task listing every task.
    """

    sequence: int
    sample_id: str
    split: SplitName
    sample: TrajectorySample | TaskSample
    files_digest: str | None = None
    train_counts: Mapping[str, int] = field(default_factory=dict)
    component: str | None = None
    group: str | None = None
    record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("a stored sample needs a positive int sequence")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("a stored sample needs a non-empty sample_id")
        if self.split not in SPLIT_NAMES:
            raise ValueError("a stored sample's split is 'train' or 'eval'")
        if isinstance(self.sample, TrajectorySample):
            if self.split != "train" or self.files_digest is not None:
                raise ValueError("a stored trajectory is in the train split and has no files digest")
        elif isinstance(self.sample, TaskSample):
            if not isinstance(self.files_digest, str) or not self.files_digest:
                raise ValueError("a stored task needs the files digest of its directory")
            task = {"name": self.sample_id, "files_digest": self.files_digest, "split": self.split}
            if self.sample.metadata.get("task") != task:
                raise ValueError(f"a stored task's sample metadata names it as {task}")
        else:
            raise TypeError("a stored sample holds a TrajectorySample or a TaskSample")
        if not isinstance(self.train_counts, Mapping) or any(
            not isinstance(component, str) or isinstance(count, bool) or not isinstance(count, int) or count < 1
            for component, count in self.train_counts.items()
        ):
            raise ValueError("a stored sample's train_counts map component names to positive ints")
        object.__setattr__(self, "train_counts", dict(self.train_counts))
        if self.component is not None and (not isinstance(self.component, str) or self.kind != "trajectory"):
            raise ValueError("only a trajectory made from records has a component, a string")
        if self.group is not None and (not isinstance(self.group, str) or not self.group):
            raise ValueError("a stored sample's group is a non-empty string or None")
        ids = self.record_ids
        if (
            not isinstance(ids, tuple)
            or any(not isinstance(record_id, str) or not record_id for record_id in ids)
            or len(set(ids)) != len(ids)
            or (not ids) != (self.component is None)
        ):
            raise ValueError("a stored sample's record_ids are distinct record IDs, empty exactly without a component")

    @property
    def kind(self) -> SampleKind:
        return "trajectory" if isinstance(self.sample, TrajectorySample) else "task"

    def train_count(self, component: str | None) -> int:
        """The number of committed jobs of the trainer ``component`` (``None`` for a scenario's only one) that held it."""
        return self.train_counts.get(component or "", 0)


class SampleStore(ABC):
    """The training samples of one scenario.

    Values: the state of a store is a finite sequence of stored samples, in
    increasing ``sequence``. No two of them have the same ``sample_id``, also
    across kinds. A stored sample's ``sample``, ``split`` and ``files_digest``
    do not change, and record retention does not remove it. Only a commit
    changes ``train_counts``: the commit history adds one, for the commit's
    trainer, for each consumed sample ID that the commit lists. Archiving the
    scenario retires the whole store.

    Purpose: the dispatcher adds the trajectories of each upload
    (``add_trajectories``) and the tasks of each task upload or deployment
    import (``add_tasks``). A trainer's pipeline adds the samples of each unit
    that its record stages make (``add_record_samples``), and a rebuilt trainer
    skips the records they consumed (``consumed_record_ids``). The trainer reads the trajectories in arrival order,
    after its cursor (``page`` and ``count`` with ``kind="trajectory"``). A
    recipe's stage, an evaluation and the task listing read the tasks
    (``page`` with ``kind="task"``).
    """

    @abstractmethod
    def add_trajectories(self, trajectories: tuple[tuple[str, TrajectorySample], ...]) -> bool:
        """Save each ``(sample_id, trajectory)`` pair in one transaction, in order. Return True if one or more is new.

        A new trajectory is in the ``train`` split and has no train count. A
        pair whose ID is stored with the same trajectory changes nothing.
        Raises ``ValueError`` if an ID is empty, and ``SampleConflict`` if an
        ID is stored with another sample. Then the store saves no sample of the
        call.
        """

    @abstractmethod
    def add_record_samples(
        self,
        trainer: str,
        group: str | None,
        record_ids: tuple[str, ...],
        trajectories: tuple[tuple[str, TrajectorySample], ...],
    ) -> bool:
        """Save the samples of one unit that ``trainer`` made from ``record_ids``, in one transaction, in order.

        ``trainer`` is a ``train_counts`` key: the component that the trainer's commits name, or ``""`` for the
        only trainer of a scenario. Each sample gets ``component`` ``trainer``, ``group`` ``group`` and
        ``record_ids`` ``record_ids``; once the transaction ends, the records are consumed for ``trainer``
        (``consumed_record_ids``). Return True if one or more sample is new. Raises ``ValueError`` for no sample
        or an empty record ID, and ``SampleConflict`` if an ID is stored with another sample; then the store saves
        no sample of the call.
        """

    @abstractmethod
    def consumed_record_ids(self, trainer: str) -> frozenset[str]:
        """The records that the stored samples of ``trainer`` (a ``train_counts`` key) consumed."""

    @abstractmethod
    def add_tasks(self, directories: tuple[Path, ...], split: SplitName) -> tuple[StoredSample, ...]:
        """Save the Harbor task in each directory to ``split`` in one transaction, in order; return the stored tasks.

        A task's ``sample_id`` is the name of its directory. A new task has no
        train count. A task that the store holds with the same files and split
        is returned as it is. Raises ``ValueError`` for another split, and
        ``SampleConflict`` if a name is stored with other files, in the other
        split, or as a trajectory. Then the store saves no sample of the call;
        the task file store may keep copies of the files.
        """

    @abstractmethod
    def page(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 256,
        kind: SampleKind | None = None,
        trainer: str | None = None,
    ) -> tuple[StoredSample, ...]:
        """Read at most ``limit`` stored samples after ``after_sequence``, in arrival order.

        Only ``kind``'s if given. With ``trainer`` (a ``train_counts`` key), only the samples that trainer reads:
        its own and those of no component.
        """

    @abstractmethod
    def count(self, *, after_sequence: int = 0, kind: SampleKind | None = None, trainer: str | None = None) -> int:
        """Return the number of stored samples after ``after_sequence`` that ``page`` would read with the filters."""

    def tasks(self, split: SplitName | None = None) -> tuple[StoredSample, ...]:
        """Every stored task, in arrival order; only ``split``'s if given."""
        found: list[StoredSample] = []
        after_sequence = 0
        while page := self.page(after_sequence=after_sequence, kind="task"):
            found.extend(stored for stored in page if split is None or stored.split == split)
            after_sequence = page[-1].sequence
        return tuple(found)


__all__ = [
    "CONSUMED_SAMPLE_PREFIX",
    "SAMPLE_KINDS",
    "SampleConflict",
    "SampleKind",
    "SampleStore",
    "StoredSample",
    "consumed_sample_id",
]
