"""Storage contract for the training samples of one scenario.

The sample store keeps the samples of one scenario in the order they arrived:
the ATIF trajectories that clients upload to train on directly, the Harbor
tasks that clients or the deployment import, each in the ``train`` or the
``eval`` split, and the samples that a trainer's stages make. Each sample has the number of committed jobs of each trainer
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
from reef.core.job_data import Sample, TaskSample, TrajectorySample
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
    * An imported task (no ``component``) has ``split`` ``train`` or ``eval``,
      and ``files_digest`` is the non-empty digest of its directory in the
      task file store. Its ``sample.task_path`` is that directory, and
      ``sample.metadata["task"]`` is ``{"name": sample_id, "files_digest":
      files_digest, "split": split}``.
    * A task that a trainer's stage made (a ``component``), from records or
      from no record set, has ``split`` ``train`` and ``files_digest``
      ``None``; its sample is as the stage made it.
    * ``train_counts`` maps strings to positive ints. ``train_counts[c]`` is
      the number of committed jobs of the trainer ``c`` whose commit lists
      ``consumed_sample_id(sample_id)``. ``c`` is the component that the
      commits name, or ``""`` for the only trainer of a scenario, whose
      commits name none. A trainer that is not a key has trained the sample
      0 times. The value keeps its own copy of ``train_counts``.
    * ``component`` is the trainer whose stage made the sample, as a
      ``train_counts`` key, or ``None`` for a sample that every trainer reads:
      an upload or an imported task.
    * ``group`` is ``None`` or a non-empty string that the samples of one unit
      share, so that they train together. ``record_ids`` is a tuple of
      distinct non-empty strings: the records that the sample's unit consumed.
      It is empty when ``component`` is ``None``, and for a sample that a
      stage made from no record set (``Record2Sample.made_samples``).

    Operations: ``kind`` is ``trajectory`` for a ``TrajectorySample`` and
    ``task`` for a ``TaskSample``. ``train_count(trainer)`` is
    ``train_counts[trainer]``, or 0 when ``trainer`` is not a key.

    Purpose: ``SampleStore`` makes it. A trainer's pipeline reads the samples
    in arrival order: ``sequence`` moves its cursor, and ``train_count`` of its
    own key tells whether one of its committed jobs trained the sample. An
    evaluation reads the imported ``eval`` tasks, and the listings read every
    sample.
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
        elif isinstance(self.sample, TaskSample) and self.component is not None:
            if self.split != "train" or self.files_digest is not None:
                raise ValueError("a task that a stage made is in the train split and has no files digest")
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
        if self.component is not None and not isinstance(self.component, str):
            raise ValueError("a stored sample's component is a string or None")
        if self.group is not None and (not isinstance(self.group, str) or not self.group):
            raise ValueError("a stored sample's group is a non-empty string or None")
        ids = self.record_ids
        if (
            not isinstance(ids, tuple)
            or any(not isinstance(record_id, str) or not record_id for record_id in ids)
            or len(set(ids)) != len(ids)
            or (ids and self.component is None)
        ):
            raise ValueError("a stored sample's record_ids are distinct record IDs, empty without a component")

    @property
    def kind(self) -> SampleKind:
        return "trajectory" if isinstance(self.sample, TrajectorySample) else "task"

    def train_count(self, trainer: str) -> int:
        """The number of committed jobs of ``trainer`` (a ``train_counts`` key) that held the sample."""
        return self.train_counts.get(trainer, 0)


class SampleStore(ABC):
    """The training samples of one scenario.

    Values: the state of a store is a pair ``(S, B)``. ``S`` is a finite
    sequence of stored samples, in increasing ``sequence``. No two of them have
    the same ``sample_id``, also across kinds. A stored sample's ``sample``,
    ``split`` and ``files_digest`` do not change, and record retention does not
    remove it. ``remove_uploads`` is the only operation that removes samples:
    the uploads (samples of no component) of the data items that the scenario
    no longer uses. A new sample always gets a ``sequence`` above every
    ``sequence`` the store gave before. ``B`` maps each trainer (a ``train_counts`` key) to its manual
    buffer, a set of sample IDs. ``insert_into_buffer`` and
    ``remove_from_buffer`` change ``B``. Only a commit changes
    ``train_counts``: the commit history adds one, for the commit's trainer,
    for each consumed sample ID that the commit lists, and removes those IDs
    from that trainer's manual buffer. Archiving the scenario retires the whole
    store.

    Purpose: the data sync (``reef.storage.data_sync``) keeps the uploads equal
    to the scenario's data items: it adds trajectories (``add_trajectories``)
    and tasks (``add_tasks``), and removes the uploads of items that left the
    scenario (``remove_uploads``). A trainer's pipeline adds the samples that its
    stages make (``add_record_samples``), and a rebuilt trainer skips the
    records they consumed (``consumed_record_ids``). The pipeline reads the
    samples in arrival order, after its cursor (``page``), and the samples of
    its manual buffer (``manual_buffer``, ``get``). An evaluation and the task
    listing read the imported tasks (``imported_tasks``).
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
        samples: tuple[tuple[str, Sample], ...],
    ) -> bool:
        """Save the samples of one unit that ``trainer`` made from ``record_ids``, in one transaction, in order.

        ``record_ids`` is empty for samples that a stage made from no record set.

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
    def get(self, sample_ids: tuple[str, ...]) -> tuple[StoredSample, ...]:
        """The stored samples of ``sample_ids`` that the store holds, in arrival order."""

    @abstractmethod
    def manual_buffer(self, trainer: str) -> tuple[str, ...]:
        """The IDs of the samples in the manual buffer of ``trainer`` (a ``train_counts`` key), in arrival order."""

    @abstractmethod
    def insert_into_buffer(self, trainer: str, sample_ids: tuple[str, ...]) -> None:
        """Add ``sample_ids`` to the manual buffer of ``trainer``, in one transaction; a held ID stays held.

        The caller checks that each ID names a stored sample that ``trainer`` can train. A commit of ``trainer``
        removes from its manual buffer the samples that it consumed, in the commit's transaction.
        """

    @abstractmethod
    def remove_from_buffer(self, trainer: str, sample_ids: tuple[str, ...]) -> None:
        """Remove ``sample_ids`` from the manual buffer of ``trainer``, in one transaction; an absent ID is skipped."""

    @abstractmethod
    def remove_uploads(self, sample_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Remove the uploads of ``sample_ids`` and their entries in every manual buffer, in one transaction.

        An upload is a sample of no component. An ID of no upload is skipped. The train counts stay, so a sample
        that the store gets again under the same ID keeps them. Return the IDs that the call removed, in arrival
        order.
        """

    def imported_tasks(self, split: SplitName | None = None) -> tuple[StoredSample, ...]:
        """Every imported task (a task of no component), in arrival order; only ``split``'s if given."""
        found: list[StoredSample] = []
        after_sequence = 0
        while page := self.page(after_sequence=after_sequence, kind="task"):
            found.extend(
                stored for stored in page if stored.component is None and (split is None or stored.split == split)
            )
            after_sequence = page[-1].sequence
        return tuple(found)


class MemorySampleStore(SampleStore):
    """A sample store in process memory, for a trainer without a scenario session, such as a test's pipeline.

    Values: ``rows`` holds ``S`` and ``buffers`` holds ``B``. There is no commit history, so every
    ``train_counts`` stays empty and nothing removes trained samples from a manual buffer; the pipeline keeps the
    samples that it trained in this process itself. Imported tasks need a task file store, so ``add_tasks`` raises
    ``ValueError``. Purpose: ``TrainingPipeline`` makes one when its context has no sample store.
    """

    def __init__(self) -> None:
        self.rows: list[StoredSample] = []
        self.buffers: dict[str, set[str]] = {}
        self.next_sequence = 1

    def stored(self, sample_id: str) -> StoredSample | None:
        return next((row for row in self.rows if row.sample_id == sample_id), None)

    def add_trajectories(self, trajectories: tuple[tuple[str, TrajectorySample], ...]) -> bool:
        return self.add_rows(trajectories, component=None, group=None, record_ids=())

    def add_record_samples(
        self,
        trainer: str,
        group: str | None,
        record_ids: tuple[str, ...],
        samples: tuple[tuple[str, Sample], ...],
    ) -> bool:
        if not samples or any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
            raise ValueError("record samples need one or more samples and non-empty record IDs")
        return self.add_rows(samples, component=trainer, group=group, record_ids=tuple(sorted(set(record_ids))))

    def add_rows(
        self,
        samples: tuple[tuple[str, Sample], ...],
        *,
        component: str | None,
        group: str | None,
        record_ids: tuple[str, ...],
    ) -> bool:
        """Store each new ``(sample_id, sample)`` pair after the stored samples, as ``add_trajectories`` does."""
        new: dict[str, StoredSample] = {}
        for sample_id, sample in samples:
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError("a sample ID must be a non-empty string")
            stored = new.get(sample_id) or self.stored(sample_id)
            if stored is None:
                new[sample_id] = StoredSample(
                    self.next_sequence + len(new), sample_id, "train", sample, None, {}, component, group, record_ids
                )
            elif stored.sample != sample:
                raise SampleConflict(f"sample {sample_id!r} is already stored with other content")
        self.rows.extend(new.values())
        self.next_sequence += len(new)
        return bool(new)

    def add_tasks(self, directories: tuple[Path, ...], split: SplitName) -> tuple[StoredSample, ...]:
        raise ValueError("a sample store in memory keeps no task files")

    def consumed_record_ids(self, trainer: str) -> frozenset[str]:
        return frozenset(record_id for row in self.rows if row.component == trainer for record_id in row.record_ids)

    def page(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 256,
        kind: SampleKind | None = None,
        trainer: str | None = None,
    ) -> tuple[StoredSample, ...]:
        if after_sequence < 0 or limit <= 0:
            raise ValueError("after_sequence must be non-negative and limit positive")
        return tuple(
            row
            for row in self.rows
            if row.sequence > after_sequence
            and (kind is None or row.kind == kind)
            and (trainer is None or row.component in (None, trainer))
        )[:limit]

    def get(self, sample_ids: tuple[str, ...]) -> tuple[StoredSample, ...]:
        return tuple(row for row in self.rows if row.sample_id in sample_ids)

    def manual_buffer(self, trainer: str) -> tuple[str, ...]:
        held = self.buffers.get(trainer, set())
        return tuple(row.sample_id for row in self.rows if row.sample_id in held)

    def insert_into_buffer(self, trainer: str, sample_ids: tuple[str, ...]) -> None:
        self.buffers.setdefault(trainer, set()).update(sample_ids)

    def remove_from_buffer(self, trainer: str, sample_ids: tuple[str, ...]) -> None:
        self.buffers.get(trainer, set()).difference_update(sample_ids)

    def remove_uploads(self, sample_ids: tuple[str, ...]) -> tuple[str, ...]:
        removed = tuple(row.sample_id for row in self.rows if row.component is None and row.sample_id in sample_ids)
        self.rows = [row for row in self.rows if row.sample_id not in removed]
        for buffer in self.buffers.values():
            buffer.difference_update(removed)
        return removed


@dataclass(frozen=True)
class TrainerSamples:
    """The samples of a scenario as one trainer reads and writes them: a sample store and the trainer's key.

    Values: ``store`` is the scenario's ``SampleStore``; ``trainer`` is the trainer's ``train_counts`` key, the
    component that its commits name or ``""`` for the only trainer of a scenario. Operations pass ``trainer``
    to the store: ``page`` reads the samples that the trainer reads (its own and those of no component),
    ``add_record_samples`` saves its own, ``consumed_record_ids`` reads the records that its samples consumed,
    ``imported_tasks`` reads the imported tasks, and ``manual_buffer``, ``insert_into_buffer`` and ``remove_from_buffer``
    read and change its manual buffer. Purpose: ``Recipe.build_trainers`` makes one for each trainer; the
    trainer's pipeline (``PipelineContext.samples``) and its stages read it.
    """

    store: SampleStore
    trainer: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.store, SampleStore):
            raise TypeError("TrainerSamples.store must be a SampleStore")
        if not isinstance(self.trainer, str):
            raise TypeError("TrainerSamples.trainer must be a string")

    def page(
        self, *, after_sequence: int = 0, limit: int = 256, kind: SampleKind | None = None
    ) -> tuple[StoredSample, ...]:
        return self.store.page(after_sequence=after_sequence, limit=limit, kind=kind, trainer=self.trainer)

    def add_record_samples(
        self, group: str | None, record_ids: tuple[str, ...], samples: tuple[tuple[str, Sample], ...]
    ) -> bool:
        return self.store.add_record_samples(self.trainer, group, record_ids, samples)

    def consumed_record_ids(self) -> frozenset[str]:
        return self.store.consumed_record_ids(self.trainer)

    def imported_tasks(self, split: SplitName | None = None) -> tuple[StoredSample, ...]:
        return self.store.imported_tasks(split)

    def get(self, sample_ids: tuple[str, ...]) -> tuple[StoredSample, ...]:
        return self.store.get(sample_ids)

    def manual_buffer(self) -> tuple[str, ...]:
        return self.store.manual_buffer(self.trainer)

    def insert_into_buffer(self, sample_ids: tuple[str, ...]) -> None:
        self.store.insert_into_buffer(self.trainer, sample_ids)

    def remove_from_buffer(self, sample_ids: tuple[str, ...]) -> None:
        self.store.remove_from_buffer(self.trainer, sample_ids)


__all__ = [
    "CONSUMED_SAMPLE_PREFIX",
    "SAMPLE_KINDS",
    "MemorySampleStore",
    "SampleConflict",
    "SampleKind",
    "SampleStore",
    "StoredSample",
    "TrainerSamples",
    "consumed_sample_id",
]
