"""The training samples of a scenario, and its training state (RFC #1028, RFC #1140).

Every training sample is an item of the deployment's data store: an uploaded trajectory, an imported Harbor task,
or a trajectory or task that a trainer's stage made. A scenario trains on the items of its owner that it uses (their
``scenarios`` metadata), in the order that the uses began (``DataStore.scenario_items``). An item that a stage of the
scenario made names that stage's trainer in its ``origin``. Every other item is an upload or an imported task, which
every trainer of the scenario reads.

The scenario storage keeps only the training state (``TrainingState``): for each item and trainer, the number of
committed jobs that trained it, and for each trainer the records that its stages consumed. A commit lists the samples
that its job data consumed (``consumed_sample_id``), and the commit history counts them for the commit's trainer in
the commit's transaction. An ``eval`` task never trains: an evaluation runs it.
"""

from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import batched
from pathlib import Path
from typing import Any, Literal, cast

from reef.core.job_data import Sample, TaskSample, TrajectorySample
from reef.storage.data import MAX_QUERY_LIMIT, DataStore, ItemConflict, MemoryDataStore, NewItem, StoredItem
from reef.storage.tasks import LocalTaskFileStore, SplitName

#: Marks the ID of a consumed sample in a commit, so that it cannot be a record ID.
CONSUMED_SAMPLE_PREFIX = "sample:"
#: The prefix of a record ID in a ``TrainingState`` that a stage ended and no commit named when it was stored.
ENDED_RECORD_PREFIX = "ended:"

SampleKind = Literal["trajectory", "task"]
SAMPLE_KINDS: tuple[SampleKind, ...] = ("trajectory", "task")


def consumed_sample_id(sample_id: str) -> str:
    """Return the ID that a commit lists for a consumed sample."""
    return f"{CONSUMED_SAMPLE_PREFIX}{sample_id}"


def stage_item_id(scenario: str, name: str) -> str:
    """The item ID of the sample that a stage of ``scenario`` made under ``name``. It holds no ``/``, so a task item
    can have a directory of that name."""
    return f"{scenario}.{name}".replace("/", ".")


@dataclass(frozen=True)
class StoredSample:
    """One item that a scenario uses, as its trainers read it, with the train counts of each trainer.

    Values: ``sequence`` is the positive sequence of the scenario's use of the item (``DataStore.scenario_items``),
    or the item's sequence when ``SampleStore.get`` read it. ``sample_id`` is the item's ID. ``split`` is the
    item's split for the scenario: ``train``, or ``eval`` for an imported task that an evaluation runs. ``sample`` is
    the ``TrajectorySample`` or ``TaskSample`` of the item. ``train_counts`` maps each trainer (its component, or
    ``""`` for the only trainer of a scenario) to the positive number of its committed jobs that listed
    ``consumed_sample_id(sample_id)``. ``component`` is the trainer whose stage made the item, or None for an upload
    or an imported task. ``group`` is None or the name that the samples of one stage unit share, so that they train
    together.

    Purpose: ``SampleStore`` makes it. A trainer's pipeline reads it, and the listings and evaluations too.
    """

    sequence: int
    sample_id: str
    split: SplitName
    sample: TrajectorySample | TaskSample
    train_counts: Mapping[str, int] = field(default_factory=dict)
    component: str | None = None
    group: str | None = None

    @property
    def kind(self) -> SampleKind:
        return "trajectory" if isinstance(self.sample, TrajectorySample) else "task"

    def train_count(self, trainer: str) -> int:
        """The number of committed jobs of ``trainer`` (a ``train_counts`` key) that held the sample."""
        return self.train_counts.get(trainer, 0)


@dataclass(frozen=True)
class ListedSample:
    """What a listing shows of one sample: the values of ``StoredSample`` without its content.

    Values: ``sequence``, ``sample_id``, ``split``, ``train_counts``, ``component`` and ``group`` are those of
    ``StoredSample``, and ``kind`` is the kind of its item.

    Purpose: ``SampleStore.listing`` makes it, and the listing routes read it.
    """

    sequence: int
    sample_id: str
    kind: SampleKind
    split: SplitName
    train_counts: Mapping[str, int]
    component: str | None
    group: str | None


class TrainingState(ABC):
    """The training state of one scenario: what its trainers trained and consumed.

    Values: the train count of each sample ID and trainer, a positive int, and the IDs of the records that each
    trainer's stages consumed. Only a commit changes a train count: the commit history adds one, for the commit's
    trainer, for each sample that the commit lists (``consumed_sample_id``), in the commit's transaction. A train
    count stays when its item leaves the scenario, so an item that comes back keeps it.

    Purpose: the scenario storage keeps it with the scenario's commits. ``SampleStore`` reads it.
    """

    @abstractmethod
    def train_counts(self, sample_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        """The train counts of each ID of ``sample_ids`` that has one, by trainer."""

    @abstractmethod
    def consumed_record_ids(self, trainer: str) -> frozenset[str]:
        """The records that the stages of ``trainer`` consumed."""

    @abstractmethod
    def consume_records(self, trainer: str, record_ids: Sequence[str]) -> None:
        """Mark ``record_ids`` consumed by the stages of ``trainer``; a consumed ID stays consumed."""


class MemoryTrainingState(TrainingState):
    """A training state in process memory, for a trainer without a scenario session. It has no commit history, so no
    train count changes."""

    def __init__(self) -> None:
        self.consumed: dict[str, set[str]] = {}

    def train_counts(self, sample_ids: Sequence[str]) -> dict[str, dict[str, int]]:
        return {}

    def consumed_record_ids(self, trainer: str) -> frozenset[str]:
        return frozenset(self.consumed.get(trainer, ()))

    def consume_records(self, trainer: str, record_ids: Sequence[str]) -> None:
        self.consumed.setdefault(trainer, set()).update(record_ids)


@dataclass(frozen=True)
class SampleStore:
    """The sample store of one scenario: the data items that it trains on, read as samples, with its training state.

    Values: ``data`` is the deployment's data store, ``owner`` the scenario's owner there, ``scenario`` its name and
    ``state`` its training state.

    Purpose: ``ScenarioStorage.sample_store`` makes it for an open scenario. The listings and evaluations read it,
    and ``Recipe.build_trainers`` gives each trainer its ``TrainerSamples`` (``for_trainer``).
    """

    data: DataStore
    owner: str
    scenario: str
    state: TrainingState

    def for_trainer(self, trainer: str) -> TrainerSamples:
        """The samples of the scenario as ``trainer``, a ``train_counts`` key, reads and writes them."""
        return TrainerSamples(self, trainer)

    def page(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 256,
        kind: SampleKind | None = None,
        trainer: str | None = None,
    ) -> tuple[StoredSample, ...]:
        """The samples whose use sequence is above ``after_sequence``, in use order, at most ``limit``; only
        ``kind``'s if given. With ``trainer``, only the samples that it reads: its own, and the uploads and imported
        tasks. A page is empty only after the last such sample."""
        # The store filters by kind: a scenario can use many record items before its first sample.
        kinds = SAMPLE_KINDS if kind is None else (kind,)
        while uses := self.data.scenario_items(
            self.owner, self.scenario, kinds=kinds, after=after_sequence, limit=min(limit, MAX_QUERY_LIMIT)
        ):
            found = tuple(
                stored for stored in self.stored(uses) if trainer is None or stored.component in (None, trainer)
            )
            if found:
                return found
            after_sequence = uses[-1][0]
        return ()

    def listing(self, *, after_sequence: int = 0, limit: int = 256) -> tuple[ListedSample, ...]:
        """The samples whose use sequence is above ``after_sequence``, in use order, at most ``limit``, without their
        content: a scenario can hold more trajectories than a listing can read in time."""
        uses = self.data.scenario_items(
            self.owner, self.scenario, kinds=SAMPLE_KINDS, after=after_sequence, limit=limit, content=False
        )
        counts = self.state.train_counts([item.item_id for _, item in uses])
        return tuple(self.listed_sample(sequence, item, counts.get(item.item_id, {})) for sequence, item in uses)

    def listed_sample(self, sequence: int, item: StoredItem, train_counts: Mapping[str, int]) -> ListedSample:
        own = self.own_origin(item)
        return ListedSample(
            sequence,
            item.item_id,
            cast(SampleKind, item.kind),
            cast(SplitName, item.scenarios[self.scenario]),
            train_counts,
            None if own is None else own["component"],
            None if own is None else item.metadata.get("group"),
        )

    def own_origin(self, item: StoredItem) -> Mapping[str, Any] | None:
        """The origin of ``item`` when a stage of this scenario made it, else None."""
        origin = item.metadata.get("origin")
        return origin if isinstance(origin, Mapping) and origin.get("scenario") == self.scenario else None

    def get(self, sample_ids: Sequence[str]) -> tuple[StoredSample, ...]:
        """The samples of ``sample_ids`` that the scenario uses, in the order of their items. The store reads them
        ``MAX_QUERY_LIMIT`` at a time: a request can name tens of thousands of samples, and one read for each would
        hold the trainer's lock for minutes."""
        items = sorted(
            (
                item
                for batch in batched(dict.fromkeys(sample_ids), MAX_QUERY_LIMIT)
                for item in self.data.read_many(self.owner, batch)
                if self.scenario in item.scenarios
            ),
            key=lambda item: item.sequence,
        )
        return tuple(
            stored
            for batch in batched(items, MAX_QUERY_LIMIT)
            for stored in self.stored([(item.sequence, item) for item in batch])
        )

    def stored(self, uses: Sequence[tuple[int, StoredItem]]) -> tuple[StoredSample, ...]:
        """The samples of the uses ``uses`` that are trajectories or tasks, with their train counts."""
        samples = [(sequence, item) for sequence, item in uses if item.kind in SAMPLE_KINDS]
        counts = self.state.train_counts([item.item_id for _, item in samples])
        return tuple(self.stored_sample(sequence, item, counts.get(item.item_id, {})) for sequence, item in samples)

    def stored_sample(self, sequence: int, item: StoredItem, train_counts: Mapping[str, int]) -> StoredSample:
        """The sample of ``item``. An item that a stage of this scenario made trains for that stage's trainer, with
        its group, its sample's group and records, and, for a task, its sample's metadata."""
        split = cast(SplitName, item.scenarios[self.scenario])
        own = self.own_origin(item)
        records = () if own is None else tuple(own["records"])
        sample: TrajectorySample | TaskSample
        if item.kind == "trajectory":
            if item.content is None:
                raise ValueError(f"trajectory item {item.item_id!r} has no content")
            sample = TrajectorySample(
                item.content, group_id=item.metadata.get("group_id"), source_agent_record_ids=records
            )
        else:
            name = {"name": item.item_id, "files_digest": item.digest, "split": split}
            metadata = {"task": name} if own is None else item.metadata["task_metadata"]
            sample = TaskSample(self.data.task_path(item), source_agent_record_ids=records, metadata=metadata)
        component = None if own is None else own["component"]
        group = None if own is None else item.metadata.get("group")
        return StoredSample(sequence, item.item_id, split, sample, train_counts, component, group)

    def imported_tasks(self, split: SplitName | None = None) -> tuple[StoredSample, ...]:
        """Every task that the scenario uses and that no stage of it made, in use order; only ``split``'s if given."""
        found: list[StoredSample] = []
        after_sequence = 0
        while page := self.page(after_sequence=after_sequence, kind="task"):
            found.extend(stored for stored in page if stored.component is None and split in (None, stored.split))
            after_sequence = page[-1].sequence
        return tuple(found)


def memory_samples(scenario: str) -> SampleStore:
    """The sample store of ``scenario`` in process memory, for a trainer without a scenario session: a data store whose
    owner is the scenario, with its task files in a temporary directory, and a training state in memory."""
    return SampleStore(MemoryDataStore(LocalTaskFileStore()), scenario, scenario, MemoryTrainingState())


@dataclass(frozen=True)
class TrainerSamples:
    """The samples of a scenario as one trainer reads and writes them.

    Values: ``store`` is the scenario's ``SampleStore``. ``trainer`` is the trainer's ``train_counts`` key: the
    component that its commits name, or ``""`` for the only trainer of a scenario.

    Purpose: ``SampleStore.for_trainer`` makes it. The trainer's pipeline (``PipelineContext.samples``) and its stages
    read the samples that the trainer trains on, save the samples that its stages make, and read the records that
    they consumed.
    """

    store: SampleStore
    trainer: str

    def page(
        self, *, after_sequence: int = 0, limit: int = 256, kind: SampleKind | None = None
    ) -> tuple[StoredSample, ...]:
        """The samples that the trainer reads: its own, and the uploads and imported tasks (``SampleStore.page``)."""
        return self.store.page(after_sequence=after_sequence, limit=limit, kind=kind, trainer=self.trainer)

    def get(self, sample_ids: Sequence[str]) -> tuple[StoredSample, ...]:
        return self.store.get(sample_ids)

    def imported_tasks(self, split: SplitName | None = None) -> tuple[StoredSample, ...]:
        return self.store.imported_tasks(split)

    def add_record_samples(
        self,
        group: str | None,
        record_ids: Sequence[str],
        samples: Sequence[tuple[str, Sample]],
        *,
        lineage: Mapping[str, str] | None = None,
    ) -> int:
        """Save the samples that the stages of ``trainer`` made from ``record_ids`` as items that the scenario uses,
        then mark the records consumed; return the number of new items. ``lineage`` adds top-level metadata that
        names the sample processing tasks of the samples (``TrainingPipeline.lineage``), so a query by group finds
        them.

        The sample named ``name`` is the item ``stage_item_id(scenario, name)``. An item that exists stays as it is,
        so a stage that makes a sample again, after a restart, changes nothing. ``record_ids`` is empty for samples
        that a stage made from no record set.
        """
        store, trainer = self.store, self.trainer
        new = 0
        for name, sample in samples:
            item_id = stage_item_id(store.scenario, name)
            if store.data.read(store.owner, item_id) is not None:
                continue
            origin = {
                "scenario": store.scenario,
                "component": trainer,
                "records": list(sample.source_agent_record_ids),
            }
            metadata = {
                "scenarios": {store.scenario: "train"},
                "origin": origin,
                **({"group": group} if group is not None else {}),
                **(lineage or {}),
            }
            try:
                if isinstance(sample, TrajectorySample):
                    group_id = {} if sample.group_id is None else {"group_id": sample.group_id}
                    item = NewItem(item_id, dict(sample.trajectory), {"kind": "trajectory", **metadata, **group_id})
                    store.data.create(store.owner, [item])
                else:
                    with tempfile.TemporaryDirectory(prefix="reef-stage-task-") as directory:
                        task = shutil.copytree(sample.task_path, Path(directory) / item_id)
                        task_metadata = {"kind": "task", **metadata, "task_metadata": dict(sample.metadata)}
                        store.data.create(store.owner, [NewItem(item_id, task, task_metadata)])
            except ItemConflict:
                # Another process made the same sample first: its item wins.
                continue
            new += 1
        store.state.consume_records(trainer, record_ids)
        return new

    def consumed_record_ids(self) -> frozenset[str]:
        """The records that the stages of ``trainer`` consumed."""
        return frozenset(
            record_id
            for record_id in self.store.state.consumed_record_ids(self.trainer)
            if not record_id.startswith(ENDED_RECORD_PREFIX)
        )

    def end_records(self, record_ids: Iterable[str]) -> None:
        """Store that the stages of ``trainer`` ended ``record_ids``, for its next commit to name.

        An ended record is not consumed until a commit names it: a restart before that commit reads it again, so a
        stage rebuilds what it decided about the record.
        """
        self.store.state.consume_records(
            self.trainer, [f"{ENDED_RECORD_PREFIX}{record_id}" for record_id in sorted(record_ids)]
        )

    def ended_record_ids(self) -> frozenset[str]:
        """The records that the stages of ``trainer`` ended (``end_records``), whether a commit names them or not."""
        return frozenset(
            record_id.removeprefix(ENDED_RECORD_PREFIX)
            for record_id in self.store.state.consumed_record_ids(self.trainer)
            if record_id.startswith(ENDED_RECORD_PREFIX)
        )


__all__ = [
    "CONSUMED_SAMPLE_PREFIX",
    "SAMPLE_KINDS",
    "ListedSample",
    "MemoryTrainingState",
    "SampleKind",
    "SampleStore",
    "StoredSample",
    "TrainerSamples",
    "TrainingState",
    "consumed_sample_id",
    "memory_samples",
    "stage_item_id",
]
