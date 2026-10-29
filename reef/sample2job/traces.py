"""Job data of harness evolution: trace samples as they come, and offline rollouts of the imported train tasks.

``TraceJobData`` puts the trace samples of the selected record sets into one
job's data. ``TrainTaskRollouts`` also hands out job data of its own: the
next train tasks of the scenario's ``imported-tasks`` store, which the
harness backend rolls out on the head tree as the step's material. That is
offline harness evolution: a step needs no traffic, so uploaded train tasks
start a run.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core import RequestType
from reef.core.config import config_option
from reef.core.job_data import Sample, TaskSample
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2sample import Record2Sample
from reef.sample2job.base import Sample2JobSettings, imported_task_sample
from reef.sample2job.distill import DistillJobData
from reef.storage.tasks import IMPORTED_TASK_STORE

#: The task store whose documents keep where the run of ``TrainTaskRollouts`` stands. A composite scenario opens
#: it under each component, so each component's run keeps its own place.
ROLLOUT_PROGRESS_STORE = "harness-rollouts"
#: The key prefixes of the progress documents: the size of each started epoch, and the end of each saved chunk.
EPOCH_KEY_PREFIX = "epoch/"
DONE_KEY_PREFIX = "done/"
#: The prefix of the ID that the commit of an offline step lists.
ROLLOUT_ID_PREFIX = "harness-rollout:"


class TraceJobData(DistillJobData):
    """Put trace samples into job data unmodified; the backends consume them without more processing.

    In ``manual`` a queued instruction takes the traces that automatic job
    data would take next, up to ``job_samples`` and possibly none, so the
    method reads the request beside them.
    """

    reads_instructions = True
    required_request_types = frozenset(RequestType)


@dataclass(frozen=True)
class TrainTaskRolloutSettings(Sample2JobSettings):
    """The config of ``TrainTaskRollouts``: the sample buffer of ``Sample2JobSettings``, and the rollout plan.

    Values: ``tasks_per_step`` ≥ 1 is the most train tasks one offline step
    rolls out; ``epochs`` ≥ 1 is how many times the run goes through the
    train split; ``shuffle_seed`` is an int that seeds each epoch's order.
    The constructor refuses other values.
    Purpose: ``TrainTaskRollouts.from_config`` parses it from
    ``pipeline.sample2job.config``.
    """

    tasks_per_step: int = config_option(8, help="The most train tasks one offline step rolls out on the head tree.")
    epochs: int = config_option(1, help="How many times the run goes through the train split.")
    shuffle_seed: int = config_option(0, help="The seed of each epoch's task order.")

    def __post_init__(self) -> None:
        super().__post_init__()
        for label, value in (("tasks_per_step", self.tasks_per_step), ("epochs", self.epochs)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be an integer of at least 1")
        if isinstance(self.shuffle_seed, bool) or not isinstance(self.shuffle_seed, int):
            raise ValueError("shuffle_seed must be an integer")


@dataclass(frozen=True)
class RolloutChunk:
    """The train tasks of one offline step.

    Values: ``epoch`` ≥ 0 and ``start`` ≥ 0 place the chunk in the run, and
    ``samples`` holds one or more task samples: the tasks at positions
    ``start`` to ``end - 1`` of epoch ``epoch``'s order.
    Purpose: ``TrainTaskRollouts`` hands one out as its own job data, and
    saves its end once the commit of its step is durable.
    """

    epoch: int
    start: int
    samples: tuple[TaskSample, ...]

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.start < 0 or not self.samples:
            raise ValueError("a rollout chunk has an epoch and a start of at least 0, and one or more samples")

    @property
    def end(self) -> int:
        """The position after the chunk's last task in its epoch's order."""
        return self.start + len(self.samples)

    @property
    def rollout_id(self) -> str:
        """The ID that the commit of the chunk's step lists: ``harness-rollout:<epoch>:<start>``."""
        return f"{ROLLOUT_ID_PREFIX}{self.epoch}:{self.start}"


class TrainTaskRollouts(TraceJobData):
    """Hand out the scenario's imported train tasks, chunk by chunk and epoch by epoch, as offline steps' job data.

    Values: ``store`` is the scenario's ``imported-tasks`` store and
    ``progress_store`` the stage's ``harness-rollouts`` store; both are
    ``None`` for a trainer without task storage, which hands out nothing.
    ``epoch_sizes`` maps each started epoch to the number of train tasks it
    covers: the train tasks published when the epoch started, which are the
    first ones in publication order. A task published later joins the next
    epoch. An epoch's order is its tasks shuffled with ``shuffle_seed`` and
    the epoch. ``epoch`` ≥ 0 and ``position`` ≥ 0 are where the run stands:
    the next chunk starts at ``position`` of epoch ``epoch``'s order, and the
    run ends when ``epoch`` reaches ``epochs``. ``handed_out`` is the chunk of
    the next step until the trainer prepares the step's commit; ``submitted``
    holds the chunks whose commit is not durable yet.
    Purpose: a deployment names the class in ``pipeline.sample2job`` for
    offline harness evolution. Traffic still makes steps, as with
    ``TraceJobData``, and goes first. The stage saves the size of each
    started epoch, and the end of each chunk once the commit of its step is
    durable, as documents of ``progress_store``. So a restart resumes after
    the last saved chunk, and a chunk whose end was not saved runs again.
    """

    # The stage trains on its own task chunks, and a rollout step has no instruction: it runs in auto only.
    supported_training_modes = frozenset({"auto"})
    accepts_imported_tasks = True

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, TrainTaskRolloutSettings.parse(config, plugin_class=cls))

    def __init__(self, context: PipelineContext, settings: TrainTaskRolloutSettings) -> None:
        super().__init__(context, settings.buffer_size, settings.job_samples)
        self.tasks_per_step = settings.tasks_per_step
        self.epochs = settings.epochs
        self.shuffle_seed = settings.shuffle_seed
        storage = context.task_storage
        self.store = None if storage is None else storage.open(IMPORTED_TASK_STORE)
        self.progress_store = None if storage is None else storage.open(ROLLOUT_PROGRESS_STORE)
        self.epoch_sizes: dict[int, int] = {}
        self.epoch = 0
        self.position = 0
        self.handed_out: RolloutChunk | None = None
        self.submitted: list[RolloutChunk] = []
        if self.progress_store is not None:
            self.restore(self.progress_store.documents())

    def restore(self, documents: Mapping[str, str]) -> None:
        """Read the saved epoch sizes, and move to the end of the last saved chunk."""
        done: list[tuple[int, int]] = []
        for key, document in documents.items():
            if key.startswith(EPOCH_KEY_PREFIX):
                self.epoch_sizes[int(key.removeprefix(EPOCH_KEY_PREFIX))] = int(json.loads(document)["size"])
            elif key.startswith(DONE_KEY_PREFIX):
                row = json.loads(document)
                done.append((int(row["epoch"]), int(row["end"])))
        if done:
            self.epoch, self.position = max(done)
            if self.position >= self.epoch_sizes[self.epoch]:
                self.epoch, self.position = self.epoch + 1, 0

    def advance(self) -> None:
        """Hand out the next chunk when none is handed out and the run has train tasks left.

        An epoch starts with the train tasks published by then, and saves its size first.
        """
        super().advance()
        if self.handed_out is not None or self.epoch >= self.epochs:
            return
        if self.store is None or self.progress_store is None:
            return
        train = self.store.tasks(split="train")
        size = self.epoch_sizes.get(self.epoch)
        if size is None:
            if not train:
                return
            size = len(train)
            self.progress_store.write_document(
                key=f"{EPOCH_KEY_PREFIX}{self.epoch}", document=json.dumps({"size": size})
            )
            self.epoch_sizes[self.epoch] = size
        order = list(train[:size])
        random.Random(f"{self.shuffle_seed}:{self.epoch}").shuffle(order)
        chunk = order[self.position : self.position + self.tasks_per_step]
        self.handed_out = RolloutChunk(
            self.epoch, self.position, tuple(imported_task_sample(self.store, stored) for stored in chunk)
        )

    def own_job_data(self) -> tuple[Sample, ...] | None:
        return None if self.handed_out is None else self.handed_out.samples

    def own_job_data_submitted(self) -> frozenset[str]:
        chunk = self.handed_out
        if chunk is None:
            raise RuntimeError("no train tasks were handed out")
        self.submitted.append(chunk)
        self.handed_out = None
        self.position = chunk.end
        if self.position >= self.epoch_sizes[chunk.epoch]:
            self.epoch, self.position = chunk.epoch + 1, 0
        return frozenset({chunk.rollout_id})

    def release_own_ids(self, ids: frozenset[str]) -> None:
        """Save the end of each submitted chunk that ``ids`` names: the commit of its step is durable."""
        durable = [chunk for chunk in self.submitted if chunk.rollout_id in ids]
        if self.progress_store is not None:
            for chunk in durable:
                self.progress_store.write_document(
                    key=f"{DONE_KEY_PREFIX}{chunk.epoch}/{chunk.end}",
                    document=json.dumps({"epoch": chunk.epoch, "end": chunk.end}),
                )
        self.submitted = [chunk for chunk in self.submitted if chunk not in durable]

    def status(self) -> Mapping[str, Any]:
        return {**super().status(), "rollout_epoch": self.epoch, "rollout_position": self.position}


__all__ = [
    "ROLLOUT_PROGRESS_STORE",
    "RolloutChunk",
    "TraceJobData",
    "TrainTaskRolloutSettings",
    "TrainTaskRollouts",
]
