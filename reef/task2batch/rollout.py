"""Rollout: Reef runs saved tasks itself and trains on their episodes.

A rollout episode is not a record. Its agent's model calls carry ``ROLLOUT_HEADER``
(:mod:`reef.core.records_types`), so the service keeps no record of them, and the episode comes back with its
answered calls. A task-to-batch stage that rolls tasks out (``RolloutTask2Batch``) owns the whole path from a
saved task to a training sample; the record stages never see a rollout episode.

The pipeline (``reef.train.pipeline.TrainingPipeline``) runs one rollout batch at a time:

1. It takes up to ``tasks_per_step`` tasks from the stage's ``TaskPool`` and submits them. Each submitted task
   gets a task use ``u``, distinct within the batch.
2. When the batch's episodes end, ``take_finished`` gives ``samples : U ⇀ TrajectoryItem⁺``: the training
   samples of each task use whose episodes can train together, its group. A use without a group is absent.
3. The pipeline drops each group whose samples a version produced that is too far behind the version the
   scenario serves. The surviving groups train as one batch.
4. A task whose group did not train, because it was stale or the backend dropped the batch, goes back to the
   pool. A task without a group stays out of the pool until a restart. A trained task is marked trained once
   the commit of its step is durable (``rollout_trained``).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.batches import TaskItem, TrajectoryItem
from reef.task2batch.base import Task2Batch

#: The prefix of the ID that a commit lists for a trained rollout group.
CONSUMED_ROLLOUT_PREFIX = "rollout:"


def consumed_rollout_id(task_use: str) -> str:
    """Return the ID that a commit lists for the trained group of ``task_use``."""
    return f"{CONSUMED_ROLLOUT_PREFIX}{task_use}"


@dataclass(frozen=True)
class PooledTask:
    """A saved task that waits to roll out.

    Values: ``task_id`` identifies the task in its pool. ``arrival_index``
    orders the tasks of a pool. ``group_key`` names what the task came from,
    such as its generation. ``item`` is the task that a rollout runs.
    Purpose: a ``TaskPool`` holds it; the pipeline takes it from the pool to
    submit it in a rollout batch, and puts it back when the batch did not train it.
    """

    task_id: Hashable
    arrival_index: int
    group_key: Hashable | None
    item: TaskItem


class TaskPool:
    """The saved tasks that wait to roll out, and the ones that a rollout batch took.

    Values: ``waiting`` and ``taken`` are disjoint finite sets of ``PooledTask`` values with distinct
    ``task_id``; ``arrival_count`` is larger than the ``arrival_index`` of each task that ``add`` made.
    ``take`` moves the oldest waiting tasks by ``arrival_index`` to ``taken``; ``put_back`` moves taken tasks
    back with their indices; ``drop`` removes taken tasks; ``add`` gives a new task a new index.
    Purpose: a rollout task-to-batch stage holds it. The stage that makes tasks adds them; the pipeline takes
    a rollout batch's tasks from it, and puts back or drops them when the batch ends.
    """

    def __init__(self) -> None:
        self.waiting: dict[Hashable, PooledTask] = {}
        self.taken: dict[Hashable, PooledTask] = {}
        self.arrival_count = 0

    def __len__(self) -> int:
        """The number of waiting tasks."""
        return len(self.waiting)

    @property
    def is_idle(self) -> bool:
        """Whether no task waits and no rollout batch holds one."""
        return not self.waiting and not self.taken

    def add(self, task_id: Hashable, item: TaskItem, *, group_key: Hashable | None = None) -> None:
        """Add a task after the waiting ones. A task with a waiting ID replaces it; a taken ID is refused."""
        if task_id in self.taken:
            raise ValueError(f"task {task_id!r} is in a rollout batch")
        self.arrival_count += 1
        self.waiting[task_id] = PooledTask(task_id, self.arrival_count, group_key, item)

    def take(self, max_count: int) -> tuple[PooledTask, ...]:
        """Move at most ``max_count`` waiting tasks to ``taken``, oldest first, and return them."""
        taken = tuple(sorted(self.waiting.values(), key=lambda task: task.arrival_index)[:max_count])
        for task in taken:
            self.taken[task.task_id] = self.waiting.pop(task.task_id)
        return taken

    def put_back(self, tasks: tuple[PooledTask, ...]) -> None:
        """Return taken tasks to the waiting ones, in their old arrival order."""
        for task in tasks:
            self.waiting[task.task_id] = self.taken.pop(task.task_id)

    def drop(self, tasks: tuple[PooledTask, ...]) -> None:
        """Remove taken tasks: they trained, or they cannot train before a restart."""
        for task in tasks:
            self.taken.pop(task.task_id)


@dataclass(frozen=True)
class RolloutResult:
    """The episodes of one rollout batch ended.

    Values: ``batch`` names the rollout batch. ``samples`` maps a task use of the batch to the non-empty
    tuple of training samples of its group; a use without a group, such as one whose episodes failed or that
    has too few episodes to compare, is absent.
    Purpose: ``RolloutTask2Batch.take_finished`` makes it; the pipeline batches the groups that are fresh.
    """

    batch: str
    samples: Mapping[str, tuple[TrajectoryItem, ...]]

    def __post_init__(self) -> None:
        if any(not group for group in self.samples.values()):
            raise ValueError("a rollout group holds at least one sample")


class RolloutTask2Batch(Task2Batch):
    """A task-to-batch stage that rolls saved tasks out and batches the samples of their episodes.

    Values: ``tasks_per_step`` is the most tasks of one rollout batch, a positive int; ``task_pool`` holds the
    tasks that wait to roll out. The pipeline takes tasks from the pool and submits them as one rollout batch.
    The stage runs them on a worker that it owns, and ``take_finished`` hands back each task use's samples
    when the episodes end. The pipeline batches the fresh groups through ``make_batch``. ``close`` stops the
    worker.
    """

    #: The most saved tasks of one rollout batch.
    tasks_per_step: int = 1
    #: The saved tasks that wait to roll out.
    task_pool: TaskPool

    @abstractmethod
    def submit(self, batch: str, tasks: tuple[TaskItem, ...]) -> tuple[str, ...]:
        """Start running the tasks as one rollout batch; return one task use ID for each task, in order."""

    @abstractmethod
    def take_finished(self) -> tuple[RolloutResult, ...]:
        """Return the rollout batches whose episodes ended since the last call."""

    @abstractmethod
    def running(self) -> bool:
        """Whether a rollout batch is running."""

    @abstractmethod
    def rollout_trained(self, batch: str, task_uses: tuple[str, ...]) -> None:
        """The commit of the step that trained these task uses' groups is durable; their tasks do not roll out again."""

    def rollout_ended(self, batch: str, *, trained: bool) -> None:
        """The rollout batch ended: it trained, or it trained nothing. The default does nothing."""
        return

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {"task_pool": len(self.task_pool)}

    def close(self) -> None:
        """Stop the rollout worker. Calling it more than one time is safe."""
        return
