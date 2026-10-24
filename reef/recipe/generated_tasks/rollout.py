"""The generated-task rollout: Reef runs saved tasks itself and trains on their episodes.

A rollout episode is not a record. Its agent's model calls carry ``ROLLOUT_HEADER``
(:mod:`reef.core.records_types`), so the service keeps no record of them, and the episode comes back with its
answered calls. ``GeneratedTaskBatches`` owns the whole path from a saved task to a training sample, and gives
the pipeline its samples as its own batch (``Task2Batch.own_batch``); the record stages never see a rollout
episode. One rollout batch runs at a time:

1. When no rollout batch runs, the stage takes up to ``tasks_per_step`` tasks from its ``TaskPool`` and
   submits them. Each submitted task gets a task use ``u``, distinct within the batch.
2. When the batch's episodes end, ``samples : U ⇀ TrajectoryItem⁺`` gives the samples of each task use whose
   episodes can train together, its group. A use without a group is absent.
3. The stage drops each group whose samples a version produced that is more than ``max_staleness`` versions
   behind the version the scenario serves. The groups that are left, ``fresh_uses``, are the own batch.
4. A task whose group did not train, because it was stale or the backend dropped the batch, goes back to the
   pool. A task without a group stays out of the pool until a restart. A trained task's commit lists
   ``rollout:<task use>``, and the task is marked trained once that ID is released after the commit is durable.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass

from reef.core.batches import TaskItem, TrajectoryItem

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
class ActiveRollout:
    """The rollout batch that runs, or that waits for its training.

    Values: ``task_uses`` are the batch's task uses, and ``tasks`` the pooled tasks that they run, in the same
    order. ``samples`` is None while the episodes run; then it maps each task use that has a group to its
    samples, a non-empty tuple. ``fresh_uses`` are the uses, in submit order, whose groups passed the
    staleness check; they train.
    Purpose: ``GeneratedTaskBatches`` keeps it from submit until the batch trains or ends.
    """

    batch: str
    task_uses: tuple[str, ...]
    tasks: tuple[PooledTask, ...]
    samples: Mapping[str, tuple[TrajectoryItem, ...]] | None = None
    fresh_uses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.task_uses) != len(self.tasks):
            raise ValueError("a rollout batch has one task use for each task")
        if self.samples is not None and any(not group for group in self.samples.values()):
            raise ValueError("a rollout group holds at least one sample")
        if not set(self.fresh_uses) <= set(self.samples or {}):
            raise ValueError("a fresh task use has a group")
