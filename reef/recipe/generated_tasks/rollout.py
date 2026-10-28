"""The generated-task task jobs: the task pool, the tasks handed out to one job, and the IDs its commit lists.

The training component rolls the tasks of a task job out itself; Reef runs no rollout of them. One job's
tasks are handed out at a time:

1. When no tasks are handed out, ``GeneratedTaskJobData`` takes up to ``tasks_per_step`` tasks from its
   ``TaskPool`` as the next job's data (``Sample2Job.own_job_data``). Each task gets a task use ``u``, distinct
   within the hand-out.
2. When the trainer commits the job's submission, the commit lists ``rollout:<task use>``; each task is
   marked trained once that ID is released after the commit is durable.
3. When the job's end is durable, the tasks leave the pool, and the stage saves each task's first result
   from the job's metrics.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from reef.core.job_data import TaskSample

#: The prefix of the ID that a commit lists for a task of a submitted task job.
CONSUMED_ROLLOUT_PREFIX = "rollout:"


def consumed_rollout_id(task_use: str) -> str:
    """Return the ID that a commit lists for the trained group of ``task_use``."""
    return f"{CONSUMED_ROLLOUT_PREFIX}{task_use}"


@dataclass(frozen=True)
class PooledTask:
    """A saved task that waits for a task job.

    Values: ``task_id`` identifies the task in its pool. ``arrival_index``
    orders the tasks of a pool. ``group_key`` names what the task came from,
    such as its generation. ``item`` is the task that the job rolls out.
    Purpose: a ``TaskPool`` holds it; the sample-to-job stage takes it from the
    pool for a task job, and drops it when the job ends.
    """

    task_id: Hashable
    arrival_index: int
    group_key: Hashable | None
    item: TaskSample


class TaskPool:
    """The saved tasks that wait for a task job, and the ones that a task job took.

    Values: ``waiting`` and ``taken`` are disjoint finite sets of ``PooledTask`` values with distinct
    ``task_id``; ``arrival_count`` is larger than the ``arrival_index`` of each task that ``add`` made.
    ``take`` moves the oldest waiting tasks by ``arrival_index`` to ``taken``; ``drop`` removes taken tasks;
    ``add`` gives a new task a new index.
    Purpose: ``GeneratedTaskJobData`` holds it. The record stage adds each round's tasks; the sample-to-job
    stage takes a job's tasks from it, and drops them when the job ends.
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
        """Whether no task waits and no task job holds one."""
        return not self.waiting and not self.taken

    def add(self, task_id: Hashable, item: TaskSample, *, group_key: Hashable | None = None) -> None:
        """Add a task after the waiting ones. A task with a waiting ID replaces it; a taken ID is refused."""
        if task_id in self.taken:
            raise ValueError(f"task {task_id!r} is in a task job")
        self.arrival_count += 1
        self.waiting[task_id] = PooledTask(task_id, self.arrival_count, group_key, item)

    def take(self, max_count: int) -> tuple[PooledTask, ...]:
        """Move at most ``max_count`` waiting tasks to ``taken``, oldest first, and return them."""
        taken = tuple(sorted(self.waiting.values(), key=lambda task: task.arrival_index)[:max_count])
        for task in taken:
            self.taken[task.task_id] = self.waiting.pop(task.task_id)
        return taken

    def drop(self, tasks: tuple[PooledTask, ...]) -> None:
        """Remove taken tasks: their task job ended."""
        for task in tasks:
            self.taken.pop(task.task_id)


@dataclass(frozen=True)
class HandedOutTasks:
    """The tasks of the next task job, from hand-out until the trainer commits the job's submission.

    Values: ``batch`` is the hand-out's ID, ``<scenario>:rollout:<hex>``. ``task_uses`` are its task uses,
    ``<batch>/<task name>``, distinct, one for each of ``tasks``, the pooled tasks, in the same order.
    Purpose: ``GeneratedTaskJobData`` makes it when it takes the tasks from the pool, and reads it when the
    submission commits.
    """

    batch: str
    task_uses: tuple[str, ...]
    tasks: tuple[PooledTask, ...]

    def __post_init__(self) -> None:
        if len(self.task_uses) != len(self.tasks) or len(set(self.task_uses)) != len(self.task_uses):
            raise ValueError("handed-out tasks have one distinct task use for each task")


@dataclass(frozen=True)
class SubmittedTaskJob:
    """The tasks of one submitted task job, until the commit of its submission is durable.

    Values: ``task_names`` are the distinct, non-empty names of the job's tasks, and ``consumed_ids`` the
    non-empty IDs that the submission's commit lists for them. Purpose: ``GeneratedTaskJobData`` makes it when
    the submission commits, and marks ``task_names`` trained once one of ``consumed_ids`` is released.
    """

    task_names: tuple[str, ...]
    consumed_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.task_names or len(set(self.task_names)) != len(self.task_names) or not all(self.task_names):
            raise ValueError("a submitted task job has distinct, non-empty task names")
        if not self.consumed_ids:
            raise ValueError("a submitted task job's commit lists one or more IDs")
