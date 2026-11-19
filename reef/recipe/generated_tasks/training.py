"""How the generated-task stage trains its published train tasks: once each, or again from a training pool.

``GeneratedTaskSamples`` paces the rounds, saves the first result of each train task, and hands the samples of the
tasks to the pipeline. A ``TaskTraining`` decides the rest: whether a round has room for more tasks, which uses of
the tasks the end of a task job ends, and which samples come after it. ``SingleUseTraining`` trains each train task
once. :class:`reef.recipe.generated_tasks.pool.TrainingPool` trains them again until a review retires them.

Both read the samples of the tasks: the published train tasks of complete generations (``published_train_tasks``),
the sample of one use of a task (``task_item``, named by ``sample_name``), and the samples that the scenario's items
hold for the trainer (``task_items``).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from reef.core.atif_payloads import canonical_json
from reef.core.job_data import TaskSample
from reef.core.tasks import read_harbor_task
from reef.core.values import require_integer, require_text
from reef.recipe.generated_tasks.generator import check_rewards
from reef.storage.samples import StoredSample, TrainerSamples
from reef.storage.tasks import GenerationState, StoredTask, TaskStore

#: The task store of the stage, and the prefix of the names of the samples that it makes.
TASK_STORE_NAME = "generated-tasks"
#: The key prefix of the documents of the task uses that a training pool saves.
USE_KEY_PREFIX = "use/"
#: The samples of each generated task that the scenario's items hold for one trainer, by task name, in use order.
TaskItems = Mapping[str, tuple[StoredSample, ...]]


def train_task_names(state: GenerationState) -> tuple[str, ...]:
    """The names of the train tasks that a generation published, in candidate order."""
    return tuple(candidate.name for candidate in state.candidates or () if candidate.split == "train")


def published_train_tasks(store: TaskStore) -> tuple[tuple[int, StoredTask], ...]:
    """The published train tasks of complete generations, with their generation, oldest first."""
    published = {stored.name: stored for stored in store.tasks(split="train")}
    return tuple(
        (state.generation, published[name])
        for state in store.generations()
        if state.is_complete
        for name in train_task_names(state)
        if name in published
    )


def task_item(store: TaskStore, generation: int, stored: StoredTask, *, use_index: int = 0) -> TaskSample:
    """The sample of a published train task, with its generation and the digest of its files.

    ``use_index`` is the use of the task that the sample trains, the ``use_index`` of its ``TaskUse``: 0 for the first
    use, and ``k`` for use ``k`` of a training pool, which the sample names as ``task.use_index``.
    """
    path = store.path(stored.files_digest)
    task = read_harbor_task(path)
    identity: dict[str, object] = {
        "name": stored.name,
        "digest": task.digest,
        "files_digest": stored.files_digest,
        "round": generation,
        "split": stored.split,
    }
    if use_index > 0:
        identity["use_index"] = use_index
    return TaskSample(
        path,
        source_agent_record_ids=task.source_agent_record_ids,
        metadata={"task": identity},
        files=store.task_files(stored.files_digest),
    )


def sample_name(task: TaskSample) -> str:
    """The name under which the stage makes the sample of a train task: ``generated-tasks/<task name>``, and
    ``generated-tasks/<task name>@<k>`` for use ``k`` of a training pool. A task name holds no ``@``, so two
    samples never share a name."""
    identity = task.metadata["task"]
    use_index = identity.get("use_index", 0)
    name = f"{TASK_STORE_NAME}/{identity['name']}"
    return name if use_index == 0 else f"{name}@{use_index}"


def sample_use_index(stored: StoredSample) -> int:
    """The use of its task that a stored generated-task sample trains: 0 for the first use (``task_item``)."""
    use_index: int = stored.sample.metadata["task"].get("use_index", 0)
    return use_index


def task_items(samples: TrainerSamples) -> dict[str, tuple[StoredSample, ...]]:
    """The generated tasks that the scenario's items hold for the trainer of ``samples``, by task name: the samples of
    each task that its stage made, in use order. Without a training pool, each task has one sample."""
    found: dict[str, list[StoredSample]] = {}
    after_sequence = 0
    while page := samples.page(after_sequence=after_sequence, kind="task"):
        for stored in page:
            if stored.component == samples.trainer:
                found.setdefault(stored.sample.metadata["task"]["name"], []).append(stored)
        after_sequence = page[-1].sequence
    return {name: tuple(sorted(stored, key=sample_use_index)) for name, stored in found.items()}


@dataclass(frozen=True)
class TaskUse:
    """One use of a train task in a task job: the rewards of the task's rollouts in that job.

    Values: ``task_name`` and ``job_id`` are non-empty strings. ``use_index`` is a non-negative int, the use of the
    task whose sample trained in the job: 0 for the task's first sample, and ``k`` for its sample of use ``k``. The
    uses of one task have distinct ``use_index`` values. ``rewards`` is a tuple of finite numbers in run order; it
    is empty when no rollout of the task ran.
    Purpose: ``TaskTraining.ended_uses`` makes one for each task use that the end of a task job ends. The stage saves
    the first result of its task from the rewards; a training pool saves it under ``use/<task_name>/<job_id>``,
    reviews the rewards, and makes the task's sample of the next use.
    """

    task_name: str
    job_id: str
    use_index: int
    rewards: tuple[float, ...]

    def __post_init__(self) -> None:
        require_text(self.task_name, "task_name")
        require_text(self.job_id, "job_id")
        require_integer(self.use_index, "a task use's use_index")
        if not isinstance(self.rewards, tuple):
            raise TypeError("a task use's rewards must be a tuple of finite numbers")
        if self.rewards:
            check_rewards(self.rewards)

    @property
    def key(self) -> str:
        """The key of the use's document."""
        return f"{USE_KEY_PREFIX}{self.task_name}/{self.job_id}"

    def document(self) -> str:
        """The use as canonical JSON text."""
        return canonical_json(
            {
                "task_name": self.task_name,
                "job_id": self.job_id,
                "use_index": self.use_index,
                "rewards": list(self.rewards),
            }
        )

    @classmethod
    def from_document(cls, text: str) -> TaskUse:
        """Read the text of :meth:`document`."""
        fields = json.loads(text)
        return cls(fields["task_name"], fields["job_id"], fields["use_index"], tuple(fields["rewards"]))


@dataclass(frozen=True)
class NextSamples:
    """What the end of a task job changes in the samples that the stage hands to the pipeline.

    Values: ``samples`` is a tuple of the samples of the next uses of train tasks, in the order that the stage hands
    them out. ``withdrawn`` is a frozenset of the names of the tasks that do not train again: the stage drops each of
    their samples that the pipeline has not stored. No sample of ``samples`` is of a task in ``withdrawn``. The
    constructor refuses other values.
    Purpose: ``TaskTraining.reuse_tasks`` makes it, and ``GeneratedTaskSamples.job_ended`` reads it.
    """

    samples: tuple[TaskSample, ...] = ()
    withdrawn: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple) or not all(isinstance(task, TaskSample) for task in self.samples):
            raise TypeError("the next samples must be a tuple of task samples")
        if not isinstance(self.withdrawn, frozenset):
            raise TypeError("the withdrawn tasks must be a frozenset of task names")
        if any(task.metadata["task"]["name"] in self.withdrawn for task in self.samples):
            raise ValueError("a withdrawn task has no next sample")


class TaskTraining(ABC):
    """How often the generated-task stage trains each published train task, and when a round has room for more.

    An implementation keeps what it needs to decide; the stage passes what it holds itself: the names of the
    unfinished train tasks (the published train tasks without a first result), and the task samples of the
    scenario's items (``task_items``).
    Purpose: ``GeneratedTaskSettings.task_training`` makes one for the stage of a scenario. ``GeneratedTaskSamples``
    asks it which samples a start adds (``start_samples``), whether a round is due (``is_idle``, ``has_room``), which
    uses the end of a task job ends (``ended_uses``), and what comes after the stage saved their first results
    (``reuse_tasks``). It tells it when a round published train tasks (``tasks_published``), and adds its
    ``status`` to the stage's status.
    """

    @abstractmethod
    def start_samples(self, items: TaskItems, results: frozenset[str]) -> tuple[TaskSample, ...]:
        """The samples that a start makes besides the first sample of each published train task that ``items`` lacks.

        ``items`` is ``task_items`` at the start, and ``results`` names the tasks that have a first result.
        """

    @abstractmethod
    def is_idle(self, unfinished: AbstractSet[str]) -> bool:
        """Whether no train task waits for a job or trains in one, so no task job's end comes."""

    @abstractmethod
    def has_room(self, unfinished: AbstractSet[str], samples: TrainerSamples | None) -> bool:
        """Whether a round may add train tasks now. ``samples`` is the scenario's samples as the trainer reads them,
        or None when the stage has none."""

    @abstractmethod
    def ended_uses(
        self, job_id: str, rewards: Mapping[str, Sequence[float]], items: TaskItems, trainer: str
    ) -> tuple[TaskUse, ...]:
        """The task uses that the end of the task job ``job_id`` ends, each with its task's rewards in ``rewards``,
        the job's ``rollout_rewards``, or no reward when ``rewards`` does not name the task. ``items`` is
        ``task_items`` at the job's end, and ``trainer`` the trainer's ``train_counts`` key. It saves nothing."""

    @abstractmethod
    def reuse_tasks(self, uses: Sequence[TaskUse], items: TaskItems, samples: TrainerSamples) -> NextSamples:
        """What follows ``uses`` (``ended_uses``) once the stage saved their first results. ``items`` is
        ``task_items`` at the job's end. The end of the same job again makes the same samples, and saves nothing
        that it saved already."""

    @abstractmethod
    def tasks_published(self) -> None:
        """A round ended and published train tasks."""

    @abstractmethod
    def status(self) -> Mapping[str, object]:
        """The keys that the stage adds to its status."""


class SingleUseTraining(TaskTraining):
    """Each published train task trains once: the end of its job ends its only use.

    Values: ``refill_tasks`` is a positive int: a round has room only while fewer train tasks than this wait for a
    job. The constructor refuses other values.
    Purpose: ``GeneratedTaskSettings.task_training`` makes it when ``pool_capacity`` is 0.
    """

    def __init__(self, refill_tasks: int) -> None:
        require_integer(refill_tasks, "refill_tasks", minimum=1)
        self.refill_tasks = refill_tasks

    def start_samples(self, items: TaskItems, results: frozenset[str]) -> tuple[TaskSample, ...]:
        return ()

    def is_idle(self, unfinished: AbstractSet[str]) -> bool:
        """Whether no train task is unfinished: a finished task does not train again."""
        return not unfinished

    def has_room(self, unfinished: AbstractSet[str], samples: TrainerSamples | None) -> bool:
        """Whether fewer than ``refill_tasks`` unfinished train tasks wait: no committed job trained any of their
        samples."""
        trainer = "" if samples is None else samples.trainer
        items = {} if samples is None else task_items(samples)
        waiting = sum(all(stored.train_count(trainer) == 0 for stored in items.get(name, ())) for name in unfinished)
        return waiting < self.refill_tasks

    def ended_uses(
        self, job_id: str, rewards: Mapping[str, Sequence[float]], items: TaskItems, trainer: str
    ) -> tuple[TaskUse, ...]:
        """The use of each task that a committed job trained: the tasks of the ended job, of an earlier job that a new
        training start cancelled, and of a job that started before the stage read this end."""
        return tuple(
            TaskUse(name, job_id, 0, tuple(rewards.get(name) or ()))
            for name, stored in items.items()
            if any(sample.train_count(trainer) > 0 for sample in stored)
        )

    def reuse_tasks(self, uses: Sequence[TaskUse], items: TaskItems, samples: TrainerSamples) -> NextSamples:
        return NextSamples()

    def tasks_published(self) -> None:
        return None

    def status(self) -> Mapping[str, object]:
        return {}


__all__ = [
    "TASK_STORE_NAME",
    "NextSamples",
    "SingleUseTraining",
    "TaskItems",
    "TaskTraining",
    "TaskUse",
    "published_train_tasks",
    "sample_name",
    "sample_use_index",
    "task_item",
    "task_items",
    "train_task_names",
]
