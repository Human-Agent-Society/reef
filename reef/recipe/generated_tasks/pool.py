"""The training pool of the generated-task recipe: each use of a train task, the reviews, and the retired tasks.

With a ``pool_capacity`` above 0, the stage trains its tasks with a ``TrainingPool``: a train task trains again.
When its task job ends, the record-to-sample stage makes the task's next sample, which waits after the samples that
wait already. The pool keeps three more values in the task store's documents, besides those of
:mod:`reef.recipe.generated_tasks.results`. Each document is written once:

- ``use/<n>/<job>``: the ``TaskUse`` of train task ``n`` in the task job ``job``, with its rewards in that job.
- ``review/<k>``: the ``PoolReview`` of review ``k``, for ``k`` in 0, 1, 2, ... in order.
- ``retired/<n>``: the ``RetiredTask`` of a train task that a review took out of the pool.

Review ``k`` is due when the uses name at least ``(k + 1) * interval`` distinct task jobs. It reads the last
``window`` rewards of each active task: a train task of a complete generation that no earlier review retired. A
task with fewer rewards stays. A review retires a task whose pass rate, the mean of those rewards, is below
``band_low`` or above ``band_high``, or whose rewards have a population standard deviation below
``min_reward_std``. The review reads the rewards that training recorded, so they can be one pool cycle old.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Literal

from reef.core.job_data import TaskSample
from reef.core.values import canonical_json, require_integer, require_text
from reef.recipe.generated_tasks.generator import check_rewards
from reef.recipe.generated_tasks.training import (
    USE_KEY_PREFIX,
    NextSamples,
    TaskItems,
    TaskTraining,
    TaskUse,
    published_train_tasks,
    sample_use_index,
    task_item,
)
from reef.storage.data.tasks import TaskStore
from reef.storage.scenario.samples import StoredSample, TrainerSamples

logger = logging.getLogger(__name__)

#: The key prefix of the documents of the retired tasks.
RETIRED_KEY_PREFIX = "retired/"
#: The key prefix of the documents of the pool reviews.
REVIEW_KEY_PREFIX = "review/"
#: Why a review retires a task, in the order that it checks.
RetirementReason = Literal["below_band", "above_band", "no_reward_spread"]
RETIREMENT_REASONS: tuple[RetirementReason, ...] = ("below_band", "above_band", "no_reward_spread")


@dataclass(frozen=True)
class PoolReviewRule:
    """The rule of the pool reviews.

    Values: ``interval`` is a non-negative int, the task jobs between two reviews; 0 means no review.
    ``window`` is an int of at least 2, the most recent rewards of a task that a review reads. ``band_low`` and
    ``band_high`` are numbers in [0, 1] with ``band_low <= band_high``, the inclusive bounds of the pass rate
    that keeps a task. ``min_reward_std`` is a non-negative finite number, the smallest population standard
    deviation of the rewards that keeps a task; 0 turns this rule off. The defaults run no review, and each of
    their rules keeps every task. The constructor refuses other values.
    Purpose: ``GeneratedTaskSettings.review_rule`` makes it from the ``review_*`` keys of
    ``pipeline.record2sample.config``; ``TrainingPool`` reads it in each review.
    """

    interval: int = 0
    window: int = 2
    band_low: float = 0.0
    band_high: float = 1.0
    min_reward_std: float = 0.0

    def __post_init__(self) -> None:
        require_integer(self.interval, "review_interval")
        require_integer(self.window, "review_window", minimum=2)
        for bound in (self.band_low, self.band_high):
            if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not 0 <= bound <= 1:
                raise ValueError("review_band_low and review_band_high must be numbers between 0 and 1")
        if self.band_low > self.band_high:
            raise ValueError("review_band_low must not be above review_band_high")
        std = self.min_reward_std
        if isinstance(std, bool) or not isinstance(std, (int, float)) or not math.isfinite(std) or std < 0:
            raise ValueError("review_min_reward_std must be a non-negative number")

    def retirement_reason(self, rewards: Sequence[float]) -> RetirementReason | None:
        """Return why a review retires a task with these rewards, or None when the review keeps the task.

        ``rewards`` holds two or more numbers. Both bounds of the band keep a task.
        """
        pass_rate = statistics.fmean(rewards)
        if pass_rate < self.band_low:
            return "below_band"
        if pass_rate > self.band_high:
            return "above_band"
        if statistics.pstdev(rewards) < self.min_reward_std:
            return "no_reward_spread"
        return None


@dataclass(frozen=True)
class RetiredTask:
    """A train task that a pool review took out of the training pool, with the rewards that decided it.

    Values: ``task_name`` is a non-empty string. ``review`` is the non-negative number of the review.
    ``rewards`` is a tuple of two or more finite numbers, the last rewards of the task that the review read.
    ``pass_rate`` is their mean and ``reward_std`` their population standard deviation. ``reason`` is one of
    ``RETIREMENT_REASONS``.
    Purpose: ``TrainingPool`` makes it in a review and saves it under ``retired/<task_name>``. The pool makes no later
    sample of the task, ends the scenario's use of its sample that waits, and reports the retired tasks in its status.
    """

    task_name: str
    review: int
    rewards: tuple[float, ...]
    reason: RetirementReason

    def __post_init__(self) -> None:
        require_text(self.task_name, "task_name")
        require_integer(self.review, "a retired task's review")
        if not isinstance(self.rewards, tuple):
            raise TypeError("a retired task's rewards must be a tuple of finite numbers")
        if len(self.rewards) < 2:
            raise ValueError("a retired task needs two or more rewards")
        check_rewards(self.rewards)
        if self.reason not in RETIREMENT_REASONS:
            raise ValueError(f"a retired task's reason must be one of {', '.join(RETIREMENT_REASONS)}")

    @property
    def key(self) -> str:
        """The key of the retired task's document."""
        return f"{RETIRED_KEY_PREFIX}{self.task_name}"

    @property
    def pass_rate(self) -> float:
        """The mean of the rewards."""
        return statistics.fmean(self.rewards)

    @property
    def reward_std(self) -> float:
        """The population standard deviation of the rewards."""
        return statistics.pstdev(self.rewards)

    def document(self) -> str:
        """The retired task as canonical JSON text."""
        return canonical_json(
            {
                "task_name": self.task_name,
                "review": self.review,
                "rewards": list(self.rewards),
                "pass_rate": self.pass_rate,
                "reward_std": self.reward_std,
                "reason": self.reason,
            }
        )

    @classmethod
    def from_document(cls, text: str) -> RetiredTask:
        """Read the text of :meth:`document`."""
        fields = json.loads(text)
        return cls(fields["task_name"], fields["review"], tuple(fields["rewards"]), fields["reason"])


@dataclass(frozen=True)
class PoolReview:
    """One review of the training pool.

    Values: ``review`` is the non-negative number of the review. ``jobs`` is a positive int, the distinct task
    jobs that the uses named when the review ran. ``kept`` is the non-negative number of active tasks that the
    review kept. ``retired`` maps the name of each task that the review retired to its reason, one of
    ``RETIREMENT_REASONS``.
    Purpose: ``TrainingPool`` makes it in a review and saves it under ``review/<review:06d>``. The number of saved
    reviews is the number of the next review; the pool reports the last review in its status.
    """

    review: int
    jobs: int
    kept: int
    retired: Mapping[str, RetirementReason]

    def __post_init__(self) -> None:
        require_integer(self.review, "a pool review's number")
        require_integer(self.jobs, "a pool review's task jobs", minimum=1)
        require_integer(self.kept, "a pool review's kept tasks")
        if not isinstance(self.retired, Mapping) or not all(
            isinstance(name, str) and name and reason in RETIREMENT_REASONS for name, reason in self.retired.items()
        ):
            raise ValueError("a pool review maps each retired task name to its reason")
        object.__setattr__(self, "retired", dict(self.retired))

    @property
    def key(self) -> str:
        """The key of the review's document."""
        return f"{REVIEW_KEY_PREFIX}{self.review:06d}"

    def document(self) -> str:
        """The review as canonical JSON text."""
        return canonical_json(
            {"review": self.review, "jobs": self.jobs, "kept": self.kept, "retired": dict(self.retired)}
        )

    @classmethod
    def from_document(cls, text: str) -> PoolReview:
        """Read the text of :meth:`document`."""
        fields = json.loads(text)
        return cls(fields["review"], fields["jobs"], fields["kept"], fields["retired"])


def task_uses(documents: Mapping[str, str]) -> dict[str, tuple[TaskUse, ...]]:
    """Return the uses of each train task in the task store's ``documents``, by task name, in use order."""
    uses: dict[str, list[TaskUse]] = {}
    for key, document in documents.items():
        if key.startswith(USE_KEY_PREFIX):
            use = TaskUse.from_document(document)
            uses.setdefault(use.task_name, []).append(use)
    return {name: tuple(sorted(found, key=lambda use: use.use_index)) for name, found in uses.items()}


def retired_tasks(documents: Mapping[str, str]) -> dict[str, RetiredTask]:
    """Return the retired tasks in the task store's ``documents``, by task name."""
    return {
        key.removeprefix(RETIRED_KEY_PREFIX): RetiredTask.from_document(document)
        for key, document in documents.items()
        if key.startswith(RETIRED_KEY_PREFIX)
    }


def pool_reviews(documents: Mapping[str, str]) -> tuple[PoolReview, ...]:
    """Return the reviews in the task store's ``documents``, in review order."""
    reviews = (
        PoolReview.from_document(document) for key, document in documents.items() if key.startswith(REVIEW_KEY_PREFIX)
    )
    return tuple(sorted(reviews, key=lambda review: review.review))


def last_ended_use_index(uses: Sequence[TaskUse], *, has_result: bool) -> int:
    """The last use of a train task whose job ended: the highest ``use_index`` of its saved ``uses``, in use order.
    Without a saved use it is -1, or 0 with ``has_result``: the task's first sample trained in a job that ended
    before a training pool saved uses, such as a job that ran before the deployment set ``pool_capacity``."""
    if uses:
        return uses[-1].use_index
    return 0 if has_result else -1


class TrainingPool(TaskTraining):
    """The training pool: the train tasks train again until a review retires them, and rounds fill the pool.

    Values: ``scenario`` is the non-empty name of the scenario, which the log of a review names. ``store`` is the
    scenario's task store ``generated-tasks``. ``capacity`` is a positive int: a round has
    room only while fewer train tasks than this are active. ``review_rule`` is the ``PoolReviewRule`` of the reviews.
    ``tasks`` are the published train tasks of complete generations with their generation, oldest first, as the
    store held them at construction, at the last ``tasks_published`` or at the last ``ended_uses``; an active task
    is one of them that ``retired`` does not name. ``uses``, ``retired`` and ``reviews`` are the store's documents
    ``use/``, ``retired/`` and ``review/``: the constructor reads them once, and the pool adds each document that it
    writes after the write succeeds. The stage of a scenario is their only writer, so they equal the store's
    documents. The constructor refuses other values.
    Purpose: ``GeneratedTaskSettings.task_training`` makes it when ``pool_capacity`` is above 0, and
    ``GeneratedTaskSamples`` asks it as its ``TaskTraining``.
    """

    def __init__(self, scenario: str, store: TaskStore, capacity: int, review_rule: PoolReviewRule) -> None:
        require_text(scenario, "scenario")
        require_integer(capacity, "a training pool's capacity", minimum=1)
        if not isinstance(review_rule, PoolReviewRule):
            raise TypeError("a training pool's review_rule must be a PoolReviewRule")
        self.scenario = scenario
        self.store = store
        self.capacity = capacity
        self.review_rule = review_rule
        documents = store.documents()
        self.uses = {name: list(found) for name, found in task_uses(documents).items()}
        self.retired = retired_tasks(documents)
        self.reviews = list(pool_reviews(documents))
        self.tasks = published_train_tasks(store)

    def active_task_count(self) -> int:
        """The number of train tasks of complete generations that no review retired: each waits for a job or trains
        in one."""
        return sum(stored.name not in self.retired for _, stored in self.tasks)

    def start_samples(self, items: TaskItems, results: frozenset[str]) -> tuple[TaskSample, ...]:
        """The next sample of each active task whose last use ended without it: after a stop, and of a task that
        trained before the deployment set ``pool_capacity``, which has a first result and no saved use."""
        return self.next_use_samples(items, results)

    def is_idle(self, unfinished: AbstractSet[str]) -> bool:
        """Whether no task is active: an active task trains again, so job ends come while one is."""
        return self.active_task_count() == 0

    def has_room(self, unfinished: AbstractSet[str], samples: TrainerSamples | None) -> bool:
        """Whether fewer train tasks than ``capacity`` are active."""
        return self.active_task_count() < self.capacity

    def ended_uses(
        self, job_id: str, rewards: Mapping[str, Sequence[float]], items: TaskItems, trainer: str
    ) -> tuple[TaskUse, ...]:
        """The uses that the end of the task job ``job_id`` ends.

        The candidates are the samples in flight: the last sample of each task that trained, whose use did not end,
        and whose task has no saved use in this job. A job trains a task once, so the sample of a task that has one
        belongs to a later job. The job's own samples are the candidates whose tasks the rewards name; their uses
        have the tasks' rewards, and the other uses have none. In ``auto``, a job takes samples after those of the
        jobs before it, so a candidate before the job's first own sample belongs to an earlier job that a new
        training start cancelled, and one after it to a job that started before the stage read this end. The end
        ends the uses of its own samples and of the candidates before them. The rewards of a failed job name no
        task, and its end ends the use of every candidate: the stage cannot tell the samples of that job from those
        of a job that started before it read the end.
        """
        self.tasks = published_train_tasks(self.store)
        candidates: dict[str, StoredSample] = {}
        for _, stored in self.tasks:
            found = self.uses.get(stored.name, [])
            last = items.get(stored.name, ())[-1:]
            if (
                last
                and last[0].train_count(trainer) > 0
                and sample_use_index(last[0]) > last_ended_use_index(found, has_result=False)
                and all(use.job_id != job_id for use in found)
            ):
                candidates[stored.name] = last[0]
        first_own = min((sample.sequence for name, sample in candidates.items() if name in rewards), default=None)
        return tuple(
            TaskUse(name, job_id, sample_use_index(sample), tuple(rewards[name] or ()) if name in rewards else ())
            for name, sample in candidates.items()
            if not rewards or name in rewards or (first_own is not None and sample.sequence < first_own)
        )

    def reuse_tasks(self, uses: Sequence[TaskUse], items: TaskItems, samples: TrainerSamples) -> NextSamples:
        """Save ``uses``, run the review that is due, end the scenario's use of each sample of a retired task that
        waits, and return the next sample of each active task whose last use ended, with the retired tasks withdrawn.

        Each step reads what the earlier steps saved, so the end of the same job again, as after a restart, saves the
        uses that a stop left out, runs a review that a stop or a failed write cut short, and makes the samples that
        it did not make.
        """
        for use in uses:
            found = self.uses.setdefault(use.task_name, [])
            # A repeated end of a job saves nothing new.
            if any(earlier.job_id == use.job_id for earlier in found):
                continue
            self.store.write_document(key=use.key, document=use.document())
            found.append(use)
            found.sort(key=lambda saved: saved.use_index)
        review = self.review()
        if review is not None:
            logger.info(
                "generated-task pool review %d on scenario %r after %d jobs: kept %d, retired %s",
                review.review,
                self.scenario,
                review.jobs,
                review.kept,
                ", ".join(f"{name} ({reason})" for name, reason in review.retired.items()) or "none",
            )
        waiting = [
            sample.sample_id
            for name in self.retired
            for sample in items.get(name, ())
            if sample.train_count(samples.trainer) == 0
        ]
        if waiting:
            samples.delete_samples(waiting)
        return NextSamples(self.next_use_samples(items, frozenset()), frozenset(self.retired))

    def review(self) -> PoolReview | None:
        """Run the review that is due, save it, and return it; return None when no review is due.

        The review reads each task of ``tasks`` that no earlier review retired. A stop or a failed write after some
        retired tasks and before the review runs the review again at a later call: a task that it retired stays
        retired with its saved rewards and reason, and the review reads the rewards of the other tasks at that call.
        """
        rule = self.review_rule
        if rule.interval == 0:
            return None
        number = len(self.reviews)
        jobs = len({use.job_id for found in self.uses.values() for use in found})
        if jobs < (number + 1) * rule.interval:
            return None
        kept = 0
        retired_reasons: dict[str, RetirementReason] = {}
        for _, stored in self.tasks:
            name = stored.name
            if name in self.retired:
                # An earlier review retired the task, or this review did before a stop or a failed write.
                if self.retired[name].review == number:
                    retired_reasons[name] = self.retired[name].reason
                continue
            rewards = tuple(reward for use in self.uses.get(name, ()) for reward in use.rewards)
            recent = rewards[-rule.window :]
            reason = rule.retirement_reason(recent) if len(recent) == rule.window else None
            if reason is None:
                kept += 1
                continue
            retired = RetiredTask(name, number, recent, reason)
            self.store.write_document(key=retired.key, document=retired.document())
            self.retired[name] = retired
            retired_reasons[name] = reason
        result = PoolReview(number, jobs, kept, retired_reasons)
        self.store.write_document(key=result.key, document=result.document())
        self.reviews.append(result)
        return result

    def next_use_samples(self, items: TaskItems, results: frozenset[str]) -> tuple[TaskSample, ...]:
        """The sample of the next use of each active task whose last use ended (``last_ended_use_index``): its last
        sample in ``items`` is of that use or an earlier one. ``results`` names the tasks whose first result counts as
        the end of their first use when they have no saved use. The samples are in the order of those last samples,
        which is the order in which the tasks trained."""
        found: list[tuple[int, TaskSample]] = []
        for generation, stored in self.tasks:
            last = items.get(stored.name, ())[-1:]
            ended = last_ended_use_index(self.uses.get(stored.name, []), has_result=stored.name in results)
            if stored.name not in self.retired and last and sample_use_index(last[0]) <= ended:
                found.append((last[0].sequence, task_item(self.store, generation, stored, use_index=ended + 1)))
        return tuple(task for _, task in sorted(found, key=lambda entry: entry[0]))

    def tasks_published(self) -> None:
        """Read the published train tasks again: a round added some."""
        self.tasks = published_train_tasks(self.store)

    def status(self) -> Mapping[str, object]:
        """The pool under ``training_pool``: its capacity, the active and retired tasks, the reviews, and the tasks
        that the last review retired with their reasons."""
        return {
            "training_pool": {
                "capacity": self.capacity,
                "active": self.active_task_count(),
                "retired": len(self.retired),
                "reviews": len(self.reviews),
                "last_review_retired": dict(self.reviews[-1].retired) if self.reviews else {},
            }
        }


__all__ = [
    "RETIREMENT_REASONS",
    "PoolReview",
    "PoolReviewRule",
    "RetiredTask",
    "RetirementReason",
    "TrainingPool",
    "pool_reviews",
    "retired_tasks",
    "task_uses",
]
