"""The training pool review of the generated-task recipe: each use of a train task, the reviews, and the retired tasks.

With ``pool_capacity`` above 0, a train task goes back to the task pool when its task job ends, so it trains
again. The recipe keeps three more values in the task store's documents, besides those of
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
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from reef.storage.tasks import TaskStore

USE_KEY_PREFIX = "use/"
RETIRED_KEY_PREFIX = "retired/"
REVIEW_KEY_PREFIX = "review/"
#: The reasons of a retirement, in the order that a review checks them.
RETIREMENT_REASONS = ("below_band", "above_band", "no_reward_spread")


def is_whole_number(value: object) -> bool:
    """Whether ``value`` is an int and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_finite_number(value: object) -> bool:
    """Whether ``value`` is a finite int or float and not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical_json(fields: Mapping[str, object]) -> str:
    """The canonical JSON text of a document: equal fields give the same text."""
    return json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ReviewSettings:
    """The rule of the pool reviews.

    Values: ``interval`` is a non-negative int, the task jobs between two reviews; 0 means no review.
    ``window`` is an int of at least 2, the most recent rewards of a task that a review reads. ``band_low`` and
    ``band_high`` are numbers in [0, 1] with ``band_low <= band_high``, the inclusive bounds of the pass rate
    that keeps a task. ``min_reward_std`` is a non-negative finite number, the smallest population standard
    deviation of the rewards that keeps a task; 0 turns this rule off. The constructor refuses other values.
    Purpose: ``TaskJobSettings.review_settings`` makes it from ``pipeline.sample2job.config``; ``review_pool``
    and ``retirement_reason`` read it.
    """

    interval: int
    window: int
    band_low: float
    band_high: float
    min_reward_std: float

    def __post_init__(self) -> None:
        if not is_whole_number(self.interval) or self.interval < 0:
            raise ValueError("review_interval must be a non-negative integer")
        if not is_whole_number(self.window) or self.window < 2:
            raise ValueError("review_window must be an integer of at least 2")
        if not all(is_finite_number(bound) and 0 <= bound <= 1 for bound in (self.band_low, self.band_high)):
            raise ValueError("review_band_low and review_band_high must be numbers between 0 and 1")
        if self.band_low > self.band_high:
            raise ValueError("review_band_low must not be above review_band_high")
        if not is_finite_number(self.min_reward_std) or self.min_reward_std < 0:
            raise ValueError("review_min_reward_std must be a non-negative number")


@dataclass(frozen=True)
class TaskUse:
    """One use of a train task in a task job: the rewards of the task's episodes in that job.

    Values: ``task_name`` and ``job_id`` are non-empty strings. ``use_index`` is a non-negative int, the number of
    uses of the task saved before this one. ``rewards`` is a tuple of finite numbers in run order; it is empty
    when no episode of the task ran.
    Purpose: ``save_task_uses`` makes one for each task of a task job that ended, and saves it under
    ``use/<task_name>/<job_id>``. ``review_pool`` reads the rewards; a restart puts the least used tasks first.
    """

    task_name: str
    job_id: str
    use_index: int
    rewards: tuple[float, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.task_name, self.job_id)):
            raise ValueError("a task use needs its task name and job id")
        if not is_whole_number(self.use_index) or self.use_index < 0:
            raise ValueError("a task use's use_index must be a non-negative integer")
        if not isinstance(self.rewards, tuple) or not all(is_finite_number(reward) for reward in self.rewards):
            raise ValueError("a task use's rewards must be a tuple of finite numbers")

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
class RetiredTask:
    """A train task that a pool review took out of the task pool, with the rewards that decided it.

    Values: ``task_name`` is a non-empty string. ``review`` is the non-negative number of the review.
    ``rewards`` is a tuple of two or more finite numbers, the last rewards of the task that the review read.
    ``pass_rate`` is their mean and ``reward_std`` their population standard deviation. ``reason`` is one of
    ``RETIREMENT_REASONS``.
    Purpose: ``review_pool`` makes it and saves it under ``retired/<task_name>``. The sample-to-job stage leaves
    the task out of the pool, and reports the retired tasks in its status.
    """

    task_name: str
    review: int
    rewards: tuple[float, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_name, str) or not self.task_name:
            raise ValueError("a retired task needs its task name")
        if not is_whole_number(self.review) or self.review < 0:
            raise ValueError("a retired task's review must be a non-negative integer")
        if not isinstance(self.rewards, tuple) or len(self.rewards) < 2:
            raise ValueError("a retired task needs two or more rewards")
        if not all(is_finite_number(reward) for reward in self.rewards):
            raise ValueError("a retired task's rewards must be finite numbers")
        if self.reason not in RETIREMENT_REASONS:
            raise ValueError(f"a retired task's reason must be one of {', '.join(RETIREMENT_REASONS)}")

    @property
    def pass_rate(self) -> float:
        """The mean of the rewards."""
        return statistics.fmean(self.rewards)

    @property
    def reward_std(self) -> float:
        """The population standard deviation of the rewards."""
        return statistics.pstdev(self.rewards)

    def document(self) -> str:
        """The retired task as canonical JSON text: a repeated review gives the same text."""
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
    Purpose: ``review_pool`` makes it and saves it under ``review/<review:06d>``. The number of saved reviews is
    the number of the next review; the sample-to-job stage reports the last review in its status.
    """

    review: int
    jobs: int
    kept: int
    retired: Mapping[str, str]

    def __post_init__(self) -> None:
        if not is_whole_number(self.review) or self.review < 0:
            raise ValueError("a pool review's number must be a non-negative integer")
        if not is_whole_number(self.jobs) or self.jobs < 1:
            raise ValueError("a pool review counts one or more task jobs")
        if not is_whole_number(self.kept) or self.kept < 0:
            raise ValueError("a pool review keeps a non-negative number of tasks")
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


def task_uses(store: TaskStore) -> dict[str, tuple[TaskUse, ...]]:
    """Return the saved uses of each train task, by task name, in use order."""
    uses: dict[str, list[TaskUse]] = {}
    for key, document in store.documents().items():
        if key.startswith(USE_KEY_PREFIX):
            use = TaskUse.from_document(document)
            uses.setdefault(use.task_name, []).append(use)
    return {name: tuple(sorted(found, key=lambda use: use.use_index)) for name, found in uses.items()}


def save_task_uses(store: TaskStore, *, job_id: str, rewards: Mapping[str, Sequence[float]]) -> None:
    """Save a use in the job ``job_id`` for each task in ``rewards``, with that task's rewards.

    A task that has a use in this job keeps it, so a repeated end of the same job saves nothing new.
    """
    saved = task_uses(store)
    for name, values in rewards.items():
        earlier = saved.get(name, ())
        if any(use.job_id == job_id for use in earlier):
            continue
        use = TaskUse(name, job_id, len(earlier), tuple(values))
        store.write_document(key=use.key, document=use.document())


def retired_tasks(store: TaskStore) -> dict[str, RetiredTask]:
    """Return the retired tasks by task name."""
    return {
        key.removeprefix(RETIRED_KEY_PREFIX): RetiredTask.from_document(document)
        for key, document in store.documents().items()
        if key.startswith(RETIRED_KEY_PREFIX)
    }


def pool_reviews(store: TaskStore) -> tuple[PoolReview, ...]:
    """Return the saved reviews in review order."""
    reviews = (
        PoolReview.from_document(document)
        for key, document in store.documents().items()
        if key.startswith(REVIEW_KEY_PREFIX)
    )
    return tuple(sorted(reviews, key=lambda review: review.review))


def retirement_reason(rewards: Sequence[float], settings: ReviewSettings) -> str | None:
    """Return why a review retires a task with these rewards, or None when the review keeps the task.

    ``rewards`` holds two or more numbers. Both bounds of the band keep a task.
    """
    pass_rate = statistics.fmean(rewards)
    if pass_rate < settings.band_low:
        return "below_band"
    if pass_rate > settings.band_high:
        return "above_band"
    if statistics.pstdev(rewards) < settings.min_reward_std:
        return "no_reward_spread"
    return None


def review_pool(store: TaskStore, settings: ReviewSettings, task_names: Sequence[str]) -> PoolReview | None:
    """Run the review that is due, save it, and return it; return None when no review is due.

    ``task_names`` are the train tasks of the complete generations. The review reads each of them that no
    earlier review retired. A stop between the retired tasks and the review repeats the same review, which
    saves the same documents again.
    """
    if settings.interval == 0:
        return None
    review = len(pool_reviews(store))
    uses = task_uses(store)
    jobs = len({use.job_id for found in uses.values() for use in found})
    if jobs < (review + 1) * settings.interval:
        return None
    earlier = {name for name, task in retired_tasks(store).items() if task.review < review}
    kept = 0
    retired_reasons: dict[str, str] = {}
    for name in task_names:
        if name in earlier:
            continue
        rewards = tuple(reward for use in uses.get(name, ()) for reward in use.rewards)
        recent = rewards[-settings.window :]
        reason = retirement_reason(recent, settings) if len(recent) == settings.window else None
        if reason is None:
            kept += 1
            continue
        retired = RetiredTask(name, review, recent, reason)
        store.write_document(key=f"{RETIRED_KEY_PREFIX}{name}", document=retired.document())
        retired_reasons[name] = reason
    result = PoolReview(review, jobs, kept, retired_reasons)
    store.write_document(key=result.key, document=result.document())
    return result
