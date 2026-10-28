"""The generated-task training pool: tasks that train again, the pool reviews, and the tasks they retire."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import pytest
from tests.support.task_service_stubs import StandInGenerator, task_for

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, SupportsReceipt
from reef.core.job_data import JobOutcome
from reef.core.tasks import HarborTask
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks import GeneratedTaskJobData
from reef.recipe.generated_tasks.generator import (
    GeneratedTaskResult,
    RolledOutTask,
    TaskGenerator,
    TaskValidationResult,
)
from reef.recipe.generated_tasks.processor import (
    GeneratedTaskRecord2Record,
    GeneratedTaskSamples,
    RoundSettings,
    TaskGroupSettings,
    TaskJobSettings,
)
from reef.recipe.generated_tasks.results import rollout_results
from reef.recipe.generated_tasks.review import PoolReview, RetiredTask, pool_reviews, retired_tasks, task_uses
from reef.record2sample.generator.inputs import TaskGenerationRequest
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import TaskStore
from reef.train.pipeline import TrainingPipeline

BATCH_SIZE = 4
TASK_A, TASK_B, TASK_C = "harbor-00000-000", "harbor-00000-001", "harbor-00000-002"


def generated(generation: int, count: int) -> tuple[HarborTask, ...]:
    """The tasks of one generation."""
    return tuple(
        task_for(f"harbor-{generation:05d}-{index:03d}", f"designer-{index}", 9000 + 10 * generation + index)
        for index in range(count)
    )


class FixedTasks(TaskGenerator):
    """A generator that returns the next scripted tasks, accepts each one, and keeps the results it receives."""

    def __init__(self, *generations: tuple[HarborTask, ...]) -> None:
        self.generations = list(generations)
        self.results: list[tuple[GeneratedTaskResult, ...]] = []

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str]) -> Self:
        return cls()

    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        return self.generations.pop(0) if self.generations else ()

    async def validate(self, task: HarborTask) -> TaskValidationResult:
        return TaskValidationResult(())

    async def record_results(self, results: tuple[GeneratedTaskResult, ...], *, generation: int) -> None:
        self.results.append(results)


class Unbound(BackgroundJobRunner):
    """The runner of a test worker that runs its jobs itself. The real worker thread never calls it."""

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("no stage bound")


class InlineWorker(BackgroundJobWorker):
    """Runs each job of the bound stage when it is submitted, on this thread; the next poll returns the outcome."""

    def __init__(self) -> None:
        super().__init__(Unbound(), thread_name="test-pool", concurrency=1)
        self.stage: GeneratedTaskSamples | None = None
        self.outcomes: list[SupportsReceipt] = []

    def submit(self, job: SupportsReceipt) -> bool:
        if self.stage is None:
            return False
        self.outcomes.append(asyncio.run(self.stage.run(job)))
        return True

    def poll(self) -> list[SupportsReceipt]:
        outcomes, self.outcomes = self.outcomes, []
        return outcomes

    def close(self) -> None:
        self.stage = None


def pipeline(
    tmp_path: Path, task_generator: TaskGenerator, *, rounds: int = 1, refill_tasks: int = 1, **job_config: float
) -> TrainingPipeline:
    """A generated-task pipeline on a SQLite task store under ``tmp_path``; ``job_config`` is the sample2job config."""
    worker = InlineWorker()
    context = PipelineContext("tasks", task_storage=SQLiteScenarioStorage(tmp_path / "records").open("tasks").tasks)
    stage = GeneratedTaskRecord2Record(context, TaskGroupSettings(refill_tasks=refill_tasks))
    samples = GeneratedTaskSamples(
        context,
        RoundSettings(rounds=rounds, description="shell tasks"),
        stage,
        task_generator=task_generator,
        generator_service=StandInGenerator(tmp_path / "service"),
        worker=worker,
    )
    jobs = GeneratedTaskJobData(
        context, TaskJobSettings(**{"tasks_per_step": 2, **job_config}), record2record=stage, record2sample=samples
    )
    worker.stage = samples
    return TrainingPipeline(context, record2record=stage, record2sample=samples, sample2job=jobs)


def task_store(p: TrainingPipeline) -> TaskStore:
    store = p.record2sample.task_store
    assert store is not None
    return store


def job_stage(p: TrainingPipeline) -> GeneratedTaskJobData:
    stage = p.sample2job
    assert isinstance(stage, GeneratedTaskJobData)
    return stage


def handed_out(p: TrainingPipeline) -> list[str]:
    """Advance the pipeline, then return the names of the tasks of the next job."""
    p.ready(BATCH_SIZE)
    own = job_stage(p).own_job_data()
    return [] if own is None else [item.metadata["task"]["name"] for item in own]


def finished(job_id: str, rewards: Mapping[str, Sequence[float]]) -> JobOutcome:
    return JobOutcome(
        job_id, "tinker", "finished", {"rollout_rewards": {name: list(r) for name, r in rewards.items()}}
    )


def train(p: TrainingPipeline, job_id: str, rewards: Mapping[str, Sequence[float]] | None = None) -> list[str]:
    """Train the next task job as the trainer does; each task's episodes earn ``rewards[name]``, 1.0 and 0.0 by default."""
    handed_out(p)
    data = p.build_job_data(BATCH_SIZE)
    names = [item.metadata["task"]["name"] for item in data.samples]
    p.acknowledge(data.data_id)
    p.release_records(p.releasable_record_ids())
    p.job_ended(finished(job_id, {name: (rewards or {}).get(name, (1.0, 0.0)) for name in names}))
    return names


def pooled(p: TrainingPipeline) -> list[str]:
    """The names of the waiting tasks, in the order that the pool hands them out."""
    waiting = sorted(job_stage(p).task_pool.waiting.values(), key=lambda task: task.arrival_index)
    return [str(task.task_id) for task in waiting]


# ------------------------------------------------------------------------------------------------- reuse


def test_without_pool_capacity_a_task_trains_once_and_leaves_the_pool(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)))
    assert train(p, "job-1") == [TASK_A, TASK_B]
    stage = job_stage(p)
    assert stage.task_pool.is_idle and handed_out(p) == []
    assert task_uses(task_store(p)) == {} and "training_pool" not in p.status()


def test_with_pool_capacity_a_task_returns_to_the_pool_after_its_job_ends(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 3)), pool_capacity=3, review_interval=0)
    assert train(p, "job-1") == [TASK_A, TASK_B]
    assert train(p, "job-2") == [TASK_C, TASK_A], "the tasks go back to the end of the pool"
    assert pooled(p) == [TASK_B, TASK_C, TASK_A]
    uses = task_uses(task_store(p))
    assert [(use.job_id, use.use_index) for use in uses[TASK_A]] == [("job-1", 0), ("job-2", 1)]


def test_a_repeated_job_end_saves_each_use_once(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=0)
    train(p, "job-1", {TASK_A: (1.0, 1.0)})
    # The trainer gives the same end again, as after a restart.
    p.job_ended(finished("job-1", {TASK_A: (0.0, 0.0), TASK_B: (0.0, 0.0)}))
    uses = task_uses(task_store(p))
    assert {name: [(use.use_index, use.rewards) for use in found] for name, found in uses.items()} == {
        TASK_A: [(0, (1.0, 1.0))],
        TASK_B: [(0, (1.0, 0.0))],
    }


def test_first_rollout_results_still_reach_the_generator_in_pool_mode(tmp_path: Path) -> None:
    task_generator = FixedTasks(generated(0, 2))
    p = pipeline(tmp_path, task_generator, pool_capacity=2, review_interval=0)
    train(p, "job-1", {TASK_A: (0.5, 0.5), TASK_B: (1.0, 0.0)})
    train(p, "job-2", {TASK_A: (0.0, 0.0), TASK_B: (0.0, 0.0)})
    (results,) = task_generator.results
    assert [(result.task.name, result.rollout_rewards) for result in results if isinstance(result, RolledOutTask)] == [
        (TASK_A, (0.5, 0.5)),
        (TASK_B, (1.0, 0.0)),
    ]
    assert rollout_results(task_store(p))[TASK_A].rollout_rewards == (0.5, 0.5), "later uses only train"


def test_rounds_start_only_while_active_tasks_are_below_capacity(tmp_path: Path) -> None:
    task_generator = FixedTasks(generated(0, 2), generated(1, 2), generated(2, 2))
    p = pipeline(tmp_path, task_generator, rounds=3, tasks_per_step=1, pool_capacity=3, review_interval=0)
    train(p, "job-1")
    handed_out(p)
    assert len(task_store(p).generations()) == 2, "two active tasks are below the capacity, so a round starts"
    for job in range(2, 6):
        train(p, f"job-{job}")
    assert len(task_store(p).generations()) == 2, "four active tasks fill the capacity, so no round starts"
    assert p.status()["training_pool"]["active"] == 4


# ------------------------------------------------------------------------------------------------- review


@pytest.mark.parametrize(
    ("rewards", "reason"),
    [((1.0, 1.0), "above_band"), ((0.0, 0.0), "below_band"), ((0.5, 0.5), "no_reward_spread")],
    ids=["above_band", "below_band", "no_reward_spread"],
)
def test_a_review_retires_a_task_outside_the_band_or_without_reward_spread(
    tmp_path: Path, rewards: tuple[float, float], reason: str
) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=1, review_window=2)
    train(p, "job-1", {TASK_A: rewards})
    store = task_store(p)
    assert retired_tasks(store) == {TASK_A: RetiredTask(TASK_A, 0, rewards, reason)}
    assert pool_reviews(store) == (PoolReview(0, 1, 1, {TASK_A: reason}),)
    assert handed_out(p) == [TASK_B], "a retired task is not handed out again"


def test_a_pass_rate_on_a_band_edge_is_kept(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=1, review_window=2)
    train(p, "job-1", {TASK_A: (1.0, 0.5), TASK_B: (0.5, 0.0)})
    assert pool_reviews(task_store(p)) == (PoolReview(0, 1, 2, {}),)
    assert handed_out(p) == [TASK_A, TASK_B]


def test_a_review_reads_only_the_most_recent_rewards(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=2, review_window=2)
    train(p, "job-1", {TASK_A: (0.0, 0.0)})
    train(p, "job-2", {TASK_A: (1.0, 0.0)})
    assert pool_reviews(task_store(p)) == (PoolReview(0, 2, 2, {}),), "the older rewards of task A are below the band"


def test_a_task_with_fewer_rewards_than_the_window_is_kept(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 3)), pool_capacity=3, review_interval=1, review_window=4)
    train(p, "job-1", {TASK_A: (1.0, 1.0)})
    assert pool_reviews(task_store(p)) == (PoolReview(0, 1, 3, {}),)
    assert retired_tasks(task_store(p)) == {}


def test_a_review_runs_after_review_interval_jobs_and_not_before(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=2, review_window=2)
    reviews = []
    for job in range(1, 5):
        train(p, f"job-{job}")
        reviews.append(len(pool_reviews(task_store(p))))
    assert reviews == [0, 1, 1, 2]
    assert [review.jobs for review in pool_reviews(task_store(p))] == [2, 4]


def test_review_interval_zero_reuses_tasks_and_retires_none(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=0)
    for job in range(1, 4):
        assert train(p, f"job-{job}", {TASK_A: (1.0, 1.0), TASK_B: (0.0, 0.0)}) == [TASK_A, TASK_B]
    store = task_store(p)
    assert pool_reviews(store) == () and retired_tasks(store) == {}


# ------------------------------------------------------------------------------------------------- restart


def test_a_restart_puts_trained_active_tasks_back_and_leaves_retired_ones_out(tmp_path: Path) -> None:
    config = {"pool_capacity": 3, "review_interval": 1, "review_window": 2}
    first = pipeline(tmp_path, FixedTasks(generated(0, 3)), **config)
    train(first, "job-1", {TASK_A: (1.0, 1.0)})
    first.close()
    again = pipeline(tmp_path, FixedTasks(), **config)
    assert pooled(again) == [TASK_C, TASK_B], "the least used task first; the retired task stays out"


def test_a_restart_after_retirement_repeats_the_same_review(tmp_path: Path) -> None:
    config = {"pool_capacity": 2, "review_interval": 1, "review_window": 2}
    first = pipeline(tmp_path, FixedTasks(generated(0, 2)), **config)
    store = task_store(first)
    write_document = store.write_document

    def stopping_write_document(*, key: str, document: str) -> None:
        if key.startswith("review/"):
            raise OSError(28, "No space left on device")
        write_document(key=key, document=document)

    # The service stops after the review saved the retired task and before it saved the review.
    store.write_document = stopping_write_document
    with pytest.raises(OSError, match="No space left"):
        train(first, "job-1", {TASK_A: (1.0, 1.0)})
    assert pool_reviews(store) == () and set(retired_tasks(store)) == {TASK_A}
    first.close()
    again = pipeline(tmp_path, FixedTasks(), **config)
    # The trainer gives the stage the last job's end again.
    again.job_ended(finished("job-1", {TASK_A: (1.0, 1.0), TASK_B: (1.0, 0.0)}))
    restored = task_store(again)
    assert pool_reviews(restored) == (PoolReview(0, 1, 1, {TASK_A: "above_band"}),)
    assert retired_tasks(restored) == {TASK_A: RetiredTask(TASK_A, 0, (1.0, 1.0), "above_band")}
    assert pooled(again) == [TASK_B]


# ------------------------------------------------------------------------------------------------- settings and status


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"pool_capacity": -1}, "pool_capacity must be a non-negative integer"),
        ({"pool_capacity": True}, "pool_capacity must be a non-negative integer"),
        ({"pool_capacity": 4, "review_interval": -1}, "review_interval must be a non-negative integer"),
        ({"pool_capacity": 4, "review_interval": True}, "review_interval must be a non-negative integer"),
        ({"pool_capacity": 4, "review_window": 1}, "review_window must be an integer of at least 2"),
        ({"pool_capacity": 4, "review_band_low": -0.1}, "between 0 and 1"),
        ({"pool_capacity": 4, "review_band_high": 1.5}, "between 0 and 1"),
        ({"pool_capacity": 4, "review_band_low": 0.8, "review_band_high": 0.5}, "must not be above"),
        ({"pool_capacity": 4, "review_min_reward_std": -0.1}, "review_min_reward_std must be a non-negative"),
        ({"review_window": 8}, "need a pool_capacity above 0"),
    ],
)
def test_pool_settings_refuse_bad_values(config: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TaskJobSettings(**config)


def test_a_pool_capacity_refuses_refill_tasks_above_1(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="leave refill_tasks at 1"):
        pipeline(tmp_path, FixedTasks(), refill_tasks=2, pool_capacity=4)


def test_the_status_reports_retired_tasks_and_their_reasons(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=1, review_window=2)
    train(p, "job-1", {TASK_A: (0.0, 0.0)})
    assert p.status()["training_pool"] == {
        "capacity": 2,
        "active": 1,
        "retired": 1,
        "reviews": 1,
        "last_review_retired": {TASK_A: "below_band"},
    }
