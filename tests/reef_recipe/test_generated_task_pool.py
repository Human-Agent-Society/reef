"""The generated-task training pool: tasks that train again, the pool reviews, and the tasks they retire."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import pytest
from tests.reef_recipe.support import CommittingPipeline
from tests.support.samples import open_sample_store
from tests.support.task_service_stubs import StandInGenerator, task_for

from reef.core.job_data import JobOutcome
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask
from reef.execution.jobs import BackgroundJobRunner, BackgroundJobWorker, SupportsReceipt
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks import GeneratedTaskJobData
from reef.recipe.generated_tasks.generator import (
    GeneratedTaskResult,
    GeneratorState,
    RolledOutTask,
    TaskGenerator,
    TaskValidationResult,
)
from reef.recipe.generated_tasks.pool import PoolReview, RetiredTask, TaskUse, pool_reviews, retired_tasks, task_uses
from reef.recipe.generated_tasks.processor import GeneratedTaskSamples, GeneratedTaskSettings
from reef.recipe.generated_tasks.results import rollout_results
from reef.record2sample.generator.inputs import TaskGenerationRequest
from reef.storage.samples import stage_item_id
from reef.storage.tasks import TaskStore
from reef.train.pipeline import TrainingPipeline

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
    def from_config(
        cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str], state: GeneratorState
    ) -> Self:
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
    tmp_path: Path,
    task_generator: TaskGenerator,
    *,
    rounds: int = 1,
    jobs_per_round: int = 1,
    refill_tasks: int = 1,
    job_samples: int = 2,
    **pool_config: Any,
) -> CommittingPipeline:
    """A generated-task pipeline of the scenario ``tasks`` on SQLite storage under ``tmp_path``. Each job takes
    ``job_samples`` tasks; ``pool_config`` is the training pool's part of ``pipeline.record2sample.config``."""
    worker = InlineWorker()
    session, samples = open_sample_store(tmp_path / "records", "tasks")
    context = PipelineContext("tasks", task_storage=session.tasks, samples=samples.for_trainer(""))
    stage = GeneratedTaskSamples(
        context,
        GeneratedTaskSettings(
            rounds=rounds,
            jobs_per_round=jobs_per_round,
            description="shell tasks",
            refill_tasks=refill_tasks,
            **pool_config,
        ),
        task_generator=task_generator,
        generator_service=StandInGenerator(tmp_path / "service"),
        worker=worker,
    )
    worker.stage = stage
    jobs = GeneratedTaskJobData.from_config(
        {"job_samples": job_samples, "buffer_size": job_samples},
        context=context,
        record2sample=stage,
        environ={},
    )
    return CommittingPipeline(context, session, record2sample=stage, sample2job=jobs)


def task_stage(stages: TrainingPipeline) -> GeneratedTaskSamples:
    stage = stages.record2sample
    assert isinstance(stage, GeneratedTaskSamples)
    return stage


def task_store(stages: TrainingPipeline) -> TaskStore:
    store = task_stage(stages).task_store
    assert store is not None
    return store


def job_samples(stages: TrainingPipeline) -> int:
    count = stages.sample2job.job_samples
    assert count is not None
    return count


def handed_out(stages: TrainingPipeline) -> list[str]:
    """Look for job data, then return the names of the tasks that the next job takes."""
    stages.ready(job_samples(stages))
    return [sample.metadata["task"]["name"] for _, sample in stages.auto_entries()]


def waiting(stages: TrainingPipeline) -> list[str]:
    """Look for job data, then return the names of the tasks whose sample waits, in the order that jobs take them."""
    stages.ready(job_samples(stages))
    samples = task_stage(stages).task_items().values()
    untrained = sorted(
        (sample for found in samples for sample in found if not sample.train_counts),
        key=lambda sample: sample.sequence,
    )
    return [sample.sample.metadata["task"]["name"] for sample in untrained]


def finished(job_id: str, rewards: Mapping[str, Sequence[float]]) -> JobOutcome:
    return JobOutcome(
        job_id, "tinker", "finished", {"rollout_rewards": {name: list(values) for name, values in rewards.items()}}
    )


def submit(stages: TrainingPipeline) -> list[str]:
    """Submit the next task job as the trainer does: its submission's commit counts its tasks trained. Return the
    names of its tasks. A round's tasks are stored on the look after the one that started the round."""
    for _ in range(3):
        if stages.ready(job_samples(stages)):
            break
    data = stages.build_job_data(job_samples(stages))
    stages.acknowledge(data.data_id)
    return [sample.metadata["task"]["name"] for sample in data.samples]


def train(stages: TrainingPipeline, job_id: str, rewards: Mapping[str, Sequence[float]] | None = None) -> list[str]:
    """Train the next task job as the trainer does; the rollouts of each task earn ``rewards[name]``, by default 1.0
    and 0.0."""
    names = submit(stages)
    stages.job_ended(finished(job_id, {name: (rewards or {}).get(name, (1.0, 0.0)) for name in names}))
    return names


# ------------------------------------------------------------------------------------------------- reuse


@pytest.mark.unit
def test_without_pool_capacity_a_task_trains_once_and_leaves_the_pool(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)))
    assert train(stages, "job-1") == [TASK_A, TASK_B]
    assert waiting(stages) == [] and handed_out(stages) == []
    assert task_uses(task_store(stages)) == {} and "training_pool" not in stages.status()


@pytest.mark.unit
def test_with_pool_capacity_a_task_returns_to_the_pool_after_its_job_ends(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 3)), pool_capacity=3, review_interval=0)
    assert train(stages, "job-1") == [TASK_A, TASK_B]
    assert train(stages, "job-2") == [TASK_C, TASK_A], "the tasks go back to the end of the pool"
    assert waiting(stages) == [TASK_B, TASK_C, TASK_A]
    uses = task_uses(task_store(stages))
    assert [(use.job_id, use.use_index) for use in uses[TASK_A]] == [("job-1", 0), ("job-2", 1)]
    samples = task_stage(stages).task_items()[TASK_A]
    assert [sample.sample_id for sample in samples] == [
        stage_item_id("tasks", f"generated-tasks/{TASK_A}"),
        stage_item_id("tasks", f"generated-tasks/{TASK_A}@1"),
        stage_item_id("tasks", f"generated-tasks/{TASK_A}@2"),
    ]
    assert [sample.train_counts for sample in samples] == [{"": 1}, {"": 1}, {}], "each sample trains once"


@pytest.mark.unit
def test_a_repeated_job_end_saves_each_use_once(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=0)
    train(stages, "job-1", {TASK_A: (1.0, 1.0)})
    # The trainer gives the same end again, as after a restart.
    stages.job_ended(finished("job-1", {TASK_A: (0.0, 0.0), TASK_B: (0.0, 0.0)}))
    uses = task_uses(task_store(stages))
    assert {name: [(use.use_index, use.rewards) for use in found] for name, found in uses.items()} == {
        TASK_A: [(0, (1.0, 1.0))],
        TASK_B: [(0, (1.0, 0.0))],
    }
    assert waiting(stages) == [TASK_A, TASK_B], "each task has one sample that waits"


@pytest.mark.unit
def test_a_job_end_repeated_while_the_next_job_runs_saves_nothing_for_the_next_job(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=0)
    train(stages, "job-1")
    assert submit(stages) == [TASK_A, TASK_B]
    stages.job_ended(finished("job-1", {TASK_A: (1.0, 0.0), TASK_B: (1.0, 0.0)}))
    stages.job_ended(finished("job-2", {TASK_A: (0.0, 0.0), TASK_B: (1.0, 1.0)}))
    uses = task_uses(task_store(stages))
    assert [(use.job_id, use.use_index, use.rewards) for use in uses[TASK_A]] == [
        ("job-1", 0, (1.0, 0.0)),
        ("job-2", 1, (0.0, 0.0)),
    ]


@pytest.mark.unit
def test_a_job_end_given_again_while_the_next_job_runs_leaves_every_task_of_that_job(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 3)), pool_capacity=3, review_interval=0)
    train(stages, "job-1")
    assert waiting(stages) == [TASK_C, TASK_A, TASK_B]
    # The trainer starts with no running job and gives the stage job-1's end again, but control submits job-2 first.
    assert submit(stages) == [TASK_C, TASK_A]
    stages.job_ended(finished("job-1", {TASK_A: (1.0, 0.0), TASK_B: (1.0, 0.0)}))
    assert waiting(stages) == [TASK_B], "task C trains in job-2, so it does not wait again"
    stages.job_ended(finished("job-2", {TASK_C: (0.0, 1.0), TASK_A: (1.0, 1.0)}))
    store = task_store(stages)
    assert {name: [(use.job_id, use.rewards) for use in found] for name, found in task_uses(store).items()} == {
        TASK_A: [("job-1", (1.0, 0.0)), ("job-2", (1.0, 1.0))],
        TASK_B: [("job-1", (1.0, 0.0))],
        TASK_C: [("job-2", (0.0, 1.0))],
    }
    assert rollout_results(store)[TASK_C].rollout_rewards == (0.0, 1.0)
    assert waiting(stages) == [TASK_B, TASK_C, TASK_A]


@pytest.mark.unit
def test_first_rollout_results_still_reach_the_generator_in_pool_mode(tmp_path: Path) -> None:
    task_generator = FixedTasks(generated(0, 2))
    stages = pipeline(tmp_path, task_generator, pool_capacity=2, review_interval=0)
    train(stages, "job-1", {TASK_A: (0.5, 0.5), TASK_B: (1.0, 0.0)})
    train(stages, "job-2", {TASK_A: (0.0, 0.0), TASK_B: (0.0, 0.0)})
    (results,) = task_generator.results
    assert [(result.task.name, result.rollout_rewards) for result in results if isinstance(result, RolledOutTask)] == [
        (TASK_A, (0.5, 0.5)),
        (TASK_B, (1.0, 0.0)),
    ]
    assert rollout_results(task_store(stages))[TASK_A].rollout_rewards == (0.5, 0.5), "later uses only train"


@pytest.mark.unit
def test_a_failed_job_returns_its_tasks_with_uses_that_have_no_rewards(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 3)), pool_capacity=3, review_interval=0)
    assert submit(stages) == [TASK_A, TASK_B]
    stages.job_ended(JobOutcome("job-1", "tinker", "failed", error="the sampler is gone"))
    assert {name: [use.rewards for use in found] for name, found in task_uses(task_store(stages)).items()} == {
        TASK_A: [()],
        TASK_B: [()],
    }
    assert waiting(stages) == [TASK_C, TASK_A, TASK_B]


@pytest.mark.unit
def test_a_job_end_returns_a_cancelled_jobs_tasks_and_leaves_a_later_jobs_tasks(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 3)), job_samples=1, pool_capacity=3, review_interval=0)
    # A rollback cancels the first job, so its end never commits; the next job takes the next task.
    assert submit(stages) == [TASK_A]
    assert submit(stages) == [TASK_B]
    # The trainer submits the third job before the samples service reads the second job's end.
    assert submit(stages) == [TASK_C]
    stages.job_ended(finished("job-2", {TASK_B: (1.0, 0.0)}))
    uses = task_uses(task_store(stages))
    assert {name: [use.job_id for use in found] for name, found in uses.items()} == {
        TASK_A: ["job-2"],
        TASK_B: ["job-2"],
    }
    assert uses[TASK_A][0].rewards == (), "the cancelled job ran no rollout that the end reports"
    assert waiting(stages) == [TASK_A, TASK_B], "task C still trains"
    stages.job_ended(finished("job-3", {TASK_C: (0.0, 1.0)}))
    assert task_uses(task_store(stages))[TASK_C][0].rewards == (0.0, 1.0)
    assert waiting(stages) == [TASK_A, TASK_B, TASK_C]


@pytest.mark.unit
def test_two_job_ends_read_together_save_each_use_once(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 3)), job_samples=1, pool_capacity=3, review_interval=0)
    # The trainer submits the second job before the samples service reads the first job's end, and the service reads
    # both ends before it stores the next samples.
    assert submit(stages) == [TASK_A] and submit(stages) == [TASK_B]
    stages.job_ended(finished("job-1", {TASK_A: (1.0, 0.0)}))
    stages.job_ended(finished("job-2", {TASK_B: (0.0, 1.0)}))
    uses = task_uses(task_store(stages))
    assert {name: [(use.job_id, use.rewards) for use in found] for name, found in uses.items()} == {
        TASK_A: [("job-1", (1.0, 0.0))],
        TASK_B: [("job-2", (0.0, 1.0))],
    }
    assert waiting(stages) == [TASK_C, TASK_A, TASK_B]


@pytest.mark.unit
def test_with_a_pool_rounds_keep_the_pace_of_jobs_per_round(tmp_path: Path) -> None:
    task_generator = FixedTasks(generated(0, 1), generated(1, 1))
    stages = pipeline(
        tmp_path, task_generator, rounds=2, jobs_per_round=2, job_samples=1, pool_capacity=5, review_interval=0
    )
    train(stages, "job-1")
    handed_out(stages)
    assert len(task_store(stages).generations()) == 1, "task A has its result, but it trains again, so a job end comes"
    train(stages, "job-2")
    handed_out(stages)
    assert len(task_store(stages).generations()) == 2


@pytest.mark.unit
def test_rounds_start_only_while_active_tasks_are_below_capacity(tmp_path: Path) -> None:
    task_generator = FixedTasks(generated(0, 2), generated(1, 2), generated(2, 2))
    stages = pipeline(tmp_path, task_generator, rounds=3, job_samples=1, pool_capacity=3, review_interval=0)
    train(stages, "job-1")
    handed_out(stages)
    assert len(task_store(stages).generations()) == 2, "two active tasks are below the capacity, so a round starts"
    for job in range(2, 6):
        train(stages, f"job-{job}")
    assert len(task_store(stages).generations()) == 2, "four active tasks fill the capacity, so no round starts"
    assert stages.status()["training_pool"]["active"] == 4


@pytest.mark.unit
def test_a_task_in_a_running_job_counts_as_active_for_the_capacity(tmp_path: Path) -> None:
    task_generator = FixedTasks(generated(0, 3), generated(1, 3))
    stages = pipeline(tmp_path, task_generator, rounds=2, job_samples=1, pool_capacity=3, review_interval=0)
    train(stages, "job-1")
    assert submit(stages) == [TASK_B]
    assert waiting(stages) == [TASK_C, TASK_A]
    assert len(task_store(stages).generations()) == 1, "two waiting tasks and one task in a job fill the capacity"


# ------------------------------------------------------------------------------------------------- review


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rewards", "reason"),
    [
        ((1.0, 1.0), "above_band"),
        ((0.0, 0.0), "below_band"),
        ((0.5, 0.5), "no_reward_spread"),
        # The population standard deviation is 0.075; the sample standard deviation, 0.106, would keep the task.
        ((0.45, 0.6), "no_reward_spread"),
    ],
    ids=["above_band", "below_band", "no_reward_spread", "no_population_spread"],
)
def test_a_review_retires_a_task_outside_the_band_or_without_reward_spread(
    tmp_path: Path, rewards: tuple[float, float], reason: str
) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=1, review_window=2)
    train(stages, "job-1", {TASK_A: rewards})
    store = task_store(stages)
    assert retired_tasks(store) == {TASK_A: RetiredTask(TASK_A, 0, rewards, reason)}
    assert pool_reviews(store) == (PoolReview(0, 1, 1, {TASK_A: reason}),)
    assert handed_out(stages) == [TASK_B], "a retired task is not handed out again"


@pytest.mark.unit
def test_a_review_retires_a_waiting_task_before_it_trains_again(tmp_path: Path) -> None:
    stages = pipeline(
        tmp_path, FixedTasks(generated(0, 2)), job_samples=1, pool_capacity=2, review_interval=2, review_window=2
    )
    train(stages, "job-1", {TASK_A: (1.0, 1.0)})
    assert waiting(stages) == [TASK_B, TASK_A]
    train(stages, "job-2")
    assert set(retired_tasks(task_store(stages))) == {TASK_A}
    assert handed_out(stages) == [TASK_B], "task A waited in the pool when the review retired it"
    assert waiting(stages) == [TASK_B], "the scenario no longer uses the sample of task A that waited"
    (trained,) = task_stage(stages).task_items()[TASK_A]
    assert trained.sample_id == stage_item_id("tasks", f"generated-tasks/{TASK_A}"), "the trained sample stays"


@pytest.mark.unit
def test_a_review_takes_a_retired_task_out_of_the_next_jobs_hand_out(tmp_path: Path) -> None:
    task_d, task_e = "harbor-00000-003", "harbor-00000-004"
    stages = pipeline(tmp_path, FixedTasks(generated(0, 5)), pool_capacity=5, review_interval=2, review_window=2)
    train(stages, "job-1", {TASK_A: (1.0, 1.0)})
    assert submit(stages) == [TASK_C, task_d]
    # While job-2 runs, a look for job data finds the next job's tasks.
    assert handed_out(stages) == [task_e, TASK_A]
    stages.job_ended(finished("job-2", {TASK_C: (1.0, 0.0), task_d: (1.0, 0.0)}))
    assert set(retired_tasks(task_store(stages))) == {TASK_A}
    assert handed_out(stages) == [task_e, TASK_B], "the next job takes the tasks without task A, in pool order"


@pytest.mark.unit
def test_a_job_whose_data_was_taken_before_the_review_trains_the_retired_task_once_more(tmp_path: Path) -> None:
    task_d, task_e = "harbor-00000-003", "harbor-00000-004"
    stages = pipeline(tmp_path, FixedTasks(generated(0, 5)), pool_capacity=5, review_interval=2, review_window=2)
    train(stages, "job-1", {TASK_A: (1.0, 1.0)})
    assert submit(stages) == [TASK_C, task_d]
    # The trainer takes the next job's data as soon as job-2 ends, before the samples service reads that end.
    data = stages.build_job_data(2)
    stages.job_ended(finished("job-2", {TASK_C: (1.0, 0.0), task_d: (1.0, 0.0)}))
    assert set(retired_tasks(task_store(stages))) == {TASK_A}
    stages.acknowledge(data.data_id)
    assert [sample.metadata["task"]["name"] for sample in data.samples] == [task_e, TASK_A]
    stages.job_ended(finished("job-3", {task_e: (1.0, 0.0), TASK_A: (1.0, 1.0)}))
    assert TASK_A not in waiting(stages), "the retired task trains no more"


@pytest.mark.unit
def test_a_later_review_leaves_a_retired_task_out(tmp_path: Path) -> None:
    stages = pipeline(
        tmp_path, FixedTasks(generated(0, 2)), job_samples=1, pool_capacity=2, review_interval=1, review_window=2
    )
    train(stages, "job-1", {TASK_A: (1.0, 1.0)})
    train(stages, "job-2", {TASK_B: (0.0, 1.0)})
    assert pool_reviews(task_store(stages)) == (PoolReview(0, 1, 1, {TASK_A: "above_band"}), PoolReview(1, 2, 1, {}))


@pytest.mark.unit
def test_a_review_retires_a_task_whose_next_sample_is_not_stored_yet(tmp_path: Path) -> None:
    stages = pipeline(
        tmp_path, FixedTasks(generated(0, 2)), job_samples=1, pool_capacity=2, review_interval=2, review_window=2
    )
    assert submit(stages) == [TASK_A] and submit(stages) == [TASK_B]
    # The samples service reads both ends before it stores the sample of task A's next use.
    stages.job_ended(finished("job-1", {TASK_A: (1.0, 1.0)}))
    stages.job_ended(finished("job-2", {TASK_B: (0.0, 1.0)}))
    assert set(retired_tasks(task_store(stages))) == {TASK_A}
    assert waiting(stages) == [TASK_B]


@pytest.mark.unit
def test_a_pass_rate_on_a_band_edge_is_kept(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=1, review_window=2)
    train(stages, "job-1", {TASK_A: (1.0, 0.5), TASK_B: (0.5, 0.0)})
    assert pool_reviews(task_store(stages)) == (PoolReview(0, 1, 2, {}),)
    assert handed_out(stages) == [TASK_A, TASK_B]


@pytest.mark.unit
def test_a_review_reads_only_the_most_recent_rewards(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=2, review_window=2)
    train(stages, "job-1", {TASK_A: (0.0, 0.0)})
    train(stages, "job-2", {TASK_A: (1.0, 0.0)})
    assert pool_reviews(task_store(stages)) == (
        PoolReview(0, 2, 2, {}),
    ), "the older rewards of task A are below the band"


@pytest.mark.unit
def test_a_task_with_fewer_rewards_than_the_window_is_kept(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 3)), pool_capacity=3, review_interval=1, review_window=4)
    train(stages, "job-1", {TASK_A: (1.0, 1.0)})
    assert pool_reviews(task_store(stages)) == (PoolReview(0, 1, 3, {}),)
    assert retired_tasks(task_store(stages)) == {}


@pytest.mark.unit
def test_a_review_runs_after_review_interval_jobs_and_not_before(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=2, review_window=2)
    reviews = []
    for job in range(1, 5):
        train(stages, f"job-{job}")
        reviews.append(len(pool_reviews(task_store(stages))))
    assert reviews == [0, 1, 1, 2]
    assert [review.jobs for review in pool_reviews(task_store(stages))] == [2, 4]


@pytest.mark.unit
def test_review_interval_zero_reuses_tasks_and_retires_none(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=0)
    for job in range(1, 4):
        assert train(stages, f"job-{job}", {TASK_A: (1.0, 1.0), TASK_B: (0.0, 0.0)}) == [TASK_A, TASK_B]
    store = task_store(stages)
    assert pool_reviews(store) == () and retired_tasks(store) == {}


# ------------------------------------------------------------------------------------------------- restart


@pytest.mark.unit
def test_a_restart_keeps_the_pool_order_and_leaves_retired_tasks_out(tmp_path: Path) -> None:
    config = {"pool_capacity": 3, "review_interval": 1, "review_window": 2}
    first = pipeline(tmp_path, FixedTasks(generated(0, 3)), **config)
    train(first, "job-1", {TASK_A: (1.0, 1.0)})
    assert waiting(first) == [TASK_C, TASK_B]
    first.close()
    again = pipeline(tmp_path, FixedTasks(), **config)
    assert waiting(again) == [TASK_C, TASK_B], "the retired task stays out"
    assert again.status()["training_pool"]["active"] == 2


@pytest.mark.unit
def test_a_restart_makes_the_next_samples_that_a_stop_left_out(tmp_path: Path) -> None:
    config = {"pool_capacity": 2, "review_interval": 0}
    first = pipeline(tmp_path, FixedTasks(generated(0, 2)), **config)
    train(first, "job-1")
    # The service stops after the job's end saved the uses and before the pipeline stored the next samples.
    task_stage(first).made.clear()
    assert waiting(first) == []
    first.close()
    again = pipeline(tmp_path, FixedTasks(), **config)
    assert waiting(again) == [TASK_A, TASK_B]


@pytest.mark.unit
def test_a_pool_set_on_a_scenario_whose_tasks_trained_once_trains_them_again(tmp_path: Path) -> None:
    first = pipeline(tmp_path, FixedTasks(generated(0, 3)))
    assert train(first, "job-1") == [TASK_A, TASK_B]
    first.close()
    again = pipeline(tmp_path, FixedTasks(), pool_capacity=3, review_interval=0)
    assert waiting(again) == [TASK_C, TASK_A, TASK_B], "the tasks with a first result go back to the pool"
    assert train(again, "job-2") == [TASK_C, TASK_A]
    uses = task_uses(task_store(again))
    assert {name: [(use.job_id, use.use_index) for use in found] for name, found in uses.items()} == {
        TASK_A: [("job-2", 1)],
        TASK_C: [("job-2", 0)],
    }
    assert waiting(again) == [TASK_B, TASK_C, TASK_A]


@pytest.mark.unit
def test_a_restart_after_retirement_repeats_the_same_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {"pool_capacity": 2, "review_interval": 1, "review_window": 2}
    first = pipeline(tmp_path, FixedTasks(generated(0, 2)), **config)
    store = task_store(first)
    write_document = store.write_document

    def stopping_write_document(*, key: str, document: str) -> None:
        if key.startswith("review/"):
            raise OSError(28, "No space left on device")
        write_document(key=key, document=document)

    # The service stops after the review saved the retired task and before it saved the review.
    monkeypatch.setattr(store, "write_document", stopping_write_document)
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
    assert waiting(again) == [TASK_B]


@pytest.mark.unit
def test_a_restart_after_a_stop_between_two_uses_reviews_every_use_of_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {"pool_capacity": 2, "review_interval": 1, "review_window": 2}
    first = pipeline(tmp_path, FixedTasks(generated(0, 2)), **config)
    store = task_store(first)
    write_document = store.write_document

    def stopping_write_document(*, key: str, document: str) -> None:
        if key == f"use/{TASK_B}/job-1":
            raise OSError(28, "No space left on device")
        write_document(key=key, document=document)

    # The service stops after it saved the use of task A in job-1 and before the use of task B.
    monkeypatch.setattr(store, "write_document", stopping_write_document)
    rewards = {TASK_A: (1.0, 0.0), TASK_B: (1.0, 1.0)}
    with pytest.raises(OSError, match="No space left"):
        train(first, "job-1", rewards)
    assert set(task_uses(store)) == {TASK_A}
    first.close()
    again = pipeline(tmp_path, FixedTasks(), **config)
    assert pool_reviews(task_store(again)) == (), "a start reviews nothing before the last job's end"
    # The trainer gives the stage the last job's end again.
    again.job_ended(finished("job-1", rewards))
    assert pool_reviews(task_store(again)) == (PoolReview(0, 1, 1, {TASK_B: "above_band"}),)
    assert waiting(again) == [TASK_A]


@pytest.mark.unit
@pytest.mark.parametrize("restart", [False, True], ids=["same_service", "restart"])
def test_a_review_that_a_failed_write_cut_short_keeps_its_retirements_at_a_later_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restart: bool
) -> None:
    config = {"pool_capacity": 2, "review_interval": 2, "review_window": 2}
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)), job_samples=1, **config)
    train(stages, "job-1", {TASK_A: (1.0, 1.0)})
    assert submit(stages) == [TASK_B]
    # The trainer submits job-3 before the samples service reads job-2's end.
    assert submit(stages) == [TASK_A]
    store = task_store(stages)
    write_document = store.write_document

    def failing_write_document(*, key: str, document: str) -> None:
        if key.startswith("review/"):
            raise OSError("connection reset")
        write_document(key=key, document=document)

    # The review of job-2's end saves the retired task A, and then the database fails once.
    monkeypatch.setattr(store, "write_document", failing_write_document)
    with pytest.raises(OSError, match="connection reset"):
        stages.job_ended(finished("job-2", {TASK_B: (1.0, 0.0)}))
    monkeypatch.undo()
    if restart:
        # job-3 runs at the restart, so the trainer does not give job-2's end again.
        stages.close()
        stages = pipeline(tmp_path, FixedTasks(), job_samples=1, **config)
    stages.job_ended(finished("job-3", {TASK_A: (0.0, 0.0)}))
    store = task_store(stages)
    assert pool_reviews(store) == (PoolReview(0, 3, 1, {TASK_A: "above_band"}),)
    assert retired_tasks(store) == {TASK_A: RetiredTask(TASK_A, 0, (1.0, 1.0), "above_band")}
    assert waiting(stages) == [TASK_B]


@pytest.mark.unit
def test_a_restart_while_a_job_runs_leaves_the_tasks_of_that_job_to_its_end(tmp_path: Path) -> None:
    config = {"pool_capacity": 4, "review_interval": 0}
    task_d = "harbor-00000-003"
    first = pipeline(tmp_path, FixedTasks(generated(0, 4)), **config)
    assert submit(first) == [TASK_A, TASK_B]
    # The trainer submits job-2 before the samples service reads job-1's end.
    assert submit(first) == [TASK_C, task_d]
    first.job_ended(finished("job-1", {TASK_A: (1.0, 0.0), TASK_B: (1.0, 0.0)}))
    assert set(rollout_results(task_store(first))) == {TASK_A, TASK_B}, "job-2 still runs the rollouts of C and D"
    first.close()
    # job-2 runs at the restart, so the trainer does not give job-1's end again.
    again = pipeline(tmp_path, FixedTasks(), **config)
    assert waiting(again) == [TASK_A, TASK_B], "C and D wait for the end of job-2"
    again.job_ended(finished("job-2", {TASK_C: (0.0, 1.0), task_d: (1.0, 0.0)}))
    store = task_store(again)
    assert {name: [(use.job_id, use.rewards) for use in found] for name, found in task_uses(store).items()} == {
        TASK_A: [("job-1", (1.0, 0.0))],
        TASK_B: [("job-1", (1.0, 0.0))],
        TASK_C: [("job-2", (0.0, 1.0))],
        task_d: [("job-2", (1.0, 0.0))],
    }
    assert rollout_results(store)[TASK_C].rollout_rewards == (0.0, 1.0)
    assert waiting(again) == [TASK_A, TASK_B, TASK_C, task_d]


# ------------------------------------------------------------------------------------------------- settings and status


@pytest.mark.unit
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
def test_a_malformed_pool_setting_is_refused(config: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GeneratedTaskSettings(**config)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"task_name": ""}, "task_name must be a non-empty string"),
        ({"job_id": " job-1"}, "job_id must be a non-empty string"),
        ({"use_index": -1}, "use_index must be a non-negative integer"),
        ({"use_index": True}, "use_index must be a non-negative integer"),
        ({"rewards": [1.0]}, "a task use's rewards must be a tuple of finite numbers"),
        ({"rewards": (1.0, float("nan"))}, "rewards must be finite numbers"),
        ({"rewards": (True,)}, "rewards must be finite numbers"),
    ],
)
def test_a_malformed_task_use_is_refused(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TaskUse(**{"task_name": TASK_A, "job_id": "job-1", "use_index": 0, "rewards": (1.0, 0.0), **fields})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"task_name": ""}, "task_name must be a non-empty string"),
        ({"review": -1}, "review must be a non-negative integer"),
        ({"rewards": ()}, "a retired task needs two or more rewards"),
        ({"rewards": (1.0,)}, "a retired task needs two or more rewards"),
        ({"rewards": [1.0, 1.0]}, "a retired task needs two or more rewards"),
        ({"rewards": (1.0, float("inf"))}, "rewards must be finite numbers"),
        ({"rewards": (1.0, "1.0")}, "rewards must be finite numbers"),
        ({"reason": "stale"}, "reason must be one of below_band, above_band, no_reward_spread"),
    ],
)
def test_a_malformed_retired_task_is_refused(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RetiredTask(**{"task_name": TASK_A, "review": 0, "rewards": (1.0, 1.0), "reason": "above_band", **fields})


@pytest.mark.unit
def test_a_pool_capacity_refuses_refill_tasks_above_1() -> None:
    config = {"refill_tasks": 2, "pool_capacity": 4}
    with pytest.raises(PluginConfigError, match="leave refill_tasks at 1"):
        GeneratedTaskSettings.parse(config, plugin_class=GeneratedTaskSamples)


@pytest.mark.unit
def test_the_status_reports_retired_tasks_and_their_reasons(tmp_path: Path) -> None:
    stages = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=1, review_window=2)
    train(stages, "job-1", {TASK_A: (0.0, 0.0)})
    assert handed_out(stages) == [TASK_B]
    assert stages.status()["training_pool"] == {
        "capacity": 2,
        "active": 1,
        "retired": 1,
        "reviews": 1,
        "last_review_retired": {TASK_A: "below_band"},
    }
