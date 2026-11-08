"""The generated-task training pool: tasks that train again, the pool reviews, and the tasks they retire."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import pytest
from tests.support.samples import open_sample_store
from tests.support.task_service_stubs import StandInGenerator, task_for

from reef.artifact import SnapshotRef
from reef.core.job_data import JobOutcome
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask
from reef.execution.jobs import BackgroundJobRunner, BackgroundJobWorker, SupportsReceipt
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks import GeneratedTaskJobData
from reef.recipe.generated_tasks.generator import (
    GeneratedTaskResult,
    RolledOutTask,
    TaskGenerator,
    TaskValidationResult,
)
from reef.recipe.generated_tasks.processor import GeneratedTaskSamples, GeneratedTaskSettings, TaskJobSettings
from reef.recipe.generated_tasks.results import rollout_results
from reef.recipe.generated_tasks.review import PoolReview, RetiredTask, pool_reviews, retired_tasks, task_uses
from reef.record2sample.generator.inputs import TaskGenerationRequest
from reef.storage.commits import CommitRecord
from reef.storage.samples import stage_item_id
from reef.storage.scenario import ScenarioStore
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


class CommittingPipeline(TrainingPipeline):
    """A pipeline whose ``acknowledge`` commits what the job data consumed, as the trainer's commit of a job's
    submission does, and whose look for job data makes the samples first, as the samples service does."""

    def __init__(self, context: PipelineContext, session: ScenarioStore, **stages: Any) -> None:
        super().__init__(context, **stages)
        self.session = session

    def advance(self) -> None:
        self.produce()
        super().advance()

    def acknowledge(self, data_id: str) -> frozenset[str]:
        consumed = super().acknowledge(data_id)
        step = len(self.session.history())
        commit = CommitRecord(
            scenario="tasks",
            step=step + 1,
            snapshot=SnapshotRef(f"release:{step + 1}", "base"),
            checkpoint=True,
            algorithm_state={},
            high_water_sequence=step + 1,
            high_water_offset=step + 1,
            consumed_ids=consumed,
            recorded_at=1000.0 + step,
        )
        self.session.commit_step(expected_step=step, commit=commit)
        return consumed


def pipeline(
    tmp_path: Path,
    task_generator: TaskGenerator,
    *,
    rounds: int = 1,
    jobs_per_round: int = 1,
    refill_tasks: int = 1,
    job_samples: int = 2,
    **pool_config: float,
) -> CommittingPipeline:
    """A generated-task pipeline of the scenario ``tasks`` on SQLite storage under ``tmp_path``. Each job takes
    ``job_samples`` tasks; ``pool_config`` is the rest of ``pipeline.sample2job.config``."""
    worker = InlineWorker()
    session, samples = open_sample_store(tmp_path / "records", "tasks")
    context = PipelineContext("tasks", task_storage=session.tasks, samples=samples.for_trainer(""))
    stage = GeneratedTaskSamples(
        context,
        GeneratedTaskSettings(
            rounds=rounds, jobs_per_round=jobs_per_round, description="shell tasks", refill_tasks=refill_tasks
        ),
        task_generator=task_generator,
        generator_service=StandInGenerator(tmp_path / "service"),
        worker=worker,
    )
    worker.stage = stage
    jobs = GeneratedTaskJobData.from_config(
        {"job_samples": job_samples, "buffer_size": job_samples, **pool_config},
        context=context,
        record2sample=stage,
        environ={},
    )
    return CommittingPipeline(context, session, record2sample=stage, sample2job=jobs)


def task_stage(p: TrainingPipeline) -> GeneratedTaskSamples:
    stage = p.record2sample
    assert isinstance(stage, GeneratedTaskSamples)
    return stage


def task_store(p: TrainingPipeline) -> TaskStore:
    store = task_stage(p).task_store
    assert store is not None
    return store


def job_samples(p: TrainingPipeline) -> int:
    count = p.sample2job.job_samples
    assert count is not None
    return count


def handed_out(p: TrainingPipeline) -> list[str]:
    """Look for job data, then return the names of the tasks that the next job takes."""
    p.ready(job_samples(p))
    return [sample.metadata["task"]["name"] for _, sample in p.auto_entries()]


def waiting(p: TrainingPipeline) -> list[str]:
    """Look for job data, then return the names of the tasks whose sample waits, in the order that jobs take them."""
    p.ready(job_samples(p))
    samples = task_stage(p).task_items().values()
    untrained = sorted(
        (sample for found in samples for sample in found if not sample.train_counts),
        key=lambda sample: sample.sequence,
    )
    return [sample.sample.metadata["task"]["name"] for sample in untrained]


def finished(job_id: str, rewards: Mapping[str, Sequence[float]]) -> JobOutcome:
    return JobOutcome(
        job_id, "tinker", "finished", {"rollout_rewards": {name: list(r) for name, r in rewards.items()}}
    )


def submit(p: TrainingPipeline) -> list[str]:
    """Submit the next task job as the trainer does: its submission's commit counts its tasks trained. Return the
    names of its tasks. A round's tasks are stored on the look after the one that started the round."""
    for _ in range(3):
        if p.ready(job_samples(p)):
            break
    data = p.build_job_data(job_samples(p))
    p.acknowledge(data.data_id)
    return [sample.metadata["task"]["name"] for sample in data.samples]


def train(p: TrainingPipeline, job_id: str, rewards: Mapping[str, Sequence[float]] | None = None) -> list[str]:
    """Train the next task job as the trainer does; each task's rollouts earn ``rewards[name]``, 1.0 and 0.0 by default."""
    names = submit(p)
    p.job_ended(finished(job_id, {name: (rewards or {}).get(name, (1.0, 0.0)) for name in names}))
    return names


# ------------------------------------------------------------------------------------------------- reuse


def test_without_pool_capacity_a_task_trains_once_and_leaves_the_pool(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)))
    assert train(p, "job-1") == [TASK_A, TASK_B]
    assert waiting(p) == [] and handed_out(p) == []
    assert task_uses(task_store(p)) == {} and "training_pool" not in p.status()


def test_with_pool_capacity_a_task_returns_to_the_pool_after_its_job_ends(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 3)), pool_capacity=3, review_interval=0)
    assert train(p, "job-1") == [TASK_A, TASK_B]
    assert train(p, "job-2") == [TASK_C, TASK_A], "the tasks go back to the end of the pool"
    assert waiting(p) == [TASK_B, TASK_C, TASK_A]
    uses = task_uses(task_store(p))
    assert [(use.job_id, use.use_index) for use in uses[TASK_A]] == [("job-1", 0), ("job-2", 1)]
    samples = task_stage(p).task_items()[TASK_A]
    assert [sample.sample_id for sample in samples] == [
        stage_item_id("tasks", f"generated-tasks/{TASK_A}"),
        stage_item_id("tasks", f"generated-tasks/{TASK_A}@1"),
        stage_item_id("tasks", f"generated-tasks/{TASK_A}@2"),
    ]
    assert [sample.train_counts for sample in samples] == [{"": 1}, {"": 1}, {}], "each sample trains once"


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
    assert waiting(p) == [TASK_A, TASK_B], "each task has one sample that waits"


def test_a_job_end_repeated_while_the_next_job_runs_saves_nothing_for_the_next_job(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=0)
    train(p, "job-1")
    assert submit(p) == [TASK_A, TASK_B]
    p.job_ended(finished("job-1", {TASK_A: (1.0, 0.0), TASK_B: (1.0, 0.0)}))
    p.job_ended(finished("job-2", {TASK_A: (0.0, 0.0), TASK_B: (1.0, 1.0)}))
    uses = task_uses(task_store(p))
    assert [(use.job_id, use.use_index, use.rewards) for use in uses[TASK_A]] == [
        ("job-1", 0, (1.0, 0.0)),
        ("job-2", 1, (0.0, 0.0)),
    ]


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


def test_a_failed_job_returns_its_tasks_with_uses_that_have_no_rewards(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 3)), pool_capacity=3, review_interval=0)
    assert submit(p) == [TASK_A, TASK_B]
    p.job_ended(JobOutcome("job-1", "tinker", "failed", error="the sampler is gone"))
    assert {name: [use.rewards for use in found] for name, found in task_uses(task_store(p)).items()} == {
        TASK_A: [()],
        TASK_B: [()],
    }
    assert waiting(p) == [TASK_C, TASK_A, TASK_B]


def test_a_job_end_returns_a_cancelled_jobs_tasks_and_leaves_a_later_jobs_tasks(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 3)), job_samples=1, pool_capacity=3, review_interval=0)
    # A rollback cancels the first job, so its end never commits; the next job takes the next task.
    assert submit(p) == [TASK_A]
    assert submit(p) == [TASK_B]
    # The trainer submits the third job before the samples service reads the second job's end.
    assert submit(p) == [TASK_C]
    p.job_ended(finished("job-2", {TASK_B: (1.0, 0.0)}))
    uses = task_uses(task_store(p))
    assert {name: [use.job_id for use in found] for name, found in uses.items()} == {
        TASK_A: ["job-2"],
        TASK_B: ["job-2"],
    }
    assert uses[TASK_A][0].rewards == (), "the cancelled job ran no rollout that the end reports"
    assert waiting(p) == [TASK_A, TASK_B], "task C still trains"
    p.job_ended(finished("job-3", {TASK_C: (0.0, 1.0)}))
    assert task_uses(task_store(p))[TASK_C][0].rewards == (0.0, 1.0)
    assert waiting(p) == [TASK_A, TASK_B, TASK_C]


def test_two_job_ends_read_together_save_each_use_once(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 3)), job_samples=1, pool_capacity=3, review_interval=0)
    # The trainer submits the second job before the samples service reads the first job's end, and the service reads
    # both ends before it stores the next samples.
    assert submit(p) == [TASK_A] and submit(p) == [TASK_B]
    p.job_ended(finished("job-1", {TASK_A: (1.0, 0.0)}))
    p.job_ended(finished("job-2", {TASK_B: (0.0, 1.0)}))
    uses = task_uses(task_store(p))
    assert {name: [(use.job_id, use.rewards) for use in found] for name, found in uses.items()} == {
        TASK_A: [("job-1", (1.0, 0.0))],
        TASK_B: [("job-2", (0.0, 1.0))],
    }
    assert waiting(p) == [TASK_C, TASK_A, TASK_B]


def test_with_a_pool_rounds_keep_the_pace_of_jobs_per_round(tmp_path: Path) -> None:
    task_generator = FixedTasks(generated(0, 1), generated(1, 1))
    p = pipeline(
        tmp_path, task_generator, rounds=2, jobs_per_round=2, job_samples=1, pool_capacity=5, review_interval=0
    )
    train(p, "job-1")
    handed_out(p)
    assert len(task_store(p).generations()) == 1, "task A has its result, but it trains again, so a job end comes"
    train(p, "job-2")
    handed_out(p)
    assert len(task_store(p).generations()) == 2


def test_rounds_start_only_while_active_tasks_are_below_capacity(tmp_path: Path) -> None:
    task_generator = FixedTasks(generated(0, 2), generated(1, 2), generated(2, 2))
    p = pipeline(tmp_path, task_generator, rounds=3, job_samples=1, pool_capacity=3, review_interval=0)
    train(p, "job-1")
    handed_out(p)
    assert len(task_store(p).generations()) == 2, "two active tasks are below the capacity, so a round starts"
    for job in range(2, 6):
        train(p, f"job-{job}")
    assert len(task_store(p).generations()) == 2, "four active tasks fill the capacity, so no round starts"
    assert p.status()["training_pool"]["active"] == 4


def test_a_task_in_a_running_job_counts_as_active_for_the_capacity(tmp_path: Path) -> None:
    task_generator = FixedTasks(generated(0, 3), generated(1, 3))
    p = pipeline(tmp_path, task_generator, rounds=2, job_samples=1, pool_capacity=3, review_interval=0)
    train(p, "job-1")
    assert submit(p) == [TASK_B]
    assert waiting(p) == [TASK_C, TASK_A]
    assert len(task_store(p).generations()) == 1, "two waiting tasks and one task in a job fill the capacity"


# ------------------------------------------------------------------------------------------------- review


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
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=1, review_window=2)
    train(p, "job-1", {TASK_A: rewards})
    store = task_store(p)
    assert retired_tasks(store) == {TASK_A: RetiredTask(TASK_A, 0, rewards, reason)}
    assert pool_reviews(store) == (PoolReview(0, 1, 1, {TASK_A: reason}),)
    assert handed_out(p) == [TASK_B], "a retired task is not handed out again"


def test_a_review_retires_a_waiting_task_before_it_trains_again(tmp_path: Path) -> None:
    p = pipeline(
        tmp_path, FixedTasks(generated(0, 2)), job_samples=1, pool_capacity=2, review_interval=2, review_window=2
    )
    train(p, "job-1", {TASK_A: (1.0, 1.0)})
    assert waiting(p) == [TASK_B, TASK_A]
    train(p, "job-2")
    assert set(retired_tasks(task_store(p))) == {TASK_A}
    assert handed_out(p) == [TASK_B], "task A waited in the pool when the review retired it"
    assert waiting(p) == [TASK_B], "the scenario no longer uses the sample of task A that waited"
    (trained,) = task_stage(p).task_items()[TASK_A]
    assert trained.sample_id == stage_item_id("tasks", f"generated-tasks/{TASK_A}"), "the trained sample stays"


def test_a_review_takes_a_retired_task_out_of_the_next_jobs_hand_out(tmp_path: Path) -> None:
    task_d, task_e = "harbor-00000-003", "harbor-00000-004"
    p = pipeline(tmp_path, FixedTasks(generated(0, 5)), pool_capacity=5, review_interval=2, review_window=2)
    train(p, "job-1", {TASK_A: (1.0, 1.0)})
    assert submit(p) == [TASK_C, task_d]
    # While job-2 runs, a look for job data finds the next job's tasks.
    assert handed_out(p) == [task_e, TASK_A]
    p.job_ended(finished("job-2", {TASK_C: (1.0, 0.0), task_d: (1.0, 0.0)}))
    assert set(retired_tasks(task_store(p))) == {TASK_A}
    assert handed_out(p) == [task_e, TASK_B], "the next job takes the tasks without task A, in pool order"


def test_a_job_whose_data_was_taken_before_the_review_trains_the_retired_task_once_more(tmp_path: Path) -> None:
    task_d, task_e = "harbor-00000-003", "harbor-00000-004"
    p = pipeline(tmp_path, FixedTasks(generated(0, 5)), pool_capacity=5, review_interval=2, review_window=2)
    train(p, "job-1", {TASK_A: (1.0, 1.0)})
    assert submit(p) == [TASK_C, task_d]
    # The trainer takes the next job's data as soon as job-2 ends, before the samples service reads that end.
    data = p.build_job_data(2)
    p.job_ended(finished("job-2", {TASK_C: (1.0, 0.0), task_d: (1.0, 0.0)}))
    assert set(retired_tasks(task_store(p))) == {TASK_A}
    p.acknowledge(data.data_id)
    assert [sample.metadata["task"]["name"] for sample in data.samples] == [task_e, TASK_A]
    p.job_ended(finished("job-3", {task_e: (1.0, 0.0), TASK_A: (1.0, 1.0)}))
    assert TASK_A not in waiting(p), "the retired task trains no more"


def test_a_later_review_leaves_a_retired_task_out(tmp_path: Path) -> None:
    p = pipeline(
        tmp_path, FixedTasks(generated(0, 2)), job_samples=1, pool_capacity=2, review_interval=1, review_window=2
    )
    train(p, "job-1", {TASK_A: (1.0, 1.0)})
    train(p, "job-2", {TASK_B: (0.0, 1.0)})
    assert pool_reviews(task_store(p)) == (PoolReview(0, 1, 1, {TASK_A: "above_band"}), PoolReview(1, 2, 1, {}))


def test_a_review_retires_a_task_whose_next_sample_is_not_stored_yet(tmp_path: Path) -> None:
    p = pipeline(
        tmp_path, FixedTasks(generated(0, 2)), job_samples=1, pool_capacity=2, review_interval=2, review_window=2
    )
    assert submit(p) == [TASK_A] and submit(p) == [TASK_B]
    # The samples service reads both ends before it stores the sample of task A's next use.
    p.job_ended(finished("job-1", {TASK_A: (1.0, 1.0)}))
    p.job_ended(finished("job-2", {TASK_B: (0.0, 1.0)}))
    assert set(retired_tasks(task_store(p))) == {TASK_A}
    assert waiting(p) == [TASK_B]


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


def test_a_restart_keeps_the_pool_order_and_leaves_retired_tasks_out(tmp_path: Path) -> None:
    config = {"pool_capacity": 3, "review_interval": 1, "review_window": 2}
    first = pipeline(tmp_path, FixedTasks(generated(0, 3)), **config)
    train(first, "job-1", {TASK_A: (1.0, 1.0)})
    assert waiting(first) == [TASK_C, TASK_B]
    first.close()
    again = pipeline(tmp_path, FixedTasks(), **config)
    assert waiting(again) == [TASK_C, TASK_B], "the retired task stays out"
    assert again.status()["training_pool"]["active"] == 2


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
        ({"pool_capacity": 4, "job_samples": 0}, "job_samples must be None or a positive int"),
    ],
)
def test_pool_settings_refuse_bad_values(config: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TaskJobSettings(**config)


def test_a_pool_capacity_refuses_refill_tasks_above_1(tmp_path: Path) -> None:
    with pytest.raises(PluginConfigError, match="leave refill_tasks at 1"):
        pipeline(tmp_path, FixedTasks(), refill_tasks=2, pool_capacity=4)


def test_the_status_reports_retired_tasks_and_their_reasons(tmp_path: Path) -> None:
    p = pipeline(tmp_path, FixedTasks(generated(0, 2)), pool_capacity=2, review_interval=1, review_window=2)
    train(p, "job-1", {TASK_A: (0.0, 0.0)})
    assert handed_out(p) == [TASK_B]
    assert p.status()["training_pool"] == {
        "capacity": 2,
        "active": 1,
        "retired": 1,
        "reviews": 1,
        "last_review_retired": {TASK_A: "below_band"},
    }
