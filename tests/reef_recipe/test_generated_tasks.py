"""The generated-task recipe: rounds of generated tasks saved and sent back, and task jobs handed to the component."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import pytest
import sqlalchemy.exc
from tests.support._trajectories import policy_trajectory
from tests.support.runtime_stubs import StubTrainingComponent
from tests.support.task_service_stubs import StandInGenerator, task_for

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord, RequestType
from reef.core.job_data import JobData, JobOutcome, TaskSample, TrajectorySample
from reef.core.plugins import PluginConfigError
from reef.core.reports import ScoredRolloutReport
from reef.core.tasks import HarborTask, write_harbor_task
from reef.inference.http import InferenceProxyRuntime
from reef.pipeline_context import PipelineContext
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks import GeneratedTaskJobData, GeneratedTaskRecipe, TaskGroupObjective
from reef.recipe.generated_tasks.generator import (
    FailedTask,
    GeneratedTaskResult,
    HeldOutTask,
    RejectedTask,
    RolledOutTask,
    TaskGenerator,
    TaskValidationResult,
)
from reef.recipe.generated_tasks.processor import (
    DeliveryJob,
    GeneratedTaskRecord2Record,
    GeneratedTaskSamples,
    GenerationJob,
    GenerationOutcome,
    RoundSettings,
    TaskGroupSettings,
    TaskJobSettings,
    train_task_names,
)
from reef.recipe.generated_tasks.results import TaskRolloutResult, delivered_generations, rollout_results
from reef.recipe.generated_tasks.rollout import HandedOutTasks, PooledTask
from reef.record2record import InferenceRecord2Record
from reef.record2sample import NoSamples
from reef.record2sample.generator.inputs import TaskGenerationRequest
from reef.runtime.jobs import TrainingComponent
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import TaskStorage
from reef.train.algos.registry import resolve_objective
from reef.train.pipeline import TrainingPipeline

MODULE = "tests.reef_recipe.test_generated_tasks"
DESCRIPTION = "shell tasks with hidden state under /var and /etc"
#: The ``rollout`` job parameter of the recipe's task jobs.
ROLLOUT = {"episodes_per_task": 2}

# ------------------------------------------------------------------------------------------- the reported half


def inference(record_id: str) -> AgentRecord:
    payload = {
        "messages": [{"role": "user", "content": "ls"}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "ls -la"}}]},
        "training": {
            "tokens": [1, 2, 3],
            "loss_mask": [0, 1, 1],
            "rollout_log_probs": [0.0, -0.5, -0.7],
            "runtime_load_id": "r1",
        },
    }
    return AgentRecord.create(
        scenario="tasks", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=record_id
    )


def report(
    record_id: str,
    reference: str,
    task: str | None,
    score: float,
    *,
    arm: str | None = "plain",
    labels: Mapping[str, str] | None = None,
) -> AgentRecord:
    payload: dict[str, object] = {"score": score, "feedback": f"verifier reward {score}", "references": [reference]}
    metadata: dict[str, object] = {}
    if task is not None:
        metadata["task"] = {"name": task, "path": f"/tasks/{task}", "digest": "ab" * 32}
    if arm is not None:
        metadata["episode"] = {"id": record_id, "labels": {"arm": arm, **(labels or {})}}
    if metadata:
        payload["metadata"] = metadata
    return AgentRecord.create(
        scenario="tasks", request_type=RequestType.REPORT, payload=payload, agent_record_id=record_id
    )


def stage_settings(config: Mapping[str, object]) -> tuple[TaskGroupSettings, RoundSettings, TaskJobSettings]:
    """Split one flat config, as a test writes it, into the settings of the three stages."""
    group_names = {field.name for field in dataclasses.fields(TaskGroupSettings)}
    round_names = {field.name for field in dataclasses.fields(RoundSettings)}
    batch_names = {field.name for field in dataclasses.fields(TaskJobSettings)}
    unknown = sorted(set(config) - group_names - round_names - batch_names)
    if unknown:
        raise ValueError(f"unknown stage settings: {', '.join(unknown)}")
    return (
        TaskGroupSettings(**{key: value for key, value in config.items() if key in group_names}),
        RoundSettings(**{key: value for key, value in config.items() if key in round_names}),
        TaskJobSettings(**{key: value for key, value in config.items() if key in batch_names}),
    )


#: The trainer's sample count. A task job is the stage's own job data, whatever this is.
BATCH_SIZE = 4


def processor(**config: object) -> TrainingPipeline:
    context = PipelineContext("tasks")
    group_settings, round_settings, batch_settings = stage_settings({"tasks_per_step": 2, **config})
    stage = GeneratedTaskRecord2Record(context, group_settings)
    generation = GeneratedTaskSamples(context, round_settings, stage)
    return TrainingPipeline(
        context,
        record2record=stage,
        record2sample=generation,
        sample2job=GeneratedTaskJobData(context, batch_settings, record2record=stage, record2sample=generation),
    )


def job_stage(p: TrainingPipeline) -> GeneratedTaskJobData:
    """The pipeline's generated-task sample-to-job stage."""
    stage = p.sample2job
    assert isinstance(stage, GeneratedTaskJobData)
    return stage


def handed_out_names(p: TrainingPipeline) -> list[str]:
    """The names of the tasks that the stage hands out as the next job's data."""
    own = job_stage(p).own_job_data()
    return [] if own is None else [item.metadata["task"]["name"] for item in own]


def sample(task: str, index: int, reward: float) -> TrajectorySample:
    """One episode's sample in the task group ``task``, as the training component's rollout makes it."""
    trajectory = policy_trajectory(f"{task}-{index}", (1, 2, 3), (1, 1), (-0.1, -0.2), reward, "r1")
    return dataclasses.replace(trajectory, group_id=task)


def test_the_objective_centers_and_scales_rewards_within_each_task_group() -> None:
    first, second = "harbor-00000-000", "harbor-00000-001"
    data = JobData("d", (sample(first, 0, 1.0), sample(first, 1, 0.0), sample(second, 0, 0.5), sample(second, 1, 0.5)))
    objective = resolve_objective("task_group")
    assert isinstance(objective, TaskGroupObjective) and objective.loss_family == "importance_sampling"
    signal = objective.prepare(data, {})
    assert signal.action == "train"
    assert signal.advantages == (1.0, -1.0, 0.0, 0.0)
    assert signal.metrics["constant_groups"] == 1


def test_a_batch_whose_groups_are_all_constant_is_skipped_and_keeps_the_step_count() -> None:
    tasks = ("harbor-00000-000", "harbor-00000-001")
    data = JobData("d", tuple(sample(task, index, 1.0) for task in tasks for index in range(2)))
    signal = resolve_objective("task_group").prepare(data, {"steps": 3})
    assert signal.action == "skip" and signal.advantages is None
    assert signal.next_algorithm_state == {"steps": 3}
    assert signal.metrics == {"constant_groups": 2, "skipped": "every group is constant"}


# ------------------------------------------------------------------------------------------------- the rounds


def pooled(name: str) -> PooledTask:
    return PooledTask(name, 0, None, TaskSample(Path(name)))


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: RolledOutTask(task_for("t-a", "r", 1), ()), "non-empty tuple"),
        (lambda: RolledOutTask(task_for("t-a", "r", 1), (float("nan"),)), "finite"),
        (lambda: FailedTask(task_for("t-a", "r", 1), "", True), "needs the error"),
        (lambda: RejectedTask(task_for("t-a", "r", 1), ""), "needs its reason"),
        (lambda: TaskRolloutResult("t-a", (1.0,), error="refused"), "no rewards"),
        (lambda: TaskRolloutResult("t-a"), "non-empty tuple"),
        (lambda: TaskRolloutResult("t-a", (1.0,), infrastructure_failure=True), "needs the error"),
        (lambda: TaskGenerationRequest((), "tasks", generation=-1), "non-negative"),
        (lambda: TaskValidationResult(("",)), "non-empty"),
        (lambda: HandedOutTasks("b", ("b/t-a", "b/t-a"), (pooled("t-a"), pooled("t-b"))), "one distinct task use"),
        (lambda: HandedOutTasks("b", ("b/t-a",), (pooled("t-a"), pooled("t-b"))), "one distinct task use"),
    ],
)
def test_the_contract_values_refuse_what_the_recipe_cannot_run(build: Any, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build()


def test_a_rollout_result_reads_back_from_its_document_as_the_generators_result() -> None:
    task = task_for("t-a", "r", 1)
    ran = TaskRolloutResult("t-a", (1.0, 0.0))
    failed = TaskRolloutResult("t-a", error="the sandbox is gone", infrastructure_failure=True)
    assert TaskRolloutResult.from_document(ran.document()) == ran and ran.generated(task) == RolledOutTask(
        task, (1.0, 0.0)
    )
    assert failed.generated(task) == FailedTask(task, "the sandbox is gone", True)
    with pytest.raises(ValueError, match="is not the result of"):
        ran.generated(task_for("t-b", "r", 1))


class ScriptedGenerator(TaskGenerator):
    """A task generator that returns scripted tasks, refuses scripted names, and keeps what it was sent."""

    built: list[dict[str, object]] = []

    def __init__(
        self,
        rounds: list[tuple[HarborTask, ...]] | None = None,
        *,
        invalid: Mapping[str, str] | None = None,
        failing: str | None = None,
        failing_results: bool = False,
    ) -> None:
        self.rounds = list(rounds) if rounds is not None else []
        self.invalid = dict(invalid or {})
        self.failing = failing
        self.failing_results = failing_results
        self.requests: list[TaskGenerationRequest] = []
        self.validated: list[str] = []
        self.results: list[tuple[GeneratedTaskResult, ...]] = []
        self.result_generations: list[int] = []

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str]) -> Self:
        cls.built.append({"config": dict(config), "scenario": scenario, "environ": dict(environ)})
        return cls()

    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        self.requests.append(request)
        if not self.rounds:
            return ()
        return self.rounds.pop(0)

    async def validate(self, task: HarborTask) -> TaskValidationResult:
        self.validated.append(task.name)
        if task.name == self.failing:
            raise RuntimeError("the checks could not run")
        error = self.invalid.get(task.name)
        return TaskValidationResult((error,) if error else ())

    async def record_results(self, results: tuple[GeneratedTaskResult, ...], *, generation: int) -> None:
        self.results.append(results)
        self.result_generations.append(generation)
        if self.failing_results:
            raise RuntimeError("the generator lost its state directory")


class SlowGenerator(ScriptedGenerator):
    """A generator whose generate call takes longer than a short deadline."""

    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        await asyncio.sleep(1.0)
        return await super().generate(request)


def generated(round_number: int, *indices: int) -> tuple[HarborTask, ...]:
    """Tasks of one round."""
    return tuple(
        task_for(f"harbor-{round_number:05d}-{index:03d}", f"designer-{index}", 9000 + 10 * round_number + index)
        for index in indices
    )


class Unbound(BackgroundJobRunner):
    """The runner of a test worker that runs rounds itself. The real worker thread never calls it."""

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("no processor bound")


class InlineWorker(BackgroundJobWorker):
    """Runs each round the moment it is submitted, on this thread, so a test reads the outcome right away."""

    def __init__(self) -> None:
        super().__init__(Unbound(), thread_name="test-rounds", concurrency=1)
        self.generation: BackgroundJobRunner | None = None
        self.submitted: list[SupportsReceipt] = []
        self.outcomes: list[SupportsReceipt] = []

    def submit(self, job: SupportsReceipt) -> bool:
        self.submitted.append(job)
        if self.generation is None:
            return False
        try:
            self.outcomes.append(asyncio.run(self.generation.run(job)))
        except Exception:
            self.outcomes.append(Failed(job.receipt))
        return True

    def poll(self) -> list[SupportsReceipt]:
        outcomes, self.outcomes = self.outcomes, []
        return outcomes

    def close(self) -> None:
        self.generation = None


class DeferredWorker(BackgroundJobWorker):
    """Takes a round the moment it is submitted and runs it on ``finish``; the outcome lands on the next poll."""

    def __init__(self) -> None:
        super().__init__(Unbound(), thread_name="test-rounds", concurrency=1)
        self.generation: BackgroundJobRunner | None = None
        self.submitted: list[SupportsReceipt] = []
        self.waiting: list[SupportsReceipt] = []
        self.finished: list[SupportsReceipt] = []

    def submit(self, job: SupportsReceipt) -> bool:
        self.submitted.append(job)
        self.waiting.append(job)
        return True

    def finish(self) -> None:
        """Run every job taken so far, on this thread; the next poll hands their outcomes back."""
        if self.generation is None:
            raise RuntimeError("no generation bound")
        jobs, self.waiting = self.waiting, []
        for job in jobs:
            self.finished.append(asyncio.run(self.generation.run(job)))

    def poll(self) -> list[SupportsReceipt]:
        outcomes, self.finished = self.finished, []
        return outcomes

    def close(self) -> None:
        self.generation = None


def task_storage(tmp_path: Path, scenario: str = "tasks") -> TaskStorage:
    """The task stores of a scenario session in a SQLite storage under ``tmp_path``."""
    return SQLiteScenarioStorage(tmp_path / "records").open(scenario).tasks


def rounds(
    tmp_path: Path,
    task_generator: ScriptedGenerator,
    service: StandInGenerator | None = None,
    *,
    worker: InlineWorker | DeferredWorker | None = None,
    **config: object,
) -> tuple[TrainingPipeline, StandInGenerator]:
    """A generated-task pipeline; ``worker`` runs the generation and delivery jobs, inline by default."""
    service = service if service is not None else StandInGenerator(tmp_path / "tasks")
    worker = worker if worker is not None else InlineWorker()
    context = PipelineContext("tasks", task_storage=task_storage(tmp_path))
    group_settings, round_settings, batch_settings = stage_settings(
        {"tasks_per_step": 2, "rounds": 3, "description": DESCRIPTION, **config}
    )
    stage = GeneratedTaskRecord2Record(context, group_settings)
    generation = GeneratedTaskSamples(
        context, round_settings, stage, task_generator=task_generator, generator_service=service, worker=worker
    )
    batches = GeneratedTaskJobData(context, batch_settings, record2record=stage, record2sample=generation)
    built = TrainingPipeline(context, record2record=stage, record2sample=generation, sample2job=batches)
    worker.generation = generation
    return built, service


def looked(p: TrainingPipeline) -> bool:
    """Look for job data three times: a generation's outcome lands on the look after its submit."""
    p.ready(BATCH_SIZE)
    p.ready(BATCH_SIZE)
    return p.ready(BATCH_SIZE)


def train_job(
    p: TrainingPipeline, rewards: Mapping[str, Sequence[float]] | None = None, *, error: str | None = None
) -> list[str]:
    """Train the handed-out task job as the trainer and the training component do, and return its task names.

    The trainer commits the job's submission, then the job's end. Each task's episodes earn ``rewards[name]``,
    1.0 and 0.0 by default. A job with ``error`` failed and reports no metrics.
    """
    data = p.build_job_data(BATCH_SIZE)
    names = [item.metadata["task"]["name"] for item in data.samples]
    p.acknowledge(data.data_id)
    # The trainer releases the consumed IDs once the submission's commit is durable.
    p.release_records(p.releasable_record_ids())
    if error is None:
        episode_rewards = {name: list((rewards or {}).get(name, (1.0, 0.0))) for name in names}
        outcome = JobOutcome("job", "tinker", "finished", {"rollout_rewards": episode_rewards, "rollout_samples": 4})
    else:
        outcome = JobOutcome("job", "tinker", "failed", error=error)
    p.job_ended(outcome)
    return names


def stage_taken(p: TrainingPipeline) -> int:
    """The tasks that the handed-out and submitted task jobs hold."""
    return len(job_stage(p).task_pool.taken)


def generation_numbers(worker: InlineWorker | DeferredWorker) -> list[int]:
    """The generations that the worker took, in order."""
    return [job.generation for job in worker.submitted if isinstance(job, GenerationJob)]


def test_the_first_look_for_a_batch_runs_round_zero_end_to_end(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1) + generated(0, 2)])
    p, service = rounds(tmp_path, task_generator, tasks_per_step=3)
    status = p.status()["task_generation"]
    assert status["generator"] == "ScriptedGenerator" and (status["next"], status["completed"], status["of"]) == (
        0,
        0,
        3,
    )
    assert looked(p), "the three saved tasks are the next job's data"
    status = p.status()["task_generation"]
    assert (status["next"], status["completed"]) == (1, 1)
    assert p.status()["task_store"]["available"] == 3, "no task trained yet"

    assert task_generator.requests == [TaskGenerationRequest((), DESCRIPTION)]
    names = ["harbor-00000-000", "harbor-00000-001", "harbor-00000-002"]
    assert task_generator.validated == names
    data = p.build_job_data(BATCH_SIZE)
    assert [item.metadata["task"]["name"] for item in data.samples] == names
    assert all(isinstance(item, TaskSample) and (item.task_path / "instruction.md").is_file() for item in data.samples)
    assert service.writes == [] and service.episode_requests == [], "Reef runs no episode of a task job"
    assert task_generator.results == [], "no result before the job ends"

    train_job(p, dict.fromkeys(names, (0.25, 0.25)))
    (results,) = task_generator.results
    assert task_generator.result_generations == [0]
    assert [result.task.name for result in results] == names
    assert all(isinstance(result, RolledOutTask) and result.rollout_rewards == (0.25, 0.25) for result in results)
    store = p.record2sample.task_store
    saved = rollout_results(store)
    assert 0 in delivered_generations(store) and sorted(saved) == sorted(names)
    assert saved[names[0]].rollout_rewards == (0.25, 0.25)
    assert p.status()["task_store"]["trained"] == 3


@pytest.mark.unit
def test_eval_tasks_are_saved_without_episodes(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    pipeline, _ = rounds(tmp_path, task_generator, eval_fraction=1.0)
    assert not looked(pipeline), "no eval task is handed out"
    assert all(isinstance(result, HeldOutTask) for result in task_generator.results[0])
    assert pipeline.status()["task_store"] == {
        "train": 0,
        "eval": 2,
        "trained": 0,
        "available": 0,
        "rejected": 0,
        "pending": 0,
    }
    assert 0 in delivered_generations(pipeline.record2sample.task_store), "no train task waits to run"
    assert pipeline.record2sample.task_store.generation(0).is_complete
    pipeline.close()


@pytest.mark.unit
def test_a_check_that_cannot_run_is_retried_from_the_saved_candidates_then_skipped(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)], failing="harbor-00000-001")
    pipeline, _ = rounds(tmp_path, task_generator, rounds=1, generation_attempts=2)
    assert not looked(pipeline), "a generation whose attempts ran out hands out none of its tasks"
    assert len(task_generator.requests) == 1, "a retry continues from the saved candidates"
    assert task_generator.validated == ["harbor-00000-000", "harbor-00000-001", "harbor-00000-001"]
    assert pipeline.record2sample.task_store.generation(0).pending_indices == (1,)
    assert "the checks could not run" in pipeline.status()["task_generation"]["last_error"]
    task_generator.failing = None
    outcome = asyncio.run(pipeline.record2sample.run(GenerationJob(receipt="generation-00000", generation=0)))
    assert isinstance(outcome, GenerationOutcome) and len(outcome.tasks) == 2
    assert len(task_generator.requests) == 1
    assert pipeline.record2sample.task_store.generation(0).is_complete
    pipeline.close()


@pytest.mark.unit
def test_a_generation_past_its_deadline_is_skipped(tmp_path: Path) -> None:
    task_generator = SlowGenerator([generated(0, 0)])
    pipeline, _ = rounds(tmp_path, task_generator, generation_deadline_seconds=0.05)
    assert not looked(pipeline)
    assert pipeline.record2sample.task_store.generation(0).candidates is None
    assert pipeline.status()["task_generation"]["last_error"]
    pipeline.close()


def sourced(name: str, *record_ids: str, port: int, designer_record_id: str = "") -> HarborTask:
    """A task made from ``record_ids``; ``designer_record_id`` names the one that is the author's own record."""
    task = task_for(name, record_ids[0], port)
    metadata = {**task.metadata, "designer_record_id": designer_record_id} if designer_record_id else task.metadata
    return dataclasses.replace(task, source_agent_record_ids=record_ids, metadata=metadata)


@pytest.mark.unit
def test_a_task_joins_the_split_of_its_sources_and_a_task_that_joins_both_splits_is_refused(tmp_path: Path) -> None:
    follower = sourced("follower", "designer-1", "eval-source", port=9102, designer_record_id="designer-1")
    bridge = sourced("bridge", "train-source", "eval-source", port=9103)
    renamed = sourced("train-task", "other-source", port=9104)
    task_generator = ScriptedGenerator([(follower, bridge, renamed)])
    pipeline, _ = rounds(tmp_path, task_generator)
    store = pipeline.record2sample.task_store
    # The author's record is a source of a train task too; it must not join the follower to the train split.
    earlier = (
        sourced("train-task", "designer-1", "train-source", port=9100, designer_record_id="designer-1"),
        sourced("eval-task", "eval-source", port=9101),
    )
    store.begin_generation(generation=100, source_record_ids=())
    store.save_candidates(
        generation=100, directories=tuple(write_harbor_task(task, tmp_path / "earlier") for task in earlier)
    )
    store.publish_task(generation=100, index=0, split="train")
    store.publish_task(generation=100, index=1, split="eval")
    store.complete_generation(generation=100)

    assert not looked(pipeline)

    results = {result.task.name: result for result in task_generator.results[0]}
    assert isinstance(results["follower"], HeldOutTask)
    bridge, train_task = results["bridge"], results["train-task"]
    assert isinstance(bridge, RejectedTask) and "split conflict" in bridge.reason
    assert isinstance(train_task, RejectedTask) and "other files" in train_task.reason
    assert {task.name: task.split for task in store.tasks()} == {
        "train-task": "train",
        "eval-task": "eval",
        "follower": "eval",
    }
    assert store.path(store.tasks(split="train")[0].files_digest).joinpath("instruction.md").is_file()
    assert pipeline.status()["task_store"] == {
        "train": 1,
        "eval": 2,
        "trained": 0,
        "available": 1,
        "rejected": 2,
        "pending": 0,
    }
    pipeline.close()


def test_an_invalid_task_comes_back_with_its_errors_and_is_never_handed_out(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator(
        [generated(0, 0, 1)], invalid={"harbor-00000-000": "oracle check refused: the oracle scored 0"}
    )
    p, _ = rounds(tmp_path, task_generator)
    assert looked(p), "the valid task is handed out"
    assert handed_out_names(p) == ["harbor-00000-001"]
    assert p.status()["task_store"]["available"] == 1
    train_job(p)
    (results,) = task_generator.results
    assert results[0] == RejectedTask(results[0].task, "oracle check refused: the oracle scored 0")
    assert isinstance(results[1], RolledOutTask)


def test_a_round_that_ran_no_task_does_not_block_the_next_one(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator([]), worker=worker)
    assert not p.ready(BATCH_SIZE) and generation_numbers(worker) == [0]
    worker.finish()
    assert not p.ready(BATCH_SIZE), "round 0 lands without a task and round 1 starts on the same look"
    assert generation_numbers(worker) == [0, 1], "no batch can come; none was acknowledged"


def test_a_restarted_processor_carries_on_from_its_task_store(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), rounds=2)
    assert looked(p) and p.status()["task_generation"]["completed"] == 1
    p.close()
    task_generator = ScriptedGenerator([generated(1, 0)])
    again, _ = rounds(tmp_path, task_generator, StandInGenerator(tmp_path / "tasks"), rounds=2)
    status = again.status()["task_generation"]
    assert (status["in_flight"], status["next"], status["completed"]) == (None, 1, 1)
    looked(again)
    assert again.status()["task_generation"]["completed"] == 2
    looked(again)
    assert again.status()["task_generation"]["completed"] == 2, "the cap is reached"
    assert len(task_generator.requests) == 1


def test_results_that_the_generator_did_not_take_are_sent_again_by_a_later_job(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)], failing_results=True)
    p, _ = rounds(tmp_path, task_generator, rounds=1)
    looked(p)
    train_job(p)
    p.ready(BATCH_SIZE)
    assert p.status()["task_generation"]["last_error"] == "the generator lost its state directory"
    assert 0 not in delivered_generations(p.record2sample.task_store)
    task_generator.failing_results = False
    assert asyncio.run(p.record2sample.deliver_results()) == ""
    assert task_generator.result_generations == [0, 0], "the second delivery reached it"
    assert 0 in delivered_generations(p.record2sample.task_store)
    assert asyncio.run(p.record2sample.deliver_results()) == "" and len(task_generator.results) == 2


def test_a_worker_that_refuses_a_round_stops_the_rounds_with_the_reason(tmp_path: Path) -> None:
    worker = InlineWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), worker=worker, rounds=2)
    worker.close()
    assert not looked(p) and not p.background_jobs_running()
    assert generation_numbers(worker) == [0], "no round is tried after the refusal"
    assert "no further round runs" in p.status()["task_generation"]["last_error"]


def test_a_restart_hands_out_the_saved_tasks_that_did_not_train_without_generating_again(tmp_path: Path) -> None:
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    looked(first)
    # The first task trained, with its result, before the stop.
    first.record2sample.save_result(TaskRolloutResult("harbor-00000-000", (1.0, 0.0)))
    first.record2sample.task_store.mark_trained(task_names=("harbor-00000-000",))
    first.close()
    task_generator = ScriptedGenerator()
    restarted, _ = rounds(tmp_path, task_generator, rounds=1)
    assert len(job_stage(restarted).task_pool) == 1, "the sibling that did not train is back in the task pool"
    looked(restarted)
    assert handed_out_names(restarted) == ["harbor-00000-001"] and task_generator.requests == []
    assert task_generator.result_generations == [], "the sibling has no result yet, so the generation waits"
    train_job(restarted)
    assert task_generator.result_generations == [0]


def test_a_task_that_an_earlier_generation_published_is_rejected_as_a_duplicate(tmp_path: Path) -> None:
    first = generated(0, 0, 1)
    task_generator = ScriptedGenerator([first, (first[0], *generated(1, 5))])
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, task_generator, worker=worker, rounds=2, jobs_per_round=1)
    for _ in range(3):
        p.ready(BATCH_SIZE)
        worker.finish()
    train_job(p)
    for _ in range(4):
        p.ready(BATCH_SIZE)
        worker.finish()
    train_job(p)
    worker.finish()
    p.ready(BATCH_SIZE)
    store = p.record2sample.task_store
    state = store.generation(1)
    assert state is not None and train_task_names(state) == ("harbor-00001-005",)
    assert state.candidates is not None and "is already published" in state.candidates[0].error
    assert task_generator.result_generations == [0, 1], "both generations get their results"


def test_a_restart_sends_results_that_the_generator_did_not_receive(tmp_path: Path) -> None:
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)], failing_results=True), rounds=1)
    looked(first)
    train_job(first)
    first.close()
    task_generator = ScriptedGenerator()
    worker = DeferredWorker()
    restarted, _ = rounds(tmp_path, task_generator, worker=worker, rounds=1)
    assert [job.receipt for job in worker.submitted] == ["delivery"], "no round is due, yet the results go out"
    worker.finish()
    restarted.ready(BATCH_SIZE)
    assert task_generator.result_generations == [0]
    assert 0 in delivered_generations(restarted.record2sample.task_store)


def test_a_generation_whose_results_fail_does_not_hold_back_later_ones(tmp_path: Path) -> None:
    task_generator = FirstResultsFail([generated(0, 0, 1), generated(1, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, rounds=2)
    looked(p)
    train_job(p)
    looked(p)
    train_job(p)
    store = p.record2sample.task_store
    assert 0 not in delivered_generations(store) and 1 in delivered_generations(store)


class FailingGenerator(ScriptedGenerator):
    """A generator whose generate call fails to run at once."""

    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        self.requests.append(request)
        raise RuntimeError("the generator service is down")


def test_a_generator_that_always_fails_stops_at_the_round_limit(tmp_path: Path) -> None:
    task_generator = FailingGenerator()
    p, _ = rounds(tmp_path, task_generator, rounds=3, generation_attempts=2)
    for _ in range(20):
        p.ready(BATCH_SIZE)
    assert len(task_generator.requests) == 6, "three rounds, two attempts each"
    assert p.record2record.rounds_stopped


class FirstResultsFail(ScriptedGenerator):
    """A generator that cannot take the results of generation 0."""

    async def record_results(self, results: tuple[GeneratedTaskResult, ...], *, generation: int) -> None:
        if generation == 0:
            raise RuntimeError("generation 0 is lost")
        await super().record_results(results, generation=generation)


# ------------------------------------------------------------------------------------------------- the task jobs


def test_saved_tasks_are_handed_out_as_one_task_job_and_marked_trained_once_its_submission_commits(
    tmp_path: Path,
) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1, 2)])
    p, _ = rounds(tmp_path, task_generator, rounds=1)
    assert not p.ready(BATCH_SIZE), "the generation saves three tasks"
    assert p.ready(BATCH_SIZE), "two of them are the next job's data, whatever the trainer's sample count"
    stage = job_stage(p)
    built = p.build_job_data(BATCH_SIZE)
    names = ["harbor-00000-000", "harbor-00000-001"]
    assert built.samples == stage.own_job_data() and handed_out_names(p) == names
    assert p.status()["handed_out"]["tasks"] == 2 and p.status()["task_pool"] == 1
    handed_out = stage.handed_out
    assert handed_out is not None
    assert p.acknowledge(built.data_id) == {f"rollout:{task_use}" for task_use in handed_out.task_uses}
    assert stage.handed_out is None and len(stage.task_pool.taken) == 2, "the job holds its tasks until it ends"
    assert p.status()["submitted_tasks"] == 2
    assert p.status()["task_store"]["trained"] == 0, "the commit is not durable yet"
    p.release_records(p.releasable_record_ids())
    assert p.record2sample.task_store.trained_task_names() == frozenset(names)
    assert task_generator.result_generations == [], "no result before the job ends"
    assert p.ready(BATCH_SIZE) and handed_out_names(p) == ["harbor-00000-002"], "the next job takes the rest"


def test_a_job_end_saves_each_tasks_first_result_from_its_rewards_and_sends_the_generation(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, rounds=1)
    looked(p)
    ran, none_ran = train_job(p, {"harbor-00000-001": ()})
    store = p.record2sample.task_store
    saved = rollout_results(store)
    assert saved[ran] == TaskRolloutResult(ran, (1.0, 0.0))
    assert saved[none_ran] == TaskRolloutResult(
        none_ran, error="no episode of the task ran", infrastructure_failure=True
    )
    assert task_generator.result_generations == [0] and 0 in delivered_generations(store)
    assert p.status()["submitted_tasks"] == 0 and stage_taken(p) == 0
    p.job_ended(JobOutcome("job", "tinker", "finished", {"rollout_rewards": {ran: [0.5, 0.5]}}))
    assert rollout_results(store)[ran].rollout_rewards == (1.0, 0.0), "a later run only trains"


def test_a_failed_job_gives_each_of_its_tasks_an_infrastructure_failure_with_the_jobs_error(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, rounds=1)
    looked(p)
    train_job(p, error="the sandbox provider is down")
    (results,) = task_generator.results
    assert results == tuple(FailedTask(result.task, "the sandbox provider is down", True) for result in results)
    assert len(results) == 2


def test_the_task_to_job_stage_needs_the_generated_task_stages() -> None:
    context = PipelineContext("tasks")
    with pytest.raises(PluginConfigError, match="needs GeneratedTaskRecord2Record"):
        GeneratedTaskJobData.from_config(
            {}, context=context, record2record=InferenceRecord2Record(), record2sample=NoSamples(), environ={}
        )


# --------------------------------------------------------------------------------------- source sessions

SOURCES = {"source_tag": "Source=session", "source_records_per_round": 2}


def session(name: str, tags: Mapping[str, str]) -> tuple[AgentRecord, AgentRecord]:
    """A recorded session: one tagged model call and the report on it, without a task or an arm."""
    call = inference(f"rec-{name}")
    call = AgentRecord.create(
        scenario="tasks",
        request_type=RequestType.INFERENCE,
        payload={**call.payload, "metadata": {"tags": dict(tags)}},
        agent_record_id=call.agent_record_id,
    )
    return call, report(f"rep-{name}", call.agent_record_id, None, 0.5, arm=None)


def sessions(p: TrainingPipeline, *names: str, tags: Mapping[str, str] | None = None) -> None:
    for name in names:
        for record in session(name, {"source": "session"} if tags is None else tags):
            p.push(record)


def source_ids(request: TaskGenerationRequest) -> list[str]:
    return [record.agent_record_id for record in request.source_records]


def test_each_round_gives_the_generator_at_most_its_share_of_tagged_sessions(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator()
    p, _ = rounds(tmp_path, task_generator, **SOURCES)
    sessions(p, "a", "b", "c")
    sessions(p, "untagged", tags={"source": "other"})
    for _ in range(3):
        looked(p)
    # Each round ran no task, so the next one followed at once.
    assert [source_ids(request) for request in task_generator.requests] == [
        ["rec-a", "rep-a", "rec-b", "rep-b"],
        ["rec-c", "rep-c"],
        [],
    ]
    assert all(request.description == DESCRIPTION for request in task_generator.requests)
    store = p.record2sample.task_store
    assert store.generation(0).source_record_ids == ("rec-a", "rep-a", "rec-b", "rep-b")
    assert store.generation(2).source_record_ids == ()


def test_a_source_sessions_records_stay_until_the_round_that_uses_it_finishes(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator(), worker=worker, **SOURCES)
    sessions(p, "a")
    sessions(p, "untagged", tags={})
    assert {"rec-untagged", "rep-untagged"} <= p.releasable_record_ids(), "an untagged session is released"
    assert not {"rec-a", "rep-a"} & p.releasable_record_ids()
    assert p.operational_metrics()["waiting_source_sessions"] == 1
    p.ready(BATCH_SIZE)
    assert [record.agent_record_id for record in worker.submitted[0].source_records] == ["rec-a", "rep-a"]
    assert not {"rec-a", "rep-a"} & p.releasable_record_ids(), "the round runs"
    worker.finish()
    p.ready(BATCH_SIZE)
    assert {"rec-a", "rep-a"} <= p.releasable_record_ids()


def test_without_a_share_of_sessions_the_rounds_read_only_the_description(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator()
    p, _ = rounds(tmp_path, task_generator, rounds=1, source_tag="source=session")
    sessions(p, "a")
    assert {"rec-a", "rep-a"} <= p.releasable_record_ids()
    looked(p)
    assert task_generator.requests == [TaskGenerationRequest((), DESCRIPTION)]


def test_no_session_waits_once_the_rounds_are_over(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator()
    p, _ = rounds(tmp_path, task_generator, rounds=1, **SOURCES)
    for _ in range(3):
        looked(p)
    sessions(p, "late")
    assert {"rec-late", "rep-late"} <= p.releasable_record_ids()
    assert len(task_generator.requests) == 1


def test_a_restart_does_not_generate_again_from_replayed_sessions_of_a_complete_generation(tmp_path: Path) -> None:
    worker = DeferredWorker()
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), worker=worker, **SOURCES)
    sessions(first, "a")
    first.ready(BATCH_SIZE)
    worker.finish()
    first.ready(BATCH_SIZE)
    first.close()
    task_generator = ScriptedGenerator()
    restarted, _ = rounds(tmp_path, task_generator, **SOURCES)
    sessions(restarted, "a")
    looked(restarted)
    assert [source_ids(request) for request in task_generator.requests][:1] == [[]], "session a is not used again"


def test_replayed_sessions_of_a_complete_generation_do_not_use_up_a_round(tmp_path: Path) -> None:
    config: dict[str, Any] = {"rounds": 2, "source_tag": "source=session", "source_records_per_round": 1}
    worker = DeferredWorker()
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), worker=worker, **config)
    sessions(first, "a")
    first.ready(BATCH_SIZE)
    worker.finish()
    first.ready(BATCH_SIZE)
    first.close()
    task_generator = ScriptedGenerator()
    restarted, _ = rounds(tmp_path, task_generator, **config)
    sessions(restarted, "a", "b")
    for _ in range(6):
        restarted.ready(BATCH_SIZE)
    assert [source_ids(request) for request in task_generator.requests] == [["rec-b", "rep-b"]]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"tasks_per_step": 0}, "tasks_per_step must be positive"),
        ({"rounds": 1, "description": "x"}, "pipeline.record2sample.config.generator"),
        ({"source_records_per_round": 1}, "needs a source_tag"),
        ({"source_tag": "session"}, "name=value"),
    ],
)
def test_a_processor_configuration_that_cannot_train_is_refused(config: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        processor(**config)


def test_rounds_need_a_description_the_generator_service_and_task_storage() -> None:
    def built(**config: object) -> GeneratedTaskSamples:
        context = PipelineContext("tasks")
        group_settings, round_settings, _ = stage_settings({"rounds": 1, **config})
        stage = GeneratedTaskRecord2Record(context, group_settings)
        return GeneratedTaskSamples(context, round_settings, stage, task_generator=ScriptedGenerator())

    with pytest.raises(ValueError, match="description"):
        built()
    with pytest.raises(ValueError, match="generator_url"):
        built(description="x")
    with pytest.raises(ValueError, match="generator_url"):
        built(description="x", generator_url="${endpoints.generator}")
    with pytest.raises(ValueError, match="task storage"):
        built(description="x", generator_url="http://g:8910")


# ------------------------------------------------------------------------------------------------- the recipe


def training_recipe(**fields: Any) -> GeneratedTaskRecipe:
    runtime = InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8000")
    return GeneratedTaskRecipe(
        StubTrainingComponent(), runtime=runtime, **{"job_parameters": {"rollout": ROLLOUT}, **fields}
    )


def test_the_recipe_binds_the_objective_and_its_stages() -> None:
    spec = GeneratedTaskRecipe.training_spec()
    assert GeneratedTaskRecipe.pipeline_spec().sample2job is GeneratedTaskJobData and spec.objective == "task_group"
    assert spec.loss_family == "importance_sampling"
    assert TaskGroupObjective.rollout_unit == "sample"
    assert GeneratedTaskRecipe.report_type.fget(GeneratedTaskRecipe) is ScoredRolloutReport  # type: ignore[union-attr]
    recipe = training_recipe()
    assert recipe.job_plan().experiment_config()["parameters"]["rollout"] == ROLLOUT, "every job is a task job"
    built = recipe.build_pipeline(PipelineContext("one"))
    assert isinstance(built.record2record, GeneratedTaskRecord2Record)
    assert (
        isinstance(built.record2sample, GeneratedTaskSamples) and built.record2sample.worker is None
    ), "no round runs"
    assert isinstance(built.sample2job, GeneratedTaskJobData) and built.sample2job.tasks_per_step == 4
    assert built.buffer_size is None, "the task pool holds the tasks that wait"
    built.close()


@pytest.mark.parametrize("job_parameters", [None, {}, {"rollout": 4}])
def test_the_recipe_refuses_jobs_without_the_rollout_job_parameter(job_parameters: Any) -> None:
    with pytest.raises(RecipeConfigError, match=r"needs the rollout parameter as an object"):
        training_recipe(job_parameters=job_parameters)


def test_the_recipe_refuses_a_training_component_that_runs_no_task_jobs() -> None:
    class TrajectoryOnly(StubTrainingComponent):
        check_task_jobs = TrainingComponent.check_task_jobs

    runtime = InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8000")
    with pytest.raises(RecipeConfigError, match="the stub training component trains no task jobs"):
        GeneratedTaskRecipe(TrajectoryOnly(), runtime=runtime, job_parameters={"rollout": ROLLOUT})


def generated_pipeline_recipe(environ: Mapping[str, str], pipeline: Mapping[str, Any]) -> WeightTrainingRecipe:
    return GeneratedTaskRecipe.from_environment(
        environ,
        config={
            "data": {"job_parameters": {"rollout": ROLLOUT}},
            "model": {"path": "Qwen/Qwen3-8B"},
            "pipeline": pipeline,
        },
        runtime=InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8000"),
        training_component=StubTrainingComponent(),
    )


def test_the_pipeline_section_configures_the_stages_and_builds_the_generator_once_for_each_scenario(
    tmp_path: Path,
) -> None:
    generator = {"module": f"{MODULE}:ScriptedGenerator", "config": {"count": 3}}
    environ = {"TASKS_KEY": "secret"}
    recipe = generated_pipeline_recipe(
        environ,
        {
            "record2record": {"module": "reef.recipe.generated_tasks:GeneratedTaskRecord2Record"},
            "record2sample": {
                "module": "reef.recipe.generated_tasks:GeneratedTaskSamples",
                "config": {
                    "rounds": 1,
                    "generator_url": "http://g:8910",
                    "description": "shell tasks",
                    "generator": generator,
                },
            },
            "sample2job": {
                "module": "reef.recipe.generated_tasks:GeneratedTaskJobData",
                "config": {"tasks_per_step": 2},
            },
        },
    )
    ScriptedGenerator.built.clear()
    built = recipe.build_pipeline(PipelineContext("one", task_storage=task_storage(tmp_path, "one")))
    assert isinstance(built.record2sample, GeneratedTaskSamples)
    assert isinstance(built.record2sample.task_generator, ScriptedGenerator)
    assert ScriptedGenerator.built == [{"config": {"count": 3}, "scenario": "one", "environ": environ}]
    assert built.record2sample.task_store is not None
    batches = built.sample2job
    assert isinstance(batches, GeneratedTaskJobData) and batches.tasks_per_step == 2
    built.close()


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        ({"record2sample": {"module": "reef.recipe.generated_tasks:GeneratedTaskJobData"}}, "inherits Record2Sample"),
        ({"record2samples": {"module": "reef.recipe.generated_tasks:GeneratedTaskSamples"}}, "unknown key"),
    ],
)
def test_a_pipeline_section_that_names_no_usable_stage_is_refused(pipeline: Mapping[str, Any], message: str) -> None:
    with pytest.raises(RecipeConfigError, match=message):
        generated_pipeline_recipe({}, pipeline)


@pytest.mark.parametrize(
    ("generator", "message"),
    [
        (f"{MODULE}:ScriptedGenerator.missing", "cannot import pipeline.record2sample.config.generator"),
        ("reef.recipe.generated_tasks:GeneratedTaskRecipe", "must name a class that inherits TaskGenerator"),
        ("reef.recipe.generated_tasks:TaskGenerator", "names an abstract class"),
    ],
)
def test_a_generator_setting_that_names_no_usable_generator_is_refused(generator: str, message: str) -> None:
    # The round settings check the generator setting when the scenario's pipeline builds.
    recipe = generated_pipeline_recipe(
        {},
        {
            "record2sample": {
                "module": "reef.recipe.generated_tasks:GeneratedTaskSamples",
                "config": {"generator": {"module": generator}},
            }
        },
    )
    with pytest.raises(RecipeConfigError, match=message):
        recipe.build_pipeline(PipelineContext("one"))


def test_the_rollout_settings_of_a_task_job_are_no_sample2job_config() -> None:
    sample2job = {"module": "reef.recipe.generated_tasks:GeneratedTaskJobData", "config": {"episodes_per_task": 4}}
    recipe = generated_pipeline_recipe({}, {"sample2job": sample2job})
    with pytest.raises(RecipeConfigError, match="unknown config fields: episodes_per_task"):
        recipe.build_pipeline(PipelineContext("one"))


def test_a_stage_without_from_config_cannot_be_named_in_the_pipeline_section() -> None:
    recipe = generated_pipeline_recipe({}, {"record2record": {"module": "reef.record2record:JudgingRecord2Record"}})
    with pytest.raises(RecipeConfigError, match="JudgingRecord2Record cannot be built from a deployment file"):
        recipe.build_pipeline(PipelineContext("one"))


def test_a_task_whose_result_cannot_be_saved_still_trains_and_the_next_job_saves_it(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]))
    store = p.record2sample.task_store
    write_document = store.write_document

    def failing_write_document(*, key: str, document: str) -> None:
        if key == "rollout/harbor-00000-000":
            raise OSError(28, "No space left on device")
        write_document(key=key, document=document)

    store.write_document = failing_write_document
    looked(p)
    train_job(p)
    assert set(rollout_results(store)) == {"harbor-00000-001"}
    assert p.status()["task_store"]["trained"] == 2, "the job's submission committed, so both tasks trained"
    store.write_document = write_document
    assert asyncio.run(p.record2sample.run(DeliveryJob(receipt="delivery"))).error == ""
    assert 0 in delivered_generations(store), "the next job saved the result, so the feedback went out"


def test_rounds_continue_when_the_task_pool_drains_before_batches_per_round(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1), generated(1, 2, 3)])
    p, _ = rounds(tmp_path, task_generator, rounds=2, jobs_per_round=2)
    for _ in range(3):
        p.ready(BATCH_SIZE)
    train_job(p)
    for _ in range(3):
        p.ready(BATCH_SIZE)
    assert p.status()["task_generation"]["completed"] == 2, "no job data can come, so the next round starts"
    handed_out = job_stage(p).handed_out
    assert handed_out is not None and all("harbor-00001" in task_use for task_use in handed_out.task_uses)


def test_a_generation_whose_task_result_failed_to_save_once_still_gets_its_feedback(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, rounds=1)
    store = p.record2sample.task_store
    write_document = store.write_document
    failures = {"count": 0}

    def flaky_write_document(*, key: str, document: str) -> None:
        if key == "rollout/harbor-00000-000" and failures["count"] == 0:
            failures["count"] += 1
            raise OSError(28, "No space left on device")
        write_document(key=key, document=document)

    store.write_document = flaky_write_document
    looked(p)
    train_job(p)
    assert failures["count"] == 1
    assert task_generator.result_generations == [0], "the delivery job saved the result again, then delivered"


def test_a_skipped_generation_is_finished_by_the_next_round(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)], failing="harbor-00000-001")
    p, _ = rounds(tmp_path, task_generator, rounds=3, generation_attempts=1)
    p.ready(BATCH_SIZE)
    p.ready(BATCH_SIZE)
    store = p.record2sample.task_store
    assert not store.generation(0).is_complete
    assert p.status()["task_store"]["train"] == 1 and p.status()["task_store"]["available"] == 0
    task_generator.failing = None
    for _ in range(3):
        p.ready(BATCH_SIZE)
    assert store.generation(0).is_complete and len(task_generator.requests) == 1
    assert p.status()["task_store"]["available"] == 2, "the same generation finished from its saved candidates"


def test_a_task_job_that_ran_across_a_restart_saves_its_rewards_when_it_ends(tmp_path: Path) -> None:
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    looked(first)
    data = first.build_job_data(BATCH_SIZE)
    first.acknowledge(data.data_id)
    first.release_records(first.releasable_record_ids())
    first.close()
    task_generator = ScriptedGenerator()
    again, _ = rounds(tmp_path, task_generator, rounds=1)
    assert rollout_results(again.record2sample.task_store) == {}, "the running job's tasks wait for its end"
    rewards = {"harbor-00000-000": [1.0, 1.0], "harbor-00000-001": [0.0, 1.0]}
    again.job_ended(JobOutcome("job", "tinker", "finished", {"rollout_rewards": rewards}))
    saved = rollout_results(again.record2sample.task_store)
    assert {name: result.rollout_rewards for name, result in saved.items()} == {
        "harbor-00000-000": (1.0, 1.0),
        "harbor-00000-001": (0.0, 1.0),
    }


def test_the_end_of_the_next_job_gives_a_cancelled_jobs_tasks_a_failure(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1, tasks_per_step=1)
    looked(p)
    # A rollback cancels the first job, so its end never commits; the next job takes the other task.
    p.acknowledge(p.build_job_data(BATCH_SIZE).data_id)
    p.release_records(p.releasable_record_ids())
    assert p.ready(BATCH_SIZE) and handed_out_names(p) == ["harbor-00000-001"]
    train_job(p)
    saved = rollout_results(p.record2sample.task_store)
    assert saved["harbor-00000-001"].rollout_rewards == (1.0, 0.0)
    assert saved["harbor-00000-000"].infrastructure_failure and stage_taken(p) == 0


def test_a_task_that_trained_before_its_result_was_saved_gets_it_when_the_restart_repeats_the_end(
    tmp_path: Path,
) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    store = p.record2sample.task_store
    write_document = store.write_document

    def failing_write_document(*, key: str, document: str) -> None:
        if key == "rollout/harbor-00000-000":
            raise OSError(28, "No space left on device")
        write_document(key=key, document=document)

    # The first task's result is never saved, but both tasks train before the stop.
    store.write_document = failing_write_document
    looked(p)
    train_job(p)
    p.close()
    task_generator = ScriptedGenerator()
    again, _ = rounds(tmp_path, task_generator, rounds=1)
    assert len(job_stage(again).task_pool) == 0, "a trained task does not run again"
    # The trainer starts with no running job, so it gives the stage the last job's end again.
    rewards = {"harbor-00000-000": [1.0, 0.0], "harbor-00000-001": [1.0, 0.0]}
    again.job_ended(JobOutcome("job", "tinker", "finished", {"rollout_rewards": rewards}))
    assert rollout_results(again.record2sample.task_store)["harbor-00000-000"].rollout_rewards == (1.0, 0.0)
    assert asyncio.run(again.record2sample.deliver_results()) == "" and task_generator.result_generations == [0]


def test_a_result_that_the_database_fails_to_save_still_trains_and_is_saved_later(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    store = p.record2sample.task_store
    write_document = store.write_document

    def locked_write_document(*, key: str, document: str) -> None:
        if key == "rollout/harbor-00000-000":
            raise sqlalchemy.exc.OperationalError("INSERT", {}, Exception("database is locked"))
        write_document(key=key, document=document)

    store.write_document = locked_write_document
    looked(p)
    train_job(p)
    assert p.status()["task_store"]["trained"] == 2, "both tasks trained"
    store.write_document = write_document
    assert asyncio.run(p.record2sample.run(DeliveryJob(receipt="delivery"))).error == ""
    assert 0 in delivered_generations(store)


def test_a_round_that_ran_out_of_attempts_is_retried_with_the_same_sessions(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1), generated(1, 5)], failing="harbor-00000-001")
    p, _ = rounds(tmp_path, task_generator, rounds=3, generation_attempts=1, **SOURCES)
    sessions(p, "a", "b", "c", "d")
    p.ready(BATCH_SIZE)
    p.ready(BATCH_SIZE)
    task_generator.failing = None
    for _ in range(3):
        looked(p)
    store = p.record2sample.task_store
    state = store.generation(0)
    assert state is not None and state.is_complete, "the next round finished the same generation"
    assert state.source_record_ids == ("rec-a", "rep-a", "rec-b", "rep-b")
    assert len(task_generator.requests) == 1


def test_an_identical_candidate_in_the_same_generation_is_rejected_as_a_duplicate(tmp_path: Path) -> None:
    task = generated(0, 0)
    task_generator = ScriptedGenerator([task + task])
    p, _ = rounds(tmp_path, task_generator, rounds=1)
    looked(p)
    state = p.record2sample.task_store.generation(0)
    assert state is not None and train_task_names(state) == ("harbor-00000-000",)
    assert state.candidates is not None and "is already published" in state.candidates[1].error


def test_a_task_whose_submission_did_not_commit_is_handed_out_again_after_a_restart(tmp_path: Path) -> None:
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    assert looked(first) and job_stage(first).handed_out is not None
    first.acknowledge(first.build_job_data(BATCH_SIZE).data_id)
    # The process stops before the submission's commit is durable, so no ID is released.
    assert first.record2sample.task_store.trained_task_names() == frozenset()
    first.close()
    restarted, _ = rounds(tmp_path, ScriptedGenerator(), rounds=1)
    assert len(job_stage(restarted).task_pool) == 2, "both tasks are handed out again"


def test_a_generation_outcome_that_is_incomplete_holds_no_task_and_names_its_error() -> None:
    with pytest.raises(ValueError, match="incomplete generation"):
        GenerationOutcome(receipt="generation-00000", generation=0, tasks=(), is_complete=False, error="")
    with pytest.raises(ValueError, match="non-negative"):
        GenerationJob(receipt="generation-x", generation=-1)
