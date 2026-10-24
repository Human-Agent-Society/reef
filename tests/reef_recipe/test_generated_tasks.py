"""The generated-task recipe: rounds of generated tasks run and sent back, episodes grouped by task and trained."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import pytest
import sqlalchemy.exc
from tests.support.runtime_stubs import StubTrainingRuntime
from tests.support.task_service_stubs import StandInGenerator, rollout_turn, task_for

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord, RequestType
from reef.core.batches import TaskItem
from reef.core.plugins import PluginConfigError
from reef.core.reports import ScoredRolloutReport
from reef.core.tasks import HarborTask, read_harbor_task, write_harbor_task
from reef.harness.client.tasks import EpisodeTurn, TaskEpisode
from reef.inference.http import InferenceProxyRuntime
from reef.pipeline_context import PipelineContext
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks import GeneratedTaskBatches, GeneratedTaskRecipe, TaskGroupObjective
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
    GeneratedTaskTasks,
    GenerationJob,
    GenerationOutcome,
    RolloutJob,
    RoundSettings,
    TaskBatchSettings,
    TaskGroupSettings,
    episode_assembly,
    rewards_of,
    train_task_names,
)
from reef.recipe.generated_tasks.results import TaskRolloutResult, delivered_generations, rollout_results
from reef.recipe.generated_tasks.rollout import ActiveRollout, PooledTask
from reef.record2record import InferenceRecord2Record
from reef.record2task import NoTasks
from reef.record2task.generator.client import GeneratorError
from reef.record2task.generator.inputs import TaskGenerationRequest
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import TaskStorage
from reef.train.algos.registry import resolve_objective
from reef.train.pipeline import TrainingPipeline

MODULE = "tests.reef_recipe.test_generated_tasks"
DESCRIPTION = "shell tasks with hidden state under /var and /etc"

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


def stage_settings(config: Mapping[str, object]) -> tuple[TaskGroupSettings, RoundSettings, TaskBatchSettings]:
    """Split one flat config, as a test writes it, into the settings of the three stages."""
    group_names = {field.name for field in dataclasses.fields(TaskGroupSettings)}
    round_names = {field.name for field in dataclasses.fields(RoundSettings)}
    batch_names = {field.name for field in dataclasses.fields(TaskBatchSettings)}
    unknown = sorted(set(config) - group_names - round_names - batch_names)
    if unknown:
        raise ValueError(f"unknown stage settings: {', '.join(unknown)}")
    return (
        TaskGroupSettings(**{key: value for key, value in config.items() if key in group_names}),
        RoundSettings(**{key: value for key, value in config.items() if key in round_names}),
        TaskBatchSettings(**{key: value for key, value in config.items() if key in batch_names}),
    )


#: The trainer's batch size of episodes. A rollout batch trains on its settled groups, whatever this is.
BATCH_SIZE = 4


def processor(**config: object) -> TrainingPipeline:
    context = PipelineContext("tasks")
    group_settings, round_settings, batch_settings = stage_settings(
        {"tasks_per_step": 2, "episodes_per_task": 2, **config}
    )
    stage = GeneratedTaskRecord2Record(context, group_settings)
    generation = GeneratedTaskTasks(context, round_settings, stage)
    return TrainingPipeline(
        context,
        record2record=stage,
        record2task=generation,
        task2batch=GeneratedTaskBatches(
            context, batch_settings, record2record=stage, record2task=generation, worker=InlineWorker()
        ),
    )


def episode(task: str, index: int, reward: float | None, *turns: EpisodeTurn) -> TaskEpisode:
    """One rollout episode of ``task`` with these answered calls, or one call of its own."""
    return TaskEpisode(
        Path(task),
        task,
        f"{task}-{index}",
        reward,
        {},
        "",
        (),
        0,
        (),
        None,
        turns=turns or (rollout_turn(None, index),),
    )


def rollout_of(p: TrainingPipeline) -> ActiveRollout | None:
    """The rollout batch of the pipeline's generated-task stage."""
    stage = p.task2batch
    assert isinstance(stage, GeneratedTaskBatches)
    return stage.rollout


def ended_rollout(p: TrainingPipeline, episodes: Mapping[str, tuple[TaskEpisode, ...]]) -> None:
    """A rollout batch of these tasks whose episodes ended: each task's samples are its group."""
    batch = "tasks:rollout:test"
    stage = p.task2batch
    assert isinstance(stage, GeneratedTaskBatches)
    samples = {
        f"{batch}/{task}": tuple(sample for sample in (stage.episode_sample(run, task) for run in runs) if sample)
        for task, runs in episodes.items()
    }
    tasks = tuple(PooledTask(task, index, None, TaskItem(Path(task))) for index, task in enumerate(episodes))
    stage.rollout = ActiveRollout(batch, tuple(samples), tasks, samples=samples, fresh_uses=tuple(samples))


def test_rewards_of_counts_an_unscored_run_as_a_loss_and_skips_an_episode_that_never_ran() -> None:
    def episode(reward: float | None, error: str = "") -> TaskEpisode:
        return TaskEpisode(Path("t"), "t", "e", reward, {}, error, (), 0, (), None)

    assert rewards_of((episode(1.0), episode(None), episode(None, "the trial raised"))) == (1.0, 0.0)
    assert rewards_of((episode(0.5, "a late error"),)) == (0.5,)
    assert (
        episode(None, "the sandbox is gone").is_infrastructure_failure and not episode(0.0).is_infrastructure_failure
    )


def test_the_assembly_spans_an_episodes_turns_and_realigns_the_think_scaffold() -> None:
    assembly = episode_assembly()
    assert assembly.settings.accept_multi_turn_policy_samples and assembly.settings.scaffold_tolerance == 8
    assert episode_assembly(2).settings.scaffold_tolerance == 2


def test_an_episode_whose_response_was_cut_at_max_tokens_still_trains() -> None:
    # Tinker's default max_tokens is 1024: a cut response plus the four-token think scaffold drifts 1028 tokens.
    scaffold, response = [90, 91, 92, 93], list(range(1000, 2024))

    def turn(tokens: list[int], response_length: int) -> EpisodeTurn:
        training = {
            "tokens": tokens,
            "loss_mask": [1] * response_length,
            "rollout_log_probs": [-0.1] * response_length,
            "runtime_load_id": "r1",
        }
        return EpisodeTurn(request={"messages": []}, response={"training": training})

    p = processor(tasks_per_step=1)
    task = "harbor-00000-000"
    cut = episode(
        task, 0, 1.0, turn([1, 2, *scaffold, *response], len(response)), turn([1, 2, *response, 3, *scaffold, 4], 1)
    )
    ended_rollout(p, {task: (cut, episode(task, 1, 0.0, turn([1, 2, 3], 1)))})
    assert p.ready(BATCH_SIZE)
    assert [sample.training.get("turn_count", 1) for sample in p.build_batch(BATCH_SIZE).items] == [2, 1]


def test_an_episode_with_an_aborted_call_has_no_sample() -> None:
    p = processor()
    stage = p.task2batch
    assert isinstance(stage, GeneratedTaskBatches)
    task = "harbor-00000-000"
    answered = rollout_turn(None, 0)
    aborted = EpisodeTurn(
        request=answered.request, response={**answered.response, "choices": [{"finish_reason": "abort"}]}
    )
    assert stage.episode_sample(episode(task, 0, 1.0, answered), task) is not None
    assert stage.episode_sample(episode(task, 1, 1.0, answered, aborted), task) is None


def test_the_objective_centers_and_scales_rewards_within_each_task_group() -> None:
    p = processor()
    first, second = "harbor-00000-000", "harbor-00000-001"
    ended_rollout(
        p,
        {
            first: (episode(first, 0, 1.0), episode(first, 1, 0.0)),
            second: (episode(second, 0, 0.5), episode(second, 1, 0.5)),
        },
    )
    objective = resolve_objective("task_group")
    assert isinstance(objective, TaskGroupObjective) and objective.loss_family == "importance_sampling"
    signal = objective.prepare(p.build_batch(BATCH_SIZE), {})
    assert signal.action == "train"
    assert signal.advantages == (1.0, -1.0, 0.0, 0.0)
    assert signal.metrics["constant_groups"] == 1


def test_a_batch_whose_groups_are_all_constant_is_skipped_and_keeps_the_step_count() -> None:
    p = processor()
    ended_rollout(
        p, {task: (episode(task, 0, 1.0), episode(task, 1, 1.0)) for task in ("harbor-00000-000", "harbor-00000-001")}
    )
    signal = resolve_objective("task_group").prepare(p.build_batch(BATCH_SIZE), {"steps": 3})
    assert signal.action == "skip" and signal.advantages is None
    assert signal.next_algorithm_state == {"steps": 3}
    assert signal.metrics == {"constant_groups": 2, "skipped": "every group is constant"}


# ------------------------------------------------------------------------------------------------- the rounds


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
    """A generated-task pipeline with an inline rollout worker; ``worker`` runs the generation jobs."""
    service = service if service is not None else StandInGenerator(tmp_path / "tasks")
    worker = worker if worker is not None else InlineWorker()
    rollout_worker = InlineWorker()
    context = PipelineContext("tasks", task_storage=task_storage(tmp_path))
    group_settings, round_settings, batch_settings = stage_settings(
        {
            "tasks_per_step": 2,
            "episodes_per_task": 2,
            "rounds": 3,
            "description": DESCRIPTION,
            "served_model": "Qwen/Qwen3-8B",
            **config,
        }
    )
    stage = GeneratedTaskRecord2Record(context, group_settings)
    generation = GeneratedTaskTasks(
        context, round_settings, stage, task_generator=task_generator, generator_service=service, worker=worker
    )
    batches = GeneratedTaskBatches(
        context, batch_settings, record2record=stage, record2task=generation, worker=rollout_worker
    )
    built = TrainingPipeline(context, record2record=stage, record2task=generation, task2batch=batches)
    worker.generation = generation
    rollout_worker.generation = batches
    return built, service


def looked(p: TrainingPipeline) -> bool:
    """Look for a batch three times: a generation's outcome, then its rollout's, land on the look after each submit."""
    p.ready(BATCH_SIZE)
    p.ready(BATCH_SIZE)
    return p.ready(BATCH_SIZE)


def train_rollout(p: TrainingPipeline, *, trained: bool = True) -> None:
    """Wait for the running rollout's episodes, then build and acknowledge its batch."""
    assert rollout_of(p) is not None, "a rollout runs"
    assert p.ready(BATCH_SIZE), "the rollout's episodes ended"
    p.acknowledge(p.build_batch(BATCH_SIZE).batch_id, trained=trained)
    # The trainer releases the consumed IDs once the step's commit is durable.
    p.release_records(p.releasable_record_ids())


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
    assert looked(p), "the rollout of the three tasks can train once its episodes end"
    status = p.status()["task_generation"]
    assert (status["next"], status["completed"]) == (1, 1)
    assert p.status()["task_store"]["available"] == 3, "no task trained yet"

    assert task_generator.requests == [TaskGenerationRequest((), DESCRIPTION)]
    names = ["harbor-00000-000", "harbor-00000-001", "harbor-00000-002"]
    assert task_generator.validated == names and service.writes == names
    arms = [(call["task"], call["arm"], call["count"], call["extra"]) for call in service.episode_requests]
    assert arms == [(name, "plain", 2, ()) for name in names]
    assert all(call["tags"]["round"] == "0" and call["model"] == "Qwen/Qwen3-8B" for call in service.episode_requests)

    (results,) = task_generator.results
    assert task_generator.result_generations == [0]
    assert [result.task.name for result in results] == names
    assert all(isinstance(result, RolledOutTask) and result.rollout_rewards == (0.25, 0.25) for result in results)
    store = p.record2task.task_store
    saved = rollout_results(store)
    assert 0 in delivered_generations(store) and sorted(saved) == sorted(names)
    assert saved[names[0]].rollout_rewards == (0.25, 0.25)


@pytest.mark.unit
def test_eval_tasks_are_saved_without_episodes(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    pipeline, service = rounds(tmp_path, task_generator, eval_fraction=1.0)
    assert not looked(pipeline)
    assert service.episode_requests == [] and service.writes == []
    assert all(isinstance(result, HeldOutTask) for result in task_generator.results[0])
    assert pipeline.status()["task_store"] == {
        "train": 0,
        "eval": 2,
        "trained": 0,
        "available": 0,
        "rejected": 0,
        "pending": 0,
    }
    assert 0 in delivered_generations(pipeline.record2task.task_store), "no train task waits to run"
    assert pipeline.record2task.task_store.generation(0).is_complete
    pipeline.close()


@pytest.mark.unit
def test_a_check_that_cannot_run_is_retried_from_the_saved_candidates_then_skipped(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)], failing="harbor-00000-001")
    pipeline, service = rounds(tmp_path, task_generator, rounds=1, generation_attempts=2)
    assert not looked(pipeline)
    assert len(task_generator.requests) == 1, "a retry continues from the saved candidates"
    assert task_generator.validated == ["harbor-00000-000", "harbor-00000-001", "harbor-00000-001"]
    assert pipeline.record2task.task_store.generation(0).pending_indices == (1,)
    assert service.writes == [], "a generation whose attempts ran out runs none of its tasks"
    assert "the checks could not run" in pipeline.status()["task_generation"]["last_error"]
    task_generator.failing = None
    outcome = asyncio.run(pipeline.record2task.run(GenerationJob(receipt="generation-00000", generation=0)))
    assert isinstance(outcome, GenerationOutcome) and len(outcome.tasks) == 2
    assert len(task_generator.requests) == 1
    assert pipeline.record2task.task_store.generation(0).is_complete
    pipeline.close()


@pytest.mark.unit
def test_a_generation_past_its_deadline_is_skipped(tmp_path: Path) -> None:
    task_generator = SlowGenerator([generated(0, 0)])
    pipeline, service = rounds(tmp_path, task_generator, generation_deadline_seconds=0.05)
    assert not looked(pipeline)
    assert service.writes == [] and pipeline.record2task.task_store.generation(0).candidates is None
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
    pipeline, service = rounds(tmp_path, task_generator)
    store = pipeline.record2task.task_store
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
    assert service.writes == []
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


def test_an_invalid_task_comes_back_with_its_errors_and_is_never_written(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator(
        [generated(0, 0, 1)], invalid={"harbor-00000-000": "oracle check refused: the oracle scored 0"}
    )
    p, service = rounds(tmp_path, task_generator)
    assert looked(p), "the valid task rolls out"
    assert service.writes == ["harbor-00000-001"]
    (results,) = task_generator.results
    assert results[0] == RejectedTask(results[0].task, "oracle check refused: the oracle scored 0")
    assert isinstance(results[1], RolledOutTask)
    assert p.status()["task_store"]["available"] == 1


def test_a_refused_write_or_a_task_the_agent_cannot_run_comes_back_without_episodes(tmp_path: Path) -> None:
    taken = StandInGenerator(tmp_path / "tasks", taken=("harbor-00000-000",))
    task_generator = ScriptedGenerator([generated(0, 0)])
    p, _ = rounds(tmp_path, task_generator, taken)
    assert not looked(p)
    (refused,) = task_generator.results[0]
    assert isinstance(refused, FailedTask) and not refused.infrastructure_failure
    assert refused.error.startswith("refused: a different task holds the name harbor-00000-000")
    assert taken.episode_requests == []

    cannot_run = StandInGenerator(tmp_path / "second" / "tasks", episode_error="Failed to start tmux session.")
    task_generator = ScriptedGenerator([generated(0, 0)])
    p, _ = rounds(tmp_path / "second", task_generator, cannot_run)
    assert not looked(p)
    assert [call["arm"] for call in cannot_run.episode_requests] == ["plain"]
    assert cannot_run.deleted == [], "the staged copy stays for the next run"
    (failed,) = task_generator.results[0]
    assert failed == FailedTask(failed.task, "the agent could not run the task: Failed to start tmux session.", True)


class PartlyReported(StandInGenerator):
    """Episodes whose first ``reported`` reach Reef; the rest end unreported, as a failed trial does."""

    def __init__(self, root: Path, *, reported: int) -> None:
        super().__init__(root)
        self.reported = reported

    async def run_episodes(self, task_path, **fields) -> tuple[TaskEpisode, ...]:
        episodes = await super().run_episodes(task_path, **fields)
        return tuple(
            episode if index < self.reported else dataclasses.replace(episode, report_agent_record_ids=())
            for index, episode in enumerate(episodes)
        )


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
    p, service = rounds(tmp_path, task_generator)
    looked(p)
    assert service.writes == ["harbor-00000-000", "harbor-00000-001"]
    p.ready(BATCH_SIZE)
    assert p.status()["task_generation"]["last_error"] == "the generator lost its state directory"
    assert 0 not in delivered_generations(p.record2task.task_store)
    task_generator.failing_results = False
    assert asyncio.run(p.record2task.deliver_results()) == ""
    assert task_generator.result_generations == [0, 0], "the second delivery reached it"
    assert 0 in delivered_generations(p.record2task.task_store)
    assert asyncio.run(p.record2task.deliver_results()) == "" and len(task_generator.results) == 2


def test_a_worker_that_refuses_a_round_stops_the_rounds_with_the_reason(tmp_path: Path) -> None:
    worker = InlineWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), worker=worker, rounds=2)
    worker.close()
    assert not looked(p) and not p.background_jobs_running()
    assert generation_numbers(worker) == [0], "no round is tried after the refusal"
    assert "no further round runs" in p.status()["task_generation"]["last_error"]


def test_a_restart_rolls_out_the_saved_tasks_that_did_not_train_without_generating_again(tmp_path: Path) -> None:
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    looked(first)
    # The rollout ran both tasks; the first one's group trained before the stop.
    first.record2task.task_store.mark_trained(task_names=("harbor-00000-000",))
    first.close()
    task_generator = ScriptedGenerator()
    service = StandInGenerator(tmp_path / "tasks")
    restarted, _ = rounds(tmp_path, task_generator, service, rounds=1)
    assert len(restarted.task2batch.task_pool) == 1, "the sibling that did not train is back in the task pool"
    looked(restarted)
    assert service.writes == ["harbor-00000-001"] and task_generator.requests == []
    assert task_generator.result_generations == [], "the generation's results went out before the stop"


def test_a_task_that_cannot_run_is_an_infrastructure_failure(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0)])
    p, _ = rounds(tmp_path, task_generator, BrokenRunner(tmp_path / "tasks"), rounds=1)
    looked(p)
    ((result,),) = task_generator.results
    assert result == FailedTask(result.task, "the sandbox provider is down", True)


class BrokenRunner(StandInGenerator):
    """A generator service whose episodes cannot run."""

    async def run_episodes(self, task_path, **fields) -> tuple[TaskEpisode, ...]:
        raise GeneratorError("the sandbox provider is down")


def test_a_task_that_an_earlier_generation_published_is_rejected_as_a_duplicate(tmp_path: Path) -> None:
    first = generated(0, 0, 1)
    task_generator = ScriptedGenerator([first, (first[0], *generated(1, 5))])
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, task_generator, worker=worker, rounds=2, batches_per_round=1)
    for _ in range(3):
        p.ready(BATCH_SIZE)
        worker.finish()
    train_rollout(p)
    for _ in range(4):
        p.ready(BATCH_SIZE)
        worker.finish()
    p.ready(BATCH_SIZE)
    store = p.record2task.task_store
    state = store.generation(1)
    assert state is not None and train_task_names(state) == ("harbor-00001-005",)
    assert state.candidates is not None and "is already published" in state.candidates[0].error
    assert task_generator.result_generations == [0, 1], "both generations get their results"


def test_a_restart_sends_results_that_the_generator_did_not_receive(tmp_path: Path) -> None:
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)], failing_results=True), rounds=1)
    looked(first)
    first.close()
    task_generator = ScriptedGenerator()
    worker = DeferredWorker()
    restarted, _ = rounds(tmp_path, task_generator, worker=worker, rounds=1)
    assert [job.receipt for job in worker.submitted] == [
        "delivery"
    ], "no round or rollout is due, yet the results go out"
    worker.finish()
    restarted.ready(BATCH_SIZE)
    assert task_generator.result_generations == [0]
    assert 0 in delivered_generations(restarted.record2task.task_store)


def test_a_generation_whose_results_fail_does_not_hold_back_later_ones(tmp_path: Path) -> None:
    task_generator = FirstResultsFail([generated(0, 0, 1), generated(1, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, rounds=2)
    looked(p)
    train_rollout(p)
    looked(p)
    store = p.record2task.task_store
    assert 0 not in delivered_generations(store) and 1 in delivered_generations(store)


def test_a_task_whose_episodes_a_stop_interrupted_is_staged_again_and_runs(tmp_path: Path) -> None:
    worker = DeferredWorker()
    service = StandInGenerator(tmp_path / "tasks")
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), service, worker=worker, rounds=1)
    first.ready(BATCH_SIZE)
    worker.finish()
    first.ready(BATCH_SIZE)
    # The rollout wrote the task through the generator service, then the process stopped before its result was saved.
    store = first.record2task.task_store
    asyncio.run(service.write_task(read_harbor_task(store.path(store.tasks()[0].files_digest))))
    first.close()
    task_generator = ScriptedGenerator()
    restarted, _ = rounds(tmp_path, task_generator, service, rounds=1)
    looked(restarted)
    ((result,),) = task_generator.results
    assert (
        isinstance(result, RolledOutTask) and service.deleted == []
    ), "staging the same task again reuses the staged copy"


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


def rollout_rounds(
    tmp_path: Path, task_generator: ScriptedGenerator, service: StandInGenerator | None = None, **config: object
) -> tuple[TrainingPipeline, StandInGenerator, InlineWorker]:
    """A generated-task pipeline that rolls saved tasks out, with inline workers for generation and rollout."""
    service = service if service is not None else StandInGenerator(tmp_path / "tasks")
    worker, rollout_worker = InlineWorker(), InlineWorker()
    context = PipelineContext("tasks", task_storage=task_storage(tmp_path))
    group_settings, round_settings, batch_settings = stage_settings(
        {
            "tasks_per_step": 2,
            "episodes_per_task": 2,
            "rounds": 1,
            "description": DESCRIPTION,
            **config,
        }
    )
    stage = GeneratedTaskRecord2Record(context, group_settings)
    generation = GeneratedTaskTasks(
        context, round_settings, stage, task_generator=task_generator, generator_service=service, worker=worker
    )
    batches = GeneratedTaskBatches(
        context, batch_settings, record2record=stage, record2task=generation, worker=rollout_worker
    )
    built = TrainingPipeline(context, record2record=stage, record2task=generation, task2batch=batches)
    worker.generation = generation
    rollout_worker.generation = batches
    return built, service, rollout_worker


def test_saved_tasks_roll_out_as_one_batch_and_train_on_the_samples_of_their_episodes(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    p, service, rollout_worker = rollout_rounds(tmp_path, task_generator)
    assert not p.ready(BATCH_SIZE), "the generation saves two tasks"
    assert not p.ready(BATCH_SIZE), "the rollout of both starts"
    (job,) = rollout_worker.submitted
    assert isinstance(job, RolloutJob)
    names = [task.metadata["task"]["name"] for _, task in job.task_uses]
    assert [call["task"] for call in service.episode_requests] == names
    assert p.ready(BATCH_SIZE), "the episodes ended, and each task's two samples form its group"
    built = p.build_batch(BATCH_SIZE)
    assert [item.group_id for item in built.items] == [names[0], names[0], names[1], names[1]]
    assert all(not item.source_agent_record_ids for item in built.items), "a rollout sample comes from no record"
    assert p.status()["ready_samples"] == 0, "no record reached the record stage"
    assert p.acknowledge(built.batch_id) == {f"rollout:{task_use}" for task_use, _ in job.task_uses}
    assert p.status()["task_store"]["trained"] == 0, "the commit is not durable yet"
    p.release_records(p.releasable_record_ids())
    assert p.status()["rollout"] is None
    assert task_generator.result_generations == [0], "the generation's results went out after its rollout"
    assert p.status()["task_store"]["trained"] == 2


def test_stale_groups_do_not_train_and_their_tasks_roll_out_again(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    service.runtime_load_id = "v1"
    p, _, rollout_worker = rollout_rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), service)
    for version in ("v1", "v2", "v3"):
        p.observe_serving_runtime_load_id(version)
    assert not p.ready(BATCH_SIZE), "the generation saves two tasks"
    assert not p.ready(BATCH_SIZE), "the rollout of both starts"
    assert not p.ready(BATCH_SIZE), "both groups are two versions behind, so none trains, and the tasks roll out again"
    first, second = rollout_worker.submitted
    assert [task.metadata["task"] for _, task in first.task_uses] == [
        task.metadata["task"] for _, task in second.task_uses
    ]
    service.runtime_load_id = "v3"
    assert not p.ready(BATCH_SIZE), "the second rollout's groups are stale too, so a third starts on v3"
    assert p.ready(BATCH_SIZE), "the groups of the version that the scenario serves train"


def test_a_rollout_without_a_group_lets_its_tasks_wait_for_a_restart(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks", episode_error="the sandbox is gone")
    p, _, _ = rollout_rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), service)
    assert not looked(p)
    stage = p.task2batch
    assert isinstance(stage, GeneratedTaskBatches)
    assert stage.rollout is None and stage.task_pool.is_idle, "no task has a group, so none goes back to the pool"


def test_a_rollout_that_the_worker_refuses_keeps_its_tasks_in_the_pool(tmp_path: Path) -> None:
    p, _, rollout_worker = rollout_rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]))
    rollout_worker.close()
    assert not p.ready(BATCH_SIZE), "the generation saves two tasks"
    with pytest.raises(RuntimeError, match="cannot run"):
        p.ready(BATCH_SIZE)
    assert len(p.task2batch.task_pool) == 2


def test_the_batch_stage_needs_the_generated_task_stages() -> None:
    context = PipelineContext("tasks")
    with pytest.raises(PluginConfigError, match="needs GeneratedTaskRecord2Record"):
        GeneratedTaskBatches.from_config(
            {}, context=context, record2record=InferenceRecord2Record(), record2task=NoTasks(), environ={}
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
    store = p.record2task.task_store
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
        ({"episodes_per_task": 1}, "at least two"),
        ({"rounds": 1, "description": "x"}, "pipeline.record2task.config.generator"),
        ({"source_records_per_round": 1}, "needs a source_tag"),
        ({"source_tag": "session"}, "name=value"),
    ],
)
def test_a_processor_configuration_that_cannot_train_is_refused(config: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        processor(**config)


def test_rounds_need_a_description_the_generator_service_and_task_storage() -> None:
    def built(**config: object) -> GeneratedTaskTasks:
        context = PipelineContext("tasks")
        group_settings, round_settings, _ = stage_settings({"rounds": 1, **config})
        stage = GeneratedTaskRecord2Record(context, group_settings)
        return GeneratedTaskTasks(context, round_settings, stage, task_generator=ScriptedGenerator())

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
    return GeneratedTaskRecipe(training_runtime=StubTrainingRuntime(), runtime=runtime, **fields)


def test_the_recipe_binds_the_objective_its_schedule_and_its_stages() -> None:
    spec = GeneratedTaskRecipe.training_spec()
    assert GeneratedTaskRecipe.pipeline_spec().task2batch is GeneratedTaskBatches and spec.objective == "task_group"
    assert spec.loss_family == "importance_sampling"
    assert spec.scheduling.unit == "sample"
    assert GeneratedTaskRecipe.report_type.fget(GeneratedTaskRecipe) is ScoredRolloutReport  # type: ignore[union-attr]
    built = training_recipe().build_pipeline(PipelineContext("one"))
    assert isinstance(built.record2record, GeneratedTaskRecord2Record)
    assert isinstance(built.record2task, GeneratedTaskTasks) and built.record2task.worker is None, "no round runs"
    assert isinstance(built.task2batch, GeneratedTaskBatches) and built.task2batch.tasks_per_step == 4
    assert built.buffer_size is None, "a rollout batch bounds its own samples"
    built.close()


def generated_pipeline_recipe(environ: Mapping[str, str], pipeline: Mapping[str, Any]) -> WeightTrainingRecipe:
    return GeneratedTaskRecipe.from_environment(
        environ,
        config={"data": {}, "model": {"path": "Qwen/Qwen3-8B"}, "pipeline": pipeline},
        runtime=InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8000"),
        training_runtime=StubTrainingRuntime(),
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
            "record2task": {
                "module": "reef.recipe.generated_tasks:GeneratedTaskTasks",
                "config": {
                    "rounds": 1,
                    "generator_url": "http://g:8910",
                    "description": "shell tasks",
                    "generator": generator,
                },
            },
            "task2batch": {
                "module": "reef.recipe.generated_tasks:GeneratedTaskBatches",
                "config": {"tasks_per_step": 2, "episodes_per_task": 3},
            },
        },
    )
    ScriptedGenerator.built.clear()
    built = recipe.build_pipeline(PipelineContext("one", task_storage=task_storage(tmp_path, "one")))
    assert isinstance(built.record2task, GeneratedTaskTasks)
    assert isinstance(built.record2task.task_generator, ScriptedGenerator)
    assert ScriptedGenerator.built == [{"config": {"count": 3}, "scenario": "one", "environ": environ}]
    assert built.record2task.task_store is not None
    batches = built.task2batch
    assert isinstance(batches, GeneratedTaskBatches) and (batches.tasks_per_step, batches.episodes_per_task) == (2, 3)
    built.close()


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        ({"record2task": {"module": "reef.recipe.generated_tasks:GeneratedTaskBatches"}}, "inherits Record2Task"),
        ({"record2tasks": {"module": "reef.recipe.generated_tasks:GeneratedTaskTasks"}}, "unknown key"),
    ],
)
def test_a_pipeline_section_that_names_no_usable_stage_is_refused(pipeline: Mapping[str, Any], message: str) -> None:
    with pytest.raises(RecipeConfigError, match=message):
        generated_pipeline_recipe({}, pipeline)


@pytest.mark.parametrize(
    ("generator", "message"),
    [
        (f"{MODULE}:ScriptedGenerator.missing", "cannot import pipeline.record2task.config.generator"),
        ("reef.recipe.generated_tasks:GeneratedTaskRecipe", "must name a class that inherits TaskGenerator"),
        ("reef.recipe.generated_tasks:TaskGenerator", "names an abstract class"),
    ],
)
def test_a_generator_setting_that_names_no_usable_generator_is_refused(generator: str, message: str) -> None:
    # The round settings check the generator setting when the scenario's pipeline builds.
    recipe = generated_pipeline_recipe(
        {},
        {
            "record2task": {
                "module": "reef.recipe.generated_tasks:GeneratedTaskTasks",
                "config": {"generator": {"module": generator}},
            }
        },
    )
    with pytest.raises(RecipeConfigError, match=message):
        recipe.build_pipeline(PipelineContext("one"))


def test_a_stage_without_from_config_cannot_be_named_in_the_pipeline_section() -> None:
    recipe = generated_pipeline_recipe({}, {"record2record": {"module": "reef.record2record:JudgingRecord2Record"}})
    with pytest.raises(RecipeConfigError, match="JudgingRecord2Record cannot be built from a deployment file"):
        recipe.build_pipeline(PipelineContext("one"))


def test_a_rollout_that_trains_nothing_lets_the_next_round_start(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1), generated(1, 2, 3)])
    p, _, _ = rollout_rounds(tmp_path, task_generator, BrokenRunner(tmp_path / "tasks"), rounds=2, batches_per_round=1)
    for _ in range(10):
        assert not p.ready(BATCH_SIZE)
    assert p.status()["task_generation"]["completed"] == 2, "the failed rollout ends without a batch to wait for"


def test_a_task_whose_result_cannot_be_saved_still_trains_and_the_next_job_saves_it(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]))
    store = p.record2task.task_store
    write_document = store.write_document

    def failing_write_document(*, key: str, document: str) -> None:
        if key == "rollout/harbor-00000-000":
            raise OSError(28, "No space left on device")
        write_document(key=key, document=document)

    store.write_document = failing_write_document
    looked(p)
    assert set(rollout_results(store)) == {"harbor-00000-001"}
    train_rollout(p)
    assert p.status()["task_store"]["trained"] == 2, "the task's episodes reached Reef, so its group trained"
    store.write_document = write_document
    assert asyncio.run(p.record2task.run(DeliveryJob(receipt="delivery"))).error == ""
    assert 0 in delivered_generations(store), "the next job saved the result, so the feedback went out"


def test_rounds_continue_when_the_task_pool_drains_before_batches_per_round(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1), generated(1, 2, 3)])
    p, _, _ = rollout_rounds(tmp_path, task_generator, rounds=2, batches_per_round=2)
    for _ in range(3):
        p.ready(BATCH_SIZE)
    train_rollout(p)
    for _ in range(3):
        p.ready(BATCH_SIZE)
    assert p.status()["task_generation"]["completed"] == 2, "no batch can come, so the next round starts"
    rollout = rollout_of(p)
    assert rollout is not None and all("harbor-00001" in task_use for task_use in rollout.task_uses)


def test_a_generation_whose_task_result_failed_to_save_once_still_gets_its_feedback(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, rounds=1)
    store = p.record2task.task_store
    write_document = store.write_document
    calls = {"count": 0}

    def flaky_write_document(*, key: str, document: str) -> None:
        calls["count"] += 1
        if key == "rollout/harbor-00000-000" and calls["count"] == 1:
            raise OSError(28, "No space left on device")
        write_document(key=key, document=document)

    store.write_document = flaky_write_document
    looked(p)
    assert task_generator.result_generations == [0], "the delivery job saved the result again, then delivered"


def test_a_skipped_generation_is_finished_by_the_next_round(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)], failing="harbor-00000-001")
    p, _ = rounds(tmp_path, task_generator, rounds=3, generation_attempts=1)
    p.ready(BATCH_SIZE)
    p.ready(BATCH_SIZE)
    store = p.record2task.task_store
    assert not store.generation(0).is_complete
    assert p.status()["task_store"]["train"] == 1 and p.status()["task_store"]["available"] == 0
    task_generator.failing = None
    for _ in range(3):
        p.ready(BATCH_SIZE)
    assert store.generation(0).is_complete and len(task_generator.requests) == 1
    assert p.status()["task_store"]["available"] == 2, "the same generation finished from its saved candidates"


def test_a_task_that_trained_before_its_result_was_saved_gets_a_lost_result_after_a_restart(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    store = p.record2task.task_store
    write_document = store.write_document

    def failing_write_document(*, key: str, document: str) -> None:
        if key == "rollout/harbor-00000-000":
            raise OSError(28, "No space left on device")
        write_document(key=key, document=document)

    # The first task's result is never saved, but both tasks train before the stop.
    store.write_document = failing_write_document
    looked(p)
    store.mark_trained(task_names=("harbor-00000-000", "harbor-00000-001"))
    p.close()
    task_generator = ScriptedGenerator()
    again, _ = rounds(tmp_path, task_generator, rounds=1)
    (lost,) = [result for result in rollout_results(again.record2task.task_store).values() if result.error]
    assert lost.task_name == "harbor-00000-000" and lost.infrastructure_failure
    assert len(again.task2batch.task_pool) == 0, "a trained task does not run again"
    assert asyncio.run(again.record2task.deliver_results()) == "" and task_generator.result_generations == [0]


def test_tasks_of_a_batch_that_did_not_train_roll_out_again(tmp_path: Path) -> None:
    p, _, _ = rollout_rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    for _ in range(3):
        p.ready(BATCH_SIZE)
    train_rollout(p, trained=False)
    p.ready(BATCH_SIZE)
    rollout = rollout_of(p)
    assert rollout is not None and len(rollout.task_uses) == 2, "both tasks roll out again"
    assert p.status()["task_store"]["trained"] == 0


def test_a_rollout_job_that_fails_gives_its_tasks_lost_results_so_the_feedback_goes_out(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    p, _, _ = rollout_rounds(tmp_path, task_generator, rounds=1)

    async def failing_run(job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("the worker lost its database connection")

    p.task2batch.run = failing_run
    for _ in range(4):
        p.ready(BATCH_SIZE)
    results = rollout_results(p.record2task.task_store).values()
    assert [result.infrastructure_failure for result in results] == [True, True]
    assert task_generator.result_generations == [0]


def test_a_result_that_the_database_fails_to_save_still_trains_and_is_saved_later(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    store = p.record2task.task_store
    write_document = store.write_document

    def locked_write_document(*, key: str, document: str) -> None:
        if key == "rollout/harbor-00000-000":
            raise sqlalchemy.exc.OperationalError("INSERT", {}, Exception("database is locked"))
        write_document(key=key, document=document)

    store.write_document = locked_write_document
    looked(p)
    train_rollout(p)
    assert p.status()["task_store"]["trained"] == 2, "both groups trained"
    store.write_document = write_document
    assert asyncio.run(p.record2task.run(DeliveryJob(receipt="delivery"))).error == ""
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
    store = p.record2task.task_store
    state = store.generation(0)
    assert state is not None and state.is_complete, "the next round finished the same generation"
    assert state.source_record_ids == ("rec-a", "rep-a", "rec-b", "rep-b")
    assert len(task_generator.requests) == 1


def test_an_identical_candidate_in_the_same_generation_is_rejected_as_a_duplicate(tmp_path: Path) -> None:
    task = generated(0, 0)
    task_generator = ScriptedGenerator([task + task])
    p, _ = rounds(tmp_path, task_generator, rounds=1)
    looked(p)
    state = p.record2task.task_store.generation(0)
    assert state is not None and train_task_names(state) == ("harbor-00000-000",)
    assert state.candidates is not None and "is already published" in state.candidates[1].error


def test_a_task_whose_step_did_not_commit_rolls_out_again_after_a_restart(tmp_path: Path) -> None:
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), rounds=1)
    assert looked(first) and rollout_of(first) is not None
    first.acknowledge(first.build_batch(BATCH_SIZE).batch_id)
    # The process stops before the step's commit is durable, so no ID is released.
    assert first.record2task.task_store.trained_task_names() == frozenset()
    first.close()
    restarted, _ = rounds(tmp_path, ScriptedGenerator(), rounds=1)
    assert len(restarted.task2batch.task_pool) == 2, "both tasks roll out again"


def test_a_generation_outcome_that_is_incomplete_holds_no_task_and_names_its_error() -> None:
    with pytest.raises(ValueError, match="incomplete generation"):
        GenerationOutcome(receipt="generation-00000", generation=0, tasks=(), is_complete=False, error="")
    with pytest.raises(ValueError, match="non-negative"):
        GenerationJob(receipt="generation-x", generation=-1)
