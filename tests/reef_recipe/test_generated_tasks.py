"""The generated-task recipe: rounds of generated tasks played and sent back, episodes grouped by task and trained."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import pytest
from tests.support.runtime_stubs import StubTrainingRuntime
from tests.support.task_service_stubs import StandInGenerator, task_for

from reef.artifact import InMemoryRepositoryBackend
from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord, RequestType
from reef.core.reports import ScoredRolloutReport
from reef.core.tasks import (
    GeneratedTask,
    HarborTask,
    PlayArm,
    TaskGenerationRequest,
    TaskGenerator,
    TaskPlayResult,
    TaskValidationResult,
)
from reef.dispatcher import Dispatcher
from reef.harness.client.tasks import TaskPlay
from reef.inference.http import InferenceProxyRuntime
from reef.pipeline_context import PipelineContext
from reef.recipe.base import Recipe, WeightTrainingRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks import GeneratedTaskBatches, GeneratedTaskRecipe, TaskGroupObjective
from reef.recipe.generated_tasks.processor import (
    GeneratedTaskRecord2Record,
    GeneratedTaskTasks,
    GenerationJob,
    GenerationOutcome,
    PlayJob,
    RoundSettings,
    TaskBatchSettings,
    TaskGroupSettings,
    episode_assembly,
    reported_task_name,
    rewards_of,
)
from reef.record2record import GroupDecision, ReportContext
from reef.record2task.generator.client import GeneratorError
from reef.record2task.store import FileTaskStore, PlayedResult
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.algos.registry import resolve_objective
from reef.train.pipeline import PipelineConfig, TrainingPipeline
from reef.train.types import TrainingBatch, trajectory_groups

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
    record_id: str, reference: str, task: str | None, score: float, *, arm: str | None = "plain"
) -> AgentRecord:
    payload: dict[str, object] = {"score": score, "feedback": f"verifier reward {score}", "references": [reference]}
    metadata: dict[str, object] = {}
    if task is not None:
        metadata["task"] = {"name": task, "path": f"/tasks/{task}", "digest": "ab" * 32}
    if arm is not None:
        metadata["episode"] = {"id": record_id, "labels": {"arm": arm}}
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


def processor(**config: object) -> TrainingPipeline:
    context = PipelineContext("tasks")
    group_settings, round_settings, batch_settings = stage_settings(
        {"tasks_per_step": 2, "plays_per_task": 2, **config}
    )
    stage = GeneratedTaskRecord2Record(context, group_settings)
    return TrainingPipeline(
        context,
        record2record=stage,
        record2task=GeneratedTaskTasks(context, round_settings, stage),
        task2batch=GeneratedTaskBatches(context, batch_settings),
    )


def played(processor: TrainingPipeline, task: str, index: int, score: float, **fields: Any) -> None:
    processor.ingest(inference(f"rec-{task}-{index}"))
    processor.ingest(report(f"rep-{task}-{index}", f"rec-{task}-{index}", task, score, **fields))


def test_rewards_of_counts_an_unscored_run_as_a_loss_and_skips_an_episode_that_never_ran(tmp_path: Path) -> None:
    def play(reward: float | None, error: str = "") -> TaskPlay:
        return TaskPlay(tmp_path, "t", "e", reward, {}, error, (), 0, (), None)

    assert rewards_of((play(1.0), play(None), play(None, "the trial raised"))) == (1.0, 0.0)
    assert rewards_of((play(0.5, "a late error"),)) == (0.5,)


def test_reported_task_name_reads_the_task_players_metadata() -> None:
    assert reported_task_name(report("r", "i", "harbor-00000-000", 1.0)) == "harbor-00000-000"
    assert reported_task_name(report("r", "i", None, 1.0)) is None


def test_episodes_of_one_task_form_a_group_and_a_batch_holds_complete_groups_only() -> None:
    p = processor()
    played(p, "harbor-00000-000", 0, 1.0)
    assert not p.ready(), "one episode of a task is not a group"
    played(p, "harbor-00000-001", 0, 0.0)
    played(p, "harbor-00000-000", 1, 0.0)
    assert not p.ready(), "one complete group is not a batch of two"
    played(p, "harbor-00000-001", 1, 1.0)
    assert p.ready()
    batch = p.build_batch()
    assert isinstance(batch, TrainingBatch) and batch.batch_id == "tasks:generated-tasks:1"
    groups = trajectory_groups(batch)
    assert [[sample.group_id for sample in group] for group in groups] == [
        ["harbor-00000-000", "harbor-00000-000"],
        ["harbor-00000-001", "harbor-00000-001"],
    ]
    assert [sample.metadata["reward"] for group in groups for sample in group] == [1.0, 0.0, 0.0, 1.0]


def test_a_training_report_without_a_task_is_refused_and_keeps_its_report_retryable() -> None:
    p = processor()
    p.ingest(inference("rec-x"))
    with pytest.raises(ValueError, match=r"requires metadata\.task\.name"):
        p.ingest(report("rep-x", "rec-x", None, 1.0))


def test_measurement_arms_and_reports_without_an_arm_are_released_not_trained() -> None:
    p = processor()
    played(p, "harbor-00000-000", 0, 0.0, arm="hint")
    played(p, "harbor-00000-000", 1, 1.0, arm="hint")
    assert not p.ready(), "measurement episodes never form a group"
    p.ingest(inference("rec-generator"))
    p.ingest(report("rep-generator", "rec-generator", "harbor-00000-000", 0.5, arm=None))
    assert not p.ready()
    released = p.releasable_record_ids()
    assert {"rep-generator", "rec-generator"} <= released
    assert {"rec-harbor-00000-000-0", "rec-harbor-00000-000-1"} <= released, "the measurement arm's calls too"
    played(p, "harbor-00000-000", 2, 1.0)
    played(p, "harbor-00000-000", 3, 0.0)
    played(p, "harbor-00000-001", 0, 1.0)
    played(p, "harbor-00000-001", 1, 0.0)
    assert p.ready() and len(p.build_batch().items) == 4


def test_a_task_group_waits_for_plays_per_task_episodes() -> None:
    grouping = processor(plays_per_task=3).record2record.grouping
    one = (ReportContext(inference("a"), 1.0, (inference("a"),)),)
    assert grouping.decide("t", one) is GroupDecision.INCOMPLETE
    assert grouping.decide("t", one * 3) is GroupDecision.READY


def test_the_assembly_spans_an_episodes_turns_and_realigns_the_think_scaffold() -> None:
    assembly = episode_assembly()
    assert assembly.settings.accept_multi_turn_policy_samples and assembly.settings.scaffold_tolerance == 8
    assert episode_assembly(2).settings.scaffold_tolerance == 2


def test_an_episode_whose_response_was_cut_at_max_tokens_still_trains() -> None:
    # Tinker's default max_tokens is 1024: a cut response plus the four-token think scaffold drifts 1028 tokens.
    scaffold, response = [90, 91, 92, 93], list(range(1000, 2024))

    def turn(record_id: str, tokens: list[int], response_length: int) -> AgentRecord:
        training = {
            "tokens": tokens,
            "loss_mask": [1] * response_length,
            "rollout_log_probs": [-0.1] * response_length,
            "runtime_load_id": "r1",
        }
        return AgentRecord.create(
            scenario="tasks",
            request_type=RequestType.INFERENCE,
            payload={"response": {"training": training}},
            agent_record_id=record_id,
        )

    p = processor(tasks_per_step=1)
    p.ingest(turn("rec-0", [1, 2, *scaffold, *response], len(response)))
    p.ingest(turn("rec-1", [1, 2, *response, 3, *scaffold, 4], 1))
    payload = {**report("rep-0", "rec-0", "harbor-00000-000", 1.0).payload, "references": ["rec-0", "rec-1"]}
    p.ingest(
        AgentRecord.create(scenario="tasks", request_type=RequestType.REPORT, payload=payload, agent_record_id="rep-0")
    )
    played(p, "harbor-00000-000", 1, 0.0)
    assert p.ready()
    assert [sample.training.get("turn_count", 1) for sample in p.build_batch().items] == [2, 1]


def test_the_objective_centers_and_scales_rewards_within_each_task_group() -> None:
    p = processor()
    for index, score in enumerate((1.0, 0.0)):
        played(p, "harbor-00000-000", index, score)
    for index, score in enumerate((0.5, 0.5)):
        played(p, "harbor-00000-001", index, score)
    objective = resolve_objective("task_group")
    assert isinstance(objective, TaskGroupObjective) and objective.loss_family == "importance_sampling"
    signal = objective.prepare(p.build_batch(), {})
    assert signal.action == "train"
    assert signal.advantages == (1.0, -1.0, 0.0, 0.0)
    assert signal.metrics["constant_groups"] == 1


def test_a_batch_whose_groups_are_all_constant_is_skipped_and_keeps_the_step_count() -> None:
    p = processor()
    for task in ("harbor-00000-000", "harbor-00000-001"):
        for index in range(2):
            played(p, task, index, 1.0)
    signal = resolve_objective("task_group").prepare(p.build_batch(), {"steps": 3})
    assert signal.action == "skip" and signal.advantages is None
    assert signal.next_algorithm_state == {"steps": 3}
    assert signal.metrics == {"constant_groups": 2, "skipped": "every group is constant"}


# ------------------------------------------------------------------------------------------------- the rounds


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: PlayArm("Hint", 1), "arm name must match"),
        (lambda: PlayArm("hint", 0), "positive integer"),
        (lambda: PlayArm("hint", 1, ("../hint.txt",)), "inside the task"),
        (lambda: PlayArm("hint", 1, ("/hint.txt",)), "inside the task"),
        (lambda: GeneratedTask(task_for("t-a", "r", 1), (PlayArm("hint", 1), PlayArm("hint", 2))), "unique"),
        (lambda: TaskPlayResult(task_for("t-a", "r", 1), (1.0,), error="refused"), "no rewards"),
        (lambda: TaskPlayResult(task_for("t-a", "r", 1), (float("nan"),)), "finite"),
        (lambda: TaskPlayResult(task_for("t-a", "r", 1), (), {"hint": [1.0]}), "tuples"),
        (lambda: TaskPlayResult(task=task_for("t-a", "r", 1), training_rewards=(1.0,), split="eval"), "eval"),
        (lambda: TaskValidationResult(("",)), "non-empty"),
    ],
)
def test_the_contract_values_refuse_what_the_recipe_cannot_play(build: Any, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build()


class ScriptedGenerator(TaskGenerator):
    """A task generator that returns scripted tasks, refuses scripted names, and keeps what it was sent."""

    built: list[dict[str, object]] = []

    def __init__(
        self,
        rounds: list[tuple[GeneratedTask, ...]] | None = None,
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
        self.results: list[tuple[TaskPlayResult, ...]] = []
        self.result_generations: list[int] = []

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str]) -> Self:
        cls.built.append({"config": dict(config), "scenario": scenario, "environ": dict(environ)})
        return cls()

    async def generate(self, request: TaskGenerationRequest) -> tuple[GeneratedTask, ...]:
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

    async def record_results(self, results: tuple[TaskPlayResult, ...], *, generation: int) -> None:
        self.results.append(results)
        self.result_generations.append(generation)
        if self.failing_results:
            raise RuntimeError("the generator lost its state directory")


class SlowGenerator(ScriptedGenerator):
    """A generator whose generate call takes longer than a short deadline."""

    async def generate(self, request: TaskGenerationRequest) -> tuple[GeneratedTask, ...]:
        await asyncio.sleep(1.0)
        return await super().generate(request)


def generated(round_number: int, *indices: int, hint: bool = True) -> tuple[GeneratedTask, ...]:
    """Tasks of one round; each asks for a hint arm of one play unless ``hint`` is off."""
    arms = (PlayArm("hint", 1, ("solution/hint.txt",)),) if hint else ()
    return tuple(
        GeneratedTask(
            task_for(f"harbor-{round_number:05d}-{index:03d}", f"designer-{index}", 9000 + 10 * round_number + index),
            arms,
        )
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
        self.generation: GeneratedTaskTasks | None = None
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
        self.generation: GeneratedTaskTasks | None = None
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


def rounds(
    tmp_path: Path,
    task_generator: ScriptedGenerator,
    player: StandInGenerator | None = None,
    *,
    worker: InlineWorker | DeferredWorker | None = None,
    **config: object,
) -> tuple[TrainingPipeline, StandInGenerator]:
    player = player if player is not None else StandInGenerator(tmp_path / "tasks")
    worker = worker if worker is not None else InlineWorker()
    context = PipelineContext("tasks")
    group_settings, round_settings, batch_settings = stage_settings(
        {
            "tasks_per_step": 2,
            "plays_per_task": 2,
            "rounds": 3,
            "description": DESCRIPTION,
            "state_dir": str(tmp_path / "state"),
            "served_model": "Qwen/Qwen3-8B",
            **config,
        }
    )
    stage = GeneratedTaskRecord2Record(context, group_settings)
    generation = GeneratedTaskTasks(
        context, round_settings, stage, task_generator=task_generator, task_player=player, worker=worker
    )
    built = TrainingPipeline(
        context,
        record2record=stage,
        record2task=generation,
        task2batch=GeneratedTaskBatches(context, batch_settings),
    )
    worker.generation = generation
    return built, player


def looked(p: TrainingPipeline) -> bool:
    """Look for a batch three times: a generation's outcome, then its play's, land on the look after each submit."""
    p.ready()
    p.ready()
    return p.ready()


def generation_numbers(worker: InlineWorker | DeferredWorker) -> list[int]:
    """The generations that the worker took, in order."""
    return [job.generation for job in worker.submitted if isinstance(job, GenerationJob)]


def generation_document(tmp_path: Path, generation: int) -> dict[str, Any]:
    return json.loads(
        (tmp_path / "state" / "tasks" / "store" / "generations" / f"gen-{generation:05d}.json").read_text()
    )


def test_the_first_look_for_a_batch_runs_round_zero_end_to_end(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1) + generated(0, 2, hint=False)])
    p, player = rounds(tmp_path, task_generator)
    status = p.status()["task_generation"]
    assert status["generator"] == "ScriptedGenerator" and (status["next"], status["completed"], status["of"]) == (
        0,
        0,
        3,
    )
    assert not looked(p)
    assert not p.background_jobs_running(), "the inline worker finished before ready() returned"
    status = p.status()["task_generation"]
    assert (status["next"], status["completed"]) == (1, 1)
    assert p.status()["task_store"]["played/train"] == 3

    assert task_generator.requests == [TaskGenerationRequest((), DESCRIPTION)]
    names = ["harbor-00000-000", "harbor-00000-001", "harbor-00000-002"]
    assert task_generator.validated == names and player.writes == names
    arms = [(call["task"], call["arm"], call["plays"], call["is_reporting"], call["extra"]) for call in player.plays]
    assert arms == [
        (names[0], "plain", 2, True, ()),
        (names[0], "hint", 1, True, ("solution/hint.txt",)),
        (names[1], "plain", 2, True, ()),
        (names[1], "hint", 1, True, ("solution/hint.txt",)),
        (names[2], "plain", 2, True, ()),
    ]
    assert all(call["tags"] == {"round": "0"} and call["model"] == "Qwen/Qwen3-8B" for call in player.plays)

    (results,) = task_generator.results
    assert task_generator.result_generations == [0]
    assert [result.task.name for result in results] == names
    assert results[0].training_rewards == (0.25, 0.25) and results[0].measurement_rewards == {"hint": (0.75,)}
    assert results[2].training_rewards == (0.25, 0.25) and results[2].measurement_rewards == {}
    assert results[2].is_played
    document = generation_document(tmp_path, 0)
    assert document["is_delivered"] and [result["task_name"] for result in document["results"]] == names
    assert document["results"][0]["training_rewards"] == [0.25, 0.25]
    assert document["results"][0]["measurement_rewards"] == {"hint": [0.75]}


@pytest.mark.unit
def test_eval_tasks_are_saved_without_training_or_measurement_plays(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    pipeline, player = rounds(tmp_path, task_generator, eval_fraction=1.0)
    assert not looked(pipeline)
    assert player.plays == [] and player.writes == []
    assert all(
        result.split == "eval" and not result.is_played and not result.error for result in task_generator.results[0]
    )
    assert pipeline.status()["task_store"] == {"held_out/eval": 2, "rejected": 0, "candidate": 0}
    assert pipeline.record2task.task_store.generation(generation=0).is_delivered, "no train task waits to play"
    assert pipeline.record2task.task_store.generation(generation=0).is_complete
    pipeline.close()


@pytest.mark.unit
def test_a_check_that_cannot_run_is_retried_from_the_saved_candidates_then_deferred(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)], failing="harbor-00000-001")
    pipeline, player = rounds(tmp_path, task_generator, rounds=1, generation_attempts=2)
    assert not looked(pipeline)
    assert len(task_generator.requests) == 1, "a retry continues from the saved candidates"
    assert task_generator.validated == ["harbor-00000-000", "harbor-00000-001", "harbor-00000-001"]
    assert pipeline.record2task.task_store.generation(generation=0).pending_indices == (1,)
    assert player.writes == [], "a deferred generation plays none of its tasks"
    assert "the checks could not run" in pipeline.status()["task_generation"]["last_error"]
    task_generator.failing = None
    outcome = asyncio.run(pipeline.record2task.run(GenerationJob(receipt="generation-00000", generation=0)))
    assert isinstance(outcome, GenerationOutcome) and not outcome.is_deferred and len(outcome.tasks) == 2
    assert len(task_generator.requests) == 1
    assert pipeline.record2task.task_store.generation(generation=0).is_complete
    pipeline.close()


@pytest.mark.unit
def test_a_generation_past_its_deadline_is_deferred(tmp_path: Path) -> None:
    task_generator = SlowGenerator([generated(0, 0)])
    pipeline, player = rounds(tmp_path, task_generator, generation_deadline_seconds=0.05)
    assert not looked(pipeline)
    assert player.writes == [] and pipeline.record2task.task_store.generation(generation=0).candidates is None
    assert pipeline.status()["task_generation"]["last_error"]
    pipeline.close()


@pytest.mark.unit
def test_legacy_rounds_are_preserved_and_do_not_resume_into_a_fresh_store(tmp_path: Path) -> None:
    legacy = tmp_path / "state" / "tasks" / "round-00000.json"
    legacy.parent.mkdir(parents=True)
    contents = '{"round": 0, "tasks": []}\n'
    legacy.write_text(contents)
    pipeline, player = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), rounds=1)
    assert pipeline.status()["task_generation"]["next"] == 0
    assert not looked(pipeline)
    assert legacy.read_text() == contents
    assert player.writes == ["harbor-00000-000"]
    pipeline.close()


@pytest.mark.integration
def test_deleting_a_scenario_archives_its_task_store_and_preserves_other_scenarios(tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    for scenario in ("first", "other"):
        with FileTaskStore(root=state_root / scenario / "store", eval_fraction=0.0, seed=0) as store:
            store.begin_generation(generation=0, source_record_ids=())
            store.save_candidates(generation=0, candidates=generated(0, 0))
            store.publish_task(generation=0, index=0)
            store.complete_generation(generation=0)
    stages = PipelineConfig.from_setting(
        {"record2task": {"config": {"state_dir": str(state_root)}}},
        spec=GeneratedTaskRecipe.pipeline_spec(),
        environ={},
    )
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    dispatcher = Dispatcher(
        recipe=Recipe(pipeline_stages=stages),
        backend_factory=InMemoryRepositoryBackend.factory(bootstrap, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(None),
    )
    try:
        dispatcher.get_or_create_scenario(scenario="first")
        archived = dispatcher.delete_scenario(scenario="first")["archived"]
        assert not (state_root / "first").exists()
        assert any((Path(path) / "store" / "generations" / "gen-00000.json").is_file() for path in archived)
        assert (state_root / "other" / "store" / "tasks" / "harbor-00000-000").is_dir()
    finally:
        dispatcher.close()


def test_an_invalid_task_comes_back_with_its_errors_and_is_never_written(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator(
        [generated(0, 0, 1)], invalid={"harbor-00000-000": "oracle check refused: the oracle scored 0"}
    )
    p, player = rounds(tmp_path, task_generator)
    assert not looked(p)
    assert player.writes == ["harbor-00000-001"]
    (results,) = task_generator.results
    assert results[0].error == "oracle check refused: the oracle scored 0" and results[0].training_rewards == ()
    assert results[1].is_played
    assert p.status()["task_store"]["played/train"] == 1


def test_a_measurement_arm_with_the_training_arms_name_is_refused_unplayed(tmp_path: Path) -> None:
    clashing = GeneratedTask(task_for("harbor-00000-000", "designer-0", 9000), (PlayArm("plain", 1),))
    task_generator = ScriptedGenerator([(clashing, *generated(0, 1))])
    p, player = rounds(tmp_path, task_generator)
    assert not looked(p)
    assert task_generator.validated == ["harbor-00000-001"] and player.writes == ["harbor-00000-001"]
    (results,) = task_generator.results
    assert results[0].error == "the measurement arm 'plain' has the name of the training arm"


def test_a_refused_write_or_a_task_the_agent_cannot_play_comes_back_unplayed(tmp_path: Path) -> None:
    taken = StandInGenerator(tmp_path / "tasks", taken=("harbor-00000-000",))
    task_generator = ScriptedGenerator([generated(0, 0)])
    p, _ = rounds(tmp_path, task_generator, taken)
    assert not looked(p)
    assert task_generator.results[0][0].error.startswith("refused: a different task holds the name harbor-00000-000")
    assert taken.plays == []

    unplayable = StandInGenerator(tmp_path / "second" / "tasks", play_error="Failed to start tmux session.")
    task_generator = ScriptedGenerator([generated(0, 0)])
    p, _ = rounds(tmp_path / "second", task_generator, unplayable)
    assert not looked(p)
    assert [call["arm"] for call in unplayable.plays] == ["plain"], "no measurement arm for a task that cannot run"
    assert unplayable.deleted == ["harbor-00000-000"]
    assert task_generator.results[0][0].error == "the agent could not play the task: Failed to start tmux session."


class PartlyReported(StandInGenerator):
    """Plays whose first ``reported`` episodes reach Reef; the rest end unreported, as a failed trial does."""

    def __init__(self, root: Path, *, reported: int) -> None:
        super().__init__(root)
        self.reported = reported

    async def play(self, task_path, **fields) -> tuple[TaskPlay, ...]:
        plays = await super().play(task_path, **fields)
        return tuple(
            play if index < self.reported else dataclasses.replace(play, report_agent_record_ids=())
            for index, play in enumerate(plays)
        )


@pytest.mark.parametrize(("reported", "is_trained"), [(2, True), (1, False)])
def test_a_task_group_completes_at_the_episodes_its_plays_reported(
    tmp_path: Path, reported: int, is_trained: bool
) -> None:
    worker = DeferredWorker()
    player = PartlyReported(tmp_path / "tasks", reported=reported)
    p, _ = rounds(
        tmp_path, ScriptedGenerator([generated(0, 0)]), player, worker=worker, plays_per_task=4, tasks_per_step=1
    )
    assert not p.ready() and generation_numbers(worker) == [0]
    for index in range(reported):
        played(p, "harbor-00000-000", index, float(index % 2))
    assert not p.ready(), "the generation runs, so the group waits for all four plays"
    worker.finish()
    assert not p.ready(), "the generation lands and its play starts"
    worker.finish()
    assert p.ready() is is_trained
    if is_trained:
        assert len(p.build_batch().items) == 2
    else:
        assert {"rep-harbor-00000-000-0", "rec-harbor-00000-000-0"} <= p.releasable_record_ids()


def test_a_restarted_processor_closes_a_group_at_the_count_its_task_store_kept(tmp_path: Path) -> None:
    player = PartlyReported(tmp_path / "tasks", reported=2)
    config: dict[str, Any] = {"plays_per_task": 4, "tasks_per_step": 1, "rounds": 1}
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), player, **config)
    assert not looked(first)
    first.close()
    restarted, _ = rounds(tmp_path, ScriptedGenerator(), player, **config)
    for index in range(2):
        played(restarted, "harbor-00000-000", index, float(index % 2))
    assert restarted.ready() and len(restarted.build_batch().items) == 2


def test_the_next_round_waits_for_batches_per_round(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1), generated(1, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, batches_per_round=2)
    assert not looked(p) and len(task_generator.requests) == 1
    assert not looked(p), "no batch was acknowledged yet"
    assert len(task_generator.requests) == 1
    for batch in range(2):
        for task in (f"t-{batch}-0", f"t-{batch}-1"):
            played(p, task, 0, 1.0)
            played(p, task, 1, 0.0)
        p.acknowledge(p.build_batch().batch_id)
        looked(p)
    assert len(task_generator.requests) == 2 and p.status()["task_generation"]["completed"] == 2
    assert sorted(entry.name for entry in (tmp_path / "state" / "tasks" / "store" / "generations").iterdir()) == [
        "gen-00000.json",
        "gen-00001.json",
    ]


def test_batches_trained_while_a_round_runs_count_toward_the_next_one(tmp_path: Path) -> None:
    worker = DeferredWorker()
    task_generator = ScriptedGenerator([generated(0, 0, 1), generated(1, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, worker=worker, batches_per_round=2)
    assert not p.ready() and generation_numbers(worker) == [0]
    assert p.background_jobs_running() and p.status()["task_generation"]["in_flight"] == 0
    for batch in range(2):
        for task in (f"t-{batch}-0", f"t-{batch}-1"):
            played(p, task, 0, 1.0)
            played(p, task, 1, 0.0)
        p.acknowledge(p.build_batch().batch_id)
        assert not p.ready() and generation_numbers(worker) == [0], "one round at a time"
    worker.finish()
    assert not p.ready(), "round 0 lands on this look and round 1 starts on it"
    assert generation_numbers(worker) == [0, 1]
    status = p.status()["task_generation"]
    assert (status["in_flight"], status["next"], status["completed"]) == (1, 2, 1)


def test_a_dropped_batch_does_not_count_toward_the_next_round(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), worker=worker)
    assert not p.ready() and generation_numbers(worker) == [0]
    for task in ("t-0", "t-1"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    batch_id = p.build_batch().batch_id
    p.acknowledge(batch_id, trained=False)
    worker.finish()
    assert not p.ready(), "round 0 lands; the dropped batch trained nothing"
    assert generation_numbers(worker) == [0]
    for task in ("t-2", "t-3"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    p.acknowledge(p.build_batch().batch_id)
    assert not p.ready(), "a trained batch counts"
    assert generation_numbers(worker) == [0, 1]


def test_a_round_that_played_no_task_does_not_block_the_next_one(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator([]), worker=worker)
    assert not p.ready() and generation_numbers(worker) == [0]
    worker.finish()
    assert not p.ready(), "round 0 lands without a task and round 1 starts on the same look"
    assert generation_numbers(worker) == [0, 1], "no batch can come; none was acknowledged"


@pytest.mark.parametrize(("retry_seconds", "generations"), [(600.0, [0]), (0.0, [0, 0])])
def test_a_generation_job_that_failed_is_deferred_and_retried_after_a_wait(
    tmp_path: Path, retry_seconds: float, generations: list[int]
) -> None:
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), worker=worker, deferred_retry_seconds=retry_seconds)
    assert not p.ready() and generation_numbers(worker) == [0]
    worker.waiting.clear()
    worker.finished.append(Failed("generation-00000"))
    assert not p.ready() and not p.ready()
    assert generation_numbers(worker) == generations, "a deferred round waits before the next one"
    assert p.status()["task_generation"]["completed"] == 0, "a deferred round does not use up a round"
    assert "generation-00000 did not finish" in p.status()["task_generation"]["last_error"]


def test_a_restarted_processor_carries_on_from_its_task_store(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), rounds=2)
    assert not looked(p) and p.status()["task_generation"]["completed"] == 1
    p.close()
    task_generator = ScriptedGenerator([generated(1, 0)])
    again, _ = rounds(tmp_path, task_generator, StandInGenerator(tmp_path / "tasks"), rounds=2)
    status = again.status()["task_generation"]
    assert (status["in_flight"], status["next"], status["completed"]) == (None, 1, 1)
    assert not looked(again) and again.status()["task_generation"]["completed"] == 2
    assert not looked(again), "the cap is reached"
    assert len(task_generator.requests) == 1


def test_results_that_the_generator_did_not_take_are_sent_again_by_a_later_job(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0, 1)], failing_results=True)
    p, player = rounds(tmp_path, task_generator)
    assert not looked(p)
    assert player.writes == ["harbor-00000-000", "harbor-00000-001"]
    assert p.status()["task_generation"]["last_error"] == "the generator lost its state directory"
    assert not p.record2task.task_store.generation(generation=0).is_delivered
    task_generator.failing_results = False
    assert asyncio.run(p.record2task.deliver_results()) == ""
    assert task_generator.result_generations == [0, 0], "the second delivery reached it"
    assert p.record2task.task_store.generation(generation=0).is_delivered
    assert asyncio.run(p.record2task.deliver_results()) == "" and len(task_generator.results) == 2


def test_a_closed_pipeline_starts_no_round_and_ignores_a_round_that_finishes_late(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), worker=worker)
    assert not p.ready() and generation_numbers(worker) == [0]
    p.close()
    worker.finished.append(
        GenerationOutcome(receipt="generation-00000", generation=0, tasks=(), is_deferred=False, error="")
    )
    assert not looked(p) and generation_numbers(worker) == [0]
    assert p.status()["task_generation"]["completed"] == 0, "the late round settled nothing"


def test_a_worker_that_refuses_a_round_stops_the_pipeline_with_the_reason(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]))
    p.record2task.worker.close()
    with pytest.raises(RuntimeError, match="generation-00000 cannot run"):
        p.ready()


def test_the_real_worker_runs_a_round_off_the_trainers_thread(tmp_path: Path) -> None:
    context = PipelineContext("tasks")
    stage = GeneratedTaskRecord2Record(context, TaskGroupSettings())
    generation = GeneratedTaskTasks(
        context,
        RoundSettings(rounds=1, description="shell tasks", state_dir=str(tmp_path / "state")),
        stage,
        task_generator=ScriptedGenerator([generated(0, 0)]),
        task_player=StandInGenerator(tmp_path / "tasks"),
    )
    p = TrainingPipeline(
        context,
        record2record=stage,
        record2task=generation,
        task2batch=GeneratedTaskBatches(context, TaskBatchSettings()),
    )
    try:
        assert not p.ready()
        deadline = time.monotonic() + 10.0
        while p.background_jobs_running() and time.monotonic() < deadline:
            time.sleep(0.02)
            p.ready()
        assert not p.background_jobs_running() and p.status()["task_generation"]["completed"] == 1
        assert generation_document(tmp_path, 0)["is_delivered"], "the generation, its play and its results ran"
    finally:
        p.close()


def test_a_generation_plays_nothing_and_its_tasks_play_from_the_task_pool(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, player = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), worker=worker)
    assert not p.ready() and generation_numbers(worker) == [0]
    worker.finish()
    assert player.writes == [] and player.plays == [], "the generation job saved its tasks and played none"
    assert not p.ready()
    (play,) = [job for job in worker.submitted if isinstance(job, PlayJob)]
    assert play.task_names == ("harbor-00000-000", "harbor-00000-001")
    assert p.status()["task_store"]["available/train"] == 2
    worker.finish()
    assert player.writes == ["harbor-00000-000", "harbor-00000-001"]
    assert p.status()["task_store"]["played/train"] == 2


def test_a_restart_plays_the_saved_tasks_that_had_no_result_without_generating_again(tmp_path: Path) -> None:
    worker = DeferredWorker()
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0, 1)]), worker=worker, rounds=1)
    first.ready()
    worker.finish()
    first.ready()
    # The first task's play ended before the stop; the second task's did not.
    first.record2task.task_store.record_result(generation=0, result=PlayedResult("harbor-00000-000", (1.0, 0.0)))
    first.close()
    task_generator = ScriptedGenerator()
    player = StandInGenerator(tmp_path / "tasks")
    restarted, _ = rounds(tmp_path, task_generator, player, rounds=1)
    assert len(restarted.record2record.task_pool) == 1, "the unplayed sibling is back in the task pool"
    looked(restarted)
    assert player.writes == ["harbor-00000-001"] and task_generator.requests == []
    assert task_generator.result_generations == [0], "the generation's results go out once both tasks have one"


def test_a_task_that_cannot_run_is_an_infrastructure_failure(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0)])
    p, _ = rounds(tmp_path, task_generator, BrokenPlayer(tmp_path / "tasks"), rounds=1)
    looked(p)
    ((result,),) = task_generator.results
    assert result.infrastructure_failure and result.error == "the sandbox provider is down"
    assert not result.is_played


class BrokenPlayer(StandInGenerator):
    """A generator service whose plays cannot run."""

    async def play(self, task_path, **fields) -> tuple[TaskPlay, ...]:
        raise GeneratorError("the sandbox provider is down")


def test_a_task_that_an_earlier_generation_published_is_rejected_as_a_duplicate(tmp_path: Path) -> None:
    first = generated(0, 0, 1)
    task_generator = ScriptedGenerator([first, (first[0], *generated(1, 5))])
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, task_generator, worker=worker, rounds=2, batches_per_round=1)
    for _ in range(3):
        p.ready()
        worker.finish()
    for task in ("t-0", "t-1"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    p.acknowledge(p.build_batch().batch_id)
    for _ in range(4):
        p.ready()
        worker.finish()
    p.ready()
    store = p.record2task.task_store
    assert store.generation(generation=1).train_task_names == ("harbor-00001-005",)
    assert "already published in generation 0" in store.generation(generation=1).outcomes[0].error
    assert task_generator.result_generations == [0, 1], "both generations get their results"


def test_a_restart_sends_results_that_the_generator_did_not_receive(tmp_path: Path) -> None:
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)], failing_results=True), rounds=1)
    looked(first)
    first.close()
    task_generator = ScriptedGenerator()
    worker = DeferredWorker()
    restarted, _ = rounds(tmp_path, task_generator, worker=worker, rounds=1)
    assert [job.receipt for job in worker.submitted] == ["delivery"], "no round or play is due, yet the results go out"
    worker.finish()
    restarted.ready()
    assert task_generator.result_generations == [0]
    assert restarted.record2task.task_store.generation(generation=0).is_delivered


def test_a_generation_whose_results_fail_does_not_hold_back_later_ones(tmp_path: Path) -> None:
    task_generator = FirstResultsFail([generated(0, 0, 1), generated(1, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, rounds=2)
    looked(p)
    for task in ("t-0", "t-1"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    p.acknowledge(p.build_batch().batch_id)
    looked(p)
    store = p.record2task.task_store
    assert not store.generation(generation=0).is_delivered and store.generation(generation=1).is_delivered


def test_a_task_whose_play_a_stop_interrupted_is_staged_again_and_plays(tmp_path: Path) -> None:
    worker = DeferredWorker()
    player = StandInGenerator(tmp_path / "tasks")
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), player, worker=worker, rounds=1)
    first.ready()
    worker.finish()
    first.ready()
    # The play wrote the task through the generator service, then the process stopped before its result was saved.
    asyncio.run(player.write_task(first.record2task.task_store.tasks()[0].generated.task))
    first.close()
    task_generator = ScriptedGenerator()
    restarted, _ = rounds(tmp_path, task_generator, player, rounds=1)
    looked(restarted)
    ((result,),) = task_generator.results
    assert result.is_played and player.deleted == ["harbor-00000-000"]


def test_a_generator_that_fails_at_once_waits_between_deferred_rounds(tmp_path: Path) -> None:
    task_generator = FailingGenerator()
    p, _ = rounds(tmp_path, task_generator, rounds=50, generation_attempts=2, **SOURCES)
    sessions(p, "a")
    for _ in range(60):
        p.ready()
    assert len(task_generator.requests) == 2, "one deferred round, two attempts, then a wait"
    assert not p.record2record.rounds_stopped and p.status()["task_generation"]["completed"] == 0
    assert not {"rec-a", "rep-a"} & p.releasable_record_ids(), "the deferred round keeps its session"


class FailingGenerator(ScriptedGenerator):
    """A generator whose generate call fails to run at once."""

    async def generate(self, request: TaskGenerationRequest) -> tuple[GeneratedTask, ...]:
        self.requests.append(request)
        raise RuntimeError("the generator service is down")


class FirstResultsFail(ScriptedGenerator):
    """A generator that cannot take the results of generation 0."""

    async def record_results(self, results: tuple[TaskPlayResult, ...], *, generation: int) -> None:
        if generation == 0:
            raise RuntimeError("generation 0 is lost")
        await super().record_results(results, generation=generation)


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
            p.ingest(record)


def source_ids(request: TaskGenerationRequest) -> list[str]:
    return [record.agent_record_id for record in request.source_records]


def test_each_round_gives_the_generator_at_most_its_share_of_tagged_sessions(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator()
    p, _ = rounds(tmp_path, task_generator, **SOURCES)
    sessions(p, "a", "b", "c")
    sessions(p, "untagged", tags={"source": "other"})
    for _ in range(3):
        looked(p)
    # Each round played no task, so the next one followed at once.
    assert [source_ids(request) for request in task_generator.requests] == [
        ["rec-a", "rep-a", "rec-b", "rep-b"],
        ["rec-c", "rep-c"],
        [],
    ]
    assert all(request.description == DESCRIPTION for request in task_generator.requests)
    assert generation_document(tmp_path, 0)["source_record_ids"] == ["rec-a", "rep-a", "rec-b", "rep-b"]
    assert generation_document(tmp_path, 2)["source_record_ids"] == []


def test_a_source_sessions_records_stay_until_the_round_that_uses_it_finishes(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator(), worker=worker, **SOURCES)
    sessions(p, "a")
    sessions(p, "untagged", tags={})
    assert {"rec-untagged", "rep-untagged"} <= p.releasable_record_ids(), "an untagged session is released"
    assert not {"rec-a", "rep-a"} & p.releasable_record_ids()
    assert p.operational_metrics()["waiting_source_sessions"] == 1
    p.ready()
    assert [record.agent_record_id for record in worker.submitted[0].source_records] == ["rec-a", "rep-a"]
    assert not {"rec-a", "rep-a"} & p.releasable_record_ids(), "the round runs"
    worker.finish()
    p.ready()
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
    first.ready()
    worker.finish()
    first.ready()
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
    first.ready()
    worker.finish()
    first.ready()
    first.close()
    task_generator = ScriptedGenerator()
    restarted, _ = rounds(tmp_path, task_generator, **config)
    sessions(restarted, "a", "b")
    for _ in range(6):
        restarted.ready()
    assert [source_ids(request) for request in task_generator.requests] == [["rec-b", "rep-b"]]


def test_a_run_takes_the_sessions_that_waited_when_it_started_and_stops_at_its_round_limit(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator()
    p, _ = rounds(
        tmp_path,
        task_generator,
        rounds=4,
        source_tag="source=session",
        source_records_per_round=1,
        max_rounds_per_run=2,
    )
    sessions(p, "a", "b", "c")
    p.ready()
    sessions(p, "d")
    for _ in range(4):
        p.ready()
    assert [source_ids(request) for request in task_generator.requests] == [
        ["rec-a", "rep-a"],
        ["rec-b", "rep-b"],
        ["rec-c", "rep-c"],
        ["rec-d", "rep-d"],
    ], "d waits for the second run; c, which the first run did not reach, opens it"


def test_a_deferred_round_runs_again_with_its_sessions_when_new_sessions_start_a_run(tmp_path: Path) -> None:
    task_generator = ScriptedGenerator([generated(0, 0)], failing="harbor-00000-000")
    p, _ = rounds(tmp_path, task_generator, rounds=4, generation_attempts=1, **SOURCES)
    sessions(p, "a")
    p.ready()
    p.ready()
    assert p.operational_metrics()["deferred_rounds"] == 1
    assert not {"rec-a", "rep-a"} & p.releasable_record_ids(), "a deferred round keeps its sessions"
    task_generator.failing = None
    sessions(p, "b")
    for _ in range(3):
        p.ready()
    # Session b makes a round due and starts a run, which first runs round 0's sessions again.
    assert [source_ids(request) for request in task_generator.requests] == [["rec-a", "rep-a"]]
    store = p.record2task.task_store
    assert store.generation(generation=0).is_complete, "the deferred generation completed from its saved candidates"
    assert store.counts()["played/train"] == 1


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"tasks_per_step": 0}, "tasks_per_step must be positive"),
        ({"plays_per_task": 1}, "at least two"),
        ({"rounds": 1, "description": "x", "state_dir": "s"}, "pipeline.record2task.config.generator"),
        ({"source_records_per_round": 1}, "needs a source_tag"),
        ({"source_tag": "session"}, "name=value"),
    ],
)
def test_a_processor_configuration_that_cannot_train_is_refused(config: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        processor(**config)


def test_rounds_need_a_description_a_state_dir_and_the_generator_service() -> None:
    def built(**config: object) -> GeneratedTaskTasks:
        context = PipelineContext("tasks")
        group_settings, round_settings, _ = stage_settings({"rounds": 1, **config})
        stage = GeneratedTaskRecord2Record(context, group_settings)
        return GeneratedTaskTasks(context, round_settings, stage, task_generator=ScriptedGenerator())

    with pytest.raises(ValueError, match="description"):
        built()
    with pytest.raises(ValueError, match="generator_url"):
        built(description="x", state_dir="s")
    with pytest.raises(ValueError, match="generator_url"):
        built(description="x", state_dir="s", generator_url="${endpoints.generator}")


# ------------------------------------------------------------------------------------------------- the recipe


def training_recipe(**fields: Any) -> GeneratedTaskRecipe:
    runtime = InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8000")
    return GeneratedTaskRecipe(training_runtime=StubTrainingRuntime(), runtime=runtime, **fields)


def test_the_recipe_binds_the_objective_its_schedule_and_its_stages() -> None:
    spec = GeneratedTaskRecipe.training_spec()
    assert GeneratedTaskRecipe.pipeline_spec().task2batch is GeneratedTaskBatches and spec.objective == "task_group"
    assert spec.loss_family == "importance_sampling"
    assert spec.scheduling.unit == "sample" and spec.scheduling.batch_size == "actual"
    assert GeneratedTaskRecipe.report_type.fget(GeneratedTaskRecipe) is ScoredRolloutReport  # type: ignore[union-attr]
    built = training_recipe().build_pipeline(PipelineContext("one"))
    assert isinstance(built.record2record, GeneratedTaskRecord2Record)
    assert isinstance(built.record2task, GeneratedTaskTasks) and built.record2task.worker is None, "no round runs"
    assert isinstance(built.task2batch, GeneratedTaskBatches) and built.batch_size == 4
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
            "record2record": {
                "module": "reef.recipe.generated_tasks:GeneratedTaskRecord2Record",
                "config": {"plays_per_task": 3},
            },
            "record2task": {
                "module": "reef.recipe.generated_tasks:GeneratedTaskTasks",
                "config": {
                    "rounds": 1,
                    "generator_url": "http://g:8910",
                    "description": "shell tasks",
                    "state_dir": str(tmp_path / "state"),
                    "generator": generator,
                },
            },
            "task2batch": {
                "module": "reef.recipe.generated_tasks:GeneratedTaskBatches",
                "config": {"tasks_per_step": 2},
            },
        },
    )
    ScriptedGenerator.built.clear()
    built = recipe.build_pipeline(PipelineContext("one"))
    assert isinstance(built.record2task, GeneratedTaskTasks)
    assert isinstance(built.record2task.task_generator, ScriptedGenerator)
    assert ScriptedGenerator.built == [{"config": {"count": 3}, "scenario": "one", "environ": environ}]
    assert built.record2task.plays_per_task == 3, "the plays come from the record stage's groups"
    assert built.record2task.state_dir == tmp_path / "state" / "one"
    assert built.batch_size == 2
    assert recipe.scenario_state_dirs("one") == (tmp_path / "state" / "one",)
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
        ("reef.core.tasks:TaskGenerator", "names an abstract class"),
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
