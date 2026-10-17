"""The generated-task recipe: rounds of generated tasks played and sent back, episodes grouped by task and trained."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import pytest
from tests.support.runtime_stubs import StubTrainingRuntime
from tests.support.task_service_stubs import StandInGenerator, task_for

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord, RequestType
from reef.core.plugins import PluginConfigError
from reef.core.reports import ScoredRolloutReport
from reef.core.tasks import (
    GeneratedTask,
    HarborTask,
    PlayArm,
    TaskGenerationRequest,
    TaskGenerator,
    TaskPlayResult,
    TaskSplit,
    TaskValidationResult,
    read_split_manifest,
    source_split,
)
from reef.harness.client.tasks import TaskPlay
from reef.inference.http import InferenceProxyRuntime
from reef.pipeline_context import PipelineContext
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks import GeneratedTaskBatches, GeneratedTaskRecipe, TaskGroupObjective
from reef.recipe.generated_tasks.processor import (
    GeneratedTaskRecord2Record,
    GeneratedTaskTasks,
    RoundJob,
    RoundSettings,
    TaskBatchSettings,
    TaskGroupSettings,
    episode_assembly,
    reported_task_name,
    rewards_of,
)
from reef.record2record import GroupDecision, ReportContext
from reef.train.algos.registry import resolve_objective
from reef.train.pipeline import TrainingPipeline
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
        (lambda: TaskPlayResult(task_for("t-a", "r", 1), split="test"), "split must be train, eval"),  # type: ignore[arg-type]
        (lambda: TaskPlayResult(task_for("t-a", "r", 1), error="refused", split="eval"), "no split"),
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

    async def record_results(self, results: tuple[TaskPlayResult, ...]) -> None:
        self.results.append(results)
        if self.failing_results:
            raise RuntimeError("the generator lost its state directory")


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
        if not isinstance(job, RoundJob):
            raise TypeError("the round worker takes RoundJob")
        try:
            self.outcomes.append(asyncio.run(self.generation.run_round(job)))
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
        self.submitted: list[RoundJob] = []
        self.waiting: list[RoundJob] = []
        self.finished: list[SupportsReceipt] = []

    def submit(self, job: SupportsReceipt) -> bool:
        if not isinstance(job, RoundJob):
            raise TypeError("the round worker takes RoundJob")
        self.submitted.append(job)
        self.waiting.append(job)
        return True

    def finish(self) -> None:
        """Run every round taken so far, on this thread; the next poll hands their outcomes back."""
        if self.generation is None:
            raise RuntimeError("no generation bound")
        jobs, self.waiting = self.waiting, []
        for job in jobs:
            self.finished.append(asyncio.run(self.generation.run_round(job)))

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
    # Built as a deployment builds it, so the batch stage tells the rounds its tasks_per_step.
    batches = GeneratedTaskBatches.from_config(
        dataclasses.asdict(batch_settings), context=context, record2record=stage, record2task=generation, environ={}
    )
    built = TrainingPipeline(context, record2record=stage, record2task=generation, task2batch=batches)
    worker.generation = generation
    return built, player


def looked(p: TrainingPipeline) -> bool:
    """Look for a batch twice: the inline worker's outcome lands on the look after the one that submitted it."""
    p.ready()
    return p.ready()


def summary(tmp_path: Path, round_number: int) -> dict[str, Any]:
    return json.loads((tmp_path / "state" / "tasks" / f"round-{round_number:05d}.json").read_text())


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
    assert (status["next"], status["completed"], status["tasks_generated"], status["tasks_played"]) == (1, 1, 3, 3)

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
    assert [result.task.name for result in results] == names
    assert results[0].training_rewards == (0.25, 0.25) and results[0].measurement_rewards == {"hint": (0.75,)}
    assert results[2].training_rewards == (0.25, 0.25) and results[2].measurement_rewards == {}
    assert results[2].is_played
    document = summary(tmp_path, 0)
    assert document["error"] == "" and [task["name"] for task in document["tasks"]] == names
    assert document["tasks"][0]["training_rewards"] == [0.25, 0.25]
    assert document["tasks"][0]["measurement_rewards"] == {"hint": [0.75]}
    assert p.operational_metrics()["rounds_completed"] == 1


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
    assert p.status()["task_generation"]["tasks_played"] == 1


@pytest.mark.parametrize(
    ("arm", "role", "eval_fraction"),
    [("plain", "training", 0.0), ("plain", "training", 0.5), ("heldout", "held out", 0.5)],
)
def test_a_measurement_arm_with_the_name_of_the_training_or_held_out_arm_is_refused(
    tmp_path: Path, arm: str, role: str, eval_fraction: float
) -> None:
    clashing = GeneratedTask(task_for("harbor-00000-000", "designer-0", 9000), (PlayArm(arm, 1),))
    task_generator = ScriptedGenerator([(clashing, *generated(0, 1))])
    player = StandInGenerator(tmp_path / "tasks", rewards=SPLIT_REWARDS)
    seed = seed_placing({"designer-1": "train"})
    p, _ = rounds(tmp_path, task_generator, player, eval_fraction=eval_fraction, split_seed=seed)
    assert not looked(p)
    assert task_generator.validated == ["harbor-00000-001"] and player.writes == ["harbor-00000-001"]
    (results,) = task_generator.results
    assert results[0].error == f"the measurement arm {arm!r} has the name of the {role} arm"


def test_without_an_eval_fraction_a_measurement_arm_may_be_named_heldout(tmp_path: Path) -> None:
    task = GeneratedTask(task_for("harbor-00000-000", "designer-0", 9000), (PlayArm("heldout", 1),))
    task_generator = ScriptedGenerator([(task,)])
    p, player = rounds(tmp_path, task_generator, StandInGenerator(tmp_path / "tasks", rewards=SPLIT_REWARDS))
    assert not looked(p)
    assert [(call["arm"], call["plays"]) for call in player.plays] == [("plain", 2), ("heldout", 1)]
    (results,) = task_generator.results
    assert results[0].is_played and results[0].split == "train" and set(results[0].measurement_rewards) == {"heldout"}


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
    assert not p.ready() and [job.round_number for job in worker.submitted] == [0]
    for index in range(reported):
        played(p, "harbor-00000-000", index, float(index % 2))
    assert not p.ready(), "the plays are still out, so the group waits for all four"
    worker.finish()
    assert p.ready() is is_trained
    if is_trained:
        assert len(p.build_batch().items) == 2
    else:
        assert {"rep-harbor-00000-000-0", "rec-harbor-00000-000-0"} <= p.releasable_record_ids()


def test_a_restarted_processor_closes_a_group_at_the_count_its_round_summary_kept(tmp_path: Path) -> None:
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
    assert sorted(entry.name for entry in (tmp_path / "state" / "tasks").iterdir()) == [
        "round-00000.json",
        "round-00001.json",
    ]


def test_batches_trained_while_a_round_runs_count_toward_the_next_one(tmp_path: Path) -> None:
    worker = DeferredWorker()
    task_generator = ScriptedGenerator([generated(0, 0, 1), generated(1, 0, 1)])
    p, _ = rounds(tmp_path, task_generator, worker=worker, batches_per_round=2)
    assert not p.ready() and [job.round_number for job in worker.submitted] == [0]
    assert p.background_jobs_running() and p.status()["task_generation"]["in_flight"] == 0
    for batch in range(2):
        for task in (f"t-{batch}-0", f"t-{batch}-1"):
            played(p, task, 0, 1.0)
            played(p, task, 1, 0.0)
        p.acknowledge(p.build_batch().batch_id)
        assert not p.ready() and [job.round_number for job in worker.submitted] == [0], "one round at a time"
    worker.finish()
    assert not p.ready(), "round 0 lands on this look and round 1 starts on it"
    assert [job.round_number for job in worker.submitted] == [0, 1]
    status = p.status()["task_generation"]
    assert (status["in_flight"], status["next"], status["completed"]) == (1, 2, 1)


def test_a_dropped_batch_does_not_count_toward_the_next_round(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), worker=worker)
    assert not p.ready() and [job.round_number for job in worker.submitted] == [0]
    for task in ("t-0", "t-1"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    batch_id = p.build_batch().batch_id
    p.acknowledge(batch_id, trained=False)
    worker.finish()
    assert not p.ready(), "round 0 lands; the dropped batch trained nothing"
    assert [job.round_number for job in worker.submitted] == [0]
    for task in ("t-2", "t-3"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    p.acknowledge(p.build_batch().batch_id)
    assert not p.ready(), "a trained batch counts"
    assert [job.round_number for job in worker.submitted] == [0, 1]


def test_a_round_that_played_no_task_does_not_block_the_next_one(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, _ = rounds(tmp_path, ScriptedGenerator([]), worker=worker)
    assert not p.ready() and [job.round_number for job in worker.submitted] == [0]
    worker.finish()
    assert not p.ready(), "round 0 lands without a task and round 1 starts on the same look"
    assert [job.round_number for job in worker.submitted] == [0, 1], "no batch can come; none was acknowledged"


def test_a_restarted_processor_carries_on_from_the_round_summaries(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    ("task_generator", "played_names"),
    [
        (ScriptedGenerator([generated(0, 0, 1)], failing="harbor-00000-001"), ["harbor-00000-000"]),
        (ScriptedGenerator([generated(0, 0, 1)], failing_results=True), ["harbor-00000-000", "harbor-00000-001"]),
    ],
)
def test_a_generator_failure_ends_the_round_with_what_ran(
    tmp_path: Path, task_generator: ScriptedGenerator, played_names: list[str]
) -> None:
    p, player = rounds(tmp_path, task_generator)
    assert not looked(p)
    assert player.writes == played_names
    assert [result.task.name for result in task_generator.results[0]] == played_names, "the results reach it"
    error = "the checks could not run" if task_generator.failing else "the generator lost its state directory"
    status = p.status()["task_generation"]
    assert status["completed"] == 1 and status["last_error"] == error
    assert summary(tmp_path, 0)["error"] == error


def test_a_round_that_never_finishes_is_named_in_the_status(tmp_path: Path) -> None:
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]))
    p.close()
    assert not looked(p)
    assert "no further round runs" in p.status()["task_generation"]["last_error"]


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
        assert (tmp_path / "state" / "tasks" / "round-00000.json").is_file()
    finally:
        p.close()


# ------------------------------------------------------------------------------------------- the splits

#: The task player's reward on each arm; the held out arm plays as the training arm does.
SPLIT_REWARDS = {"plain": 0.25, "heldout": 0.5, "hint": 0.75}


def seed_placing(splits: Mapping[str, str], eval_fraction: float = 0.5) -> int:
    """The first seed that places a task made from each record in the split given for it."""
    return next(
        seed
        for seed in range(1000)
        if all(
            source_split(seed, "unused", (record,), eval_fraction=eval_fraction) == split
            for record, split in splits.items()
        )
    )


def test_an_eval_task_plays_on_the_held_out_arm_and_never_reaches_a_batch(tmp_path: Path) -> None:
    worker = DeferredWorker()
    player = StandInGenerator(tmp_path / "tasks", rewards=SPLIT_REWARDS)
    task_generator = ScriptedGenerator([generated(0, 0, 1)])
    seed = seed_placing({"designer-0": "eval", "designer-1": "train"})
    p, _ = rounds(
        tmp_path, task_generator, player, worker=worker, eval_fraction=0.5, split_seed=seed, tasks_per_step=1
    )
    assert not p.ready()
    worker.finish()
    assert not p.ready(), "round 0 lands; its train task makes a batch to wait for"
    assert [job.round_number for job in worker.submitted] == [0]
    assert [(call["task"], call["arm"], call["is_reporting"], call["tags"]) for call in player.plays] == [
        ("harbor-00000-000", "heldout", True, {"round": "0"}),
        ("harbor-00000-000", "hint", True, {"round": "0"}),
        ("harbor-00000-001", "plain", True, {"round": "0"}),
        ("harbor-00000-001", "hint", True, {"round": "0"}),
    ]
    stage = p.record2record
    assert isinstance(stage, GeneratedTaskRecord2Record)
    assert stage.task_grouping.reported_plays == {"harbor-00000-001": 2}, "an eval task's plays are not counted"
    assert [task["reported_plays"] for task in summary(tmp_path, 0)["tasks"]] == [None, 2]

    played(p, "harbor-00000-000", 0, 1.0, arm="heldout")
    played(p, "harbor-00000-000", 1, 0.0, arm="heldout")
    assert not p.ready(), "the held out plays form no group"
    held_out = {f"{kind}-harbor-00000-000-{index}" for kind in ("rep", "rec") for index in range(2)}
    assert held_out <= p.releasable_record_ids()
    played(p, "harbor-00000-001", 0, 1.0)
    played(p, "harbor-00000-001", 1, 0.0)
    assert p.ready()
    groups = trajectory_groups(p.build_batch())
    assert [[sample.group_id for sample in group] for group in groups] == [["harbor-00000-001", "harbor-00000-001"]]

    (results,) = task_generator.results
    assert [(result.task.name, result.split, result.training_rewards) for result in results] == [
        ("harbor-00000-000", "eval", (0.5, 0.5)),
        ("harbor-00000-001", "train", (0.25, 0.25)),
    ]
    assert results[0].measurement_rewards == {"hint": (0.75,)}
    assert [task["split"] for task in summary(tmp_path, 0)["tasks"]] == ["eval", "train"]


@pytest.mark.parametrize("config", [{}, {"eval_fraction": 0.0, "split_seed": 5}])
def test_without_an_eval_fraction_every_task_trains_as_before(tmp_path: Path, config: dict[str, Any]) -> None:
    worker = DeferredWorker()
    task_generator = ScriptedGenerator([generated(0, 0, 1), generated(1, 0)])
    p, player = rounds(tmp_path, task_generator, worker=worker, **config)
    assert not p.ready()
    worker.finish()
    assert not p.ready() and [job.round_number for job in worker.submitted] == [0], "round 1 waits for a batch"
    assert [(call["arm"], call["is_reporting"], call["tags"]) for call in player.plays] == [
        ("plain", True, {"round": "0"}),
        ("hint", True, {"round": "0"}),
        ("plain", True, {"round": "0"}),
        ("hint", True, {"round": "0"}),
    ]
    for task in ("harbor-00000-000", "harbor-00000-001"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    assert p.ready()
    p.acknowledge(p.build_batch().batch_id)
    assert not p.ready() and [job.round_number for job in worker.submitted] == [0, 1]
    assert [result.split for result in task_generator.results[0]] == ["train", "train"]
    assert not (tmp_path / "state" / "tasks" / "split-manifest.json").exists()
    document = summary(tmp_path, 0)
    assert set(document) == {"round", "error", "source_record_ids", "tasks"}
    assert set(document["tasks"][0]) == {"name", "error", "training_rewards", "measurement_rewards", "reported_plays"}


def test_tasks_made_from_the_same_records_land_in_one_split(tmp_path: Path) -> None:
    player = StandInGenerator(tmp_path / "tasks", rewards=SPLIT_REWARDS)
    tasks = tuple(
        GeneratedTask(task_for(f"harbor-00000-{index:03d}", record, 9000 + index))
        for index, record in enumerate(("session-a", "session-b", "session-a", "session-b"))
    )
    task_generator = ScriptedGenerator([tasks])
    seed = seed_placing({"session-a": "eval", "session-b": "train"})
    p, _ = rounds(tmp_path, task_generator, player, eval_fraction=0.5, split_seed=seed)
    assert not looked(p)
    assert [result.split for result in task_generator.results[0]] == ["eval", "train", "eval", "train"]
    assert [call["arm"] for call in player.plays] == ["heldout", "plain", "heldout", "plain"]


def test_a_round_of_eval_tasks_only_does_not_block_the_next_one(tmp_path: Path) -> None:
    worker = DeferredWorker()
    player = StandInGenerator(tmp_path / "tasks", rewards=SPLIT_REWARDS)
    seed = seed_placing({"designer-0": "eval"})
    p, _ = rounds(
        tmp_path, ScriptedGenerator([generated(0, 0)]), player, worker=worker, eval_fraction=0.5, split_seed=seed
    )
    assert not p.ready() and [job.round_number for job in worker.submitted] == [0]
    worker.finish()
    assert not p.ready(), "round 0 lands with only an eval task, and round 1 starts on the same look"
    assert [job.round_number for job in worker.submitted] == [0, 1], "no batch can come; none was acknowledged"
    assert p.status()["task_generation"]["tasks_played"] == 1


def test_the_split_manifest_lists_every_played_task_and_a_restart_places_tasks_the_same_way(tmp_path: Path) -> None:
    state = tmp_path / "state" / "tasks"
    player = StandInGenerator(tmp_path / "tasks", rewards=SPLIT_REWARDS)
    seed = seed_placing({"designer-0": "eval", "designer-1": "train"})
    # One train group fills a batch, so no round follows at once.
    config: dict[str, Any] = {"eval_fraction": 0.5, "split_seed": seed, "tasks_per_step": 1}
    task_generator = ScriptedGenerator([generated(0, 0, 1, 2)], invalid={"harbor-00000-002": "oracle check refused"})
    first, _ = rounds(tmp_path, task_generator, player, **config)
    assert not looked(first)
    first.close()
    manifest = read_split_manifest(state / "split-manifest.json")
    assert manifest == TaskSplit(("harbor-00000-001",), ("harbor-00000-000",), seed, 0.5)
    assert (summary(tmp_path, 0)["eval_fraction"], summary(tmp_path, 0)["split_seed"]) == (0.5, seed)

    restarted_generator = ScriptedGenerator([generated(1, 0, 1)])
    restarted, _ = rounds(tmp_path, restarted_generator, player, **config)
    assert not looked(restarted)
    assert [(result.task.name, result.split) for result in restarted_generator.results[0]] == [
        ("harbor-00001-000", "eval"),
        ("harbor-00001-001", "train"),
    ], "tasks from the records of round 0 go where round 0 placed them"
    manifest = read_split_manifest(state / "split-manifest.json")
    assert manifest == TaskSplit(
        ("harbor-00000-001", "harbor-00001-001"), ("harbor-00000-000", "harbor-00001-000"), seed, 0.5
    )


def test_a_restart_with_other_split_settings_is_refused(tmp_path: Path) -> None:
    seed = seed_placing({"designer-0": "eval"})
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), eval_fraction=0.5, split_seed=seed)
    assert not looked(first)
    first.close()
    changed: tuple[dict[str, Any], ...] = ({"eval_fraction": 0.5, "split_seed": seed + 1}, {"eval_fraction": 0.25}, {})
    for config in changed:
        with pytest.raises(PluginConfigError, match="keeps both for the life of its state_dir"):
            rounds(tmp_path, ScriptedGenerator(), **config)
    rounds(tmp_path, ScriptedGenerator(), eval_fraction=0.5, split_seed=seed)[0].close()


def test_a_summary_without_split_settings_counts_as_no_eval_fraction(tmp_path: Path) -> None:
    state = tmp_path / "state" / "tasks"
    state.mkdir(parents=True)
    old_tasks = [{"name": "harbor-old-000", "error": ""}, {"name": "harbor-old-001", "error": "oracle check refused"}]
    (state / "round-00000.json").write_text(json.dumps({"round": 0, "error": "", "tasks": old_tasks}))
    with pytest.raises(PluginConfigError, match=r"eval_fraction 0\.0 and split_seed None, not 0\.5"):
        rounds(tmp_path, ScriptedGenerator(), eval_fraction=0.5)
    # Without an eval fraction the seed places nothing, so it may change.
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(1, 0)]), split_seed=9)
    assert not looked(p)
    assert summary(tmp_path, 1)["tasks"][0]["name"] == "harbor-00001-000"


def test_a_split_cannot_start_on_rounds_that_trained_without_one(tmp_path: Path) -> None:
    first, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]))
    assert not looked(first)
    first.close()
    with pytest.raises(PluginConfigError, match=r"round 0 under .* eval_fraction 0\.0 and split_seed None, not 0\.5"):
        rounds(tmp_path, ScriptedGenerator(), eval_fraction=0.5)


def test_a_split_with_no_eval_task_yet_logs_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    seed = seed_placing({"designer-0": "train"})
    p, _ = rounds(tmp_path, ScriptedGenerator([generated(0, 0)]), eval_fraction=0.5, split_seed=seed)
    with caplog.at_level(logging.WARNING, logger="reef.recipe.generated_tasks.processor"):
        assert not looked(p)
    assert "has no eval task yet: none of its 1 played tasks hashed to eval" in caplog.text


@pytest.mark.parametrize(
    ("indices", "eval_fraction", "follows"),
    [((0, 1, 2, 3), 0.0, False), ((0, 1, 2, 3), 0.5, True), ((0, 1, 2), 0.5, False)],
)
def test_a_round_its_eval_tasks_left_short_of_the_batches_is_followed_at_once(
    tmp_path: Path, indices: tuple[int, ...], eval_fraction: float, follows: bool
) -> None:
    # Two batches of two groups: four train groups make the batches the next round waits for.
    worker = DeferredWorker()
    seed = seed_placing({"designer-0": "eval", "designer-1": "train", "designer-2": "train", "designer-3": "train"})
    task_generator = ScriptedGenerator([generated(0, *indices)])
    player = StandInGenerator(tmp_path / "tasks", rewards=SPLIT_REWARDS)
    config: dict[str, Any] = {"eval_fraction": eval_fraction, "split_seed": seed, "batches_per_round": 2}
    p, _ = rounds(tmp_path, task_generator, player, worker=worker, **config)
    p.ready()
    worker.finish()
    assert not p.ready()
    splits = [result.split for result in task_generator.results[0]]
    assert splits == ["eval" if eval_fraction and index == 0 else "train" for index in indices]
    # Three train groups of four tasks cannot fill the batches. Three tasks could not without an eval task either,
    # so that round waits for batches, as before.
    assert [job.round_number for job in worker.submitted] == ([0, 1] if follows else [0])


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
    assert summary(tmp_path, 0)["source_record_ids"] == ["rec-a", "rep-a", "rec-b", "rep-b"]
    assert summary(tmp_path, 2)["source_record_ids"] == []


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


def test_a_source_session_takes_part_in_the_split_as_its_report(tmp_path: Path) -> None:
    # The draw of each session's report places the task; the draw of a model call in the session does not.
    seed = seed_placing({"rep-a": "train", "rec-a": "eval", "rep-b": "eval", "rec-b": "train"})
    whole = dataclasses.replace(task_for("harbor-00000-000", "x", 9000), source_agent_record_ids=("rec-a", "rep-a"))
    call = dataclasses.replace(task_for("harbor-00000-001", "x", 9001), source_agent_record_ids=("rec-b",))
    task_generator = ScriptedGenerator([(GeneratedTask(whole), GeneratedTask(call))])
    player = StandInGenerator(tmp_path / "tasks", rewards=SPLIT_REWARDS)
    config: dict[str, Any] = {**SOURCES, "eval_fraction": 0.5, "split_seed": seed}
    p, _ = rounds(tmp_path, task_generator, player, **config)
    sessions(p, "a", "b")
    assert not looked(p)
    assert source_ids(task_generator.requests[0]) == ["rec-a", "rep-a", "rec-b", "rep-b"]
    assert [result.split for result in task_generator.results[0]] == ["train", "eval"]


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


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"tasks_per_step": 0}, "tasks_per_step must be positive"),
        ({"plays_per_task": 1}, "at least two"),
        ({"rounds": 1, "description": "x", "state_dir": "s"}, "pipeline.record2task.config.generator"),
        ({"source_records_per_round": 1}, "needs a source_tag"),
        ({"source_tag": "session"}, "name=value"),
        ({"eval_fraction": 1.0}, "eval_fraction must be at least 0 and below 1"),
        ({"eval_fraction": -0.1}, "eval_fraction must be at least 0 and below 1"),
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
                    "eval_fraction": 0.25,
                    "split_seed": 7,
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
    assert (built.record2task.eval_fraction, built.record2task.split_seed) == (0.25, 7)
    assert built.batch_size == 2 and built.record2task.tasks_per_step == 2, "the batch stage tells the rounds"
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
