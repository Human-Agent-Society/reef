"""The generated-task recipe: rounds of generated tasks played and sent back, episodes grouped by task and trained."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime
from reef_service.task_service_stubs import StandInGenerator, task_for

from reef.core import AgentRecord, RequestType
from reef.core.plugins import PluginReference
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
from reef.core.trajectories import make_trajectory
from reef.harness.client.tasks import TaskPlay
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.errors import RecipeConfigError
from reef.record2dataset.generated_tasks import (
    GeneratedTaskProcessor,
    GeneratedTaskRecipe,
    TaskGenerationConfig,
    TaskGroupObjective,
)
from reef.record2dataset.generated_tasks.processor import RoundJob, reported_task_name, rewards_of
from reef.train.algos.registry import resolve_objective
from reef.train.processors.computed import Failed, JudgingWorker, SupportsReceipt
from reef.train.processors.reported import GroupDecision
from reef.train.types import ProcessorContext, TrainingBatch, TrajectoryItem, trajectory_groups

MODULE = "reef_service.test_generated_tasks"
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


def processor(**config: object) -> GeneratedTaskProcessor:
    return GeneratedTaskProcessor(ProcessorContext("tasks", {"tasks_per_step": 2, "plays_per_task": 2, **config}))


def played(processor: GeneratedTaskProcessor, task: str, index: int, score: float, **fields: Any) -> None:
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


def test_decide_group_waits_for_plays_per_task_episodes() -> None:
    p = processor(plays_per_task=3)
    one: tuple[TrajectoryItem, ...] = (make_trajectory((inference("a"),), 1.0),)
    assert p.decide_group("t", one) is GroupDecision.INCOMPLETE
    assert p.decide_group("t", one * 3) is GroupDecision.READY


def test_the_assembly_spans_an_episodes_turns_and_realigns_the_think_scaffold() -> None:
    assembly = processor().assembly
    assert assembly.accept_multi_turn and assembly.scaffold_tolerance == 8
    assert processor(scaffold_tolerance=2).assembly.scaffold_tolerance == 2


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


class InlineWorker(JudgingWorker):
    """Runs each round the moment it is submitted, on this thread, so a test reads the outcome right away."""

    def __init__(self) -> None:
        super().__init__(self.unbound, concurrency=1)
        self.processor: GeneratedTaskProcessor | None = None
        self.submitted: list[SupportsReceipt] = []
        self.outcomes: list[SupportsReceipt] = []

    async def unbound(self, job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("no processor bound")

    def submit(self, job: SupportsReceipt) -> bool:
        self.submitted.append(job)
        if self.processor is None:
            return False
        if not isinstance(job, RoundJob):
            raise TypeError("the round worker takes RoundJob")
        try:
            self.outcomes.append(asyncio.run(self.processor.run_round(job)))
        except Exception:
            self.outcomes.append(Failed(job.receipt))
        return True

    def poll(self) -> list[SupportsReceipt]:
        outcomes, self.outcomes = self.outcomes, []
        return outcomes

    def close(self) -> None:
        self.processor = None


class DeferredWorker(JudgingWorker):
    """Takes a round the moment it is submitted and runs it on ``finish``; the outcome lands on the next poll."""

    def __init__(self) -> None:
        super().__init__(self.unbound, concurrency=1)
        self.processor: GeneratedTaskProcessor | None = None
        self.submitted: list[RoundJob] = []
        self.waiting: list[RoundJob] = []
        self.finished: list[SupportsReceipt] = []

    async def unbound(self, job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("no processor bound")

    def submit(self, job: SupportsReceipt) -> bool:
        if not isinstance(job, RoundJob):
            raise TypeError("the round worker takes RoundJob")
        self.submitted.append(job)
        self.waiting.append(job)
        return True

    def finish(self) -> None:
        """Run every round taken so far, on this thread; the next poll hands their outcomes back."""
        if self.processor is None:
            raise RuntimeError("no processor bound")
        jobs, self.waiting = self.waiting, []
        for job in jobs:
            self.finished.append(asyncio.run(self.processor.run_round(job)))

    def poll(self) -> list[SupportsReceipt]:
        outcomes, self.finished = self.finished, []
        return outcomes

    def close(self) -> None:
        self.processor = None


def rounds(
    tmp_path: Path,
    task_generator: ScriptedGenerator,
    player: StandInGenerator | None = None,
    *,
    worker: InlineWorker | DeferredWorker | None = None,
    **config: object,
) -> tuple[GeneratedTaskProcessor, StandInGenerator]:
    player = player if player is not None else StandInGenerator(tmp_path / "tasks")
    worker = worker if worker is not None else InlineWorker()
    built = GeneratedTaskProcessor(
        ProcessorContext(
            "tasks",
            {
                "tasks_per_step": 2,
                "plays_per_task": 2,
                "rounds": 3,
                "description": DESCRIPTION,
                "state_dir": str(tmp_path / "state"),
                "served_model": "Qwen/Qwen3-8B",
                **config,
            },
        ),
        task_generator=task_generator,
        task_player=player,
        worker=worker,
    )
    worker.processor = built
    return built, player


def looked(p: GeneratedTaskProcessor) -> bool:
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
    assert not p.derivation_pending(), "the inline worker finished before ready() returned"
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
    assert p.derivation_pending() and p.status()["task_generation"]["in_flight"] == 0
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
    p.dropped(batch_id)
    p.acknowledge(batch_id)
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
    p = GeneratedTaskProcessor(
        ProcessorContext("tasks", {"rounds": 1, "description": "shell tasks", "state_dir": str(tmp_path / "state")}),
        task_generator=ScriptedGenerator([generated(0, 0)]),
        task_player=StandInGenerator(tmp_path / "tasks"),
    )
    try:
        assert not p.ready()
        deadline = time.monotonic() + 10.0
        while p.derivation_pending() and time.monotonic() < deadline:
            time.sleep(0.02)
            p.ready()
        assert not p.derivation_pending() and p.status()["task_generation"]["completed"] == 1
        assert (tmp_path / "state" / "tasks" / "round-00000.json").is_file()
    finally:
        p.close()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"tasks_per_step": 0}, "tasks_per_step must be positive"),
        ({"plays_per_task": 1}, "at least two"),
        ({"rounds": 1}, "task-generation section"),
    ],
)
def test_a_processor_configuration_that_cannot_train_is_refused(config: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        processor(**config)


def test_rounds_need_a_description_a_state_dir_and_the_generator_service() -> None:
    def built(**config: object) -> GeneratedTaskProcessor:
        return GeneratedTaskProcessor(
            ProcessorContext("tasks", {"rounds": 1, **config}), task_generator=ScriptedGenerator()
        )

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


def test_the_recipe_binds_the_processor_the_objective_and_its_schedule() -> None:
    spec = GeneratedTaskRecipe.training_spec()
    assert spec.processor is GeneratedTaskProcessor and spec.objective == "task_group"
    assert spec.loss_family == "importance_sampling"
    assert spec.scheduling.unit == "sample" and spec.scheduling.batch_size == "actual"
    assert GeneratedTaskRecipe.report_type.fget(GeneratedTaskRecipe) is ScoredRolloutReport  # type: ignore[union-attr]
    config = training_recipe().processor_config()
    assert config["tasks_per_step"] == 4 and config["plays_per_task"] == 4 and config["scaffold_tolerance"] == 8
    assert config["rounds"] == 0 and config["generator_url"] == ""
    with pytest.raises(ValueError, match="scaffold_tolerance"):
        training_recipe(scaffold_tolerance=-1)
    with pytest.raises(ValueError, match="task-generation section"):
        training_recipe(rounds=2)


def test_the_section_selects_a_generator_class_built_once_for_each_scenario(tmp_path: Path) -> None:
    setting = {"module": f"{MODULE}:ScriptedGenerator", "config": {"count": 3}}
    environ = {"TASKS_KEY": "secret"}
    recipe = GeneratedTaskRecipe.from_environment(
        environ,
        config={
            "data": {
                "rounds": 1,
                "generator_url": "http://g:8910",
                "description": "shell tasks",
                "state_dir": str(tmp_path / "state"),
            },
            "model": {"path": "Qwen/Qwen3-8B"},
            "task_generation": setting,
        },
        runtime=InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8000"),
        training_runtime=StubTrainingRuntime(),
    )
    assert isinstance(recipe, GeneratedTaskRecipe)
    assert recipe.task_generation == TaskGenerationConfig(PluginReference(ScriptedGenerator, {"count": 3}), environ)
    ScriptedGenerator.built.clear()
    built = recipe.build_processor(ProcessorContext("one"))
    assert isinstance(built, GeneratedTaskProcessor) and isinstance(built.task_generator, ScriptedGenerator)
    assert ScriptedGenerator.built == [{"config": {"count": 3}, "scenario": "one", "environ": environ}]
    assert built.state_dir == tmp_path / "state" / "one"
    assert recipe.scenario_state_dirs("one") == (tmp_path / "state" / "one",)
    built.close()


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        (f"{MODULE}:ScriptedGenerator.missing", "cannot import task-generation"),
        ("reef.record2dataset.generated_tasks:GeneratedTaskRecipe", "must name a class that inherits TaskGenerator"),
        ("reef.core.tasks:TaskGenerator", "names an abstract class"),
    ],
)
def test_a_section_that_names_no_usable_generator_is_refused(setting: str, message: str) -> None:
    with pytest.raises(RecipeConfigError, match=message):
        GeneratedTaskRecipe.from_environment(
            {},
            config={"data": {}, "model": {"path": "m"}, "task_generation": setting},
            runtime=InferenceProxyRuntime(model_path="m", base_url="http://127.0.0.1:8000"),
            training_runtime=StubTrainingRuntime(),
        )
