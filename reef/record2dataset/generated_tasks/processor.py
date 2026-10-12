"""The generated-task processor: rounds of generated tasks played here, their training episodes trained here.

Two contracts meet in this class. As a reported feedback processor it takes the task player's reports:
every episode of the training arm names its task under ``metadata.task``, the episodes of one task form
a group, a group is complete at ``plays_per_task`` episodes, and a batch holds ``tasks_per_step``
complete groups, in arrival order. As the driver of task generation it runs rounds with the
deployment's :class:`~reef.core.tasks.TaskGenerator`.

One round is one job on a private worker, off the trainer's thread:

1. Ask the generator for tasks.
2. Validate each task with the generator.
3. Write each valid task through the generator service and play it: ``plays_per_task`` times on the
   training arm, then each measurement arm the generator asked for.
4. Send the results to the generator.

The first round starts when the processor first looks for a batch. The next round starts after
``batches_per_round`` batches were acknowledged since the previous round started. Its episodes train
while it runs, so the generator always writes for the policy that trains now. A round that played no
task is followed at once. A summary of each round goes under ``state_dir/<scenario>``; a restart
reads it to continue with the next round.

Only the training arm, labeled ``plain``, trains. The processor releases reports of other arms, and
reports without an arm, such as a generator's own reports, without assembling them. A generated task
whose measurement arm is named ``plain`` is not played.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from reef.core import AgentRecord
from reef.core.tasks import GeneratedTask, TaskGenerationRequest, TaskGenerator, TaskPlayResult
from reef.harness.client.tasks import TaskPlay
from reef.record2dataset.client import DuplicateTask, Generator, HttpGenerator, TaskNameConflict
from reef.train.processors.computed import Failed, JudgingWorker, SupportsReceipt
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

DEFAULT_TASKS_PER_STEP = 4
DEFAULT_PLAYS_PER_TASK = 4
# Qwen3 with thinking off ends the generation prompt with an empty think block of four tokens.
DEFAULT_SCAFFOLD_TOLERANCE = 8
DEFAULT_BATCHES_PER_ROUND = 1
ERROR_EXCERPT_CHARS = 300
ROUND_PREFIX = "round-"
# The arm label of the plays this recipe trains on: played as they are, with no extra instruction files.
TRAINING_ARM = "plain"


def reported_task_name(report: AgentRecord) -> str | None:
    """The task a report names under ``metadata.task.name``, the way the task player sends it."""
    metadata = report.payload.get("metadata")
    task = metadata.get("task") if isinstance(metadata, Mapping) else None
    name = task.get("name") if isinstance(task, Mapping) else None
    return name if isinstance(name, str) and name else None


def reported_arm(report: AgentRecord) -> str | None:
    """The ``arm`` label of an episode, under ``metadata.episode.labels`` where the task player puts the labels."""
    metadata = report.payload.get("metadata")
    episode = metadata.get("episode") if isinstance(metadata, Mapping) else None
    labels = episode.get("labels") if isinstance(episode, Mapping) else None
    arm = labels.get("arm") if isinstance(labels, Mapping) else None
    return arm if isinstance(arm, str) else None


def rewards_of(plays: tuple[TaskPlay, ...]) -> tuple[float, ...]:
    """The rewards of the episodes that ran. An episode that ran without a reward counts as 0."""
    return tuple(
        play.reward if play.reward is not None else 0.0 for play in plays if play.reward is not None or not play.error
    )


def recorded_rounds(state_dir: Path) -> tuple[int, ...]:
    """The rounds whose summaries are under the state directory, in order."""
    if not state_dir.is_dir():
        return ()
    numbers = []
    for path in sorted(state_dir.glob(f"{ROUND_PREFIX}*.json")):
        stem = path.name[len(ROUND_PREFIX) : -len(".json")]
        if stem.isdigit():
            numbers.append(int(stem))
    return tuple(numbers)


@dataclass(frozen=True)
class RoundJob(SupportsReceipt):
    """One round to run; the receipt names it."""

    round_number: int


@dataclass(frozen=True)
class RoundOutcome(SupportsReceipt):
    """What the worker hands back for one round."""

    round_number: int
    results: tuple[TaskPlayResult, ...]
    error: str


class GeneratedTaskProcessor(ReportedFeedbackProcessor):
    """Groups the training arm's episodes by task and runs the rounds of task generation that produce them."""

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(
        self,
        context: ProcessorContext,
        *,
        task_generator: TaskGenerator | None = None,
        task_player: Generator | None = None,
        worker: JudgingWorker | None = None,
    ) -> None:
        config = dict(context.config)
        self.tasks_per_step = int(config.get("tasks_per_step", DEFAULT_TASKS_PER_STEP))
        self.plays_per_task = int(config.get("plays_per_task", DEFAULT_PLAYS_PER_TASK))
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")
        if self.plays_per_task < 2:
            raise ValueError("plays_per_task must be at least two: a group of one has no relative reward")
        # An agent's episode is many model calls that extend one conversation; the sample spans them.
        config.setdefault("accept_multi_turn_policy_samples", True)
        config.setdefault("scaffold_tolerance", DEFAULT_SCAFFOLD_TOLERANCE)
        self.assembly = SampleAssembly.from_config(context.with_config(config))
        # One unit of the batch is one complete task group.
        super().__init__(context.with_config({**config, "batch_size": self.tasks_per_step}))

        self.rounds = int(config.get("rounds", 0))
        self.batches_per_round = int(config.get("batches_per_round", DEFAULT_BATCHES_PER_ROUND))
        self.description = str(config.get("description", "") or "")
        served_model = config.get("served_model")
        self.served_model = str(served_model) if served_model else None
        state_dir = str(config.get("state_dir", "") or "")
        # Every scenario keeps its own rounds.
        self.state_dir = Path(state_dir).expanduser() / self.scenario if state_dir else None
        if self.rounds < 0 or self.batches_per_round < 1:
            raise ValueError("rounds must be non-negative and batches_per_round positive")
        if self.rounds > 0:
            if task_generator is None:
                raise ValueError("rounds need a task generator: add a task-generation section")
            if not self.description.strip() or self.state_dir is None:
                raise ValueError("rounds need a description (what the tasks are about) and a state_dir")
            if task_player is None:
                generator_url = str(config.get("generator_url", "") or "")
                if not generator_url or "${" in generator_url:
                    raise ValueError(
                        "rounds need generator_url: the generator service's address, ${endpoints.generator} "
                        "when the deployment has a generator section"
                    )
                task_player = HttpGenerator(generator_url)
        self.task_generator = task_generator
        self.task_player = task_player
        self.worker = worker
        if self.worker is None and self.rounds > 0:
            self.worker = JudgingWorker(self.run_round, concurrency=1)
        if self.task_generator is not None:
            logger.info("scenario %r generates tasks with %s", self.scenario, type(self.task_generator).__name__)
        recorded = recorded_rounds(self.state_dir) if self.state_dir is not None else ()
        self.next_round = recorded[-1] + 1 if recorded else 0
        self.completed_rounds = len(recorded)
        self.round_in_flight: int | None = None
        self.tasks_generated = 0
        self.tasks_played = 0
        # Batches acknowledged since the last round started; the next one is due at batches_per_round.
        self.batches_since_round = 0
        self.has_landed = False
        # A round without a played task makes no batch; waiting for one would stall the loop until a restart.
        self.landed_empty = False
        # Batches the backend dropped as stale trained nothing, so they do not pace the rounds.
        self.dropped_batches: set[str] = set()
        self.last_error = ""

    # ------------------------------------------------------- the reported half

    def is_training_report(self, report: AgentRecord) -> bool:
        # Measurement arms and the generator's own reports share the scenario; only the training arm trains.
        return reported_arm(report) == TRAINING_ARM

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        task_name = reported_task_name(context.report)
        if task_name is None:
            raise ValueError("a generated task's report requires metadata.task.name, as the task player sends it")
        sample = self.assembly.build(context, context.require_score())
        return replace(sample, group_id=task_name)

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return reported_task_name(context.report), None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        return GroupDecision.READY if len(items) >= self.plays_per_task else GroupDecision.INCOMPLETE

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:generated-tasks:{batch_number}", items)

    # ---------------------------------------------------------- the round half

    def ready(self) -> bool:
        self.catch_up()
        return super().ready()

    def dropped(self, batch_id: str) -> None:
        self.dropped_batches.add(batch_id)

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        consumed = super().acknowledge(batch_id)
        if batch_id in self.dropped_batches:
            self.dropped_batches.discard(batch_id)
        else:
            self.batches_since_round += 1
        return consumed

    def derivation_pending(self) -> bool:
        return self.round_in_flight is not None

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {
            **super().operational_metrics(),
            "rounds_completed": self.completed_rounds,
            "round_in_flight": 0 if self.round_in_flight is None else 1,
        }

    def status(self) -> Mapping[str, Any]:
        return {
            **super().status(),
            "task_generation": {
                "generator": None if self.task_generator is None else type(self.task_generator).__name__,
                "in_flight": self.round_in_flight,
                "next": self.next_round,
                "completed": self.completed_rounds,
                "of": self.rounds,
                "tasks_generated": self.tasks_generated,
                "tasks_played": self.tasks_played,
                "last_error": self.last_error,
            },
        }

    def catch_up(self) -> None:
        """Absorb the round the worker finished, then start the next one when it is due."""
        if self.worker is None:
            return
        for outcome in self.worker.poll():
            self.round_in_flight = None
            self.has_landed = True
            if isinstance(outcome, Failed):
                self.last_error = f"{outcome.receipt} did not finish; see the service log"
                logger.error("generated-task %s failed on scenario %r", outcome.receipt, self.scenario)
                continue
            if not isinstance(outcome, RoundOutcome):
                raise TypeError("the round worker must hand back a RoundOutcome")
            played = sum(1 for result in outcome.results if result.is_played)
            self.completed_rounds += 1
            self.tasks_generated += len(outcome.results)
            self.tasks_played += played
            self.last_error = outcome.error
            self.landed_empty = played == 0
            logger.info(
                "generated-task round %d on scenario %r: %d tasks played of %d",
                outcome.round_number,
                self.scenario,
                played,
                len(outcome.results),
            )
        if self.round_in_flight is not None or self.next_round >= self.rounds:
            return
        is_due = not self.has_landed or self.landed_empty or self.batches_since_round >= self.batches_per_round
        if not is_due:
            return
        job = RoundJob(receipt=f"round-{self.next_round:05d}", round_number=self.next_round)
        if self.worker.submit(job):
            self.round_in_flight = self.next_round
            self.next_round += 1
            # The batches this round's episodes make count toward the next one.
            self.batches_since_round = 0
            self.landed_empty = False
        else:
            self.last_error = "the round worker is closed or broken; no further round runs"

    async def run_round(self, job: RoundJob) -> RoundOutcome:
        """The worker's job: generate, validate, play, and send the results back; then the round's summary."""
        if self.task_generator is None or self.state_dir is None:
            raise RuntimeError("a round needs a task generator and a state_dir")
        results: list[TaskPlayResult] = []
        error = ""
        # The generator is a deployment's plugin: whatever it raises stops the round, and what ran is kept.
        try:
            generated = await self.task_generator.generate(TaskGenerationRequest((), self.description))
            for item in generated:
                if any(arm.name == TRAINING_ARM for arm in item.measurement_arms):
                    refusal = f"the measurement arm {TRAINING_ARM!r} has the name of the training arm"
                    results.append(TaskPlayResult(item.task, error=refusal))
                    continue
                validation = await self.task_generator.validate(item.task)
                if validation.is_valid:
                    results.append(await self.played(item, job.round_number))
                else:
                    results.append(TaskPlayResult(item.task, error="; ".join(validation.errors)))
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            logger.exception("generated-task round %d stopped on scenario %r", job.round_number, self.scenario)
        try:
            await self.task_generator.record_results(tuple(results))
        except Exception as exc:
            error = error or str(exc) or type(exc).__name__
            logger.exception("the task generator did not take the results of round %d", job.round_number)
        self.write_round_summary(job.round_number, tuple(results), error)
        return RoundOutcome(receipt=job.receipt, round_number=job.round_number, results=tuple(results), error=error)

    async def played(self, item: GeneratedTask, round_number: int) -> TaskPlayResult:
        """Write one valid task and play it: the training arm, then each measurement arm.

        A task the agent could not play at all (every training episode ended before the agent ran) is
        removed and comes back with the first episode's error, and its measurement arms are not played.
        """
        if self.task_player is None:
            raise RuntimeError("playing a task needs the generator service")
        task = item.task
        try:
            written = await self.task_player.write_task(task)
        except (DuplicateTask, TaskNameConflict) as exc:
            return TaskPlayResult(task, error=f"refused: {exc}")
        tags = {"round": str(round_number)}
        training = await self.task_player.play(
            written.path,
            scenario=self.scenario,
            arm=TRAINING_ARM,
            plays=self.plays_per_task,
            is_reporting=True,
            extra_instruction_files=(),
            tags=tags,
            model=self.served_model,
        )
        if training and all(play.reward is None and play.error for play in training):
            await self.task_player.delete_task(task.name)
            return TaskPlayResult(
                task, error=f"the agent could not play the task: {training[0].error[:ERROR_EXCERPT_CHARS]}"
            )
        measurement_rewards: dict[str, tuple[float, ...]] = {}
        for arm in item.measurement_arms:
            plays = await self.task_player.play(
                written.path,
                scenario=self.scenario,
                arm=arm.name,
                plays=arm.count,
                is_reporting=True,
                extra_instruction_files=arm.extra_instruction_files,
                tags=tags,
                model=self.served_model,
            )
            measurement_rewards[arm.name] = rewards_of(plays)
        return TaskPlayResult(task, rewards_of(training), measurement_rewards)

    def write_round_summary(self, round_number: int, results: tuple[TaskPlayResult, ...], error: str) -> Path:
        """Keep each task's result and the round's error; a restart counts these files."""
        if self.state_dir is None:
            raise RuntimeError("a round summary needs a state_dir")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "round": round_number,
            "error": error,
            "tasks": [
                {
                    "name": result.task.name,
                    "error": result.error,
                    "training_rewards": list(result.training_rewards),
                    "measurement_rewards": {arm: list(rewards) for arm, rewards in result.measurement_rewards.items()},
                }
                for result in results
            ],
        }
        path = self.state_dir / f"{ROUND_PREFIX}{round_number:05d}.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path
