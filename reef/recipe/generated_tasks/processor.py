"""The generated-task stages: rounds of generated tasks played here, their training episodes trained here.

The record stage (``RoundRecord2Record``, built in ``recipe.py``) groups the task player's reports: every
episode of the training arm names its task under ``metadata.task``, the episodes of one task form a
group, and a group is complete at ``plays_per_task`` episodes. A training play that reports nothing, such
as a failed trial, never joins its group. So when a task's plays end, its group is complete at the number
of plays that reported, and a group of one is discarded. ``GeneratedTaskBatches`` batches
``tasks_per_step`` complete groups, in arrival order. ``GeneratedTaskTasks`` runs the rounds with the
deployment's :class:`~reef.core.tasks.TaskGenerator`.

One round is one job on a private worker, off the trainer's thread:

1. Ask the generator for tasks, with the round's source sessions when the record stage gave any.
2. Validate each task with the generator.
3. Write each valid task through the generator service and play it: ``plays_per_task`` times on the
   training arm, then each measurement arm the generator asked for.
4. Send the results to the generator.

The first round starts when the pipeline first looks for a batch. The next round starts after
``batches_per_round`` batches were acknowledged since the previous round started. Its episodes train
while it runs, so the generator always writes for the policy that trains now. A round that played no
task is followed at once. A summary of each round goes under ``state_dir/<scenario>``; a restart
reads it to continue with the next round.

Only the training arm, labeled ``plain``, trains. The record stage releases reports of other arms, and
reports without an arm, such as a generator's own reports, without assembling them. A generated task
whose measurement arm is named ``plain`` is not played.
"""

from __future__ import annotations

import json
import logging
import queue
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord
from reef.core.config import config_option
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.core.tasks import GeneratedTask, TaskGenerationRequest, TaskGenerator, TaskPlayResult
from reef.harness.client.tasks import TaskPlay
from reef.pipeline_context import PipelineContext
from reef.record2record import GroupDecision, Record2Record, ReportContext, RoundGrouping, RoundRecord2Record
from reef.record2task import Record2Task, ReportTasks
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2task.generator.client import DuplicateTask, Generator, HttpGenerator, TaskNameConflict
from reef.task2batch import Task2Batch
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

DEFAULT_TASKS_PER_STEP = 4
DEFAULT_PLAYS_PER_TASK = 4
# Qwen3 with thinking off ends the generation prompt with an empty think block of four tokens.
DEFAULT_SCAFFOLD_TOLERANCE = 8
# A realigned drift spans the whole previous response, so the budget must exceed max_tokens plus the scaffold.
REALIGN_THRESHOLD = 32768
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


def recorded_reported_plays(state_dir: Path) -> dict[str, int]:
    """Each played task's count of training episodes that reached Reef, from the round summaries.

    A summary written before the counts were kept names none.
    """
    counts: dict[str, int] = {}
    for round_number in recorded_rounds(state_dir):
        document = json.loads((state_dir / f"{ROUND_PREFIX}{round_number:05d}.json").read_text(encoding="utf-8"))
        for task in document["tasks"]:
            if task.get("reported_plays") is not None:
                counts[task["name"]] = task["reported_plays"]
    return counts


@dataclass(frozen=True)
class RoundJob(SupportsReceipt):
    """One round to run, with its source sessions' records; the receipt names it."""

    round_number: int
    source_records: tuple[AgentRecord, ...] = ()


def session_records(sessions: tuple[ReportContext, ...]) -> tuple[AgentRecord, ...]:
    """The records of the source sessions for the generator: each session's inference records, then its report.

    Sessions that share an inference record send it once.
    """
    records: dict[str, AgentRecord] = {}
    for session in sessions:
        for record in (*session.inferences, session.report):
            records.setdefault(record.agent_record_id, record)
    return tuple(records.values())


@dataclass(frozen=True)
class RoundOutcome(SupportsReceipt):
    """What the worker hands back for one round."""

    round_number: int
    results: tuple[TaskPlayResult, ...]
    error: str


def episode_assembly(scaffold_tolerance: int = DEFAULT_SCAFFOLD_TOLERANCE) -> SampleAssembly:
    """The sample assembly for an agent's episode: many model calls that extend one conversation."""
    return SampleAssembly(
        SampleAssemblySettings(
            accept_multi_turn_policy_samples=True,
            realign_threshold=REALIGN_THRESHOLD,
            scaffold_tolerance=scaffold_tolerance,
        )
    )


def source_tag_of(setting: str) -> tuple[str, str] | None:
    """Read ``source_tag`` as a tag name and value: ``"source=session"`` is the tag ``x-reef-tag-source: session``.

    An empty setting selects no source sessions.
    """
    if not setting:
        return None
    name, separator, value = setting.partition("=")
    if not separator or not name.strip() or not value:
        raise ValueError("source_tag must be name=value, the x-reef-tag-<name> header that marks a source session")
    # The service keeps tag names in lower case.
    return name.strip().lower(), value


def inference_tags(record: AgentRecord) -> Mapping[str, object]:
    """The ``x-reef-tag-*`` pairs that the service keeps under an inference record's ``metadata.tags``."""
    metadata = record.payload.get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, Mapping) else None
    return tags if isinstance(tags, Mapping) else {}


class TaskGroupGrouping(RoundGrouping):
    """The training arm's episodes of one task form one group, ready at ``plays_per_task`` episodes.

    A group is ready at fewer episodes when fewer of the task's plays reported (``reported_plays``), and a
    group of one is discarded. Measurement arms and the generator's own reports share the scenario; only
    the training arm trains. A source session is a report whose inference records all carry ``source_tag``.
    """

    def __init__(self, plays_per_task: int, source_tag: tuple[str, str] | None = None) -> None:
        self.plays_per_task = plays_per_task
        self.source_tag = source_tag
        #: How many training episodes of each played task reached Reef, by task name, once its plays ended.
        self.reported_plays: dict[Hashable, int] = {}

    def is_training_report(self, report: AgentRecord) -> bool:
        return reported_arm(report) == TRAINING_ARM

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return reported_task_name(context.report), None

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) < self.reported_plays.get(key, self.plays_per_task):
            return GroupDecision.INCOMPLETE
        # A group of one has no relative reward.
        return GroupDecision.READY if len(members) >= 2 else GroupDecision.DISCARD

    def is_source_report(self, context: ReportContext) -> bool:
        if self.source_tag is None or not context.inferences:
            return False
        name, value = self.source_tag
        return all(inference_tags(inference).get(name) == value for inference in context.inferences)


@dataclass(frozen=True)
class TaskGroupSettings(PluginSettings):
    """The config of ``GeneratedTaskRecord2Record``."""

    plays_per_task: int = config_option(
        DEFAULT_PLAYS_PER_TASK,
        help="The training plays of each task; a group is complete at this count, or at the plays that reported.",
    )
    batches_per_round: int = config_option(
        DEFAULT_BATCHES_PER_ROUND, help="The trained batches after which the next round starts."
    )
    source_records_per_round: int = config_option(0, help="The source sessions that each round takes, at most.")
    source_tag: str = config_option("", help="The name=value tag that marks a source session.")

    def __post_init__(self) -> None:
        if self.plays_per_task < 2:
            raise ValueError("plays_per_task must be at least two: a group of one has no relative reward")
        if self.batches_per_round < 1:
            raise ValueError("batches_per_round must be positive")
        if self.source_records_per_round < 0:
            raise ValueError("source_records_per_round must be non-negative")
        if self.source_records_per_round > 0 and not self.source_tag:
            raise ValueError("source_records_per_round needs a source_tag that marks the source sessions")
        source_tag_of(self.source_tag)


@dataclass(frozen=True)
class RoundSettings(PluginSettings):
    """The config of ``GeneratedTaskTasks``."""

    rounds: int = config_option(0, help="The rounds of task generation; 0 generates nothing.")
    generator: Mapping[str, Any] | None = config_option(
        None, help="The TaskGenerator class (module) and its config. Required with rounds."
    )
    description: str = config_option("", help="What the tasks are about; every generation request carries it.")
    generator_url: str = config_option("", help="The generator service, ${endpoints.generator}.")
    state_dir: str = config_option("", help="Where each scenario keeps its round summaries.")
    served_model: str = config_option("", help="The model the task player asks for; the deployment's by default.")
    scaffold_tolerance: int = config_option(
        DEFAULT_SCAFFOLD_TOLERANCE, help="Masked tokens the sample assembly may realign before a response."
    )

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must be non-negative")
        if self.scaffold_tolerance < 0:
            raise ValueError("scaffold_tolerance must be non-negative")
        if self.rounds > 0 and (not self.description.strip() or not self.state_dir):
            raise ValueError("rounds need a description (what the tasks are about) and a state_dir")
        if self.generator is not None:
            TaskGenerator.reference(self.generator, setting="pipeline.record2task.config.generator")


@dataclass(frozen=True)
class TaskBatchSettings(PluginSettings):
    """The config of ``GeneratedTaskBatches``."""

    tasks_per_step: int = config_option(DEFAULT_TASKS_PER_STEP, help="The complete task groups in one batch.")

    def __post_init__(self) -> None:
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")


class GeneratedTaskRecord2Record(RoundRecord2Record):
    """The generated-task record stage: one group per task, and a round request every ``batches_per_round`` batches.

    Config: ``plays_per_task`` (the training plays of each task, at least 2; a
    group is complete at this count, or at the plays that reported), ``batches_per_round``,
    ``source_records_per_round`` (the source sessions that each request takes,
    at most), and ``source_tag`` (``name=value``; required when
    ``source_records_per_round`` is above 0). A terminal report owns its sources.
    """

    def __init__(self, context: PipelineContext, settings: TaskGroupSettings) -> None:
        self.task_grouping = TaskGroupGrouping(settings.plays_per_task, source_tag_of(settings.source_tag))
        super().__init__(
            context,
            self.task_grouping,
            batches_per_round=settings.batches_per_round,
            sources_per_round=settings.source_records_per_round,
            exclusive_sources=True,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(context, TaskGroupSettings.parse(config, plugin_class=cls))

    def plays_reported(self, task_name: str, count: int) -> None:
        """A task's training plays ended with ``count`` episodes reported, so its group is complete at that count."""
        self.task_grouping.reported_plays[task_name] = count
        # The count can arrive before the task's first report.
        if task_name in self.groups:
            self.refresh_group(task_name)


class GeneratedTaskTasks(ReportTasks, BackgroundJobRunner):
    """The record-to-task stage of generated tasks: played episodes as tasks, and the rounds.

    Each time the pipeline looks for a batch, the stage starts a round when
    the record stage says one is due, on the worker. The round's source
    sessions go to the generator as the request's ``source_records``. The
    round generates tasks, validates them, has the generator service play
    them, and sends the results to the generator; the plays come back as
    records. This stage owns the worker, the round numbers, and the round
    summaries under ``state_dir/<scenario>``; a restart reads them to continue
    with the next round.
    """

    def __init__(
        self,
        context: PipelineContext,
        settings: RoundSettings,
        round_stage: GeneratedTaskRecord2Record,
        *,
        task_generator: TaskGenerator | None = None,
        task_player: Generator | None = None,
        worker: BackgroundJobWorker | None = None,
    ) -> None:
        #: The record stage that requests the rounds; ``poll`` tells it when each one finished.
        self.round_stage = round_stage
        self.assembly = episode_assembly(settings.scaffold_tolerance)
        self.scenario = context.scenario
        # The record stage's groups complete at the plays that each round makes.
        self.plays_per_task = round_stage.task_grouping.plays_per_task
        self.rounds = settings.rounds
        self.description = settings.description
        self.served_model = settings.served_model or None
        # Every scenario keeps its own rounds.
        self.state_dir = Path(settings.state_dir).expanduser() / self.scenario if settings.state_dir else None
        if self.rounds > 0:
            if task_generator is None:
                raise ValueError("rounds need a task generator: set pipeline.record2task.config.generator")
            if task_player is None:
                generator_url = settings.generator_url
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
            self.worker = BackgroundJobWorker(self, thread_name="generated-task-rounds", concurrency=1)
        if self.task_generator is not None:
            logger.info("scenario %r generates tasks with %s", self.scenario, type(self.task_generator).__name__)
        recorded = recorded_rounds(self.state_dir) if self.state_dir is not None else ()
        # The round summaries keep how many plays of each task reported, so replayed reports close the groups.
        if self.state_dir is not None:
            for task_name, count in recorded_reported_plays(self.state_dir).items():
                round_stage.plays_reported(task_name, count)
        # The worker puts each task's count here as its plays end; the trainer's thread takes them in poll.
        self.reported_play_counts: queue.SimpleQueue[tuple[str, int]] = queue.SimpleQueue()
        self.next_round = recorded[-1] + 1 if recorded else 0
        self.completed_rounds = len(recorded)
        self.round_in_flight: int | None = None
        self.tasks_generated = 0
        self.tasks_played = 0
        self.last_error = ""

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the rounds of one scenario, with the ``TaskGenerator`` that ``generator`` names."""
        settings = RoundSettings.parse(config, plugin_class=cls)
        if not isinstance(record2record, GeneratedTaskRecord2Record):
            raise PluginConfigError(
                f"{cls.__name__} needs GeneratedTaskRecord2Record as record2record, not {type(record2record).__name__}"
            )
        task_generator = None
        if settings.generator is not None:
            generator = TaskGenerator.reference(settings.generator, setting="pipeline.record2task.config.generator")
            task_generator = TaskGenerator.referenced_class(generator).from_config(
                generator.config, scenario=context.scenario, environ=environ
            )
        return cls(context, settings, record2record, task_generator=task_generator)

    @classmethod
    def scenario_state_dirs(cls, config: Mapping[str, Any], scenario: str) -> tuple[Path, ...]:
        state_dir = RoundSettings.parse(config, plugin_class=cls).state_dir
        return (Path(state_dir).expanduser() / scenario,) if state_dir else ()

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        task_name = reported_task_name(context.report)
        if task_name is None:
            raise ValueError("a generated task's report requires metadata.task.name, as the task player sends it")
        sample = self.assembly.build(context, context.require_score())
        return replace(sample, group_id=task_name)

    def poll(self) -> None:
        """Absorb the finished plays and rounds, then start the next round when the record stage says one is due."""
        while not self.reported_play_counts.empty():
            self.round_stage.plays_reported(*self.reported_play_counts.get_nowait())
        if self.worker is not None:
            self.absorb_finished_rounds()
        if self.round_stage.round_due():
            self.start_round()

    def start_round(self) -> None:
        """Start the next round on the worker with the waiting source sessions, or stop the rounds when none can run."""
        if self.worker is None or self.next_round >= self.rounds:
            self.round_stage.stop_rounds()
            return
        sources = self.round_stage.start_round()
        job = RoundJob(
            receipt=f"round-{self.next_round:05d}",
            round_number=self.next_round,
            source_records=session_records(sources),
        )
        if not self.worker.submit(job):
            self.last_error = "the round worker is closed or broken; no further round runs"
            self.round_stage.stop_rounds()
            return
        self.round_in_flight = self.next_round
        self.next_round += 1

    def absorb_finished_rounds(self) -> None:
        """Take the rounds that the worker finished, and tell the record stage whether each one played no task."""
        if self.worker is None:
            raise RuntimeError("no round worker to poll")
        for outcome in self.worker.poll():
            self.round_in_flight = None
            if isinstance(outcome, Failed):
                self.last_error = f"{outcome.receipt} did not finish; see the service log"
                logger.error("generated-task %s failed on scenario %r", outcome.receipt, self.scenario)
                self.round_stage.round_finished(played_none=False)
                continue
            if not isinstance(outcome, RoundOutcome):
                raise TypeError("the round worker must hand back a RoundOutcome")
            played = sum(1 for result in outcome.results if result.is_played)
            self.completed_rounds += 1
            self.tasks_generated += len(outcome.results)
            self.tasks_played += played
            self.last_error = outcome.error
            logger.info(
                "generated-task round %d on scenario %r: %d tasks played of %d",
                outcome.round_number,
                self.scenario,
                played,
                len(outcome.results),
            )
            self.round_stage.round_finished(played_none=played == 0)

    def background_jobs_running(self) -> bool:
        return self.round_in_flight is not None

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {"rounds_completed": self.completed_rounds, "round_in_flight": 0 if self.round_in_flight is None else 1}

    def status(self) -> Mapping[str, Any]:
        return {
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

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Run one job on the worker: a task-generation round."""
        if not isinstance(job, RoundJob):
            raise TypeError(f"the round worker takes RoundJob, not {type(job).__name__}")
        return await self.run_round(job)

    async def run_round(self, job: RoundJob) -> RoundOutcome:
        """The worker's job: generate, validate, play, and send the results back; then the round's summary."""
        if self.task_generator is None or self.state_dir is None:
            raise RuntimeError("a round needs a task generator and a state_dir")
        results: list[TaskPlayResult] = []
        reported_plays: dict[str, int] = {}
        error = ""
        # The generator is a deployment's plugin: whatever it raises stops the round, and what ran is kept.
        try:
            generated = await self.task_generator.generate(TaskGenerationRequest(job.source_records, self.description))
            for item in generated:
                if any(arm.name == TRAINING_ARM for arm in item.measurement_arms):
                    refusal = f"the measurement arm {TRAINING_ARM!r} has the name of the training arm"
                    results.append(TaskPlayResult(item.task, error=refusal))
                    continue
                validation = await self.task_generator.validate(item.task)
                if validation.is_valid:
                    result, reported = await self.played(item, job.round_number)
                    results.append(result)
                    if result.is_played:
                        reported_plays[item.task.name] = reported
                        # The task's group can close now, before the round ends.
                        self.reported_play_counts.put((item.task.name, reported))
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
        self.write_round_summary(job.round_number, tuple(results), error, reported_plays, job.source_records)
        return RoundOutcome(receipt=job.receipt, round_number=job.round_number, results=tuple(results), error=error)

    async def played(self, item: GeneratedTask, round_number: int) -> tuple[TaskPlayResult, int]:
        """Write one valid task and play it: the training arm, then each measurement arm.

        Returns the task's result and how many of its training episodes reached Reef, 0 for a task that was
        not played. A task the agent could not play at all (every training episode ended before the agent
        ran) is removed and comes back with the first episode's error, and its measurement arms are not played.
        """
        if self.task_player is None:
            raise RuntimeError("playing a task needs the generator service")
        task = item.task
        try:
            written = await self.task_player.write_task(task)
        except (DuplicateTask, TaskNameConflict) as exc:
            return TaskPlayResult(task, error=f"refused: {exc}"), 0
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
            error = f"the agent could not play the task: {training[0].error[:ERROR_EXCERPT_CHARS]}"
            return TaskPlayResult(task, error=error), 0
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
        reported = sum(play.is_reported for play in training)
        return TaskPlayResult(task, rewards_of(training), measurement_rewards), reported

    def write_round_summary(
        self,
        round_number: int,
        results: tuple[TaskPlayResult, ...],
        error: str,
        reported_plays: Mapping[str, int],
        source_records: tuple[AgentRecord, ...] = (),
    ) -> Path:
        """Keep each task's result and reported plays, the round's error, and its source records' IDs.

        A restart counts these files and reads the reported plays back.
        """
        if self.state_dir is None:
            raise RuntimeError("a round summary needs a state_dir")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "round": round_number,
            "error": error,
            "source_record_ids": [record.agent_record_id for record in source_records],
            "tasks": [
                {
                    "name": result.task.name,
                    "error": result.error,
                    "training_rewards": list(result.training_rewards),
                    "measurement_rewards": {arm: list(rewards) for arm, rewards in result.measurement_rewards.items()},
                    "reported_plays": reported_plays.get(result.task.name),
                }
                for result in results
            ],
        }
        path = self.state_dir / f"{ROUND_PREFIX}{round_number:05d}.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path


class GeneratedTaskBatches(Task2Batch):
    """Batch the training arm's complete task groups, ``tasks_per_step`` groups to a batch."""

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, TaskBatchSettings.parse(config, plugin_class=cls))

    def __init__(self, context: PipelineContext, settings: TaskBatchSettings) -> None:
        # One record set of the batch is one complete task group.
        super().__init__(context, settings.tasks_per_step)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:generated-tasks:{batch_number}", tasks)
