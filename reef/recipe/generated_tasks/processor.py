"""The generated-task stages: tasks generated and saved here, played from the task pool, and trained here.

The record stage (``GeneratedTaskRecord2Record``, a ``RoundRecord2Record``) groups the task player's reports: every
episode of the training arm names its task under ``metadata.task``, the episodes of one task form a
group, and a group is complete at ``plays_per_task`` episodes. A training play that reports nothing, such
as a failed trial, never joins its group. So when a task's plays end, its group is complete at the number
of plays that reported, and a group of one is discarded. ``GeneratedTaskBatches`` batches
``tasks_per_step`` complete groups, in arrival order.

When a round is due, the record stage forms a round set; when saved tasks wait in its task pool and no
play runs, it forms a play set. ``GeneratedTaskTasks`` runs each set as one job on a private worker, off
the trainer's thread, and returns pending until the job finishes. A generation job:

1. Asks the generator for tasks, with the round's source sessions when the record stage gave any.
2. Saves the candidates and validates each task with the generator.
3. Publishes valid tasks with their fixed train/eval split. Eval tasks are never played.

A play job stages each train task through the generator service, plays its training and measurement arms,
and saves the result. When every train task of a generation has a result, the next job sends the
generation's results to the generator.

The first round starts when the pipeline first looks for a batch. The next round starts after
``batches_per_round`` batches were acknowledged since the previous round started, and only while the task
pool holds fewer than ``refill_tasks`` tasks. A round that made no train task and a play that played no task
are followed at once; a deferred round waits ``deferred_retry_seconds`` or for new sessions. A restart
continues from the task store.

Only the training arm, labeled ``plain``, trains. The record stage releases reports of other arms, and
reports without an arm, such as a generator's own reports, without assembling them. A generated task
whose measurement arm is named ``plain`` is not played.
"""

from __future__ import annotations

import asyncio
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
from reef.core.tasks.metadata import TRAINING_ARM, reported_arm, reported_task_name
from reef.core.tasks.split import TaskSplitConflict
from reef.harness.client.tasks import TaskPlay
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks.store import FileTaskStore, PlayedResult, StoredTask, TaskConflict
from reef.record2record import (
    GroupDecision,
    PlaySet,
    Record2Record,
    RecordSet,
    ReportContext,
    RoundGrouping,
    RoundRecord2Record,
    RoundSet,
    TaskStatus,
)
from reef.record2task import Record2Task, ReportTasks, TaskResult
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2task.generator.client import DuplicateTask, Generator, HttpGenerator, TaskNameConflict
from reef.task2batch import Task2Batch
from reef.train.types import TaskItem, TrainDataItem, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

DEFAULT_TASKS_PER_STEP = 4
DEFAULT_PLAYS_PER_TASK = 4
# Qwen3 with thinking off ends the generation prompt with an empty think block of four tokens.
DEFAULT_SCAFFOLD_TOLERANCE = 8
# A realigned drift spans the whole previous response, so the budget must exceed max_tokens plus the scaffold.
REALIGN_THRESHOLD = 32768
DEFAULT_BATCHES_PER_ROUND = 1
ERROR_EXCERPT_CHARS = 300
STORE_DIRECTORY = "store"
DEFAULT_EVAL_FRACTION = 0.0
DEFAULT_SPLIT_SEED = 0
DEFAULT_GENERATION_ATTEMPTS = 3
DEFAULT_GENERATION_DEADLINE_SECONDS = 1800.0
DELIVERY_RECEIPT = "delivery"


def rewards_of(plays: tuple[TaskPlay, ...]) -> tuple[float, ...]:
    """The rewards of the episodes that ran. An episode that ran without a reward counts as 0."""
    return tuple(
        play.reward if play.reward is not None else 0.0 for play in plays if play.reward is not None or not play.error
    )


@dataclass(frozen=True)
class GenerationJob(SupportsReceipt):
    """One generation to run, with its source sessions' records; the receipt names it."""

    generation: int
    source_records: tuple[AgentRecord, ...] = ()


@dataclass(frozen=True)
class GenerationOutcome(SupportsReceipt):
    """What the worker hands back for one generation: its train tasks, or a deferral after its attempts ran out."""

    generation: int
    tasks: tuple[TaskItem, ...]
    is_deferred: bool
    error: str


@dataclass(frozen=True)
class DeliveryJob(SupportsReceipt):
    """Send the generator the results that it has not received; a restart submits it."""


@dataclass(frozen=True)
class DeliveryOutcome(SupportsReceipt):
    """The error of a delivery that failed, empty when every delivery went through."""

    error: str


@dataclass(frozen=True)
class PlayJob(SupportsReceipt):
    """Saved train tasks to play, by name; the receipt names the play."""

    task_names: tuple[str, ...]


@dataclass(frozen=True)
class PlayOutcome(SupportsReceipt):
    """What the worker hands back for one play: how many of its tasks played."""

    played: int
    error: str


def session_records(sessions: tuple[ReportContext, ...]) -> tuple[AgentRecord, ...]:
    """The records of the source sessions for the generator: each session's inference records, then its report.

    Sessions that share an inference record send it once.
    """
    records: dict[str, AgentRecord] = {}
    for session in sessions:
        for record in (*session.inferences, session.report):
            records.setdefault(record.agent_record_id, record)
    return tuple(records.values())


def task_item(saved: StoredTask) -> TaskItem:
    """The task that the task pool holds for a saved train task, with its generation, split and arms."""
    task = saved.generated.task
    return TaskItem(
        saved.path,
        source_agent_record_ids=task.source_agent_record_ids,
        metadata={
            "task": {
                "name": task.name,
                "digest": task.digest,
                "round": saved.generation,
                "split": saved.split,
                "arms": [arm.name for arm in saved.generated.measurement_arms],
            }
        },
    )


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
    refill_tasks: int = config_option(
        1, help="The next round starts only while the task pool holds fewer saved tasks than this."
    )
    max_rounds_per_run: int = config_option(0, help="The rounds that one run of source sessions takes; 0 is no limit.")
    max_run_seconds: float = config_option(0.0, help="The seconds after which a run stops; 0 is no limit.")
    deferred_retry_seconds: float = config_option(
        600.0, help="The seconds after a deferred round before the next round is due, unless new sessions wait."
    )

    def __post_init__(self) -> None:
        if self.refill_tasks < 1:
            raise ValueError("refill_tasks must be positive")
        if self.max_rounds_per_run < 0 or self.max_run_seconds < 0 or self.deferred_retry_seconds < 0:
            raise ValueError("max_rounds_per_run, max_run_seconds and deferred_retry_seconds must be non-negative")
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

    rounds: int = config_option(0, help="The generations to finish; 0 generates nothing.")
    generator: Mapping[str, Any] | None = config_option(
        None, help="The TaskGenerator class (module) and its config. Required with rounds."
    )
    description: str = config_option("", help="What the tasks are about; every generation request carries it.")
    generator_url: str = config_option("", help="The generator service, ${endpoints.generator}.")
    state_dir: str = config_option("", help="Where each scenario keeps its task store.")
    served_model: str = config_option("", help="The model the task player asks for; the deployment's by default.")
    eval_fraction: float = config_option(
        DEFAULT_EVAL_FRACTION, help="The fraction of new source components held out before rollout."
    )
    split_seed: int = config_option(DEFAULT_SPLIT_SEED, help="The fixed seed for the task store's train/eval split.")
    scaffold_tolerance: int = config_option(
        DEFAULT_SCAFFOLD_TOLERANCE, help="Masked tokens the sample assembly may realign before a response."
    )
    generation_attempts: int = config_option(
        DEFAULT_GENERATION_ATTEMPTS, help="The attempts of a generation whose generator or checks fail to run."
    )
    generation_deadline_seconds: float = config_option(
        DEFAULT_GENERATION_DEADLINE_SECONDS, help="The seconds that all attempts of one generation may take."
    )

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must be non-negative")
        if self.generation_attempts < 1 or self.generation_deadline_seconds <= 0:
            raise ValueError("generation_attempts and generation_deadline_seconds must be positive")
        if self.scaffold_tolerance < 0:
            raise ValueError("scaffold_tolerance must be non-negative")
        if isinstance(self.eval_fraction, bool) or not 0 <= self.eval_fraction <= 1:
            raise ValueError("eval_fraction must be between 0 and 1")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise ValueError("split_seed must be an integer")
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
            refill_tasks=settings.refill_tasks,
            max_rounds_per_run=settings.max_rounds_per_run,
            max_run_seconds=settings.max_run_seconds,
            deferred_retry_seconds=settings.deferred_retry_seconds,
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
    """The record-to-task stage of generated tasks: played episodes as tasks, and the generation and play jobs.

    The record stage forms a ``RoundSet`` when a round is due and a
    ``PlaySet`` when saved tasks wait to play. ``make_tasks`` submits either
    one to the worker and returns ``TaskResult.pending()``.

    A generation job generates tasks from the round's description and source
    sessions, validates them and saves them in the task store; it plays none.
    A generator or check that fails to run is retried up to
    ``generation_attempts`` times within ``generation_deadline_seconds``. After
    that the round is deferred, and its sessions wait for the next run. The
    round comes back ``READY`` with the saved train tasks, or ``SKIPPED`` when
    it made none.

    A play job writes each task through the generator service, plays it, and
    saves its result. The plays come back as records. When every train task of
    a generation has a result, the next job sends the generation's results to
    the generator. A restart puts the unplayed train tasks back in the record
    stage's task pool and sends the results that the generator did not receive.
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
        #: The record stage that forms the round and play sets and closes the task groups.
        self.round_stage = round_stage
        self.assembly = episode_assembly(settings.scaffold_tolerance)
        self.scenario = context.scenario
        # The record stage's groups complete at the plays that each task makes.
        self.plays_per_task = round_stage.task_grouping.plays_per_task
        self.rounds = settings.rounds
        self.description = settings.description
        self.served_model = settings.served_model or None
        self.generation_attempts = settings.generation_attempts
        self.generation_deadline_seconds = settings.generation_deadline_seconds
        # Every scenario keeps its own task store.
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
        self.task_store = (
            FileTaskStore(
                root=self.state_dir / STORE_DIRECTORY, eval_fraction=settings.eval_fraction, seed=settings.split_seed
            )
            if self.state_dir is not None and self.rounds > 0
            else None
        )
        # The worker puts each task's count here as its plays end; the trainer's thread takes them in take_finished.
        self.reported_play_counts: queue.SimpleQueue[tuple[str, int]] = queue.SimpleQueue()
        round_stage.resume_rounds(
            completed_rounds=self.task_store.complete_generation_count() if self.task_store is not None else 0,
            round_limit=self.rounds,
        )
        if self.task_store is not None:
            for saved in self.task_store.available_tasks():
                round_stage.add_tasks((task_item(saved),), round_number=saved.generation)
            # The saved results close the groups of replayed reports.
            for state in self.task_store.generations.values():
                for result in state.results:
                    round_stage.plays_reported(result.task_name, result.reported_plays)
        #: The set ID of each submitted job, by the job's receipt.
        self.job_set_ids: dict[str, Hashable] = {}
        self.generation_in_flight: int | None = None
        self.last_error = ""
        # Results that a stop kept from the generator go out now, even when no other job is due.
        if self.task_store is not None and self.worker is not None and self.task_store.deliverable_generations():
            self.worker.submit(DeliveryJob(receipt=DELIVERY_RECEIPT))

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

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        """Submit a round or play set to the worker and return pending; make a sample from any other set."""
        if isinstance(record_set, PlaySet):
            job: SupportsReceipt = PlayJob(
                receipt=f"play-{record_set.play_number:05d}",
                task_names=tuple(task.task_path.name for task in record_set.tasks),
            )
        elif isinstance(record_set, RoundSet):
            if self.task_store is None:
                raise RuntimeError("a round needs a task store: set pipeline.record2task.config.state_dir")
            records = session_records(record_set.sources)
            completed = self.task_store.completed_source_record_ids()
            remaining = tuple(record for record in records if record.agent_record_id not in completed)
            # A restart replays sessions that a complete generation already used; their tasks are saved.
            if records and not remaining:
                return TaskResult(TaskStatus.INGESTED)
            existing = self.task_store.find_generation(
                source_record_ids=tuple(record.agent_record_id for record in remaining)
            )
            generation = self.task_store.next_generation() if existing is None else existing.generation
            job = GenerationJob(
                receipt=f"generation-{generation:05d}", generation=generation, source_records=remaining
            )
            self.generation_in_flight = generation
        else:
            return super().make_tasks(record_set)
        if self.worker is None or not self.worker.submit(job):
            raise RuntimeError(f"the round worker is closed or broken; {job.receipt} cannot run")
        self.job_set_ids[job.receipt] = record_set.set_id
        return TaskResult.pending()

    def take_finished(self) -> tuple[tuple[Hashable, TaskResult], ...]:
        """Close the task groups whose plays ended, and hand back each finished generation and play."""
        while not self.reported_play_counts.empty():
            self.round_stage.plays_reported(*self.reported_play_counts.get_nowait())
        if self.worker is None:
            return ()
        finished: list[tuple[Hashable, TaskResult]] = []
        for outcome in self.worker.poll():
            if outcome.receipt == DELIVERY_RECEIPT:
                self.last_error = (
                    outcome.error if isinstance(outcome, DeliveryOutcome) else "the delivery did not finish"
                )
                continue
            set_id = self.job_set_ids.pop(outcome.receipt)
            if isinstance(outcome, Failed):
                self.last_error = f"{outcome.receipt} did not finish; see the service log"
                logger.error("generated-task %s failed on scenario %r", outcome.receipt, self.scenario)
                if outcome.receipt.startswith("generation-"):
                    self.generation_in_flight = None
                    finished.append((set_id, TaskResult.deferred()))
                else:
                    finished.append((set_id, TaskResult.skipped()))
                continue
            if isinstance(outcome, GenerationOutcome):
                self.generation_in_flight = None
                self.last_error = outcome.error
                logger.info(
                    "generated-task generation %d on scenario %r: %d train tasks%s",
                    outcome.generation,
                    self.scenario,
                    len(outcome.tasks),
                    ", deferred" if outcome.is_deferred else "",
                )
                if outcome.is_deferred:
                    finished.append((set_id, TaskResult.deferred()))
                else:
                    finished.append(
                        (set_id, TaskResult.ready(outcome.tasks) if outcome.tasks else TaskResult.skipped())
                    )
                continue
            if not isinstance(outcome, PlayOutcome):
                raise TypeError("the round worker must hand back a GenerationOutcome or a PlayOutcome")
            self.last_error = outcome.error
            finished.append((set_id, TaskResult.ready(()) if outcome.played else TaskResult.skipped()))
        return tuple(finished)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
        if self.task_store is not None:
            self.task_store.close()

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {"generation_in_flight": 0 if self.generation_in_flight is None else 1}

    def status(self) -> Mapping[str, Any]:
        completed = self.task_store.complete_generation_count() if self.task_store is not None else 0
        if self.generation_in_flight is not None:
            next_generation = self.generation_in_flight + 1
        else:
            next_generation = self.task_store.next_generation() if self.task_store is not None else 0
        return {
            "task_store": self.task_store.counts() if self.task_store is not None else {},
            "task_generation": {
                "generator": None if self.task_generator is None else type(self.task_generator).__name__,
                "in_flight": self.generation_in_flight,
                "next": next_generation,
                "completed": completed,
                "of": self.rounds,
                "last_error": self.last_error,
            },
        }

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Run one job on the worker, after sending the generator any results that it has not received."""
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("a job needs a task generator and a state_dir")
        delivery_error = await self.deliver_results()
        if isinstance(job, DeliveryJob):
            return DeliveryOutcome(receipt=job.receipt, error=delivery_error)
        if isinstance(job, GenerationJob):
            generation = await self.run_generation(job)
            # A generation without train tasks has its complete results now.
            delivery_error = await self.deliver_results() or delivery_error
            return replace(generation, error=generation.error or delivery_error)
        if isinstance(job, PlayJob):
            play = await self.run_play(job)
            delivery_error = await self.deliver_results() or delivery_error
            return replace(play, error=play.error or delivery_error)
        raise TypeError(f"the round worker takes GenerationJob or PlayJob, not {type(job).__name__}")

    async def run_generation(self, job: GenerationJob) -> GenerationOutcome:
        """Generate, validate and save one generation; retry a failure to run within the attempts and deadline."""
        if self.task_store is None:
            raise RuntimeError("a generation needs a state_dir")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.generation_deadline_seconds
        error = ""
        for attempt in range(1, self.generation_attempts + 1):
            remaining_seconds = deadline - loop.time()
            if remaining_seconds <= 0:
                break
            # The generator is a deployment's plugin: whatever it raises is a failure to run.
            try:
                await asyncio.wait_for(self.generated(job), remaining_seconds)
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                logger.exception(
                    "generated-task generation %d attempt %d failed on scenario %r",
                    job.generation,
                    attempt,
                    self.scenario,
                )
                continue
            state = self.task_store.generation(generation=job.generation)
            saved = {saved.generated.task.name: saved for saved in self.task_store.tasks(split="train")}
            tasks = tuple(task_item(saved[name]) for name in state.train_task_names)
            return GenerationOutcome(
                receipt=job.receipt, generation=job.generation, tasks=tasks, is_deferred=False, error=""
            )
        return GenerationOutcome(
            receipt=job.receipt,
            generation=job.generation,
            tasks=(),
            is_deferred=True,
            error=error or "the generation deadline passed",
        )

    async def generated(self, job: GenerationJob) -> None:
        """One attempt: continue from the saved candidates and validation outcomes, then complete the generation."""
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("a generation needs a task generator and a state_dir")
        generation = self.task_store.begin_generation(
            generation=job.generation,
            source_record_ids=tuple(record.agent_record_id for record in job.source_records),
        )
        if generation.candidates is None:
            generated = await self.task_generator.generate(
                TaskGenerationRequest(job.source_records, self.description, generation=job.generation)
            )
            generation = self.task_store.save_candidates(generation=job.generation, candidates=generated)
        for index in generation.pending_indices:
            item = generation.candidate(index=index)
            if any(arm.name == TRAINING_ARM for arm in item.measurement_arms):
                refusal = f"the measurement arm {TRAINING_ARM!r} has the name of the training arm"
                self.task_store.reject_task(generation=job.generation, index=index, reason=refusal)
                continue
            validation = await self.task_generator.validate(item.task)
            if validation.is_valid:
                try:
                    self.task_store.publish_task(generation=job.generation, index=index)
                except (TaskConflict, TaskSplitConflict) as conflict:
                    self.task_store.reject_task(generation=job.generation, index=index, reason=str(conflict))
            else:
                self.task_store.reject_task(
                    generation=job.generation, index=index, reason="; ".join(validation.errors)
                )
        self.task_store.complete_generation(generation=job.generation)

    async def run_play(self, job: PlayJob) -> PlayOutcome:
        """Play each task and save its result; a task that cannot run has an infrastructure failure as its result."""
        if self.task_store is None:
            raise RuntimeError("a play needs a state_dir")
        saved_by_name = {saved.generated.task.name: saved for saved in self.task_store.tasks(split="train")}
        played = 0
        for name in job.task_names:
            saved = saved_by_name[name]
            try:
                result, reported = await self.played(saved.generated, saved.generation)
            except Exception as exc:
                logger.exception("generated-task %s could not play on scenario %r", name, self.scenario)
                result = TaskPlayResult(saved.generated.task, error=str(exc) or type(exc).__name__)
                result = replace(result, infrastructure_failure=True)
                reported = 0
            self.task_store.record_result(
                generation=saved.generation,
                result=PlayedResult(
                    task_name=name,
                    training_rewards=result.training_rewards,
                    measurement_rewards=dict(result.measurement_rewards),
                    error=result.error,
                    infrastructure_failure=result.infrastructure_failure,
                    reported_plays=reported,
                ),
            )
            if result.is_played:
                played += 1
                # The task's group can close now, before the play ends.
                self.reported_play_counts.put((name, reported))
        return PlayOutcome(receipt=job.receipt, played=played, error="")

    async def played(self, item: GeneratedTask, generation: int) -> tuple[TaskPlayResult, int]:
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
        except DuplicateTask:
            # A play that a stop interrupted left the task staged; stage it again.
            await self.task_player.delete_task(task.name)
            written = await self.task_player.write_task(task)
        except TaskNameConflict as exc:
            return TaskPlayResult(task, error=f"refused: {exc}"), 0
        tags = {"round": str(generation)}
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
            return TaskPlayResult(task, error=error, infrastructure_failure=True), 0
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

    async def deliver_results(self) -> str:
        """Send each generation whose train tasks all have results to the generator, oldest first.

        Returns the error of the last delivery that failed; a failed generation waits for the next job, and later
        generations still go out.
        """
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("delivering results needs a task generator and a state_dir")
        error = ""
        for state in self.task_store.deliverable_generations():
            try:
                await self.task_generator.record_results(state.task_results(), generation=state.generation)
            except Exception as exc:
                logger.exception("the task generator did not take the results of generation %d", state.generation)
                error = str(exc) or type(exc).__name__
                continue
            self.task_store.mark_delivered(generation=state.generation)
        return error


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
