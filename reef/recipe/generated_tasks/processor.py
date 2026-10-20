"""The generated-task stages: tasks generated and saved here, played from the task pool, and trained here.

The record stage (``GeneratedTaskRecord2Record``) groups the task player's reports and paces the jobs: every
episode of the training arm names its task under ``metadata.task``, the episodes of one task form a
group, and a group is complete at ``plays_per_task`` episodes. A training play that reports nothing, such
as a failed trial, never joins its group. So when a task's plays end, its group is complete at the number
of plays that reported, and a group of one is discarded. ``GeneratedTaskBatches`` batches
``tasks_per_step`` complete groups, in arrival order.

When a round is due, the record stage gives a ``TaskGenerationRound`` to ``GeneratedTaskTasks``. When saved
train tasks wait in its task pool and no play runs, it gives them to the same stage to play. A round and a
play are not record sets and do not train; the episodes that a play reports do. Each runs as one job on a
private worker, off the trainer's thread. A generation job:

1. Asks the generator for tasks, with the round's source sessions when the record stage gave any.
2. Saves the candidates in the scenario's task store, and validates each task with the generator.
3. Publishes each valid task with its train/eval split. Eval tasks are never played.

A play job writes each train task through the generator service, plays it ``plays_per_task`` times on the
training arm, then each measurement arm the generator asked for, and saves the result in the task store.
When every train task of a generation has a result, the next job sends the generation's results to the
generator.

The first round starts when the pipeline first looks for a batch. The next round starts after
``batches_per_round`` batches were acknowledged since the previous round started, and only while the task
pool holds fewer than ``refill_tasks`` tasks. A round that made no train task and a play that played no
task are followed at once; a deferred round waits ``deferred_retry_seconds`` or for new sessions. A restart
continues from the task store.

Only the training arm, labeled ``plain``, trains. The record stage releases reports of other arms, and
reports without an arm, such as a generator's own reports, without assembling them. A generated task
whose measurement arm is named ``plain`` is not played.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Self

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord
from reef.core.config import config_option
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.core.tasks import GeneratedTask, TaskGenerationRequest, TaskGenerator, TaskPlayResult
from reef.core.tasks.metadata import TRAINING_ARM, reported_arm, reported_task_name
from reef.core.tasks.split import TaskSplit, TaskSplitConflict, extend_split_by_source
from reef.harness.client.tasks import TaskPlay
from reef.pipeline_context import PipelineContext
from reef.record2record import (
    GroupDecision,
    Record2Record,
    RecordSetUnit,
    ReportContext,
    ReportGrouping,
    ReportRecord2Record,
)
from reef.record2task import Record2Task, ReportTasks
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2task.generator.client import DuplicateTask, Generator, HttpGenerator, TaskNameConflict
from reef.storage.tasks import GenerationState, PlayedResult, SplitName, SplitSettings, TaskConflict, TaskStore
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
TASK_STORE_NAME = "generated-tasks"
DEFAULT_EVAL_FRACTION = 0.0
DEFAULT_SPLIT_SEED = 0
DEFAULT_GENERATION_ATTEMPTS = 3
DEFAULT_GENERATION_DEADLINE_SECONDS = 1800.0
DEFAULT_DEFERRED_RETRY_SECONDS = 600.0
DELIVERY_RECEIPT = "delivery"


def rewards_of(plays: tuple[TaskPlay, ...]) -> tuple[float, ...]:
    """The rewards of the episodes that ran. An episode that ran without a reward counts as 0."""
    return tuple(
        play.reward if play.reward is not None else 0.0 for play in plays if play.reward is not None or not play.error
    )


def split_source_ids(generated: GeneratedTask) -> tuple[str, ...]:
    """The source record ids that place a task in a split: its sources without the author's own record."""
    designer_record_id = generated.task.metadata.get("designer_record_id")
    return tuple(record_id for record_id in generated.task.source_agent_record_ids if record_id != designer_record_id)


def task_counts(store: TaskStore) -> dict[str, int]:
    """Count the published tasks of each split, the train tasks that played, and the rejected and pending candidates."""
    splits = Counter(task.split for task in store.tasks())
    states = store.generations()
    candidates = [candidate for state in states for candidate in state.candidates or ()]
    return {
        "train": splits["train"],
        "eval": splits["eval"],
        "played": sum(len(state.play_results) for state in states),
        "rejected": sum(bool(candidate.error) for candidate in candidates),
        "pending": sum(candidate.is_pending for candidate in candidates),
    }


def next_generation(store: TaskStore) -> int:
    """Return the number after the highest saved generation, 0 for an empty store."""
    return max((state.generation for state in store.generations()), default=-1) + 1


def unplayed_train_tasks(store: TaskStore) -> tuple[str, ...]:
    """Return the name of each train task of a complete generation without a play result.

    The names are in generation and candidate order.
    """
    return tuple(
        name
        for state in store.generations()
        if state.is_complete
        for name in state.train_task_names
        if name not in {result.task_name for result in state.play_results}
    )


def assigned_split(store: TaskStore, settings: SplitSettings, generated: GeneratedTask) -> SplitName:
    """Return the split of a task: the split of its published name, or else the split of its source records.

    Tasks that share a source record get the same split, and a published task keeps its split. Raises
    ``TaskSplitConflict`` if the sources of the task join tasks of both splits.
    """
    published = {task.generated.task.name: task for task in store.tasks()}
    existing = published.get(generated.task.name)
    if existing is not None:
        return existing.split
    previous = TaskSplit(
        train=tuple(name for name, task in published.items() if task.split == "train"),
        eval=tuple(name for name, task in published.items() if task.split == "eval"),
        seed=settings.seed,
        eval_fraction=settings.eval_fraction,
    )
    assigned = extend_split_by_source(
        previous=previous,
        previous_sources={name: split_source_ids(task.generated) for name, task in published.items()},
        sources={generated.task.name: split_source_ids(generated)},
    )
    return "eval" if generated.task.name in assigned.eval else "train"


@dataclass(frozen=True)
class GenerationJob(SupportsReceipt):
    """One generation for the worker to run.

    Values: ``generation`` is the number of the generation in the task store.
    ``source_records`` are the records of the round's source sessions, without
    the records that a complete generation already used. The receipt is
    ``generation-<generation>``.
    Purpose: ``GeneratedTaskTasks.start_round`` makes it, and the worker runs it
    (``GeneratedTaskTasks.run_generation``).
    """

    generation: int
    source_records: tuple[AgentRecord, ...] = ()


@dataclass(frozen=True)
class GenerationOutcome(SupportsReceipt):
    """How one generation job ended.

    Values: ``train_task_names`` names the train tasks that the generation
    published. A deferred generation (``is_deferred``) did not complete and has
    no train task. ``error`` is the last failure to run of a deferred
    generation, or the error of a delivery that failed; it is empty otherwise.
    Purpose: the worker hands it back, and ``GeneratedTaskTasks.finished_jobs``
    turns it into a ``RoundResult``.
    """

    generation: int
    train_task_names: tuple[str, ...]
    is_deferred: bool
    error: str

    def __post_init__(self) -> None:
        if self.is_deferred and self.train_task_names:
            raise ValueError("a deferred generation has no train task")


@dataclass(frozen=True)
class PlayJob(SupportsReceipt):
    """One play of saved train tasks for the worker to run.

    Values: ``task_names`` names one or more train tasks of complete
    generations that have no play result. The receipt is ``play-<number>``.
    Purpose: ``GeneratedTaskTasks.start_play`` makes it, and the worker runs it
    (``GeneratedTaskTasks.run_play``).
    """

    task_names: tuple[str, ...]


@dataclass(frozen=True)
class PlayOutcome(SupportsReceipt):
    """How one play job ended.

    Values: ``played_count`` is the number of the play's tasks that played, not
    negative. ``error`` is the error of a delivery that failed after the play,
    or empty.
    Purpose: the worker hands it back, and ``GeneratedTaskTasks.finished_jobs``
    turns it into a ``PlayResult``.
    """

    played_count: int
    error: str


@dataclass(frozen=True)
class DeliveryJob(SupportsReceipt):
    """A job that only sends the generator the results that it did not receive. The receipt is ``delivery``.

    Purpose: ``GeneratedTaskTasks`` submits it at startup when a stop kept results from the generator.
    """


@dataclass(frozen=True)
class DeliveryOutcome(SupportsReceipt):
    """How a delivery job ended: ``error`` is the error of a delivery that failed, or empty.

    Purpose: the worker hands it back, and ``GeneratedTaskTasks.finished_jobs`` keeps its error in the status.
    """

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


def session_record_ids(sessions: tuple[ReportContext, ...]) -> frozenset[str]:
    """The IDs of the records of source sessions: each session's inference records and its report."""
    return frozenset(
        record_id for session in sessions for record_id in (*session.references, session.report.agent_record_id)
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


@dataclass(frozen=True)
class TaskGenerationRound:
    """One task-generation round.

    Values: ``sources`` holds at most ``source_records_per_round`` source
    sessions, in arrival order. It is empty for a round that reads only the
    description.
    Purpose: ``GeneratedTaskRecord2Record`` makes it when a round is due, and
    holds the records of its sources until the round ends.
    ``GeneratedTaskTasks.start_round`` runs it.
    """

    sources: tuple[ReportContext, ...]

    def record_ids(self) -> frozenset[str]:
        """Return the IDs of the records of the sources, which the record stage holds while the round runs."""
        return session_record_ids(self.sources)


class RoundEnd(Enum):
    """How a task-generation round ended."""

    #: The round finished a generation. Its train tasks wait in the task pool.
    GENERATED = "generated"
    #: A complete generation already used every source record of the round, so the round made nothing.
    REPLAYED = "replayed"
    #: The generation could not finish within its attempts and deadline. Its sources wait for the next run.
    DEFERRED = "deferred"


@dataclass(frozen=True)
class RoundResult:
    """How a round ended, and the train tasks that it made.

    Values: ``end`` says how the round ended. ``train_task_names`` names the
    train tasks that the round's generation published. It is empty unless
    ``end`` is ``GENERATED``.
    Purpose: ``GeneratedTaskTasks.finished_jobs`` makes it when a round ends.
    ``GeneratedTaskRecord2Record.advance`` reads it to fill the task pool,
    count the round, and decide when the next round is due.
    """

    end: RoundEnd
    train_task_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.end is not RoundEnd.GENERATED and self.train_task_names:
            raise ValueError("only a round that generated has train tasks")


@dataclass(frozen=True)
class PlayResult:
    """How a play ended.

    Values: ``played_count`` is the number of the play's tasks that played, not
    negative.
    Purpose: ``GeneratedTaskTasks.finished_jobs`` makes it when a play ends.
    ``GeneratedTaskRecord2Record.advance`` reads it: a play that played no task
    makes no batch, so the next round does not wait for one.
    """

    played_count: int

    def __post_init__(self) -> None:
        if self.played_count < 0:
            raise ValueError("played_count must not be negative")


class TaskGroupGrouping(ReportGrouping):
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
        """Whether a report that does not train is a recorded session to generate tasks from."""
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
        DEFAULT_DEFERRED_RETRY_SECONDS,
        help="The seconds after a deferred round before the next round is due, unless new sessions wait.",
    )

    def __post_init__(self) -> None:
        if self.plays_per_task < 2:
            raise ValueError("plays_per_task must be at least two: a group of one has no relative reward")
        if self.batches_per_round < 1:
            raise ValueError("batches_per_round must be positive")
        if self.source_records_per_round < 0:
            raise ValueError("source_records_per_round must be non-negative")
        if self.source_records_per_round > 0 and not self.source_tag:
            raise ValueError("source_records_per_round needs a source_tag that marks the source sessions")
        if self.refill_tasks < 1:
            raise ValueError("refill_tasks must be positive")
        if self.max_rounds_per_run < 0 or self.max_run_seconds < 0 or self.deferred_retry_seconds < 0:
            raise ValueError("max_rounds_per_run, max_run_seconds and deferred_retry_seconds must be non-negative")
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
    served_model: str = config_option("", help="The model the task player asks for; the deployment's by default.")
    eval_fraction: float = config_option(
        DEFAULT_EVAL_FRACTION, help="The fraction of new source groups that go to the eval split."
    )
    split_seed: int = config_option(DEFAULT_SPLIT_SEED, help="The fixed seed of the task store's train/eval split.")
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
        if self.rounds > 0 and not self.description.strip():
            raise ValueError("rounds need a description: what the tasks are about")
        if self.generator is not None:
            TaskGenerator.reference(self.generator, setting="pipeline.record2task.config.generator")


@dataclass(frozen=True)
class TaskBatchSettings(PluginSettings):
    """The config of ``GeneratedTaskBatches``."""

    tasks_per_step: int = config_option(DEFAULT_TASKS_PER_STEP, help="The complete task groups in one batch.")

    def __post_init__(self) -> None:
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")


class GeneratedTaskRecord2Record(ReportRecord2Record):
    """The generated-task record stage: one group per task, the task pool, and the pace of rounds and plays.

    Config: ``plays_per_task`` (the training plays of each task, at least 2; a
    group is complete at this count, or at the plays that reported),
    ``batches_per_round``, ``source_records_per_round`` (the source sessions
    that each request takes, at most), ``source_tag`` (``name=value``;
    required when ``source_records_per_round`` is above 0), ``refill_tasks``,
    ``max_rounds_per_run``, ``max_run_seconds`` and ``deferred_retry_seconds``.
    A terminal report owns its sources.

    The stage starts no round or play until ``resume_rounds`` gives it the
    record-to-task stage that runs them, the rounds that finished a generation
    and the round limit. When no play runs and the task pool holds tasks, the
    stage gives every task of the pool to one play. A round is due when no
    round runs, the round limit is not reached, the task pool holds fewer than
    ``refill_tasks`` tasks, and one of these is true:

    * no round has started yet (the first round after a start);
    * the last round made no train task, or the last play played no task, so
      no batch comes;
    * the last round was deferred, and ``deferred_retry_seconds`` passed or new
      sessions wait;
    * ``batches_per_round`` batches trained since the last round started. A batch
      that the backend dropped as stale does not count.

    The round limit counts the rounds that finished a generation. A deferred
    round and a round of replayed sessions do not count. At the limit, the
    stage stops the rounds and holds no more source sessions; the plays go on.
    If the worker refuses a job, no round or play starts again.

    A source session is a report that does not train and that
    ``TaskGroupGrouping.is_source_report`` accepts, with its inference records.
    The stage holds source sessions in arrival order, and rounds take them
    from a run. A run starts with the sessions that wait at that moment;
    sessions that arrive later wait for the next run. Each round takes at most
    ``source_records_per_round`` sessions of the run. A run ends when it has no
    session left, after ``max_rounds_per_run`` rounds, or after
    ``max_run_seconds``; a limit of 0 is no limit. Sessions that a run did not
    reach wait for the next run. A deferred round keeps its sessions, and the
    next run, which new sessions start, runs that round again before it takes
    new sessions. The stage holds the records of every session that waits or
    that a running round uses.
    """

    def __init__(self, context: PipelineContext, settings: TaskGroupSettings) -> None:
        self.task_grouping = TaskGroupGrouping(settings.plays_per_task, source_tag_of(settings.source_tag))
        super().__init__(context, self.task_grouping, exclusive_sources=True)
        self.batches_per_round = settings.batches_per_round
        self.sources_per_round = settings.source_records_per_round
        self.refill_tasks = settings.refill_tasks
        self.max_rounds_per_run = settings.max_rounds_per_run
        self.max_run_seconds = settings.max_run_seconds
        self.deferred_retry_seconds = settings.deferred_retry_seconds
        self.runner: GeneratedTaskTasks | None = None
        self.completed_rounds = 0
        self.round_limit = 0
        self.running_round: TaskGenerationRound | None = None
        self.play_running = False
        # The first round after a start is due at once.
        self.follow_at_once = True
        self.rounds_stopped = False
        self.jobs_refused = False
        # After a deferred round, the next round is due at this monotonic time or when new sessions wait.
        self.retry_at: float | None = None
        # Batches that trained since the last round started; the next round is due at batches_per_round.
        self.trained_batches = 0
        self.waiting_sources: list[ReportContext] = []
        #: The sources of each deferred round; each one runs again as one round of the next run.
        self.deferred_rounds: list[tuple[ReportContext, ...]] = []
        #: The deferred rounds that the current run has not run again yet.
        self.run_deferred: list[tuple[ReportContext, ...]] = []
        #: The sessions that the current run has not given to a round yet; None between runs.
        self.run_sources: list[ReportContext] | None = None
        self.run_started_at = 0.0
        self.run_round_count = 0
        #: The names of the saved train tasks that wait to play, in the order they arrived.
        self.task_pool: list[str] = []

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(context, TaskGroupSettings.parse(config, plugin_class=cls))

    def plays_reported(self, task_name: str, count: int) -> None:
        """A task's training plays ended with ``count`` episodes reported, so its group is complete at that count."""
        self.task_grouping.reported_plays[task_name] = count
        # The count can arrive before the task's first report.
        if task_name in self.groups:
            self.refresh_group(task_name)

    def resume_rounds(self, runner: GeneratedTaskTasks, *, completed_rounds: int, round_limit: int) -> None:
        """Set the runner, the rounds that finished a generation and the round limit; a restart reads them from its store."""
        self.runner = runner
        self.completed_rounds = completed_rounds
        self.round_limit = round_limit

    def add_tasks(self, task_names: tuple[str, ...]) -> None:
        """Put saved train tasks in the task pool. A restart puts back the tasks that have not played."""
        self.task_pool.extend(name for name in task_names if name not in self.task_pool)

    def handle_other_report(self, report: AgentRecord) -> None:
        # The report ends here. releasable_record_ids holds back a session's records until its round takes it.
        super().handle_other_report(report)
        if self.sources_per_round == 0 or self.rounds_stopped:
            return
        if any(reference not in self.inferences for reference in report.references):
            return
        context = self.report_context(report, None)
        if self.task_grouping.is_source_report(context):
            self.waiting_sources.append(context)

    def advance(self, now: float) -> None:
        """End the rounds and plays that finished, then start a play and the next round when they are due."""
        super().advance(now)
        if self.runner is None:
            return
        for result in self.runner.finished_jobs():
            if isinstance(result, PlayResult):
                self.play_running = False
                # A play that played no task makes no batch to wait for.
                self.follow_at_once = self.follow_at_once or result.played_count == 0
            else:
                self.round_ended(result, now)
        if self.jobs_refused:
            return
        if not self.play_running and self.task_pool:
            if not self.runner.start_play(tuple(self.task_pool)):
                self.refuse_jobs()
                return
            self.task_pool = []
            self.play_running = True
        if not self.round_due(now):
            return
        if self.completed_rounds >= self.round_limit:
            self.stop_rounds()
            return
        generation_round = TaskGenerationRound(sources=self.take_run_sources(now))
        if not self.runner.start_round(generation_round):
            self.refuse_jobs()
            return
        self.running_round = generation_round
        self.follow_at_once = False
        self.retry_at = None
        # The batches that the round's plays make count toward the next one.
        self.trained_batches = 0

    def round_ended(self, result: RoundResult, now: float) -> None:
        """Count a round that generated and pool its train tasks, or keep the sources of a deferred one."""
        sources = () if self.running_round is None else self.running_round.sources
        self.running_round = None
        if result.end is RoundEnd.DEFERRED:
            self.retry_at = now + self.deferred_retry_seconds
            if sources and not self.rounds_stopped:
                self.deferred_rounds.append(sources)
            return
        if result.end is RoundEnd.GENERATED:
            self.completed_rounds += 1
            self.add_tasks(result.train_task_names)
        # A round that made no train task makes no batch to wait for.
        self.follow_at_once = not result.train_task_names

    def round_due(self, now: float) -> bool:
        """Whether the next round should start now."""
        if self.rounds_stopped or self.running_round is not None or len(self.task_pool) >= self.refill_tasks:
            return False
        if self.retry_at is not None and (now >= self.retry_at or self.waiting_sources):
            return True
        return self.follow_at_once or self.trained_batches >= self.batches_per_round

    def take_run_sources(self, now: float) -> tuple[ReportContext, ...]:
        """The next round's sessions from the current run; a run that is over gives way to a new one."""
        if self.sources_per_round == 0:
            return ()
        if self.run_sources is not None and (
            not (self.run_sources or self.run_deferred)
            or 0 < self.max_rounds_per_run <= self.run_round_count
            or 0 < self.max_run_seconds <= now - self.run_started_at
        ):
            self.waiting_sources = [*self.run_sources, *self.waiting_sources]
            self.deferred_rounds = [*self.run_deferred, *self.deferred_rounds]
            self.run_sources, self.run_deferred = None, []
        if self.run_sources is None:
            # Deferred rounds wait for a run that new sessions start.
            if not self.waiting_sources:
                return ()
            self.run_sources, self.waiting_sources = self.waiting_sources, []
            self.run_deferred, self.deferred_rounds = self.deferred_rounds, []
            self.run_started_at = now
            self.run_round_count = 0
        self.run_round_count += 1
        if self.run_deferred:
            return self.run_deferred.pop(0)
        sources = tuple(self.run_sources[: self.sources_per_round])
        self.run_sources = self.run_sources[self.sources_per_round :]
        return sources

    def stop_rounds(self) -> None:
        """No round runs again, so no session waits for one."""
        self.rounds_stopped = True
        self.waiting_sources = []
        self.deferred_rounds, self.run_deferred = [], []
        self.run_sources = None

    def refuse_jobs(self) -> None:
        """The worker refused a job, so it is closed or broken: no round or play starts again."""
        self.jobs_refused = True
        self.stop_rounds()

    def consume(self, units: tuple[RecordSetUnit, ...], *, trained: bool) -> frozenset[str]:
        if trained:
            self.trained_batches += 1
        return super().consume(units, trained=trained)

    def releasable_record_ids(self) -> frozenset[str]:
        deferred = tuple(session for sources in (*self.deferred_rounds, *self.run_deferred) for session in sources)
        held = session_record_ids((*self.waiting_sources, *deferred, *(self.run_sources or ())))
        if self.running_round is not None:
            held |= self.running_round.record_ids()
        return super().releasable_record_ids() - held

    def background_jobs_running(self) -> bool:
        """Return True while a round or a play runs: its end can start the next job without a new record."""
        return self.running_round is not None or self.play_running or super().background_jobs_running()

    def status(self) -> Mapping[str, Any]:
        return {**super().status(), "task_pool": len(self.task_pool)}

    def operational_metrics(self) -> dict[str, float | int]:
        return {
            **super().operational_metrics(),
            "waiting_source_sessions": len(self.waiting_sources),
            "deferred_rounds": len(self.deferred_rounds) + len(self.run_deferred),
            "task_pool": len(self.task_pool),
        }


class GeneratedTaskTasks(ReportTasks, BackgroundJobRunner):
    """The record-to-task stage of generated tasks: played episodes as tasks, and the generation and play jobs.

    The record stage starts a round when it is due (``start_round``), and a
    play of the saved train tasks that wait in its task pool
    (``start_play``). The stage runs each one as a job on its worker.

    A generation job generates tasks from the description and the round's
    source sessions, validates them and saves them in the task store. It
    plays none. A generator or check that fails to run is retried, up to
    ``generation_attempts`` times within ``generation_deadline_seconds``. After
    that the round is deferred. A round whose source records a complete
    generation already used makes nothing, because a restart replays those
    sessions. A round whose sources an incomplete generation used continues
    that generation from its saved candidates.

    A play job writes each task through the generator service, plays it, and
    saves its result. The plays come back as records and train as reports. A
    task that cannot run at all has an infrastructure failure as its result.
    When every train task of a generation has a result, the next job sends the
    generation's results to the generator. A generation whose delivery fails
    waits for the next job and does not hold back later generations.

    ``finished_jobs`` returns each round and play that ended. This stage owns
    the worker. The tasks, their splits, their play results and the
    deliveries go in the task store ``generated-tasks`` of the scenario
    session. A restart puts the unplayed train tasks back in the record
    stage's task pool, and sends the results that the generator did not
    receive.
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
        super().__init__()
        #: The record stage that paces the rounds and plays and closes the task groups.
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
            if context.task_storage is None:
                raise ValueError("rounds need the task storage of a scenario session")
        self.task_generator = task_generator
        self.task_player = task_player
        self.worker = worker
        if self.worker is None and self.rounds > 0:
            self.worker = BackgroundJobWorker(self, thread_name="generated-task-rounds", concurrency=1)
        if self.task_generator is not None:
            logger.info("scenario %r generates tasks with %s", self.scenario, type(self.task_generator).__name__)
        self.split_settings = SplitSettings(eval_fraction=settings.eval_fraction, seed=settings.split_seed)
        self.task_store = (
            context.task_storage.open(TASK_STORE_NAME, split=self.split_settings)
            if context.task_storage is not None and self.rounds > 0
            else None
        )
        # The worker puts each task's count here as its plays end; the trainer's thread takes them in finished_jobs.
        self.reported_play_counts: queue.SimpleQueue[tuple[str, int]] = queue.SimpleQueue()
        states = self.task_store.generations() if self.task_store is not None else ()
        round_stage.resume_rounds(
            self, completed_rounds=sum(state.is_complete for state in states), round_limit=self.rounds
        )
        if self.task_store is not None:
            round_stage.add_tasks(unplayed_train_tasks(self.task_store))
        # The saved results close the groups of replayed reports.
        for state in states:
            for result in state.play_results:
                round_stage.plays_reported(result.task_name, result.reported_plays)
        #: The generation that runs on the worker. One round runs at a time.
        self.generation_in_flight: int | None = None
        #: The ends of rounds that needed no job, for the next ``finished_jobs``.
        self.ended_rounds: list[RoundResult] = []
        #: The number of the next play, which names its job.
        self.next_play = 0
        self.last_error = ""
        # Results that a stop kept from the generator go out now, even when no other job is due.
        if self.worker is not None and any(state.is_feedback_complete and not state.is_delivered for state in states):
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

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        task_name = reported_task_name(context.report)
        if task_name is None:
            raise ValueError("a generated task's report requires metadata.task.name, as the task player sends it")
        sample = self.assembly.build(context, context.require_score())
        return replace(sample, group_id=task_name)

    def start_round(self, generation_round: TaskGenerationRound) -> bool:
        """Submit the round's generation to the worker. A worker that refuses it is closed or broken.

        A round whose source records a complete generation already used ends at once with ``REPLAYED``.
        """
        if self.task_store is None:
            raise RuntimeError("a round needs a task store: set pipeline.record2task.config.rounds")
        records = session_records(generation_round.sources)
        states = self.task_store.generations()
        completed = {record_id for state in states if state.is_complete for record_id in state.source_record_ids}
        remaining = tuple(record for record in records if record.agent_record_id not in completed)
        # A restart replays sessions that a complete generation already used; their tasks are saved.
        if records and not remaining:
            self.ended_rounds.append(RoundResult(RoundEnd.REPLAYED))
            return True
        source_record_ids = tuple(record.agent_record_id for record in remaining)
        # A deferred round runs again as its generation, which continues from the saved candidates.
        existing = next(
            (state for state in states if source_record_ids and state.source_record_ids == source_record_ids), None
        )
        generation = next_generation(self.task_store) if existing is None else existing.generation
        job = GenerationJob(receipt=f"generation-{generation:05d}", generation=generation, source_records=remaining)
        if self.worker is None or not self.worker.submit(job):
            self.last_error = (
                f"the round worker is closed or broken; {job.receipt} cannot run, and no further job runs"
            )
            return False
        self.generation_in_flight = generation
        return True

    def start_play(self, task_names: tuple[str, ...]) -> bool:
        """Submit a play of saved train tasks to the worker. A worker that refuses it is closed or broken."""
        job = PlayJob(receipt=f"play-{self.next_play:05d}", task_names=task_names)
        if self.worker is None or not self.worker.submit(job):
            self.last_error = (
                f"the round worker is closed or broken; {job.receipt} cannot run, and no further job runs"
            )
            return False
        self.next_play += 1
        return True

    def finished_jobs(self) -> tuple[RoundResult | PlayResult, ...]:
        """Close the task groups whose plays ended, and return each round and play that ended."""
        while not self.reported_play_counts.empty():
            self.round_stage.plays_reported(*self.reported_play_counts.get_nowait())
        finished: list[RoundResult | PlayResult] = list(self.ended_rounds)
        self.ended_rounds = []
        if self.worker is None:
            return tuple(finished)
        for outcome in self.worker.poll():
            if outcome.receipt == DELIVERY_RECEIPT:
                self.last_error = (
                    outcome.error if isinstance(outcome, DeliveryOutcome) else "the delivery did not finish"
                )
                continue
            if isinstance(outcome, Failed):
                self.last_error = f"{outcome.receipt} did not finish; see the service log"
                logger.error("generated-task %s failed on scenario %r", outcome.receipt, self.scenario)
                if outcome.receipt.startswith("generation-"):
                    self.generation_in_flight = None
                    finished.append(RoundResult(RoundEnd.DEFERRED))
                else:
                    finished.append(PlayResult(played_count=0))
                continue
            if isinstance(outcome, GenerationOutcome):
                self.generation_in_flight = None
                self.last_error = outcome.error
                logger.info(
                    "generated-task generation %d on scenario %r: %d train tasks%s",
                    outcome.generation,
                    self.scenario,
                    len(outcome.train_task_names),
                    ", deferred" if outcome.is_deferred else "",
                )
                if outcome.is_deferred:
                    finished.append(RoundResult(RoundEnd.DEFERRED))
                else:
                    finished.append(RoundResult(RoundEnd.GENERATED, outcome.train_task_names))
                continue
            if not isinstance(outcome, PlayOutcome):
                raise TypeError("the round worker must hand back a GenerationOutcome, a PlayOutcome or a delivery")
            self.last_error = outcome.error
            finished.append(PlayResult(played_count=outcome.played_count))
        return tuple(finished)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {
            "rounds_completed": self.round_stage.completed_rounds,
            "round_in_flight": 0 if self.generation_in_flight is None else 1,
        }

    def status(self) -> Mapping[str, Any]:
        if self.generation_in_flight is not None:
            next_number = self.generation_in_flight + 1
        else:
            next_number = next_generation(self.task_store) if self.task_store is not None else 0
        return {
            "task_store": task_counts(self.task_store) if self.task_store is not None else {},
            "task_generation": {
                "generator": None if self.task_generator is None else type(self.task_generator).__name__,
                "in_flight": self.generation_in_flight,
                "next": next_number,
                "completed": self.round_stage.completed_rounds,
                "of": self.rounds,
                "last_error": self.last_error,
            },
        }

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Run one job on the worker, after sending the generator any results that it has not received."""
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
            return replace(play, error=delivery_error)
        raise TypeError(f"the round worker takes GenerationJob, PlayJob or DeliveryJob, not {type(job).__name__}")

    async def run_generation(self, job: GenerationJob) -> GenerationOutcome:
        """Generate, validate and save one generation; retry a failure to run within the attempts and deadline."""
        if self.task_store is None:
            raise RuntimeError("a generation needs a task store")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.generation_deadline_seconds
        error = ""
        for attempt in range(1, self.generation_attempts + 1):
            remaining_seconds = deadline - loop.time()
            if remaining_seconds <= 0:
                break
            # The generator is a deployment's plugin: whatever it raises is a failure to run.
            try:
                state = await asyncio.wait_for(self.generated(job), remaining_seconds)
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                logger.exception(
                    "generated-task generation %d attempt %d failed on scenario %r",
                    job.generation,
                    attempt,
                    self.scenario,
                )
                continue
            return GenerationOutcome(
                receipt=job.receipt,
                generation=job.generation,
                train_task_names=state.train_task_names,
                is_deferred=False,
                error="",
            )
        return GenerationOutcome(
            receipt=job.receipt,
            generation=job.generation,
            train_task_names=(),
            is_deferred=True,
            error=error or "the generation deadline passed",
        )

    async def generated(self, job: GenerationJob) -> GenerationState:
        """One attempt: continue from the saved candidates and validation outcomes, then complete the generation."""
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("a generation needs a task generator and a task store")
        generation = self.task_store.begin_generation(
            generation=job.generation,
            source_record_ids=tuple(record.agent_record_id for record in job.source_records),
        )
        if generation.candidates is None:
            generated = await self.task_generator.generate(
                TaskGenerationRequest(job.source_records, self.description, generation=job.generation)
            )
            generation = self.task_store.save_candidates(generation=job.generation, candidates=generated)
        candidates = generation.candidates or ()
        for index in generation.pending_indices:
            item = candidates[index].generated
            if any(arm.name == TRAINING_ARM for arm in item.measurement_arms):
                refusal = f"the measurement arm {TRAINING_ARM!r} has the name of the training arm"
                self.task_store.reject_task(generation=job.generation, index=index, reason=refusal)
                continue
            # A task has one play result, in the generation that published it first.
            earlier = next(
                (
                    task
                    for task in self.task_store.tasks()
                    if task.generated.task.name == item.task.name and task.generation != job.generation
                ),
                None,
            )
            if earlier is not None:
                duplicate = f"task {item.task.name!r} is already published in generation {earlier.generation}"
                self.task_store.reject_task(generation=job.generation, index=index, reason=duplicate)
                continue
            validation = await self.task_generator.validate(item.task)
            if validation.is_valid:
                try:
                    split = assigned_split(self.task_store, self.split_settings, item)
                    self.task_store.publish_task(generation=job.generation, index=index, split=split)
                except (TaskConflict, TaskSplitConflict) as conflict:
                    self.task_store.reject_task(generation=job.generation, index=index, reason=str(conflict))
            else:
                self.task_store.reject_task(
                    generation=job.generation, index=index, reason="; ".join(validation.errors)
                )
        return self.task_store.complete_generation(generation=job.generation)

    async def run_play(self, job: PlayJob) -> PlayOutcome:
        """Play each task and save its result; a task that cannot run has an infrastructure failure as its result."""
        if self.task_store is None:
            raise RuntimeError("a play needs a task store")
        saved_by_name = {saved.generated.task.name: saved for saved in self.task_store.tasks(split="train")}
        played_count = 0
        for name in job.task_names:
            saved = saved_by_name[name]
            try:
                result, reported = await self.played(saved.generated, saved.generation)
            except Exception as exc:
                logger.exception("generated-task %s could not play on scenario %r", name, self.scenario)
                result = TaskPlayResult(
                    saved.generated.task, error=str(exc) or type(exc).__name__, infrastructure_failure=True
                )
                reported = 0
            self.task_store.save_play_result(
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
                played_count += 1
                # The task's group can close now, before the play ends.
                self.reported_play_counts.put((name, reported))
        return PlayOutcome(receipt=job.receipt, played_count=played_count, error="")

    async def played(self, item: GeneratedTask, generation: int) -> tuple[TaskPlayResult, int]:
        """Write one valid task and play it: the training arm, then each measurement arm.

        Returns the task's result and how many of its training episodes reached Reef, 0 for a task that was
        not played. A task the agent could not play at all (every training episode ended before the agent
        ran) is removed and comes back with the first episode's error as an infrastructure failure, and its
        measurement arms are not played.
        """
        if self.task_player is None:
            raise RuntimeError("playing a task needs the generator service")
        task = item.task
        try:
            written = await self.task_player.write_task(task)
        except DuplicateTask:
            # A play that a stop interrupted left the task written; write it again.
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

        Returns the error of the last delivery that failed. A generation whose delivery failed waits for the
        next job, and later generations still go out.
        """
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("delivering results needs a task generator and a task store")
        error = ""
        for state in self.task_store.generations():
            if not state.is_feedback_complete or state.is_delivered:
                continue
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

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)
