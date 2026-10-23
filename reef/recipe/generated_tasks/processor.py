"""The generated-task stages: rounds of generated tasks played here, their training episodes trained here.

The record stage (``GeneratedTaskRecord2Record``) groups the task player's reports and paces the rounds: every
episode of the training arm names its task under ``metadata.task``, the episodes of one task form a
group, and a group is complete at ``plays_per_task`` episodes. A training play that reports nothing, such
as a failed trial, never joins its group. So when a task's plays end, its group is complete at the number
of plays that reported, and a group of one is discarded. ``GeneratedTaskBatches`` batches complete
groups, in arrival order, until the batch holds the backend's batch size of episodes. When a round is
due, the record stage gives a
``TaskGenerationRound`` to ``GeneratedTaskTasks``. That stage runs the round with the
deployment's :class:`~reef.recipe.generated_tasks.TaskGenerator` and returns a ``RoundResult`` when the round ends. A
round is not a record set and does not train; the episodes that its plays report do.

One round is one job on a private worker, off the trainer's thread:

1. Ask the generator for tasks, with the round's source sessions when the record stage gave any.
2. Save the candidates in the scenario's task store, and validate each task with the generator.
3. Publish each valid task with its train/eval split.
4. Write each training task through the generator service and play it ``plays_per_task`` times on the
   training arm. Eval tasks are not played.
5. Send the results to the generator.

The first round starts when the pipeline first looks for a batch. The next round starts after
``batches_per_round`` batches were acknowledged since the previous round started. Its episodes train
while it runs, so the generator always writes for the policy that trains now. A round that played no
task is followed at once. A summary of each round goes under ``state_dir/<scenario>``; a restart
reads it to continue with the next round.

Only the training arm, labeled ``plain``, trains. The record stage releases reports of other arms, such as
the plays that a generator runs itself, and reports without an arm, such as a generator's own reports,
without assembling them.
"""

from __future__ import annotations

import json
import logging
import queue
import tempfile
from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord
from reef.core.config import config_option
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.core.tasks import HarborTask, read_harbor_task, write_harbor_task
from reef.core.tasks.split import TaskSplit, TaskSplitConflict, extend_split_by_source
from reef.harness.client.tasks import TaskPlay
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks.generator import TaskGenerator, TaskPlayResult
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
from reef.record2task.generator.inputs import TaskGenerationRequest
from reef.storage.tasks import SplitName, TaskConflict, TaskStore
from reef.task2batch import BufferSettings, Task2Batch
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

DEFAULT_PLAYS_PER_TASK = 4
# Qwen3 with thinking off ends the generation prompt with an empty think block of four tokens.
DEFAULT_SCAFFOLD_TOLERANCE = 8
# A realigned drift spans the whole previous response, so the budget must exceed max_tokens plus the scaffold.
REALIGN_THRESHOLD = 32768
DEFAULT_BATCHES_PER_ROUND = 1
ERROR_EXCERPT_CHARS = 300
ROUND_PREFIX = "round-"
TASK_STORE_NAME = "generated-tasks"
DEFAULT_EVAL_FRACTION = 0.0
DEFAULT_SPLIT_SEED = 0
#: The arm label of the training plays. Only reports with this label train.
TRAINING_ARM = "plain"


def reported_task_name(report: AgentRecord) -> str | None:
    """Read the task player's task name from a report."""
    metadata = report.payload.get("metadata")
    task = metadata.get("task") if isinstance(metadata, Mapping) else None
    name = task.get("name") if isinstance(task, Mapping) else None
    return name if isinstance(name, str) and name else None


def reported_arm(report: AgentRecord) -> str | None:
    """Read the episode's arm label from a task-player report."""
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


def split_source_ids(task: HarborTask) -> tuple[str, ...]:
    """The source record ids that place a task in a split: its sources without the author's own record."""
    designer_record_id = task.metadata.get("designer_record_id")
    return tuple(record_id for record_id in task.source_agent_record_ids if record_id != designer_record_id)


def task_counts(store: TaskStore) -> dict[str, int]:
    """Count the published tasks of each split, and the rejected and pending candidates."""
    splits = Counter(task.split for task in store.tasks())
    candidates = [candidate for state in store.generations() for candidate in state.candidates or ()]
    return {
        "train": splits["train"],
        "eval": splits["eval"],
        "rejected": sum(bool(candidate.error) for candidate in candidates),
        "pending": sum(candidate.is_pending for candidate in candidates),
    }


def assigned_split(store: TaskStore, task: HarborTask, *, seed: int, eval_fraction: float) -> SplitName:
    """Return the split of a task: the split of its published name, or else the split of its source records.

    Tasks that share a source record get the same split, and a published task keeps its split. Raises
    ``TaskSplitConflict`` if the sources of the task join tasks of both splits.
    """
    published = {stored.name: stored for stored in store.tasks()}
    existing = published.get(task.name)
    if existing is not None:
        return existing.split
    previous = TaskSplit(
        train=tuple(name for name, stored in published.items() if stored.split == "train"),
        eval=tuple(name for name, stored in published.items() if stored.split == "eval"),
        seed=seed,
        eval_fraction=eval_fraction,
    )
    assigned = extend_split_by_source(
        previous=previous,
        previous_sources={
            name: split_source_ids(read_harbor_task(store.path(stored.files_digest)))
            for name, stored in published.items()
        },
        sources={task.name: split_source_ids(task)},
    )
    return "eval" if task.name in assigned.eval else "train"


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


@dataclass(frozen=True)
class TaskGenerationRound:
    """One task-generation round.

    Values: ``round_number`` is the number of the round. The rounds of one
    scenario have the numbers 0, 1, 2, and so on, in order. ``sources`` holds
    at most ``source_records_per_round`` source sessions, in arrival order.
    Purpose: ``GeneratedTaskRecord2Record`` makes it when a round is due.
    ``GeneratedTaskTasks`` runs it.
    """

    round_number: int
    sources: tuple[ReportContext, ...]

    def record_ids(self) -> frozenset[str]:
        """Return the IDs of the records of the sources, which the record stage holds while the round runs."""
        return frozenset(
            record_id for source in self.sources for record_id in (*source.references, source.report.agent_record_id)
        )


@dataclass(frozen=True)
class RoundResult:
    """How a round ended.

    Values: ``wait_for_batches`` is True when the next round waits for
    ``batches_per_round`` trained batches. It is False when the next round can
    start at once, because the round made no batch.
    Purpose: ``GeneratedTaskTasks.finished_rounds`` makes it when a round ends.
    ``GeneratedTaskRecord2Record.advance`` reads it to decide when the next round is due.
    """

    wait_for_batches: bool


class TaskGroupGrouping(ReportGrouping):
    """The training arm's episodes of one task form one group, ready at ``plays_per_task`` episodes.

    A group is ready at fewer episodes when fewer of the task's plays reported (``reported_plays``), and a
    group of one is discarded. The generator's own plays and reports share the scenario; only the training
    arm trains. A source session is a report whose inference records all carry ``source_tag``.
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
    eval_fraction: float = config_option(
        DEFAULT_EVAL_FRACTION, help="The fraction of new source groups that go to the eval split."
    )
    split_seed: int = config_option(DEFAULT_SPLIT_SEED, help="The fixed seed of the task store's train/eval split.")
    scaffold_tolerance: int = config_option(
        DEFAULT_SCAFFOLD_TOLERANCE, help="Masked tokens the sample assembly may realign before a response."
    )

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must be non-negative")
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


class GeneratedTaskRecord2Record(ReportRecord2Record):
    """The generated-task record stage: one group per task, and the pace of the task-generation rounds.

    Config: ``plays_per_task`` (the training plays of each task, at least 2; a
    group is complete at this count, or at the plays that reported), ``batches_per_round``,
    ``source_records_per_round`` (the source sessions that each request takes,
    at most), and ``source_tag`` (``name=value``; required when
    ``source_records_per_round`` is above 0). A terminal report owns its sources.

    The stage starts no round until ``resume_rounds`` gives it the
    record-to-task stage that runs the rounds, the first round number and the
    round limit. A round is due when no round is running, the round limit is
    not reached, and one of these is true:

    * no round has started yet (the first round after a start);
    * the last round ended with ``wait_for_batches`` False, for example a round
      that played no task, so it makes no batch to wait for;
    * ``batches_per_round`` batches trained since the last round started. A batch
      that the backend dropped as stale does not count.

    At the limit, the stage stops the rounds and holds no more source sessions.
    A source session is a report that does not train and that
    ``TaskGroupGrouping.is_source_report`` accepts, with its inference records.
    The stage holds source sessions in arrival order. Each round takes at most
    ``source_records_per_round`` of them. The stage holds the records of a
    round's sources until the round ends.
    """

    def __init__(self, context: PipelineContext, settings: TaskGroupSettings) -> None:
        self.task_grouping = TaskGroupGrouping(settings.plays_per_task, source_tag_of(settings.source_tag))
        super().__init__(context, self.task_grouping, exclusive_sources=True)
        self.batches_per_round = settings.batches_per_round
        self.sources_per_round = settings.source_records_per_round
        self.runner: GeneratedTaskTasks | None = None
        self.next_round = 0
        self.round_limit = 0
        self.running_round: TaskGenerationRound | None = None
        # The first round after a start is due at once.
        self.follow_at_once = True
        self.rounds_stopped = False
        # Batches that trained since the last round started; the next round is due at batches_per_round.
        self.trained_batches = 0
        self.waiting_sources: list[ReportContext] = []

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(context, TaskGroupSettings.parse(config, plugin_class=cls))

    def plays_reported(self, task_name: str, count: int) -> None:
        """A task's training plays ended with ``count`` episodes reported, so its group is complete at that count."""
        self.task_grouping.reported_plays[task_name] = count
        # The count can arrive before the task's first report.
        if task_name in self.groups:
            self.refresh_group(task_name)

    def resume_rounds(self, runner: GeneratedTaskTasks, *, next_round: int, round_limit: int) -> None:
        """Set the runner, the next round number and the round limit. After a restart, the next round follows the recorded ones."""
        self.runner = runner
        self.next_round = next_round
        self.round_limit = round_limit

    def handle_other_report(self, report: AgentRecord) -> None:
        # The report ends here. releasable_record_ids holds back a session's records until its round ends.
        super().handle_other_report(report)
        if self.sources_per_round == 0 or self.rounds_stopped:
            return
        if any(reference not in self.inferences for reference in report.references):
            return
        context = self.report_context(report, None)
        if self.task_grouping.is_source_report(context):
            self.waiting_sources.append(context)

    def advance(self, now: float) -> None:
        """End the rounds that finished, then start the next round when it is due, or stop the rounds at the limit."""
        super().advance(now)
        if self.runner is None:
            return
        for result in self.runner.finished_rounds():
            self.running_round = None
            self.follow_at_once = not result.wait_for_batches
        if self.rounds_stopped or self.running_round is not None:
            return
        if not self.follow_at_once and self.trained_batches < self.batches_per_round:
            return
        if self.next_round >= self.round_limit:
            self.stop_rounds()
            return
        sources = tuple(self.waiting_sources[: self.sources_per_round])
        generation_round = TaskGenerationRound(round_number=self.next_round, sources=sources)
        if not self.runner.start_round(generation_round):
            self.stop_rounds()
            return
        self.waiting_sources = self.waiting_sources[self.sources_per_round :]
        self.running_round = generation_round
        self.next_round += 1
        self.follow_at_once = False
        # The batches that the round's plays make count toward the next one.
        self.trained_batches = 0

    def stop_rounds(self) -> None:
        """No round runs again, so no session waits for one."""
        self.rounds_stopped = True
        self.waiting_sources = []

    def consume(self, units: tuple[RecordSetUnit, ...], *, trained: bool) -> frozenset[str]:
        if trained:
            self.trained_batches += 1
        return super().consume(units, trained=trained)

    def releasable_record_ids(self) -> frozenset[str]:
        held = {
            record_id
            for context in self.waiting_sources
            for record_id in (*context.references, context.report.agent_record_id)
        }
        if self.running_round is not None:
            held |= self.running_round.record_ids()
        return super().releasable_record_ids() - held

    def background_jobs_running(self) -> bool:
        """Return True while a round runs: its end can start the next round without a new record."""
        return self.running_round is not None or super().background_jobs_running()

    def operational_metrics(self) -> dict[str, float | int]:
        return {**super().operational_metrics(), "waiting_source_sessions": len(self.waiting_sources)}


class GeneratedTaskTasks(ReportTasks, BackgroundJobRunner):
    """The record-to-task stage of generated tasks: played episodes as tasks, and the runner of the rounds.

    The record stage starts a round when it is due (``start_round``). The
    stage runs it on its worker. The round's source sessions go to the
    generator as the request's ``source_records``. The round generates tasks,
    validates them, has the generator service play them, and sends the
    results to the generator. The plays come back as records and train as reports.

    ``finished_rounds`` returns each round that ended. After a round that
    played a task, or whose job failed, the next round waits for batches.
    After a round that played no task, the next round starts at once.

    This stage owns the worker and the round summaries under
    ``state_dir/<scenario>``. A restart reads the summaries to continue with
    the next round. The tasks and their splits go in the task store
    ``generated-tasks`` of the scenario session.
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
        #: The record stage that paces the rounds and closes the task groups.
        super().__init__()
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
            if context.task_storage is None:
                raise ValueError("rounds need the task storage of a scenario session")
        self.task_generator = task_generator
        self.task_player = task_player
        self.worker = worker
        if self.worker is None and self.rounds > 0:
            self.worker = BackgroundJobWorker(self, thread_name="generated-task-rounds", concurrency=1)
        if self.task_generator is not None:
            logger.info("scenario %r generates tasks with %s", self.scenario, type(self.task_generator).__name__)
        self.eval_fraction = settings.eval_fraction
        self.split_seed = settings.split_seed
        self.task_store = (
            context.task_storage.open(TASK_STORE_NAME)
            if context.task_storage is not None and self.rounds > 0
            else None
        )
        recorded = recorded_rounds(self.state_dir) if self.state_dir is not None else ()
        # The round summaries keep how many plays of each task reported, so replayed reports close the groups.
        if self.state_dir is not None:
            for task_name, count in recorded_reported_plays(self.state_dir).items():
                round_stage.plays_reported(task_name, count)
        # The worker puts each task's count here as its plays end; the trainer's thread takes them in finished_rounds.
        self.reported_play_counts: queue.SimpleQueue[tuple[str, int]] = queue.SimpleQueue()
        round_stage.resume_rounds(self, next_round=recorded[-1] + 1 if recorded else 0, round_limit=self.rounds)
        self.completed_rounds = len(recorded)
        #: The round that runs on the worker. One round runs at a time.
        self.round_in_flight: TaskGenerationRound | None = None
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

    def start_round(self, generation_round: TaskGenerationRound) -> bool:
        """Submit the round to the worker. A worker that refuses it is closed or broken, so no round runs again."""
        job = RoundJob(
            receipt=f"round-{generation_round.round_number:05d}",
            round_number=generation_round.round_number,
            source_records=session_records(generation_round.sources),
        )
        if self.worker is None or not self.worker.submit(job):
            self.last_error = "the round worker is closed or broken; no further round runs"
            return False
        self.round_in_flight = generation_round
        return True

    def finished_rounds(self) -> tuple[RoundResult, ...]:
        """Close the task groups whose plays ended, and return each round that ended."""
        while not self.reported_play_counts.empty():
            self.round_stage.plays_reported(*self.reported_play_counts.get_nowait())
        if self.worker is None or self.round_in_flight is None:
            return ()
        finished: list[RoundResult] = []
        for outcome in self.worker.poll():
            self.round_in_flight = None
            if isinstance(outcome, Failed):
                self.last_error = f"{outcome.receipt} did not finish; see the service log"
                logger.error("generated-task %s failed on scenario %r", outcome.receipt, self.scenario)
                # A failing worker must not start round after round, so the next round waits for batches.
                finished.append(RoundResult(wait_for_batches=True))
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
            finished.append(RoundResult(wait_for_batches=played > 0))
        return tuple(finished)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {"rounds_completed": self.completed_rounds, "round_in_flight": 0 if self.round_in_flight is None else 1}

    def status(self) -> Mapping[str, Any]:
        return {
            "task_store": task_counts(self.task_store) if self.task_store is not None else {},
            "task_generation": {
                "generator": None if self.task_generator is None else type(self.task_generator).__name__,
                "in_flight": None if self.round_in_flight is None else self.round_in_flight.round_number,
                "next": self.round_stage.next_round,
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
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("a round needs a task generator and a task store")
        results: list[TaskPlayResult] = []
        reported_plays: dict[str, int] = {}
        error = ""
        # The generator is a deployment's plugin: whatever it raises stops the round, and what ran is kept.
        try:
            generation = self.task_store.begin_generation(
                generation=job.round_number,
                source_record_ids=tuple(record.agent_record_id for record in job.source_records),
            )
            if generation.candidates is None:
                generated = await self.task_generator.generate(
                    TaskGenerationRequest(job.source_records, self.description)
                )
                with tempfile.TemporaryDirectory(prefix="reef-candidates-") as staging:
                    # One directory for each candidate, because two candidates can have the same name.
                    directories = tuple(
                        write_harbor_task(task, Path(staging) / str(index)) for index, task in enumerate(generated)
                    )
                    generation = self.task_store.save_candidates(generation=job.round_number, directories=directories)
            candidates = generation.candidates or ()
            for index in generation.pending_indices:
                task = read_harbor_task(self.task_store.path(candidates[index].files_digest))
                validation = await self.task_generator.validate(task)
                if validation.is_valid:
                    try:
                        split = assigned_split(
                            self.task_store, task, seed=self.split_seed, eval_fraction=self.eval_fraction
                        )
                        self.task_store.publish_task(generation=job.round_number, index=index, split=split)
                    except (TaskConflict, TaskSplitConflict) as conflict:
                        self.task_store.reject_task(generation=job.round_number, index=index, reason=str(conflict))
                else:
                    self.task_store.reject_task(
                        generation=job.round_number, index=index, reason="; ".join(validation.errors)
                    )
            self.task_store.complete_generation(generation=job.round_number)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            logger.exception("generated-task round %d stopped on scenario %r", job.round_number, self.scenario)
        # Play already validated tasks even when a later validation stopped the round.
        saved = self.task_store.generation(job.round_number)
        saved_candidates = () if saved is None or saved.candidates is None else saved.candidates
        try:
            for candidate in saved_candidates:
                if candidate.is_pending:
                    continue
                task = read_harbor_task(self.task_store.path(candidate.files_digest))
                if candidate.error:
                    results.append(TaskPlayResult(task=task, error=candidate.error))
                elif candidate.split == "eval":
                    results.append(TaskPlayResult(task=task, split="eval"))
                else:
                    result, reported = await self.played(task, job.round_number)
                    results.append(result)
                    if result.is_played:
                        reported_plays[task.name] = reported
                        # The task's group can close now, before the round ends.
                        self.reported_play_counts.put((task.name, reported))
        except Exception as exc:
            error = error or str(exc) or type(exc).__name__
            logger.exception("generated-task rollout %d stopped on scenario %r", job.round_number, self.scenario)
        try:
            await self.task_generator.record_results(tuple(results))
        except Exception as exc:
            error = error or str(exc) or type(exc).__name__
            logger.exception("the task generator did not take the results of round %d", job.round_number)
        self.write_round_summary(job.round_number, tuple(results), error, reported_plays, job.source_records)
        return RoundOutcome(receipt=job.receipt, round_number=job.round_number, results=tuple(results), error=error)

    async def played(self, task: HarborTask, round_number: int) -> tuple[TaskPlayResult, int]:
        """Write one valid task and play it on the training arm.

        Returns the task's result and how many of its training episodes reached Reef, 0 for a task that was
        not played. A task the agent could not play at all (every training episode ended before the agent
        ran) is removed and comes back with the first episode's error.
        """
        if self.task_player is None:
            raise RuntimeError("playing a task needs the generator service")
        try:
            written = await self.task_player.write_task(task)
        except (DuplicateTask, TaskNameConflict) as exc:
            return TaskPlayResult(task, error=f"refused: {exc}"), 0
        training = await self.task_player.play(
            written.path,
            scenario=self.scenario,
            arm=TRAINING_ARM,
            plays=self.plays_per_task,
            is_reporting=True,
            extra_instruction_files=(),
            tags={"round": str(round_number)},
            model=self.served_model,
        )
        if training and all(play.reward is None and play.error for play in training):
            await self.task_player.delete_task(task.name)
            error = f"the agent could not play the task: {training[0].error[:ERROR_EXCERPT_CHARS]}"
            return TaskPlayResult(task, error=error), 0
        reported = sum(play.is_reported for play in training)
        return TaskPlayResult(task, rewards_of(training)), reported

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
                    "split": result.split,
                    "training_rewards": list(result.training_rewards),
                    "reported_plays": reported_plays.get(result.task.name),
                }
                for result in results
            ],
        }
        path = self.state_dir / f"{ROUND_PREFIX}{round_number:05d}.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path


class GeneratedTaskBatches(Task2Batch):
    """Batch the training arm's complete task groups; one record set of a batch is one group."""

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
        return cls(context, BufferSettings.parse(config, plugin_class=cls).buffer_size)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)
