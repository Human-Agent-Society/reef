"""The generated-task stages: tasks generated and saved here, rolled out from the task pool, and trained here.

When a round is due, the record stage (``GeneratedTaskRecord2Record``) gives a ``TaskGenerationRound`` to
``GeneratedTaskTasks``, which runs it as one generation job on a private worker, off the trainer's thread. A
generation job:

1. Asks the generator for tasks, with the round's source sessions when the record stage gave any.
2. Saves the candidates in the scenario's task store, and validates each task with the generator.
3. Publishes each valid task with its train/eval split. Eval tasks run no episodes.

The round's train tasks wait in the task pool of ``GeneratedTaskJobData``. The pipeline takes up to
``tasks_per_step`` of them as one task job's data. The training component rolls the tasks out on its own
sampler and trains on their episodes; Reef runs no episode of them. When the job's end commits, each task's
first rollout saves its result from the job's metrics; when every train task of a generation has a result,
the next job sends the generation's results to the generator.

The first round starts when the pipeline first looks for job data. The next round starts after
``jobs_per_round`` jobs' data were acknowledged since the previous round started, and only while the task
pool holds fewer than ``refill_tasks`` tasks. A round that made no train task, a rollout that trained
nothing, and an empty task pool with no tasks handed out are followed at once. A restart continues from the
task store: train tasks that did not train go back to the task pool.

The record stage trains on no report. It holds the reports that are source sessions for the rounds, and
releases the others, such as a generator's own reports.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import tempfile
import uuid
from collections import Counter
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord
from reef.core.config import config_option
from reef.core.job_data import JobOutcome, TaskItem
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.core.tasks import HarborTask, read_harbor_task, write_harbor_task
from reef.core.tasks.split import TaskSplit, TaskSplitConflict, extend_split_by_source
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks.generator import GeneratedTaskResult, HeldOutTask, RejectedTask, TaskGenerator
from reef.recipe.generated_tasks.results import (
    TaskRolloutResult,
    delivered_generations,
    mark_delivered,
    rollout_results,
    save_rollout_result,
)
from reef.recipe.generated_tasks.rollout import (
    HandedOutTasks,
    PooledTask,
    SubmittedTaskJob,
    TaskPool,
    consumed_rollout_id,
)
from reef.record2record import (
    Record2Record,
    Record2RecordSettings,
    RecordSet,
    ReportContext,
    ReportGrouping,
    ReportRecord2Record,
)
from reef.record2task import Record2Task, TaskResult
from reef.record2task.generator.client import Generator, HttpGenerator
from reef.record2task.generator.inputs import TaskGenerationRequest
from reef.storage.tasks import GenerationState, SplitName, StoredTask, TaskConflict, TaskStore, TaskStoreError
from reef.task2job import Task2Job
from reef.train.types import JobData, TrainDataItem

logger = logging.getLogger(__name__)

DEFAULT_TASKS_PER_STEP = 4
DEFAULT_JOBS_PER_ROUND = 1
TASK_STORE_NAME = "generated-tasks"
DEFAULT_EVAL_FRACTION = 0.0
DEFAULT_SPLIT_SEED = 0
DEFAULT_GENERATION_ATTEMPTS = 3
DEFAULT_GENERATION_DEADLINE_SECONDS = 1800.0
DELIVERY_RECEIPT = "delivery"


def split_source_ids(task: HarborTask) -> tuple[str, ...]:
    """The source record ids that place a task in a split: its sources without the author's own record."""
    designer_record_id = task.metadata.get("designer_record_id")
    return tuple(record_id for record_id in task.source_agent_record_ids if record_id != designer_record_id)


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


# ---------------------------------------------------------------- store lookups


def train_task_names(state: GenerationState) -> tuple[str, ...]:
    """The names of the train tasks that a generation published, in candidate order."""
    return tuple(candidate.name for candidate in state.candidates or () if candidate.split == "train")


def find_generation(store: TaskStore, source_record_ids: tuple[str, ...]) -> GenerationState | None:
    """Return the generation made from exactly these source records, if one exists.

    Without source records, return the newest incomplete description-only generation, so the next round finishes it.
    """
    if not source_record_ids:
        incomplete = [state for state in store.generations() if not state.source_record_ids and not state.is_complete]
        return max(incomplete, key=lambda state: state.generation, default=None)
    return next((state for state in store.generations() if state.source_record_ids == source_record_ids), None)


def completed_source_record_ids(store: TaskStore) -> frozenset[str]:
    """The source records of complete generations, which no new generation uses again."""
    return frozenset(
        record_id for state in store.generations() if state.is_complete for record_id in state.source_record_ids
    )


def next_generation(store: TaskStore) -> int:
    """The number after the highest saved generation, 0 for an empty store."""
    return max((state.generation for state in store.generations()), default=-1) + 1


def complete_generation_count(store: TaskStore) -> int:
    return sum(state.is_complete for state in store.generations())


def available_tasks(store: TaskStore) -> tuple[tuple[int, StoredTask], ...]:
    """The train tasks of complete generations that have not trained, with their generation, oldest first."""
    published = {stored.name: stored for stored in store.tasks(split="train")}
    trained = store.trained_task_names()
    return tuple(
        (state.generation, published[name])
        for state in store.generations()
        if state.is_complete
        for name in train_task_names(state)
        if name in published and name not in trained
    )


def deliverable_generations(store: TaskStore) -> tuple[GenerationState, ...]:
    """The deliverable generations, oldest first: ``complete(g)``, ``train(g) ⊆ dom rollout`` and ``g ∉ Δ``."""
    saved = rollout_results(store)
    delivered = delivered_generations(store)
    return tuple(
        state
        for state in store.generations()
        if state.is_complete
        and state.generation not in delivered
        and all(name in saved for name in train_task_names(state))
    )


def task_results(store: TaskStore, state: GenerationState) -> tuple[GeneratedTaskResult, ...]:
    """The result of each candidate of a complete generation, in candidate order, as the generator receives them.

    A rejected candidate is a ``RejectedTask``, an eval task a ``HeldOutTask``, and a train task the result of
    its first rollout, ``rollout(n)``.
    """
    saved = rollout_results(store)
    results: list[GeneratedTaskResult] = []
    for candidate in state.candidates or ():
        task = read_harbor_task(store.path(candidate.files_digest))
        if candidate.error:
            results.append(RejectedTask(task, candidate.error))
        elif candidate.split == "eval":
            results.append(HeldOutTask(task))
        else:
            results.append(saved[candidate.name].generated(task))
    return tuple(results)


def task_counts(store: TaskStore) -> dict[str, int]:
    """Count the published tasks of each split, the train tasks by training state, and the other candidates."""
    states = store.generations()
    complete = {name for state in states if state.is_complete for name in train_task_names(state)}
    trained = store.trained_task_names()
    train = store.tasks(split="train")
    splits = Counter(task.split for task in store.tasks())
    candidates = [candidate for state in states for candidate in state.candidates or ()]
    return {
        "train": splits["train"],
        "eval": splits["eval"],
        "trained": sum(task.name in trained for task in train),
        "available": sum(task.name in complete and task.name not in trained for task in train),
        "rejected": sum(bool(candidate.error) for candidate in candidates),
        "pending": sum(candidate.is_pending for candidate in candidates),
    }


def task_item(store: TaskStore, generation: int, stored: StoredTask) -> TaskItem:
    """The task that the task pool holds for a saved train task, with its generation and the digest of its files."""
    path = store.path(stored.files_digest)
    task = read_harbor_task(path)
    return TaskItem(
        path,
        source_agent_record_ids=task.source_agent_record_ids,
        metadata={
            "task": {
                "name": stored.name,
                "digest": task.digest,
                "files_digest": stored.files_digest,
                "round": generation,
                "split": stored.split,
            }
        },
    )


# ---------------------------------------------------------------- rounds and jobs


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
class TaskGenerationRound:
    """One task-generation round.

    Values: ``round_number`` is the number of the round; the rounds of one
    process have increasing numbers. ``sources`` holds at most
    ``source_records_per_round`` source sessions, in arrival order.
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

    Values: ``tasks`` are the train tasks that the round's generation saved
    and that have not trained, for the task pool. ``counts`` is False for a
    round that made nothing new, because a complete generation already used
    its sessions; such a round does not count toward the round limit.
    ``retry`` is True when the generation is incomplete, so the next round
    takes the same sessions again and finishes it.
    Purpose: ``GeneratedTaskTasks.finished_rounds`` makes it when a round ends.
    ``GeneratedTaskRecord2Record.advance`` reads it.
    """

    tasks: tuple[TaskItem, ...] = ()
    counts: bool = True
    retry: bool = False


@dataclass(frozen=True)
class GenerationJob(SupportsReceipt):
    """One generation to run.

    Values: ``receipt`` is ``generation-<generation>``. ``generation`` is a non-negative generation number of
    the task store. ``source_records`` holds the records of the round's source sessions that no complete
    generation used, each record once; it is empty for a description-only generation.
    Purpose: ``GeneratedTaskTasks.start_round`` makes it; its worker runs it in ``run_generation``.
    """

    generation: int
    source_records: tuple[AgentRecord, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")


@dataclass(frozen=True)
class GenerationOutcome(SupportsReceipt):
    """How one generation job ended.

    Values: ``receipt`` and ``generation`` are those of the job. ``is_complete`` says whether the generation
    is complete in the task store. A complete generation holds in ``tasks`` its published train tasks that
    have not trained, and its ``error`` is empty or names a delivery to the generator that failed. An
    incomplete generation ran out of attempts: it holds no task, and ``error`` names the last failure.
    Purpose: the worker hands it back from ``run``; ``GeneratedTaskTasks.finished_rounds`` turns it into a
    ``RoundResult``.
    """

    generation: int
    tasks: tuple[TaskItem, ...]
    is_complete: bool
    error: str

    def __post_init__(self) -> None:
        if not self.is_complete and (self.tasks or not self.error):
            raise ValueError("an incomplete generation holds no task and names its error")


@dataclass(frozen=True)
class DeliveryJob(SupportsReceipt):
    """A job that sends the generator the results it has not received.

    Values: ``receipt`` is ``DELIVERY_RECEIPT``.
    Purpose: ``GeneratedTaskTasks`` submits it at a start and after each task job's end; its worker runs it.
    """


@dataclass(frozen=True)
class DeliveryOutcome(SupportsReceipt):
    """How a delivery job ended.

    Values: ``receipt`` is ``DELIVERY_RECEIPT``. ``error`` is the error of the last generation whose delivery
    failed, or empty when every deliverable generation reached the generator.
    Purpose: the worker hands it back; ``GeneratedTaskTasks.finished_rounds`` reads ``error`` into its status.
    """

    error: str


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


class SourceSessionGrouping(ReportGrouping):
    """No report trains; a source session carries ``source_tag``.

    The generated-task recipe trains on its rollout episodes, which are not records. A report is either a
    source session for the rounds, whose inference records all carry ``source_tag``, or released.
    """

    def __init__(self, source_tag: tuple[str, str] | None = None) -> None:
        self.source_tag = source_tag

    def is_training_report(self, report: AgentRecord) -> bool:
        return False

    def is_source_report(self, context: ReportContext) -> bool:
        """Whether a report that does not train is a recorded session to generate tasks from."""
        if self.source_tag is None or not context.inferences:
            return False
        name, value = self.source_tag
        return all(inference_tags(inference).get(name) == value for inference in context.inferences)


@dataclass(frozen=True)
class TaskGroupSettings(Record2RecordSettings):
    """The config of ``GeneratedTaskRecord2Record``: the pace of the rounds, the source sessions, and the staleness limit.

    Values: ``jobs_per_round`` and ``refill_tasks`` are positive ints; ``source_records_per_round`` is a
    non-negative int, and above 0 it needs a ``source_tag`` of the form ``name=value``; ``max_staleness`` is
    as in ``Record2RecordSettings``. The constructor refuses other values.
    Purpose: ``GeneratedTaskRecord2Record.from_config`` parses it from ``pipeline.record2record.config``.
    """

    jobs_per_round: int = config_option(
        DEFAULT_JOBS_PER_ROUND, help="The finished task jobs after which the next round starts."
    )
    source_records_per_round: int = config_option(0, help="The source sessions that each round takes, at most.")
    source_tag: str = config_option("", help="The name=value tag that marks a source session.")
    refill_tasks: int = config_option(
        1, help="The next round starts only while the task pool holds fewer saved tasks than this."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.jobs_per_round < 1:
            raise ValueError("jobs_per_round must be positive")
        if self.refill_tasks < 1:
            raise ValueError("refill_tasks must be positive")
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
    eval_fraction: float = config_option(
        DEFAULT_EVAL_FRACTION, help="The fraction of new source groups that go to the eval split."
    )
    split_seed: int = config_option(DEFAULT_SPLIT_SEED, help="The fixed seed of the task store's train/eval split.")
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
        if isinstance(self.eval_fraction, bool) or not 0 <= self.eval_fraction <= 1:
            raise ValueError("eval_fraction must be between 0 and 1")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise ValueError("split_seed must be an integer")
        if self.rounds > 0 and not self.description.strip():
            raise ValueError("rounds need a description (what the tasks are about)")
        if self.generator is not None:
            TaskGenerator.reference(self.generator, setting="pipeline.record2task.config.generator")


@dataclass(frozen=True)
class TaskJobSettings(PluginSettings):
    """The config of ``GeneratedTaskJobData``: ``tasks_per_step``, a positive int, the saved tasks of one task job.

    The rollout of the tasks is the training component's: its ``rollout`` job parameter sets the episodes of
    each task and the agent that runs them.
    """

    tasks_per_step: int = config_option(DEFAULT_TASKS_PER_STEP, help="The saved tasks of one task job.")

    def __post_init__(self) -> None:
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")


class GeneratedTaskRecord2Record(ReportRecord2Record):
    """The generated-task record stage: the source sessions and the pace of the rounds.

    Config: ``jobs_per_round``, ``source_records_per_round`` (the source sessions that each request takes, at most), ``source_tag``
    (``name=value``; required when ``source_records_per_round`` is above 0), ``refill_tasks``, and
    ``max_staleness``. A terminal report owns its sources. No report trains: the training component trains on
    the episodes of the task jobs, which are not records.

    The stage starts no round until ``resume_rounds`` gives it the record-to-task stage that runs the rounds,
    the next round number, the complete generations of the task store, and the round limit. The limit counts
    the rounds that finished a generation and the rounds whose generation ran out of attempts; a round of
    sessions that a complete generation already used does not count. A restart counts complete generations
    only, so a generation that ran out of attempts gets new attempts after a restart. A round is due when no
    round is running, the limit is not reached, the task pool that ``attach_task_pool`` gave holds fewer than
    ``refill_tasks`` tasks, and one of these is true:

    * it is the first round after a start;
    * the last round made no train task, or the last task job failed, so no job data comes;
    * ``jobs_per_round`` task jobs finished since the last round started;
    * the task pool is empty and no task job holds a task, so no job data can come.

    At the limit, the stage stops the rounds and holds no more source sessions. A source session is a report
    that ``SourceSessionGrouping.is_source_report`` accepts, with its inference records.
    The stage holds source sessions in arrival order, and each round takes at most
    ``source_records_per_round`` of them. A round whose generation is incomplete gives its sessions to the
    next round. The stage holds the records of a round's sources until the round ends.
    """

    def __init__(self, context: PipelineContext, settings: TaskGroupSettings) -> None:
        self.source_grouping = SourceSessionGrouping(source_tag_of(settings.source_tag))
        super().__init__(context, self.source_grouping, exclusive_sources=True, max_staleness=settings.max_staleness)
        #: The task pool, which the task-to-job stage owns; a round's train tasks join it.
        self.task_pool: TaskPool | None = None
        self.jobs_per_round = settings.jobs_per_round
        self.sources_per_round = settings.source_records_per_round
        self.refill_tasks = settings.refill_tasks
        self.runner: GeneratedTaskTasks | None = None
        self.next_round = 0
        self.completed_rounds = 0
        self.round_limit = 0
        self.running_round: TaskGenerationRound | None = None
        # The first round after a start is due at once.
        self.follow_at_once = True
        self.rounds_stopped = False
        # Task jobs that finished since the last round started; the next round is due at jobs_per_round.
        self.trained_jobs = 0
        self.waiting_sources: list[ReportContext] = []
        #: The sessions of a round whose generation is incomplete; the next round takes exactly these.
        self.retry_sources: tuple[ReportContext, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(context, TaskGroupSettings.parse(config, plugin_class=cls))

    def attach_task_pool(self, task_pool: TaskPool) -> None:
        """Take the task pool of the stage that hands tasks out; the rounds' train tasks join it."""
        self.task_pool = task_pool

    def resume_rounds(
        self, runner: GeneratedTaskTasks, *, next_round: int, completed_rounds: int, round_limit: int
    ) -> None:
        """Set the runner, the next round number, the rounds that finished a generation, and the round limit."""
        self.runner = runner
        self.next_round = next_round
        self.completed_rounds = completed_rounds
        self.round_limit = round_limit

    def handle_other_report(self, report: AgentRecord) -> None:
        # The report ends here. releasable_record_ids holds back a session's records until its round ends.
        super().handle_other_report(report)
        if self.sources_per_round == 0 or self.rounds_stopped:
            return
        if any(reference not in self.inferences for reference in report.references):
            return
        context = self.report_context(report, None)
        if self.source_grouping.is_source_report(context):
            self.waiting_sources.append(context)

    def round_due(self) -> bool:
        """Whether the next round should start now."""
        pool = self.task_pool
        if pool is None or self.rounds_stopped or self.running_round is not None or len(pool) >= self.refill_tasks:
            return False
        if self.follow_at_once or self.trained_jobs >= self.jobs_per_round:
            return True
        # No task waits and no task job holds one, so no job data comes to count toward the next round.
        return pool.is_idle

    def advance(self, now: float) -> None:
        """End the rounds that finished, then start the next round when it is due, or stop the rounds at the limit."""
        super().advance(now)
        if self.runner is None:
            return
        for result in self.runner.finished_rounds():
            finished = self.running_round
            self.running_round = None
            self.follow_at_once = not result.tasks
            if result.counts:
                self.completed_rounds += 1
            if result.retry and finished is not None and not self.rounds_stopped:
                self.retry_sources = finished.sources
            if result.tasks and self.task_pool is None:
                raise RuntimeError("a round's train tasks need the task pool of the stage that hands them out")
            for task in result.tasks:
                if self.task_pool is not None:
                    self.task_pool.add(task.metadata["task"]["name"], task, group_key=task.metadata["task"]["round"])
        if not self.round_due():
            return
        if self.completed_rounds >= self.round_limit:
            self.stop_rounds()
            return
        sources = self.retry_sources or tuple(self.waiting_sources[: self.sources_per_round])
        generation_round = TaskGenerationRound(round_number=self.next_round, sources=sources)
        if not self.runner.start_round(generation_round):
            self.stop_rounds()
            return
        if self.retry_sources:
            self.retry_sources = ()
        else:
            self.waiting_sources = self.waiting_sources[self.sources_per_round :]
        self.running_round = generation_round
        self.next_round += 1
        self.follow_at_once = False
        # The task jobs of the round's tasks count toward the next one.
        self.trained_jobs = 0

    def task_job_ended(self, *, finished: bool) -> None:
        """A task job ended: one that finished counts toward the next round; after one that failed, no job
        data comes to wait for, so the next round may follow."""
        if finished:
            self.trained_jobs += 1
        else:
            self.follow_at_once = True

    def stop_rounds(self) -> None:
        """No round runs again, so no session waits for one."""
        self.rounds_stopped = True
        self.waiting_sources = []
        self.retry_sources = ()

    def releasable_record_ids(self) -> frozenset[str]:
        held = {
            record_id
            for context in (*self.retry_sources, *self.waiting_sources)
            for record_id in (*context.references, context.report.agent_record_id)
        }
        if self.running_round is not None:
            held |= self.running_round.record_ids()
        return super().releasable_record_ids() - held

    def background_jobs_running(self) -> bool:
        """Return True while a round runs: its end can start the next round without a new record."""
        return self.running_round is not None or super().background_jobs_running()

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {**super().operational_metrics(), "waiting_source_sessions": len(self.waiting_sources)}


class GeneratedTaskTasks(Record2Task, BackgroundJobRunner):
    """The record-to-task stage of generated tasks: the runner of the generations, and of each task's episodes.

    The record stage forms no record set, so the stage makes no task from records.

    The record stage starts a round when it is due (``start_round``). The stage runs the round's generation
    on its worker: it generates tasks from the round's description and source sessions, validates them and
    saves them in the task store ``generated-tasks`` of the scenario session; it runs no episodes. A generator or
    check that fails to run is retried up to ``generation_attempts`` times within
    ``generation_deadline_seconds``. ``finished_rounds`` hands back each round with its saved train tasks.

    ``GeneratedTaskJobData`` saves a task's first rollout result through ``save_result``, ``rollout(n)`` of
    :mod:`reef.recipe.generated_tasks.results`, when the task's job ends. When every train task of a
    generation has a result, the next job sends the generation's results to the generator. A restart sends the results that the generator did not receive.
    """

    def __init__(
        self,
        context: PipelineContext,
        settings: RoundSettings,
        round_stage: GeneratedTaskRecord2Record,
        *,
        task_generator: TaskGenerator | None = None,
        generator_service: Generator | None = None,
        worker: BackgroundJobWorker | None = None,
    ) -> None:
        super().__init__()
        #: The record stage that paces the rounds.
        self.round_stage = round_stage
        self.scenario = context.scenario
        self.rounds = settings.rounds
        self.description = settings.description
        self.generation_attempts = settings.generation_attempts
        self.generation_deadline_seconds = settings.generation_deadline_seconds
        if self.rounds > 0:
            if task_generator is None:
                raise ValueError("rounds need a task generator: set pipeline.record2task.config.generator")
            if generator_service is None:
                generator_url = settings.generator_url
                if not generator_url or "${" in generator_url:
                    raise ValueError(
                        "rounds need generator_url: the generator service's address, ${endpoints.generator} "
                        "when the deployment has a generator section"
                    )
                generator_service = HttpGenerator(generator_url)
            if context.task_storage is None:
                raise ValueError("rounds need the task storage of a scenario session")
        self.task_generator = task_generator
        self.generator_service = generator_service
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
        #: Rollout results that the store could not save; the next job saves them before it delivers.
        self.unsaved_results: queue.SimpleQueue[TaskRolloutResult] = queue.SimpleQueue()
        #: Rounds that ended without a job, such as a round of sessions that a complete generation already used.
        self.ended_rounds: list[RoundResult] = []
        #: The generation that runs on the worker. One generation runs at a time.
        self.generation_in_flight: int | None = None
        self.last_error = ""
        store = self.task_store
        round_stage.resume_rounds(
            self,
            next_round=next_generation(store) if store is not None else 0,
            completed_rounds=complete_generation_count(store) if store is not None else 0,
            round_limit=self.rounds,
        )
        # A trained task without a result waits for the end of its task job (GeneratedTaskJobData.job_ended).
        # Results that a stop kept from the generator go out now, even when no other job is due.
        if store is not None and self.worker is not None and deliverable_generations(store):
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

    def push(self, record_set: RecordSet) -> None:
        raise TypeError(f"{type(self).__name__} makes no tasks from records: its record stage forms no record set")

    def pull(self) -> tuple[tuple[Hashable, TaskResult], ...]:
        return ()

    def start_round(self, generation_round: TaskGenerationRound) -> bool:
        """Submit the round's generation to the worker.

        A round of sessions that a complete generation already used ends at once and makes nothing new. A
        worker that refuses the round is closed or broken, so no round runs again: the result is False.
        """
        if self.task_store is None:
            raise RuntimeError("a round needs the task storage of a scenario session")
        records = session_records(generation_round.sources)
        completed = completed_source_record_ids(self.task_store)
        remaining = tuple(record for record in records if record.agent_record_id not in completed)
        # A restart replays sessions that a complete generation already used; their tasks are saved.
        if records and not remaining:
            self.ended_rounds.append(RoundResult(counts=False))
            return True
        existing = find_generation(self.task_store, tuple(record.agent_record_id for record in remaining))
        generation = next_generation(self.task_store) if existing is None else existing.generation
        job = GenerationJob(receipt=f"generation-{generation:05d}", generation=generation, source_records=remaining)
        if self.worker is None or not self.worker.submit(job):
            self.last_error = "the round worker is closed or broken; no further round runs"
            return False
        self.generation_in_flight = generation
        return True

    def finished_rounds(self) -> tuple[RoundResult, ...]:
        """Return each round that ended since the last call."""
        finished, self.ended_rounds = list(self.ended_rounds), []
        if self.worker is None:
            return tuple(finished)
        for outcome in self.worker.poll():
            if outcome.receipt == DELIVERY_RECEIPT:
                self.last_error = (
                    outcome.error if isinstance(outcome, DeliveryOutcome) else "the delivery did not finish"
                )
                continue
            self.generation_in_flight = None
            if isinstance(outcome, Failed):
                self.last_error = f"{outcome.receipt} did not finish; see the service log"
                logger.error("generated-task %s failed on scenario %r", outcome.receipt, self.scenario)
                finished.append(RoundResult(retry=True))
                continue
            if not isinstance(outcome, GenerationOutcome):
                raise TypeError("the round worker must hand back a GenerationOutcome")
            self.last_error = outcome.error
            logger.info(
                "generated-task generation %d on scenario %r: %d train tasks",
                outcome.generation,
                self.scenario,
                len(outcome.tasks),
            )
            finished.append(RoundResult(tasks=outcome.tasks, retry=not outcome.is_complete))
        return tuple(finished)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {"generation_in_flight": 0 if self.generation_in_flight is None else 1}

    def status(self) -> Mapping[str, Any]:
        store = self.task_store
        return {
            "task_store": task_counts(store) if store is not None else {},
            "task_generation": {
                "generator": None if self.task_generator is None else type(self.task_generator).__name__,
                "in_flight": self.generation_in_flight,
                "next": self.round_stage.next_round,
                "completed": self.round_stage.completed_rounds,
                "of": self.rounds,
                "last_error": self.last_error,
            },
        }

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Run one job on the worker, after sending the generator any results that it has not received."""
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("a job needs a task generator and a task store")
        self.save_unsaved_results()
        delivery_error = await self.deliver_results()
        if isinstance(job, DeliveryJob):
            return DeliveryOutcome(receipt=job.receipt, error=delivery_error)
        if isinstance(job, GenerationJob):
            generation = await self.run_generation(job)
            # A generation without train tasks has its complete results now.
            delivery_error = await self.deliver_results() or delivery_error
            return replace(generation, error=generation.error or delivery_error)
        raise TypeError(f"the round worker takes GenerationJob or DeliveryJob, not {type(job).__name__}")

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
            tasks = tuple(
                task_item(self.task_store, generation, stored)
                for generation, stored in available_tasks(self.task_store)
                if generation == job.generation
            )
            return GenerationOutcome(
                receipt=job.receipt, generation=job.generation, tasks=tasks, is_complete=True, error=""
            )
        return GenerationOutcome(
            receipt=job.receipt,
            generation=job.generation,
            tasks=(),
            is_complete=False,
            error=error or "the generation deadline passed",
        )

    async def generated(self, job: GenerationJob) -> None:
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
            with tempfile.TemporaryDirectory(prefix="reef-candidates-") as staging:
                # One directory for each candidate, because two candidates can have the same name.
                directories = tuple(
                    write_harbor_task(task, Path(staging) / str(index)) for index, task in enumerate(generated)
                )
                generation = self.task_store.save_candidates(generation=job.generation, directories=directories)
        candidates = generation.candidates or ()
        for index in generation.pending_indices:
            candidate = candidates[index]
            # The store would take the same files again as the earlier publication, which this generation
            # does not own, so its results would never be complete.
            if any(stored.files_digest == candidate.files_digest for stored in self.task_store.tasks()):
                refusal = f"task {candidate.name!r} is already published"
                self.task_store.reject_task(generation=job.generation, index=index, reason=refusal)
                continue
            task = read_harbor_task(self.task_store.path(candidate.files_digest))
            validation = await self.task_generator.validate(task)
            if validation.is_valid:
                try:
                    split = assigned_split(
                        self.task_store, task, seed=self.split_seed, eval_fraction=self.eval_fraction
                    )
                    self.task_store.publish_task(generation=job.generation, index=index, split=split)
                except (TaskConflict, TaskSplitConflict) as conflict:
                    self.task_store.reject_task(generation=job.generation, index=index, reason=str(conflict))
            else:
                self.task_store.reject_task(
                    generation=job.generation, index=index, reason="; ".join(validation.errors)
                )
        self.task_store.complete_generation(generation=job.generation)

    def save_result(self, result: TaskRolloutResult) -> None:
        """Save ``rollout(n)`` unless the task has a result.

        A result that the store fails to save waits for the next job.
        """
        if self.task_store is None:
            raise RuntimeError("a result needs a task store")
        try:
            if result.task_name not in rollout_results(self.task_store):
                save_rollout_result(self.task_store, result)
        except TaskStoreError:
            logger.exception("the result of generated-task %s conflicts with the task store", result.task_name)
        except Exception:
            # The task still trained; the next job's end saves the result again before it delivers.
            logger.exception(
                "the result of generated-task %s was not saved on scenario %r", result.task_name, self.scenario
            )
            self.unsaved_results.put(result)

    def save_unsaved_results(self) -> None:
        """Save the results that an earlier job's end could not save; one that still fails waits for the next job."""
        unsaved: list[TaskRolloutResult] = []
        while not self.unsaved_results.empty():
            unsaved.append(self.unsaved_results.get_nowait())
        for result in unsaved:
            self.save_result(result)

    def deliver_later(self) -> None:
        """Send the generator the results that it has not received, on the worker, after the jobs before it."""
        if self.worker is not None:
            self.worker.submit(DeliveryJob(receipt=DELIVERY_RECEIPT))

    async def deliver_results(self) -> str:
        """Send each generation whose train tasks all have results to the generator, oldest first.

        Returns the error of the last delivery that failed; a failed generation waits for the next job, and later
        generations still go out.
        """
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("delivering results needs a task generator and a task store")
        error = ""
        for state in deliverable_generations(self.task_store):
            try:
                await self.task_generator.record_results(
                    task_results(self.task_store, state), generation=state.generation
                )
            except Exception as exc:
                logger.exception("the task generator did not take the results of generation %d", state.generation)
                error = str(exc) or type(exc).__name__
                continue
            mark_delivered(self.task_store, state.generation)
        return error


class GeneratedTaskJobData(Task2Job):
    """Hand saved tasks to the training component as task jobs, and save each task's result when its job ends.

    The stage owns the task pool: the record stage adds each round's train tasks to it, and a restart puts the
    train tasks that did not train back in it. On each ``advance`` with no tasks handed out, the stage takes up
    to ``tasks_per_step`` of them as the next job's data: the ``TaskItem`` values themselves. The training
    component rolls them out on its own sampler and trains on their episodes (a task job; see the Tinker
    component's ``rollout`` job parameter). When the trainer commits the job's submission, the commit lists
    ``rollout:<task use>``; the store marks the tasks trained once that commit is durable. The tasks stay
    taken in the pool until the job ends.

    When the job's end is durable, ``job_ended`` gives a result to each trained task without one: its rewards
    in the job's ``rollout_rewards`` metric, or an infrastructure failure. One task job runs at a time, so
    such a task belongs to the ended job, or to an earlier job that a rollback cancelled. A second call for
    the same end saves nothing new. The record-to-task stage then sends the finished generations' results to
    the generator.
    """

    #: The tasks wait in the task pool, not in a buffer, and a job trains ``tasks_per_step`` tasks.
    trains_own_job_data_only = True

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
        settings = TaskJobSettings.parse(config, plugin_class=cls)
        if not isinstance(record2record, GeneratedTaskRecord2Record) or not isinstance(
            record2task, GeneratedTaskTasks
        ):
            raise PluginConfigError(
                f"{cls.__name__} needs GeneratedTaskRecord2Record and GeneratedTaskTasks, "
                f"not {type(record2record).__name__} and {type(record2task).__name__}"
            )
        return cls(context, settings, record2record=record2record, record2task=record2task)

    def __init__(
        self,
        context: PipelineContext,
        settings: TaskJobSettings,
        *,
        record2record: GeneratedTaskRecord2Record,
        record2task: GeneratedTaskTasks,
    ) -> None:
        super().__init__(context, None, None)
        self.record2record = record2record
        self.record2task = record2task
        self.tasks_per_step = settings.tasks_per_step
        self.task_pool = TaskPool()
        record2record.attach_task_pool(self.task_pool)
        store = record2task.task_store
        if store is not None:
            for generation, stored in available_tasks(store):
                self.task_pool.add(stored.name, task_item(store, generation, stored), group_key=generation)
        #: The tasks of the next job, from hand-out until the trainer commits the job's submission.
        self.handed_out: HandedOutTasks | None = None
        #: The submitted jobs whose submission's commit is not yet durable. Their tasks are marked trained when
        #: one of the IDs that the commit lists is released.
        self.trained_rollouts: list[SubmittedTaskJob] = []
        #: The tasks of the jobs submitted in this process that did not end; they leave the pool at the end.
        self.submitted_tasks: tuple[PooledTask, ...] = ()

    def make_job_data(self, tasks: tuple[TrainDataItem, ...], data_id: str) -> JobData:
        return JobData(data_id, tasks)

    def advance(self) -> None:
        """Hand out the next job's tasks when none are handed out."""
        if self.handed_out is not None:
            return
        taken = self.task_pool.take(self.tasks_per_step)
        if not taken:
            return
        batch = f"{self.scenario}:rollout:{uuid.uuid4().hex[:12]}"
        task_uses = tuple(f"{batch}/{task_name_of(task)}" for task in taken)
        self.handed_out = HandedOutTasks(batch, task_uses, taken)

    def own_job_data(self) -> tuple[TrainDataItem, ...] | None:
        if self.handed_out is None:
            return None
        return tuple(task.item for task in self.handed_out.tasks)

    def own_job_data_submitted(self) -> frozenset[str]:
        handed_out = self.handed_out
        if handed_out is None:
            raise RuntimeError("no tasks were handed out")
        ids = frozenset(consumed_rollout_id(task_use) for task_use in handed_out.task_uses)
        self.trained_rollouts.append(SubmittedTaskJob(tuple(task_name_of(task) for task in handed_out.tasks), ids))
        self.submitted_tasks = (*self.submitted_tasks, *handed_out.tasks)
        self.handed_out = None
        return ids

    def release_own_ids(self, ids: frozenset[str]) -> None:
        durable = [job for job in self.trained_rollouts if job.consumed_ids & ids]
        for job in durable:
            if self.record2task.task_store is not None:
                self.record2task.task_store.mark_trained(task_names=job.task_names)
        self.trained_rollouts = [job for job in self.trained_rollouts if job not in durable]

    def job_ended(self, outcome: JobOutcome) -> None:
        """Save a result for each trained task without one: its rewards in the job's ``rollout_rewards``, or why
        no episode ran."""
        rewards = outcome.metrics.get("rollout_rewards", {})
        if not isinstance(rewards, Mapping):
            raise TypeError("a task job's rollout_rewards metric must map task names to rewards")
        error = outcome.error or "no episode of the task ran"
        store = self.record2task.task_store
        if store is not None:
            trained = store.trained_task_names()
            saved = rollout_results(store)
            for state in store.generations():
                for name in train_task_names(state):
                    if name not in trained or name in saved:
                        continue
                    values = tuple(rewards.get(name) or ())
                    if values:
                        self.record2task.save_result(TaskRolloutResult(name, values))
                    else:
                        self.record2task.save_result(TaskRolloutResult(name, error=error, infrastructure_failure=True))
        if self.submitted_tasks:
            self.task_pool.drop(self.submitted_tasks)
            self.submitted_tasks = ()
        self.record2record.task_job_ended(finished=outcome.event == "finished")
        self.record2task.deliver_later()

    def status(self) -> Mapping[str, Any]:
        handed_out = self.handed_out
        return {
            "task_pool": len(self.task_pool),
            "handed_out": None if handed_out is None else {"batch": handed_out.batch, "tasks": len(handed_out.tasks)},
            "submitted_tasks": len(self.submitted_tasks),
        }


def task_name_of(task: PooledTask) -> str:
    """The name of a pooled generated task."""
    return str(task.item.metadata["task"]["name"])
