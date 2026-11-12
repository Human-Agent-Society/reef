"""The generated-task stages: tasks generated and saved here, saved as task items, and trained as task jobs.

When a round is due, ``GeneratedTaskSamples`` makes a ``TaskGenerationRound`` and runs it as one generation job on
a private worker, off the trainer's thread. A generation job:

1. Asks the generator for tasks, with the round's source sessions when the round has any.
2. Saves the candidates in the scenario's task store, and validates each task with the generator.
3. Publishes each valid task with its train/eval split. Eval tasks run no rollouts.

The stage hands each train task of a complete generation to the pipeline
(``made_samples``), which saves it as a task item that the scenario uses (``stage_item_id``). The
tasks then train like any other sample: ``GeneratedTaskJobData`` makes one task job of the next job's
tasks, and the commit that submits the job counts them trained. The training component rolls the
tasks out on its own sampler and trains on their rollouts; Reef runs no rollout of them. When the job's end
commits, the stage saves the result of each task that trained and has none, from the job's metrics; when
every train task of a generation has a result, the next job sends the generation's results to the generator.

A train task waits until a committed job trains it, and is unfinished until it has a result. The first round
starts when the pipeline first looks for job data. The next round starts after ``jobs_per_round`` task jobs
ended since the previous round started, and only while fewer than ``refill_tasks`` train tasks wait. A round
that made no train task, a failed task job, and no unfinished task are followed at once. A restart continues from
the task store and the scenario's items: the scenario still uses the train tasks that did not train, and a later job takes
them again.

With ``designer_delay`` d above 0, the task jobs also train the Designer's replies. When the generator has taken the
results of generation g, it gives the advantage of each Designer reply of g that trains
(``TaskGenerator.designer_advantages``). The delivery makes the delayed sample of each such reply from the record
that the stage holds, and saves the replies as ``designer/<g>`` before it marks g delivered. The replies of g train in
the first task job that holds a task of round g + d, beside its task rollouts (``JobData.delayed``). A round that
published no train task passes them to the next round that did. The tasks of that round wait until g is delivered.
``GeneratedTaskJobData`` reads the replies from the task store, so control builds the job data in its own process.

The stage trains on no report. It holds the reports that are source sessions for the rounds, and releases the
others, such as a generator's own reports.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from reef.core import AgentRecord
from reef.core.config import config_option
from reef.core.job_data import DelayedSample, JobOutcome, TaskSample
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask, read_harbor_task, write_harbor_task
from reef.core.tasks.split import TaskSplit, TaskSplitConflict, extend_split_by_source
from reef.execution.jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks.generator import (
    GeneratedTaskResult,
    GeneratorState,
    HeldOutTask,
    RejectedTask,
    TaskGenerator,
)
from reef.recipe.generated_tasks.results import (
    DesignerReplies,
    TaskRolloutResult,
    checked_advantages,
    delivered_generations,
    designer_replies,
    has_designer_replies,
    mark_delivered,
    rollout_results,
    save_designer_replies,
    save_rollout_result,
)
from reef.record2sample import Record2Sample, Record2SampleSettings, ReportContext, ReportGrouping, ReportSamples
from reef.record2sample.generator.client import Generator, HttpGenerator
from reef.record2sample.generator.inputs import TaskGenerationRequest
from reef.record2sample.reports import report_score
from reef.record2sample.samples import make_policy_trajectory
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.storage.samples import StoredSample, TrainerSamples
from reef.storage.tasks import GenerationState, SplitName, StoredTask, TaskConflict, TaskStore, TaskStoreError
from reef.train.types import JobData, Sample

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_SIZE = 4
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


def published_train_tasks(store: TaskStore) -> tuple[tuple[int, StoredTask], ...]:
    """The published train tasks of complete generations, with their generation, oldest first."""
    published = {stored.name: stored for stored in store.tasks(split="train")}
    return tuple(
        (state.generation, published[name])
        for state in store.generations()
        if state.is_complete
        for name in train_task_names(state)
        if name in published
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


def task_counts(store: TaskStore, trained: frozenset[str]) -> dict[str, int]:
    """Count the published tasks of each split, the train tasks by training state, and the other candidates.

    ``trained`` holds the names of the train tasks that a committed job trained.
    """
    states = store.generations()
    complete = {name for state in states if state.is_complete for name in train_task_names(state)}
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


def task_item(store: TaskStore, generation: int, stored: StoredTask) -> TaskSample:
    """The sample of a published train task, with its generation and the digest of its files."""
    path = store.path(stored.files_digest)
    task = read_harbor_task(path)
    return TaskSample(
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


def task_round(sample: Sample) -> int | None:
    """The generation of a generated train task, its metadata ``task.round``; None for any other sample."""
    if not isinstance(sample, TaskSample):
        return None
    task = sample.metadata.get("task")
    generation = task.get("round") if isinstance(task, Mapping) else None
    return generation if isinstance(generation, int) and not isinstance(generation, bool) else None


def carrier_round(generation: int, train_rounds: tuple[int, ...], designer_delay: int) -> int | None:
    """The round whose first task job carries the Designer replies of ``generation``, or None while there is none.

    ``train_rounds`` holds the complete generations that published a train task, in increasing order. The carrier
    is the first of them at or after ``generation + designer_delay``, so a round without a train task passes the
    replies on.
    """
    return next((number for number in train_rounds if number >= generation + designer_delay), None)


def carried_generations(round_number: int, train_rounds: tuple[int, ...], designer_delay: int) -> tuple[int, ...]:
    """The generations whose Designer replies the first task job of ``round_number`` carries, in increasing order:
    the generations of ``train_rounds`` whose carrier round (``carrier_round``) is ``round_number``."""
    return tuple(
        generation
        for generation in train_rounds
        if carrier_round(generation, train_rounds, designer_delay) == round_number
    )


def train_rounds_of(store: TaskStore) -> tuple[int, ...]:
    """The complete generations that published a train task, in increasing order."""
    return tuple(sorted({generation for generation, _ in published_train_tasks(store)}))


def stored_tasks(samples: TrainerSamples) -> dict[str, StoredSample]:
    """The generated tasks that the scenario's items hold for the trainer of ``samples``, by task name: the tasks that
    its generated-task stage made."""
    trainer = samples.trainer
    found: dict[str, StoredSample] = {}
    after_sequence = 0
    while page := samples.page(after_sequence=after_sequence, kind="task"):
        for stored in page:
            if stored.component == trainer:
                found[stored.sample.metadata["task"]["name"]] = stored
        after_sequence = page[-1].sequence
    return found


def trained_rounds(samples: TrainerSamples) -> frozenset[int]:
    """The rounds that have a generated task that a committed job of the trainer of ``samples`` trained."""
    return frozenset(
        number
        for stored in stored_tasks(samples).values()
        if stored.train_count(samples.trainer) > 0 and (number := task_round(stored.sample)) is not None
    )


def designer_record_ids(task: HarborTask) -> frozenset[str]:
    """The records that can hold the Designer reply that wrote ``task``: its ``designer_record_id`` and its sources.

    A task that the Designer wrote from the description alone has that reply as its only source.
    """
    designer_record_id = task.metadata.get("designer_record_id")
    named = {designer_record_id} if isinstance(designer_record_id, str) and designer_record_id else set()
    return frozenset({*task.source_agent_record_ids, *named})


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

    Values: ``sources`` holds at most ``source_records_per_round`` source
    sessions, in arrival order.
    Purpose: ``GeneratedTaskSamples`` makes it when a round is due.
    ``GeneratedTaskSamples`` runs it.
    """

    sources: tuple[ReportContext, ...]

    def record_ids(self) -> frozenset[str]:
        """Return the IDs of the records of the sources, which the stage holds while the round runs."""
        return frozenset(
            record_id for source in self.sources for record_id in (*source.references, source.report.agent_record_id)
        )


@dataclass(frozen=True)
class RoundResult:
    """How a round ended.

    Values: ``made_train_tasks`` says whether the round's generation published
    a train task. ``counts`` is False for a
    round that made nothing new, because a complete generation already used
    its sessions; such a round does not count toward the round limit.
    ``retry`` is True when the generation is incomplete, so the next round
    takes the same sessions again and finishes it.
    Purpose: ``GeneratedTaskSamples.finished_rounds`` makes it when a round ends.
    ``GeneratedTaskSamples.advance`` reads it.
    """

    made_train_tasks: bool = False
    counts: bool = True
    retry: bool = False


@dataclass(frozen=True)
class DesignerSampleCounts:
    """The Designer replies that deliveries saved with their delayed samples, and the named records that they skipped.

    Values: three non-negative ints. ``saved`` counts the replies saved with their delayed samples, ``uncaptured``
    the named records whose response holds no exact sample that a delayed sample takes, and ``missing`` the named
    records that the delivery did not hold. The constructor refuses other values.
    Purpose: ``GenerationRunner.deliver`` counts them in its outcome; ``GeneratedTaskSamples`` adds them up for its
    operational metrics.
    """

    saved: int = 0
    uncaptured: int = 0
    missing: int = 0

    def __post_init__(self) -> None:
        for count in (self.saved, self.uncaptured, self.missing):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"a count of Designer replies must be a non-negative int, got {count!r}")

    def __add__(self, other: DesignerSampleCounts) -> DesignerSampleCounts:
        return DesignerSampleCounts(
            self.saved + other.saved, self.uncaptured + other.uncaptured, self.missing + other.missing
        )


@dataclass(frozen=True)
class GenerationJob(SupportsReceipt):
    """One generation to run.

    Values: ``receipt`` is ``generation-<generation>``. ``generation`` is a non-negative generation number of
    the task store. ``source_records`` holds the records of the round's source sessions that no complete
    generation used, each record once; it is empty for a description-only generation. ``designer_records`` holds
    the Designer records that the job's deliveries make delayed samples of (``GeneratedTaskSamples.designer_records``).
    Purpose: ``GeneratedTaskSamples.start_round`` makes it; its worker runs it in ``run_generation``.
    """

    generation: int
    source_records: tuple[AgentRecord, ...] = ()
    designer_records: tuple[AgentRecord, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")


@dataclass(frozen=True)
class GenerationOutcome(SupportsReceipt):
    """How one generation job ended.

    Values: ``receipt`` and ``generation`` are those of the job. ``is_complete`` says whether the generation
    is complete in the task store. A complete generation holds in ``tasks`` its published train tasks, and its
    ``error`` is empty or names a delivery to the generator that failed. An
    incomplete generation ran out of attempts: it holds no task, and ``error`` names the last failure.
    ``undelivered`` holds the generations whose delivery failed in the job's last delivery, in increasing order.
    ``designer`` counts the Designer replies that the job's deliveries saved, and ``saved_record_ids`` names their
    held records, as in ``DeliveryOutcome``.
    Purpose: the worker hands it back from ``run``; ``GeneratedTaskSamples.finished_rounds`` turns it into a
    ``RoundResult``.
    """

    generation: int
    tasks: tuple[TaskSample, ...]
    is_complete: bool
    error: str
    undelivered: tuple[int, ...] = ()
    designer: DesignerSampleCounts = DesignerSampleCounts()
    saved_record_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.is_complete and (self.tasks or not self.error):
            raise ValueError("an incomplete generation holds no task and names its error")


@dataclass(frozen=True)
class DeliveryJob(SupportsReceipt):
    """A job that sends the generator the results it has not received.

    Values: ``receipt`` is ``DELIVERY_RECEIPT``. ``designer_records`` holds the Designer records that the delivery
    makes delayed samples of (``GeneratedTaskSamples.designer_records``).
    Purpose: ``GeneratedTaskSamples`` submits it at a start and after each task job's end; its worker runs it.
    """

    designer_records: tuple[AgentRecord, ...] = ()


@dataclass(frozen=True)
class DeliveryOutcome(SupportsReceipt):
    """How a delivery job ended.

    Values: ``receipt`` is ``DELIVERY_RECEIPT``. ``error`` is the error of the last generation whose delivery
    failed, or empty when every deliverable generation reached the generator. ``undelivered`` holds the
    generations whose delivery failed, in increasing order; it is empty when ``error`` is empty. ``designer`` counts
    the Designer replies that the delivery saved. ``saved_record_ids`` names the held records of the replies that
    the delivery saved, with or without a delayed sample: the task store keeps what trains of them.
    Purpose: the worker hands it back; ``GeneratedTaskSamples.finished_rounds`` reads ``error`` into its status,
    ``undelivered`` into the rounds whose tasks stop waiting for Designer replies, ``designer`` into its
    operational metrics, and ``saved_record_ids`` into the records that it releases.
    """

    error: str
    undelivered: tuple[int, ...] = ()
    designer: DesignerSampleCounts = DesignerSampleCounts()
    saved_record_ids: frozenset[str] = frozenset()


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

    The generated-task recipe trains on its rollouts, which are not records. A report is either a
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
class GeneratedTaskSettings(Record2SampleSettings):
    """The config of ``GeneratedTaskSamples``: the rounds, their pace, the source sessions, and the generations.

    Values:

    * ``rounds`` is a non-negative int; above 0 it needs a ``description``.
    * ``jobs_per_round`` and ``refill_tasks`` are positive ints.
    * ``source_records_per_round`` is a non-negative int; above 0 it needs a ``source_tag`` of the form
      ``name=value``.
    * ``generator`` is ``None`` or a ``TaskGenerator`` reference ``{module, config}``.
    * ``eval_fraction`` is a number in [0, 1]; ``split_seed`` an int.
    * ``generation_attempts`` and ``generation_deadline_seconds`` are positive.
    * ``designer_delay`` is a non-negative int; 0 trains no Designer reply.
    * ``max_staleness`` is as in ``Record2SampleSettings``.

    The constructor refuses other values.
    Purpose: ``GeneratedTaskSamples.from_config`` parses it from ``pipeline.record2sample.config``.
    """

    rounds: int = config_option(0, help="The generations to finish; 0 generates nothing.")
    jobs_per_round: int = config_option(
        DEFAULT_JOBS_PER_ROUND, help="The finished task jobs after which the next round starts."
    )
    source_records_per_round: int = config_option(0, help="The source sessions that each round takes, at most.")
    source_tag: str = config_option("", help="The name=value tag that marks a source session.")
    refill_tasks: int = config_option(
        1, help="The next round starts only while fewer saved tasks than this wait to train."
    )

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
    designer_delay: int = config_option(
        0,
        help=(
            "The rounds after which the Designer replies of a generation train, beside the task rollouts of "
            "the first job of that later round; 0 trains no Designer reply."
        ),
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
        if self.rounds < 0:
            raise ValueError("rounds must be non-negative")
        if self.generation_attempts < 1 or self.generation_deadline_seconds <= 0:
            raise ValueError("generation_attempts and generation_deadline_seconds must be positive")
        if isinstance(self.eval_fraction, bool) or not 0 <= self.eval_fraction <= 1:
            raise ValueError("eval_fraction must be between 0 and 1")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise ValueError("split_seed must be an integer")
        if (
            isinstance(self.designer_delay, bool)
            or not isinstance(self.designer_delay, int)
            or self.designer_delay < 0
        ):
            raise ValueError("designer_delay must be a non-negative integer")
        if self.rounds > 0 and not self.description.strip():
            raise ValueError("rounds need a description (what the tasks are about)")
        if self.generator is not None:
            TaskGenerator.reference(self.generator, setting="pipeline.record2sample.config.generator")


class StoredGeneratorState(GeneratorState):
    """The generator state ``(S, F)`` of one scenario in the task store of the generated-task stage.

    Values: ``store`` is a ``TaskStore``. ``S`` is the state documents of the
    store, and ``F`` is the directories that the store saved in the task
    files of the scenario.
    Purpose: ``GeneratedTaskSamples.from_config`` makes it for the generator of
    the scenario. The stage's own generations and documents are not in
    ``S``, so the generator does not read or change them.
    """

    def __init__(self, store: TaskStore) -> None:
        if not isinstance(store, TaskStore):
            raise TypeError("store must be a TaskStore")
        self.store = store

    def read(self, key: str) -> str | None:
        return self.store.state_document(key)

    def write(self, key: str, document: str) -> None:
        self.store.write_state_document(key=key, document=document)

    def remove(self, key: str) -> None:
        self.store.remove_state_document(key=key)

    def keys(self, prefix: str = "") -> tuple[str, ...]:
        return self.store.state_document_keys(prefix=prefix)

    def save_files(self, directory: Path) -> str:
        return self.store.save_files(directory)

    def files(self, digest: str) -> Path:
        return self.store.path(digest)


class GenerationRunner(BackgroundJobRunner):
    """Generate and publish tasks, save training results, and deliver completed generations.

    Values: ``scenario`` is non-empty text and ``description`` is text. ``task_store`` is a TaskStore or None;
    ``task_generator`` is a TaskGenerator or None. ``run``, ``generated`` and ``deliver`` require both
    dependencies and raise RuntimeError otherwise. ``run_generation`` requires the store; a missing generator
    exhausts its attempts and returns an incomplete outcome. Result saving requires the store alone.
    ``generation_attempts`` is a positive int.
    ``generation_deadline_seconds`` is a positive number, including infinity for no time limit.
    ``eval_fraction`` is a number in [0, 1], and
    ``split_seed`` is a non-bool int. ``designer_delay`` is a non-negative int; above 0, each delivery saves the
    generation's Designer replies (``designer/<g>``) before its delivery marker. ``unsaved_results`` is a FIFO
    queue of TaskRolloutResult values whose saves failed; it may be empty and contain repeated task names, since
    the store keeps only each task's first result.

    Purpose: GeneratedTaskSamples submits jobs to this runner on one background worker. The runner writes
    candidates, validation results, task publications, Designer replies and delivery markers. The stage reads
    them to schedule rounds and emit samples. The trainer's job-end path also calls save_result. Both paths write
    the shared task store through its existing operations. The retry queue is the runner's only mutable in-memory
    state shared between them. The scenario session owns and closes the store
    after the stage has stopped its worker.
    """

    def __init__(
        self,
        scenario: str,
        task_store: TaskStore | None,
        task_generator: TaskGenerator | None,
        *,
        description: str,
        generation_attempts: int,
        generation_deadline_seconds: float,
        eval_fraction: float,
        split_seed: int,
        designer_delay: int = 0,
    ) -> None:
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be non-empty text")
        if not isinstance(description, str):
            raise TypeError("description must be text")
        if task_store is not None and not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore or None")
        if task_generator is not None and not isinstance(task_generator, TaskGenerator):
            raise TypeError("task_generator must be a TaskGenerator or None")
        if not isinstance(generation_attempts, int) or generation_attempts < 1:
            raise ValueError("generation_attempts must be a positive integer")
        if not isinstance(generation_deadline_seconds, (int, float)) or not generation_deadline_seconds > 0:
            raise ValueError("generation_deadline_seconds must be positive")
        if (
            isinstance(eval_fraction, bool)
            or not isinstance(eval_fraction, (int, float))
            or not 0 <= eval_fraction <= 1
        ):
            raise ValueError("eval_fraction must be between 0 and 1")
        if isinstance(split_seed, bool) or not isinstance(split_seed, int):
            raise ValueError("split_seed must be an integer")
        if isinstance(designer_delay, bool) or not isinstance(designer_delay, int) or designer_delay < 0:
            raise ValueError("designer_delay must be a non-negative integer")
        self.scenario = scenario
        self.task_store = task_store
        self.task_generator = task_generator
        self.description = description
        self.generation_attempts = generation_attempts
        self.generation_deadline_seconds = generation_deadline_seconds
        self.eval_fraction = eval_fraction
        self.split_seed = split_seed
        self.designer_delay = designer_delay
        self.unsaved_results: queue.SimpleQueue[TaskRolloutResult] = queue.SimpleQueue()

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Run one job on the worker, after sending the generator any results that it has not received."""
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("a job needs a task generator and a task store")
        self.save_unsaved_results()
        designer_records = job.designer_records if isinstance(job, (GenerationJob, DeliveryJob)) else ()
        delivery = await self.deliver(designer_records)
        if isinstance(job, DeliveryJob):
            return delivery
        if isinstance(job, GenerationJob):
            generation = await self.run_generation(job)
            # A generation without train tasks has its complete results now.
            last_delivery = await self.deliver(designer_records)
            return replace(
                generation,
                error=generation.error or last_delivery.error or delivery.error,
                undelivered=last_delivery.undelivered,
                designer=delivery.designer + last_delivery.designer,
                saved_record_ids=delivery.saved_record_ids | last_delivery.saved_record_ids,
            )
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
                for generation, stored in published_train_tasks(self.task_store)
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

    async def deliver(self, designer_records: Sequence[AgentRecord] = ()) -> DeliveryOutcome:
        """Send each generation whose train tasks all have results to the generator, oldest first.

        A failed generation waits for the next job, and later generations still go out. With ``designer_delay``
        above 0, a generation that the generator took gets its Designer replies saved before its delivery marker
        (``save_designer_replies``), with delayed samples made of ``designer_records``. The outcome names the error
        of the last failure, the generations that the generator did not take, the replies saved and their records.
        """
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("delivering results needs a task generator and a task store")
        held = {record.agent_record_id: record for record in designer_records}
        error = ""
        undelivered: list[int] = []
        designer = DesignerSampleCounts()
        saved_record_ids: set[str] = set()
        for state in deliverable_generations(self.task_store):
            try:
                await self.task_generator.record_results(
                    task_results(self.task_store, state), generation=state.generation
                )
                if self.designer_delay > 0:
                    designer_error, counts, record_ids = self.save_designer_replies(state.generation, held)
                    error = designer_error or error
                    designer += counts
                    saved_record_ids |= record_ids
            except Exception as exc:
                logger.exception("the task generator did not take the results of generation %d", state.generation)
                error = str(exc) or type(exc).__name__
                undelivered.append(state.generation)
                continue
            mark_delivered(self.task_store, state.generation)
        return DeliveryOutcome(
            receipt=DELIVERY_RECEIPT,
            error=error,
            undelivered=tuple(undelivered),
            designer=designer,
            saved_record_ids=frozenset(saved_record_ids),
        )

    def save_designer_replies(
        self, generation: int, held: Mapping[str, AgentRecord]
    ) -> tuple[str, DesignerSampleCounts, frozenset[str]]:
        """Save ``designer(generation)`` unless the store has it: the advantages that the generator gives the
        generation's Designer replies, and the delayed sample of each reply, made of its record in ``held``.

        A generator that fails to give the advantages, or gives a value that is not a finite number by a record ID,
        trains no reply of the generation: the saved advantages are empty, and the result names the error. A reply
        whose record ``held`` lacks, or whose record holds no exact sample, gets no delayed sample and does not train;
        the counts of the result name both. The result also names the records of ``held`` whose replies it saved. A
        store that fails to save the replies raises, so the delivery fails and runs again later.
        """
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("Designer replies need a task generator and a task store")
        if has_designer_replies(self.task_store, generation):
            return "", DesignerSampleCounts(), frozenset()
        # The generator is a deployment's plugin: whatever it raises leaves the generation's replies untrained.
        try:
            advantages = checked_advantages(self.task_generator.designer_advantages(generation))
            error = ""
        except Exception as exc:
            logger.exception("the task generator gave no Designer advantages for generation %d", generation)
            advantages, error = {}, str(exc) or type(exc).__name__
        samples: list[DelayedSample] = []
        uncaptured = missing = 0
        for record_id, advantage in sorted(advantages.items()):
            record = held.get(record_id)
            if record is None:
                missing += 1
                continue
            try:
                samples.append(DelayedSample(make_policy_trajectory(record, advantage), advantage))
            except (TypeError, ValueError):
                uncaptured += 1
        save_designer_replies(self.task_store, generation, DesignerReplies(advantages, tuple(samples)))
        if uncaptured or missing:
            logger.warning(
                "generation %d of scenario %r trains %d of %d Designer replies: %d records hold no exact sample, "
                "and %d records were not held",
                generation,
                self.scenario,
                len(samples),
                len(advantages),
                uncaptured,
                missing,
            )
        saved_record_ids = frozenset(record_id for record_id in advantages if record_id in held)
        return error, DesignerSampleCounts(len(samples), uncaptured, missing), saved_record_ids


class GeneratedTaskSamples(ReportSamples, BackgroundJobRunner):
    """The generated-task stage: the source sessions, the pace of the rounds, and the source of their tasks.

    GenerationRunner owns generation execution, publication and result delivery.

    Config: ``GeneratedTaskSettings``. No report trains: the training component trains on the rollouts of the task
    jobs, which are not records, so the stage makes no sample from records. A terminal report owns its sources.

    **Rounds.** The round limit is ``rounds``. It counts the rounds that finished a generation and the rounds whose
    generation ran out of attempts; a round of sessions that a complete generation already used does not count. A
    start reads the next round number and the complete generations from the task store. A restart counts complete
    generations only, so a generation that ran out of attempts gets new attempts after a restart. A round is due
    when no round is running, the limit is not reached, fewer than ``refill_tasks`` train tasks wait to train
    (``waiting_task_count``), and one of these is true:

    * it is the first round after a start;
    * the last round made no train task, or the last task job failed, so no job data comes;
    * ``jobs_per_round`` task jobs finished since the last round started;
    * no train task is unfinished (``unfinished_task_count``), so no job end can come.

    At the limit, the stage stops the rounds and holds no more source sessions.

    **Source sessions.** A source session is a report that ``SourceSessionGrouping.is_source_report`` accepts, with
    its inference records. The stage holds source sessions in arrival order, and each round takes at most
    ``source_records_per_round`` of them. A round whose generation is incomplete gives its sessions to the next
    round. The stage holds the records of a round's sources until the round ends.

    **Generations.** The stage runs a due round's generation on its worker (``start_round``): it generates tasks
    from the round's description and source sessions, validates them and saves them in the task store
    ``generated-tasks`` of the scenario session; it runs no rollouts. A generator or check that fails to run is
    retried up to ``generation_attempts`` times within ``generation_deadline_seconds``. ``finished_rounds`` hands
    back each round with its published train tasks.

    **Tasks.** The stage hands each published train task of a complete generation to the pipeline
    (``made_samples``), which saves it as a task item; a start hands back each one that the scenario's items do
    not hold. A train task is unfinished until it has a result. When a job's end commits
    (``job_ended``), the stage saves ``rollout(n)`` of :mod:`reef.recipe.generated_tasks.results` for each task that
    a committed job trained and that has no result. When every train task of a generation has a result, the next
    job sends the generation's results to the generator. A restart sends the results that the generator did not
    receive.

    **Designer replies.** With ``designer_delay`` d above 0, each delivery of generation g also saves
    ``designer(g)`` of :mod:`reef.recipe.generated_tasks.results`: the generator's advantages of g's Designer
    replies, and the delayed sample of each reply, made of the record that the stage holds. The stage holds the
    records that can hold a Designer reply of g's train tasks (``held_designer_record_ids``) from the publication of
    g's train tasks until g is delivered, so a restart before that delivery reads them again. Then the record of each
    saved reply ends, whether or not a report owns it, unless a live report or a round still needs it
    (``saved_designer_records``). A job's deliveries
    get the held records (``designer_records``); after a job's end, the delivery waits for the next ``advance``,
    which comes after the samples service read the records. Let the carrier round of g be the first complete
    generation at or after g + d that published a train task (``carrier_round``). The stage hands the tasks of a
    round to the pipeline only when each generation that the round carries reached the generator, or failed to reach
    it since this start. ``GeneratedTaskJobData`` reads ``designer(g)`` into the job data of the carrier round's first
    task job.
    """

    def __init__(
        self,
        context: PipelineContext,
        settings: GeneratedTaskSettings,
        *,
        task_generator: TaskGenerator | None = None,
        generator_service: Generator | None = None,
        worker: BackgroundJobWorker | None = None,
    ) -> None:
        self.source_grouping = SourceSessionGrouping(source_tag_of(settings.source_tag))
        super().__init__(context, self.source_grouping, exclusive_sources=True, max_staleness=settings.max_staleness)
        self.jobs_per_round = settings.jobs_per_round
        self.sources_per_round = settings.source_records_per_round
        self.refill_tasks = settings.refill_tasks
        self.running_round: TaskGenerationRound | None = None
        # The first round after a start is due at once.
        self.follow_at_once = True
        self.rounds_stopped = False
        # Task jobs that finished since the last round started; the next round is due at jobs_per_round.
        self.trained_jobs = 0
        self.waiting_sources: list[ReportContext] = []
        #: The sessions of a round whose generation is incomplete; the next round takes exactly these.
        self.retry_sources: tuple[ReportContext, ...] = ()
        self.rounds = settings.rounds
        if self.rounds > 0:
            if task_generator is None:
                raise ValueError("rounds need a task generator: set pipeline.record2sample.config.generator")
            if generator_service is None:
                generator_url = settings.generator_url
                if not generator_url or "${" in generator_url:
                    raise ValueError(
                        "rounds need generator_url: the generator service's address, ${endpoints.generator} "
                        "when the deployment has a generator section"
                    )
                generator_service = HttpGenerator(generator_url)
            if context.task_storage is None or context.samples is None:
                raise ValueError("rounds need the task storage and the samples of a scenario session")
        self.task_generator = task_generator
        self.generator_service = generator_service
        self.worker = worker
        if self.task_generator is not None:
            logger.info("scenario %r generates tasks with %s", self.scenario, type(self.task_generator).__name__)
        # With no rounds, the stage still trains the train tasks that earlier generations saved.
        self.task_store = None if context.task_storage is None else context.task_storage.open(TASK_STORE_NAME)
        self.generation_runner = GenerationRunner(
            self.scenario,
            self.task_store,
            task_generator,
            description=settings.description,
            generation_attempts=settings.generation_attempts,
            generation_deadline_seconds=settings.generation_deadline_seconds,
            eval_fraction=settings.eval_fraction,
            split_seed=settings.split_seed,
            designer_delay=settings.designer_delay,
        )
        if self.worker is None and self.rounds > 0:
            self.worker = BackgroundJobWorker(
                self.generation_runner, thread_name="generated-task-rounds", concurrency=1
            )
        #: Rounds that ended without a job, such as a round of sessions that a complete generation already used.
        self.ended_rounds: list[RoundResult] = []
        #: The scenario's samples as this trainer reads them, where the published train tasks train from.
        self.samples: TrainerSamples | None = context.samples
        #: The published train tasks that the pipeline has not stored yet (``made_samples``).
        self.made: list[TaskSample] = []
        #: The names of the published train tasks without a result.
        self.unfinished: set[str] = set()
        #: The generation that runs on the worker. One generation runs at a time.
        self.generation_in_flight: int | None = None
        self.last_error = ""
        #: The rounds after which a generation's Designer replies train; 0 trains none.
        self.designer_delay = settings.designer_delay
        #: The generations whose results reached the generator, as ``refresh_designer_state`` last read them.
        self.delivered: frozenset[int] = frozenset()
        #: The complete generations that published a train task, in increasing order, as last read.
        self.train_rounds: tuple[int, ...] = ()
        #: The generations whose delivery failed since this start; the tasks of their carrier rounds do not wait.
        self.undelivered: set[int] = set()
        #: The records that can hold the Designer replies of each complete generation's train tasks.
        self.generation_designer_records: dict[int, frozenset[str]] = {}
        #: Whether a job's end asked for a delivery that the next ``advance`` submits; only with designer_delay.
        self.delivery_due = False
        #: The delivery jobs on the worker whose outcome the stage has not read; counted only with designer_delay.
        self.deliveries_in_flight = 0
        #: The Designer replies that deliveries saved since this start, and the records that they skipped.
        self.designer_counts = DesignerSampleCounts()
        #: The read records of the Designer replies that deliveries saved and that the stage has not released.
        self.saved_designer_records: set[str] = set()
        store = self.task_store
        self.next_round = next_generation(store) if store is not None else 0
        self.completed_rounds = complete_generation_count(store) if store is not None else 0
        if store is not None:
            saved = rollout_results(store)
            stored = self.stored_tasks()
            for generation, published in published_train_tasks(store):
                if published.name not in saved:
                    self.unfinished.add(published.name)
                # A stop can fall between the generation's end and the pipeline storing its tasks.
                if published.name not in stored:
                    self.made.append(task_item(store, generation, published))
        if self.designer_delay > 0:
            self.refresh_designer_state()
        # A trained task without a result waits for the end of its task job (job_ended).
        # Results that a stop kept from the generator go out now, even when no other job is due.
        if store is not None and self.worker is not None and deliverable_generations(store):
            self.deliver_later()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the rounds of one scenario, with the ``TaskGenerator`` that ``generator`` names.

        The generator gets its state in the stage's task store. Raises ``PluginConfigError`` if the config names
        a generator and the context has no task storage.
        """
        settings = GeneratedTaskSettings.parse(config, plugin_class=cls)
        task_generator = None
        if settings.generator is not None:
            generator = TaskGenerator.reference(settings.generator, setting="pipeline.record2sample.config.generator")
            generator_class = TaskGenerator.referenced_class(generator)
            if context.task_storage is None:
                raise PluginConfigError(
                    "pipeline.record2sample.config.generator needs the task storage of a scenario session: "
                    "the generator keeps its state in the task store"
                )
            task_generator = generator_class.from_config(
                generator.config,
                scenario=context.scenario,
                environ=environ,
                state=StoredGeneratorState(context.task_storage.open(TASK_STORE_NAME)),
            )
        return cls(context, settings, task_generator=task_generator)

    def handle_other_report(self, report: AgentRecord) -> None:
        # The report ends here. releasable_record_ids holds back a session's records until its round ends.
        super().handle_other_report(report)
        if self.sources_per_round == 0 or self.rounds_stopped:
            return
        if any(reference not in self.inferences for reference in report.references):
            return
        calls = self.report_calls(report)
        if calls is None:
            return
        context = ReportContext(report, report_score(report), tuple(self.inferences[call] for call in calls))
        if self.source_grouping.is_source_report(context):
            self.waiting_sources.append(context)

    def round_due(self) -> bool:
        """Whether the next round should start now."""
        if self.rounds_stopped or self.running_round is not None:
            return False
        # With no unfinished task, no job end comes to count toward the next round.
        if not (self.follow_at_once or self.trained_jobs >= self.jobs_per_round or self.unfinished_task_count() == 0):
            return False
        return self.waiting_task_count() < self.refill_tasks

    def advance(self, now: float) -> None:
        """Submit a delivery that a job's end asked for, end the rounds that finished, then start the next round
        when it is due, or stop the rounds at the limit."""
        super().advance(now)
        if self.delivery_due:
            self.delivery_due = False
            self.submit_delivery()
        for result in self.finished_rounds():
            finished = self.running_round
            self.running_round = None
            self.follow_at_once = not result.made_train_tasks
            if result.counts:
                self.completed_rounds += 1
            if result.retry and finished is not None and not self.rounds_stopped:
                self.retry_sources = finished.sources
        if not self.round_due():
            return
        if self.completed_rounds >= self.rounds:
            self.stop_rounds()
            return
        sources = self.retry_sources or tuple(self.waiting_sources[: self.sources_per_round])
        generation_round = TaskGenerationRound(sources=sources)
        if not self.start_round(generation_round):
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
        held |= self.held_designer_record_ids()
        # The task store keeps the saved replies, so their records end, even those that no report owns.
        saved = self.saved_designer_records - self.live_references() - held
        return (super().releasable_record_ids() - held) | saved

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        super().release_records(agent_record_ids)
        self.saved_designer_records -= agent_record_ids

    def background_jobs_running(self) -> bool:
        """Return True while a round runs: its end can start the next round without a new record. With
        ``designer_delay`` above 0, also while a delivery is due or runs: its end can release the tasks that wait
        for it."""
        return (
            self.running_round is not None
            or self.delivery_due
            or self.deliveries_in_flight > 0
            or super().background_jobs_running()
        )

    def made_samples(self) -> tuple[tuple[str, TaskSample], ...]:
        """The published train tasks that the pipeline has not stored and that do not wait for Designer replies
        (``is_released``), each named ``generated-tasks/<task name>``."""
        released: list[TaskSample] = []
        waiting: list[TaskSample] = []
        for task in self.made:
            (released if self.is_released(task) else waiting).append(task)
        self.made = waiting
        return tuple((f"{TASK_STORE_NAME}/{task.metadata['task']['name']}", task) for task in released)

    # ------------------------------------------------------------- Designer replies

    def refresh_designer_state(self) -> None:
        """Read the delivered generations and the generations that published a train task again."""
        if self.task_store is None:
            return
        self.delivered = delivered_generations(self.task_store)
        self.train_rounds = train_rounds_of(self.task_store)

    def is_released(self, task: TaskSample) -> bool:
        """Whether the pipeline may store ``task``: with ``designer_delay`` above 0, each generation that its round
        carries reached the generator, so ``designer(g)`` is saved, or failed to reach it since this start."""
        round_number = task_round(task)
        if self.designer_delay == 0 or round_number is None:
            return True
        return all(
            generation in self.delivered or generation in self.undelivered
            for generation in carried_generations(round_number, self.train_rounds, self.designer_delay)
        )

    def designer_records_of(self, generation: int) -> frozenset[str]:
        """The records that can hold the Designer replies of the train tasks of the complete ``generation``."""
        cached = self.generation_designer_records.get(generation)
        if cached is not None:
            return cached
        store = self.task_store
        state = None if store is None else store.generation(generation)
        if store is None or state is None:
            return frozenset()
        published = {stored.name: stored for stored in store.tasks(split="train")}
        found = frozenset(
            record_id
            for name in train_task_names(state)
            if name in published
            for record_id in designer_record_ids(read_harbor_task(store.path(published[name].files_digest)))
        )
        self.generation_designer_records[generation] = found
        return found

    def held_designer_record_ids(self) -> frozenset[str]:
        """The records that the stage holds so that a delivery makes the delayed samples of the Designer replies: with
        ``designer_delay`` above 0, those that can hold a reply of the train tasks of each complete generation that
        the generator has not taken."""
        if self.designer_delay == 0:
            return frozenset()
        return frozenset(
            record_id
            for generation in self.train_rounds
            if generation not in self.delivered
            for record_id in self.designer_records_of(generation)
        )

    def designer_records(self) -> tuple[AgentRecord, ...]:
        """The held Designer records (``held_designer_record_ids``) that the stage read, in record ID order: what a
        job's deliveries make delayed samples of."""
        return tuple(
            self.inferences[record_id]
            for record_id in sorted(self.held_designer_record_ids())
            if record_id in self.inferences
        )

    def stored_tasks(self) -> dict[str, StoredSample]:
        """The generated tasks that the scenario's items hold for this trainer, by task name: the tasks that its
        stage made."""
        return {} if self.samples is None else stored_tasks(self.samples)

    def unfinished_task_count(self) -> int:
        """The number of published train tasks without a result: each waits for a job, or for its job's end."""
        return len(self.unfinished)

    def waiting_task_count(self) -> int:
        """The number of unfinished train tasks that no committed job trained: each waits for a job."""
        trainer = "" if self.samples is None else self.samples.trainer
        stored = self.stored_tasks()
        return sum(name not in stored or stored[name].train_count(trainer) == 0 for name in self.unfinished)

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
        job = GenerationJob(
            receipt=f"generation-{generation:05d}",
            generation=generation,
            source_records=remaining,
            designer_records=self.designer_records(),
        )
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
        outcomes = self.worker.poll()
        for outcome in outcomes:
            if outcome.receipt == DELIVERY_RECEIPT:
                self.deliveries_in_flight = max(0, self.deliveries_in_flight - 1)
                if isinstance(outcome, DeliveryOutcome):
                    self.last_error = outcome.error
                    self.undelivered.update(outcome.undelivered)
                    self.designer_counts += outcome.designer
                    self.saved_designer_records.update(self.inferences.keys() & outcome.saved_record_ids)
                else:
                    self.last_error = "the delivery did not finish"
                    if self.designer_delay > 0 and self.task_store is not None:
                        # No round waits for a delivery that cannot run.
                        self.undelivered.update(state.generation for state in deliverable_generations(self.task_store))
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
            finished.append(RoundResult(made_train_tasks=bool(outcome.tasks), retry=not outcome.is_complete))
            self.made.extend(outcome.tasks)
            self.unfinished.update(task.metadata["task"]["name"] for task in outcome.tasks)
            self.undelivered.update(outcome.undelivered)
            self.designer_counts += outcome.designer
            self.saved_designer_records.update(self.inferences.keys() & outcome.saved_record_ids)
        # A delivery saves Designer replies, and a generation publishes train tasks: both change which tasks wait.
        if outcomes and self.designer_delay > 0:
            self.refresh_designer_state()
        return tuple(finished)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()

    def operational_metrics(self) -> Mapping[str, float | int]:
        metrics: dict[str, float | int] = {
            **super().operational_metrics(),
            "waiting_source_sessions": len(self.waiting_sources),
            "generation_in_flight": 0 if self.generation_in_flight is None else 1,
        }
        if self.designer_delay > 0:
            metrics.update(
                saved_designer_samples=self.designer_counts.saved,
                uncaptured_designer_records=self.designer_counts.uncaptured,
                missing_designer_records=self.designer_counts.missing,
            )
        return metrics

    def status(self) -> Mapping[str, Any]:
        store = self.task_store
        trainer = "" if self.samples is None else self.samples.trainer
        trained = frozenset(name for name, stored in self.stored_tasks().items() if stored.train_count(trainer) > 0)
        return {
            **super().status(),
            "task_store": task_counts(store, trained) if store is not None else {},
            "task_generation": {
                "generator": None if self.task_generator is None else type(self.task_generator).__name__,
                "in_flight": self.generation_in_flight,
                "next": self.next_round,
                "completed": self.completed_rounds,
                "of": self.rounds,
                "last_error": self.last_error,
            },
        }

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Run a submitted generation or delivery through the stage's execution owner."""
        return await self.generation_runner.run(job)

    async def deliver_results(self) -> str:
        """Deliver completed generations through the stage's execution owner; return the error of the last delivery
        that failed."""
        return (await self.generation_runner.deliver(self.designer_records())).error

    def save_result(self, result: TaskRolloutResult) -> None:
        """Save a task's first training result through the stage's execution owner."""
        self.generation_runner.save_result(result)

    def job_ended(self, outcome: JobOutcome) -> None:
        """Save a result for each task that a committed job trained and that has none: its rewards in the job's
        ``rollout_rewards``, or why no rollout ran.

        One task job runs at a time, so such a task belongs to the ended job, or to an earlier job that a new
        training start cancelled. A second call for the same end saves nothing new. Then the stage sends the
        finished generations' results to the generator.
        """
        rewards = outcome.metrics.get("rollout_rewards", {})
        if not isinstance(rewards, Mapping):
            raise TypeError("a task job's rollout_rewards metric must map task names to rewards")
        error = outcome.error or "no rollout of the task ran"
        store = self.task_store
        if store is not None and self.samples is not None:
            saved = rollout_results(store)
            for name, stored in self.stored_tasks().items():
                if stored.train_count(self.samples.trainer) == 0 or name in saved:
                    continue
                values = tuple(rewards.get(name) or ())
                if values:
                    self.generation_runner.save_result(TaskRolloutResult(name, values))
                else:
                    self.generation_runner.save_result(
                        TaskRolloutResult(name, error=error, infrastructure_failure=True)
                    )
                self.unfinished.discard(name)
        self.task_job_ended(finished=outcome.event == "finished")
        self.deliver_later()

    def deliver_later(self) -> None:
        """Send the generator the results that it has not received, on the worker, after the jobs before it.

        With ``designer_delay`` above 0, the next ``advance`` submits the delivery: it comes after the samples service
        read the records, so the delivery gets the Designer records that a restart reads again.
        """
        if self.designer_delay > 0:
            self.delivery_due = True
            return
        self.submit_delivery()

    def submit_delivery(self) -> None:
        """Submit a delivery with the held Designer records (``designer_records``)."""
        job = DeliveryJob(receipt=DELIVERY_RECEIPT, designer_records=self.designer_records())
        if self.worker is None or not self.worker.submit(job):
            return
        if self.designer_delay > 0:
            # Tasks can wait for the delivery's Designer replies: the samples service polls until it ends.
            self.deliveries_in_flight += 1


class GeneratedTaskJobData(Sample2Job):
    """The tasks of the next job as one task job's data: the ``TaskSample`` values themselves, and the Designer
    replies that the job carries.

    The generated tasks train like any other sample. A job takes at least ``job_samples``
    tasks, 1 by default, and at most ``buffer_size``, 4 by default or ``job_samples`` when that is more. The
    training component rolls the tasks out on its own sampler and trains on their rollouts (a task job; see the
    training component's job parameters). The commit that submits the job counts the tasks trained;
    ``GeneratedTaskSamples.job_ended`` saves their results when the job ends. Imported train tasks train the same
    way; only generated tasks get results.

    ``designer_delay`` is the record-to-sample stage's setting, a non-negative int. Above 0, the context gives the
    training component's ``DelayedSampleLimit`` and the scenario session's task storage and samples, and the job
    data's ``delayed`` holds the Designer replies that the job carries (``delayed_samples``). The stage reads them
    from the task store and the scenario's samples, so control builds the data in whichever process it runs.
    The constructor refuses other values.
    """

    accepts_imported_tasks = True

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = Sample2JobSettings.parse(config, plugin_class=cls)
        if not isinstance(record2sample, GeneratedTaskSamples):
            raise PluginConfigError(f"{cls.__name__} needs GeneratedTaskSamples, not {type(record2sample).__name__}")
        return cls(
            context,
            buffer_size=settings.buffer_size,
            job_samples=settings.job_samples,
            designer_delay=record2sample.designer_delay,
        )

    def __init__(
        self,
        context: PipelineContext,
        *,
        buffer_size: int | None = None,
        job_samples: int | None = None,
        designer_delay: int = 0,
    ) -> None:
        job_samples = 1 if job_samples is None else job_samples
        buffer_size = max(DEFAULT_BUFFER_SIZE, job_samples) if buffer_size is None else buffer_size
        super().__init__(context, buffer_size, job_samples)
        if isinstance(designer_delay, bool) or not isinstance(designer_delay, int) or designer_delay < 0:
            raise PluginConfigError("designer_delay must be a non-negative integer")
        #: The rounds after which a generation's Designer replies train; 0 trains none.
        self.designer_delay = designer_delay
        #: The delayed samples that a task job of the training component trains.
        self.delayed_sample_limit = context.delayed_sample_limit
        #: The scenario's samples as this trainer reads them, whose train counts name the rounds that trained.
        self.samples = context.samples
        #: The task store that holds the Designer replies (``designer_replies``); None with ``designer_delay`` 0.
        self.task_store: TaskStore | None = None
        if designer_delay > 0:
            if self.delayed_sample_limit is None:
                raise PluginConfigError(
                    "designer_delay above 0 needs a training component whose task jobs train delayed samples"
                )
            if context.task_storage is None or context.samples is None:
                raise PluginConfigError("designer_delay above 0 needs the task storage and the samples of a scenario")
            self.task_store = context.task_storage.open(TASK_STORE_NAME)

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, samples, delayed=self.delayed_samples(samples))

    def delayed_samples(self, samples: tuple[Sample, ...]) -> tuple[DelayedSample, ...]:
        """The Designer replies that the job data of ``samples`` carries, each with its advantage.

        The data carries the samples of ``designer(g)`` for each generation g whose carrier round has a task in
        ``samples`` and no task that a committed job trained, in generation order and then record ID order. Of
        those, it holds the ones that ``delayed_sample_limit`` keeps (``DelayedSampleLimit.trainable``), and logs
        the others. A generation without ``designer(g)``, which the generator did not take, carries nothing. With
        ``designer_delay`` 0 the result is empty.
        """
        if self.task_store is None or self.samples is None or self.delayed_sample_limit is None:
            return ()
        rounds = {number for sample in samples if (number := task_round(sample)) is not None}
        if not rounds:
            return ()
        train_rounds = train_rounds_of(self.task_store)
        carrying = [
            (round_number, generations)
            for round_number in sorted(rounds)
            if (generations := carried_generations(round_number, train_rounds, self.designer_delay))
        ]
        if not carrying:
            return ()
        # The train counts page every task of the scenario, so only a round that carries replies reads them.
        trained = trained_rounds(self.samples)
        carried: list[DelayedSample] = []
        for round_number, generations in carrying:
            if round_number in trained:
                continue
            for generation in generations:
                replies = designer_replies(self.task_store, generation)
                if replies is not None:
                    carried.extend(replies.samples)
        kept = self.delayed_sample_limit.trainable(carried)
        if len(kept) < len(carried):
            logger.warning(
                "scenario %r leaves %d of %d Designer replies out of a task job: the training component trains at "
                "most %d delayed samples, each from %s",
                self.scenario,
                len(carried) - len(kept),
                len(carried),
                self.delayed_sample_limit.count,
                self.delayed_sample_limit.distribution,
            )
        return kept
