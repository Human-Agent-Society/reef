"""The training pipeline: records become job data through two stages.

::

    record ─► record2sample ─► sample ─► sample2job ─► job data ─► trainer

``TrainingPipeline`` holds a recipe's two stages and is the only object
that the trainer gives records to and takes job data from. A recipe writes the
stages; it does not subclass the pipeline:

* the record-to-sample stage (:class:`~reef.record2sample.Record2Sample`)
  decides which records train, in which record sets and units, and turns each
  record set into samples;
* the sample-to-job stage (:class:`~reef.sample2job.Sample2Job`) bounds
  the size of a job and shapes its samples into the job's data.

The pipeline owns what is the same for every recipe: it routes records, saves the samples of each
ready unit as data items that the scenario uses, selects the samples of the next job
(in ``auto``, the first ``buffer_size`` untrained samples in store order, and the
rest of each group that those touch; in ``manual``, the samples that the
request names), hands out the data of one job at a time until the
trainer acknowledges or releases it, and runs the training modes and the
training requests, which the scenario gives it from the training request queue
(``reef.train.training_requests``). ``ready`` and ``build_job_data`` take a sample count:
the trainer asks for its stage's ``job_samples``, and a test or a tool may ask
for another count. Automatic job data holds at least that many samples. It
takes whole groups, so it can hold more.

A client can also upload an ATIF trajectory. It is not a record, so the
record-to-sample stage never reads it as a record; the stage renders it into
its training sample (``Record2Sample.uploaded_sample``), which counts as one
sample. It waits in the store with the other samples, in store order.

A record-to-sample stage can also make samples from no record set, such as the
generated-task recipe's tasks (``Record2Sample.made_samples``). The pipeline
stores them like the samples of a unit, and they train like any other sample. The stage reads each job's end (``Record2Sample.job_ended``).
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from reef.core.job_data import JobOutcome
from reef.core.plugins import Plugin, PluginConfigError, PluginReference
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.core.training_request import TrainingRequest
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.record2sample import NoSamples, Record2Sample, RecordSetUnit
from reef.record2sample.staleness import ServedArtifacts
from reef.sample2job import NoJobData, Sample2Job
from reef.storage.samples import SAMPLE_KINDS, StoredSample, consumed_sample_id, memory_samples
from reef.train.types import JobData, Sample, TrajectorySample

#: The keys of the deployment's ``pipeline`` section, one for each stage.
PIPELINE_STAGES = ("record2sample", "sample2job")

logger = logging.getLogger(__name__)

#: The most records whose reading task a pipeline remembers (``TrainingPipeline.read_by``).
MAX_READ_RECORDS = 100_000


@dataclass(frozen=True)
class PipelineSpec:
    """A recipe's default stage classes. A deployment's ``pipeline`` section can name others."""

    record2sample: type[Record2Sample] = NoSamples
    sample2job: type[Sample2Job] = NoJobData


@dataclass(frozen=True)
class PipelineConfig:
    """The two stages of a recipe's pipeline: each stage's class and its config.

    It comes from the deployment's ``pipeline`` section over the recipe's
    ``PipelineSpec``. Each key of the section is ``{module, config}``; a key
    without ``module`` configures the recipe's own class, and a missing key
    builds it with no config. Reading the section imports and checks each
    class; a stage parses its config when a scenario builds it, in pipeline
    order.
    """

    record2sample: PluginReference
    sample2job: PluginReference
    environ: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_setting(cls, value: object, *, spec: PipelineSpec, environ: Mapping[str, str]) -> PipelineConfig:
        """Read the ``pipeline`` section over ``spec``; ``None`` is the recipe's stages with no config."""
        if value is not None and not isinstance(value, Mapping):
            raise PluginConfigError(f"pipeline must be a mapping with {', '.join(PIPELINE_STAGES)}")
        section: Mapping[str, Any] = {} if value is None else value
        unknown = sorted(str(key) for key in section if key not in PIPELINE_STAGES)
        if unknown:
            raise PluginConfigError(
                f"pipeline has unknown key(s) {', '.join(unknown)}; expected {', '.join(PIPELINE_STAGES)}"
            )
        record2sample, sample2job = (section.get(stage) for stage in PIPELINE_STAGES)
        return cls(
            record2sample=stage_reference(spec.record2sample, record2sample, "record2sample")
            or Record2Sample.reference(record2sample, setting="pipeline.record2sample"),
            sample2job=stage_reference(spec.sample2job, sample2job, "sample2job")
            or Sample2Job.reference(sample2job, setting="pipeline.sample2job"),
            environ=dict(environ),
        )

    def build(self, context: PipelineContext) -> TrainingPipeline:
        """Build the pipeline of one scenario: each stage from its config, the job stage with the sample stage."""
        record2sample = Record2Sample.referenced_class(self.record2sample).from_config(
            self.record2sample.config, context=context, environ=self.environ
        )
        sample2job = Sample2Job.referenced_class(self.sample2job).from_config(
            self.sample2job.config, context=context, record2sample=record2sample, environ=self.environ
        )
        return TrainingPipeline(context, record2sample=record2sample, sample2job=sample2job)


def stage_reference(default: type[Plugin], value: object, stage: str) -> PluginReference | None:
    """The recipe's class for one stage of the ``pipeline`` section, absent or ``{config}``; None when it names a module."""
    if value is None:
        return PluginReference(default, {})
    if not isinstance(value, Mapping) or "module" in value:
        return None
    unknown = sorted(str(key) for key in value if key != "config")
    if unknown:
        raise PluginConfigError(
            f"pipeline.{stage} has unknown key(s) {', '.join(unknown)}; expected module and config"
        )
    config = value.get("config", {})
    if not isinstance(config, Mapping):
        raise PluginConfigError(f"pipeline.{stage}.config must be a mapping")
    return PluginReference(default, dict(config))


def grouped(stored: StoredSample, sample: Sample) -> Sample:
    """``sample`` as job data trains it: a trajectory of a stored group without its own ``group_id`` gets the group."""
    if isinstance(sample, TrajectorySample) and sample.group_id is None and stored.group is not None:
        return replace(sample, group_id=stored.group)
    return sample


class TrainingPipeline:
    """Turn a scenario's records into typed job data through a recipe's two stages.

    Without stages it is the no-update default: it reads records for audit
    but never becomes ready and never produces job data.

    A ready unit's samples become data items that the scenario uses; an uploaded
    trajectory is one sample too. In ``auto``, ``ready(sample_count)`` is True if
    and only if the next automatic job (``auto_entries``) holds at least ``sample_count``
    samples, and ``build_job_data(sample_count)`` takes them, in store order. Other
    samples stay in the store for later job data. In ``manual``, a queued request's
    job takes the samples that it names (``request_entries``).

    The trainer-facing methods (``push``, ``ready``, ``build_job_data``, and
    the others) run under the trainer's lock, whichever thread calls them (the
    trainer's worker, the worker's ``background_jobs_running``), and must not block on network or model
    latency. Stages that call models run that work on a
    background worker and release it in ``close``, which the owning scenario
    calls when it drops the pipeline. State that lives only in memory is
    rebuilt after a restart: the trainer replays the records that no committed
    job data consumed.
    """

    def __init__(
        self,
        context: PipelineContext,
        *,
        record2sample: Record2Sample | None = None,
        sample2job: Sample2Job | None = None,
    ) -> None:
        self.context = context
        self.scenario = context.scenario
        #: The record-to-sample stage that turns the records into samples.
        self.record2sample = NoSamples() if record2sample is None else record2sample
        #: The sample-to-job stage that shapes the samples of a job into its data.
        self.sample2job = NoJobData(context) if sample2job is None else sample2job
        self.supported_training_modes = self.sample2job.supported_training_modes
        self.training_requests: dict[str, TrainingRequest] = {}
        #: Whether this trainer takes each queued request, by ID. ``request_entries`` decides it once: it reads every
        #: sample of the request, and ``holds_job_data`` runs on each poll.
        self.takes_request: dict[str, bool] = {}
        self.consumed_requests: set[str] = set()
        #: What the failed step of each failed request said, by request ID.
        self.failed_requests: dict[str, str] = {}
        #: The job data handed out and not yet acknowledged. While it exists the
        #: pipeline is ready, hands out the same object, and must not
        #: reshuffle what it references.
        self.pending: JobData | None = None
        self.data_number = 0
        #: The scenario's samples as this trainer reads them; in memory for a trainer without a scenario session.
        self.store = memory_samples(context.scenario).for_trainer("") if context.samples is None else context.samples
        #: The sequence of the last stored sample that the pipeline read.
        self.store_sequence = 0
        #: Whether the pipeline stored a new sample or read a new training request since ``produce`` returned.
        self.received_input = False
        #: The number of new samples that this pipeline stored.
        self.stored_sample_count = 0
        #: The ID of the sample processing task whose step runs now, or None (``Trainer.produce_samples``).
        self.selection_task: str | None = None
        #: The read number and the task of each of the last ``MAX_READ_RECORDS`` records that a task read, by record
        #: ID, oldest first. The read number counts the reads of this pipeline.
        self.read_by: OrderedDict[str, tuple[int, str]] = OrderedDict()
        self.reads = 0
        #: The stored samples that this trainer has not trained, in store order, each with the sample that job data
        #: trains: an upload rendered by the record-to-sample stage, else the stored sample. Imported tasks do not train.
        self.held: dict[str, tuple[StoredSample, Sample]] = {}
        #: The IDs of the stored samples in the handed out job data.
        self.reserved_sample_ids: tuple[str, ...] = ()
        #: The stored samples that job data of this process consumed; their train count may not be durable yet.
        self.trained_sample_ids: set[str] = set()
        #: The ended records of the stage that this pipeline stored and no commit named yet.
        self.stored_ended_record_ids: set[str] = set()
        #: The artifacts that the scenario served, oldest first, for the staleness check.
        self.served_artifacts = ServedArtifacts()
        self.set_training_mode(context.training_mode)

    @property
    def required_request_types(self) -> frozenset[RequestType]:
        """The kinds of record that the trainer gives the pipeline: the stage's."""
        return self.sample2job.required_request_types

    @property
    def accepts_trajectories(self) -> bool:
        """Whether the trainer gives the pipeline uploaded trajectories."""
        return self.sample2job.accepts_trajectories

    @property
    def accepts_imported_tasks(self) -> bool:
        """Whether the recipe trains the scenario's imported ``train`` tasks (``Sample2Job.accepts_imported_tasks``)."""
        return self.sample2job.accepts_imported_tasks

    @property
    def buffer_size(self) -> int | None:
        """The most samples that one automatic job takes before group completion, or None for every sample."""
        return self.sample2job.buffer_size

    @property
    def training_mode(self) -> str:
        """The training mode selected for this pipeline."""
        return self.context.training_mode

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, pipeline, and backend."""
        return self.context.experiment_logger

    def set_training_mode(self, training_mode: str) -> None:
        """Select future job data while preserving the handed out job data."""
        if training_mode not in ("auto", "manual"):
            raise ValueError("training_mode must be 'auto' or 'manual'")
        if training_mode not in self.supported_training_modes:
            raise NotImplementedError(
                f"{type(self.sample2job).__name__} does not implement training_mode={training_mode!r}"
            )
        self.context = replace(self.context, training_mode=training_mode)
        if training_mode == "manual":
            # A later switch to auto reads the store again: the train counts skip the samples that trained.
            self.held.clear()
            self.store_sequence = 0
        self.record2sample.context_changed(self.context)

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        """Name the contract the scenario's ingress admits when it is wider than this recipe's own."""
        self.context = replace(self.context, admitted_report_type=report_type)
        self.record2sample.context_changed(self.context)

    # ------------------------------------------------------------ requests

    def queue_requests(self, requests: Iterable[TrainingRequest], failures: Mapping[str, str] | None = None) -> None:
        """Hold the scenario's queued requests, oldest first: each one that this pipeline did not consume yet.

        The scenario reads them from the training request queue before each step. ``failures`` holds what the failed
        step of each failed request said, by ID, as the queue stores it; they join the failures that this pipeline
        holds. A request that left the queue, because a commit consumed it, leaves the pipeline with its failure.
        """
        queued = {request.id: request for request in requests if request.id not in self.consumed_requests}
        if set(queued) - set(self.training_requests):
            self.received_input = True
        self.training_requests = queued
        self.takes_request = {key: value for key, value in self.takes_request.items() if key in queued}
        held = {**self.failed_requests, **(failures or {})}
        self.failed_requests = {key: value for key, value in held.items() if key in queued}

    def request_failure(self, request_id: str) -> str | None:
        """What the request's failed step said, when it failed at all."""
        return self.failed_requests.get(request_id)

    def mark_request_failed(self, request_id: str, error: str) -> None:
        """Record that the request's step failed with ``error``; its next job data is consumed with a skip row."""
        self.failed_requests[request_id] = error

    def discard_request(self, request_id: str) -> frozenset[str]:
        """Consume one queued request without job data; the committed row that names it is what recovery skips."""
        if self.pending is not None and self.pending.request is not None and self.pending.request.id == request_id:
            raise ValueError(f"training request {request_id!r} is reserved; release its job data first")
        if request_id not in self.training_requests:
            raise ValueError(f"unknown training request {request_id!r}")
        self.consume_request(request_id)
        return frozenset({request_id})

    def consume_request(self, request_id: str) -> None:
        """Take a queued request out of the pipeline; ``queue_requests`` no longer holds it."""
        self.training_requests.pop(request_id, None)
        self.takes_request.pop(request_id, None)
        self.failed_requests.pop(request_id, None)
        self.consumed_requests.add(request_id)

    # ------------------------------------------------------------- records

    def note_read(self, record_id: str) -> None:
        """Remember that the task whose step runs now read ``record_id``; nothing happens outside a task."""
        if self.selection_task is None:
            return
        self.reads += 1
        self.read_by.pop(record_id, None)
        self.read_by[record_id] = (self.reads, self.selection_task)
        if len(self.read_by) > MAX_READ_RECORDS:
            self.read_by.popitem(last=False)

    def lineage(self, record_ids: tuple[str, ...]) -> dict[str, str]:
        """The metadata that names the tasks of samples made from ``record_ids`` now: ``stored_by_task``, the task
        whose step stores them, and ``read_by_task``, the task that read the last of their records. Each is left out
        when it is unknown, as outside a task or after a restart."""
        lineage: dict[str, str] = {}
        if self.selection_task is not None:
            lineage["stored_by_task"] = self.selection_task
        reads = [self.read_by[record_id] for record_id in record_ids if record_id in self.read_by]
        if reads:
            lineage["read_by_task"] = max(reads)[1]
        return lineage

    def add_record_samples(
        self, group: str | None, record_ids: tuple[str, ...], samples: tuple[tuple[str, Sample], ...]
    ) -> bool:
        """Save samples (``TrainerSamples.add_record_samples``) with their tasks (``lineage``), count the new ones,
        and return whether one is new."""
        added = self.store.add_record_samples(group, record_ids, samples, lineage=self.lineage(record_ids))
        self.stored_sample_count += added
        return added > 0

    def store_samples(self) -> bool:
        """Store the samples of each unit that the stage made ready, and the samples that it made from no set.

        When the stage makes a unit ready (``take_ready_units``), the pipeline saves its samples as items
        (``TrainerSamples.add_record_samples``), which then consumes its records, and the stage ends
        the unit. Return whether one or more stored sample is new.
        """
        ready_units = self.record2sample.take_ready_units()
        stored = False
        for unit in ready_units:
            stored = self.store_unit(unit) or stored
        made = self.record2sample.made_samples()
        if made:
            stored = (
                self.add_record_samples(
                    None, (), tuple((f"{self.sample_id_prefix}{name}", sample) for name, sample in made)
                )
                or stored
            )
        if ready_units or made:
            self.read_store()
        self.received_input = self.received_input or stored
        return stored

    @property
    def sample_id_prefix(self) -> str:
        """The prefix of the IDs of the samples that this trainer stores: ``<component>:``, or none for the only
        trainer of a scenario (key ``""``)."""
        return f"{self.store.trainer}:" if self.store.trainer else ""

    def store_unit(self, unit: RecordSetUnit) -> bool:
        """Save the samples of the ready ``unit`` as items, and end the unit in the stage; return
        whether one or more of them is new.

        Sample ``i`` of record set ``s`` has the ID ``<trainer>:<s>:<i>``, or ``<s>:<i>`` for the only trainer of a
        scenario. The samples of a unit of more than one sample share the group ``<trainer>:<first set>`` (or
        ``<first set>``): a record set's ID is a record ID, which forms one set only, so the group names this unit
        and no later unit of the same group key.
        """
        prefix = self.sample_id_prefix
        samples = tuple(
            (f"{prefix}{set_id}:{index}", sample)
            for set_id, set_samples in self.record2sample.unit_samples(unit)
            for index, sample in enumerate(set_samples)
        )
        group = f"{prefix}{unit.set_ids()[0]}" if len(samples) > 1 else None
        stored = self.add_record_samples(group, tuple(unit.consumed_ids()), samples)
        self.record2sample.consume(unit)
        return stored

    def push(self, item: AgentRecord) -> None:
        """Read one record through the record-to-sample stage."""
        self.record2sample.push(item)
        self.store_samples()

    def job_ended(self, outcome: JobOutcome) -> None:
        """Give the record-to-sample stage the outcome of a job whose end the trainer committed."""
        self.record2sample.job_ended(outcome)

    def observe_serving_artifact_id(self, artifact_id: str) -> None:
        """Add the artifact that serves the scenario now to those that served it, for the staleness check."""
        self.served_artifacts.observe(artifact_id)

    # --------------------------------------------------------- job data cycle

    # Make job data when the trainer's sample count waits (auto) or a request is queued (manual),
    # hand the same data out until it is acknowledged, then count its samples trained.

    def read_store(self) -> None:
        """Read the samples after the store cursor that this trainer reads: hold each untrained one that it can train
        (``trainable``). A sample whose use began again, such as a task whose split changed, replaces the held one,
        or leaves when the trainer cannot train it now.

        Manual mode reads nothing: a request names its samples (``request_entries``), and a scenario can hold more
        trajectories than this process can keep in memory."""
        if self.training_mode == "manual":
            return
        while stored_samples := self.store.page(after_sequence=self.store_sequence):
            for stored in stored_samples:
                self.store_sequence = stored.sequence
                self.held.pop(stored.sample_id, None)
                if stored.train_count(self.store.trainer) > 0 or stored.sample_id in self.trained_sample_ids:
                    continue
                sample = self.trainable(stored)
                if sample is not None:
                    self.held[stored.sample_id] = (stored, sample)

    def trainable(self, stored: StoredSample) -> Sample | None:
        """The sample that job data trains for ``stored``, or None when this trainer cannot train it.

        A sample that this trainer's stages made trains as it is, and another trainer's does not. A task of no
        component is an imported task: a ``train`` task trains as it is when the sample-to-job stage accepts
        imported tasks, and an ``eval`` task never trains, since an evaluation runs it. A trajectory of no component
        is an upload, which trains only when the sample-to-job stage accepts uploads, rendered by the
        record-to-sample stage; one that it cannot render now, for example after ``max_tokens`` was lowered, is
        skipped with a warning.
        """
        if stored.component is not None:
            # Another trainer's sample is that trainer's to train.
            return stored.sample if stored.component == self.store.trainer else None
        if not isinstance(stored.sample, TrajectorySample):
            return stored.sample if stored.split == "train" and self.accepts_imported_tasks else None
        if not self.accepts_trajectories:
            return None
        try:
            return self.record2sample.uploaded_sample(stored.sample)
        except ValueError:
            logger.warning(
                "uploaded trajectory %s of scenario %s cannot train and is skipped",
                stored.sample_id,
                self.scenario,
                exc_info=True,
            )
            return None

    def has_unread_samples(self) -> bool:
        """Whether the store holds samples after the store cursor that this trainer reads."""
        return self.training_mode == "auto" and bool(self.store.page(after_sequence=self.store_sequence, limit=1))

    def auto_entries(self) -> tuple[tuple[StoredSample, Sample], ...]:
        """The samples that the next automatic job takes, in store order, each with the sample that it trains.

        The candidates are the held samples that no job data reserved, of a kind that the recipe gives job parameters
        (``PipelineContext.job_parameters``; every kind for a trainer without a job loop), of the kind of the first
        of them, and that are not stale: a sample that the record-to-sample stage made is stale when the stage's
        staleness check finds its data too many versions behind (``Record2Sample.is_stale``). An artifact that the
        scenario serves beside its active selection is not behind. Every sample of a stale sample's group is stale. The job takes the first ``buffer_size`` candidates, or all of them without a ``buffer_size``,
        and the other candidates of every group that those touch, so a group trains whole.
        """
        # A kind that the recipe gives no parameters trains only with a request that gives them.
        kinds = set(self.context.job_parameters) or set(SAMPLE_KINDS)
        unreserved = [
            (stored, sample)
            for sample_id, (stored, sample) in self.held.items()
            if sample_id not in self.reserved_sample_ids and stored.kind in kinds
        ]
        candidates = [(stored, sample) for stored, sample in unreserved if stored.kind == unreserved[0][0].kind]
        stale_groups = {
            stored.group or stored.sample_id
            for stored, sample in candidates
            if stored.component is not None
            and not self.served_artifacts.served_beside((sample,))
            and self.record2sample.is_stale((sample,), self.served_artifacts.artifact_ids)
        }
        fresh = [entry for entry in candidates if (entry[0].group or entry[0].sample_id) not in stale_groups]
        prefix = fresh if self.buffer_size is None else fresh[: self.buffer_size]
        groups = {stored.group for stored, _ in prefix if stored.group is not None}
        prefix_ids = {stored.sample_id for stored, _ in prefix}
        return tuple(entry for entry in fresh if entry[0].sample_id in prefix_ids or entry[0].group in groups)

    def request_entries(self, request: TrainingRequest) -> tuple[tuple[StoredSample, Sample], ...]:
        """The samples that ``request`` names and the store holds, in store order, each with the sample that it
        trains. A sample that trained before trains again; staleness does not apply, since the client chose.

        Raise ``ValueError`` when this trainer cannot train one of them (another trainer's sample, an ``eval`` or an
        imported task that it does not take, an upload that its stage cannot render), for samples of more than one
        kind, and for more than the stage's ``max_job_samples``.
        """
        return self.sample_entries(self.store.get(request.samples))

    def sample_entries(self, samples: Sequence[StoredSample]) -> tuple[tuple[StoredSample, Sample], ...]:
        """``samples`` of a request, each with the sample that it trains; raise ``ValueError`` as
        ``request_entries`` does."""
        entries: list[tuple[StoredSample, Sample]] = []
        untrainable: list[str] = []
        for stored in samples:
            sample = self.trainable(stored)
            if sample is None:
                untrainable.append(stored.sample_id)
            else:
                entries.append((stored, sample))
        if untrainable:
            raise ValueError(f"this trainer cannot train the samples {sorted(untrainable)}")
        if len({stored.kind for stored, _ in entries}) > 1:
            raise ValueError("a request names samples of one kind: trajectories or tasks")
        limit = self.sample2job.max_job_samples
        if limit is not None and len(entries) > limit:
            raise ValueError(f"one job's data holds at most {limit} samples")
        return tuple(entries)

    def check_request(self, request: TrainingRequest) -> None:
        """Raise ``ValueError`` unless this trainer can run ``request``: the store holds each sample that it names,
        and ``request_entries`` takes them; a request that names none needs a stage that
        ``reads_requests``."""
        if not request.samples:
            if not self.sample2job.reads_requests:
                raise ValueError("the request names no samples to train")
            return
        samples = self.store.get(request.samples)
        unknown = sorted(set(request.samples) - {stored.sample_id for stored in samples})
        if unknown:
            raise ValueError(f"the scenario holds no samples {unknown}")
        self.sample_entries(samples)

    def next_job_listing(self) -> tuple[Mapping[str, Any], ...]:
        """The samples of the next automatic job (``auto_entries``) as JSON: ``sample_id``, ``kind``, ``group``,
        ``component`` and ``train_count`` of this trainer."""
        return tuple(
            {
                "sample_id": stored.sample_id,
                "kind": stored.kind,
                "group": stored.group,
                "component": stored.component,
                "train_count": stored.train_count(self.store.trainer),
            }
            for stored, _ in self.auto_entries()
        )

    def advance(self) -> None:
        """Read the samples stored since the last call; the samples service makes them (``produce``).

        Call it on the same serialized path as other pipeline operations; it starts no new thread.
        """
        self.read_store()

    def produce(self) -> bool:
        """Advance the record-to-sample work: take in background work and store the samples that it made;
        return whether the trainer received new input since the last call: a sample stored here or by ``push`` when
        a record completed a set, or a training request that ``push`` read.

        The samples service calls it (``Trainer.produce_samples``); ``advance`` only reads the sample store.
        """
        # Background work, such as judgments or finished rounds, forms sets or makes samples without a new record.
        self.record2sample.advance(time.monotonic())
        self.store_samples()
        stored, self.received_input = self.received_input, False
        return stored

    def settle_ended_records(self, committed_record_ids: frozenset[str] | set[str]) -> None:
        """Let the stage release the ended records that a commit named, and store the ended records that no commit
        named yet, for the trainer's next commit (``TrainerSamples.end_records``).

        ``committed_record_ids`` holds the record IDs that the trainer's commits named, in whichever process they ran.
        """
        ended = self.record2sample.releasable_record_ids()
        committed = ended & committed_record_ids
        if committed:
            self.record2sample.release_records(committed)
        uncommitted = ended - committed - self.stored_ended_record_ids
        if uncommitted:
            self.store.end_records(uncommitted)
            self.stored_ended_record_ids |= uncommitted
        self.stored_ended_record_ids -= committed

    def ready(self, sample_count: int | None) -> bool:
        """Advance the pipeline and check readiness; auto mode needs ``sample_count`` samples for the next job.

        With ``sample_count`` None, held samples never make the pipeline ready in ``auto``. A held sample that left
        the scenario, or left its ``train`` split, since the pipeline read it leaves the next job.
        """
        self.advance()
        while self.pending is None and self.training_mode == "auto" and self.holds_job_data(sample_count):
            ids = [stored.sample_id for stored, _ in self.auto_entries()]
            used = {stored.sample_id for stored in self.store.get(ids) if stored.split == "train"}
            left = [sample_id for sample_id in ids if sample_id not in used]
            if not left:
                return True
            for sample_id in left:
                self.held.pop(sample_id, None)
        return self.holds_job_data(sample_count)

    def holds_job_data(self, sample_count: int | None) -> bool:
        """Whether the pipeline holds the next job's data now, as ``ready`` decides it, without advancing it.

        Handed out job data holds. Else ``auto`` needs ``sample_count`` samples for the next job (``auto_entries``).
        ``manual`` needs a queued request whose samples this trainer trains (``request_entries``), one or
        more, whatever ``sample_count``. A stage that reads requests (``Sample2Job.reads_requests``) runs a
        request with no sample too.
        """
        if self.pending is not None:
            return True
        if self.training_mode == "auto":
            return sample_count is not None and len(self.auto_entries()) >= sample_count
        request = next(iter(self.training_requests.values()), None)
        if request is None:
            return False
        if not request.samples:
            return self.sample2job.reads_requests
        if request.id not in self.takes_request:
            try:
                self.takes_request[request.id] = bool(self.request_entries(request))
            except ValueError:
                # Another trainer's request.
                self.takes_request[request.id] = False
        return self.takes_request[request.id]

    def build_job_data(self, sample_count: int | None) -> JobData:
        """Return the handed out job data, or build the next job's data when ``ready`` for ``sample_count``."""
        if self.pending is None:
            if not self.ready(sample_count):
                raise RuntimeError(f"scenario {self.scenario!r} has no job data ready")
            self.data_number += 1
            # Manual runs the oldest queued request with the samples that it names; auto leaves the queue to manual.
            request = next(iter(self.training_requests.values())) if self.training_mode == "manual" else None
            entries = self.auto_entries() if request is None else self.request_entries(request)
            self.reserved_sample_ids = tuple(stored.sample_id for stored, _ in entries)
            samples = tuple(grouped(stored, sample) for stored, sample in entries)
            if request is None:
                self.pending = self.sample2job.make_job_data(samples, f"{self.scenario}:data:{self.data_number}")
            else:
                data = self.sample2job.make_job_data(samples, f"{self.scenario}:request:{request.id}")
                self.pending = replace(data, request=request)
        return self.pending

    def acknowledge(self, data_id: str) -> frozenset[str]:
        """The commit that submits the job data's job consumed it: return the consumed IDs, ``sample:<id>`` for each
        stored sample of the data, and the request's ID. The commit counts the samples trained, whatever the
        job's outcome; this process no longer holds them."""
        if self.pending is None or self.pending.data_id != data_id:
            raise ValueError(f"unknown data_id {data_id!r}")
        for sample_id in self.reserved_sample_ids:
            self.held.pop(sample_id, None)
            self.trained_sample_ids.add(sample_id)
        consumed = frozenset(consumed_sample_id(sample_id) for sample_id in self.reserved_sample_ids)
        self.reserved_sample_ids = ()
        if self.pending.request is not None:
            self.consume_request(self.pending.request.id)
            consumed = consumed | {self.pending.request.id}
        self.pending = None
        return consumed

    def release_job_data(self, data_id: str) -> None:
        """Forget the handed out job data without consuming anything; the next ``build_job_data`` selects again."""
        if self.pending is None or self.pending.data_id != data_id:
            raise ValueError(f"unknown data_id {data_id!r}")
        self.pending = None
        self.reserved_sample_ids = ()

    # ------------------------------------------------------------- lifecycle

    def release_requests(self, request_ids: frozenset[str]) -> None:
        """Forget the consumed requests ``request_ids``, whose commit is durable: the queue no longer gives them."""
        self.consumed_requests -= request_ids

    def background_jobs_running(self) -> bool:
        """Return True while background work of the record-to-sample stage runs, such as a judgment.

        A finished job can make the pipeline ready without a new record.
        While this is True, the scenario's worker polls at a fixed interval.
        Otherwise it sleeps until the next accepted record. It must not block.
        """
        return self.record2sample.background_jobs_running()

    def next_due(self) -> float | None:
        """The monotonic time at which ``produce`` can make samples without a new record (``Record2Sample.next_due``)."""
        return self.record2sample.next_due()

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Numeric queue state, sampled under the trainer lock without advancing work."""
        return dict(self.record2sample.operational_metrics())

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that callers need while waiting.

        ``ready_samples`` against the trainer's ``job_samples`` says how close
        the next job's data is; ``buffer_size`` is the most samples of one
        automatic job, beside the rest of a group, or None for every sample. A caller waiting on a
        release needs it to tell job data that is still filling from data
        that never will be: a scenario whose producer stopped short sits
        below ``job_samples`` with nothing else to report.
        The stages add terminal outcomes that can never train, so a bounded
        external wait can fail explicitly.
        """
        state = {"ready_samples": len(self.auto_entries()), "buffer_size": self.buffer_size}
        return {**state, **self.record2sample.status(), **self.sample2job.status()}

    def close(self) -> None:
        """Close the stages; safe to call more than once.

        After ``close`` returns, no background worker of a stage may touch
        shared state or deliver further results.
        """
        try:
            self.sample2job.close()
        finally:
            self.record2sample.close()
