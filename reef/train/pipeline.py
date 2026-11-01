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
  the sample buffer and shapes the samples of the buffer into one job's data.

The pipeline owns what is the same for every recipe: it routes records, saves the samples of each
ready unit to the scenario's sample store, computes the buffer from the store
(the first ``buffer_size`` untrained samples in store order, and the rest of
each group that those touch), hands out the data of one job at a time until the
trainer acknowledges or releases it, and runs the training modes and the
training instructions, which the scenario gives it from the instructions queue
(``reef.train.instructions``). ``ready`` and ``build_job_data`` take a sample count:
the trainer asks for its stage's ``job_samples``, and a test or a tool may ask
for another count. Automatic job data holds at least that many samples. It
takes whole groups, so it can hold more.

A client can also upload an ATIF trajectory. It is not a record, so the
record-to-sample stage never reads it as a record; the stage renders it into
its training sample (``Record2Sample.uploaded_sample``), which counts as one
sample. It waits in the store with the other samples, in store order.

A record-to-sample stage can also make samples from no record set, such as the
generated-task recipe's tasks (``Record2Sample.made_samples``). The pipeline
stores them like the samples of a unit, and they train from the buffer like any
other sample. The stage reads each job's end (``Record2Sample.job_ended``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from reef.core.errors import ManualBufferRefused
from reef.core.job_data import JobOutcome
from reef.core.plugins import Plugin, PluginConfigError, PluginReference
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.core.training_request import TrainingRequest
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.record2sample import NoSamples, Record2Sample, RecordSetUnit
from reef.sample2job import NoJobData, Sample2Job
from reef.storage.samples import SAMPLE_KINDS, MemorySampleStore, StoredSample, TrainerSamples, consumed_sample_id
from reef.train.types import JobData, Sample, TrajectorySample

#: The keys of the deployment's ``pipeline`` section, one for each stage.
PIPELINE_STAGES = ("record2sample", "sample2job")

logger = logging.getLogger(__name__)


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

    A ready unit's samples are stored in the scenario's sample store; an uploaded
    trajectory is one stored sample. In ``auto``, ``ready(sample_count)`` is True if
    and only if the buffer (``buffer``) holds at least ``sample_count`` samples.
    ``build_job_data(sample_count)`` takes the samples of the buffer, in store
    order. Samples beyond the buffer stay in the store for later job data.

    The trainer-facing methods (``push``, ``ready``, ``build_job_data``, and
    the others) run under the trainer's lock, whichever thread calls them (the
    trainer's worker, admission's ``uploaded_sample``, the worker's
    ``background_jobs_running``), and must not block on network or model
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
        #: The sample-to-job stage that shapes the samples of the buffer into job data.
        self.sample2job = NoJobData(context) if sample2job is None else sample2job
        self.supported_training_modes = self.sample2job.supported_training_modes
        self.training_requests: dict[str, TrainingRequest] = {}
        self.consumed_requests: set[str] = set()
        # The error of each buffered instruction whose step failed; its next job data is a skip row, not a run.
        #: What the failed step of each failed instruction said, by instruction ID.
        self.failed_requests: dict[str, str] = {}
        #: The job data handed out and not yet acknowledged. While it exists the
        #: pipeline is ready, hands out the same object, and must not
        #: reshuffle what it references.
        self.pending: JobData | None = None
        self.data_number = 0
        #: The scenario's samples as this trainer reads them; in memory for a trainer without a scenario session.
        self.store = TrainerSamples(MemorySampleStore()) if context.samples is None else context.samples
        #: The sequence of the last stored sample that the pipeline read.
        self.store_sequence = 0
        #: Whether the pipeline stored a new sample or read a new training instruction since ``produce`` returned.
        self.received_input = False
        #: The stored samples that this trainer has not trained, in store order, each with the sample that job data
        #: trains: an upload rendered by the record-to-sample stage, else the stored sample. Imported tasks do not train.
        self.held: dict[str, tuple[StoredSample, Sample]] = {}
        #: The IDs of the stored samples in the handed out job data.
        self.reserved_sample_ids: tuple[str, ...] = ()
        #: The stored samples that job data of this process consumed; their train count may not be durable yet.
        self.trained_sample_ids: set[str] = set()
        self.set_training_mode(context.training_mode)

    @property
    def output_schema(self) -> type[JobData]:
        """The job data type ``build_job_data`` returns; the trainer validates it."""
        return self.sample2job.output_schema

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

    def set_training_mode(self, training_mode: str, *, fill_manual_buffer: bool = False) -> None:
        """Select future job data while preserving the handed out job data.

        With ``fill_manual_buffer``, a switch from ``auto`` to ``manual`` adds the buffer that automatic job data
        would take next to the manual buffer: a person who switches starts from what auto would train. A mode that
        a reload restores fills nothing. A switch back leaves the manual buffer stored, unused.
        """
        if training_mode not in ("auto", "manual"):
            raise ValueError("training_mode must be 'auto' or 'manual'")
        if training_mode not in self.supported_training_modes:
            raise NotImplementedError(
                f"{type(self.sample2job).__name__} does not implement training_mode={training_mode!r}"
            )
        if fill_manual_buffer and self.context.training_mode == "auto" and training_mode == "manual":
            # Manual starts from what automatic job data would take next, with the samples stored since the last read.
            self.read_store()
            self.store.insert_into_buffer(self.buffer())
        self.context = replace(self.context, training_mode=training_mode)
        self.record2sample.context_changed(self.context)

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        """Name the contract the scenario's ingress admits when it is wider than this recipe's own."""
        self.context = replace(self.context, admitted_report_type=report_type)
        self.record2sample.context_changed(self.context)

    # ------------------------------------------------------------ instructions

    def buffered_requests(self) -> int:
        """How many instructions are read into memory and not yet consumed."""
        return len(self.training_requests)

    def queue_instructions(self, requests: Iterable[TrainingRequest]) -> None:
        """Hold the scenario's queued instructions, oldest first: each one that this pipeline did not consume yet.

        The scenario reads them from the instructions queue before each step. An instruction that left the queue,
        because a commit consumed it, leaves the pipeline with its failure.
        """
        queued = {request.id: request for request in requests if request.id not in self.consumed_requests}
        if set(queued) - set(self.training_requests):
            self.received_input = True
        self.training_requests = queued
        self.failed_requests = {key: value for key, value in self.failed_requests.items() if key in queued}

    def request_failure(self, request_id: str) -> str | None:
        """What the instruction's failed step said, when it failed at all."""
        return self.failed_requests.get(request_id)

    def request_failures(self) -> Mapping[str, str]:
        """The failed instructions still buffered, by id, with what their step said."""
        return dict(self.failed_requests)

    def mark_request_failed(self, request_id: str, error: str) -> None:
        """Record that the instruction's step failed with ``error``; its next job data is consumed with a skip row."""
        self.failed_requests[request_id] = error

    def set_request_failures(self, failures: Mapping[str, str]) -> None:
        """Carry the failed instructions of a replaced pipeline into this one."""
        self.failed_requests = dict(failures)

    def discard_request(self, request_id: str) -> frozenset[str]:
        """Consume one queued instruction without job data; the committed row that names it is what recovery skips."""
        if self.pending is not None and self.pending.request is not None and self.pending.request.id == request_id:
            raise ValueError(f"training request {request_id!r} is reserved; release its job data first")
        if request_id not in self.training_requests:
            raise ValueError(f"unknown training request {request_id!r}")
        self.consume_request(request_id)
        return frozenset({request_id})

    def consume_request(self, request_id: str) -> None:
        """Take a queued instruction out of the pipeline; ``queue_instructions`` no longer holds it."""
        self.training_requests.pop(request_id, None)
        self.failed_requests.pop(request_id, None)
        self.consumed_requests.add(request_id)

    # ------------------------------------------------------------- records

    def store_samples(self) -> bool:
        """Store the samples of each unit that the stage made ready, and the samples that it made from no set.

        When the stage makes a unit ready (``take_ready_units``), the pipeline saves its samples to the sample store
        in one transaction (``TrainerSamples.add_record_samples``), which consumes its records, and the stage ends
        the unit. Return whether one or more stored sample is new.
        """
        ready_units = self.record2sample.take_ready_units()
        stored = False
        for unit in ready_units:
            stored = self.store_unit(unit) or stored
        made = self.record2sample.made_samples()
        if made:
            stored = (
                self.store.add_record_samples(
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
        """Save the samples of the ready ``unit`` to the sample store, and end the unit in the stage; return
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
        stored = self.store.add_record_samples(group, tuple(unit.consumed_ids()), samples)
        self.record2sample.consume(unit)
        return stored

    def push(self, item: AgentRecord) -> None:
        """Read one record through the record-to-sample stage."""
        self.record2sample.push(item)
        self.store_samples()

    def uploaded_sample(self, trajectory: TrajectorySample) -> TrajectorySample:
        """The training sample of an uploaded trajectory (``Record2Sample.uploaded_sample``); ``ValueError`` when the
        recipe does not accept uploads or the trajectory cannot become a sample. Pure: admission calls it too."""
        if not self.accepts_trajectories:
            raise ValueError(f"{type(self.sample2job).__name__} does not accept uploaded trajectories")
        return self.record2sample.uploaded_sample(trajectory)

    def job_ended(self, outcome: JobOutcome) -> None:
        """Give the record-to-sample stage the outcome of a job whose end the trainer committed."""
        self.record2sample.job_ended(outcome)

    def observe_serving_artifact_id(self, artifact_id: str) -> None:
        """Give the record-to-sample stage the artifact that serves the scenario now, for its staleness check."""
        self.record2sample.observe_serving_artifact_id(artifact_id)

    # --------------------------------------------------------- job data cycle

    # Make job data when the buffer holds the trainer's sample count (auto) or an instruction is queued (manual),
    # hand the same data out until it is acknowledged, then count its samples trained.

    def forget_samples(self, sample_ids: tuple[str, ...]) -> None:
        """Stop holding ``sample_ids``: the data sync removed their uploads from the store.

        A sample in handed out job data stays in it; its commit counts it as usual. A sample that the store gets
        again has a new ``sequence``, so ``read_store`` reads it again.
        """
        for sample_id in sample_ids:
            self.held.pop(sample_id, None)

    def read_store(self) -> None:
        """Read the stored samples after the store cursor that this trainer reads: hold each untrained one that it
        can train (``trainable``)."""
        while stored_samples := self.store.page(after_sequence=self.store_sequence):
            for stored in stored_samples:
                self.store_sequence = stored.sequence
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
        return bool(self.store.page(after_sequence=self.store_sequence, limit=1))

    def buffer(self) -> tuple[str, ...]:
        """The IDs of the samples that the next job data takes, in store order (``buffer_entries``)."""
        return tuple(stored.sample_id for stored, _ in self.buffer_entries())

    def buffer_entries(self) -> tuple[tuple[StoredSample, Sample], ...]:
        """The samples that the next job data takes, in store order, each with the sample that it trains.

        In ``manual``, the buffer is the trainer's manual buffer (``TrainerSamples.manual_buffer``), which only a
        client changes (``insert_into_buffer``, ``remove_from_buffer``), less the reserved samples and those that
        this process trained, of the kind of the first of them. Staleness does not apply: the client decides.

        In ``auto``, the candidates are the held samples that no job data reserved, of a kind that the recipe gives
        job parameters (``PipelineContext.job_parameters``; every kind for a trainer without a job loop), of the
        kind of the first of them, and that are not stale: a sample that the record-to-sample stage made is stale when
        the stage's staleness check finds its data too many versions behind (``Record2Sample.is_stale``), and so is every
        sample of its group. The buffer is the first ``buffer_size`` candidates, or all of them without a
        ``buffer_size``, and the other candidates of every group that those touch, so a group trains whole.
        """
        if self.training_mode == "manual":
            excluded = {*self.reserved_sample_ids, *self.trained_sample_ids}
            held = tuple(sample_id for sample_id in self.store.manual_buffer() if sample_id not in excluded)
            entries = [
                (stored, sample) for stored in self.store.get(held) if (sample := self.trainable(stored)) is not None
            ]
            return tuple(entry for entry in entries if entry[0].kind == entries[0][0].kind)
        # A kind that the recipe gives no parameters trains only with an instruction that gives them.
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
            if stored.component is not None and self.record2sample.is_stale((sample,))
        }
        fresh = [entry for entry in candidates if (entry[0].group or entry[0].sample_id) not in stale_groups]
        prefix = fresh if self.buffer_size is None else fresh[: self.buffer_size]
        groups = {stored.group for stored, _ in prefix if stored.group is not None}
        prefix_ids = {stored.sample_id for stored, _ in prefix}
        return tuple(entry for entry in fresh if entry[0].sample_id in prefix_ids or entry[0].group in groups)

    def insert_into_buffer(self, sample_ids: tuple[str, ...]) -> None:
        """Add ``sample_ids`` to the manual buffer, also samples that trained before; only in ``manual``.

        Raise ``ManualBufferRefused`` in ``auto``, and ``ValueError`` for an ID that the store does not hold, a
        sample that this trainer cannot train (an eval or an imported task, an upload that its stage does not
        take), samples of another kind than the buffer's, and a buffer of more than the stage's
        ``max_job_samples``. Then the buffer does not change.
        """
        if self.training_mode != "manual":
            raise ManualBufferRefused("the buffer changes by hand only in manual training mode")
        stored = {item.sample_id: item for item in self.store.get(sample_ids)}
        unknown = sorted(set(sample_ids) - set(stored))
        if unknown:
            raise ValueError(f"the scenario holds no samples {unknown}")
        untrainable = sorted(sample_id for sample_id, item in stored.items() if self.trainable(item) is None)
        if untrainable:
            raise ValueError(f"this trainer cannot train the samples {untrainable}")
        buffered = self.store.get(self.store.manual_buffer())
        if len({item.kind for item in (*stored.values(), *buffered)}) > 1:
            raise ValueError("the buffer holds samples of one kind: trajectories or tasks")
        limit = self.sample2job.max_job_samples
        if limit is not None and len(set(sample_ids) | {item.sample_id for item in buffered}) > limit:
            raise ValueError(f"one job's data holds at most {limit} samples")
        self.trained_sample_ids -= set(sample_ids)
        self.store.insert_into_buffer(tuple(sample_ids))

    def remove_from_buffer(self, sample_ids: tuple[str, ...]) -> None:
        """Remove ``sample_ids`` from the manual buffer; only in ``manual``, else ``ManualBufferRefused``."""
        if self.training_mode != "manual":
            raise ManualBufferRefused("the buffer changes by hand only in manual training mode")
        self.store.remove_from_buffer(tuple(sample_ids))

    def buffer_listing(self) -> tuple[Mapping[str, Any], ...]:
        """The samples of the buffer as JSON: ``sample_id``, ``kind``, ``group``, ``component``, ``train_count``
        of this trainer, and ``stale``, whether the staleness check finds a record sample's data too old."""
        return tuple(
            {
                "sample_id": stored.sample_id,
                "kind": stored.kind,
                "group": stored.group,
                "component": stored.component,
                "train_count": stored.train_count(self.store.trainer),
                "stale": stored.component is not None and self.record2sample.is_stale((sample,)),
            }
            for stored, sample in self.buffer_entries()
        )

    def advance(self) -> None:
        """Read the samples stored since the last call; the samples service makes them (``produce``).

        Call it on the same serialized path as other pipeline operations; it starts no new thread.
        """
        self.read_store()

    def produce(self) -> bool:
        """Advance the record-to-sample work: take in background work and store the samples that it made;
        return whether the trainer received new input since the last call: a sample stored here or by ``push`` when
        a record completed a set, or a training instruction that ``push`` read.

        The samples service calls it (``Trainer.produce_samples``); ``advance`` only reads the sample store.
        """
        # Background work, such as judgments or finished rounds, forms sets or makes samples without a new record.
        self.record2sample.advance(time.monotonic())
        self.store_samples()
        stored, self.received_input = self.received_input, False
        return stored

    def ready(self, sample_count: int | None) -> bool:
        """Advance the pipeline and check readiness; auto mode needs ``sample_count`` samples in the buffer.

        With ``sample_count`` None, held samples never make the pipeline ready in ``auto``.
        """
        self.advance()
        return self.holds_job_data(sample_count)

    def holds_job_data(self, sample_count: int | None) -> bool:
        """Whether the pipeline holds the next job's data now, as ``ready`` decides it, without advancing it.

        Handed out job data holds. Else ``auto`` needs ``sample_count`` samples in the buffer. ``manual`` needs a
        queued instruction and one or more samples in the manual buffer, whatever ``sample_count``: the instruction
        trains the samples that the client chose. A stage that reads instructions (``Sample2Job.reads_instructions``)
        runs an instruction with no sample too.
        """
        if self.pending is not None:
            return True
        if self.training_mode == "manual":
            return bool(self.training_requests) and (self.sample2job.reads_instructions or bool(self.buffer_entries()))
        return sample_count is not None and len(self.buffer_entries()) >= sample_count

    def build_job_data(self, sample_count: int | None) -> JobData:
        """Return the handed out job data, or build the data of the buffer when ``ready`` for ``sample_count``."""
        if self.pending is None:
            if not self.ready(sample_count):
                raise RuntimeError(f"scenario {self.scenario!r} has no job data ready")
            self.data_number += 1
            # Manual runs the oldest queued instruction with the samples of the manual buffer; auto leaves the
            # queue to manual.
            request = next(iter(self.training_requests.values())) if self.training_mode == "manual" else None
            entries = self.buffer_entries()
            self.reserved_sample_ids = tuple(stored.sample_id for stored, _ in entries)
            samples = tuple(grouped(stored, sample) for stored, sample in entries)
            if request is None:
                self.pending = self.sample2job.make_job_data(samples, f"{self.scenario}:data:{self.data_number}")
            else:
                data = self.sample2job.make_job_data(samples, f"{self.scenario}:instruction:{request.id}")
                self.pending = replace(data, request=request)
        return self.pending

    def acknowledge(self, data_id: str) -> frozenset[str]:
        """The commit that submits the job data's job consumed it: return the consumed IDs, ``sample:<id>`` for each
        stored sample of the data, and the instruction's ID. The commit counts the samples trained, whatever the
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

    def releasable_record_ids(self) -> frozenset[str]:
        """Records whose held state can be released after a durable commit."""
        return frozenset(self.consumed_requests) | self.record2sample.releasable_record_ids()

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        """Release committed in-memory state without changing stored records."""
        self.consumed_requests -= agent_record_ids
        self.record2sample.release_records(agent_record_ids)

    def background_jobs_running(self) -> bool:
        """Return True while background work of the record-to-sample stage runs, such as a judgment.

        A finished job can make the pipeline ready without a new record.
        While this is True, the scenario's worker polls at a fixed interval.
        Otherwise it sleeps until the next accepted record. It must not block.
        """
        return self.record2sample.background_jobs_running()

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Numeric queue state, sampled under the trainer lock without advancing work."""
        return {
            "buffered_requests": self.buffered_requests(),
            **self.record2sample.operational_metrics(),
        }

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
        state: dict[str, Any] = {
            "ready_samples": len(self.buffer_entries()),
            "buffer_size": self.buffer_size,
        }
        if "manual" in self.supported_training_modes:
            state["buffered_requests"] = self.buffered_requests()
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
