"""The training pipeline: records become job data through three stages.

::

    record ─► record2record ─► record set ─► record2sample ─► sample ─► sample2job ─► job data ─► trainer

``TrainingPipeline`` holds a recipe's three stages and is the only object
that the trainer gives records to and takes job data from. A recipe writes the
stages; it does not subclass the pipeline:

* the record-to-record stage (:class:`~reef.record2record.Record2Record`)
  decides which records train, in which record sets and units, and in
  what order;
* the record-to-sample stage (:class:`~reef.record2sample.Record2Sample`) turns each
  record set into samples;
* the sample-to-job stage (:class:`~reef.sample2job.Sample2Job`) bounds
  the sample buffer and shapes the samples of the selected record set units into
  one job's data.

The pipeline owns what is the same for every recipe: it routes records, gives
each record set's sample result back to its stage, keeps the samples of held
sets in the buffer, releases the oldest sets when the buffer is over the
stage's ``buffer_size``, hands out the data of one job at a time until the
trainer acknowledges or releases it, and runs the training modes and the
training instructions. ``ready`` and ``build_job_data`` take a sample count:
the trainer asks for its stage's ``job_samples``, and a test or a tool may ask
for another count. Automatic job data holds at least that many samples. It
takes whole record sets, so it can hold more.

A client can also upload an ATIF trajectory. It is not a record, so the
record-to-record stage never sees it; the record-to-sample stage renders it into
its training sample (``Record2Sample.uploaded_sample``), which counts as one
sample. Job data takes uploaded trajectories first.

A sample-to-job stage can also make samples itself, from no record, such as
the generated-task recipe's rollouts. While ``Sample2Job.own_job_data``
returns samples, the pipeline is ready, and its next job data holds exactly
those, unless automatic job data is also ready: held samples, such as
uploaded trajectories, train first, so a recipe that trains both trains its
held samples before its own.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from reef.core.job_data import JobOutcome
from reef.core.plugins import Plugin, PluginConfigError, PluginReference
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.core.training_request import TrainingRequest
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.record2record import NoRecordSets, Record2Record, RecordSet, RecordSetUnit, SampleStatus, UnitDropReason
from reef.record2record.staleness import producing_runtime_load_ids
from reef.record2sample import NoSamples, Record2Sample
from reef.sample2job import NoJobData, Sample2Job
from reef.storage.trajectories import consumed_trajectory_id
from reef.train.types import JobData, Sample, TrajectorySample

#: The keys of the deployment's ``pipeline`` section, one for each stage.
PIPELINE_STAGES = ("record2record", "record2sample", "sample2job")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSpec:
    """A recipe's default stage classes. A deployment's ``pipeline`` section can name others."""

    record2record: type[Record2Record] = NoRecordSets
    record2sample: type[Record2Sample] = NoSamples
    sample2job: type[Sample2Job] = NoJobData


@dataclass(frozen=True)
class PipelineConfig:
    """The three stages of a recipe's pipeline: each stage's class and its config.

    It comes from the deployment's ``pipeline`` section over the recipe's
    ``PipelineSpec``. Each key of the section is ``{module, config}``; a key
    without ``module`` configures the recipe's own class, and a missing key
    builds it with no config. Reading the section imports and checks each
    class; a stage parses its config when a scenario builds it, in pipeline
    order.
    """

    record2record: PluginReference
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
        record2record, record2sample, sample2job = (section.get(stage) for stage in PIPELINE_STAGES)
        return cls(
            record2record=stage_reference(spec.record2record, record2record, "record2record")
            or Record2Record.reference(record2record, setting="pipeline.record2record"),
            record2sample=stage_reference(spec.record2sample, record2sample, "record2sample")
            or Record2Sample.reference(record2sample, setting="pipeline.record2sample"),
            sample2job=stage_reference(spec.sample2job, sample2job, "sample2job")
            or Sample2Job.reference(sample2job, setting="pipeline.sample2job"),
            environ=dict(environ),
        )

    def build(self, context: PipelineContext) -> TrainingPipeline:
        """Build the pipeline of one scenario: each stage from its config, later stages with the earlier ones."""
        record2record = Record2Record.referenced_class(self.record2record).from_config(
            self.record2record.config, context=context, environ=self.environ
        )
        record2sample = Record2Sample.referenced_class(self.record2sample).from_config(
            self.record2sample.config, context=context, record2record=record2record, environ=self.environ
        )
        sample2job = Sample2Job.referenced_class(self.sample2job).from_config(
            self.sample2job.config,
            context=context,
            record2record=record2record,
            record2sample=record2sample,
            environ=self.environ,
        )
        return TrainingPipeline(
            context, record2record=record2record, record2sample=record2sample, sample2job=sample2job
        )


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


@dataclass(frozen=True)
class InstructionFailure:
    """A failed instruction and the attempt metadata carried through an in-process trainer reload."""

    error: str
    metrics: Mapping[str, Any] = field(default_factory=dict)


class TrainingPipeline:
    """Turn a scenario's records into typed job data through a recipe's three stages.

    Without stages it is the no-update default: it reads records for audit
    but never becomes ready and never produces job data.

    A held unit holds the samples of its record sets; an uploaded
    trajectory is one sample. In ``auto``, ``ready(sample_count)`` is True if
    and only if the held units and uploaded trajectories hold at least
    ``sample_count`` samples. ``build_job_data(sample_count)`` takes the uploaded
    trajectories first, then whole units in selection order until the data
    holds ``sample_count`` samples or the units run out. When ``buffer_size``
    is set, after it moves record sets, the unreserved units hold at most
    ``buffer_size`` samples, or are one unit: it releases the oldest ones
    beyond that and keeps the newest, so a unit larger than the buffer still
    trains.

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
        record2record: Record2Record | None = None,
        record2sample: Record2Sample | None = None,
        sample2job: Sample2Job | None = None,
    ) -> None:
        self.context = context
        self.scenario = context.scenario
        #: The record-to-record stage that turns the records into record sets.
        self.record2record = NoRecordSets() if record2record is None else record2record
        #: The record-to-sample stage that turns each record set into samples.
        self.record2sample = NoSamples() if record2sample is None else record2sample
        #: The sample-to-job stage that shapes the samples of the selected units into job data.
        self.sample2job = NoJobData(context) if sample2job is None else sample2job
        self.supported_training_modes = self.sample2job.supported_training_modes
        # The two modes that take an instruction go together: a recipe that cannot run one cannot run it in either.
        if not context.manual_enabled:
            self.supported_training_modes = self.supported_training_modes - {"manual", "hybrid"}
        self.training_requests: dict[str, TrainingRequest] = {}
        self.consumed_requests: set[str] = set()
        # The error of each buffered instruction whose step failed; its next job data is a skip row, not a run.
        self.failed_requests: dict[str, InstructionFailure] = {}
        #: The job data handed out and not yet acknowledged. While it exists the
        #: pipeline is ready, hands out the same object, and must not
        #: reshuffle what it references.
        self.pending: JobData | None = None
        self.data_number = 0
        #: Samples by the ID of the record set that made them, until the set trains or ends.
        self.samples: dict[Hashable, tuple[Sample, ...]] = {}
        #: Record sets pushed to the record-to-sample stage that have no result yet, by set ID. Their records stay held.
        self.waiting_sets: dict[Hashable, RecordSet] = {}
        #: Uploaded trajectories by trajectory ID, in upload order, until job data consumes them.
        self.uploaded_trajectories: dict[str, TrajectorySample] = {}
        #: The IDs of the uploaded trajectories in the handed out job data.
        self.reserved_trajectory_ids: tuple[str, ...] = ()
        #: The commit IDs of the uploaded trajectories that job data consumed. A second read of one does not queue it.
        self.consumed_trajectory_ids: set[str] = set()
        #: Whether the handed out job data holds the sample-to-job stage's own samples (``Sample2Job.own_job_data``).
        self.pending_is_own = False
        #: The IDs that the stage's consumed own job data listed, until their commit is durable.
        self.consumed_own_ids: set[str] = set()
        self.buffer_release_warned = False
        self.set_training_mode(context.training_mode)

    @property
    def output_schema(self) -> type[JobData]:
        """The job data type ``build_job_data`` returns; the trainer validates it."""
        return self.sample2job.output_schema

    @property
    def required_request_types(self) -> frozenset[RequestType]:
        """The kinds of record that the trainer gives the pipeline."""
        return self.sample2job.required_request_types

    @property
    def accepts_trajectories(self) -> bool:
        """Whether the trainer gives the pipeline uploaded trajectories."""
        return self.sample2job.accepts_trajectories

    @property
    def accepts_imported_tasks(self) -> bool:
        """Whether the recipe reads the scenario's imported tasks."""
        return self.sample2job.accepts_imported_tasks

    @property
    def buffer_size(self) -> int | None:
        """The most samples that the held record sets keep, or None for every sample; the sample-to-job stage sets it."""
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
        """Select future job data while preserving the held units and the reservation."""
        if training_mode not in ("auto", "manual", "hybrid"):
            raise ValueError("training_mode must be 'auto', 'manual' or 'hybrid'")
        if training_mode not in self.supported_training_modes:
            raise NotImplementedError(
                f"{type(self.sample2job).__name__} does not implement training_mode={training_mode!r}"
            )
        self.context = replace(self.context, training_mode=training_mode)
        self.record2record.context_changed(self.context)
        self.forget_ended_sets()

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        """Name the contract the scenario's ingress admits when it is wider than this recipe's own."""
        self.context = replace(self.context, admitted_report_type=report_type)
        self.record2record.context_changed(self.context)

    # ------------------------------------------------------------ instructions

    def buffered_requests(self) -> int:
        """How many instructions are read into memory and not yet consumed."""
        return len(self.training_requests)

    def request_failure(self, request_id: str) -> str | None:
        """What the instruction's failed step said, when it failed at all."""
        failure = self.failed_requests.get(request_id)
        return None if failure is None else failure.error

    def request_failure_metrics(self, request_id: str) -> Mapping[str, Any]:
        """The exact failed attempt's metadata, if the backend had produced any."""
        failure = self.failed_requests.get(request_id)
        return {} if failure is None else failure.metrics

    def request_failures(self) -> Mapping[str, InstructionFailure]:
        """The failed instructions still buffered, by id, with what their step said."""
        return dict(self.failed_requests)

    def mark_request_failed(self, request_id: str, error: str, metrics: Mapping[str, Any] | None = None) -> None:
        """Record that the instruction's step failed; its next job data is consumed with a skip row."""
        self.failed_requests[request_id] = InstructionFailure(error, dict(metrics or {}))

    def set_request_failures(self, failures: Mapping[str, InstructionFailure]) -> None:
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
        """Take a queued instruction out of the queue; a replay of its record no longer queues it."""
        self.training_requests.pop(request_id)
        self.failed_requests.pop(request_id, None)
        self.consumed_requests.add(request_id)

    # ------------------------------------------------------------- records

    def forget_ended_sets(self) -> None:
        """Forget the samples of the sets that the record stage ended since the last call."""
        for set_id in self.record2record.take_ended():
            self.samples.pop(set_id, None)

    def count_unit_samples(self, unit: RecordSetUnit) -> int:
        """The number of samples in one held record set unit: those of its record sets."""
        return sum(len(self.samples[set_id]) for set_id in unit.set_ids())

    def count_held_samples(self) -> int:
        """The samples that job data can take now: those of the held units and the uploaded trajectories."""
        pool = self.record2record.pool
        return sum(self.count_unit_samples(unit) for unit in pool.units()) + len(self.uploaded_trajectories)

    def release_over_buffer(self) -> None:
        """Release the oldest unreserved units while their samples are more than ``buffer_size``.

        The reserved job data and the newest unreserved unit stay. Uploaded trajectories do not count: their store
        keeps them. Without a ``buffer_size`` nothing is released.
        """
        buffer_size = self.buffer_size
        if buffer_size is None:
            return
        pool = self.record2record.pool
        reserved = {unit.unit_id for unit in pool.reserved_units()}
        unreserved = sorted(
            (unit for unit in pool.units() if unit.unit_id not in reserved), key=lambda unit: unit.arrival_index
        )
        held = sum(self.count_unit_samples(unit) for unit in unreserved)
        for unit in unreserved[:-1]:
            if held <= buffer_size:
                break
            held -= self.count_unit_samples(unit)
            self.record2record.drop_held_unit(unit, reason=UnitDropReason.BUFFER_LIMIT)
            if not self.buffer_release_warned:
                logger.warning(
                    "scenario %r released a held record set beyond the buffer size %d (further releases are not logged)",
                    self.scenario,
                    buffer_size,
                )
                self.buffer_release_warned = True
        self.forget_ended_sets()

    def move_record_sets(self) -> None:
        """Push each record set that the record stage formed to the sample stage, and settle each sample result.

        The sample stage returns the result of each pushed set one time. Until
        then the set waits, and its records stay held. A ready set's samples
        enter the buffer, and the oldest sets leave it when it is over its size.
        """
        for record_set in self.record2record.pull():
            self.record2sample.push(record_set)
            self.waiting_sets[record_set.set_id] = record_set
        for set_id, result in self.record2sample.pull():
            if set_id not in self.waiting_sets:
                raise ValueError(
                    f"{type(self.record2sample).__name__} returned a result for set {set_id!r}, which no set waits for"
                )
            record_set = self.waiting_sets.pop(set_id)
            if result.status is SampleStatus.READY:
                self.samples[set_id] = result.samples
            self.record2record.settle(record_set, result.status, producing_runtime_load_ids(result.samples))
        self.forget_ended_sets()
        self.release_over_buffer()

    def push(self, item: AgentRecord) -> None:
        """Read one record. Training instructions stay here; every other record goes through the record stage."""
        if item.request_type is RequestType.TRAIN:
            if item.scenario != self.scenario:
                raise ValueError("training records must belong to the pipeline's scenario")
            request = TrainingRequest.from_dict(item.payload, request_id=item.agent_record_id)
            if request.id not in self.consumed_requests:
                self.training_requests.setdefault(request.id, request)
            return
        self.record2record.push(item)
        self.move_record_sets()

    def uploaded_sample(self, trajectory: TrajectorySample) -> TrajectorySample:
        """The training sample of an uploaded trajectory (``Record2Sample.uploaded_sample``); ``ValueError`` when the
        recipe does not accept uploads or the trajectory cannot become a sample. Pure: admission calls it too."""
        if not self.accepts_trajectories:
            raise ValueError(f"{type(self.sample2job).__name__} does not accept uploaded trajectories")
        return self.record2sample.uploaded_sample(trajectory)

    def push_trajectory(self, trajectory_id: str, trajectory: TrajectorySample) -> None:
        """Read one uploaded trajectory as its training sample. It waits here for job data."""
        if consumed_trajectory_id(trajectory_id) not in self.consumed_trajectory_ids:
            self.uploaded_trajectories.setdefault(trajectory_id, self.uploaded_sample(trajectory))

    def job_ended(self, outcome: JobOutcome) -> None:
        """Give the sample-to-job stage the outcome of a job whose end the trainer committed."""
        self.sample2job.job_ended(outcome)

    def observe_serving_runtime_load_id(self, runtime_load_id: str) -> None:
        """Give the record-to-record stage the runtime load ID that serves the scenario now, for its staleness check."""
        self.record2record.observe_serving_runtime_load_id(runtime_load_id)

    # --------------------------------------------------------- job data cycle
    #
    # Make job data when the held units and uploaded trajectories hold the trainer's
    # sample count (auto, hybrid) or an instruction is queued (manual, hybrid), hand
    # the same data out until it is acknowledged, then end the units and
    # trajectories it consumed.

    def advance(self) -> None:
        """Advance background work and settle its results without reserving job data.

        Apply buffer limits and, without pending job data, staleness checks.
        This operation needs no backend sample count. Call it on the same
        serialized path as other pipeline operations; it starts no new thread.
        """
        # Background work, such as judgments or finished rounds, forms sets without a new record.
        self.record2record.advance(time.monotonic())
        self.move_record_sets()
        self.sample2job.advance()
        if self.pending is None:
            self.record2record.drop_stale_units()
            self.forget_ended_sets()

    def ready(self, sample_count: int | None) -> bool:
        """Advance the pipeline and check readiness; auto mode needs ``sample_count`` held samples.

        The stage's own job data (``Sample2Job.own_job_data``) makes the pipeline ready, whatever its sample count.
        With ``sample_count`` None, as for a stage that ``trains_own_job_data_only``, held samples never make
        the pipeline ready.
        """
        self.advance()
        if self.pending is not None or self.sample2job.own_job_data() is not None:
            return True
        return self.held_data_ready(sample_count)

    def held_data_ready(self, sample_count: int | None) -> bool:
        """Whether the training mode lets held samples and instructions make job data of ``sample_count`` samples."""
        if self.training_mode == "manual":
            return bool(self.training_requests)
        if self.training_mode == "hybrid" and self.training_requests:
            return True
        return sample_count is not None and self.count_held_samples() >= sample_count

    def build_job_data(self, sample_count: int | None) -> JobData:
        """Return the handed out job data, or build data of at least ``sample_count`` samples when ``ready``."""
        if self.pending is None:
            if not self.ready(sample_count):
                raise RuntimeError(f"scenario {self.scenario!r} has no job data ready")
            self.data_number += 1
            own = self.sample2job.own_job_data()
            # Held samples go first; the stage's own samples wait while they are ready.
            if own is not None and not self.held_data_ready(sample_count):
                # The stage's own samples train as one job, however many they are.
                self.pending = self.sample2job.make_job_data(own, f"{self.scenario}:data:{self.data_number}")
                self.pending_is_own = True
                return self.pending
            # Manual and hybrid run the oldest queued instruction first; auto leaves the queue to a mode that takes it.
            request = (
                next(iter(self.training_requests.values()))
                if self.training_mode != "auto" and self.training_requests
                else None
            )
            # In manual mode an instruction runs alone; in hybrid it takes the samples that automatic job data would.
            selected_samples = (
                0 if sample_count is None or (request is not None and self.training_mode == "manual") else sample_count
            )
            # Uploaded trajectories go first, in upload order. Whole record set units, in selection order, fill
            # the rest of the data until it holds ``selected_samples`` samples.
            self.reserved_trajectory_ids = tuple(self.uploaded_trajectories)[:selected_samples]
            missing = selected_samples - len(self.reserved_trajectory_ids)
            units: list[RecordSetUnit] = []
            for unit in self.record2record.pool.ordered_units():
                if missing <= 0:
                    break
                units.append(unit)
                missing -= self.count_unit_samples(unit)
            selected = self.record2record.pool.reserve(tuple(units))
            samples = (
                *(self.uploaded_trajectories[trajectory_id] for trajectory_id in self.reserved_trajectory_ids),
                *(sample for unit in selected for set_id in unit.set_ids() for sample in self.samples[set_id]),
            )
            if request is None:
                self.pending = self.sample2job.make_job_data(samples, f"{self.scenario}:data:{self.data_number}")
            else:
                data = self.sample2job.make_job_data(samples, f"{self.scenario}:instruction:{request.id}")
                self.pending = replace(data, request=request)
        return self.pending

    def acknowledge(self, data_id: str) -> frozenset[str]:
        """The commit that submits the job data's job consumed it: end its units and uploaded trajectories, and
        return the consumed IDs. The records are consumed whatever the job's outcome."""
        if self.pending is None or self.pending.data_id != data_id:
            raise ValueError(f"unknown data_id {data_id!r}")
        if self.pending_is_own:
            # The stage's own job data consumes no record: the stage names what the commit lists.
            own_ids = self.sample2job.own_job_data_submitted()
            self.consumed_own_ids |= own_ids
            self.pending = None
            self.pending_is_own = False
            return own_ids
        consumed_units = self.record2record.pool.consume_reserved()
        consumed = self.record2record.consume(consumed_units)
        self.forget_ended_sets()
        for trajectory_id in self.reserved_trajectory_ids:
            self.uploaded_trajectories.pop(trajectory_id)
            self.consumed_trajectory_ids.add(consumed_trajectory_id(trajectory_id))
        consumed = consumed | {consumed_trajectory_id(trajectory_id) for trajectory_id in self.reserved_trajectory_ids}
        self.reserved_trajectory_ids = ()
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
        self.pending_is_own = False
        self.reserved_trajectory_ids = ()
        self.record2record.pool.release_reservation()

    # ------------------------------------------------------------- lifecycle

    def releasable_record_ids(self) -> frozenset[str]:
        """Records whose held state can be released after a durable commit. A waiting set's records stay held."""
        waiting = {record_id for record_set in self.waiting_sets.values() for record_id in record_set.record_ids()}
        return (
            frozenset(self.consumed_requests)
            | frozenset(self.consumed_trajectory_ids)
            | frozenset(self.consumed_own_ids)
            | self.record2record.releasable_record_ids()
        ) - waiting

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        """Release committed in-memory state without changing stored records."""
        self.consumed_requests -= agent_record_ids
        self.consumed_trajectory_ids -= agent_record_ids
        self.record2record.release_records(agent_record_ids)
        durable_own_ids = frozenset(self.consumed_own_ids & agent_record_ids)
        if durable_own_ids:
            self.consumed_own_ids -= durable_own_ids
            self.sample2job.release_own_ids(durable_own_ids)

    def background_jobs_running(self) -> bool:
        """Return True while background work runs: a waiting set, or a record stage job such as a judgment.

        A finished job can make the pipeline ready without a new record.
        While this is True, the scenario's worker polls at a fixed interval.
        Otherwise it sleeps until the next accepted record. It must not block.
        """
        return (
            bool(self.waiting_sets)
            or self.sample2job.background_jobs_running()
            or self.record2record.background_jobs_running()
        )

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Numeric queue state, sampled under the trainer lock without advancing work."""
        return {
            "buffered_requests": self.buffered_requests(),
            **self.record2record.operational_metrics(),
            **self.record2sample.operational_metrics(),
        }

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that callers need while waiting.

        ``ready_samples`` against the trainer's ``job_samples`` says how close
        the next job's data is; ``buffer_size`` is the most samples the held
        record sets keep, or None for every sample. A caller waiting on a
        release needs it to tell job data that is still filling from data
        that never will be: a scenario whose producer stopped short sits
        below ``job_samples`` with nothing else to report.
        The stages add terminal outcomes that can never train, so a bounded
        external wait can fail explicitly.
        """
        state: dict[str, Any] = {
            "ready_samples": self.count_held_samples(),
            "buffer_size": self.buffer_size,
        }
        if self.supported_training_modes & {"manual", "hybrid"}:
            state["buffered_requests"] = self.buffered_requests()
        return {**state, **self.record2record.status(), **self.record2sample.status(), **self.sample2job.status()}

    def close(self) -> None:
        """Close the stages; safe to call more than once.

        After ``close`` returns, no background worker of a stage may touch
        shared state or deliver further results.
        """
        try:
            self.record2record.close()
        finally:
            try:
                # A stage's own work, such as a rollout, can still save results through the record-to-sample
                # stage, so it closes first.
                self.sample2job.close()
            finally:
                self.record2sample.close()
