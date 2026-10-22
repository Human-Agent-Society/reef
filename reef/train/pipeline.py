"""The training pipeline: records become batches through three stages.

::

    record ─► record2record ─► record set ─► record2task ─► task ─► task2batch ─► batch ─► trainer

``TrainingPipeline`` holds a recipe's three stages and is the only object
that the trainer gives records to and takes batches from. A recipe writes the
stages; it does not subclass the pipeline:

* the record-to-record stage (:class:`~reef.record2record.Record2Record`)
  decides which records train, in which record sets and batch units, and in
  what order;
* the record-to-task stage (:class:`~reef.record2task.Record2Task`) turns each
  record set into tasks;
* the task-to-batch stage (:class:`~reef.task2batch.Task2Batch`) shapes
  the tasks of the selected batch units into a batch.

The pipeline owns what is the same for every recipe: it routes records, gives
each record set's task result back to its stage, keeps the tasks of held sets,
hands out one batch at a time until the trainer acknowledges or releases it,
and runs the training modes and the training instructions.

A client can also upload an ATIF trajectory. It is not a record, so it does not
go through the first two stages. The pipeline gives it to the task-to-batch
stage. A batch takes uploaded trajectories first.
"""

from __future__ import annotations

import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from reef.core.plugins import Plugin, PluginConfigError, PluginReference
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.core.training_request import TrainingRequest
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.record2record import NoRecordSets, Record2Record, RecordSet, TaskStatus
from reef.record2record.staleness import producing_runtime_load_ids
from reef.record2task import NoTasks, Record2Task
from reef.storage.trajectories import consumed_trajectory_id
from reef.task2batch import NoBatches, Task2Batch
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem


#: In manual mode the pipeline holds at most this many batches of units; it releases older ones.
MANUAL_HELD_BATCHES = 4


#: The keys of the deployment's ``pipeline`` section, one for each stage.
PIPELINE_STAGES = ("record2record", "record2task", "task2batch")


@dataclass(frozen=True)
class PipelineSpec:
    """A recipe's default stage classes. A deployment's ``pipeline`` section can name others."""

    record2record: type[Record2Record] = NoRecordSets
    record2task: type[Record2Task] = NoTasks
    task2batch: type[Task2Batch] = NoBatches


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
    record2task: PluginReference
    task2batch: PluginReference
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
        record2record, record2task, task2batch = (section.get(stage) for stage in PIPELINE_STAGES)
        return cls(
            record2record=stage_reference(spec.record2record, record2record, "record2record")
            or Record2Record.reference(record2record, setting="pipeline.record2record"),
            record2task=stage_reference(spec.record2task, record2task, "record2task")
            or Record2Task.reference(record2task, setting="pipeline.record2task"),
            task2batch=stage_reference(spec.task2batch, task2batch, "task2batch")
            or Task2Batch.reference(task2batch, setting="pipeline.task2batch"),
            environ=dict(environ),
        )

    def build(self, context: PipelineContext) -> TrainingPipeline:
        """Build the pipeline of one scenario: each stage from its config, later stages with the earlier ones."""
        record2record = Record2Record.referenced_class(self.record2record).from_config(
            self.record2record.config, context=context, environ=self.environ
        )
        record2task = Record2Task.referenced_class(self.record2task).from_config(
            self.record2task.config, context=context, record2record=record2record, environ=self.environ
        )
        task2batch = Task2Batch.referenced_class(self.task2batch).from_config(
            self.task2batch.config,
            context=context,
            record2record=record2record,
            record2task=record2task,
            environ=self.environ,
        )
        return TrainingPipeline(context, record2record=record2record, record2task=record2task, task2batch=task2batch)


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
    """Turn a scenario's records into typed training batches through a recipe's three stages.

    Without stages it is the no-update default: it reads records for audit
    but never becomes ready and never produces a batch.

    The trainer-facing methods (``push``, ``ready``, ``build_batch``, and
    the others) run on the trainer's thread under its lock and must not block
    on network or model latency. Stages that call models run that work on a
    background worker and release it in ``close``, which the owning scenario
    calls when it drops the pipeline. State that lives only in memory is
    rebuilt after a restart: the trainer replays the records that no committed
    batch consumed.
    """

    def __init__(
        self,
        context: PipelineContext,
        *,
        record2record: Record2Record | None = None,
        record2task: Record2Task | None = None,
        task2batch: Task2Batch | None = None,
    ) -> None:
        self.context = context
        self.scenario = context.scenario
        #: The record-to-record stage that turns the records into record sets.
        self.record2record = NoRecordSets() if record2record is None else record2record
        #: The record-to-task stage that turns each record set into tasks.
        self.record2task = NoTasks() if record2task is None else record2task
        #: The task-to-batch stage that shapes the tasks of the selected units into a batch.
        self.task2batch = NoBatches(context) if task2batch is None else task2batch
        self.supported_training_modes = self.task2batch.supported_training_modes
        # The two modes that take an instruction go together: a recipe that cannot run one cannot run it in either.
        if not context.manual_enabled:
            self.supported_training_modes = self.supported_training_modes - {"manual", "hybrid"}
        self.training_requests: dict[str, TrainingRequest] = {}
        self.consumed_requests: set[str] = set()
        # The error of each buffered instruction whose step failed; its next batch is a skip row, not a run.
        self.failed_requests: dict[str, InstructionFailure] = {}
        #: The batch handed out and not yet acknowledged. While it exists the
        #: pipeline is ready, hands out the same object, and must not
        #: reshuffle what it references.
        self.pending: TrainingBatch | None = None
        self.batch_number = 0
        #: Tasks by the ID of the record set that made them, until the set trains or ends.
        self.tasks: dict[Hashable, tuple[TrainDataItem, ...]] = {}
        #: Record sets pushed to the record-to-task stage that have no result yet, by set ID. Their records stay held.
        self.waiting_sets: dict[Hashable, RecordSet] = {}
        #: Uploaded trajectories by trajectory ID, in upload order, until a batch consumes them.
        self.uploaded_trajectories: dict[str, TrajectoryItem] = {}
        #: The IDs of the uploaded trajectories in the handed out batch.
        self.reserved_trajectory_ids: tuple[str, ...] = ()
        #: The commit IDs of the uploaded trajectories that a batch consumed. A second read of one does not queue it.
        self.consumed_trajectory_ids: set[str] = set()
        self.set_training_mode(context.training_mode)

    @property
    def output_schema(self) -> type[TrainingBatch]:
        """The batch type ``build_batch`` returns; the trainer validates it."""
        return self.task2batch.output_schema

    @property
    def required_request_types(self) -> frozenset[RequestType]:
        """The kinds of record that the trainer gives the pipeline."""
        return self.task2batch.required_request_types

    @property
    def accepts_trajectories(self) -> bool:
        """Whether the trainer gives the pipeline uploaded trajectories."""
        return self.task2batch.accepts_trajectories

    @property
    def batch_size(self) -> int:
        """The number of batch units in one batch. An uploaded trajectory counts as one unit."""
        return self.task2batch.batch_size

    @property
    def training_mode(self) -> str:
        """The batching policy selected for this pipeline."""
        return self.context.training_mode

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, pipeline, and backend."""
        return self.context.experiment_logger

    def set_training_mode(self, training_mode: str) -> None:
        """Select future batches while preserving the held units and the reservation."""
        if training_mode not in ("auto", "manual", "hybrid"):
            raise ValueError("training_mode must be 'auto', 'manual' or 'hybrid'")
        if training_mode not in self.supported_training_modes:
            raise NotImplementedError(
                f"{type(self.task2batch).__name__} does not implement training_mode={training_mode!r}"
            )
        # Manual mode batches on instructions, not units, so the held units are bounded instead.
        held_unit_limit = self.batch_size * MANUAL_HELD_BATCHES if training_mode == "manual" else None
        self.context = replace(self.context, training_mode=training_mode, held_unit_limit=held_unit_limit)
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
        """Record that the instruction's step failed; its next batch is consumed with a skip row."""
        self.failed_requests[request_id] = InstructionFailure(error, dict(metrics or {}))

    def set_request_failures(self, failures: Mapping[str, InstructionFailure]) -> None:
        """Carry the failed instructions of a replaced pipeline into this one."""
        self.failed_requests = dict(failures)

    def discard_request(self, request_id: str) -> frozenset[str]:
        """Consume one queued instruction without a batch; the committed row that names it is what recovery skips."""
        if self.pending is not None and self.pending.request is not None and self.pending.request.id == request_id:
            raise ValueError(f"training request {request_id!r} is reserved; release its batch first")
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
        """Forget the tasks of the sets that the record stage ended since the last call."""
        for set_id in self.record2record.take_ended():
            self.tasks.pop(set_id, None)

    def move_record_sets(self) -> None:
        """Push each record set that the record stage formed to the task stage, and settle each task result.

        The task stage returns the result of each pushed set one time. Until
        then the set waits, and its records stay held.
        """
        for record_set in self.record2record.pull():
            self.record2task.push(record_set)
            self.waiting_sets[record_set.set_id] = record_set
        for set_id, result in self.record2task.pull():
            if set_id not in self.waiting_sets:
                raise ValueError(
                    f"{type(self.record2task).__name__} returned a result for set {set_id!r}, which no set waits for"
                )
            record_set = self.waiting_sets.pop(set_id)
            if result.status is TaskStatus.READY:
                self.tasks[set_id] = result.tasks
            self.record2record.settle(record_set, result.status, producing_runtime_load_ids(result.tasks))
        self.forget_ended_sets()

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

    def push_trajectory(self, trajectory_id: str, trajectory: TrajectoryItem) -> None:
        """Read one uploaded trajectory. It waits here for a batch."""
        if not self.accepts_trajectories:
            raise ValueError(f"{type(self.task2batch).__name__} does not accept uploaded trajectories")
        if consumed_trajectory_id(trajectory_id) not in self.consumed_trajectory_ids:
            self.uploaded_trajectories.setdefault(trajectory_id, trajectory)

    def observe_serving_runtime_load_id(self, runtime_load_id: str) -> None:
        """Give the record stage the runtime load ID that serves the scenario now, for its staleness check."""
        self.record2record.observe_serving_runtime_load_id(runtime_load_id)

    # ------------------------------------------------------------ batch cycle
    #
    # Batch when enough units and uploaded trajectories are held (auto, hybrid)
    # or an instruction is queued (manual, hybrid), hand the same batch out
    # until it is acknowledged, then end the units and trajectories it consumed.

    def ready(self) -> bool:
        # Background work, such as judgments or finished rounds, forms sets without a new record.
        self.record2record.advance(time.monotonic())
        self.move_record_sets()
        if self.pending is not None:
            return True
        self.record2record.drop_stale_units(self.context.max_staleness)
        self.forget_ended_sets()
        if self.training_mode == "manual":
            return bool(self.training_requests)
        if self.training_mode == "hybrid" and self.training_requests:
            return True
        return len(self.record2record.pool) + len(self.uploaded_trajectories) >= self.batch_size

    def build_batch(self) -> TrainingBatch:
        if self.pending is None:
            if not self.ready():
                raise RuntimeError(f"scenario {self.scenario!r} has no batch ready")
            self.batch_number += 1
            # Manual and hybrid run the oldest queued instruction first; auto leaves the queue to a mode that takes it.
            request = (
                next(iter(self.training_requests.values()))
                if self.training_mode != "auto" and self.training_requests
                else None
            )
            # In manual mode an instruction runs alone; in hybrid it takes the units that an automatic batch would.
            unit_count = 0 if request is not None and self.training_mode == "manual" else self.batch_size
            # Uploaded trajectories go first, in upload order. Record set units fill the rest of the batch.
            self.reserved_trajectory_ids = tuple(self.uploaded_trajectories)[:unit_count]
            selected = self.record2record.pool.reserve(unit_count - len(self.reserved_trajectory_ids))
            tasks = (
                *(self.uploaded_trajectories[trajectory_id] for trajectory_id in self.reserved_trajectory_ids),
                *(task for unit in selected for set_id in unit.set_ids() for task in self.tasks[set_id]),
            )
            if request is None:
                self.pending = self.task2batch.make_batch(tasks, f"{self.scenario}:batch:{self.batch_number}")
            else:
                batch = self.task2batch.make_batch(tasks, f"{self.scenario}:instruction:{request.id}")
                self.pending = replace(batch, request=request)
        return self.pending

    def acknowledge(self, batch_id: str, *, trained: bool = True) -> frozenset[str]:
        """The training step consumed the batch: end its units and uploaded trajectories, and return the consumed IDs.

        ``trained`` is False when the backend dropped the batch as stale, for
        a record stage that paces work on what trained.
        """
        if self.pending is None or self.pending.batch_id != batch_id:
            raise ValueError(f"unknown batch_id {batch_id!r}")
        consumed_units = self.record2record.pool.consume_reserved()
        consumed = self.record2record.consume(consumed_units, trained=trained)
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

    def release_batch(self, batch_id: str) -> None:
        """Forget the handed out batch without consuming anything; the next ``build_batch`` selects again."""
        if self.pending is None or self.pending.batch_id != batch_id:
            raise ValueError(f"unknown batch_id {batch_id!r}")
        self.pending = None
        self.reserved_trajectory_ids = ()
        self.record2record.pool.release_reservation()

    # ------------------------------------------------------------- lifecycle

    def releasable_record_ids(self) -> frozenset[str]:
        """Records whose held state can be released after a durable commit. A waiting set's records stay held."""
        waiting = {record_id for record_set in self.waiting_sets.values() for record_id in record_set.record_ids()}
        return (
            frozenset(self.consumed_requests)
            | frozenset(self.consumed_trajectory_ids)
            | self.record2record.releasable_record_ids()
        ) - waiting

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        """Release committed in-memory state without changing stored records."""
        self.consumed_requests -= agent_record_ids
        self.consumed_trajectory_ids -= agent_record_ids
        self.record2record.release_records(agent_record_ids)

    def background_jobs_running(self) -> bool:
        """Return True while background work runs: a waiting set, or a record stage job such as a judgment.

        A finished job can make the pipeline ready without a new record.
        While this is True, the training thread polls at a fixed interval.
        Otherwise it sleeps until the next accepted record. The training
        thread reads this between drains. It must not block.
        """
        return bool(self.waiting_sets) or self.record2record.background_jobs_running()

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Numeric queue state, sampled under the trainer lock without advancing work."""
        return {
            "buffered_requests": self.buffered_requests(),
            **self.record2record.operational_metrics(),
            **self.record2task.operational_metrics(),
        }

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that callers need while waiting.

        ``ready_units`` against ``batch_size`` says how close the next batch
        is. A caller waiting on a release needs it to tell a batch that is
        still filling from one that never will: a scenario whose producer
        stopped short sits below the batch size with nothing else to report.
        The stages add terminal outcomes that can never train, so a bounded
        external wait can fail explicitly.
        """
        state: dict[str, Any] = {
            "ready_units": len(self.record2record.pool) + len(self.uploaded_trajectories),
            "batch_size": self.batch_size,
        }
        if self.supported_training_modes & {"manual", "hybrid"}:
            state["buffered_requests"] = self.buffered_requests()
        return {**state, **self.record2record.status(), **self.record2task.status()}

    def close(self) -> None:
        """Close the record and task stages; safe to call more than once.

        After ``close`` returns, no background worker of a stage may touch
        shared state or deliver further results.
        """
        try:
            self.record2record.close()
        finally:
            self.record2task.close()
