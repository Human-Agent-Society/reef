"""The trainer: record consumption, candidate selection, and commit preparation.

A trainer pages records into its processor and runs every training step
through its bound backend. Dispatched backends reserve the batch first so
long-running work happens outside scenario locks.
Backends expose prepare/evaluate/settle phases; the trainer executes one
configured candidate evaluator between preparation and settlement, defaulting
to backend evaluation plus ``AlwaysSelectMixin``. Commit and compaction are split
so the scenario committer can make the commit record durable before any
row is deleted.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

from reef.core.evaluation import CandidateEvaluationPlugin, SelectionDecision, UpdateCandidate
from reef.core.records_types import RequestType
from reef.core.reports import ReportBase
from reef.core.training_request import TrainingRequest
from reef.observability import ExperimentLogger, NullExperimentLogger
from reef.observability.operations import OperationMetrics
from reef.storage.records import RecordStore
from reef.train.backend import CandidateBackend, PreparedStep, StepExecution
from reef.train.evaluation.evaluators import BackendAlwaysSelectPlugin
from reef.train.processors.base import DataProcessor, InstructionFailure
from reef.train.types import PreparedCommit, ProcessorContext, TrainingBatch, TrainStepResult


@dataclass
class _PendingStep:
    """One reserved batch and, once known, the result that will be committed.

    ``result`` stays ``None`` while a dispatched backend works outside the
    trainer lock; an inline backend fills it during ``run_once``.
    """

    batch: TrainingBatch
    result: TrainStepResult | None
    prepared_commit: PreparedCommit | None = None
    # Set once the processor has the batch back and the step consumed these ids on its own, so no acknowledgement.
    consumed_ids: frozenset[str] | None = None
    #: The release served when the batch was reserved: what the step was prepared against.
    base_release_id: str | None = None
    #: The prepared step behind ``result``; kept across a stale refusal when the candidate is evaluated again.
    prepared: PreparedStep | None = None

    @property
    def batch_id(self) -> str:
        return self.batch.batch_id


@dataclass(frozen=True)
class ComponentTrainer:
    """One trainer and the release component it evolves."""

    component: str
    trainer: Trainer

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("component must be a non-empty string")
        if not isinstance(self.trainer, Trainer):
            raise TypeError("trainer must be a Trainer")


class Trainer:
    """Turn scenario data into transactional local or backend training steps."""

    _DATA_READ_BATCH_SIZE = 256

    @classmethod
    def build(
        cls,
        scenario: str,
        records: RecordStore,
        *,
        processor_factory: Callable[[ProcessorContext], DataProcessor],
        candidate_backend: CandidateBackend | None = None,
        candidate_evaluator: CandidateEvaluationPlugin | None = None,
        algorithm_state: Mapping[str, Any] | None = None,
        report_type: type[ReportBase] | None = None,
        experiment_logger: ExperimentLogger | None = None,
        training_mode: str = "auto",
    ) -> Trainer:
        if candidate_backend is None and candidate_evaluator is not None:
            raise ValueError("candidate evaluation requires a candidate backend")
        if candidate_evaluator is not None and not isinstance(candidate_evaluator, CandidateEvaluationPlugin):
            raise TypeError("candidate_evaluator must inherit CandidateEvaluationPlugin")
        processor = processor_factory(
            ProcessorContext(
                scenario=scenario,
                report_type=report_type,
                experiment_logger=(experiment_logger if experiment_logger is not None else NullExperimentLogger()),
                training_mode=training_mode,
            )
        )
        if processor.training_mode != training_mode:
            processor.close()
            raise ValueError("processor_factory must preserve the requested training_mode")
        default_state = candidate_backend.initial_state() if candidate_backend is not None else {}
        initial_state = dict(default_state if algorithm_state is None else algorithm_state)
        return cls(
            scenario=scenario,
            records=records,
            processor=processor,
            candidate_backend=candidate_backend,
            candidate_evaluator=candidate_evaluator,
            state=initial_state,
        )

    def __init__(
        self,
        *,
        scenario: str,
        records: RecordStore,
        processor: DataProcessor,
        candidate_backend: CandidateBackend | None,
        candidate_evaluator: CandidateEvaluationPlugin | None,
        state: Mapping[str, Any],
    ) -> None:
        self._scenario = scenario
        self._records = records
        self._processor = processor
        self._candidate_backend = candidate_backend
        if candidate_backend is None:
            self._candidate_evaluator = None
        elif candidate_evaluator is not None:
            self._candidate_evaluator = candidate_evaluator
        else:
            self._candidate_evaluator = BackendAlwaysSelectPlugin(candidate_backend)
        self._state = dict(state)
        self._data_offset = 0
        self._data_sequence = 0
        self._pending: _PendingStep | None = None
        # Rows this trainer no longer needs that retention still keeps stored:
        # consumed by a committed step, or of a type its processor never
        # ingests. The processor never sees them, so the trainer itself keeps
        # releasing them until every trainer has.
        self._released_stored_ids: set[str] = set()
        self._lock = Lock()
        self.operations = OperationMetrics(("execution",))

    def operational_metrics(self) -> dict[str, float | int]:
        """Sample backlog and execution separately; a reserved batch can be running.

        Unread records are not necessarily trainable. Processor counts retain
        their recipe-defined names rather than being guessed into batch depth.
        A busy processor omits its gauges for this sample instead of blocking
        execution metrics behind ingestion or commit work.
        """
        values = {f"training/{key}": value for key, value in self.operations.snapshot().items()}
        if self._candidate_backend is not None:
            values.update(self._candidate_backend.operational_metrics())
        if not self._lock.acquire(blocking=False):
            return values
        try:
            values["records/unread_count"] = self._records.count(self.scenario, after_sequence=self._data_sequence)
            oldest = self._records.replay_page(self.scenario, after_sequence=self._data_sequence, limit=1)
            values["records/oldest_unread_age_seconds"] = (
                max(0.0, time.time() - oldest[0][1].created_at) if oldest else 0.0
            )
            values["training/reserved_batches"] = int(self._pending is not None)
            values["training/auto_enabled"] = int(self.training_mode in {"auto", "hybrid"})
            processor_metrics = {**self._processor.status(), **self._processor.operational_metrics()}
            for name, value in processor_metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values[f"processor/{name}"] = int(value) if isinstance(value, bool) else value
        finally:
            self._lock.release()
        return values

    @property
    def scenario(self) -> str:
        return self._scenario

    @property
    def training_mode(self) -> str:
        """The mode selected for subsequent batches."""
        return self._processor.training_mode

    def set_training_mode(self, training_mode: str) -> None:
        """Serialize mode selection with record ingestion and batch reservation."""
        with self._lock:
            self._processor.set_training_mode(training_mode)

    def supports_training_mode(self, training_mode: str) -> bool:
        """Whether the processor can run in ``training_mode``."""
        return training_mode in self._processor.supported_training_modes

    def pending_instructions(self) -> int:
        """Instructions accepted and not yet consumed: the ones the processor buffers plus those unread in storage."""
        with self._lock:
            unread = self._records.count(
                self.scenario, request_type=RequestType.TRAIN, after_sequence=self._data_sequence
            )
            return self._processor.buffered_requests() + unread

    def fail_pending_instruction(self, error: str) -> bool:
        """Mark the reserved instruction as failed, so its next batch is a skip row; False when none is reserved."""
        with self._lock:
            pending = self._pending
            if pending is None or pending.result is not None or pending.batch.request is None:
                return False
            metadata = {} if self._candidate_backend is None else self._candidate_backend.failed_step_metrics()
            self._processor.mark_request_failed(pending.batch.request.id, error, metadata)
            return True

    def instruction_failures(self) -> Mapping[str, InstructionFailure]:
        """The failed instructions this trainer still holds, for the trainer that replaces it."""
        with self._lock:
            return self._processor.request_failures()

    def set_instruction_failures(self, failures: Mapping[str, InstructionFailure]) -> None:
        """Carry failed instructions into this trainer's processor; a rebuilt scenario starts without them."""
        with self._lock:
            self._processor.set_request_failures(failures)

    @property
    def processor(self) -> DataProcessor:
        return self._processor

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The report type selected by the recipe that built this trainer."""
        return self._processor.context.report_type

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        """Name the contract the scenario's ingress admits, when several trainers share one scenario."""
        with self._lock:
            self._processor.admit_reports_of(report_type)

    @property
    def candidate_backend(self) -> CandidateBackend | None:
        return self._candidate_backend

    @property
    def candidate_evaluator(self) -> CandidateEvaluationPlugin | None:
        """The external or built-in candidate evaluator, when training."""
        return self._candidate_evaluator

    @property
    def state(self) -> Mapping[str, Any]:
        return self._state

    def algorithm_state_dict(self) -> Mapping[str, Any]:
        """Serialize the current algorithm state for artifact metadata."""
        return dict(self._state)

    @property
    def data_offset(self) -> int:
        return self._data_offset

    @property
    def pending_base_release_id(self) -> str | None:
        """The release the reserved batch was prepared against, if a batch is reserved."""
        with self._lock:
            return None if self._pending is None else self._pending.base_release_id

    @property
    def pending_batch(self) -> TrainingBatch | None:
        with self._lock:
            return None if self._pending is None else self._pending.batch

    def batch_ready(self) -> bool:
        """Whether a reserved or buildable batch is waiting on the training thread."""
        with self._lock:
            return self._pending is not None or self._processor.ready()

    def processor_status(self) -> Mapping[str, Any]:
        """Return caller-visible processor state under the processor's lock."""
        with self._lock:
            return dict(self._processor.status())

    def _build_validated_batch(self) -> TrainingBatch:
        """Build the next batch and hold the processor to its declared schema.

        ``DataProcessor.output_schema`` is the processor's published contract
        for what a candidate backend will receive; enforcing it at the only
        place batches enter the trainer turns a drifting processor into a loud
        error instead of a backend-side shape failure.
        """
        batch = self._processor.build_batch()
        schema = self._processor.output_schema
        if not isinstance(batch, schema):
            raise TypeError(
                f"{type(self._processor).__name__} declared output_schema "
                f"{schema.__name__} but built {type(batch).__name__}"
            )
        return batch

    def _consume_data(self) -> None:
        while True:
            items = self._records.replay_page(
                self.scenario,
                after_sequence=self._data_sequence,
                limit=self._DATA_READ_BATCH_SIZE,
            )
            if not items:
                return
            for sequence, item in items:
                if item.request_type in self.processor.required_request_types:
                    self._processor.ingest(item)
                else:
                    self._released_stored_ids.add(item.agent_record_id)
                self._data_offset += 1
                self._data_sequence = sequence
                if self._processor.ready():
                    return

    def run_once(self, scenario_step: int = 0, *, base_release_id: str | None = None) -> TrainStepResult | None:
        """Consume available data and, with a candidate backend, prepare one step.

        Returns ``None`` when this trainer has no candidate backend (it
        only advances record consumption, because a non-training scenario still
        has to drain and compact its store) or when the processor is not yet
        ready to produce a batch. ``base_release_id`` names the release served
        now; a batch reserved by this call is prepared against it.
        """
        if self._candidate_backend is not None and self._candidate_backend.dispatched:
            raise RuntimeError("dispatched candidate backends must reserve a batch before execution")
        with self._lock:
            if self._candidate_backend is None:
                self._consume_data()
                return None
            kept: PreparedStep | None = None
            if self._pending is not None:
                result = self._pending.result
                if result is not None:
                    return result
                if self._pending.base_release_id is None:
                    self._pending.base_release_id = base_release_id
                batch = self._pending.batch
                kept = self._pending.prepared
            else:
                self._consume_data()
                if not self._processor.ready():
                    return None
                batch = self._build_validated_batch()
                self._pending = _PendingStep(batch=batch, result=None, base_release_id=base_release_id)
        # Local candidate generation and evaluation can take minutes. Keep the
        # batch reserved, but release the trainer lock so status remains live.
        with self.operations.measure("execution"):
            execution = (
                self._reevaluate(kept) if kept is not None else self._execute_backend_step(batch, scenario_step)
            )
        if execution.outcome != "commit" or execution.result is None:
            raise RuntimeError(f"inline candidate backend returned {execution.outcome!r}")
        with self._lock:
            if self._pending is None or self._pending.batch_id != batch.batch_id:
                raise RuntimeError("inline trainer reservation changed while its backend was executing")
            self._pending.result = execution.result
            self._pending.prepared = execution.prepared
            return execution.result

    def _reevaluate(self, prepared: PreparedStep) -> StepExecution:
        """Evaluate a kept candidate against the release served now and settle it again."""
        backend = self._candidate_backend
        if backend is None:
            raise RuntimeError("cannot evaluate a candidate without a backend")
        prepared = backend.prepare_reevaluation(prepared)
        candidate = prepared.candidate
        if not isinstance(candidate, UpdateCandidate):
            raise TypeError("a kept step must carry an UpdateCandidate")
        try:
            decision = self._evaluate_candidate(candidate)
            return StepExecution("commit", backend.settle_step(prepared, decision), prepared=prepared)
        except BaseException:
            backend.abort_step(prepared)
            raise

    def _execute_backend_step(self, batch: TrainingBatch, scenario_step: int) -> StepExecution:
        backend = self._candidate_backend
        if backend is None:
            raise RuntimeError("cannot execute a training step without a backend")
        request = batch.request
        error = None if request is None else self._processor.request_failure(request.id)
        if request is not None and error is not None:
            # Committed without the backend: the failed instruction is consumed alone and the catalog row names why.
            return StepExecution("commit", self._skip_failed_instruction(batch, request, error))
        prepared = backend.prepare_step(batch, self._state, scenario_step)
        if not isinstance(prepared, PreparedStep):
            raise TypeError(f"{type(backend).__name__}.prepare_step must return PreparedStep")
        if prepared.outcome == "retry":
            if prepared.storage is None:
                raise RuntimeError("retry preparation must carry storage status")
            return StepExecution("retry", storage=prepared.storage)
        if prepared.outcome == "drop":
            return StepExecution("drop", metrics=prepared.metrics)
        if prepared.outcome == "skip":
            return StepExecution("commit", TrainStepResult(prepared.state, prepared.metrics))
        candidate = prepared.candidate
        if not isinstance(candidate, UpdateCandidate):
            raise TypeError("candidate preparation must carry an UpdateCandidate")
        try:
            decision = self._evaluate_candidate(candidate)
            return StepExecution("commit", backend.settle_step(prepared, decision), prepared=prepared)
        except BaseException:
            backend.abort_step(prepared)
            raise

    def _skip_failed_instruction(self, batch: TrainingBatch, request: TrainingRequest, error: str) -> TrainStepResult:
        """Consume the instruction alone; the units beside it go back to the processor for a batch a proposer reads."""
        with self._lock:
            pending = self._pending
            if pending is None or pending.batch_id != batch.batch_id:
                raise RuntimeError("trainer reservation changed while its instruction was being skipped")
            metadata = dict(self._processor.request_failure_metrics(request.id))
            self._processor.release_batch(batch.batch_id)
            pending.consumed_ids = self._processor.discard_request(request.id)
        metrics = {**metadata, "skipped": "instruction failed", "error": error}
        return TrainStepResult(dict(self._state), metrics)

    def _evaluate_candidate(self, candidate: UpdateCandidate) -> SelectionDecision:
        evaluator = self._candidate_evaluator
        if evaluator is None:
            raise RuntimeError("candidate preparation has no candidate evaluator")
        evaluation = evaluator.evaluate(candidate)
        decision = evaluator.decide(candidate, evaluation)
        if decision.evaluation is not evaluation:
            raise ValueError("candidate evaluator must retain the evaluation result supplied by Reef")
        return decision

    def reserve_training_batch(self, *, base_release_id: str | None = None) -> TrainingBatch | None:
        """Reserve one batch for a dispatched backend, prepared against the release served now."""
        backend = self._candidate_backend
        if backend is None or not backend.dispatched:
            raise RuntimeError("trainer has no dispatched candidate backend")
        with self._lock:
            if self._pending is not None:
                if self._pending.base_release_id is None:
                    self._pending.base_release_id = base_release_id
                return self._pending.batch
            self._consume_data()
            if not self._processor.ready():
                return None
            batch = self._build_validated_batch()
            self._pending = _PendingStep(batch=batch, result=None, base_release_id=base_release_id)
            return batch

    def retry_pending(self, *, keep_candidate: bool = False) -> None:
        """Keep the reserved batch but forget its result, so the next step prepares it again.

        The scenario calls this when a result was prepared against a release
        that another component's commit has since replaced: the batch is still
        the right data, and the backend must evaluate it against the release
        served now. ``keep_candidate`` keeps the prepared candidate too, so the
        next step evaluates it again instead of proposing anew.
        """
        with self._lock:
            if self._pending is None:
                return
            self._pending = _PendingStep(
                batch=self._pending.batch,
                result=None,
                prepared=self._pending.prepared if keep_candidate else None,
            )

    def execute_reserved_step(self, scenario_step: int) -> StepExecution:
        """Run the dispatched backend for the currently reserved batch."""
        with self._lock:
            if self._pending is None:
                raise RuntimeError("trainer has no reserved training batch")
            if self._pending.result is not None:
                return StepExecution("commit", self._pending.result)
            batch = self._pending.batch
        with self.operations.measure("execution"):
            execution = self._execute_backend_step(batch, scenario_step)
        if execution.outcome == "commit":
            if execution.result is None:
                raise RuntimeError("commit execution must carry a training result")
            with self._lock:
                if self._pending is None or self._pending.batch_id != batch.batch_id:
                    raise RuntimeError("trainer reservation changed while its backend was executing")
                self._pending.result = execution.result
        return execution

    def releasable_agent_record_ids(self) -> frozenset[str]:
        """The rows this trainer no longer needs and does not protect."""
        with self._lock:
            return self._releasable_ids()

    def _releasable_ids(self) -> frozenset[str]:
        retention = self._processor.retention_decision()
        released = retention.releasable_agent_record_ids | self._released_stored_ids
        return frozenset(released - retention.protected_agent_record_ids)

    def prepare_commit(
        self, result: TrainStepResult | None, *, compactable: frozenset[str] | None = None
    ) -> PreparedCommit:
        """Prepare the pending result without exposing its state as committed.

        Acknowledging a processor mutates its in-memory batch bookkeeping, so
        the prepared value is cached and reused after a publication retry. The
        trainer's algorithm state and pending reservation remain unchanged
        until :meth:`commit` is called after the scenario's artifact and commit
        record settle. ``compactable`` limits the rows this commit may retire
        to those every other trainer of the scenario has released too.
        """
        with self._lock:
            if self._pending is None:
                return PreparedCommit(
                    algorithm_state=self.algorithm_state_dict(),
                    high_water_sequence=self._data_sequence,
                    high_water_offset=self._data_offset,
                    compacted_ids=frozenset(),
                )
            if self._pending.result is not result:
                raise RuntimeError("training result does not match the pending step")
            if self._pending.prepared_commit is not None:
                return self._pending.prepared_commit
            batch_id, result = self._pending.batch_id, self._pending.result
            if result is None:
                raise RuntimeError("trainer pending batch has no result")
            if not isinstance(result.state, Mapping):
                raise TypeError("training step state must be a mapping")
            consumed = self._pending.consumed_ids
            if consumed is None:
                consumed = self._processor.acknowledge(batch_id)
            compacted = self._releasable_ids()
            if compactable is not None:
                compacted = compacted & compactable
            metrics = dict(result.metrics)
            request = self._pending.batch.request
            if request is not None:
                # The backend's own dict, when it wrote one, carries what its proposer added to ``requires``.
                metrics.setdefault("training_request", {"id": request.id, **request.to_dict()})
            prepared = PreparedCommit(
                algorithm_state=dict(result.state),
                high_water_sequence=self._data_sequence,
                high_water_offset=self._data_offset,
                compacted_ids=frozenset(compacted),
                consumed_ids=consumed,
                metrics=metrics or None,
                training_job_id=result.training_job_id,
                base_release_id=self._pending.base_release_id,
            )
            self._pending.prepared_commit = prepared
            return prepared

    def shipped_content_update(self, published_tree: Path) -> TrainStepResult | None:
        """The backend's update of its shipped content; ``None`` while a step is pending or nothing is stale."""
        with self._lock:
            if self._candidate_backend is None or self._pending is not None:
                return None
            state = dict(self._state)
        return self._candidate_backend.shipped_content_update(state, published_tree)

    def apply_committed_state(self, state: Mapping[str, Any]) -> None:
        """Expose state a commit outside a training step recorded, such as a shipped content update."""
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("cannot apply committed state while a training step is pending")
            self._state = dict(state)

    def commit(self, prepared: PreparedCommit) -> None:
        """Expose one prepared state after its scenario commit has settled."""
        with self._lock:
            if self._pending is None:
                return
            if self._pending.prepared_commit is not prepared:
                raise RuntimeError("prepared commit does not match the pending training step")
            self._state = dict(prepared.algorithm_state)
            self._pending = None

    def add_commit_metrics(self, result: TrainStepResult, metrics: Mapping[str, Any]) -> TrainStepResult:
        """Attach provider correlation fields to the exact pending result.

        Dispatcher calls this immediately before the durable commit. Updating
        the pending value under the trainer lock keeps the commit record and
        the post-commit experiment event on one immutable result snapshot.
        """
        if not metrics:
            return result
        with self._lock:
            if self._pending is None or self._pending.result is None:
                raise RuntimeError("cannot annotate a training result without a pending step")
            if self._pending.result is not result:
                raise RuntimeError("cannot annotate a result that is not the pending training step")
            annotated = replace(result, metrics={**dict(result.metrics), **dict(metrics)})
            self._pending.result = annotated
            return annotated

    def reject_pending(
        self, metrics: Mapping[str, Any] | None = None, *, compactable: frozenset[str] | None = None
    ) -> frozenset[str]:
        """Drop the reserved batch; returns the rows retired with it."""
        with self._lock:
            if self._pending is None:
                return frozenset()
            batch_id = self._pending.batch_id
            self._processor.dropped(batch_id)
            self._processor.acknowledge(batch_id)
            compacted = self._releasable_ids()
            if compactable is not None:
                compacted = compacted & compactable
            self._records.compact(
                self.scenario,
                compacted,
                receipt_id=batch_id,
                receipt_metadata={"outcome": "stale", "metrics": dict(metrics or {})},
            )
            self._processor.compaction_applied(compacted)
            self._released_stored_ids -= compacted
            self._pending = None
            return compacted

    def apply_compaction(self, compacted_ids: frozenset[str]) -> None:
        """Retire rows and notify the processor for standalone trainer callers.

        Scenario commits settle records through their store and then call
        :meth:`compaction_applied` to update processor memory.
        """
        if not compacted_ids:
            return
        with self._lock:
            self._records.compact(self.scenario, compacted_ids)
            self._processor.compaction_applied(compacted_ids)
            self._released_stored_ids -= compacted_ids

    def compaction_applied(self, compacted_ids: frozenset[str]) -> None:
        """Notify the processor after the scenario store retires committed rows."""
        if compacted_ids:
            with self._lock:
                self._processor.compaction_applied(compacted_ids)
                self._released_stored_ids -= compacted_ids

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Notify the backend after ``state`` enters the durable commit log."""
        backend = self._candidate_backend
        if backend is not None:
            backend.commit_applied(state)

    def close(self) -> None:
        """Release processor and backend resources owned by the trainer.

        Held under the trainer lock so a processor is never closed while a
        batch is being ingested or built on the training thread.
        """
        with self._lock:
            try:
                self._processor.close()
            finally:
                if self._candidate_backend is not None:
                    self._candidate_backend.close()

    def reingest(self, *, up_to_sequence: int, consumed_ids: frozenset[str]) -> None:
        """Rebuild processor memory from retained rows at or below a watermark.

        The committed high-water mark is a consumption cursor, not a liveness
        boundary: consumption stops the moment a batch is ready, so the cursor
        passes rows of the next, still-incomplete step, and retention keeps
        them stored. Only processor memory knew about them, and a rebuilt
        processor starts empty: without this replay, a report arriving after
        recovery waits forever on a reference that can never be ingested
        again. ``consumed_ids`` names the rows committed batches consumed —
        the commit log records them per step because retention may keep a
        consumed row stored (audit-only retention is contract-legal), and a
        row a committed batch consumed must never train twice. Skipping them
        while replaying the retained prefix reconstructs the live state the
        crash destroyed and nothing more; consumption accounting stays
        untouched, since the recovered mark already covers every replayed row.
        """
        if up_to_sequence < 0:
            raise ValueError("up_to_sequence must be non-negative")
        with self._lock:
            sequence = 0
            while True:
                items = self._records.replay_page(
                    self.scenario,
                    after_sequence=sequence,
                    limit=self._DATA_READ_BATCH_SIZE,
                )
                if not items:
                    return
                for sequence, item in items:
                    if sequence > up_to_sequence:
                        return
                    if item.agent_record_id in consumed_ids:
                        # Still stored, already trained: this trainer has released it and says so
                        # until every other trainer has too. The processor learns of it, so a report
                        # that arrives later on this row is settled instead of resolved against it.
                        self._released_stored_ids.add(item.agent_record_id)
                        self._processor.restore_consumed(item)
                        continue
                    if item.request_type in self.processor.required_request_types:
                        self._processor.ingest(item)
                    else:
                        self._released_stored_ids.add(item.agent_record_id)

    def restore_record_progress(self, *, after_sequence: int, offset: int) -> None:
        """Resume consumption from a recovered commit record's high-water mark.

        Without this, a recovered trainer would page the record store from
        sequence 0 and re-ingest consumed rows the retention policy protected
        from deletion — re-training them. Restoring the watermark makes
        replay start exactly where the recovered step left off.
        """
        if after_sequence < 0 or offset < 0:
            raise ValueError("record progress must be non-negative")
        with self._lock:
            self._data_sequence = after_sequence
            self._data_offset = offset
