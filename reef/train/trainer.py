"""The trainer: record consumption, candidate selection, and commit preparation.

A trainer pages records into its processor and runs every training step
through its bound backend. Dispatched backends reserve the batch first so
long-running work happens outside scenario locks.
Backends expose prepare/evaluate/settle phases; the trainer executes one
configured candidate evaluator between preparation and settlement, defaulting
to backend evaluation plus ``AlwaysSelect``. Commit and compaction are split
so the scenario commit protocol can make the commit record durable before any
row is deleted.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

from reef.core.reports import ReportBase
from reef.observability import ExperimentLogger, NullExperimentLogger
from reef.records import RecordStore
from reef.train.backend import PreparedStep, StepExecution, TrainingBackend
from reef.train.evaluation.contracts import CandidateEvaluationPlugin, SelectionDecision, UpdateCandidate
from reef.train.evaluation.evaluators import DefaultCandidateEvaluationPlugin
from reef.train.processors.base import DataProcessor
from reef.train.types import PreparedCommit, ProcessorContext, TrainingBatch, TrainStepResult


@dataclass
class _PendingStep:
    """One reserved batch and, once known, the result that will be committed.

    ``result`` stays ``None`` while a dispatched backend works outside the
    trainer lock; an inline backend fills it during ``run_once``.
    """

    batch: TrainingBatch
    result: TrainStepResult | None

    @property
    def batch_id(self) -> str:
        return self.batch.batch_id


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
        training_backend: TrainingBackend | None = None,
        candidate_evaluator: CandidateEvaluationPlugin | None = None,
        algorithm_state: Mapping[str, Any] | None = None,
        report_type: type[ReportBase] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        if training_backend is None and candidate_evaluator is not None:
            raise ValueError("candidate evaluation requires a training backend")
        processor = processor_factory(
            ProcessorContext(
                scenario=scenario,
                report_type=report_type,
                experiment_logger=(experiment_logger if experiment_logger is not None else NullExperimentLogger()),
            )
        )
        default_state = training_backend.initial_state() if training_backend is not None else {}
        initial_state = dict(default_state if algorithm_state is None else algorithm_state)
        return cls(
            scenario=scenario,
            records=records,
            processor=processor,
            training_backend=training_backend,
            candidate_evaluator=candidate_evaluator,
            state=initial_state,
        )

    def __init__(
        self,
        *,
        scenario: str,
        records: RecordStore,
        processor: DataProcessor,
        training_backend: TrainingBackend | None,
        candidate_evaluator: CandidateEvaluationPlugin | None,
        state: Mapping[str, Any],
    ) -> None:
        self._scenario = scenario
        self._records = records
        self._processor = processor
        self._training_backend = training_backend
        if training_backend is None:
            self._candidate_evaluator = None
        elif candidate_evaluator is not None:
            self._candidate_evaluator = candidate_evaluator
        else:
            self._candidate_evaluator = DefaultCandidateEvaluationPlugin(training_backend)
        self._state = dict(state)
        self._data_offset = 0
        self._data_sequence = 0
        self._pending: _PendingStep | None = None
        self._lock = Lock()

    @property
    def scenario(self) -> str:
        return self._scenario

    @property
    def processor(self) -> DataProcessor:
        return self._processor

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The report type selected by the recipe that built this trainer."""
        return self._processor.context.report_type

    @property
    def training_backend(self) -> TrainingBackend | None:
        return self._training_backend

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
        for what a training backend will receive; enforcing it at the only
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
                self._data_offset += 1
                self._data_sequence = sequence
                if self._processor.ready():
                    return

    def run_once(self, scenario_step: int = 0) -> TrainStepResult | None:
        """Consume available data and, with a training backend, prepare one step.

        Returns ``None`` when this trainer has no training backend (it
        only advances record consumption, because a non-training scenario still
        has to drain and compact its store) or when the processor is not yet
        ready to produce a batch.
        """
        if self._training_backend is not None and self._training_backend.dispatched:
            raise RuntimeError("dispatched training backends must reserve a batch before execution")
        with self._lock:
            if self._training_backend is None:
                self._consume_data()
                return None
            if self._pending is not None:
                result = self._pending.result
                if result is not None:
                    return result
                batch = self._pending.batch
            else:
                self._consume_data()
                if not self._processor.ready():
                    return None
                batch = self._build_validated_batch()
                self._pending = _PendingStep(batch=batch, result=None)
        # Local candidate generation and evaluation can take minutes. Keep the
        # batch reserved, but release the trainer lock so status remains live.
        execution = self._execute_backend_step(batch, scenario_step)
        if execution.outcome != "commit" or execution.result is None:
            raise RuntimeError(f"inline training backend returned {execution.outcome!r}")
        with self._lock:
            if self._pending is None or self._pending.batch_id != batch.batch_id:
                raise RuntimeError("inline trainer reservation changed while its backend was executing")
            self._pending.result = execution.result
            return execution.result

    def _execute_backend_step(self, batch: TrainingBatch, scenario_step: int) -> StepExecution:
        backend = self._training_backend
        if backend is None:
            raise RuntimeError("cannot execute a training step without a backend")
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
            return StepExecution("commit", backend.settle_step(prepared, decision))
        except BaseException:
            backend.abort_step(prepared)
            raise

    def _evaluate_candidate(self, candidate: UpdateCandidate) -> SelectionDecision:
        evaluator = self._candidate_evaluator
        if evaluator is None:
            raise RuntimeError("candidate preparation has no candidate evaluator")
        evaluation = evaluator.evaluate(candidate)
        decision = evaluator.decide(candidate, evaluation)
        if decision.evaluation is not evaluation:
            raise ValueError("candidate evaluator must retain the evaluation result supplied by Reef")
        return decision

    def reserve_training_batch(self) -> TrainingBatch | None:
        """Reserve one batch for a dispatched backend."""
        backend = self._training_backend
        if backend is None or not backend.dispatched:
            raise RuntimeError("trainer has no dispatched training backend")
        with self._lock:
            if self._pending is not None:
                return self._pending.batch
            self._consume_data()
            if not self._processor.ready():
                return None
            batch = self._build_validated_batch()
            self._pending = _PendingStep(batch=batch, result=None)
            return batch

    def execute_reserved_step(self, scenario_step: int) -> StepExecution:
        """Run the dispatched backend for the currently reserved batch."""
        with self._lock:
            if self._pending is None:
                raise RuntimeError("trainer has no reserved training batch")
            if self._pending.result is not None:
                return StepExecution("commit", self._pending.result)
            batch = self._pending.batch
        execution = self._execute_backend_step(batch, scenario_step)
        if execution.outcome == "commit":
            if execution.result is None:
                raise RuntimeError("commit execution must carry a training result")
            with self._lock:
                if self._pending is None or self._pending.batch_id != batch.batch_id:
                    raise RuntimeError("trainer reservation changed while its backend was executing")
                self._pending.result = execution.result
        return execution

    def commit(self) -> PreparedCommit:
        """Commit the pending result on the trainer side, without compacting.

        With a pending result this acknowledges its batch, advances the
        algorithm state, and computes what compaction would delete — but deletes
        nothing, so the caller can make a commit record durable first and then
        call :meth:`apply_compaction`. Without a pending result (a commit
        driven externally, e.g. a direct weight publication) nothing changes
        and the returned commit describes the current state.
        """
        with self._lock:
            if self._pending is None:
                return PreparedCommit(
                    algorithm_state=self.algorithm_state_dict(),
                    high_water_sequence=self._data_sequence,
                    high_water_offset=self._data_offset,
                    compacted_ids=frozenset(),
                )
            batch_id, result = self._pending.batch_id, self._pending.result
            if result is None:
                raise RuntimeError("trainer pending batch has no result")
            if not isinstance(result.state, Mapping):
                raise TypeError("training step state must be a mapping")
            consumed = self._processor.acknowledge(batch_id)
            retention = self._processor.retention_decision()
            compacted = retention.releasable_agent_record_ids - retention.protected_agent_record_ids
            self._state = dict(result.state)
            self._pending = None
            return PreparedCommit(
                algorithm_state=self.algorithm_state_dict(),
                high_water_sequence=self._data_sequence,
                high_water_offset=self._data_offset,
                compacted_ids=frozenset(compacted),
                consumed_ids=consumed,
                metrics=dict(result.metrics) or None,
                training_job_id=result.training_job_id,
            )

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

    def reject_pending(self, metrics: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            if self._pending is None:
                return
            batch_id = self._pending.batch_id
            self._processor.acknowledge(batch_id)
            retention = self._processor.retention_decision()
            compacted = frozenset(retention.releasable_agent_record_ids - retention.protected_agent_record_ids)
            self._records.compact(
                self.scenario,
                compacted,
                receipt_id=batch_id,
                receipt_metadata={"outcome": "stale", "metrics": dict(metrics or {})},
            )
            self._processor.compaction_applied(compacted)
            self._pending = None

    def apply_compaction(self, compacted_ids: frozenset[str]) -> None:
        """Physically delete the rows a prepared commit marked disposable."""
        if not compacted_ids:
            return
        with self._lock:
            self._records.compact(self.scenario, compacted_ids)
            self._processor.compaction_applied(compacted_ids)

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Notify the backend after ``state`` enters the durable commit log."""
        backend = self._training_backend
        if backend is not None:
            backend.commit_applied(state)

    def close(self) -> None:
        """Release the processor's resources; the trainer owns its lifecycle.

        Held under the trainer lock so a processor is never closed while a
        batch is being ingested or built on the training thread.
        """
        with self._lock:
            self._processor.close()

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
                        continue
                    if item.request_type in self.processor.required_request_types:
                        self._processor.ingest(item)

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
