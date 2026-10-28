"""The trainer: record consumption, training jobs or harness steps, and commit preparation.

A trainer pages records into its pipeline. A weight trainer hands each ready
job's data to its ``JobLoop``, which submits training jobs and turns their saved
checkpoints into commits (``reef.train.jobs``). A harness trainer runs each
step through its candidate backend: the backend prepares a candidate, the
trainer runs one candidate evaluator, and the backend settles the step.
Consumption progress is committed before pipeline memory is released. Storage
owns disk eviction independently; new training commits never retire record
bodies.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

from reef.core.evaluation import CandidateEvaluationPlugin, SelectionDecision, UpdateCandidate
from reef.core.job_data import JobOutcome, TrajectorySample
from reef.core.plugins import PluginConfigError
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.core.training_request import TrainingRequest
from reef.evaluation.evaluators import BackendAlwaysSelectPlugin
from reef.observability import ExperimentLogger, NullExperimentLogger
from reef.observability.operations import OperationMetrics
from reef.pipeline_context import PipelineContext
from reef.storage.records import RecordStore
from reef.storage.tasks import TaskStorage
from reef.storage.trajectories import TrajectoryStore, consumed_trajectory_id
from reef.train.backend import CandidateBackend, PreparedStep
from reef.train.jobs import JobLoop, JobLoopState
from reef.train.pipeline import InstructionFailure, TrainingPipeline
from reef.train.types import JobData, PreparedCommit, TrainStepResult


@dataclass
class _PendingStep:
    """One commit in preparation: the job data it consumes, if any, and once known its result.

    The trainer makes one when it reserves job data or when a running job's
    poll returns a result, and its commit path reads it. ``data`` is ``None``
    for a commit about a running job, which consumes no records; such a step
    is made only with its result, so ``data is None`` implies ``result is not
    None``. With data, ``result`` stays ``None`` while the step's work runs
    outside the trainer lock.
    """

    data: JobData | None
    result: TrainStepResult | None
    prepared_commit: PreparedCommit | None = None
    #: The outcome of the job whose end this commit records, which the pipeline learns once it is durable.
    ended: JobOutcome | None = None
    # Set once the pipeline has the job data back and the step consumed these ids on its own, so no acknowledgement.
    consumed_ids: frozenset[str] | None = None
    #: The values ``result`` held before commit metrics replaced it: a caller may still hold one.
    earlier_results: tuple[TrainStepResult, ...] = ()

    def __post_init__(self) -> None:
        if self.data is None and self.result is None:
            raise ValueError("a pending step without job data must carry its result")

    @property
    def data_id(self) -> str | None:
        return None if self.data is None else self.data.data_id


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


logger = logging.getLogger(__name__)


class Trainer:
    """Turn a scenario's records into job data, and each training job's progress into commits.

    A weight trainer submits jobs through its ``JobLoop``; a harness trainer runs a step on its candidate
    backend. A trainer with neither trains nothing: it reads every record and is never ready.
    """

    _DATA_READ_BATCH_SIZE = 256

    @classmethod
    def build(
        cls,
        scenario: str,
        records: RecordStore,
        *,
        pipeline_factory: Callable[[PipelineContext], TrainingPipeline],
        candidate_backend: CandidateBackend | None = None,
        candidate_evaluator: CandidateEvaluationPlugin | None = None,
        job_loop: JobLoop | None = None,
        algorithm_state: Mapping[str, Any] | None = None,
        report_type: type[ReportBase] | None = None,
        experiment_logger: ExperimentLogger | None = None,
        training_mode: str = "auto",
        task_storage: TaskStorage | None = None,
        default_job_samples: int | None = None,
    ) -> Trainer:
        """Build a trainer and its pipeline.

        A trainer trains through ``job_loop`` (weights) or ``candidate_backend``
        (harness steps), or through neither, when it only consumes records.
        ``default_job_samples`` is None or a positive int: ``evolution.batch_size``
        for harness evolution, and None for a weight trainer, whose
        deployment sets the size. The trainer's ``job_samples`` is the
        sample-to-job stage's ``job_samples``, which is ``default_job_samples``
        unless the stage config sets it; the stage checks its own size rules.
        A trainer that trains with no ``job_samples``, through a stage that
        trains more than its own job data, is a ``PluginConfigError``.
        """
        if candidate_backend is None and candidate_evaluator is not None:
            raise ValueError("candidate evaluation requires a candidate backend; a job loop holds its own")
        if candidate_evaluator is not None and not isinstance(candidate_evaluator, CandidateEvaluationPlugin):
            raise TypeError("candidate_evaluator must inherit CandidateEvaluationPlugin")
        trains = candidate_backend is not None or job_loop is not None
        pipeline = pipeline_factory(
            PipelineContext(
                scenario=scenario,
                report_type=report_type,
                experiment_logger=(experiment_logger if experiment_logger is not None else NullExperimentLogger()),
                training_mode=training_mode,
                task_storage=task_storage,
                # A trainer that trains nothing has no job size, so its pipeline keeps every sample.
                default_job_samples=default_job_samples if trains else None,
            )
        )
        if pipeline.training_mode != training_mode:
            pipeline.close()
            raise ValueError("pipeline_factory must preserve the requested training_mode")
        stage = pipeline.sample2job
        if trains and stage.job_samples is None and not stage.trains_own_job_data_only:
            pipeline.close()
            raise PluginConfigError(
                f"{type(stage).__name__} has no job_samples: set pipeline.sample2job.config.job_samples, "
                "the fewest samples one training job trains"
            )
        if candidate_backend is not None:
            default_state: Mapping[str, Any] = candidate_backend.initial_state()
        elif job_loop is not None:
            default_state = JobLoopState().to_dict()
        else:
            default_state = {}
        initial_state = dict(default_state if algorithm_state is None else algorithm_state)
        try:
            trainer = cls(
                scenario=scenario,
                records=records,
                pipeline=pipeline,
                candidate_backend=candidate_backend,
                candidate_evaluator=candidate_evaluator,
                job_loop=job_loop,
                state=initial_state,
            )
        except BaseException:
            pipeline.close()
            raise
        if job_loop is not None:
            restored = JobLoopState.from_dict(initial_state)
            if restored.job is None and restored.last_job is not None:
                # A restart can fall between the durable commit of a job's end and the pipeline learning it.
                pipeline.job_ended(restored.last_job)
        return trainer

    def __init__(
        self,
        *,
        scenario: str,
        records: RecordStore,
        pipeline: TrainingPipeline,
        candidate_backend: CandidateBackend | None,
        candidate_evaluator: CandidateEvaluationPlugin | None,
        state: Mapping[str, Any],
        job_loop: JobLoop | None = None,
    ) -> None:
        if candidate_backend is not None and job_loop is not None:
            raise ValueError("a trainer trains through a job loop or a candidate backend, not both")
        self._scenario = scenario
        self._records = records
        self._pipeline = pipeline
        self._candidate_backend = candidate_backend
        self._job_loop = job_loop
        if candidate_backend is None:
            self._candidate_evaluator = None
        elif candidate_evaluator is not None:
            self._candidate_evaluator = candidate_evaluator
        else:
            self._candidate_evaluator = BackendAlwaysSelectPlugin(candidate_backend)
        #: The fewest samples one job trains: automatic job data holds at least this many. None when the
        #: trainer trains nothing (it reads every record and is never ready) and when its stage trains only
        #: its own job data.
        self.job_samples = pipeline.sample2job.job_samples if self.trains else None
        self._state = dict(state)
        self._data_offset = 0
        self._data_sequence = 0
        #: The uploaded trajectories of the scenario, after ``read_uploaded_trajectories``.
        self.trajectory_store: TrajectoryStore | None = None
        #: The sequence of the last uploaded trajectory that the trainer read.
        self.trajectory_sequence = 0
        self.consumed_record_ids: set[str] = set()
        self.skipped_record_ids: set[str] = set()
        self._pending: _PendingStep | None = None
        # The result values of the last pending step a commit put on record: committing one again is a second step.
        self.recorded_results: tuple[TrainStepResult, ...] = ()
        self._lock = Lock()
        self.operations = OperationMetrics(("execution",))

    def operational_metrics(self) -> dict[str, float | int]:
        """Sample backlog and execution separately; a job of reserved data can be running.

        Unread records are not necessarily trainable. Pipeline counts retain
        their recipe-defined names rather than being guessed into a queue depth.
        A busy pipeline omits its gauges for this sample instead of blocking
        execution metrics behind record push or commit work.
        """
        loss = self._records.loss(self.scenario)
        values = {f"training/{key}": value for key, value in self.operations.snapshot().items()}
        values.update(
            {
                "records/evicted_count": loss.record_count,
                "records/evicted_body_bytes": loss.body_bytes,
                "records/data_incomplete": int(loss.record_count > 0),
            }
        )
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
            values["training/reserved_job_data"] = int(self._pending is not None and self._pending.data is not None)
            values["training/auto_enabled"] = int(self._pipeline.training_mode in {"auto", "hybrid"})
            pipeline_metrics = {
                **self._pipeline.status(),
                "job_samples": self.job_samples,
                **self._pipeline.operational_metrics(),
            }
            for name, value in pipeline_metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values[f"processor/{name}"] = int(value) if isinstance(value, bool) else value
        finally:
            self._lock.release()
        return values

    @property
    def scenario(self) -> str:
        return self._scenario

    def set_training_mode(self, training_mode: str) -> None:
        """Serialize mode selection with record push and job data reservation."""
        with self._lock:
            self._pipeline.set_training_mode(training_mode)

    def pending_instructions(self) -> int:
        """Instructions accepted and not yet consumed: the ones the pipeline buffers plus those unread in storage."""
        with self._lock:
            unread = self._records.count(
                self.scenario, request_type=RequestType.TRAIN, after_sequence=self._data_sequence
            )
            return self._pipeline.buffered_requests() + unread

    def fail_pending_instruction(self, error: str) -> bool:
        """Mark the reserved instruction as failed, so its next job data is a skip row; False when none is reserved."""
        with self._lock:
            pending = self._pending
            if pending is None or pending.result is not None or pending.data is None or pending.data.request is None:
                return False
            metadata = {} if self._candidate_backend is None else self._candidate_backend.failed_step_metrics()
            self._pipeline.mark_request_failed(pending.data.request.id, error, metadata)
            return True

    def instruction_failures(self) -> Mapping[str, InstructionFailure]:
        """The failed instructions this trainer still holds, for the trainer that replaces it."""
        with self._lock:
            return self._pipeline.request_failures()

    def set_instruction_failures(self, failures: Mapping[str, InstructionFailure]) -> None:
        """Carry failed instructions into this trainer's pipeline; a rebuilt scenario starts without them."""
        with self._lock:
            self._pipeline.set_request_failures(failures)

    @property
    def pipeline(self) -> TrainingPipeline:
        return self._pipeline

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        """Name the contract the scenario's ingress admits, when several trainers share one scenario."""
        with self._lock:
            self._pipeline.admit_reports_of(report_type)

    @property
    def candidate_backend(self) -> CandidateBackend | None:
        return self._candidate_backend

    @property
    def job_loop(self) -> JobLoop | None:
        return self._job_loop

    @property
    def trains(self) -> bool:
        """Whether this trainer trains: through a job loop or a candidate backend."""
        return self._job_loop is not None or self._candidate_backend is not None

    @property
    def job_running(self) -> bool:
        """Whether this trainer's committed state names a running job, which its worker polls."""
        with self._lock:
            return self._job_loop is not None and JobLoopState.from_dict(self._state).job is not None

    def needs_polling(self) -> bool:
        """Whether the trainer's worker polls at a fixed interval: a job runs, or the pipeline's own work runs.

        Otherwise the worker sleeps until the next accepted record.
        """
        with self._lock:
            if self._job_loop is not None:
                state = JobLoopState.from_dict(self._state)
                if state.job is not None:
                    return True
            return self._pipeline.background_jobs_running()

    def uploaded_sample(self, trajectory: TrajectorySample) -> TrajectorySample:
        """The training sample of an uploaded trajectory (``TrainingPipeline.uploaded_sample``), under the lock.

        Admission calls it from a request thread; the lock keeps the pipeline's stages, such as a tokenizer,
        to one thread at a time.
        """
        with self._lock:
            return self._pipeline.uploaded_sample(trajectory)

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

    def job_loop_required(self) -> JobLoop:
        """The job loop of a weight trainer; raise ``RuntimeError`` for a harness trainer."""
        if self._job_loop is None:
            raise RuntimeError("only a weight trainer has a job loop")
        return self._job_loop

    @property
    def has_pending_commit(self) -> bool:
        """Whether a commit is in preparation: reserved job data, or a job commit not yet recorded."""
        with self._lock:
            return self._pending is not None

    @property
    def pending_job_data(self) -> JobData | None:
        with self._lock:
            return None if self._pending is None else self._pending.data

    def job_data_ready(self) -> bool:
        """Whether reserved or buildable job data is waiting; a pending job commit holds no job data."""
        with self._lock:
            return (self._pending is not None and self._pending.data is not None) or self.pipeline_ready()

    def holds_untrained_records(self) -> bool:
        """Whether this trainer has records that its backend would still train on.

        That is the case when rows are unread, job data is ready or reserved,
        or a background job runs. A trainer that trains nothing (``trains``
        is False) has none. The rows that the pipeline keeps for later job data do
        not count, because a rebuilt trainer reads them again from storage.
        """
        if not self.trains:
            return False
        with self._lock:
            if self._pending is not None or self._pipeline.background_jobs_running() or self.pipeline_ready():
                return True
            trajectories = self.trajectory_store
            if (
                trajectories is not None
                and self._pipeline.accepts_trajectories
                and trajectories.count(after_sequence=self.trajectory_sequence) > 0
            ):
                return True
            return self._records.count(self.scenario, after_sequence=self._data_sequence) > 0

    def read_uploaded_trajectories(self, store: TrajectoryStore) -> None:
        """Read the uploaded trajectories of the scenario from ``store``, from the first one.

        The scenario factory calls this when it builds the trainer. An uploaded
        trajectory that a commit consumed is in ``consumed_record_ids`` after
        recovery, so the trainer skips it.
        """
        with self._lock:
            self.trajectory_store = store
            self.trajectory_sequence = 0

    def pipeline_status(self) -> Mapping[str, Any]:
        """Return caller-visible pipeline state under the pipeline's lock."""
        with self._lock:
            loss = self._records.loss(self.scenario)
            status = {**self._pipeline.status(), "job_samples": self.job_samples}
            if not loss.record_count:
                return status
            return {
                **status,
                "record_data_incomplete": loss.record_count > 0,
                "evicted_record_count": loss.record_count,
                "evicted_body_bytes": loss.body_bytes,
            }

    def pipeline_ready(self) -> bool:
        """Whether the pipeline holds the next job's data; the caller holds the lock. A trainer that trains nothing is never ready."""
        return self.trains and self._pipeline.ready(self.job_samples)

    def _build_validated_job_data(self) -> JobData:
        """Build the next job's data and hold the pipeline to its declared schema.

        ``Sample2Job.output_schema`` is the recipe's published contract for
        what the training component or candidate backend receives; enforcing it at the only place
        job data enters the trainer turns a drifting recipe into a loud error
        instead of a backend-side shape failure.
        """
        if not self.trains:
            raise RuntimeError("a trainer that trains nothing builds no job data")
        data = self._pipeline.build_job_data(self.job_samples)
        schema = self._pipeline.output_schema
        if not isinstance(data, schema):
            raise TypeError(
                f"{type(self._pipeline.sample2job).__name__} declared output_schema "
                f"{schema.__name__} but built {type(data).__name__}"
            )
        return data

    def push_retained(self, item: AgentRecord) -> None:
        """Consumption state is independent of record visibility in storage."""
        if item.agent_record_id in self.consumed_record_ids:
            return
        if item.request_type is RequestType.REPORT:
            if any(ref in self.consumed_record_ids for ref in item.references):
                self.skipped_record_ids.add(item.agent_record_id)
                return
            missing = [ref for ref in item.references if self._records.get(self.scenario, ref) is None]
            if missing:
                logger.warning(
                    "Skipping report %s in scenario %s: %d referenced records are no longer stored; "
                    "training data is incomplete",
                    item.agent_record_id,
                    self.scenario,
                    len(missing),
                )
                self.skipped_record_ids.add(item.agent_record_id)
                return
        if item.request_type in self.pipeline.required_request_types:
            self._pipeline.push(item)

    def _consume_data(self) -> None:
        # Uploaded trajectories go first in job data, so the trainer reads them before the records.
        while self.trajectory_store is not None and self._pipeline.accepts_trajectories:
            stored_trajectories = self.trajectory_store.page(
                after_sequence=self.trajectory_sequence, limit=self._DATA_READ_BATCH_SIZE
            )
            if not stored_trajectories:
                break
            for stored in stored_trajectories:
                self.trajectory_sequence = stored.sequence
                if consumed_trajectory_id(stored.trajectory_id) in self.consumed_record_ids:
                    continue
                try:
                    self._pipeline.push_trajectory(stored.trajectory_id, stored.trajectory)
                except ValueError:
                    # Admission rendered it; a config that changed since, such as a lower max_tokens, can refuse it.
                    logger.warning(
                        "uploaded trajectory %s of scenario %s cannot train and is skipped",
                        stored.trajectory_id,
                        self.scenario,
                        exc_info=True,
                    )
                    continue
                if self.pipeline_ready():
                    return
        while True:
            items = self._records.replay_page(
                self.scenario,
                after_sequence=self._data_sequence,
                limit=self._DATA_READ_BATCH_SIZE,
            )
            if not items:
                return
            for sequence, item in items:
                self.push_retained(item)
                self._data_offset += 1
                self._data_sequence = sequence
                if self.pipeline_ready():
                    return

    def run_once(
        self,
        scenario_step: int = 0,
        *,
        serving_runtime_load_id: str | None = None,
    ) -> TrainStepResult | None:
        """Consume available data and prepare the next commit, if any.

        Returns ``None`` when this trainer trains nothing (it only advances
        record consumption, because a non-training scenario still has to
        advance its record cursor), when no job data is ready, or while a job runs
        with nothing new to commit. ``serving_runtime_load_id`` is the version
        that serves the scenario now, for the staleness check of the pipeline.
        """
        with self._lock:
            if not self.trains:
                self._consume_data()
                return None
            if self._pending is not None and self._pending.result is not None:
                return self._pending.result
            observed_state = self._state
            loop_state = None if self._job_loop is None else JobLoopState.from_dict(observed_state)
            job = None if loop_state is None else loop_state.job
            data = None if job is not None else self.reserve_job_data(serving_runtime_load_id)
            if job is None and data is None:
                return None
        # Submitting a job, evaluating a checkpoint or preparing a harness candidate can take minutes.
        # Keep the job data reserved, but release the trainer lock so status remains live.
        with self.operations.measure("execution"):
            result: TrainStepResult | None
            ended: JobOutcome | None = None
            if data is None:
                polled = self.job_loop_required().poll(JobLoopState.from_dict(observed_state))
                result, ended = (None, None) if polled is None else (polled.result, polled.ended)
            elif self._job_loop is None:
                result = self._execute_backend_step(data, scenario_step)
            else:
                result = self._job_loop.submit(JobLoopState.from_dict(observed_state), data).result
        with self._lock:
            if data is None:
                # A rollback that ran meanwhile replaced the state this poll read: its result is stale.
                if result is None or self._state is not observed_state:
                    return None
                self._pending = _PendingStep(data=None, result=result, ended=ended)
            else:
                if self._pending is None or self._pending.data_id != data.data_id:
                    raise RuntimeError("trainer reservation changed while its step was being prepared")
                self._pending.result = result
            return result

    def _execute_backend_step(self, data: JobData, scenario_step: int) -> TrainStepResult:
        backend = self._candidate_backend
        if backend is None:
            raise RuntimeError("cannot execute a training step without a backend")
        request = data.request
        error = None if request is None else self._pipeline.request_failure(request.id)
        if request is not None and error is not None:
            # Committed without the backend: the failed instruction is consumed alone and the catalog row names why.
            return self._skip_failed_instruction(data, request, error)
        prepared = backend.prepare_step(data, self._state, scenario_step)
        if not isinstance(prepared, PreparedStep):
            raise TypeError(f"{type(backend).__name__}.prepare_step must return PreparedStep")
        if prepared.outcome == "skip":
            return TrainStepResult(prepared.state, prepared.metrics)
        candidate = prepared.candidate
        if not isinstance(candidate, UpdateCandidate):
            raise TypeError("candidate preparation must carry an UpdateCandidate")
        return self.settle_candidate(backend, prepared, candidate)

    def settle_candidate(
        self, backend: CandidateBackend, prepared: PreparedStep, candidate: UpdateCandidate
    ) -> TrainStepResult:
        """Evaluate ``prepared``'s candidate and settle the step; a failure aborts it at the backend."""
        try:
            decision = self._evaluate_candidate(candidate)
            return backend.settle_step(prepared, decision)
        except BaseException:
            backend.abort_step(prepared)
            raise

    def _skip_failed_instruction(self, data: JobData, request: TrainingRequest, error: str) -> TrainStepResult:
        """Consume the instruction alone; the units beside it go back to the pipeline for job data that a proposer reads."""
        with self._lock:
            pending = self._pending
            if pending is None or pending.data_id != data.data_id:
                raise RuntimeError("trainer reservation changed while its instruction was being skipped")
            metadata = dict(self._pipeline.request_failure_metrics(request.id))
            self._pipeline.release_job_data(data.data_id)
            pending.consumed_ids = self._pipeline.discard_request(request.id)
        metrics = {**metadata, "skipped": "instruction failed", "error": error}
        return TrainStepResult(dict(self._state), metrics)

    def _evaluate_candidate(self, candidate: UpdateCandidate) -> SelectionDecision:
        evaluator = self._candidate_evaluator
        if evaluator is None:
            raise RuntimeError("candidate preparation has no candidate evaluator")
        return evaluator.evaluate_and_decide(candidate)

    def reserve_job_data(self, serving_runtime_load_id: str | None) -> JobData | None:
        """Return the reserved job data, or reserve the next data when the pipeline is ready. The caller holds the lock.

        ``serving_runtime_load_id`` is the version that serves the scenario now.
        The staleness check of the pipeline measures the lag of each held set against it.
        """
        if self._pending is not None:
            return self._pending.data
        if serving_runtime_load_id is not None:
            self._pipeline.observe_serving_runtime_load_id(serving_runtime_load_id)
        self._consume_data()
        if not self.pipeline_ready():
            return None
        data = self._build_validated_job_data()
        self._pending = _PendingStep(data=data, result=None)
        return data

    @property
    def pending_prepared_commit(self) -> PreparedCommit | None:
        """The prepared commit of the pending step, once an attempt prepared it."""
        with self._lock:
            return None if self._pending is None else self._pending.prepared_commit

    def prepare_commit(self, result: TrainStepResult | None) -> PreparedCommit:
        """Prepare the pending result without exposing its state as committed.

        Acknowledging a pipeline mutates its in-memory job data bookkeeping, so
        the prepared value is cached and reused after a publication retry. The
        trainer's algorithm state and pending reservation remain unchanged
        until :meth:`commit` is called after the scenario's artifact and commit
        record settle.
        """
        with self._lock:
            if self._pending is None:
                if any(result is recorded for recorded in self.recorded_results):
                    raise RuntimeError("training result is on record already: committing it again is a second step")
                return PreparedCommit(
                    algorithm_state=self.algorithm_state_dict(),
                    high_water_sequence=self._data_sequence,
                    high_water_offset=self._data_offset,
                )
            if self._pending.result is not result:
                raise RuntimeError("training result does not match the pending step")
            if self._pending.prepared_commit is not None:
                return self._pending.prepared_commit
            data_id, result = self._pending.data_id, self._pending.result
            if result is None:
                raise RuntimeError("trainer pending job data has no result")
            if not isinstance(result.state, Mapping):
                raise TypeError("training step state must be a mapping")
            consumed = self._pending.consumed_ids
            if consumed is None:
                # A commit about a running job consumes no job data.
                consumed = frozenset() if data_id is None else self._pipeline.acknowledge(data_id)
            released = self._pipeline.releasable_record_ids()
            loss = self._records.loss(self.scenario)
            metrics = dict(result.metrics)
            if loss.record_count > 0:
                metrics.update({"records/evicted_count": loss.record_count, "records/data_incomplete": 1})
            request = None if self._pending.data is None else self._pending.data.request
            if request is not None:
                # The backend's own dict, when it wrote one, carries what its proposer added to ``requires``.
                metrics.setdefault("training_request", {"id": request.id, **request.to_dict()})
            prepared = PreparedCommit(
                algorithm_state=dict(result.state),
                high_water_sequence=self._data_sequence,
                high_water_offset=self._data_offset,
                consumed_ids=frozenset(consumed | released | self.skipped_record_ids),
                released_ids=released,
                metrics=metrics or None,
                training_job_id=result.training_job_id,
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
            self._pipeline.release_records(prepared.released_ids)
            if self._pending.ended is not None:
                # The job's end is durable: the pipeline learns its outcome, such as a task job's rewards.
                self._pipeline.job_ended(self._pending.ended)
            self._state = dict(prepared.algorithm_state)
            self.consumed_record_ids.update(prepared.consumed_ids)
            self.skipped_record_ids.difference_update(prepared.consumed_ids)
            held = self._pending.result
            self.recorded_results = (*self._pending.earlier_results, *(() if held is None else (held,)))
            self._pending = None

    def add_commit_metrics(self, result: TrainStepResult, metrics: Mapping[str, Any]) -> TrainStepResult:
        """Attach provider correlation fields to the exact pending result.

        The committer calls this before preparing the durable commit. Updating
        the pending value under the trainer lock keeps the commit record and
        the storage-observed experiment event on one immutable result snapshot.
        """
        if not metrics:
            return result
        with self._lock:
            if self._pending is None or self._pending.result is None:
                raise RuntimeError("cannot annotate a training result without a pending step")
            if self._pending.result is not result:
                raise RuntimeError("cannot annotate a result that is not the pending training step")
            annotated = replace(result, metrics={**dict(result.metrics), **dict(metrics)})
            self._pending.earlier_results = (*self._pending.earlier_results, result)
            self._pending.result = annotated
            return annotated

    def cancel_job(self) -> None:
        """Cancel the running training job at its component; nothing for a trainer without one."""
        if self._job_loop is None:
            return
        with self._lock:
            state = JobLoopState.from_dict(self._state)
        self._job_loop.cancel(state)

    def import_rollback_release(self, release: Path) -> str | None:
        """Import a rolled-back release into the weight trainer's component and return it as a start.

        The component raises ``ValueError`` for weights it cannot start from.
        """
        if self._job_loop is None:
            raise RuntimeError("only a weight trainer imports a rolled-back release")
        return self._job_loop.import_start(release)

    def apply_rollback_start(self, start: str | None) -> None:
        """Start the next job from an imported rolled-back release, after the rollback committed.

        The running job is cancelled. The trainer's next commit records the
        new start; until then, recovery reads it back from the rollback.
        """
        if self._job_loop is None:
            raise RuntimeError("only a weight trainer takes a rolled-back start")
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("cannot restore a start while a commit is pending")
            self._state = self._job_loop.restored(JobLoopState.from_dict(self._state), start).to_dict()

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        """Notify the backend after ``state`` enters the durable commit history."""
        backend = self._candidate_backend
        if backend is not None:
            backend.commit_applied(state)

    def close(self) -> None:
        """Release pipeline and backend resources owned by the trainer.

        Held under the trainer lock so a pipeline is never closed while a
        record is being pushed or job data is built on the training thread.
        """
        with self._lock:
            try:
                self._pipeline.close()
            finally:
                if self._candidate_backend is not None:
                    self._candidate_backend.close()

    def replay_retained(self, *, up_to_sequence: int, consumed_ids: frozenset[str]) -> None:
        """Rebuild pipeline memory from retained rows at or below a watermark.

        The committed high-water mark is a consumption cursor, not a liveness
        boundary: consumption stops the moment job data is ready, so the cursor
        passes rows of the next, still-incomplete step, and retention keeps
        them stored. Only pipeline memory knew about them, and a rebuilt
        pipeline starts empty: without this replay, a report arriving after
        recovery waits forever on a reference that can never be pushed
        again. ``consumed_ids`` names the rows that committed job data consumed —
        the commit history records them per step because retention may keep a
        consumed row stored (audit-only retention is contract-legal), and a
        row that committed job data consumed must never train twice. Skipping them
        while replaying the retained prefix reconstructs the live state the
        crash destroyed and nothing more; consumption accounting stays
        untouched, since the recovered mark already covers every replayed row.
        """
        if up_to_sequence < 0:
            raise ValueError("up_to_sequence must be non-negative")
        with self._lock:
            self.consumed_record_ids.update(consumed_ids)
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
                    self.push_retained(item)

    def restore_record_progress(self, *, after_sequence: int, offset: int) -> None:
        """Resume consumption from a recovered commit record's high-water mark.

        Without this, a recovered trainer would page the record store from
        sequence 0 and push again the consumed rows that the retention policy
        protected from deletion — re-training them. Restoring the watermark makes
        replay start exactly where the recovered step left off.
        """
        if after_sequence < 0 or offset < 0:
            raise ValueError("record progress must be non-negative")
        with self._lock:
            self._data_sequence = after_sequence
            self._data_offset = offset
