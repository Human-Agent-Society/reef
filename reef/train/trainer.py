"""The trainer: record consumption, training jobs, and commit preparation.

A trainer pages records into its pipeline. A trainer that trains hands each
ready job's data to its ``JobLoop``, which submits training jobs and turns their
saved checkpoints into commits (``reef.train.jobs``): weight jobs, and the
harness steps of the harness training component. Consumption progress is
committed before pipeline memory is released. Storage owns disk eviction
independently; new training commits never retire record bodies.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import CommitPendingError, TrainingStartRefusedError
from reef.core.plugins import PluginConfigError
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.core.training_request import TrainingRequest
from reef.observability import ExperimentLogger, NullExperimentLogger
from reef.observability.operations import OperationMetrics
from reef.pipeline_context import PipelineContext
from reef.record2sample.service import ProducedSamples, Selection
from reef.record2sample.staleness import ServedArtifacts
from reef.storage.commits import CommitRecord
from reef.storage.record_items import used_record, used_records
from reef.storage.records import RecordStore
from reef.storage.samples import SAMPLE_KINDS, SampleKind, TrainerSamples, memory_samples
from reef.storage.tasks import TaskStorage
from reef.train.component import JobCheckPendingError, ScenarioRoot
from reef.train.jobs import JobLoop, JobLoopState
from reef.train.pipeline import TrainingPipeline
from reef.train.types import JobData, PreparedCommit, TrainingCommit


@dataclass(frozen=True)
class JobDataReservation:
    """Job data held while submission runs outside the trainer lock.

    Values: ``data`` is JobData. ``consumed_ids`` is None while the pipeline holds the reservation. Otherwise,
    ``data.request`` exists and ``consumed_ids == frozenset({data.request.id})``: a failed request released
    its samples and consumed only that request.
    Purpose: Trainer.reserve_job_data makes this phase; submit_job replaces it with PendingTrainingResult once
    submission or request skipping returns. It contains no training result or prepared commit.
    """

    data: JobData
    consumed_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, JobData):
            raise TypeError("a reservation needs JobData")
        if self.consumed_ids is not None and (
            not isinstance(self.consumed_ids, frozenset)
            or self.data.request is None
            or self.consumed_ids != frozenset({self.data.request.id})
        ):
            raise ValueError("consumed_ids must name only the reserved request, or be None")

    @property
    def data_id(self) -> str:
        """The ID of the reserved data."""
        return self.data.data_id


@dataclass(frozen=True)
class PendingTrainingResult:
    """A completed submission or job update awaiting its durable scenario commit.

    Values: ``result`` is TrainingCommit. ``data`` is the reserved JobData for a submission or None for a polled
    job update. ``consumed_ids`` is None,
    or ``data.request`` exists and ``consumed_ids == frozenset({data.request.id})`` for a skipped request.
    ``prepared_commit`` is None until Trainer.prepare_commit prepares the commit. When set,
    ``result.state`` is a mapping, ``prepared_commit.algorithm_state == result.state`` and
    ``prepared_commit.training_job_id == result.training_job_id``.
    Purpose: Trainer.poll_jobs and Trainer.submit_job make this phase. prepare_commit acknowledges its data once and records the snapshot,
    which a retry of the preparation returns. Trainer.settle_commit reads it only after scenario settlement and then clears
    the pending phase.
    """

    data: JobData | None
    result: TrainingCommit
    consumed_ids: frozenset[str] | None = None
    prepared_commit: PreparedCommit | None = None

    def __post_init__(self) -> None:
        if self.data is not None and not isinstance(self.data, JobData):
            raise TypeError("data must be JobData or None")
        if not isinstance(self.result, TrainingCommit):
            raise TypeError("a pending result needs TrainingCommit")
        if self.consumed_ids is not None and (
            self.data is None
            or not isinstance(self.consumed_ids, frozenset)
            or self.data.request is None
            or self.consumed_ids != frozenset({self.data.request.id})
        ):
            raise ValueError("consumed_ids must name only the reserved request, or be None")
        if self.prepared_commit is not None:
            if not isinstance(self.result.state, Mapping):
                raise TypeError("a prepared training result needs a state mapping")
            if self.prepared_commit.algorithm_state != self.result.state:
                raise ValueError("prepared state must match the training result state")
            if self.prepared_commit.training_job_id != self.result.training_job_id:
                raise ValueError("prepared job ID must match the training result job ID")

    @property
    def data_id(self) -> str | None:
        """The ID of the submission's data, or None for a polled job update."""
        return None if self.data is None else self.data.data_id


@dataclass(frozen=True)
class ComponentTrainer:
    """One trainer and the snapshot component it evolves."""

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

    A trainer that trains submits jobs through its ``JobLoop``. A trainer without one trains nothing: it reads
    every record and is never ready.
    """

    _DATA_READ_BATCH_SIZE = 256

    @classmethod
    def build(
        cls,
        scenario: str,
        records: RecordStore,
        *,
        pipeline_factory: Callable[[PipelineContext], TrainingPipeline],
        job_loop: JobLoop | None = None,
        root: ScenarioRoot | None = None,
        algorithm_state: Mapping[str, Any] | None = None,
        report_type: type[ReportBase] | None = None,
        experiment_logger: ExperimentLogger | None = None,
        training_mode: str = "auto",
        task_storage: TaskStorage | None = None,
        samples: TrainerSamples | None = None,
        job_parameters: Mapping[SampleKind, Mapping[str, Any]] | None = None,
    ) -> Trainer:
        """Build a trainer and its pipeline.

        A trainer trains through ``job_loop`` (weight jobs, or harness steps as jobs of the harness training
        component), or without one, when it only consumes records.
        The trainer's ``job_samples`` is the sample-to-job stage's
        ``job_samples``: the stage config's value, or the stage's own default,
        and the stage checks its own size rules. A trainer that trains with no
        ``job_samples`` is a ``PluginConfigError``.
        ``job_parameters`` is the parameters of each job of ``job_loop``, by
        the kind of its samples (``PipelineContext.job_parameters``); the
        pipeline's stage gives them with each job's data. ``None`` gives
        every kind empty parameters, so the training component's defaults
        apply. ``algorithm_state`` is the trainer's committed state, or
        ``None`` for a trainer that has none. Such a trainer that trains
        starts its first job from ``root``, its scenario's root snapshot
        (``TrainingComponent.root_start``); ``root`` is then required, and a
        root that the component cannot start from raises
        ``TrainingStartRefusedError``.
        """
        trains = job_loop is not None
        if samples is None and job_loop is not None:
            # Without a scenario session, the samples are in memory. Their task files go to the store that the
            # jobs' data directories name them in.
            samples = memory_samples(scenario, job_loop.data_store.task_files).for_trainer("")
        pipeline = pipeline_factory(
            PipelineContext(
                scenario=scenario,
                report_type=report_type,
                experiment_logger=(experiment_logger if experiment_logger is not None else NullExperimentLogger()),
                training_mode=training_mode,
                task_storage=task_storage,
                samples=samples,
                # None gives every kind empty parameters: the training component's own defaults.
                job_parameters={kind: {} for kind in SAMPLE_KINDS} if job_parameters is None else job_parameters,
            )
        )
        if pipeline.training_mode != training_mode:
            pipeline.close()
            raise ValueError("pipeline_factory must preserve the requested training_mode")
        stage = pipeline.sample2job
        if trains and stage.job_samples is None:
            pipeline.close()
            raise PluginConfigError(
                f"{type(stage).__name__} has no job_samples: set pipeline.sample2job.config.job_samples, "
                "the fewest samples one training job trains"
            )
        try:
            initial_state: dict[str, Any]
            if algorithm_state is not None:
                initial_state = dict(algorithm_state)
            elif job_loop is None:
                initial_state = {}
            elif root is None:
                raise ValueError(f"a new trainer of scenario {scenario!r} starts from the scenario's root: give root")
            else:
                try:
                    start = job_loop.component.root_start(root)
                except ValueError as exc:
                    raise TrainingStartRefusedError(
                        f"scenario {scenario!r} cannot train from its root snapshot {root.artifact.ref.revision}: {exc}"
                    ) from exc
                initial_state = JobLoopState(start).to_dict()
            trainer = cls(
                scenario=scenario,
                records=records,
                pipeline=pipeline,
                job_loop=job_loop,
                state=initial_state,
            )
        except BaseException:
            pipeline.close()
            raise
        return trainer

    def __init__(
        self,
        *,
        scenario: str,
        records: RecordStore,
        pipeline: TrainingPipeline,
        state: Mapping[str, Any],
        job_loop: JobLoop | None = None,
    ) -> None:
        self._scenario = scenario
        self._records = records
        self._pipeline = pipeline
        self._job_loop = job_loop
        #: The fewest samples one job trains: automatic job data holds at least this many. None when the
        #: trainer trains nothing (it reads every record and is never ready).
        self.job_samples = pipeline.sample2job.job_samples if self.trains else None
        self._state = dict(state)
        self._data_offset = 0
        self._data_sequence = 0
        # The records of other scenarios that this one uses. Commits name the ones that trained, so a new process reads
        # them from the start and ``push_retained`` skips those.
        self._used_sequence = 0
        self._used_read = 0
        self.consumed_record_ids: set[str] = set()
        self.skipped_record_ids: set[str] = set()
        #: The record IDs that a commit of this trainer named; the next commit names the other consumed records.
        self.committed_record_ids: set[str] = set()
        #: The number of job ends that the pipeline has read (``read_job_ends``). A restart can fall between the
        #: durable commit of a job's end and the pipeline reading it, so a new pipeline with no running job reads the
        #: last end again.
        self.job_ends_read = 0
        if job_loop is not None:
            restored = JobLoopState.from_dict(self._state)
            self.job_ends_read = restored.finished_jobs - (not restored.jobs and restored.last_job is not None)
        self.pending_commit: JobDataReservation | PendingTrainingResult | None = None
        #: Whether the last job that the trainer tried to submit waits for the training component's check
        #: (``JobCheckPendingError``): the worker then polls, and the next poll submits the job again.
        self.job_check_waits = False
        # The result of the last pending commit that went on record: committing it again is a second commit.
        self.recorded_result: TrainingCommit | None = None
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
        if not self._lock.acquire(blocking=False):
            return values
        try:
            values["records/unread_count"] = self._records.count(self.scenario, after_sequence=self._data_sequence)
            oldest = self._records.replay_page(self.scenario, after_sequence=self._data_sequence, limit=1)
            values["records/oldest_unread_age_seconds"] = (
                max(0.0, time.time() - oldest[0][1].created_at) if oldest else 0.0
            )
            values["training/reserved_job_data"] = int(
                self.pending_commit is not None and self.pending_commit.data is not None
            )
            values["training/auto_enabled"] = int(self._pipeline.training_mode == "auto")
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
        """Serialize mode selection with record push and job data reservation (``TrainingPipeline.set_training_mode``)."""
        with self._lock:
            self._pipeline.set_training_mode(training_mode)

    def consumed_among(self, ids: Iterable[str]) -> frozenset[str]:
        """The IDs among ``ids`` that a commit of this trainer consumed, or that its recovery read as consumed."""
        with self._lock:
            return frozenset(ids) & self.consumed_record_ids

    def queue_requests(self, requests: Iterable[TrainingRequest], failures: Mapping[str, str] | None = None) -> None:
        """Give the pipeline the scenario's queued requests (``TrainingPipeline.queue_requests``)."""
        with self._lock:
            self._pipeline.queue_requests(requests, failures)

    def fail_pending_request(self, error: str) -> str | None:
        """Mark the reserved request as failed, so its next job data is a skip row; return its ID, or None when no
        request is reserved. ``Scenario.fail_pending_request`` stores the failure."""
        with self._lock:
            pending = self.pending_commit
            if not isinstance(pending, JobDataReservation) or pending.data.request is None:
                return None
            self._pipeline.mark_request_failed(pending.data.request.id, error)
            return pending.data.request.id

    @property
    def pipeline(self) -> TrainingPipeline:
        return self._pipeline

    @property
    def accepts_imported_tasks(self) -> bool:
        """Whether the trainer's pipeline trains the scenario's imported train tasks
        (``TrainingPipeline.accepts_imported_tasks``). Admission refuses a train-split task upload when no trainer
        of the scenario accepts them.
        """
        return self._pipeline.accepts_imported_tasks

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        """Name the contract the scenario's ingress admits, when several trainers share one scenario."""
        with self._lock:
            self._pipeline.admit_reports_of(report_type)

    @property
    def job_loop(self) -> JobLoop | None:
        return self._job_loop

    @property
    def trains(self) -> bool:
        """Whether this trainer trains: through a job loop."""
        return self._job_loop is not None

    @property
    def job_running(self) -> bool:
        """Whether this trainer's committed state names a running job, which its worker polls."""
        with self._lock:
            return self._job_loop is not None and bool(JobLoopState.from_dict(self._state).jobs)

    def accepts_job(self) -> bool:
        """Whether the trainer submits its next job now (``JobLoop.accepts_job``). The next job continues the
        training start unless it is the job of a request that names ``train_from``. The caller holds the lock."""
        if self._job_loop is None:
            return False
        request = self._pipeline.next_request()
        continues = request is None or request.train_from is None
        return self._job_loop.accepts_job(JobLoopState.from_dict(self._state), continues=continues)

    def job_slot_free(self) -> bool:
        """Whether ``reserve_job`` can give job data now: data is reserved, or the trainer accepts a job."""
        with self._lock:
            return isinstance(self.pending_commit, JobDataReservation) or (
                self.pending_commit is None and self.accepts_job()
            )

    def needs_polling(self) -> bool:
        """Whether the trainer's worker polls at a fixed interval: a job runs, a job waits for its check, or the
        pipeline's own work runs.

        Otherwise the worker sleeps until the next accepted record.
        """
        with self._lock:
            if self.job_check_waits:
                return True
            if self._job_loop is not None and JobLoopState.from_dict(self._state).jobs:
                return True
            return self._pipeline.background_jobs_running()

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
        """The job loop of a trainer that trains; raise ``RuntimeError`` for one that does not."""
        if self._job_loop is None:
            raise RuntimeError("only a trainer that trains has a job loop")
        return self._job_loop

    @property
    def has_pending_commit(self) -> bool:
        """Whether a commit is in preparation: reserved job data, or a job commit not yet recorded."""
        with self._lock:
            return self.pending_commit is not None

    @property
    def pending_job_data(self) -> JobData | None:
        with self._lock:
            return None if self.pending_commit is None else self.pending_commit.data

    def job_data_ready(self) -> bool:
        """Whether reserved or buildable job data is waiting; a pending job commit holds no job data."""
        with self._lock:
            if self.pending_commit is not None and self.pending_commit.data is not None:
                return True
            return self.trains and self._pipeline.holds_job_data(self.job_samples)

    def holds_untrained_records(self) -> bool:
        """Whether this trainer has records that its jobs would still train on.

        That is the case when rows are unread, job data is ready or reserved,
        or a background job runs. A trainer that trains nothing (``trains``
        is False) has none. The rows that the pipeline keeps for later job data do
        not count, because a rebuilt trainer reads them again from storage.
        """
        if not self.trains:
            return False
        with self._lock:
            if (
                self.pending_commit is not None
                or self._pipeline.background_jobs_running()
                or self._pipeline.holds_job_data(self.job_samples)
            ):
                return True
            if self._pipeline.has_unread_samples():
                return True
            if self._records.count(self.scenario, after_sequence=self._data_sequence) > 0:
                return True
            samples = self._pipeline.store.store
            used = used_records(
                samples.data, samples.owner, self.scenario, after_sequence=self._used_sequence, limit=1
            )
            return bool(used[1])

    def next_job_listing(self) -> Mapping[str, Any]:
        """The trainer's training mode and the samples of its next automatic job (``TrainingPipeline.next_job_listing``)."""
        with self._lock:
            self._pipeline.read_store()
            return {"training_mode": self._pipeline.training_mode, "samples": list(self._pipeline.next_job_listing())}

    def check_request(self, request: TrainingRequest) -> None:
        """Raise ``ValueError`` unless this trainer can run ``request`` (``TrainingPipeline.check_request``)."""
        with self._lock:
            self._pipeline.check_request(request)

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

    def push_retained(self, item: AgentRecord) -> None:
        """Consumption state is independent of record visibility in storage."""
        if item.agent_record_id in self.consumed_record_ids:
            return
        if item.request_type is RequestType.REPORT:
            if any(ref in self.consumed_record_ids for ref in item.references):
                self.skipped_record_ids.add(item.agent_record_id)
                return
            missing = [
                ref
                for ref in item.references
                if self._records.read(self.scenario, ref) is None and self.used_record(ref) is None
            ]
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

    def produce_samples(
        self, selection: Selection | None = None, served: ServedArtifacts | None = None
    ) -> ProducedSamples:
        """Read the job ends of this trainer's state, judge the stored samples against ``served``
        (``TrainingPipeline.mark_stale``; nothing is stale without it), then push every record that arrived since the last
        call into the pipeline and make samples of the formed record sets, which are judged as they are stored.
        Return whether the trainer received new input (samples, requests, or a new judgment), whether background
        work of the pipeline still runs, and when a set that waits for time alone is due, so the samples service comes
        back. ``selection`` gets the counts so far after each page of records and after the stage runs, and names the
        task in the samples that the step stores."""
        with self._lock:
            if not self.trains:
                return ProducedSamples(changed=False, background=False)
            self.read_job_end(JobLoopState.from_dict(self._state))
            # Another samples process can have stored the samples of later records since this trainer loaded;
            # the records that stored samples consumed never form a record set again.
            self.consumed_record_ids.update(self._pipeline.store.consumed_record_ids())
            before_records = self._data_offset + self._used_read
            before_samples = self._pipeline.stored_sample_count

            def report() -> None:
                if selection is not None:
                    selection.progress(
                        self._data_offset + self._used_read - before_records,
                        self._pipeline.stored_sample_count - before_samples,
                    )

            # Each write of the step checks first that the selection's worker still holds it.
            self._pipeline.selection = selection
            try:
                # The samples service judges the stored samples first, so that the samples that this step stores are
                # judged against the same artifacts when they are stored.
                judged = self._pipeline.mark_stale(ServedArtifacts() if served is None else served)
                self._pipeline.selection_task = None if selection is None else selection.task_id
                try:
                    first_sequence = self.read_records(report)
                    stored = self._pipeline.produce()
                finally:
                    self._pipeline.selection_task = None
                report()
                self._pipeline.settle_ended_records(self.committed_record_ids)
                self.skipped_record_ids -= self.committed_record_ids
                if self.skipped_record_ids:
                    self._pipeline.check_lease()
                    self._pipeline.store.end_records(self.skipped_record_ids)
            finally:
                self._pipeline.selection = None
            due = self._pipeline.next_due()
            return ProducedSamples(
                changed=stored or judged,
                background=self._pipeline.background_jobs_running(),
                records_read=self._data_offset + self._used_read - before_records,
                samples_created=self._pipeline.stored_sample_count - before_samples,
                due_in_seconds=None if due is None else max(0.0, due - time.monotonic()),
                record_sequences=None if first_sequence is None else (first_sequence, self._data_sequence),
            )

    def read_commits(self, records: Iterable[CommitRecord]) -> None:
        """Read ``records``, this trainer's newer training commits in commit number order: give the pipeline each job end that
        it has not read, and hold the record IDs that they name. The samples service calls it
        (``Scenario.produce_samples``), also when control runs in another process and this trainer's own state stays
        behind."""
        with self._lock:
            for record in records:
                self.read_job_end(JobLoopState.from_dict(record.algorithm_state or {}))
                self.consumed_record_ids.update(record.consumed_ids)
                self.committed_record_ids.update(record.consumed_ids)

    def read_job_end(self, state: JobLoopState) -> None:
        """Give the pipeline ``state``'s last job end unless it has read that end. The caller holds the lock."""
        if state.finished_jobs > self.job_ends_read and state.last_job is not None:
            self.job_ends_read = state.finished_jobs
            self._pipeline.job_ended(state.last_job)

    def read_new_records(self) -> None:
        """Move the record cursor of a trainer that trains nothing past the new records, which its pipeline ends. The
        dispatcher calls it when records arrive."""
        with self._lock:
            self.read_records()

    def read_records(self, page_read: Callable[[], None] | None = None) -> int | None:
        """Push every record after the record cursor into the pipeline and move the cursor; call ``page_read`` after
        each page. Return the data sequence of the first record of this scenario that it read, or None. The samples
        service calls it for a trainer that trains (``produce_samples``); ``read_new_records`` calls it for one that
        trains nothing. The caller holds the lock."""
        self.read_used_records()
        first: int | None = None
        while True:
            items = self._records.replay_page(
                self.scenario,
                after_sequence=self._data_sequence,
                limit=self._DATA_READ_BATCH_SIZE,
            )
            if not items:
                return first
            for sequence, item in items:
                self._pipeline.note_read(item.agent_record_id)
                self.push_retained(item)
                self._data_offset += 1
                self._data_sequence = sequence
                first = sequence if first is None else first
            if page_read is not None:
                page_read()

    def used_record(self, record_id: str) -> AgentRecord | None:
        """The record item of another scenario that this scenario uses, read as its record, or None."""
        samples = self._pipeline.store.store
        return used_record(samples.data, samples.owner, self.scenario, record_id)

    def read_used_records(self) -> None:
        """Push the record items of other scenarios that this scenario uses and that this process has not read. The
        caller holds the lock."""
        samples = self._pipeline.store.store
        while True:
            cursor, records = used_records(
                samples.data,
                samples.owner,
                self.scenario,
                after_sequence=self._used_sequence,
                limit=self._DATA_READ_BATCH_SIZE,
            )
            self._used_sequence = cursor
            for item in records:
                self._pipeline.note_read(item.agent_record_id)
                self.push_retained(item)
                self._used_read += 1
            if not records:
                return

    def poll_jobs(self) -> TrainingCommit | None:
        """Read the progress of the running jobs. Return the next commit about one of them, or None: when a result
        waits for its commit or job data is reserved, when no job runs, and while every job runs with nothing new."""
        with self._lock:
            if self._job_loop is None or self.pending_commit is not None:
                return None
            observed_state = self._state
        # Reading a job's checkpoints can take a while: the trainer lock stays free, so status remains live.
        with self.operations.measure("execution"):
            polled = self._job_loop.poll(JobLoopState.from_dict(observed_state))
        with self._lock:
            # A new training start, or a reservation, since the poll read the state: the next cycle polls again.
            if polled is None or self._state is not observed_state or self.pending_commit is not None:
                return None
            self.pending_commit = PendingTrainingResult(data=None, result=polled.result)
            return polled.result

    def pending_result(self) -> TrainingCommit | None:
        """The result that waits for its commit, or None."""
        with self._lock:
            return self.pending_commit.result if isinstance(self.pending_commit, PendingTrainingResult) else None

    def reserve_job(self, *, automatic_jobs: bool = True) -> JobData | None:
        """Reserve the data of the job that the trainer submits next, or return the reserved data. Return None when
        the trainer accepts no job now (``accepts_job``), when no job data is ready, and while a result waits for
        its commit. Without ``automatic_jobs``, only a request's job data is ready."""
        with self._lock:
            if isinstance(self.pending_commit, PendingTrainingResult):
                return None
            if self.pending_commit is None and not self.accepts_job():
                return None
            return self.reserve_job_data(automatic_jobs)

    def submit_job(
        self, data: JobData, *, start: str | None = None, train_from_artifact_id: str | None = None
    ) -> TrainingCommit | None:
        """Submit the job of ``data``, which ``reserve_job`` gave, and return its commit. Return None while the
        training component waits for the check of the job (``JobCheckPendingError``): nothing is submitted, and the
        reservation is released without consuming a sample or a request, so the scenario's other operations and the
        commits of running jobs go on. The worker polls (``needs_polling``), and the next poll reserves the data
        and submits again: the same data names the same job, whose check then has its answer.

        The job of a request that names ``train_from`` starts from ``start``, what ``import_start`` gave for that
        artifact. Any other job gives no ``start`` and continues the training start; ``train_from_artifact_id`` is
        then the artifact of the training start, which the job records for the job listing. A failed request
        (``request_failure``) submits no job: its commit consumes the request alone.
        """
        with self._lock:
            observed_state = self._state
        # Submitting a job writes its data directory, which can take a while.
        # Keep the job data reserved, but release the trainer lock so status remains live.
        with self.operations.measure("execution"):
            failure = None if data.request is None else self._pipeline.request_failure(data.request.id)
            if data.request is not None and failure is not None:
                # Committed without a job: the failed request is consumed alone and the catalog row names why.
                result = self.skip_failed_request(data, data.request, failure)
            else:
                try:
                    result = (
                        self.job_loop_required()
                        .submit(
                            JobLoopState.from_dict(observed_state),
                            data,
                            self._pipeline.sample2job.job_parameters(data),
                            start=start,
                            train_from=None if start is None or data.request is None else data.request.train_from,
                            train_from_artifact_id=train_from_artifact_id,
                        )
                        .result
                    )
                except JobCheckPendingError:
                    with self._lock:
                        reservation = self.pending_commit
                        if isinstance(reservation, JobDataReservation) and reservation.data_id == data.data_id:
                            self._pipeline.release_job_data(data.data_id)
                            self.pending_commit = None
                        self.job_check_waits = True
                    return None
                except BaseException:
                    with self._lock:
                        self.job_check_waits = False
                    raise
        with self._lock:
            self.job_check_waits = False
            reservation = self.pending_commit
            if not isinstance(reservation, JobDataReservation) or reservation.data_id != data.data_id:
                raise RuntimeError("trainer reservation changed while its step was being prepared")
            self.pending_commit = PendingTrainingResult(
                data=data, result=result, consumed_ids=reservation.consumed_ids
            )
            return result

    def advance_job(self, *, train_from_artifact_id: str | None = None) -> TrainingCommit | None:
        """Move the trainer's jobs forward and return the next commit, if any: the result that waits for its commit,
        else news of a running job (``poll_jobs``), else the submission of the next job, which continues the
        training start (``reserve_job``, ``submit_job``). A scenario gives the job of a request that names
        ``train_from`` its own start (``Scenario.prepare_job``)."""
        result = self.pending_result() or self.poll_jobs()
        if result is not None or not self.trains:
            return result
        data = self.reserve_job()
        return None if data is None else self.submit_job(data, train_from_artifact_id=train_from_artifact_id)

    def skip_failed_request(self, data: JobData, request: TrainingRequest, error: str) -> TrainingCommit:
        """Consume the request alone; the samples of its job data stay untrained."""
        with self._lock:
            pending = self.pending_commit
            if not isinstance(pending, JobDataReservation) or pending.data_id != data.data_id:
                raise RuntimeError("trainer reservation changed while its request was being skipped")
            self._pipeline.release_job_data(data.data_id)
            self.pending_commit = replace(pending, consumed_ids=self._pipeline.discard_request(request.id))
        return TrainingCommit(dict(self._state), {"skipped": "request failed", "error": error})

    def reserve_job_data(self, automatic_jobs: bool = True) -> JobData | None:
        """Return the reserved job data, or reserve the next data when the pipeline is ready. The caller holds the
        lock."""
        if self.pending_commit is not None:
            return self.pending_commit.data
        # The samples service makes the samples (``produce_samples``); the cycle only reads the sample store.
        # Without a sample count the pipeline starts no automatic job (``TrainingPipeline.holds_job_data``).
        sample_count = self.job_samples if automatic_jobs else None
        if not (self.trains and self._pipeline.ready(sample_count)):
            # The data of a job that waited for its check is gone, such as after its request was cancelled.
            self.job_check_waits = False
            return None
        data = self._pipeline.build_job_data(sample_count)
        self.pending_commit = JobDataReservation(data)
        return data

    def prepare_commit(
        self, result: TrainingCommit | None, correlation_metrics: Mapping[str, Any] | None = None
    ) -> PreparedCommit:
        """Prepare the pending result without exposing its state as committed.

        ``correlation_metrics`` are the experiment tracker's fields, which join the result's metrics in the commit.

        Acknowledging a pipeline mutates its in-memory job data bookkeeping, so
        the prepared value is cached and reused after a publication retry. The
        trainer's algorithm state and pending reservation remain unchanged
        until :meth:`commit` is called after the scenario's artifact and commit
        record settle.
        """
        with self._lock:
            if self.pending_commit is None:
                if result is not None and result is self.recorded_result:
                    raise RuntimeError("training result is on record already: committing it again is a second commit")
                return PreparedCommit(
                    algorithm_state=self.algorithm_state_dict(),
                    high_water_sequence=self._data_sequence,
                    high_water_offset=self._data_offset,
                )
            pending = self.pending_commit
            if isinstance(pending, JobDataReservation):
                if result is not None:
                    raise RuntimeError("training result does not match the pending commit")
                raise RuntimeError("trainer pending job data has no result")
            if pending.result is not result:
                raise RuntimeError("training result does not match the pending commit")
            if pending.prepared_commit is not None:
                return pending.prepared_commit
            data_id, result = pending.data_id, pending.result
            if not isinstance(result.state, Mapping):
                raise TypeError("training step state must be a mapping")
            consumed = pending.consumed_ids
            if consumed is None:
                # A commit about a running job consumes no job data.
                consumed = frozenset() if data_id is None else self._pipeline.acknowledge(data_id)
            released = frozenset(self._pipeline.consumed_requests)
            # The records that the stages ended since the last commit, in whichever process the stages run.
            ended_records = self._pipeline.store.ended_record_ids() - self.committed_record_ids
            loss = self._records.loss(self.scenario)
            metrics = {**dict(result.metrics), **dict(correlation_metrics or {})}
            if loss.record_count > 0:
                metrics.update({"records/evicted_count": loss.record_count, "records/data_incomplete": 1})
            request = None if pending.data is None else pending.data.request
            if request is not None:
                # The step's own dict, when it wrote one, carries what its method added to ``requires``.
                metrics.setdefault("training_request", {"id": request.id, **request.to_dict()})
            prepared = PreparedCommit(
                algorithm_state=dict(result.state),
                high_water_sequence=self._data_sequence,
                high_water_offset=self._data_offset,
                consumed_ids=frozenset(consumed | released | ended_records),
                released_ids=released,
                metrics=metrics or None,
                training_job_id=result.training_job_id,
                train_from=result.train_from,
            )
            self.pending_commit = replace(pending, prepared_commit=prepared)
            return prepared

    def settle_commit(self, prepared: PreparedCommit) -> None:
        """Expose one prepared state after its scenario commit has settled."""
        with self._lock:
            if self.pending_commit is None:
                return
            if (
                not isinstance(self.pending_commit, PendingTrainingResult)
                or self.pending_commit.prepared_commit is not prepared
            ):
                raise RuntimeError("prepared commit does not match the pending training commit")
            self._pipeline.release_requests(prepared.released_ids)
            self._state = dict(prepared.algorithm_state)
            self.consumed_record_ids.update(prepared.consumed_ids)
            self.committed_record_ids.update(prepared.consumed_ids)
            self.recorded_result = self.pending_commit.result
            self.pending_commit = None

    def cancel_job(self) -> None:
        """Cancel the running training jobs at their component; nothing for a trainer without one."""
        if self._job_loop is None:
            return
        with self._lock:
            state = JobLoopState.from_dict(self._state)
        self._job_loop.cancel(state)

    @property
    def start_files(self) -> tuple[str, ...] | None:
        """The files of a start that ``import_start`` reads (``TrainingComponent.start_files``); None for all of
        them, and for a trainer that trains nothing."""
        return None if self._job_loop is None else self._job_loop.component.start_files

    def import_start(self, artifact: Artifact) -> str:
        """The start of a job that trains from ``artifact``, the trainer's component of a snapshot, local with at
        least the files of ``start_files`` (``TrainingComponent.snapshot_start``): control's component checks the artifact and names its snapshot, and
        the training service imports it when it takes the job. Raise ``TrainingStartRefusedError`` when this
        trainer cannot start from ``artifact``."""
        if artifact.local_path is None:
            raise TrainingStartRefusedError(f"artifact {artifact.ref.revision!r} has no local files to train from")
        if self._job_loop is None:
            raise TrainingStartRefusedError(f"the trainer of scenario {self.scenario!r} trains nothing to start")
        try:
            return self._job_loop.component.snapshot_start(artifact)
        except ValueError as exc:
            raise TrainingStartRefusedError(
                f"scenario {self.scenario!r} cannot train from this artifact: {exc}"
            ) from exc

    def start_from(self, start: str) -> None:
        """Set the training start to ``start``, what ``import_start`` gave for an artifact, and cancel the running
        job that continues the start before (``JobLoop.started_from``). Raise ``CommitPendingError`` while a commit is
        in preparation."""
        with self._lock:
            if self.pending_commit is not None:
                raise CommitPendingError("cannot set the training start while a training result waits for its commit")
            loop_state = JobLoopState.from_dict(self._state)
            self._state = self.job_loop_required().started_from(loop_state, start).to_dict()

    def close(self) -> None:
        """Release the pipeline, which the trainer owns.

        Held under the trainer lock so a pipeline is never closed while a
        record is being pushed or job data is built on the training thread.
        """
        with self._lock:
            self._pipeline.close()

    def restore_consumed_records(self, consumed_ids: frozenset[str]) -> None:
        """Set the records that committed job data or stored samples consumed, which a recovered trainer skips.

        A recovered trainer reads every retained record again, so the records of a step that was not complete reach
        its pipeline again. Retention can keep a consumed record stored, and a consumed record never trains twice.
        """
        with self._lock:
            self.consumed_record_ids.update(consumed_ids)

    def restore_committed_records(self, record_ids: frozenset[str]) -> None:
        """Set the record IDs that the commits of this trainer named, so that its next commit does not name them again."""
        with self._lock:
            self.committed_record_ids = set(record_ids)
