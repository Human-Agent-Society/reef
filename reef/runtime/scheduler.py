"""Scenario scheduling, serialized training execution, and resource handoffs.

Backbone objects, in the order a training step meets them:

- :class:`RuntimeScheduler` — the recipe-facing pair of training and inference
  runtimes; keeps unpublished weights unavailable to requests.
- :class:`TrainingCoordinator` — the worker-side object that serializes backend
  jobs with their publication behind Reef's commit barrier.
- :class:`TrainingExecution` — runs one durable job through its checkpoint,
  replaying or refusing based on the recorded marker.
- :class:`InferenceMemory` — pairs a colocated engine's memory releases with
  their resumes.

The record-to-record stage checks staleness before a batch forms
(``reef.record2record.staleness``). The jobs here carry each sample's
producing runtime load ID only as training metadata.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import replace
from threading import Lock, RLock
from typing import Any

from reef.core.batches import StepScheduling, TrainingBatch
from reef.observability.operations import OperationMetrics
from reef.runtime.interfaces import (
    ActivatedModel,
    CandidateTrainingDeferred,
    InferenceBackend,
    InferenceMemoryOperations,
    InferenceRuntime,
    ModelCandidate,
    PreparedTrainingJob,
    PreparedTrainingStep,
    RuntimeContractError,
    StaleCandidate,
    TrainingBackend,
    TrainingJobResult,
    TrainingJobState,
    TrainingJobStore,
    TrainingMethod,
    TrainingRuntime,
    TrainingRuntimeError,
)
from reef.runtime.publication import PUBLISHED_STATES, BackendWeightPublisher, TrainingPublication
from reef.runtime.recovery import (
    COMMITTED_MARKER_KEY,
    FileTrainingJobStore,
    TrainingRecovery,
    marker_checkpoint_result,
    marker_committed,
    marker_disposition,
    marker_in_flight,
    marker_path,
    marker_result,
)

#: Job states that hold an activated candidate Reef still has to commit or finish.
COMMIT_PENDING_STATES = frozenset({"UPDATING_WEIGHTS", *PUBLISHED_STATES})


# -- Colocated memory ---------------------------------------------------------


class InferenceMemory:
    """Pair releases and resumes, including cold startup and partial failures.

    Repeating an acknowledged operation is safe. A failed operation can have
    changed only part of an engine, so its state remains uncertain until the
    owning deployment replaces that engine.
    """

    def __init__(self, operations: InferenceMemoryOperations, regions: Sequence[str]) -> None:
        self._operations = operations
        self._regions = tuple(regions)
        self._released: set[str] = set()
        self._uncertain = False
        self._lock = RLock()

    def _selection(self, regions: Sequence[str] | None) -> tuple[str, ...]:
        if self._uncertain:
            raise RuntimeError("inference memory state is uncertain; replace the engine before reuse")
        selected = self._regions if regions is None else tuple(dict.fromkeys(regions))
        if unknown := set(selected).difference(self._regions):
            raise ValueError(f"unknown inference memory regions: {sorted(unknown)}")
        return selected

    def release(self, regions: Sequence[str] | None = None) -> None:
        with self._lock:
            selected = tuple(region for region in self._selection(regions) if region not in self._released)
            if not selected:
                return
            try:
                self._operations.release(selected)
            except BaseException:
                self._uncertain = True
                raise
            self._released.update(selected)

    def resume(self, regions: Sequence[str] | None = None) -> None:
        with self._lock:
            selected = tuple(region for region in self._selection(regions) if region in self._released)
            if not selected:
                return
            try:
                self._operations.resume(selected)
            except BaseException:
                self._uncertain = True
                raise
            self._released.difference_update(selected)


# -- Durable job execution ----------------------------------------------------


#: The payload key naming the scenario that owns a job. The marker records it; the job identity leaves it out, since
#: a job is its batch whichever scenario asked for it.
JOB_OWNER_KEY = "owner"


def training_job_id(payload: Mapping[str, Any]) -> str:
    """Preserve the retry-stable identity of the shared training payload: its batch."""
    identity = dict(payload)
    identity.pop(JOB_OWNER_KEY, None)
    # The other components of a composite advance the scenario step while a job is out; its retry must replay.
    identity.pop("scenario_step", None)
    # The processor numbers batches per process; a reload numbers the same rows again.
    identity.pop("batch_id", None)
    # A retry after a newer publication trains the same batch. It must not repeat an optimizer step.
    identity.pop("parent_runtime_load_id", None)
    encoded = json.dumps(identity, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def job_scenario_step(payload: Mapping[str, Any]) -> int:
    """The scenario step a training payload trains, validated as a non-negative integer."""
    scenario_step = payload.get("scenario_step")
    if not isinstance(scenario_step, int) or isinstance(scenario_step, bool) or scenario_step < 0:
        raise ValueError("training job scenario_step must be non-negative")
    return scenario_step


class TrainingExecution:
    """Own retry classification, RUNNING/CHECKPOINT transitions and train ordering.

    The caller serializes execution with publication and shutdown. Replay never
    prepares a backend job. RUNNING is intentionally ambiguous after a failure:
    Reef cannot infer whether an optimizer stepped, so automatic retry refuses.

    Inside a coordinator, ``publisher`` performs the colocated device
    handoff before a job trains; standalone execution needs none.
    """

    def __init__(
        self,
        store: TrainingJobStore | None,
        backend: TrainingBackend,
        state: TrainingJobState,
        *,
        publisher: BackendWeightPublisher | None = None,
    ) -> None:
        self._store = store
        self._backend = backend
        self._state = state
        self._publisher = publisher

    def recover(self) -> dict[str, Any] | None:
        """Read restart state without guessing whether an optimizer step completed."""
        marker = self._store.read() if self._store is not None else None
        if marker is not None and marker["status"] == "RUNNING":
            raise RuntimeError(f"ambiguous training job {marker['job_id']}")
        return marker

    def execute(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        job_id = training_job_id(payload)
        scenario_step = job_scenario_step(payload)
        if self._store is None:
            raise RuntimeError("training job checkpoint path is not configured")
        marker = self._store.read()
        disposition = marker_disposition(marker, job_id)
        if marker is not None and disposition == "conflict":
            raise RuntimeError(
                f"training marker is {marker['status']} for job {marker['job_id']}, not this batch's job "
                f"{job_id}; operator recovery required (see Training-step coordination in the executors guide)"
            )
        if marker is not None and disposition != "fresh":
            return marker_result(marker) if marker["status"] == "COMPLETE" else marker_checkpoint_result(marker)
        with self._backend.prepare(
            payload, job_id=job_id, scenario_step=scenario_step, prior_marker=marker
        ) as prepared:
            if isinstance(prepared, TrainingJobResult):
                if prepared.outcome not in {"stale", "storage_blocked"}:
                    raise RuntimeError("training preparation may only return stale or storage_blocked")
                return prepared
            return self._run(prepared, self._store, job_id, payload, marker)
        raise RuntimeError("training preparation suppressed an execution failure")

    def _run(
        self,
        prepared: PreparedTrainingJob,
        store: TrainingJobStore,
        job_id: str,
        payload: Mapping[str, Any],
        prior_marker: Mapping[str, Any] | None,
    ) -> TrainingJobResult:
        """Train and checkpoint one job, recording RUNNING then CHECKPOINT."""
        checkpoint = prepared.checkpoint
        # Reef reasons in scenario steps; the marker's rollout_id is the backend's own checkpoint index.
        running: dict[str, Any] = {
            "status": "RUNNING",
            "job_id": job_id,
            "rollout_id": checkpoint.rollout_id,
            "scenario_step": checkpoint.scenario_step,
        }
        if prior_marker is not None and marker_committed(prior_marker):
            # RUNNING overwrites the settled marker, so keep a copy: the
            # Slime preflight restores it when a kill leaves nothing saved.
            running[COMMITTED_MARKER_KEY] = {
                key: value for key, value in prior_marker.items() if key != COMMITTED_MARKER_KEY
            }
        parent_runtime_load_id = payload.get("parent_runtime_load_id")
        if isinstance(parent_runtime_load_id, str) and parent_runtime_load_id:
            running["parent_runtime_load_id"] = parent_runtime_load_id
        # The owner: the adapter's scenario on a runtime training several, else the scenario the payload names.
        owner = checkpoint.scenario if checkpoint.scenario is not None else payload.get(JOB_OWNER_KEY)
        if isinstance(owner, str) and owner:
            running["scenario"] = owner
        store.write(running)
        self._state.phase = "training"
        try:
            if self._publisher is not None:
                self._publisher.prepare_training()
            metrics = prepared.train()
            self._state.phase = "checkpointing"
            prepared.save_checkpoint()
            if checkpoint.path.is_symlink() or not checkpoint.path.is_dir():
                raise RuntimeError(f"checkpoint is missing or unsafe: {checkpoint.path}")
            # Replay must see all telemetry with the checkpoint, even if
            # the process dies immediately after this transition.
            updates: dict[str, Any] = {"checkpoint_path": str(checkpoint.path)}
            if metrics.durable:
                updates["metrics"] = dict(metrics.durable)
            if metrics.training:
                updates["train_metrics"] = dict(metrics.training)
            store.transition(running, "CHECKPOINT", **updates)
        except BaseException:
            self._state.phase = "training_failed" if self._state.phase == "training" else "checkpoint_failed"
            # Later retries replace the RPC error; retain the original
            # worker/checkpoint failure in the coordinator's process log.
            traceback.print_exc(file=sys.stderr)
            raise
        return marker_checkpoint_result(running)


# -- Recipe-facing scheduling -------------------------------------------------


class RuntimeScheduler:
    """Schedule backend work while keeping unpublished weights unavailable.

    Each scenario invokes recovery with its own durable commit identity. When
    backends share one inference service, admission remains engine-wide while
    acknowledgement remains scoped to the scenario that owns the pending job.
    A colocated backend stops admitting inference for the whole training step;
    a disaggregated one stops only for the serving-weight update.
    """

    def __init__(self, training_runtime: TrainingRuntime, inference_runtime: InferenceRuntime) -> None:
        self.operations = OperationMetrics(("weight_sync",), counters=("stale_batches_total",))
        self.training_runtime = training_runtime
        self.inference_runtime = inference_runtime
        status = training_runtime.training_job_status()
        self._colocated = bool(status and status.get("colocate"))
        if status is None:
            inference_runtime.mark_published()
        else:
            self._sync_inference_admission(status)

    # -- Training

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        method: TrainingMethod,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
        scenario_step: int,
    ) -> PreparedTrainingStep:
        return self.training_runtime.prepare_training_step(
            batch,
            method,
            algorithm_state,
            scheduling,
            scenario_step,
            parent_runtime_load_id=self.inference_runtime.current_runtime_load_id(),
        )

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        """Train a checkpoint, preserving the currently published source version."""
        current = self.inference_runtime.current_runtime_load_id()
        try:
            with self._colocated_pause():
                candidate = self.training_runtime.train_candidate(payload)
        except StaleCandidate:
            self.operations.increment("stale_batches_total")
            raise
        if not isinstance(candidate, ModelCandidate):
            raise RuntimeContractError("training runtime must return ModelCandidate")
        return replace(candidate, current_runtime_load_id=current)

    @property
    def colocated(self) -> bool:
        """Whether the backend hands the engine's devices to training for the whole job."""
        return self._colocated

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        """Stage selected weights behind closed inference admission."""
        self.inference_runtime.pause_admission()
        with self.operations.measure("weight_sync"):
            return self.inference_runtime.activate_candidate(candidate)

    def reject_candidate(self, candidate: ModelCandidate) -> None:
        """Discard a candidate before reopening the unchanged serving version."""
        self.training_runtime.reject_candidate(candidate)
        self.inference_runtime.resume_admission()

    @contextmanager
    def _colocated_pause(self) -> Iterator[None]:
        """Hold admission closed while a colocated backend hands its devices to training.

        New requests wait while the workers move shared devices from inference
        to training; backend operations preserve already admitted requests
        across their own memory release and restore. A refused or stale batch
        reopens admission at once. Any other failure reopens it only when the
        durable status proves that no training or checkpoint work started.
        """
        if self._colocated:
            self.inference_runtime.pause_admission()
        try:
            yield
        except (CandidateTrainingDeferred, StaleCandidate):
            self._resume_if_colocated()
            raise
        except BaseException:
            if self._colocated and self._backend_idle():
                self.inference_runtime.resume_admission()
            raise

    def _resume_if_colocated(self) -> None:
        if self._colocated:
            self.inference_runtime.resume_admission()

    def _backend_idle(self) -> bool:
        with suppress(Exception):
            return (self.training_runtime.training_job_status() or {}).get("status") == "IDLE"
        return False

    # -- Recovery and commit

    def recover_pending_step(
        self,
        scenario_step: int,
        *,
        scenario: str | None = None,
        committed_training_job_id: str | None = None,
        committed_training_without_job_id: bool = False,
    ) -> None:
        """Reconcile a backend journal only against the owning scenario commit."""
        if committed_training_job_id is not None and (
            not isinstance(committed_training_job_id, str) or not committed_training_job_id
        ):
            raise TrainingRuntimeError("committed_training_job_id must be a non-empty string or None")
        if not isinstance(committed_training_without_job_id, bool):
            raise TrainingRuntimeError("committed_training_without_job_id must be a boolean")
        scenario = scenario if self.training_runtime.concurrent_training_scenarios else None
        training_job = self.training_runtime.training_job_status()
        if training_job is None:
            if committed_training_job_id is not None:
                self.finish_committed_training_job(committed_training_job_id)
            return
        status = training_job["status"]
        if self._sync_inference_admission(training_job):
            return
        job_scenario = training_job.get("scenario")
        if scenario is not None and isinstance(job_scenario, str) and job_scenario != scenario:
            # The pending job belongs to another scenario sharing this
            # runtime; admission is engine-global and already synced above,
            # but its commit handshake is that scenario's to finish.
            return
        if status == "REJECTING":
            training_job_id = training_job.get("training_job_id")
            if not isinstance(training_job_id, str) or not training_job_id:
                raise TrainingRuntimeError("rejecting training job is missing its durable identity")
            self.training_runtime.reject_training_job(training_job_id)
            self.inference_runtime.resume_admission()
            return
        if status not in COMMIT_PENDING_STATES:
            return
        if "scenario_step" not in training_job and not marker_in_flight(training_job):
            # A settled job an earlier release recorded without its scenario step has nothing left to finish.
            return
        job_scenario_step, training_job_id = pending_job_identity(training_job)
        if status == "UPDATING_WEIGHTS":
            with self.operations.measure("weight_sync"):
                self.inference_runtime.resume_weight_update(training_job_id)
        # Other components can advance the scenario; only this job's commit can finish it.
        if committed_training_job_id is not None:
            committed = scenario_step > job_scenario_step and committed_training_job_id == training_job_id
        else:
            # Older bridges lack job IDs and require an unacknowledged COMPLETE marker at the exact next step.
            committed = (
                committed_training_without_job_id
                and status == "COMPLETE"
                and training_job.get("commit_acknowledged") is not True
                and scenario_step == job_scenario_step + 1
            )
        if committed:
            self.finish_committed_training_job(training_job_id)

    def acknowledge_commit(self, scenario_step: int, training_job_id: str, *, scenario: str | None = None) -> None:
        """Release a selected update only after its matching durable commit."""
        self.recover_pending_step(scenario_step, scenario=scenario, committed_training_job_id=training_job_id)

    def finish_committed_training_job(self, training_job_id: str) -> None:
        """Acknowledge the matched job before publishing its identity and reopening admission.

        If either acknowledgement fails, admission stays closed; retry completes the same job.
        """
        self.inference_runtime.acknowledge_publication(training_job_id)
        self.training_runtime.commit_candidate(training_job_id)
        self.inference_runtime.mark_published()
        self.inference_runtime.resume_admission()

    def _sync_inference_admission(self, training_job: Mapping[str, Any]) -> bool:
        """Apply states that need no weight-update recovery; return if settled."""
        status = training_job["status"]
        if status in {"IDLE", "REJECTED"} or (
            status == "COMPLETE" and training_job.get("commit_acknowledged") is True
        ):
            self.inference_runtime.mark_published()
            self.inference_runtime.resume_admission()
            return True
        if status in {"RUNNING", "CHECKPOINT"}:
            if self._colocated:
                self.inference_runtime.pause_admission()
            else:
                if self.inference_runtime.current_runtime_load_id() is None:
                    self.inference_runtime.mark_published()
                self.inference_runtime.resume_admission()
            return True
        self.inference_runtime.pause_admission()
        return False


def pending_job_identity(training_job: Mapping[str, Any]) -> tuple[int, str]:
    job_scenario_step = training_job.get("scenario_step")
    training_job_id = training_job.get("training_job_id")
    if (
        not isinstance(job_scenario_step, int)
        or isinstance(job_scenario_step, bool)
        or not isinstance(training_job_id, str)
        or not training_job_id
    ):
        raise TrainingRuntimeError("training-job status is missing its durable identity")
    return job_scenario_step, training_job_id


# -- Worker-side coordination -------------------------------------------------

#: Publication phases in which the coordinator reports itself unhealthy.
_FAILED_PHASES = frozenset({"training_failed", "checkpoint_failed", "weight_sync_failed", "stopped"})


class TrainingCoordinator:
    """Serialize backend jobs and their publication behind Reef's commit barrier."""

    def __init__(self, training: TrainingBackend, inference: InferenceBackend, *, owns_training: bool = True) -> None:
        self._training = training
        self._owns_training = owns_training
        self._context = training.context
        self._config = training.config
        path = marker_path(self._config.save_hf_template) if self._config.save_hf_template is not None else None
        self._store = FileTrainingJobStore(path) if path is not None else None
        state = TrainingJobState()
        self._weight_publisher = BackendWeightPublisher(training, inference, state, self._store)
        self._publication = TrainingPublication(self._store, self._weight_publisher, state)
        self._execution = TrainingExecution(self._store, training, state, publisher=self._weight_publisher)
        self._closed = False
        self._completed_train_steps = 0
        self._last_train_rollout_id: int | None = None
        self._last_train_metrics: dict[str, Any] = {}
        self._operation_lock = Lock()
        self._training.start()
        recovered = self._execution.recover()
        self._inference_url = TrainingRecovery(self._publication, self._weight_publisher).restore(recovered)

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        method: TrainingMethod,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
    ) -> PreparedTrainingStep:
        return self._training.prepare_training_step(batch, method, algorithm_state, scheduling)

    def shutdown(self) -> None:
        """Close training workers; deployment ownership closes inference separately."""
        with self._operation_lock:
            if self._closed:
                return
            self._closed = True
            self._publication.phase = "stopped"
            if self._owns_training:
                self._training.close()

    def health(self) -> dict[str, Any]:
        """Return a lightweight liveness marker for container health checks."""
        self._training.check_health()
        training_job = self._training_job_health()
        ok = self._publication.phase not in _FAILED_PHASES
        return {
            "ok": ok,
            # A publication failure with a durable UPDATING_WEIGHTS marker is
            # replayable in place: ``update_serving_weights`` recovers the
            # engines and republishes from the checkpoint. The coordinator decides
            # which failures are retryable so callers never re-derive it from
            # phase and marker.
            "recoverable": not ok
            and self._publication.phase == "weight_sync_failed"
            and training_job.get("status") == "UPDATING_WEIGHTS",
            "start_rollout_id": self._context.next_rollout_id,
            "phase": self._publication.phase,
            "colocate": self._config.colocate,
            # Where the serving engines answer; Reef dials this when the
            # deployment leaves ``reef.inference_url`` unset.
            "inference_url": self._inference_url,
            "lora_adapter": None,
            "lora_mode": "scenario" if self._config.lora else None,
            "lora_adapters": {} if self._context.history is None else self._context.history.status(),
            "adapter_residency": (
                None if self._weight_publisher.residency is None else self._weight_publisher.residency.status()
            ),
            "completed_train_steps": self._completed_train_steps,
            "last_train_rollout_id": self._last_train_rollout_id,
            "last_train_metrics": dict(self._last_train_metrics),
            "training_job": training_job,
        }

    def _training_job_health(self) -> dict[str, Any]:
        """The durable job block of :meth:`health`, read from the marker when one exists."""
        training_job: dict[str, Any] = {
            "deferred_weight_update": self._config.save_hf_template is not None,
            "status": "COMPLETE" if self._config.save_hf_template is None else "IDLE",
        }
        marker = self._store.read() if self._store is not None else None
        if marker is None:
            return training_job
        training_job.update(
            status=marker["status"],
            training_job_id=marker["job_id"],
            runtime_load_id=marker.get("runtime_load_id"),
            commit_acknowledged=marker.get("commit_acknowledged", False),
        )
        if "scenario" in marker:
            training_job["scenario"] = marker["scenario"]
        if "scenario_step" in marker:
            # Reef reasons in scenario steps; the marker's rollout_id is the backend's own checkpoint index. Every
            # job still out names its step; only a settled marker an earlier release wrote has none.
            training_job["scenario_step"] = marker["scenario_step"]
        return training_job

    def start_rollout_id(self) -> int:
        return self._context.next_rollout_id

    def serving_runtime_load_id(self) -> str:
        """Return the last successfully published serving-runtime load ID.

        Failed swaps can consume a backend counter before raising, so this
        caches only completed publications. Reef recovery uses the value to
        reconcile the serving engine with its recovered head.
        """
        return self._context.runtime_load_id

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        """Delegate job replay, train ordering and checkpoint recording to Reef."""
        if self._config.save_hf_template is None:
            raise RuntimeError("training checkpoint path is not configured")
        with self._operation_lock:
            return self._execution.execute(payload)

    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        """Delegate durable publication ordering to Reef's shared coordinator."""
        with self._operation_lock:
            publication = self._publication.publish(training_job_id)
            marker = publication.marker
            if publication.published:
                self._record_completed_step(marker)
            return marker_result(marker)

    def _record_completed_step(self, marker: Mapping[str, Any]) -> None:
        rollout_id = int(marker["rollout_id"])
        self._context.next_rollout_id = max(self._context.next_rollout_id, rollout_id + 1)
        self._completed_train_steps += 1
        self._last_train_rollout_id = rollout_id
        train_metrics = marker.get("train_metrics")
        self._last_train_metrics = dict(train_metrics) if isinstance(train_metrics, Mapping) else {}

    def reject_training_candidate(self, training_job_id: str) -> None:
        """Finish a checkpointed job without changing the serving weights."""
        with self._operation_lock:
            marker = self._publication.reject(training_job_id)
            self._context.next_rollout_id = max(self._context.next_rollout_id, int(marker["rollout_id"]) + 1)

    def acknowledge_training_commit(self, training_job_id: str) -> None:
        """Resume requests only through Reef's durable commit gate."""
        with self._operation_lock:
            self._publication.acknowledge(training_job_id)
