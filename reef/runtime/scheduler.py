"""Scenario scheduling, serialized training execution, admission, and resource handoffs.

Backbone objects, in the order a training step meets them:

- :class:`RuntimeScheduler` — the recipe-facing pair of training and inference
  runtimes; keeps unpublished weights unavailable to requests.
- :class:`TrainingCoordinator` — the worker-side object that serializes backend
  jobs with their publication behind Reef's commit barrier.
- :class:`TrainingExecution` — runs one durable job through its checkpoint,
  replaying or refusing based on the recorded marker.
- :class:`InferenceMemory` — pairs a colocated engine's memory releases with
  their resumes.

The module-level functions implement staleness admission: whether a batch's
producing versions are close enough to the serving version to train on, and
what to report when they are not.
"""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from threading import Lock, RLock
from typing import Any, Literal

from reef.core.artifact_ref import parse_runtime_load_spans
from reef.core.batches import StepScheduling, TrainingBatch
from reef.core.evaluation import SelectionDecision
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
    RuntimeLoadId,
    ScenarioHistoryStore,
    StaleCandidate,
    TrainingBackend,
    TrainingContext,
    TrainingJobResult,
    TrainingJobState,
    TrainingJobStore,
    TrainingRuntime,
    TrainingRuntimeError,
)
from reef.runtime.publication import PUBLISHED_STATES, BackendWeightPublisher, TrainingPublication
from reef.runtime.recovery import (
    FileTrainingJobStore,
    TrainingRecovery,
    marker_checkpoint_result,
    marker_disposition,
    marker_path,
    marker_result,
)

#: Job states that hold an activated candidate Reef still has to commit or finish.
COMMIT_PENDING_STATES = frozenset({"UPDATING_WEIGHTS", *PUBLISHED_STATES})

# -- Staleness admission ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _StalenessDecision:
    action: Literal["admit", "drop"]
    metrics: Mapping[str, Any]


def max_staleness(payload: Mapping[str, Any]) -> int:
    value = payload.get("max_staleness", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("training job max_staleness must be a non-negative integer")
    return value


def uses_staleness_admission(payload: Mapping[str, Any]) -> bool:
    """Whether the serving version is an admission fence, not job identity."""
    return max_staleness(payload) > 0 or "producing_runtime_load_ids" in payload


def _source_agent_record_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    samples = payload.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, str | bytes):
        return ()
    return tuple(
        str(row[0]) for row in samples if isinstance(row, Sequence) and not isinstance(row, str | bytes) and row
    )


def _producing_runtime_load_ids(payload: Mapping[str, Any]) -> Sequence[Any]:
    versions = payload.get("producing_runtime_load_ids")
    if not isinstance(versions, Sequence) or isinstance(versions, str | bytes) or not versions:
        raise ValueError("bounded staleness admission requires producing_runtime_load_ids")
    source_ids = _source_agent_record_ids(payload)
    if len(versions) != len(source_ids):
        raise ValueError(
            "bounded staleness admission requires one producing runtime load ID "
            f"per sample: {len(versions)} versions for {len(source_ids)} samples"
        )
    return versions


def _admission_runtime_load_id_groups(payload: Mapping[str, Any]) -> list[list[Any]]:
    """Return each sample's exact span versions for bounded admission."""
    versions = _producing_runtime_load_ids(payload)
    raw_groups = payload.get("producing_runtime_load_spans")
    if raw_groups is None:
        return [[version] for version in versions]
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, str | bytes) or len(raw_groups) != len(versions):
        raise ValueError("producing_runtime_load_spans must contain one span list per sample")
    samples = payload["samples"]
    groups: list[list[Any]] = []
    for sample_index, (raw_spans, scalar) in enumerate(zip(raw_groups, versions, strict=True)):
        if not raw_spans:
            groups.append([scalar])
            continue
        row = samples[sample_index]
        response_length = len(row[2]) if isinstance(row, Sequence) and len(row) > 2 else None
        spans = parse_runtime_load_spans(
            raw_spans,
            field_name=f"producing_runtime_load_spans[{sample_index}]",
            response_length=response_length,
        )
        group = [span.runtime_load_id for span in spans]
        if scalar is not None and set(group) != {scalar}:
            raise ValueError(f"producing runtime load ID for sample {sample_index} disagrees with its token spans")
        groups.append(group)
    return groups


def _stale_drop_decision(
    payload: Mapping[str, Any],
    *,
    serving_runtime_load_id: str,
    producing_runtime_load_ids: Sequence[Any],
    reason: str,
    policy_lags: Sequence[int] = (),
) -> _StalenessDecision:
    source_ids = _source_agent_record_ids(payload)
    metrics: dict[str, Any] = {
        "staleness/samples_dropped": len(source_ids) or len(producing_runtime_load_ids),
        "staleness/drop_reason": reason,
        "staleness/source_agent_record_ids": list(source_ids),
        "staleness/producing_runtime_load_ids": [
            None if version is None else str(version) for version in producing_runtime_load_ids
        ],
        "staleness/serving_runtime_load_id": serving_runtime_load_id,
    }
    if policy_lags:
        metrics["staleness/drop_policy_lags"] = list(policy_lags)
    return _StalenessDecision(action="drop", metrics=metrics)


def _lag_decision(
    payload: Mapping[str, Any],
    *,
    serving_runtime_load_id: str,
    groups: Sequence[Sequence[Any]],
    lags: Sequence[int],
    sample_lags: Sequence[int],
    drop_reason: str | None,
    extra_metrics: Mapping[str, Any] | None = None,
) -> _StalenessDecision:
    """Turn a lag walk's outcome into a decision; a drop reports the lags seen so far."""
    if drop_reason is not None:
        return _stale_drop_decision(
            payload,
            serving_runtime_load_id=serving_runtime_load_id,
            producing_runtime_load_ids=[version for group in groups for version in group],
            reason=drop_reason,
            policy_lags=lags,
        )
    return _StalenessDecision(
        action="admit",
        metrics={
            "staleness/samples_fresh": sum(lag == 0 for lag in sample_lags),
            "staleness/samples_admitted_stale": sum(lag > 0 for lag in sample_lags),
            **(extra_metrics or {}),
        },
    )


def _producing_problem(value: Any) -> str | None:
    """Why ``value`` cannot name a producing version, or ``None`` when it parses."""
    if not isinstance(value, str) or not value:
        return "missing_producing_runtime_load_id"
    try:
        RuntimeLoadId.parse(value)
    except (TypeError, ValueError):
        return "malformed_producing_runtime_load_id"
    return None


def _canonical_serving(serving_runtime_load_id: str) -> RuntimeLoadId:
    try:
        serving = RuntimeLoadId.parse(serving_runtime_load_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot classify staleness from serving runtime load ID {serving_runtime_load_id!r}"
        ) from exc
    if str(serving) != serving_runtime_load_id:
        raise RuntimeError(f"cannot classify staleness from non-canonical serving version {serving_runtime_load_id!r}")
    return serving


def _staleness_admission(
    payload: Mapping[str, Any],
    *,
    serving_runtime_load_id: str,
    max_staleness: int,
) -> _StalenessDecision:
    """Bounded admission against the engine-global version sequence.

    A sample's lag is the sequence distance between its producing version and
    the serving version; both must share one incarnation, and a multi-span
    sample's versions must ascend.
    """
    serving = _canonical_serving(serving_runtime_load_id)
    groups = _admission_runtime_load_id_groups(payload)
    lags: list[int] = []
    sample_lags: list[int] = []
    drop_reason: str | None = None
    for group in groups:
        group_lags: list[int] = []
        previous_sequence: int | None = None
        for value in group:
            drop_reason = _producing_problem(value)
            if drop_reason is not None:
                break
            producing = RuntimeLoadId.parse(value)
            if str(producing) != value:
                drop_reason = "malformed_producing_runtime_load_id"
            elif producing.incarnation != serving.incarnation:
                drop_reason = "cross_incarnation"
            elif previous_sequence is not None and producing.sequence <= previous_sequence:
                drop_reason = "non_monotonic_producing_runtime_load_ids"
            if drop_reason is not None:
                break
            previous_sequence = producing.sequence
            lag = serving.sequence - producing.sequence
            lags.append(lag)
            group_lags.append(lag)
            if lag < 0:
                drop_reason = "future_producing_runtime_load_id"
            elif lag > max_staleness:
                drop_reason = "policy_lag_exceeded"
            if drop_reason is not None:
                break
        if drop_reason is not None:
            break
        sample_lags.append(max(group_lags))
    return _lag_decision(
        payload,
        serving_runtime_load_id=serving_runtime_load_id,
        groups=groups,
        lags=lags,
        sample_lags=sample_lags,
        drop_reason=drop_reason,
    )


def _scenario_staleness_admission(
    payload: Mapping[str, Any],
    *,
    scenario: str,
    history: ScenarioHistoryStore,
    serving_runtime_load_id: str,
    max_staleness: int,
) -> _StalenessDecision:
    """Bounded admission against one scenario's own publication history.

    The engine's runtime load ID advances on every scenario's publication, so
    the global sequence gap overstates this scenario's staleness. A sample's
    lag is the number of *this* scenario's publications that postdate the
    version its tokens were produced under.
    """
    if uses_staleness_admission(payload):
        groups = _admission_runtime_load_id_groups(payload)
    else:
        groups = [[payload.get("expected_runtime_load_id")]]
    lags: list[int] = []
    sample_lags: list[int] = []
    drop_reason: str | None = None
    for group in groups:
        group_lags: list[int] = []
        for value in group:
            drop_reason = _producing_problem(value)
            if drop_reason is not None:
                break
            lag = history.lag(scenario, RuntimeLoadId.parse(value))
            if lag is None:
                drop_reason = "cross_incarnation"
                break
            lags.append(lag)
            group_lags.append(lag)
            if lag > max_staleness:
                drop_reason = "policy_lag_exceeded"
                break
        if drop_reason is not None:
            break
        sample_lags.append(max(group_lags))
    return _lag_decision(
        payload,
        serving_runtime_load_id=serving_runtime_load_id,
        groups=groups,
        lags=lags,
        sample_lags=sample_lags,
        drop_reason=drop_reason,
        extra_metrics={"staleness/scenario": scenario},
    )


def _admission_decision(payload: Mapping[str, Any], context: TrainingContext) -> _StalenessDecision | None:
    """Decide whether the coordinator may train on ``payload`` at the context's serving version.

    ``None`` means exact-version admission passed and there is nothing to
    report; a drop with empty metrics means the exact version simply mismatched.
    """
    serving = context.runtime_load_id
    window = max_staleness(payload)
    if context.history is not None:
        scenario = payload.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("per-scenario LoRA training jobs must name their scenario")
        return _scenario_staleness_admission(
            payload,
            scenario=scenario,
            history=context.history,
            serving_runtime_load_id=serving,
            max_staleness=window,
        )
    if uses_staleness_admission(payload):
        if payload.get("expected_runtime_load_id") != serving:
            groups = _admission_runtime_load_id_groups(payload)
            return _stale_drop_decision(
                payload,
                serving_runtime_load_id=serving,
                producing_runtime_load_ids=[value for group in groups for value in group],
                reason="execution_fence_mismatch",
            )
        return _staleness_admission(payload, serving_runtime_load_id=serving, max_staleness=window)
    if payload.get("expected_runtime_load_id") != serving:
        return _StalenessDecision(action="drop", metrics={})
    return None


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


#: The payload key naming the scenario that owns a job. The marker records it; the job identity leaves it out, so
#: a job an earlier build started, whose payload named no owner, keeps its identity.
JOB_OWNER_KEY = "owner"


def training_job_id(payload: Mapping[str, Any]) -> str:
    """Preserve the retry-stable identity of the shared training payload: its batch and admission fence."""
    identity = dict(payload)
    identity.pop("max_staleness", None)
    identity.pop(JOB_OWNER_KEY, None)
    # The other components of a composite advance the scenario step while a job is out; its retry must replay.
    identity.pop("rollout_id", None)
    # The processor numbers batches per process; a reload numbers the same rows again.
    identity.pop("batch_id", None)
    if uses_staleness_admission(payload):
        # A newer admission fence on retry must not repeat an optimizer step.
        identity.pop("expected_runtime_load_id", None)
    encoded = json.dumps(identity, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def legacy_training_job_id(payload: Mapping[str, Any], rollout_id: int) -> str:
    """The identity an earlier build wrote into its job marker: the payload with the scenario step in it."""
    identity = {**payload, "rollout_id": rollout_id}
    identity.pop("max_staleness", None)
    identity.pop(JOB_OWNER_KEY, None)
    if uses_staleness_admission(payload):
        identity.pop("expected_runtime_load_id", None)
    encoded = json.dumps(identity, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rollout_id(payload: Mapping[str, Any]) -> int:
    rollout_id = payload.get("rollout_id")
    if not isinstance(rollout_id, int) or isinstance(rollout_id, bool) or rollout_id < 0:
        raise ValueError("training job rollout_id must be non-negative")
    return rollout_id


class TrainingExecution:
    """Own retry classification, RUNNING/CHECKPOINT transitions and train ordering.

    The caller serializes execution with publication and shutdown. Replay never
    prepares a backend job. RUNNING is intentionally ambiguous after a failure:
    Reef cannot infer whether an optimizer stepped, so automatic retry refuses.

    Inside a coordinator, ``context`` adds staleness admission before the
    backend prepares a job and ``publisher`` performs the colocated device
    handoff before it trains; standalone execution needs neither.
    """

    def __init__(
        self,
        store: TrainingJobStore | None,
        backend: TrainingBackend,
        state: TrainingJobState,
        *,
        context: TrainingContext | None = None,
        publisher: BackendWeightPublisher | None = None,
    ) -> None:
        self._store = store
        self._backend = backend
        self._state = state
        self._context = context
        self._publisher = publisher

    def recover(self) -> dict[str, Any] | None:
        """Read restart state without guessing whether an optimizer step completed."""
        marker = self._store.read() if self._store is not None else None
        if marker is not None and marker["status"] == "RUNNING":
            raise RuntimeError(f"ambiguous training job {marker['job_id']}")
        return marker

    def execute(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        job_id = training_job_id(payload)
        rollout_id = _rollout_id(payload)
        if self._store is None:
            raise RuntimeError("training job checkpoint path is not configured")
        marker = self._store.read()
        if marker is not None and marker["job_id"] != job_id:
            # A marker an earlier build wrote names the same batch by the step it ran at. Its job replays or
            # resumes under that name; a batch that trains again from the start carries the current identity.
            legacy = legacy_training_job_id(payload, marker.get("scenario_step", marker["rollout_id"]))
            if marker["job_id"] == legacy and marker_disposition(marker, legacy) != "fresh":
                job_id = legacy
        disposition = marker_disposition(marker, job_id)
        if disposition == "conflict":
            if marker is None:
                raise RuntimeError("conflicting training disposition has no marker")
            raise RuntimeError(f"training marker is {marker['status']}; operator recovery required")
        if disposition != "fresh":
            if marker is None:
                raise RuntimeError("replayed training disposition has no marker")
            return marker_result(marker) if marker["status"] == "COMPLETE" else marker_checkpoint_result(marker)
        admission_metrics: Mapping[str, Any] = {}
        if self._context is not None:
            decision = _admission_decision(payload, self._context)
            if decision is not None and decision.action == "drop":
                return TrainingJobResult(
                    outcome="stale", runtime_load_id=self._context.runtime_load_id, metrics=decision.metrics or None
                )
            if decision is not None:
                admission_metrics = decision.metrics
        with self._backend.prepare(payload, job_id=job_id, rollout_id=rollout_id, prior_marker=marker) as prepared:
            if isinstance(prepared, TrainingJobResult):
                if prepared.outcome not in {"stale", "storage_blocked"}:
                    raise RuntimeError("training preparation may only return stale or storage_blocked")
                return prepared
            return self._run(prepared, self._store, job_id, payload, admission_metrics)
        raise RuntimeError("training preparation suppressed an execution failure")

    def _run(
        self,
        prepared: PreparedTrainingJob,
        store: TrainingJobStore,
        job_id: str,
        payload: Mapping[str, Any],
        admission_metrics: Mapping[str, Any],
    ) -> TrainingJobResult:
        """Train and checkpoint one admitted job, recording RUNNING then CHECKPOINT."""
        checkpoint = prepared.checkpoint
        running: dict[str, Any] = {"status": "RUNNING", "job_id": job_id, "rollout_id": checkpoint.rollout_id}
        parent_runtime_load_id = payload.get("expected_runtime_load_id")
        if isinstance(parent_runtime_load_id, str) and parent_runtime_load_id:
            running["parent_runtime_load_id"] = parent_runtime_load_id
        # The owner: the adapter's scenario on a runtime training several, else the scenario the payload names.
        owner = checkpoint.scenario if checkpoint.scenario is not None else payload.get(JOB_OWNER_KEY)
        if isinstance(owner, str) and owner:
            running["scenario"] = owner
        if checkpoint.scenario_step is not None:
            # Reef reasons in scenario steps; the marker's rollout_id is the backend's own checkpoint index.
            running["scenario_step"] = checkpoint.scenario_step
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
            durable = {**admission_metrics, **metrics.durable}
            if durable:
                updates["metrics"] = durable
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
        objective: str,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
        scenario_step: int,
    ) -> PreparedTrainingStep:
        return self.training_runtime.prepare_training_step(
            batch,
            objective,
            algorithm_state,
            scheduling,
            scenario_step,
            serving_runtime_load_id=(
                self.inference_runtime.serving_runtime_load_id()
                if self.training_runtime.max_staleness > 0
                else self.inference_runtime.current_runtime_load_id()
            ),
        )

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        """Execute a native checkpoint job and stage its uncommitted weights."""
        with self._colocated_pause():
            checkpoint = self._validated_result(self.training_runtime.execute_training_job(payload))
        if checkpoint.outcome == "stale":
            self.operations.increment("stale_batches_total")
        if checkpoint.outcome in {"stale", "storage_blocked"}:
            self._resume_if_colocated()
            return checkpoint
        if checkpoint.outcome == "complete":
            if (self.training_runtime.training_job_status() or {}).get("commit_acknowledged") is True:
                self.inference_runtime.resume_admission()
            else:
                self.inference_runtime.pause_admission()
            return checkpoint
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise TrainingRuntimeError("deferred weight updates require a checkpoint with a training_job_id")
        if not self._colocated:
            # Training and checkpointing may overlap inference on disjoint
            # GPUs. Close admission only for the short serving-weight update.
            self.inference_runtime.pause_admission()
        with self.operations.measure("weight_sync"):
            updated = self.inference_runtime.resume_weight_update(checkpoint.training_job_id)
        return TrainingJobResult(
            "complete",
            updated.runtime_load_id,
            checkpoint.checkpoint_path,
            metrics=checkpoint.metrics,
            training_job_id=checkpoint.training_job_id,
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

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        """Discard a candidate before reopening the unchanged serving version."""
        self.training_runtime.reject_candidate(candidate, decision)
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
                self._finish_committed_training_job(committed_training_job_id)
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
        rollout_id, training_job_id = _pending_job_identity(training_job)
        if status == "UPDATING_WEIGHTS":
            with self.operations.measure("weight_sync"):
                self.inference_runtime.resume_weight_update(training_job_id)
        if (
            status == "COMPLETE"
            and training_job.get("commit_acknowledged") is not True
            and scenario_step == rollout_id + 1
            and committed_training_job_id is None
            and committed_training_without_job_id
        ):
            # Older bridges resumed before Reef committed and could not write
            # their job identity into the old commit schema. The exact
            # next-step training record is the strongest durable migration
            # proof available; rollback/non-training commits are excluded.
            self._finish_committed_training_job(training_job_id)
            return
        if scenario_step > rollout_id and committed_training_job_id == training_job_id:
            self._finish_committed_training_job(training_job_id)

    def acknowledge_commit(self, scenario_step: int, training_job_id: str, *, scenario: str | None = None) -> None:
        """Release a selected update only after its matching durable commit."""
        self.recover_pending_step(scenario_step, scenario=scenario, committed_training_job_id=training_job_id)

    def _finish_committed_training_job(self, training_job_id: str) -> None:
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

    @staticmethod
    def _validated_result(result: Any) -> TrainingJobResult:
        if not isinstance(result, TrainingJobResult):
            raise TrainingRuntimeError(f"train group handle returned invalid training result: {type(result).__name__}")
        return result


def _pending_job_identity(training_job: Mapping[str, Any]) -> tuple[int, str]:
    rollout_id = training_job.get("rollout_id")
    training_job_id = training_job.get("training_job_id")
    if (
        not isinstance(rollout_id, int)
        or isinstance(rollout_id, bool)
        or not isinstance(training_job_id, str)
        or not training_job_id
    ):
        raise TrainingRuntimeError("training-job status is missing its durable identity")
    return rollout_id, training_job_id


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
        self._execution = TrainingExecution(
            self._store, training, state, context=self._context, publisher=self._weight_publisher
        )
        self._closed = False
        self._completed_train_steps = 0
        self._last_train_rollout_id: int | None = None
        self._last_train_metrics: dict[str, Any] = {}
        self._operation_lock = Lock()
        self._training.start()
        self._inference_url = TrainingRecovery(self._publication, self._weight_publisher).restore(
            self._execution.recover()
        )

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        objective: str,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
    ) -> PreparedTrainingStep:
        return self._training.prepare_training_step(batch, objective, algorithm_state, scheduling)

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
            # Reef reasons in scenario steps; the marker's rollout id is the
            # backend's own checkpoint index (older markers carry only that).
            rollout_id=marker.get("scenario_step", marker["rollout_id"]),
            runtime_load_id=marker.get("runtime_load_id"),
            commit_acknowledged=marker.get("commit_acknowledged", False),
        )
        if "scenario" in marker:
            training_job["scenario"] = marker["scenario"]
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

    def republish_serving(self) -> str:
        """Recover serving actors and republish unchanged weights in place.

        This path is for an inference-engine replacement, not a training step.
        Keep the current token because the checkpoint/model tensors have not
        changed; the next optimizer-backed publication advances it normally.
        """
        with self._operation_lock:
            return self._publication.republish(self._context.runtime_load_id)

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
