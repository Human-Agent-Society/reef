"""Runtime adapter that drives a Ray training actor group directly.

This adapter is for training backends (e.g. slime's Megatron actors) that are
exposed as Ray actors rather than as HTTP services. The reef process must run
inside the same Ray cluster as the training actors.

The runtime does not import a concrete training backend. Backends implement
a train group handle (:class:`RayTrainGroupHandle`) around their own actor
groups; ``RayRuntime`` drives training only through that handle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

from reef.core.errors import ReefError
from reef.runtime.base import PreparedTrainingStep, TrainingJobResult, TrainingRuntime
from reef.runtime.candidates import ActivatedModel, CandidateTrainingDeferred, ModelCandidate, StaleCandidate
from reef.runtime.inference import InferenceBackend, InferenceBackendFactory, build_http_inference_backend
from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.runtime.registry import RuntimeConfigError, RuntimeFactory, register_runtime_kind
from reef.train.evaluation.contracts import SelectionDecision
from reef.train.types import TrainingBatch, policy_samples


class RayRuntimeError(ReefError):
    """Raised when the Ray runtime fails to drive the training actors."""


class RayTrainGroupHandle(ABC):
    """Train group handle: the contract every Ray training backend implements.

    Each backend wraps its own actor group and payload format behind this
    handle. ``RayRuntime`` depends only on this contract.
    """

    def serving_weight_version(self) -> str | None:
        """Return the serving version, or ``None`` when unavailable."""
        return None

    @abstractmethod
    def health(self) -> Mapping[str, Any]:
        """Return backend health and durable training-job state."""
        ...

    @abstractmethod
    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        """Activate one checkpointed candidate in the serving engine."""
        ...

    @abstractmethod
    def reject_training_candidate(self, training_job_id: str) -> None:
        """Finish a rejected candidate without changing serving weights."""
        ...

    @abstractmethod
    def acknowledge_training_commit(self, training_job_id: str) -> None:
        """Acknowledge Reef's durable commit for an activated candidate."""
        ...

    @abstractmethod
    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
    ) -> PreparedTrainingStep:
        """Prepare the step signal and backend payload for one reserved batch."""

    @abstractmethod
    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        """Execute one idempotent training job, through its durable checkpoint."""


class RayRuntime(TrainingRuntime):
    """Training runtime that drives a Ray actor group in the local cluster.

    ``inference_backend`` still proxies HTTP to the serving engines (e.g. an
    SGLang router); only the training lifecycle uses Ray.
    """

    def __init__(
        self,
        *,
        train_group_handle: RayTrainGroupHandle,
        inference_url: str | None = None,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        max_staleness: int = 0,
        inference_backend_factory: InferenceBackendFactory = build_http_inference_backend,
        inference_backend_config: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(max_staleness, int) or isinstance(max_staleness, bool) or max_staleness < 0:
            raise ValueError("max_staleness must be a non-negative integer")
        self._train_group_handle = train_group_handle
        if not inference_url:
            # The training backend started the serving engines, so it is the
            # authority on where they listen; a deployment only overrides
            # this when it fronts the engines with something else.
            reported = self._train_group_handle.health().get("inference_url")
            if not isinstance(reported, str) or not reported:
                raise RayRuntimeError(
                    "inference_url is unset and the training actor does not report one; "
                    "set reef.inference_url or update the training actor"
                )
            inference_url = reported
        super().__init__(base_url=inference_url, inference_timeout_s=inference_timeout_s)
        self._model_path = model_path
        self._max_staleness = max_staleness
        self._inference_backend = inference_backend_factory(
            self.base_url,
            model_path=model_path,
            timeout_s=self.inference_timeout_s,
            **dict(inference_backend_config or {}),
        )
        training_job = self._training_job_status()
        self._colocated = bool(training_job.get("colocate", False))
        adapter = training_job.get("lora_adapter")
        self._serving_adapter_name = adapter if isinstance(adapter, str) and adapter else None
        self._per_scenario_adapters = training_job.get("lora_mode") == "scenario"
        self._current_weight_version: str | None = None
        self._sync_inference_admission(training_job)

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def train_group_handle(self) -> RayTrainGroupHandle:
        return self._train_group_handle

    @property
    def max_staleness(self) -> int:
        return self._max_staleness

    @property
    def inference_backend(self) -> InferenceBackend:
        return self._inference_backend

    def serving_adapter_name(self) -> str | None:
        return self._serving_adapter_name

    @property
    def concurrent_training_scenarios(self) -> bool:
        return self._per_scenario_adapters

    def serving_adapter_version(self, scenario: str) -> str | None:
        if not self._per_scenario_adapters:
            return None
        adapters = self._training_job_status().get("lora_adapters") or {}
        entry = adapters.get(scenario)
        if not isinstance(entry, Mapping):
            return None
        version = entry.get("weight_version")
        return version if isinstance(version, str) and version else None

    def adapter_residency_status(self) -> Mapping[str, Any] | None:
        """The training bridge's adapter residency, when it serves per-scenario adapters."""
        if not self._per_scenario_adapters:
            return None
        status = self._training_job_status().get("adapter_residency")
        return status if isinstance(status, Mapping) else None

    def serving_weight_version(self) -> str | None:
        version = self._train_group_handle.serving_weight_version()
        if version is not None and (not isinstance(version, str) or not version):
            raise RayRuntimeError("train group handle must return a non-empty serving weight version or None")
        return version

    def current_weight_version(self) -> str | None:
        return self._current_weight_version

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedTrainingStep:
        prepared = self._train_group_handle.prepare_training_step(batch, step_preparer, algorithm_state)
        if not isinstance(prepared, PreparedTrainingStep):
            raise RayRuntimeError(
                f"train group handle returned invalid prepared training step: {type(prepared).__name__}"
            )
        if prepared.action == "skip":
            return prepared
        if prepared.payload is None:
            raise RayRuntimeError("non-skip training preparation must carry a payload")
        payload = dict(prepared.payload)
        samples = policy_samples(batch)
        source_rows = payload.pop("source_rows", None)
        if source_rows is not None:
            # Wire rows follow the step schedule (epochs repeat rows, shuffle
            # reorders rollouts); provenance must line up with them, not with
            # the batch's own order.
            try:
                samples = tuple(samples[row] for row in source_rows)
            except (IndexError, TypeError) as exc:
                raise RayRuntimeError(f"prepared payload names invalid source rows: {exc}") from exc
        versions = tuple(sample.weight_version for sample in samples)
        if not samples:
            raise RayRuntimeError("a training job requires at least one policy sample")
        version_spans = [
            [
                {
                    "start": span.start,
                    "end": span.end,
                    "weight_version": span.weight_version,
                }
                for span in sample.weight_version_spans
            ]
            for sample in samples
        ]
        if any(version_spans):
            payload["producing_weight_version_spans"] = version_spans
        if self._max_staleness == 0 and any(
            version is None and not spans for version, spans in zip(versions, version_spans, strict=True)
        ):
            raise RayRuntimeError("a training job requires a recorded producing weight version for every sample")
        span_versions = {span["weight_version"] for spans in version_spans for span in spans}
        recorded_versions = span_versions | {version for version in versions if version is not None}
        requires_staleness_admission = (
            self._max_staleness > 0 or any(version is None for version in versions) or len(recorded_versions) != 1
        )
        if not requires_staleness_admission:
            expected_weight_version = recorded_versions.pop()
            if expected_weight_version is None:
                raise RayRuntimeError("recorded producing weight version cannot be null")
        else:
            expected_weight_version = self.serving_weight_version()
            if expected_weight_version is None:
                raise RayRuntimeError("token staleness admission requires a verified serving weight version")
            payload["max_staleness"] = self._max_staleness
            payload["producing_weight_versions"] = list(versions)
        # The scenario step crosses into the backend job as ``rollout_id`` —
        # the training backend's own (wire) name for the same integer.
        payload.update(rollout_id=scenario_step, expected_weight_version=expected_weight_version)
        return PreparedTrainingStep(
            action="train",
            payload=payload,
            next_algorithm_state=prepared.next_algorithm_state,
            metrics=prepared.metrics,
        )

    def execute_training_job(
        self,
        payload: Mapping[str, Any],
    ) -> TrainingJobResult:
        if self._colocated:
            # New requests wait without occupying a service worker and bind
            # the head committed below. Already-admitted requests stay inside
            # SGLang: the colocated bridge retracts their KV before handing
            # the GPUs to Megatron, then SGLang re-prefills them after resume.
            self._inference_admission.close()

        try:
            checkpoint = self._validated_result(self._train_group_handle.execute_training_job(payload))
        except BaseException:
            # A colocated pause may have succeeded before the backend rejected
            # the job. Reopen only when the durable status proves that no
            # training or checkpoint work started.
            job_state = None
            if self._colocated:
                with suppress(Exception):
                    job_state = self._training_job_status().get("status")
            if job_state == "IDLE":
                self._inference_admission.open()
            raise
        if checkpoint.outcome in {"stale", "storage_blocked"}:
            if self._colocated:
                self._inference_admission.open()
            return checkpoint
        if checkpoint.outcome == "complete":
            if self._training_job_status().get("commit_acknowledged") is True:
                self._inference_admission.open()
            else:
                self._inference_admission.close()
            return checkpoint
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise RayRuntimeError("deferred weight updates require a checkpoint with a training_job_id")

        if not self._colocated:
            # Training and checkpointing may overlap inference on disjoint
            # GPUs. Close admission only for the short serving-weight update.
            self._inference_admission.close()
        updated = self._validated_result(self._train_group_handle.update_serving_weights(checkpoint.training_job_id))
        if updated.outcome != "complete" or updated.training_job_id != checkpoint.training_job_id:
            raise RayRuntimeError("serving-weight update returned an invalid completed result")
        return updated

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        """Train through checkpoint export without changing serving weights."""
        current = self.current_weight_version()
        if self._colocated:
            self._inference_admission.close()
        try:
            checkpoint = self._validated_result(self._train_group_handle.execute_training_job(payload))
        except BaseException:
            job_state = None
            if self._colocated:
                with suppress(Exception):
                    job_state = self._training_job_status().get("status")
            if job_state == "IDLE":
                self._inference_admission.open()
            raise
        if checkpoint.outcome == "storage_blocked":
            if self._colocated:
                self._inference_admission.open()
            if not isinstance(checkpoint.storage, Mapping):
                raise RayRuntimeError("training runtime returned invalid checkpoint storage status")
            raise CandidateTrainingDeferred(checkpoint.storage)
        if checkpoint.outcome == "stale":
            if self._colocated:
                self._inference_admission.open()
            raise StaleCandidate(checkpoint.metrics)
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise RayRuntimeError("candidate training must stop after exporting a checkpoint")
        if checkpoint.checkpoint_path is None:
            raise RayRuntimeError("exported checkpoint must carry a checkpoint path")
        return ModelCandidate(
            candidate_id=checkpoint.training_job_id,
            current_version=current,
            training_job_id=checkpoint.training_job_id,
            checkpoint_path=checkpoint.checkpoint_path,
            current_weight_version=current,
            training_metrics=dict(checkpoint.metrics or {}),
        )

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        """Apply one selected checkpoint to serving."""
        if not self._colocated:
            self._inference_admission.close()
        updated = self._validated_result(self._train_group_handle.update_serving_weights(candidate.training_job_id))
        if updated.outcome != "complete" or updated.training_job_id != candidate.training_job_id:
            raise RayRuntimeError("serving-weight update returned an invalid completed result")
        return ActivatedModel(candidate.candidate_id, updated.weight_version)

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        self._train_group_handle.reject_training_candidate(candidate.training_job_id)
        self._inference_admission.open()

    def reconcile_training_job(
        self,
        scenario_step: int,
        *,
        committed_training_job_id: str | None = None,
        committed_training_without_job_id: bool = False,
        scenario: str | None = None,
    ) -> None:
        if committed_training_job_id is not None and (
            not isinstance(committed_training_job_id, str) or not committed_training_job_id
        ):
            raise RayRuntimeError("committed_training_job_id must be a non-empty string or None")
        if not isinstance(committed_training_without_job_id, bool):
            raise RayRuntimeError("committed_training_without_job_id must be a boolean")
        training_job = self._training_job_status()
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
                raise RayRuntimeError("rejecting training job is missing its durable identity")
            self._train_group_handle.reject_training_candidate(training_job_id)
            self._inference_admission.open()
            return
        if status not in {"UPDATING_WEIGHTS", "READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}:
            return
        rollout_id = training_job.get("rollout_id")
        training_job_id = training_job.get("training_job_id")
        if (
            not isinstance(rollout_id, int)
            or isinstance(rollout_id, bool)
            or not isinstance(training_job_id, str)
            or not training_job_id
        ):
            raise RayRuntimeError("training-job status is missing its durable identity")
        if status == "UPDATING_WEIGHTS":
            recovered = self._validated_result(self._train_group_handle.update_serving_weights(training_job_id))
            if recovered.outcome != "complete" or recovered.training_job_id != training_job_id:
                raise RayRuntimeError("recovered serving-weight update returned an invalid completed result")
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

    def _finish_committed_training_job(self, training_job_id: str) -> None:
        self._train_group_handle.acknowledge_training_commit(training_job_id)
        current_weight_version = self.serving_weight_version()
        self._inference_admission.open()
        self._current_weight_version = current_weight_version

    def _sync_inference_admission(self, training_job: Mapping[str, Any]) -> bool:
        """Apply states that need no weight-update recovery; return if settled."""
        status = training_job["status"]
        if status in {"IDLE", "REJECTED"} or (
            status == "COMPLETE" and training_job.get("commit_acknowledged") is True
        ):
            current_weight_version = self.serving_weight_version()
            self._inference_admission.open()
            self._current_weight_version = current_weight_version
            return True
        if status in {"RUNNING", "CHECKPOINT"}:
            if self._colocated:
                self._inference_admission.close()
            else:
                if self._current_weight_version is None:
                    self._current_weight_version = self.serving_weight_version()
                self._inference_admission.open()
            return True
        self._inference_admission.close()
        return False

    def _training_job_status(self) -> Mapping[str, Any]:
        health = self._train_group_handle.health()
        if not isinstance(health, Mapping):
            raise RayRuntimeError(f"train group handle returned invalid health: {type(health).__name__}")
        healthy = health.get("ok")
        if healthy is not None and not isinstance(healthy, bool):
            raise RayRuntimeError("train group returned malformed health status")
        if healthy is False:
            phase = health.get("phase")
            detail = f" in phase {phase!r}" if isinstance(phase, str) and phase else ""
            raise RayRuntimeError(f"train group is unhealthy{detail}")
        status = health.get("training_job")
        if not isinstance(status, Mapping):
            raise RayRuntimeError("train group health is missing training_job status")
        deferred = status.get("deferred_weight_update")
        colocate = health.get("colocate", False)
        lora_adapter = health.get("lora_adapter")
        state = status.get("status", "COMPLETE")
        if deferred is not True:
            raise RayRuntimeError("Reef requires deferred serving-weight updates")
        if not isinstance(colocate, bool) or not isinstance(state, str):
            raise RayRuntimeError("train group returned malformed training-job status")
        if lora_adapter is not None and (not isinstance(lora_adapter, str) or not lora_adapter):
            raise RayRuntimeError("train group returned a malformed serving adapter name")
        lora_mode = health.get("lora_mode")
        if lora_mode is not None and lora_mode not in {"shared", "scenario"}:
            raise RayRuntimeError(f"train group returned unknown LoRA mode: {lora_mode!r}")
        lora_adapters = health.get("lora_adapters")
        if lora_adapters is not None and not isinstance(lora_adapters, Mapping):
            raise RayRuntimeError("train group returned malformed per-scenario adapters")
        adapter_residency = health.get("adapter_residency")
        if adapter_residency is not None and not isinstance(adapter_residency, Mapping):
            raise RayRuntimeError("train group returned malformed adapter residency")
        if "commit_acknowledged" in status and not isinstance(status["commit_acknowledged"], bool):
            raise RayRuntimeError("train group returned malformed commit acknowledgement")
        if state not in {
            "IDLE",
            "RUNNING",
            "CHECKPOINT",
            "UPDATING_WEIGHTS",
            "READY_TO_COMMIT",
            "HEAD_COMMITTED",
            "COMPLETE",
            "REJECTED",
            "REJECTING",
        }:
            raise RayRuntimeError(f"train group returned unknown training-job status: {state!r}")
        return {
            **dict(status),
            "colocate": colocate,
            "lora_adapter": lora_adapter,
            "lora_mode": lora_mode,
            "lora_adapters": dict(lora_adapters) if lora_adapters else {},
            "adapter_residency": dict(adapter_residency) if adapter_residency is not None else None,
        }

    @staticmethod
    def _validated_result(result: Any) -> TrainingJobResult:
        if not isinstance(result, TrainingJobResult):
            raise RayRuntimeError(f"train group handle returned invalid training result: {type(result).__name__}")
        return result


def _require_ray():
    """Import ray lazily so no-update deployments need not install it."""
    try:
        import ray
    except ImportError as exc:
        raise RayRuntimeError(
            "remote Ray runtimes require the 'ray' package; install it to connect to a training backend"
        ) from exc
    return ray


class RemoteRayTrainGroupHandle(RayTrainGroupHandle):
    """Train group handle for a training backend exposed as a named Ray actor.

    Sends framework-agnostic sample rows; the backend's train group actor
    converts them into its native training format.
    """

    def __init__(self, train_group_actor: Any, *, timeout_s: float = 300.0) -> None:
        if timeout_s <= 0:
            raise RayRuntimeError("Ray training timeout must be positive")
        self._train_group_actor = train_group_actor
        self._timeout_s = timeout_s

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
    ) -> PreparedTrainingStep:
        return self._get(
            self._train_group_actor.prepare_training_step.remote(batch, step_preparer, dict(algorithm_state)),
            timeout=self._timeout_s,
        )

    def serving_weight_version(self) -> str | None:
        version = self._get(
            self._train_group_actor.serving_weight_version.remote(),
            timeout=self._timeout_s,
        )
        return None if version is None else str(version)

    def health(self) -> Mapping[str, Any]:
        try:
            method = self._train_group_actor.health
        except AttributeError as exc:
            raise RayRuntimeError(
                "remote training actor predates deferred serving-weight updates; "
                "restart the Reef service and training actor together"
            ) from exc
        value = self._get(
            method.remote(),
            timeout=self._timeout_s,
        )
        if not isinstance(value, Mapping):
            raise RayRuntimeError("remote train group returned invalid health")
        if not isinstance(value.get("training_job"), Mapping):
            raise RayRuntimeError(
                "remote training actor predates deferred serving-weight updates; "
                "restart the Reef service and training actor together"
            )
        return dict(value)

    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        return self._get(
            self._train_group_actor.update_serving_weights.remote(training_job_id),
            timeout=self._timeout_s,
        )

    def reject_training_candidate(self, training_job_id: str) -> None:
        self._get(
            self._train_group_actor.reject_training_candidate.remote(training_job_id),
            timeout=self._timeout_s,
        )

    def acknowledge_training_commit(self, training_job_id: str) -> None:
        self._get(
            self._train_group_actor.acknowledge_training_commit.remote(training_job_id),
            timeout=self._timeout_s,
        )

    def execute_training_job(
        self,
        payload: Mapping[str, Any],
    ) -> TrainingJobResult:
        return self._get(
            self._train_group_actor.execute_training_job.remote(dict(payload)),
            timeout=self._timeout_s,
        )

    @staticmethod
    def _get(value: Any, *, timeout: float = 300) -> Any:
        return _require_ray().get(value, timeout=timeout)


def connect_ray_runtime(
    *,
    inference_url: str | None = None,
    actor_name: str = DEFAULT_ACTOR_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    ray_address: str | None = None,
    model_path: str = "",
    inference_timeout_s: float = 300.0,
    train_timeout_s: float | None = None,
    max_staleness: int = 0,
    inference_backend_factory: InferenceBackendFactory = build_http_inference_backend,
    inference_backend_config: Mapping[str, Any] | None = None,
) -> RayRuntime:
    """Connect to a named training actor and return a ready :class:`RayRuntime`.

    Reef and the backend run as separate services in one Ray cluster.
    ``namespace`` must match the namespace used when the backend actor was
    created. ``inference_url`` defaults to the address the actor reports.
    """
    ray = _require_ray()
    if not ray.is_initialized():
        ray.init(address=ray_address or "auto", namespace=namespace)
    train_group_actor = ray.get_actor(actor_name)
    # A training step legitimately outlasts an inference request.
    return RayRuntime(
        train_group_handle=RemoteRayTrainGroupHandle(
            train_group_actor, timeout_s=train_timeout_s if train_timeout_s is not None else inference_timeout_s
        ),
        inference_url=inference_url,
        model_path=model_path,
        inference_timeout_s=inference_timeout_s,
        max_staleness=max_staleness,
        inference_backend_factory=inference_backend_factory,
        inference_backend_config=inference_backend_config,
    )


@register_runtime_kind
class RayTrainingRuntimeFactory(RuntimeFactory):
    """Build (connect) a :class:`RayRuntime` from a runtime config section.

    The config mirrors :func:`connect_ray_runtime`'s keyword arguments. A
    ``connect`` entry may inject an alternative connector callable (tests use
    this to stub the Ray cluster); it defaults to :func:`connect_ray_runtime`.
    """

    kind = "ray_training"

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> TrainingRuntime:
        connect = config.get("connect", connect_ray_runtime)
        if not callable(connect):
            raise RuntimeConfigError("runtime.connect must be callable")
        kwargs: dict[str, Any] = {"model_path": model_path}
        for key in (
            "inference_url",
            "actor_name",
            "namespace",
            "ray_address",
            "inference_timeout_s",
            "train_timeout_s",
            "max_staleness",
            "inference_backend_factory",
            "inference_backend_config",
        ):
            if key in config:
                kwargs[key] = config[key]
        runtime = connect(**kwargs)
        if not isinstance(runtime, TrainingRuntime):
            raise RuntimeConfigError(f"runtime connector returned {type(runtime).__name__}, not a TrainingRuntime")
        return runtime
