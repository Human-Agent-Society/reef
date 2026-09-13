"""Backend-neutral training runtime and executor-backed configuration."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from reef.core.batches import TrainingBatch, policy_samples
from reef.core.config import config_option
from reef.core.evaluation import SelectionDecision
from reef.runtime.adapters.backend_runtime import BackendInferenceRuntime
from reef.runtime.base import (
    InferenceAdmissionHandle,
    InferenceRuntime,
    ModelRuntime,
    PreparedTrainingStep,
    TrainingJobResult,
)
from reef.runtime.candidates import ActivatedModel, CandidateTrainingDeferred, ModelCandidate, StaleCandidate
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.inference import InferenceBackendFactory, build_http_inference_backend
from reef.runtime.registry import RuntimeConfigError, RuntimeFactory, register_runtime_kind
from reef.runtime.settings import TrainingRuntimeSettings
from reef.runtime.training_group import ExecutorTrainGroupHandle, TrainingGroupHandle, TrainingRuntimeError


class ExecutorModelRuntime(ModelRuntime):
    """Training lifecycle independent of worker placement and RPC transport.

    The training handle owns backend-specific payloads and worker control.
    Serving uses the separately configured inference backend.
    """

    def __init__(
        self,
        *,
        train_group_handle: TrainingGroupHandle,
        inference: InferenceRuntime | None = None,
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
        if inference is not None and inference_url is not None:
            raise ValueError("pass inference or inference_url, not both")
        self._discover_inference_url = inference is None and not inference_url
        if inference is None and not inference_url:
            # The deployment coordinator reports its independently owned
            # inference endpoint through the existing control connection.
            reported = self._train_group_handle.health().get("inference_url")
            if not isinstance(reported, str) or not reported:
                raise TrainingRuntimeError(
                    "inference_url is unset and the training actor does not report one; "
                    "set reef.inference_url or update the training actor"
                )
            inference_url = reported
        if inference is None:
            if inference_url is None:
                raise TrainingRuntimeError("inference_url is unset")
            inference = BackendInferenceRuntime(
                backend=inference_backend_factory(
                    inference_url.rstrip("/"),
                    model_path=model_path,
                    timeout_s=inference_timeout_s,
                    **dict(inference_backend_config or {}),
                ),
                base_url=inference_url,
                inference_timeout_s=inference_timeout_s,
            )
        super().__init__(inference=inference, training=train_group_handle)
        self._model_path = model_path
        self._max_staleness = max_staleness
        training_job = self._training_job_status()
        self._colocated = bool(training_job.get("colocate", False))
        adapter = training_job.get("lora_adapter")
        self._serving_adapter_name = adapter if isinstance(adapter, str) and adapter else None
        self._per_scenario_adapters = training_job.get("lora_mode") == "scenario"
        self._current_runtime_load_id: str | None = None
        self._sync_inference_admission(training_job)

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def train_group_handle(self) -> TrainingGroupHandle:
        return self._train_group_handle

    @property
    def max_staleness(self) -> int:
        return self._max_staleness

    async def acquire_inference(self) -> InferenceAdmissionHandle:
        if not self._train_group_handle.reconnects:
            return await super().acquire_inference()
        # Admission stays owned by training/commit reconciliation. A request
        # may verify or wait for serving, but must never reopen a gate that a
        # concurrent publication closed after this health snapshot was read.
        deadline = asyncio.get_running_loop().time() + self.inference_timeout_s
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TrainingRuntimeError("inference is unavailable during model recovery")
            status = await asyncio.wait_for(asyncio.to_thread(self._training_job_status), timeout=remaining)
            state = status["status"]
            available = status["serving_healthy"] and (
                state in {"IDLE", "REJECTED"}
                or (state == "COMPLETE" and status.get("commit_acknowledged") is True)
                or (state in {"RUNNING", "CHECKPOINT"} and not self._colocated)
            )
            if available:
                try:
                    return await asyncio.wait_for(super().acquire_inference(), timeout=min(0.25, remaining))
                except asyncio.TimeoutError:
                    continue
            await asyncio.sleep(min(0.25, remaining))

    def serving_adapter_name(self) -> str | None:
        return self._serving_adapter_name

    @property
    def concurrent_training_scenarios(self) -> bool:
        return self._per_scenario_adapters

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        if not self._per_scenario_adapters:
            return None
        adapters = self._training_job_status().get("lora_adapters") or {}
        entry = adapters.get(scenario)
        if not isinstance(entry, Mapping):
            return None
        version = entry.get("runtime_load_id")
        return version if isinstance(version, str) and version else None

    def adapter_residency_status(self) -> Mapping[str, Any] | None:
        """The training bridge's adapter residency, when it serves per-scenario adapters."""
        if not self._per_scenario_adapters:
            return None
        status = self._training_job_status().get("adapter_residency")
        return status if isinstance(status, Mapping) else None

    def serving_runtime_load_id(self) -> str | None:
        version = self._train_group_handle.serving_runtime_load_id()
        if version is not None and (not isinstance(version, str) or not version):
            raise TrainingRuntimeError("train group handle must return a non-empty serving runtime load ID or None")
        return version

    def current_runtime_load_id(self) -> str | None:
        return self._current_runtime_load_id

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedTrainingStep:
        prepared = self.training.prepare_training_step(batch, step_preparer, algorithm_state)
        if not isinstance(prepared, PreparedTrainingStep):
            raise TrainingRuntimeError(
                f"train group handle returned invalid prepared training step: {type(prepared).__name__}"
            )
        if prepared.action == "skip":
            return prepared
        if prepared.payload is None:
            raise TrainingRuntimeError("non-skip training preparation must carry a payload")
        payload = dict(prepared.payload)
        samples = policy_samples(batch)
        source_rows = payload.pop("source_rows", None)
        if source_rows is not None:
            # Wire rows follow the step schedule (epochs repeat rows, shuffle
            # reorders rollouts); producing versions and timestamps must follow
            # that same order.
            try:
                samples = tuple(samples[row] for row in source_rows)
            except (IndexError, TypeError) as exc:
                raise TrainingRuntimeError(f"prepared payload names invalid source rows: {exc}") from exc
        versions = tuple(sample.runtime_load_id for sample in samples)
        if not samples:
            raise TrainingRuntimeError("a training job requires at least one policy sample")
        version_spans = [
            [
                {
                    "start": span.start,
                    "end": span.end,
                    "runtime_load_id": span.runtime_load_id,
                }
                for span in sample.runtime_load_spans
            ]
            for sample in samples
        ]
        if any(version_spans):
            payload["producing_runtime_load_spans"] = version_spans
        if self._max_staleness == 0 and any(
            version is None and not spans for version, spans in zip(versions, version_spans, strict=True)
        ):
            raise TrainingRuntimeError("a training job requires a recorded producing runtime load ID for every sample")
        span_versions = {span["runtime_load_id"] for spans in version_spans for span in spans}
        recorded_versions = span_versions | {version for version in versions if version is not None}
        requires_staleness_admission = (
            self._max_staleness > 0 or any(version is None for version in versions) or len(recorded_versions) != 1
        )
        if not requires_staleness_admission:
            expected_runtime_load_id = recorded_versions.pop()
            if expected_runtime_load_id is None:
                raise TrainingRuntimeError("recorded producing runtime load ID cannot be null")
        else:
            expected_runtime_load_id = self.serving_runtime_load_id()
            if expected_runtime_load_id is None:
                raise TrainingRuntimeError("token staleness admission requires a verified serving runtime load ID")
            payload["max_staleness"] = self._max_staleness
            payload["producing_runtime_load_ids"] = list(versions)
        # The scenario step crosses into the backend job as ``rollout_id`` —
        # the training backend's own (wire) name for the same integer.
        payload.update(rollout_id=scenario_step, expected_runtime_load_id=expected_runtime_load_id)
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
            self.inference.pause_admission()

        try:
            checkpoint = self._validated_result(self.training.execute_training_job(payload))
        except BaseException:
            # A colocated pause may have succeeded before the backend rejected
            # the job. Reopen only when the durable status proves that no
            # training or checkpoint work started.
            job_state = None
            if self._colocated:
                with suppress(Exception):
                    job_state = self._training_job_status().get("status")
            if job_state == "IDLE":
                self.inference.resume_admission()
            raise
        if checkpoint.outcome in {"stale", "storage_blocked"}:
            if self._colocated:
                self.inference.resume_admission()
            return checkpoint
        if checkpoint.outcome == "complete":
            if self._training_job_status().get("commit_acknowledged") is True:
                self.inference.resume_admission()
            else:
                self.inference.pause_admission()
            return checkpoint
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise TrainingRuntimeError("deferred weight updates require a checkpoint with a training_job_id")

        if not self._colocated:
            # Training and checkpointing may overlap inference on disjoint
            # GPUs. Close admission only for the short serving-weight update.
            self.inference.pause_admission()
        updated = self._validated_result(self._train_group_handle.update_serving_weights(checkpoint.training_job_id))
        if updated.outcome != "complete" or updated.training_job_id != checkpoint.training_job_id:
            raise TrainingRuntimeError("serving-weight update returned an invalid completed result")
        return updated

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        """Train through checkpoint export without changing serving weights."""
        current = self.current_runtime_load_id()
        if self._colocated:
            self.inference.pause_admission()
        try:
            checkpoint = self._validated_result(self.training.execute_training_job(payload))
        except BaseException:
            job_state = None
            if self._colocated:
                with suppress(Exception):
                    job_state = self._training_job_status().get("status")
            if job_state == "IDLE":
                self.inference.resume_admission()
            raise
        if checkpoint.outcome == "storage_blocked":
            if self._colocated:
                self.inference.resume_admission()
            if not isinstance(checkpoint.storage, Mapping):
                raise TrainingRuntimeError("training runtime returned invalid checkpoint storage status")
            raise CandidateTrainingDeferred(checkpoint.storage)
        if checkpoint.outcome == "stale":
            if self._colocated:
                self.inference.resume_admission()
            raise StaleCandidate(checkpoint.metrics)
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise TrainingRuntimeError("candidate training must stop after exporting a checkpoint")
        if checkpoint.checkpoint_path is None:
            raise TrainingRuntimeError("exported checkpoint must carry a checkpoint path")
        return ModelCandidate(
            candidate_id=checkpoint.training_job_id,
            training_job_id=checkpoint.training_job_id,
            checkpoint_path=checkpoint.checkpoint_path,
            current_runtime_load_id=current,
            training_metrics=dict(checkpoint.metrics or {}),
        )

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        """Apply one selected checkpoint to serving."""
        if not self._colocated:
            self.inference.pause_admission()
        updated = self._validated_result(self._train_group_handle.update_serving_weights(candidate.training_job_id))
        if updated.outcome != "complete" or updated.training_job_id != candidate.training_job_id:
            raise TrainingRuntimeError("serving-weight update returned an invalid completed result")
        return ActivatedModel(candidate.candidate_id, updated.runtime_load_id)

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        self._train_group_handle.reject_training_candidate(candidate.training_job_id)
        self.inference.resume_admission()

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
            raise TrainingRuntimeError("committed_training_job_id must be a non-empty string or None")
        if not isinstance(committed_training_without_job_id, bool):
            raise TrainingRuntimeError("committed_training_without_job_id must be a boolean")
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
                raise TrainingRuntimeError("rejecting training job is missing its durable identity")
            self._train_group_handle.reject_training_candidate(training_job_id)
            self.inference.resume_admission()
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
            raise TrainingRuntimeError("training-job status is missing its durable identity")
        if status == "UPDATING_WEIGHTS":
            recovered = self._validated_result(self._train_group_handle.update_serving_weights(training_job_id))
            if recovered.outcome != "complete" or recovered.training_job_id != training_job_id:
                raise TrainingRuntimeError("recovered serving-weight update returned an invalid completed result")
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
        current_runtime_load_id = self.serving_runtime_load_id()
        self.inference.resume_admission()
        self._current_runtime_load_id = current_runtime_load_id

    def _sync_inference_admission(self, training_job: Mapping[str, Any]) -> bool:
        """Apply states that need no weight-update recovery; return if settled."""
        status = training_job["status"]
        if status in {"IDLE", "REJECTED"} or (
            status == "COMPLETE" and training_job.get("commit_acknowledged") is True
        ):
            current_runtime_load_id = self.serving_runtime_load_id()
            self.inference.resume_admission()
            self._current_runtime_load_id = current_runtime_load_id
            return True
        if status in {"RUNNING", "CHECKPOINT"}:
            if self._colocated:
                self.inference.pause_admission()
            else:
                if self._current_runtime_load_id is None:
                    self._current_runtime_load_id = self.serving_runtime_load_id()
                self.inference.resume_admission()
            return True
        self.inference.pause_admission()
        return False

    def _training_job_status(self) -> Mapping[str, Any]:
        health = self._train_group_handle.health()
        if not isinstance(health, Mapping):
            raise TrainingRuntimeError(f"train group handle returned invalid health: {type(health).__name__}")
        healthy = health.get("ok")
        if healthy is not None and not isinstance(healthy, bool):
            raise TrainingRuntimeError("train group returned malformed health status")
        # A group that reports its failure as recoverable is retried through
        # the normal reconciliation path instead of being declared dead.
        if healthy is False and health.get("recoverable") is not True:
            phase = health.get("phase")
            detail = f" in phase {phase!r}" if isinstance(phase, str) and phase else ""
            raise TrainingRuntimeError(f"train group is unhealthy{detail}")
        if self._discover_inference_url:
            reported = health.get("inference_url")
            if not isinstance(reported, str) or not reported:
                raise TrainingRuntimeError("training coordinator stopped reporting its inference endpoint")
            if reported.rstrip("/") != self.base_url:
                self.inference.reconnect(reported)
        status = health.get("training_job")
        if not isinstance(status, Mapping):
            raise TrainingRuntimeError("train group health is missing training_job status")
        deferred = status.get("deferred_weight_update")
        colocate = health.get("colocate", False)
        lora_adapter = health.get("lora_adapter")
        state = status.get("status", "COMPLETE")
        if deferred is not True:
            raise TrainingRuntimeError("Reef requires deferred serving-weight updates")
        if not isinstance(colocate, bool) or not isinstance(state, str):
            raise TrainingRuntimeError("train group returned malformed training-job status")
        if lora_adapter is not None and (not isinstance(lora_adapter, str) or not lora_adapter):
            raise TrainingRuntimeError("train group returned a malformed serving adapter name")
        lora_mode = health.get("lora_mode")
        if lora_mode is not None and lora_mode not in {"shared", "scenario"}:
            raise TrainingRuntimeError(f"train group returned unknown LoRA mode: {lora_mode!r}")
        lora_adapters = health.get("lora_adapters")
        if lora_adapters is not None and not isinstance(lora_adapters, Mapping):
            raise TrainingRuntimeError("train group returned malformed per-scenario adapters")
        adapter_residency = health.get("adapter_residency")
        if adapter_residency is not None and not isinstance(adapter_residency, Mapping):
            raise TrainingRuntimeError("train group returned malformed adapter residency")
        if "commit_acknowledged" in status and not isinstance(status["commit_acknowledged"], bool):
            raise TrainingRuntimeError("train group returned malformed commit acknowledgement")
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
            raise TrainingRuntimeError(f"train group returned unknown training-job status: {state!r}")
        return {
            **dict(status),
            "serving_healthy": healthy is not False and health.get("phase") != "recovering",
            "colocate": colocate,
            "lora_adapter": lora_adapter,
            "lora_mode": lora_mode,
            "lora_adapters": dict(lora_adapters) if lora_adapters else {},
            "adapter_residency": dict(adapter_residency) if adapter_residency is not None else None,
        }

    @staticmethod
    def _validated_result(result: Any) -> TrainingJobResult:
        if not isinstance(result, TrainingJobResult):
            raise TrainingRuntimeError(f"train group handle returned invalid training result: {type(result).__name__}")
        return result


def _executor_config(value: Mapping[str, Any]) -> ExecutorConfig:
    workers = value.get("workers", ())
    if not isinstance(workers, Sequence) or isinstance(workers, (str, bytes)):
        raise RuntimeConfigError("runtime.executor.workers must be a sequence of worker specifications")
    specs = []
    for worker in workers:
        if isinstance(worker, WorkerSpec):
            specs.append(worker)
        elif isinstance(worker, Mapping):
            try:
                specs.append(WorkerSpec(**dict(worker)))
            except (TypeError, ValueError) as exc:
                raise RuntimeConfigError(f"invalid runtime.executor worker: {exc}") from exc
        else:
            raise RuntimeConfigError("runtime.executor.workers entries must be WorkerSpec objects or mappings")
    try:
        return ExecutorConfig(
            backend=value.get("backend", "auto"),
            workers=tuple(specs),
            options=value.get("options", {}),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"invalid runtime.executor configuration: {exc}") from exc


@dataclass(frozen=True)
class ExecutorRuntimeSettings(TrainingRuntimeSettings):
    coordinator_rank: int = config_option(0, help="Rank of the training coordinator worker.")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.coordinator_rank < 0:
            raise ValueError("runtime.coordinator_rank must be non-negative")


@register_runtime_kind
class ExecutorModelRuntimeFactory(RuntimeFactory):
    """Create a training coordinator using a configured executor.

    The executor entry accepts an existing Executor, an ExecutorConfig, or a
    mapping with backend, workers, and options. The worker at coordinator_rank
    (default 0) implements TrainingGroupHandle's methods; it may manage its
    own model-parallel worker group.
    """

    kind = "executor_training"

    def config_type(self) -> type:
        return ExecutorRuntimeSettings

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        injected = {
            key: config[key] for key in ("executor", "inference", "inference_backend_factory") if key in config
        }
        values = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        return {**values, **injected}

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> ModelRuntime:
        value = config.get("executor")
        if isinstance(value, Mapping):
            value = _executor_config(value)
        created = False
        if isinstance(value, ExecutorConfig):
            executor = Executor.create(value)
            created = True
        elif isinstance(value, Executor):
            executor = value
        else:
            raise RuntimeConfigError("runtime.executor must be an Executor, ExecutorConfig, or configuration mapping")
        try:
            handle = ExecutorTrainGroupHandle(
                executor,
                rank=config.get("coordinator_rank", 0),
                timeout_s=(
                    config["train_timeout_s"]
                    if config.get("train_timeout_s") is not None
                    else config.get("inference_timeout_s", 300.0)
                ),
            )
            kwargs: dict[str, Any] = {"train_group_handle": handle, "model_path": model_path}
            for key in (
                "inference",
                "inference_url",
                "inference_timeout_s",
                "max_staleness",
                "inference_backend_factory",
                "inference_backend_config",
            ):
                if key in config:
                    kwargs[key] = config[key]
            return ExecutorModelRuntime(**kwargs)
        except BaseException:
            if created:
                with suppress(Exception):
                    executor.shutdown()
            raise
