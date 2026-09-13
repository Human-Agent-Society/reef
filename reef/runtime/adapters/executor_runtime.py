"""Backend-neutral training runtime and executor-backed configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from reef.core.batches import TrainingBatch, policy_samples
from reef.core.config import config_option
from reef.core.evaluation import SelectionDecision
from reef.runtime.base import InferenceRuntime, PreparedTrainingStep, TrainingJobResult, TrainingRuntime
from reef.runtime.candidates import CandidateTrainingDeferred, ModelCandidate, StaleCandidate
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.inference import InferenceBackendFactory, build_http_inference_backend
from reef.runtime.registry import RuntimeConfigError, RuntimeFactory, register_runtime_kind
from reef.runtime.settings import TrainingRuntimeSettings
from reef.runtime.training_group import (
    ExecutorTrainGroupHandle,
    TrainingGroupHandle,
    TrainingRuntimeError,
    training_job_status,
)


class ExecutorTrainingRuntime(TrainingRuntime):
    """Train and export checkpoints through a supplied worker control connection."""

    def __init__(self, train_group_handle: TrainingGroupHandle, *, max_staleness: int = 0) -> None:
        if not isinstance(max_staleness, int) or isinstance(max_staleness, bool) or max_staleness < 0:
            raise ValueError("max_staleness must be a non-negative integer")
        self._train_group_handle = train_group_handle
        self._max_staleness = max_staleness

    @property
    def train_group_handle(self) -> TrainingGroupHandle:
        return self._train_group_handle

    @property
    def max_staleness(self) -> int:
        return self._max_staleness

    @property
    def concurrent_training_scenarios(self) -> bool:
        return self.training_job_status().get("lora_mode") == "scenario"

    def training_job_status(self) -> Mapping[str, Any]:
        return training_job_status(self._train_group_handle)

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
        *,
        serving_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        prepared = self._train_group_handle.prepare_training_step(batch, step_preparer, algorithm_state)
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
            expected_runtime_load_id = serving_runtime_load_id
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

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        return self._validated_result(self._train_group_handle.execute_training_job(payload))

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        checkpoint = self.execute_training_job(payload)
        if checkpoint.outcome == "storage_blocked":
            if not isinstance(checkpoint.storage, Mapping):
                raise TrainingRuntimeError("training runtime returned invalid checkpoint storage status")
            raise CandidateTrainingDeferred(checkpoint.storage)
        if checkpoint.outcome == "stale":
            raise StaleCandidate(checkpoint.metrics)
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise TrainingRuntimeError("candidate training must stop after exporting a checkpoint")
        if checkpoint.checkpoint_path is None:
            raise TrainingRuntimeError("exported checkpoint must carry a checkpoint path")
        return ModelCandidate(
            candidate_id=checkpoint.training_job_id,
            training_job_id=checkpoint.training_job_id,
            checkpoint_path=checkpoint.checkpoint_path,
            current_runtime_load_id=None,
            training_metrics=dict(checkpoint.metrics or {}),
        )

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        self.reject_training_job(candidate.training_job_id)

    def reject_training_job(self, training_job_id: str) -> None:
        self._train_group_handle.reject_training_candidate(training_job_id)

    def shutdown(self) -> None:
        self._train_group_handle.shutdown()

    @staticmethod
    def _validated_result(result: Any) -> TrainingJobResult:
        if not isinstance(result, TrainingJobResult):
            raise TrainingRuntimeError(f"train group handle returned invalid training result: {type(result).__name__}")
        return result


def connect_executor_runtimes(
    *,
    train_group_handle: TrainingGroupHandle,
    inference: InferenceRuntime | None = None,
    inference_url: str | None = None,
    model_path: str = "",
    inference_timeout_s: float = 300.0,
    max_staleness: int = 0,
    inference_backend_factory: InferenceBackendFactory = build_http_inference_backend,
    inference_backend_config: Mapping[str, Any] | None = None,
) -> tuple[TrainingRuntime, InferenceRuntime]:
    """Assemble independent components over the existing deployment connection."""
    from reef.runtime.adapters.executor_inference import ExecutorInferenceRuntime

    if inference is not None and inference_url is not None:
        raise ValueError("pass inference or inference_url, not both")
    training = ExecutorTrainingRuntime(train_group_handle, max_staleness=max_staleness)
    if inference is None:
        inference = ExecutorInferenceRuntime(
            control=train_group_handle,
            inference_url=inference_url,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            inference_backend_factory=inference_backend_factory,
            inference_backend_config=inference_backend_config,
        )
    return training, inference


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
class ExecutorTrainingRuntimeFactory(RuntimeFactory):
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
    ) -> tuple[TrainingRuntime, InferenceRuntime]:
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
            return connect_executor_runtimes(**kwargs)
        except BaseException:
            if created:
                with suppress(Exception):
                    executor.shutdown()
            raise
