"""Inference requests and weight publication over an existing control connection."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from reef.runtime.adapters.backend_runtime import BackendInferenceRuntime
from reef.runtime.base import InferenceAdmissionHandle, TrainingJobResult
from reef.runtime.candidates import ActivatedModel, ModelCandidate
from reef.runtime.inference import InferenceBackendFactory, build_http_inference_backend
from reef.runtime.training_group import TrainingGroupHandle, TrainingRuntimeError, training_job_status


class ExecutorInferenceRuntime(BackendInferenceRuntime):
    """Own request/serving state; borrow the deployment's control connection."""

    def __init__(
        self,
        *,
        control: TrainingGroupHandle,
        inference_url: str | None = None,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        inference_backend_factory: InferenceBackendFactory = build_http_inference_backend,
        inference_backend_config: Mapping[str, Any] | None = None,
    ) -> None:
        self._control = control
        self._discover_inference_url = not inference_url
        if not inference_url:
            inference_url = training_job_status(control).get("inference_url")
            if not isinstance(inference_url, str) or not inference_url:
                raise TrainingRuntimeError("inference_url is unset and the deployment does not report one")
        super().__init__(
            base_url=inference_url,
            inference_timeout_s=inference_timeout_s,
            backend=inference_backend_factory(
                inference_url.rstrip("/"),
                model_path=model_path,
                timeout_s=inference_timeout_s,
                **dict(inference_backend_config or {}),
            ),
        )
        self._model_path = model_path
        # Only the coordinator can reconcile pending publication with Reef's
        # durable head and authorize requests after attachment.
        self.pause_admission()

    @property
    def model_path(self) -> str:
        return self._model_path

    def _publication_status(self) -> Mapping[str, Any]:
        status = training_job_status(self._control)
        if self._discover_inference_url:
            reported = status.get("inference_url")
            if not isinstance(reported, str) or not reported:
                raise TrainingRuntimeError("deployment stopped reporting its inference endpoint")
            if reported.rstrip("/") != self.base_url:
                self.reconnect(reported)
        return status

    async def acquire_inference(self) -> InferenceAdmissionHandle:
        if not self._control.reconnects:
            return await super().acquire_inference()
        # Admission stays owned by training/commit reconciliation. A request
        # may verify or wait for serving, but must never reopen a gate that a
        # concurrent publication closed after this health snapshot was read.
        deadline = asyncio.get_running_loop().time() + self.inference_timeout_s
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TrainingRuntimeError("inference is unavailable during model recovery")
            status = await asyncio.wait_for(asyncio.to_thread(self._publication_status), timeout=remaining)
            state = status["status"]
            available = status["serving_healthy"] and (
                state in {"IDLE", "REJECTED"}
                or (state == "COMPLETE" and status.get("commit_acknowledged") is True)
                or (state in {"RUNNING", "CHECKPOINT"} and not status.get("colocate", False))
            )
            if available:
                try:
                    return await asyncio.wait_for(super().acquire_inference(), timeout=min(0.25, remaining))
                except asyncio.TimeoutError:
                    continue
            await asyncio.sleep(min(0.25, remaining))

    def serving_adapter_name(self) -> str | None:
        return self._publication_status().get("lora_adapter")

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        adapters = self._publication_status().get("lora_adapters") or {}
        entry = adapters.get(scenario)
        if not isinstance(entry, Mapping):
            return None
        version = entry.get("runtime_load_id")
        return version if isinstance(version, str) and version else None

    def adapter_residency_status(self) -> Mapping[str, Any] | None:
        return self._publication_status().get("adapter_residency")

    def serving_runtime_load_id(self) -> str | None:
        self._publication_status()
        version = self._control.serving_runtime_load_id()
        if version is not None and (not isinstance(version, str) or not version):
            raise TrainingRuntimeError("train group handle must return a non-empty serving runtime load ID or None")
        return version

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        return self.update_serving_weights(candidate.training_job_id, candidate_id=candidate.candidate_id)

    def update_serving_weights(self, training_job_id: str, *, candidate_id: str | None = None) -> ActivatedModel:
        updated = self._control.update_serving_weights(training_job_id)
        if not isinstance(updated, TrainingJobResult):
            raise TrainingRuntimeError("inference control returned invalid weight-update result")
        if updated.outcome != "complete" or updated.training_job_id != training_job_id:
            raise TrainingRuntimeError("serving-weight update returned an invalid completed result")
        return ActivatedModel(candidate_id or training_job_id, updated.runtime_load_id)

    def resume_weight_update(self, training_job_id: str) -> ActivatedModel:
        return self.update_serving_weights(training_job_id)

    def acknowledge_publication(self, training_job_id: str) -> None:
        self._control.acknowledge_training_commit(training_job_id)

    def shutdown(self) -> None:
        self.pause_admission()
