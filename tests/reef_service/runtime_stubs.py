"""Test doubles for Reef runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.evaluation import SelectionDecision
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.base import PreparedTrainingStep, TrainingRuntime
from reef.runtime.candidates import ActivatedModel, ModelCandidate
from reef.runtime.inference import InferenceBackend
from reef.train.types import TrainingBatch


class StubTrainingRuntime(TrainingRuntime):
    """Training and a separate inference fixture for in-process recipe smoke tests.

    It satisfies the constructor and typing contract so a training recipe can
    be built and its data path driven — records in, processor pairing, batch
    reservation, step preparation — without a running model service. The
    inference surface is disabled and training-path methods raise
    ``NotImplementedError``; tests reaching model execution supply their own
    implementations.
    """

    def __init__(self, base_url: str = "http://training-runtime", *, max_staleness: int = 0) -> None:
        if not isinstance(max_staleness, int) or isinstance(max_staleness, bool) or max_staleness < 0:
            raise ValueError("max_staleness must be a non-negative integer")
        self.inference = StubInferenceRuntime(self, base_url=base_url)
        self._max_staleness = max_staleness

    @property
    def max_staleness(self) -> int:
        return self._max_staleness

    @property
    def inference_backend(self) -> InferenceBackend | None:  # type: ignore[override]
        """No backend: smoke tests never route inference through the stub."""
        return None

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
        *,
        serving_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        raise NotImplementedError("StubTrainingRuntime does not prepare training steps")

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        raise NotImplementedError("StubTrainingRuntime does not train candidates")

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        raise NotImplementedError("StubTrainingRuntime does not activate candidates")

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        raise NotImplementedError("StubTrainingRuntime does not reject candidates")

    @property
    def base_url(self):
        return self.inference.base_url

    def serving_runtime_load_id(self):
        return None

    def serving_adapter_name(self):
        return None

    def serving_adapter_runtime_load_id(self, scenario):
        return None

    def current_runtime_load_id(self):
        return self.inference.current_runtime_load_id()

    def restore_checkpoint(self, artifact):
        return None

    def restore_serving_checkpoint(self, artifact):
        raise RuntimeError("test inference runtime does not restore checkpoints")


class StubInferenceRuntime(InferenceProxyRuntime):
    """Receiver hooks for the in-process training test fixtures."""

    def __init__(self, training, **kwargs):
        super().__init__(**kwargs)
        self.fixture = training

    @property
    def inference_backend(self):
        return self.fixture.inference_backend

    def serving_runtime_load_id(self):
        return self.fixture.serving_runtime_load_id()

    def serving_adapter_name(self):
        return self.fixture.serving_adapter_name()

    def serving_adapter_runtime_load_id(self, scenario):
        return self.fixture.serving_adapter_runtime_load_id(scenario)

    def activate_candidate(self, candidate):
        return self.fixture.activate_candidate(candidate)

    def restore_checkpoint(self, artifact):
        return self.fixture.restore_serving_checkpoint(artifact)


def runtime_bindings(value):
    """Bind independent components from a test fixture or deployment result."""
    from reef.train.runtime_backend import RuntimeTrainingBackend

    if isinstance(value, RuntimeTrainingBackend):
        return {"runtime": value.inference_runtime, "training_runtime": value.training_runtime}
    if isinstance(value, tuple):
        return {"runtime": value[1], "training_runtime": value[0]}
    if isinstance(value, StubTrainingRuntime):
        return {"runtime": value.inference, "training_runtime": value}
    return {"runtime": value}


def training_backend(value, step_preparer, **kwargs):
    from reef.train.runtime_backend import RuntimeTrainingBackend

    bindings = runtime_bindings(value)
    return RuntimeTrainingBackend(
        bindings["training_runtime"],
        step_preparer,
        inference_runtime=bindings["runtime"],
        **kwargs,
    )


from reef.train.runtime_backend import RuntimeTrainingBackend


class ExecutorRuntimeFixture(RuntimeTrainingBackend):
    """Test fixture assembling two runtimes and the actual training backend."""

    def __init__(self, components=None, **kwargs):
        from reef.runtime.adapters.executor_runtime import connect_executor_runtimes

        training, inference = components if components is not None else connect_executor_runtimes(**kwargs)
        super().__init__(training, "sft", inference_runtime=inference)

    @property
    def inference(self):
        return self.inference_runtime

    @property
    def training(self):
        return self.training_runtime

    @property
    def train_group_handle(self):
        return self.training_runtime.train_group_handle

    @property
    def base_url(self):
        return self.inference_runtime.base_url

    @property
    def model_path(self):
        return self.inference_runtime.model_path

    @property
    def inference_timeout_s(self):
        return self.inference_runtime.inference_timeout_s

    @property
    def max_staleness(self):
        return self.training_runtime.max_staleness

    @property
    def concurrent_training_scenarios(self):
        return self.training_runtime.concurrent_training_scenarios

    @property
    def inference_backend(self):
        return self.inference_runtime.inference_backend

    @property
    def inference_admission_status(self):
        return self.inference_runtime.inference_admission_status

    def serving_runtime_load_id(self):
        return self.inference_runtime.serving_runtime_load_id()

    def current_runtime_load_id(self):
        return self.inference_runtime.current_runtime_load_id()

    def serving_adapter_name(self):
        return self.inference_runtime.serving_adapter_name()

    def serving_adapter_runtime_load_id(self, scenario):
        return self.inference_runtime.serving_adapter_runtime_load_id(scenario)

    async def acquire_inference(self):
        return await self.inference_runtime.acquire_inference()

    def reconcile_training_job(self, scenario_step, **kwargs):
        self._scenario = kwargs.pop("scenario", None)
        return self.recover_pending_step(scenario_step, **kwargs)

    def shutdown(self):
        self.inference_runtime.pause_admission()
        try:
            self.training_runtime.shutdown()
        finally:
            self.inference_runtime.shutdown()


def runtime_fixture(value):
    return ExecutorRuntimeFixture(value) if isinstance(value, tuple) else value
