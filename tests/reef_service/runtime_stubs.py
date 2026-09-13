"""Test doubles for Reef runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.evaluation import SelectionDecision
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.base import ModelRuntime, PreparedTrainingStep, TrainingRuntime
from reef.runtime.candidates import ActivatedModel, ModelCandidate
from reef.runtime.inference import InferenceBackend
from reef.train.types import TrainingBatch


class StubTrainingRuntime(ModelRuntime):
    """A ``ModelRuntime`` for in-process recipe smoke tests.

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
        super().__init__(inference=InferenceProxyRuntime(base_url=base_url), training=StubTrainingWorker())
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
    ) -> PreparedTrainingStep:
        raise NotImplementedError("StubTrainingRuntime does not prepare training steps")

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        raise NotImplementedError("StubTrainingRuntime does not train candidates")

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        raise NotImplementedError("StubTrainingRuntime does not activate candidates")

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        raise NotImplementedError("StubTrainingRuntime does not reject candidates")


class StubTrainingWorker(TrainingRuntime):
    """Training-only component for coordinated runtime test doubles."""

    def health(self):
        return {"ok": True}

    def prepare_training_step(self, batch, step_preparer, algorithm_state):
        raise NotImplementedError("test worker does not prepare training")

    def execute_training_job(self, payload):
        raise NotImplementedError("test worker does not execute training")
