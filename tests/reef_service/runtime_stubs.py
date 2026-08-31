"""Test doubles for Reef runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.base import PreparedTrainingStep, TrainingRuntime
from reef.runtime.candidates import ActivatedModel, ModelCandidate
from reef.runtime.inference import InferenceBackend
from reef.train.evaluation.contracts import SelectionDecision
from reef.train.types import TrainingBatch


class StubTrainingRuntime(TrainingRuntime):
    """A ``TrainingRuntime`` for in-process recipe smoke tests.

    It satisfies the constructor and typing contract so a training recipe can
    be built and its data path driven — records in, processor pairing, batch
    reservation, step preparation — with no backend anywhere. It composes no
    inference backend, and every training-path method raises
    ``NotImplementedError``: a test that reaches the real backend surface
    needs a real runtime, not this stub.
    """

    def __init__(self, base_url: str = "http://training-runtime", *, max_staleness: int = 0) -> None:
        if not isinstance(max_staleness, int) or isinstance(max_staleness, bool) or max_staleness < 0:
            raise ValueError("max_staleness must be a non-negative integer")
        super().__init__(base_url=base_url)
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
