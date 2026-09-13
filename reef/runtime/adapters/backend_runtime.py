"""Inference runtime for an independently selected request backend."""

from reef.runtime.base import InferenceRuntime
from reef.runtime.inference import InferenceBackend


class BackendInferenceRuntime(InferenceRuntime):
    """Compose native request execution with inference admission and endpoint state."""

    def __init__(self, *, backend: InferenceBackend, base_url: str, inference_timeout_s: float = 300.0) -> None:
        super().__init__(base_url=base_url, inference_timeout_s=inference_timeout_s)
        self._backend = backend

    @property
    def inference_backend(self) -> InferenceBackend:
        return self._backend
