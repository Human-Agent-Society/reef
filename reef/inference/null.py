"""The inference runtime of a deployment that runs no engine: ``inference.backend: none``.

A deployment selects it to train on data that does not come from its own model,
such as uploaded trajectories (``POST /reef/trajectories``) or imported records
(``POST /reef/records``). It takes no GPUs and serves no requests: every
inference request fails with :class:`~reef.core.errors.InferenceUnavailable`,
which the HTTP service returns as 503. Reef still publishes and commits every
saved checkpoint as a release, which another deployment can serve.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailable
from reef.runtime.deployment import RuntimeFactory
from reef.runtime.interfaces import InferenceAdmissionHandle, InferenceHandler, InferenceRuntime

#: The response body of every inference request.
UNAVAILABLE_MESSAGE = (
    "this deployment runs no inference engine (inference.backend: none) and serves no inference requests; "
    "serve a published checkpoint from a separate deployment"
)
#: The base URL of the runtime. It names no endpoint: every request fails before it would reach one.
NULL_INFERENCE_URL = "none://no-inference-engine"


class NullInferenceHandler(InferenceHandler):
    """A request handler that refuses every request with :class:`InferenceUnavailable`."""

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise InferenceUnavailable(UNAVAILABLE_MESSAGE)


class NullInferenceRuntime(InferenceRuntime):
    """Serve nothing: admission refuses every request, and activating a release loads nothing."""

    def __init__(self, *, model_path: str = "", inference_timeout_s: float = 300.0) -> None:
        super().__init__(base_url=NULL_INFERENCE_URL, inference_timeout_s=inference_timeout_s)
        self._handler = NullInferenceHandler()
        self._model_path = model_path

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def shared_by_training_scenarios(self) -> bool:
        # Nothing is served, so no scenario's release replaces another's.
        return True

    async def acquire_inference(self) -> InferenceAdmissionHandle:
        raise InferenceUnavailable(UNAVAILABLE_MESSAGE)

    def current_runtime_load_id(self) -> str | None:
        # The runtime loads nothing, so no load names what serves.
        return None

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        return artifact.ref.release_id


class NullRuntimeFactory(RuntimeFactory):
    """Build the runtime of a deployment with ``inference.backend: none``."""

    kind = "none"

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        return NullInferenceRuntime(
            model_path=model_path, inference_timeout_s=config.get("inference_timeout_s", 300.0)
        )
