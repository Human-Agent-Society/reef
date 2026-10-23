"""The inference runtime of a training-only deployment: it has no engines and serves no requests.

A Slime deployment with ``inference.num-gpus: 0`` trains on every model GPU and
starts no inference engines. Reef still commits each training step through the
coordinator's publication protocol, so this runtime keeps the coordinator's
serving version and its commit handshake. Every inference request fails with
:class:`~reef.core.errors.InferenceUnavailable`, which the HTTP service returns
as 503.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailable
from reef.inference.runtime import EngineRuntimeFactory, ExecutorInferenceRuntime
from reef.runtime.deployment import RuntimeConfigError
from reef.runtime.executor.connection import CoordinatorClient
from reef.runtime.interfaces import InferenceAdmissionHandle, InferenceHandler

#: Why a request fails; the HTTP response body.
UNAVAILABLE_MESSAGE = (
    "this deployment trains without inference engines (inference.num-gpus: 0) and serves no inference requests; "
    "serve a published checkpoint from a separate deployment"
)
#: The runtime's base URL. It names no endpoint: every request fails before it reaches one.
TRAIN_ONLY_URL = "train-only:no-engines"


class UnavailableInferenceHandler(InferenceHandler):
    """A request handler that refuses every request with :class:`InferenceUnavailable`."""

    @classmethod
    def from_config(
        cls,
        upstream_url: str,
        *,
        model_path: str,
        timeout_s: float,
        **config: Any,
    ) -> InferenceHandler:
        return cls()

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise InferenceUnavailable(UNAVAILABLE_MESSAGE)


class TrainOnlyInferenceRuntime(ExecutorInferenceRuntime):
    """Follow the coordinator's serving version and commit handshake without engines.

    Weight activation and commit acknowledgement go to the coordinator as they
    do with engines; the coordinator assigns each trained version a runtime
    load ID without a transfer. Admission refuses every request at once.
    """

    def __init__(
        self,
        *,
        control: CoordinatorClient,
        inference_url: str | None = None,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        inference_handler_factory: type[InferenceHandler] | None = None,
        inference_handler_config: Mapping[str, Any] | None = None,
    ) -> None:
        if inference_url or inference_handler_factory is not None or inference_handler_config:
            raise RuntimeConfigError(
                "a training-only deployment has no inference engines; remove inference.url, "
                "inference.handler-factory and inference.handler-config"
            )
        super().__init__(
            control=control,
            inference_url=TRAIN_ONLY_URL,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            inference_handler_factory=UnavailableInferenceHandler,
        )

    async def acquire_inference(self) -> InferenceAdmissionHandle:
        raise InferenceUnavailable(UNAVAILABLE_MESSAGE)


class TrainOnlyRuntimeFactory(EngineRuntimeFactory):
    """Connect a training-only deployment's coordinator without an inference receiver."""

    kind = "train_only"
    engine_name = "training-only"
    runtime_type = TrainOnlyInferenceRuntime
