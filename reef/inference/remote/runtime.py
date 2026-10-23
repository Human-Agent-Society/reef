"""The HTTP service's inference runtime for engines that run on a remote component.

Requests go to the component, which forwards them to its engines; Reef's
coordinator still controls admission and publication, as with local engines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.config import config_option
from reef.inference.http import ProviderRequestHeaders
from reef.inference.remote.config import DEFAULT_INFERENCE_TOKEN_ENV
from reef.inference.runtime import EngineRuntimeFactory, ExecutorInferenceRuntime
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.inference.vllm.chat import VLLMInferenceHandler
from reef.runtime.deployment import RuntimeConfigError, RuntimeConnectionConfig
from reef.runtime.executor.connection import CoordinatorClient
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime

#: The request dialect of each built-in engine the component can run.
ENGINE_HANDLERS: dict[str, type[InferenceHandler]] = {
    "sglang": SGLangInferenceHandler,
    "vllm": VLLMInferenceHandler,
}


@dataclass(frozen=True)
class RemoteRuntimeConfig(RuntimeConnectionConfig):
    """Request options of the HTTP service for engines on a remote component.

    Values: the ``RuntimeConnectionConfig`` values, where also ``engine`` is a
    non-empty backend name that the component runs (``sglang``, ``vllm`` or a
    ``module:factory`` reference), and ``inference_token_env`` is a non-empty
    environment variable name. ``RemoteRuntimeFactory`` reads them: ``engine``
    selects the request dialect, unless ``inference_handler_factory`` names one,
    and the variable holds the bearer token that every request carries.
    """

    engine: str = config_option("sglang", help="Engine backend that the remote component runs.")
    inference_token_env: str = config_option(
        DEFAULT_INFERENCE_TOKEN_ENV, help="Environment variable with the component's inference token."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (("engine", self.engine), ("inference_token_env", self.inference_token_env)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"runtime.{name} must be a non-empty string")


class RemoteRuntimeFactory(EngineRuntimeFactory):
    """Connect the HTTP service to a remote component through the coordinator's control client.

    The engine's own request handler speaks to the component, with the
    inference token as a bearer header. A ``module:factory`` engine needs an
    explicit ``inference_handler_factory``.
    """

    kind = "remote"
    engine_name = "remote"
    runtime_type = ExecutorInferenceRuntime

    def config_type(self) -> type:
        return RemoteRuntimeConfig

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        control = config.get("control")
        if not isinstance(control, CoordinatorClient):
            raise RuntimeConfigError("the remote runtime requires a Reef coordinator client")
        engine = config.get("engine", "sglang")
        handler_factory = config.get("inference_handler_factory") or ENGINE_HANDLERS.get(engine)
        if handler_factory is None:
            raise RuntimeConfigError(f"remote engine {engine!r} needs an explicit inference handler factory")
        variable = config.get("inference_token_env", DEFAULT_INFERENCE_TOKEN_ENV)
        token = environ.get(variable, "").strip()
        if not token:
            raise RuntimeConfigError(f"the remote inference runtime requires the environment variable {variable}")
        return ExecutorInferenceRuntime(
            control=control,
            model_path=model_path,
            inference_url=config.get("inference_url"),
            inference_timeout_s=config.get("inference_timeout_s", 300.0),
            inference_handler_factory=handler_factory,
            inference_handler_config={
                **dict(config.get("inference_handler_config") or {}),
                "request_headers": ProviderRequestHeaders(token),
            },
        )
