"""The HTTP service's inference runtimes for engines that run on an inference component.

Requests go to the component, which forwards them to its engines. With
``remote``, Reef's coordinator controls admission and publication, as with
local engines. With ``remote_adapters``, the control plane controls the
component itself and serves each scenario's saved adapter
(``reef.runtime.serving.AdapterServingRuntime``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.config import config_metadata, config_option
from reef.inference.http import ProviderRequestHeaders
from reef.inference.remote.config import DEFAULT_INFERENCE_TOKEN_ENV, RemoteInferenceSettings
from reef.inference.remote.service import RemoteInferenceService
from reef.inference.runtime import EngineRuntimeFactory, ExecutorInferenceRuntime
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.inference.vllm.chat import VLLMInferenceHandler
from reef.runtime.deployment import RuntimeConfigError, RuntimeConnectionConfig, RuntimeFactory
from reef.runtime.executor.connection import CoordinatorClient
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime
from reef.runtime.serving import AdapterServingRuntime

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


@dataclass(frozen=True)
class RemoteAdapterRuntimeConfig:
    """Options of the control plane's runtime for per-scenario adapters served by an inference component.

    Values: ``remote`` is an ``inference.remote`` section that
    ``RemoteInferenceSettings.from_mapping`` accepts; ``engine`` is the
    non-empty engine backend name the component starts, with its own input
    ``engine_config``, on ``num_gpus`` GPUs; ``adapter_capacity`` bounds the
    resident adapters; ``num_gpus``, ``adapter_capacity`` and
    ``inference_timeout_s`` are positive; ``inference_handler_config`` holds
    the request handler's own options. ``RemoteAdapterRuntimeFactory`` reads
    them; the deployment of a training component writes them.
    """

    remote: Mapping[str, Any] = field(default_factory=dict, metadata=config_metadata("The inference.remote section."))
    engine: str = config_option("sglang", help="Engine backend that the component starts.")
    engine_config: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("The engine backend's own input.")
    )
    num_gpus: int = config_option(1, help="GPUs the component's engines use.")
    adapter_capacity: int = config_option(1, help="Adapters the engines keep resident.")
    inference_timeout_s: float = config_option(300.0, help="Inference request timeout in seconds.")
    inference_handler_config: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Inference adapter-owned options.")
    )

    def __post_init__(self) -> None:
        RemoteInferenceSettings.from_mapping(self.remote)
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("runtime.engine must be a non-empty string")
        for name, value in (("num_gpus", self.num_gpus), ("adapter_capacity", self.adapter_capacity)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"runtime.{name} must be a positive integer")
        if self.inference_timeout_s <= 0:
            raise ValueError("runtime.inference_timeout_s must be positive")


class RemoteAdapterRuntimeFactory(RuntimeFactory):
    """Start the component's engines and serve saved per-scenario adapters on them.

    The engine's own request handler speaks to the component, with the
    inference token as a bearer header; a ``module:factory`` engine needs an
    explicit ``inference_handler_factory``. The runtime stops the engines when
    it shuts down.
    """

    kind = "remote_adapters"

    def config_type(self) -> type:
        return RemoteAdapterRuntimeConfig

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        injected = {key: config[key] for key in ("inference_handler_factory",) if key in config}
        parsed = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        return {**parsed, **injected}

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        remote = RemoteInferenceSettings.from_mapping(config["remote"])
        engine = config["engine"]
        handler_factory = config.get("inference_handler_factory") or ENGINE_HANDLERS.get(engine)
        if handler_factory is None:
            raise RuntimeConfigError(f"remote engine {engine!r} needs an explicit inference handler factory")
        try:
            control_token = remote.token(remote.control_token_env, environ)
            inference_token = remote.token(remote.inference_token_env, environ)
        except ValueError as exc:
            raise RuntimeConfigError(str(exc)) from exc
        service = RemoteInferenceService(
            remote,
            engine=engine,
            engine_config=config["engine_config"],
            num_gpus=config["num_gpus"],
            control_token=control_token,
        )
        try:
            backend = service.backend(service.start())
            handler = handler_factory.from_config(
                remote.url,
                model_path=model_path,
                timeout_s=config["inference_timeout_s"],
                **dict(config["inference_handler_config"]),
                request_headers=ProviderRequestHeaders(inference_token),
            )
            return AdapterServingRuntime(
                backend,
                handler,
                adapter_capacity=config["adapter_capacity"],
                service=service,
                model_path=model_path,
                inference_timeout_s=config["inference_timeout_s"],
            )
        except BaseException:
            service.close()
            raise
