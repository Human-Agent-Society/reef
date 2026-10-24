"""The HTTP service's inference runtimes for engines that run on an inference component.

Requests go to the component, which forwards them to its engines. The
control plane controls the component itself and serves published releases
(kind ``remote_serving``): each scenario's adapter
(``reef.runtime.serving.AdapterServingRuntime``) or one scenario's full
weights (``reef.runtime.serving.FullWeightServingRuntime``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.config import config_metadata, config_option
from reef.inference.http import ProviderRequestHeaders
from reef.inference.remote.config import RemoteInferenceSettings
from reef.inference.remote.service import RemoteInferenceService
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.inference.vllm.chat import VLLMInferenceHandler
from reef.runtime.deployment import RuntimeConfigError, RuntimeFactory
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime
from reef.runtime.serving import AdapterServingRuntime, FullWeightServingRuntime, SavedWeightServingRuntime

#: The request dialect of each built-in engine the component can run.
ENGINE_HANDLERS: dict[str, type[InferenceHandler]] = {
    "sglang": SGLangInferenceHandler,
    "vllm": VLLMInferenceHandler,
}


@dataclass(frozen=True)
class RemoteServingRuntimeConfig:
    """Options of the control plane's runtime for published weights served by an inference component.

    Values: ``layout`` is ``adapter`` (each scenario's PEFT adapter) or
    ``full`` (one scenario's full weights); ``remote`` is an
    ``inference.remote`` section that ``RemoteInferenceSettings.from_mapping``
    accepts; ``engine`` is the non-empty engine backend name the component
    starts, with its own input mapping ``engine_config``, on ``num_gpus`` GPUs;
    ``adapter_capacity`` bounds the resident adapters of the ``adapter``
    layout; ``num_gpus``, ``adapter_capacity`` and ``inference_timeout_s`` are
    positive; the mapping ``inference_handler_config`` holds the request
    handler's own options. ``RemoteServingRuntimeFactory`` reads them; the deployment of a
    training component writes them.
    """

    layout: str = config_option("adapter", help="adapter: one PEFT adapter per scenario; full: full weights.")
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
        if self.layout not in ("adapter", "full"):
            raise ValueError("runtime.layout must be adapter or full")
        RemoteInferenceSettings.from_mapping(self.remote)
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("runtime.engine must be a non-empty string")
        for name, options in (
            ("engine_config", self.engine_config),
            ("inference_handler_config", self.inference_handler_config),
        ):
            if not isinstance(options, Mapping):
                raise ValueError(f"runtime.{name} must be an object")
        for name, value in (("num_gpus", self.num_gpus), ("adapter_capacity", self.adapter_capacity)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"runtime.{name} must be a positive integer")
        if self.inference_timeout_s <= 0:
            raise ValueError("runtime.inference_timeout_s must be positive")


class RemoteServingRuntimeFactory(RuntimeFactory):
    """Start the component's engines and serve published releases on them, in the configured layout.

    The engine's own request handler speaks to the component, with the
    inference token as a bearer header; a ``module:factory`` engine needs an
    explicit ``inference_handler_factory``. The runtime stops the engines when
    it shuts down.
    """

    kind = "remote_serving"

    def config_type(self) -> type:
        return RemoteServingRuntimeConfig

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
            serving: SavedWeightServingRuntime
            if config["layout"] == "adapter":
                serving = AdapterServingRuntime(
                    backend,
                    handler,
                    adapter_capacity=config["adapter_capacity"],
                    service=service,
                    model_path=model_path,
                    inference_timeout_s=config["inference_timeout_s"],
                )
            else:
                serving = FullWeightServingRuntime(
                    backend,
                    handler,
                    service=service,
                    model_path=model_path,
                    inference_timeout_s=config["inference_timeout_s"],
                )
            return serving
        except BaseException:
            service.close()
            raise
