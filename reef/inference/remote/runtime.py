"""The HTTP service's inference runtimes for engines that run on an inference component.

Requests go to the component, which forwards them to its engines. The
control plane controls the component itself and serves published releases
(kind ``remote_serving``): each scenario's adapter
(``reef.runtime.serving.AdapterServingRuntime``) or one scenario's full
weights (``reef.runtime.serving.FullWeightServingRuntime``), as the training
component's ``ReleaseFormat`` says. The runtime builds the engines' input from
the deployment's ``inference`` settings and that format; no training backend
takes part.
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
from reef.runtime.jobs import ReleaseFormat
from reef.runtime.serving import AdapterServingRuntime, FullWeightServingRuntime, SavedWeightServingRuntime

#: The request dialect of each built-in engine the component can run.
ENGINE_HANDLERS: dict[str, type[InferenceHandler]] = {
    "sglang": SGLangInferenceHandler,
    "vllm": VLLMInferenceHandler,
}


def serving_engine_config(
    engine: str,
    *,
    model_path: str,
    num_gpus: int,
    tensor_parallel_size: int,
    options: Mapping[str, Any],
    release_format: ReleaseFormat,
    adapter_capacity: int,
    prefix_sharing: bool,
    executor: str,
    executor_options: Mapping[str, Any],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """The input of ``engine``'s inference service for serving engines on one inference component's host.

    The engines load ``model_path`` on ``num_gpus`` GPUs of one node, as
    tensor-parallel engines of ``tensor_parallel_size`` GPUs. ``options`` are
    the engine's native options; Reef adds the ones its serving needs, and
    SGLang's ``router-*`` options configure the router (``router-port`` binds
    the first one). An
    ``adapter`` release format enables the engine's adapter table, of
    ``adapter_capacity`` adapters. ``prefix_sharing`` retracts the requests
    in flight at each load, which lets the engines keep their prefix cache
    where that is safe (SGLang decides; see ``SGLangConfig``). ``vllm`` serves full
    weights only, with one engine. Any other engine, such as a
    ``module:factory``, gets SGLang's input. Raise ``ValueError`` for an
    input the engine cannot serve.
    """
    values = {key.replace("-", "_"): value for key, value in options.items()}
    placement = {
        "model_path": model_path,
        "num_gpus": num_gpus,
        "gpus_per_engine": tensor_parallel_size,
        "gpus_per_node": max(num_gpus, tensor_parallel_size),
        "pause_mode": "retract" if prefix_sharing else "in_place",
        "executor": executor,
        "executor_options": dict(executor_options),
    }
    if engine == "vllm":
        # The serving assembly already refused more than one vLLM engine (``prepare_engine_serving``).
        if release_format.layout != "full":
            raise ValueError("inference.backend: vllm serves full weights only, not PEFT adapter releases")
        values.setdefault("trust_remote_code", True)
        return {**placement, "options": values, "env_vars": {}}
    # Router options keep their native names after the router- prefix; the first router binds router-port.
    router_options = {
        name.removeprefix("router_"): values.pop(name) for name in list(values) if name.startswith("router_")
    }
    if "ip" in router_options:
        raise ValueError("inference.options.router-ip names an external router, which serving engines do not use")
    placement.update(router_port=router_options.pop("port", None), router_options=router_options)
    values.update(
        model_path=model_path,
        trust_remote_code=True,
        skip_server_warmup=True,
        enable_metrics=True,
        incremental_streaming_output=True,
    )
    if release_format.layout == "adapter":
        values.update(
            enable_lora=True,
            max_lora_rank=release_format.adapter_rank,
            max_loaded_loras=adapter_capacity,
            max_loras_per_batch=adapter_capacity,
            lora_target_modules=list(release_format.adapter_target_modules),
            tokenizer_worker_num=1,
        )
    # SGLang reads its own variables on the component's host.
    env_vars = {name: value for name, value in environ.items() if name.startswith("SGLANG_")}
    return {**placement, "options": values, "env_vars": env_vars}


@dataclass(frozen=True)
class RemoteServingRuntimeConfig:
    """Options of the control plane's runtime for published weights that engines on an inference component serve.

    Values: ``remote`` is an ``inference.remote`` section that
    ``RemoteInferenceSettings.from_mapping`` accepts. ``release_format`` is a
    ``ReleaseFormat`` as ``to_dict`` writes it. ``engine`` is the non-empty
    engine backend the component starts. ``num_gpus``,
    ``tensor_parallel_size``, ``adapter_capacity`` and ``inference_timeout_s``
    are positive, and ``num_gpus`` is a multiple of ``tensor_parallel_size``.
    ``options`` holds the engine's native options and
    ``inference_handler_config`` the request handler's own options.
    ``executor`` and ``executor_options`` select the engines' executor (the
    ``rollout`` role). Purpose: the serving assembly writes the value from the
    deployment's ``inference`` settings and the training component's release
    format; ``RemoteServingRuntimeFactory`` reads it.
    """

    remote: Mapping[str, Any] = field(metadata=config_metadata("The inference.remote section."))
    release_format: Mapping[str, Any] = field(
        metadata=config_metadata("The training component's release format, as ReleaseFormat.to_dict writes it.")
    )
    engine: str = config_option("sglang", help="Engine backend that the component starts.")
    num_gpus: int = config_option(1, help="GPUs the component's engines use.")
    tensor_parallel_size: int = config_option(1, help="GPUs of one tensor-parallel engine.")
    options: Mapping[str, Any] = field(default_factory=dict, metadata=config_metadata("Native engine options."))
    adapter_capacity: int = config_option(1, help="Adapters the engines keep resident.")
    prefix_sharing: bool = config_option(False, help="Share prefix-cache entries; each load retracts requests.")
    executor: str = config_option("auto", help="Executor backend of the serving engines.")
    executor_options: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Options of the engines' executor.")
    )
    inference_timeout_s: float = config_option(300.0, help="Inference request timeout in seconds.")
    inference_handler_config: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Inference adapter-owned options.")
    )

    def __post_init__(self) -> None:
        RemoteInferenceSettings.from_mapping(self.remote)
        ReleaseFormat.from_dict(self.release_format)
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("runtime.engine must be a non-empty string")
        for name, options in (
            ("options", self.options),
            ("executor_options", self.executor_options),
            ("inference_handler_config", self.inference_handler_config),
        ):
            if not isinstance(options, Mapping):
                raise ValueError(f"runtime.{name} must be an object")
        for name, value in (
            ("num_gpus", self.num_gpus),
            ("tensor_parallel_size", self.tensor_parallel_size),
            ("adapter_capacity", self.adapter_capacity),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"runtime.{name} must be a positive integer")
        if self.num_gpus % self.tensor_parallel_size:
            raise ValueError("runtime.num_gpus must be a multiple of runtime.tensor_parallel_size")
        if self.inference_timeout_s <= 0:
            raise ValueError("runtime.inference_timeout_s must be positive")


class RemoteServingRuntimeFactory(RuntimeFactory):
    """Start the component's engines and serve published releases on them, in the layout of their release format.

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
        release_format = ReleaseFormat.from_dict(config["release_format"])
        try:
            engine_config = serving_engine_config(
                engine,
                model_path=remote.model_path or model_path,
                num_gpus=config["num_gpus"],
                tensor_parallel_size=config["tensor_parallel_size"],
                options=config["options"],
                release_format=release_format,
                adapter_capacity=config["adapter_capacity"],
                prefix_sharing=config["prefix_sharing"],
                executor=config["executor"],
                executor_options=config["executor_options"],
                environ=environ,
            )
        except ValueError as exc:
            raise RuntimeConfigError(str(exc)) from exc
        service = RemoteInferenceService(
            remote,
            engine=engine,
            engine_config=engine_config,
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
            if release_format.layout == "adapter":
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
