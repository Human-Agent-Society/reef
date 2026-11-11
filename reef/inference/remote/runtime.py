"""The HTTP service's inference runtimes for engines that run on an inference component.

Requests go to the component, which forwards them to its engines. The
control plane controls the component itself and serves published snapshots
(kind ``remote_serving``): each scenario's adapter
(``reef.inference.serving.AdapterServingRuntime``) or one scenario's full
weights (``reef.inference.serving.FullWeightServingRuntime``), as the training
component's ``ArtifactFormat`` says. With further components
(``inference.remote.slot-urls``), one scenario serves a full-weight checkpoint
on each of them (``reef.inference.checkpoint_slots.CheckpointSlotsRuntime``).
The runtime builds the engines' input from the deployment's ``inference``
settings and that format; no training backend takes part.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from reef.core.config import config_metadata, config_option
from reef.core.errors import RuntimeConfigError
from reef.core.weight_layout import ArtifactFormat, ArtifactWeights
from reef.inference.checkpoint_slots import CheckpointSlotsRuntime
from reef.inference.deployment import BUILTIN_ENGINES, InferenceConnection
from reef.inference.http import ProviderRequestHeaders
from reef.inference.interfaces import InferenceHandler, InferenceRuntime
from reef.inference.registry import RuntimeFactory
from reef.inference.remote.config import RemoteInferenceSettings, require_no_external_kv_cache
from reef.inference.remote.service import RemoteInferenceService
from reef.inference.serving import AdapterServingRuntime, FullWeightServingRuntime, OpenedEngines, ServingEngines


def serving_engine_config(
    engine: str,
    *,
    model_path: str,
    num_gpus: int,
    tensor_parallel_size: int,
    options: Mapping[str, Any],
    artifact_format: ArtifactFormat,
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
    ``adapter_capacity`` adapters. ``prefix_sharing`` selects the retracting
    pause, which clears the prefix cache before each load, so the engines
    keep their prefix cache where that is safe (the engine's settings decide;
    see ``SGLangSettings`` and ``VLLMSettings``). A component that keeps each
    request on its weights drains its requests before that pause, so the
    pause retracts none. ``vllm`` serves full
    weights only, with one engine. Any other engine, such as a
    ``module:ClassName``, gets SGLang's input. Raise ``ValueError`` for an
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
        if artifact_format.layout != "full":
            raise ValueError("inference.backend: vllm serves full weights only, not PEFT adapter artifacts")
        values.setdefault("trust_remote_code", True)
        return {**placement, "options": values, "env_vars": {}}
    # Router options keep their native names after the router- prefix; the first router binds router-port.
    router_options = {
        name.removeprefix("router_"): values.pop(name) for name in list(values) if name.startswith("router_")
    }
    placement.update(router_port=router_options.pop("port", None), router_options=router_options)
    values.update(
        model_path=model_path,
        trust_remote_code=True,
        skip_server_warmup=True,
        enable_metrics=True,
        incremental_streaming_output=True,
    )
    if artifact_format.layout == "adapter":
        values.update(
            enable_lora=True,
            max_lora_rank=artifact_format.adapter_rank,
            max_loaded_loras=adapter_capacity,
            max_loras_per_batch=adapter_capacity,
            lora_target_modules=list(artifact_format.adapter_target_modules),
            tokenizer_worker_num=1,
        )
    # SGLang reads its own variables on the component's host.
    env_vars = {name: value for name, value in environ.items() if name.startswith("SGLANG_")}
    return {**placement, "options": values, "env_vars": env_vars}


@dataclass(frozen=True)
class RemoteServingRuntimeSettings:
    """Options of the inference service's runtime for published weights that engines on an inference component serve.

    Values: ``remote`` is an ``inference.remote`` section that
    ``RemoteInferenceSettings.from_mapping`` accepts. ``artifact_format`` is a
    ``ArtifactFormat`` as ``to_dict`` writes it. ``engine`` is the non-empty
    engine backend the component starts. ``num_gpus``,
    ``tensor_parallel_size``, ``adapter_capacity`` and ``inference_timeout_s``
    are positive, and ``num_gpus`` is a multiple of ``tensor_parallel_size``.
    Every value but ``remote`` applies to each component that ``remote`` names, so each one uses ``num_gpus`` GPUs.
    ``options`` holds the engine's native options and
    ``inference_handler_config`` the request handler's own options.
    ``executor`` and ``executor_options`` select the engines' executor (the
    ``rollout`` role). ``weights`` is ``None``, which reads each snapshot's own
    files (``ArtifactWeightFiles``), or a ``ArtifactWeights`` plugin reference.
    ``prefix_sharing`` is a bool. When true, requests on the same weights share their prefix cache: the component
    drains its requests, then the engines clear the cache, before each change of weights. The engine's config
    refuses a cache that it cannot keep to one set of weights, and the factory refuses ``options`` that keep KV
    outside the engines. The component keeps each request on its weights, with no setting to turn that off.
    Purpose: the serving assembly writes the value from the deployment's
    ``inference`` settings and the training backend's engine weights;
    ``RemoteServingRuntimeFactory`` reads it.
    """

    remote: Mapping[str, Any] = field(metadata=config_metadata("The inference.remote section."))
    artifact_format: Mapping[str, Any] = field(
        metadata=config_metadata("The format of the weights the engines load, as ArtifactFormat.to_dict writes it.")
    )
    engine: str = config_option("sglang", help="Engine backend that the component starts.")
    num_gpus: int = config_option(1, help="GPUs the engines of each component use.")
    tensor_parallel_size: int = config_option(1, help="GPUs of one tensor-parallel engine.")
    options: Mapping[str, Any] = field(default_factory=dict, metadata=config_metadata("Native engine options."))
    adapter_capacity: int = config_option(1, help="Adapters the engines keep resident.")
    prefix_sharing: bool = config_option(
        False, help="Share the prefix cache among requests on the same weights; each change of weights clears it."
    )
    executor: str = config_option("auto", help="Executor backend of the serving engines.")
    executor_options: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Options of the engines' executor.")
    )
    inference_timeout_s: float = config_option(300.0, help="Inference request timeout in seconds.")
    inference_handler_config: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Inference adapter-owned options.")
    )
    weights: str | Mapping[str, Any] | None = config_option(
        None, help="A ArtifactWeights plugin that gives each snapshot's weight files; unset reads the snapshot's own."
    )

    def __post_init__(self) -> None:
        RemoteInferenceSettings.from_mapping(self.remote)
        ArtifactFormat.from_dict(self.artifact_format)
        if not isinstance(self.prefix_sharing, bool):
            raise ValueError("runtime.prefix_sharing must be a bool")
        if self.weights is not None:
            ArtifactWeights.reference(self.weights, setting="runtime.weights")
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


class ComponentEngines(ServingEngines):
    """The engines of a remote inference component, which outlive control.

    ``open`` starts a new engine stack, which replaces the engines that the component runs. A new serving owner,
    such as a restarted inference worker or one that takes over, therefore ends the requests in flight: the
    engines' load IDs cannot prove which weights they hold, and a late operation of the previous owner could still
    change them. ``close`` stops no engine: the component's engines are the inference service's, and stop with the
    component.

    ``request_pinning`` is the service's own. A service that requests the protection requires the component to
    confirm it at every engine start and health check, so the declaration holds only for a component that
    enforces it.
    """

    def __init__(self, service: RemoteInferenceService) -> None:
        self.service = service

    @property
    def base_url(self) -> str:
        return self.service.settings.url

    @property
    def request_pinning(self) -> bool:
        return self.service.request_pinning

    def set_owner_sequence(self, sequence: int) -> None:
        self.service.set_owner_sequence(sequence)

    def open(self) -> OpenedEngines:
        if self.service.started and not self.service.closed:
            # A prior open may have failed after startup but before initialization or pause was acknowledged.
            self.service.restart_engines()
            connection = InferenceConnection(None)
        else:
            # Adoption cannot prove adapter residency or exclude a late operation from the previous owner.
            connection = self.service.start()
        return OpenedEngines(self.service.backend(connection), attached=False)

    def check_health(self) -> None:
        self.service.check_health()


class RemoteServingRuntimeFactory(RuntimeFactory):
    """Start the component's engines and serve published snapshots on them, in the layout of their snapshot format.

    The engine's own request handler speaks to the component, with the
    inference token as a bearer header. The engines outlive the runtime
    (``ComponentEngines``).

    With ``slot_urls`` in the ``inference.remote`` section, each URL names its own component. The factory then
    builds a ``FullWeightServingRuntime`` for each one, with its own service and request handler and the same
    engine input and tokens, and returns a ``CheckpointSlotsRuntime`` over them, which serves one checkpoint on
    each component. That needs full-weight artifacts: ``slot_urls`` with an adapter format raises
    ``RuntimeConfigError``. Without ``slot_urls`` the factory returns the one runtime of the component at ``url``.

    The factory always requests the component's request protection (``ServingEngines.request_pinning``), and the
    component must confirm it. Only the built-in engines with their built-in request handlers qualify: the
    component knows their generation routes, and their handlers name the selected weights on every request. The
    factory raises ``RuntimeConfigError`` for any other engine or handler, so published weights never serve
    without the protection. It raises the same error for an engine option that keeps KV outside the engines
    (``require_no_external_kv_cache``), because no pause clears that KV before a change of weights.
    """

    kind = "remote_serving"

    def config_type(self) -> type:
        return RemoteServingRuntimeSettings

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
        handler_factory = (
            InferenceHandler.resolve_class(BUILTIN_ENGINES[engine].handler, setting="inference handler")
            if engine in BUILTIN_ENGINES
            else None
        )
        if handler_factory is None or config.get("inference_handler_factory") not in (None, handler_factory):
            raise RuntimeConfigError(
                f"remote engine {engine!r} cannot serve published weights: the inference component keeps each "
                "request on its weights only for the built-in sglang and vllm engines with their built-in "
                "request handlers"
            )
        try:
            control_token = remote.token(remote.control_token_env, environ)
            inference_token = remote.token(remote.inference_token_env, environ)
        except ValueError as exc:
            raise RuntimeConfigError(str(exc)) from exc
        artifact_format = ArtifactFormat.from_dict(config["artifact_format"])
        if remote.slot_urls and artifact_format.layout != "full":
            raise RuntimeConfigError(
                "inference.remote.slot-urls adds components that each serve one full-weight checkpoint; engines "
                "that load adapters serve every scenario on the component at inference.remote.url, so remove "
                "slot-urls"
            )
        weights = (
            None
            if config.get("weights") is None
            else ArtifactWeights.from_setting(config["weights"], setting="runtime.weights")
        )
        try:
            engine_config = serving_engine_config(
                engine,
                model_path=remote.model_path or model_path,
                num_gpus=config["num_gpus"],
                tensor_parallel_size=config["tensor_parallel_size"],
                options=config["options"],
                artifact_format=artifact_format,
                adapter_capacity=config["adapter_capacity"],
                prefix_sharing=config["prefix_sharing"],
                executor=config["executor"],
                executor_options=config["executor_options"],
                environ=environ,
            )
            require_no_external_kv_cache(engine, engine_config)
        except ValueError as exc:
            raise RuntimeConfigError(str(exc)) from exc
        # Building the runtime touches no engine: only the inference owner opens it, in every layout.
        components: list[tuple[ComponentEngines, InferenceHandler]] = []
        for url in (remote.url, *remote.slot_urls):
            service = RemoteInferenceService(
                # The settings of one component name no other component.
                replace(remote, url=url, slot_urls=()),
                engine=engine,
                engine_config=engine_config,
                num_gpus=config["num_gpus"],
                control_token=control_token,
                request_pinning=True,
                weight_layout=artifact_format.layout,
                drain_timeout_s=config["inference_timeout_s"],
            )
            handler = handler_factory.from_config(
                url,
                model_path=model_path,
                timeout_s=config["inference_timeout_s"],
                **dict(config["inference_handler_config"]),
                request_headers=ProviderRequestHeaders(inference_token),
            )
            components.append((ComponentEngines(service), handler))
        if artifact_format.layout == "adapter":
            engines, handler = components[0]
            return AdapterServingRuntime(
                engines,
                handler,
                adapter_capacity=config["adapter_capacity"],
                model_path=model_path,
                inference_timeout_s=config["inference_timeout_s"],
                weights=weights,
            )
        slots = [
            FullWeightServingRuntime(
                engines,
                handler,
                model_path=model_path,
                inference_timeout_s=config["inference_timeout_s"],
                weights=weights,
            )
            for engines, handler in components
        ]
        return slots[0] if len(slots) == 1 else CheckpointSlotsRuntime(slots)
