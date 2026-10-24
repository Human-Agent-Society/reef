"""Lightweight training deployment contracts, separate from model execution."""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.errors import DeployConfigError
from reef.runtime.deployment import runtime_factory_for
from reef.runtime.executor.arguments import native_arguments, normalize_native_options
from reef.runtime.executor.config import role_executor_settings, select_executor
from reef.runtime.interfaces import InferenceHandler

#: The ``inference.backend`` that runs no engine: no weights leave the trainer and no request is served.
NO_INFERENCE_BACKEND = "none"

# These alter engine creation or topology outside Reef's inference request.
# Reject argparse abbreviations as well as full names, even for omitted values.
INFERENCE_LAUNCH_OPTIONS = frozenset(
    {
        "rollout-num-gpus",
        "rollout-num-gpus-per-engine",
        "rollout-external",
        "rollout-external-engine-addrs",
        "prefill-num-servers",
    }
)
# Training and serving never share GPUs; these native options would make them.
PLACEMENT_OPTIONS = frozenset({"colocate", "offload-rollout"})
#: Native SGLang options Reef binds itself for a managed engine.
INFERENCE_RESERVED_OPTIONS = frozenset(
    {
        "model",
        "model-path",
        "served-model-name",
        "config",
        "config-file",
        "yaml-config",
        "tp",
        "tp-size",
        "tensor-parallel-size",
        "dp",
        "dp-size",
        "data-parallel-size",
        "pp",
        "pp-size",
        "pipeline-parallel-size",
        "nnodes",
        "node-rank",
        "dist-init-addr",
        "port",
        "host",
        "base-gpu-id",
        "gpu-id-step",
        "nccl-port",
        "api-key",
        "random-seed",
        "trust-remote-code",
        "enable-memory-saver",
        "skip-server-warmup",
        "enable-return-routed-experts",
    }
)


def remote_engine_config(engine_config: Mapping[str, Any], model_path: str) -> dict[str, Any]:
    """``engine_config`` for engines on a remote component, which loads ``model_path``.

    ``model_path`` is ``inference.remote.model_path``. It replaces the model
    in the input and in its ``options``, which hold this host's path once
    deployment resolution has downloaded the model here.
    """
    config = dict(engine_config)
    config["model_path"] = model_path
    options = dict(config.get("options") or {})
    if "model_path" in options:
        options["model_path"] = model_path
    config["options"] = options
    return config


def prepare_inference_config(
    config: dict[str, Any], settings: Mapping[str, Any], training_options: Mapping[str, Any]
) -> None:
    """Resolve managed inference capacity before downloads or Ray allocation.

    Public engine options use SGLang names without Slime's prefix. The first
    managed topology uses tensor parallel engines, with independent replicas
    when the total GPU budget exceeds the per-engine tensor parallel size.
    ``inference.backend: none`` takes no inference GPUs.
    """
    for name in training_options:
        if (
            name.startswith(("sglang-", "router-"))
            or "sglang".startswith(name)
            or "router".startswith(name)
            or any(flag.startswith(name) for flag in INFERENCE_LAUNCH_OPTIONS)
        ):
            raise DeployConfigError(
                f"training.options.{name} configures inference; use inference.num-gpus, "
                "inference.tensor-parallel-size or inference.options instead"
            )
        if any(flag.startswith(name) for flag in PLACEMENT_OPTIONS):
            raise DeployConfigError(f"training.options.{name} shares GPUs between training and serving; remove it")
    if settings["inference_backend"] == NO_INFERENCE_BACKEND:
        config["reef"].update(inference_num_gpus=0, inference_options={})
        return
    parallel_size = settings["tensor_parallel_size"]
    parallel_size = 1 if parallel_size is None else parallel_size
    num_gpus = settings["inference_num_gpus"]
    num_gpus = parallel_size if num_gpus is None else num_gpus
    if parallel_size <= 0 or num_gpus <= 0 or num_gpus % parallel_size:
        raise DeployConfigError(
            "inference.num-gpus and inference.tensor-parallel-size must be positive; "
            "num-gpus must be divisible by tensor-parallel-size"
        )
    options = normalize_native_options(settings["inference_options"])
    native_arguments(options, reserved=set(INFERENCE_RESERVED_OPTIONS))
    if any(name.startswith("sglang-") for name in options):
        raise DeployConfigError("inference.options uses native SGLang names without the sglang- prefix")
    config["reef"].update(
        # A selected dotted factory keeps its name; it receives the same SGLang-shaped input.
        inference_backend=settings["inference_backend"] or "sglang",
        inference_num_gpus=num_gpus,
        tensor_parallel_size=parallel_size,
        inference_options=options,
    )


def inference_handler_factory_for(path: str | None) -> type[InferenceHandler] | None:
    """Load an optional request handler factory selected by deployment config."""
    if path is None:
        return None
    if not isinstance(path, str) or not path.strip():
        raise ValueError("inference.handler-factory must be a non-empty dotted path")
    module_path, separator, attribute = path.strip().rpartition(".")
    if not separator or not module_path or not attribute:
        raise ValueError("inference.handler-factory must be a dotted path")
    try:
        factory = getattr(importlib.import_module(module_path), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load inference.handler-factory {path!r}") from exc
    if not isinstance(factory, type) or not issubclass(factory, InferenceHandler):
        raise ValueError(f"inference.handler-factory {path!r} must inherit InferenceHandler")
    return factory


def require_ray_roles(config: dict[str, Any], *roles: str, backend: str) -> None:
    """Default the named execution roles to Ray and refuse any other executor for them."""
    execution = config.setdefault("execution", {})
    for role in roles:
        execution.setdefault(role, "ray")
        if select_executor(role_executor_settings(config, role), role=role).settings.backend != "ray":
            raise DeployConfigError(f"automatic {backend} training requires execution.{role}.backend: ray")


class TrainingDeployment(ABC):
    """Lightweight integration definition; preparation never allocates model resources."""

    requires_local_model: bool = True
    supports_teacher_engine: bool = False
    #: Whether its serving engines can run on a remote component (an ``inference.remote`` section).
    supports_remote_inference: bool = False

    @abstractmethod
    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        """Validate integration inputs and describe dependencies of the HTTP process.

        Bind derived component values in config, using existing process definitions
        for any children. Return an empty tuple for an in-process runtime.
        """

    @abstractmethod
    def runtime_config(self, settings: Mapping[str, Any], *, loss_family: str) -> dict[str, Any]:
        """Describe the training component and inference runtime for RuntimeRegistry, without constructing them.

        ``loss_family`` is the recipe's startup loss family, which a backend
        that starts its workers with one loss family reads.
        """

    def http_environment(self, settings: Mapping[str, Any]) -> Mapping[str, str]:
        """Variables the HTTP service process gets, where the training component runs.

        Values may name ``${endpoints.<service>}`` of a service that
        :meth:`prepare` returned; ``reef serve`` resolves them before it starts
        the HTTP service.
        """
        return {}

    def serves_on_inference_component(self, settings: Mapping[str, Any]) -> bool:
        """Whether the control plane serves saved weights on an inference component for these settings.

        The component is the one ``inference.remote`` names. Without that
        section, ``reef serve`` starts one on this host and records its address
        there before :meth:`prepare` runs.
        """
        return False


class InProcessTrainingDeployment(TrainingDeployment):
    """An integration whose registered runtime factory builds its training component and inference runtime in process.

    Subclasses name their runtime_type. The runtime factory owns option parsing
    and imports its execution dependencies only when constructing the runtime.
    """

    runtime_type: str

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        unsupported = {
            "ray_address",
            "ray_namespace",
            "inference_url",
            "inference_backend",
            "inference_handler_factory",
            "inference_handler_config",
            "tensor_parallel_size",
            "inference_num_gpus",
            "inference_options",
        } & config["reef"].keys()
        if unsupported or {"training", "rollout"} & config.get("execution", {}).keys():
            raise DeployConfigError(
                "in-process training owns its execution and inference; remove Ray, executor and inference engine settings"
            )
        factory = runtime_factory_for(self.runtime_type)
        if factory is None:
            raise DeployConfigError(f"unknown training runtime type {self.runtime_type!r}")
        # Use the runtime's schema for both CLI and YAML options, without a model.
        factory.parse_config(self.runtime_config(settings, loss_family=""), os.environ)
        return ()

    def runtime_config(self, settings: Mapping[str, Any], *, loss_family: str) -> dict[str, Any]:
        options = {
            key.replace("-", "_"): value
            for key, value in normalize_native_options(settings["training_backend_options"]).items()
        }
        managed = {"type", "model_path", "inference_timeout_s", "train_timeout_s", "connect"}
        if managed & options.keys():
            raise DeployConfigError("training.options cannot override managed runtime fields")
        return {
            **options,
            "type": self.runtime_type,
            "inference_timeout_s": settings["inference_timeout_s"],
            "train_timeout_s": settings["train_timeout_s"],
        }
