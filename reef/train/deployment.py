"""Lightweight training deployment contracts, separate from model execution."""

from __future__ import annotations

import importlib
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.errors import DeployConfigError
from reef.runtime.deployment import CoordinatorConfig, DeploymentResources, TrainingService, runtime_factory_for
from reef.runtime.executor.arguments import native_arguments, normalize_native_options
from reef.runtime.executor.config import role_executor_settings, select_executor
from reef.runtime.interfaces import InferenceHandler

#: What the model driver writes once every component is ready; ``reef serve`` probes for it.
READY_PROBE = (
    "import os, pathlib, sys; "
    "p = pathlib.Path(os.environ['REEF_BRIDGE_READY_FILE']); "
    "sys.exit(0 if p.is_file() and p.read_text().strip() == 'reef-training-ready' else 1)"
)

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
# Colocation is Reef's placement decision (training.colocate); its native
# spellings would let the trainer and the reservation disagree.
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


def prepare_inference_config(
    config: dict[str, Any], settings: Mapping[str, Any], training_options: Mapping[str, Any]
) -> None:
    """Resolve managed inference capacity before downloads or Ray allocation.

    Public engine options use SGLang names without Slime's prefix. The first
    managed topology uses tensor parallel engines, with independent replicas
    when the total GPU budget exceeds the per-engine tensor parallel size.
    LoRA and colocated deployments retain their existing runtime lifecycle.
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
            raise DeployConfigError(f"training.options.{name} places the model; use training.colocate instead")
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
        colocate=bool(settings["colocate"]),
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


def driver_service(name: str, settings: Mapping[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """The model driver process definition ``reef serve`` starts before the HTTP service."""
    python = os.environ.get("REEF_PYTHON", sys.executable)
    return {
        "name": name,
        "executor": "uni",
        "command": [python, "-m", "reef.service.training_driver"],
        "ready": [python, "-c", READY_PROBE],
        "ready_timeout": settings["training_ready_timeout"],
        "env": {
            **env,
            "REEF_RAY_NAMESPACE": "${reef.ray_namespace}",
            "REEF_RAY_ACTOR_NAME": "${reef.ray_actor_name}",
        },
    }


@dataclass(frozen=True)
class TrainingDeploymentPlan:
    """Unallocated trainer and its launch requirements for Reef's assembler.

    Native inference options are data passed to the separately selected inference
    factory. The trainer never constructs or owns an inference service.
    """

    resources: DeploymentResources
    training: TrainingService
    inference_config: Mapping[str, Any] = field(default_factory=dict)
    coordinator: CoordinatorConfig | None = None
    monitor_components: bool = True


class TrainingDeployment(ABC):
    """Lightweight integration definition; preparation never allocates model resources."""

    requires_local_model: bool = True

    @abstractmethod
    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        """Validate integration inputs and describe dependencies of the HTTP process.

        Bind derived component values in config, using existing process definitions
        for any children. Return an empty tuple for an in-process runtime.
        """

    @abstractmethod
    def runtime_config(
        self, settings: Mapping[str, Any], *, max_staleness: int, connector: Any = None
    ) -> dict[str, Any]:
        """Describe the runtime for RuntimeRegistry, without constructing it."""

    def create_training_plan(self, config: Mapping[str, Any], *, loss_family: str) -> TrainingDeploymentPlan:
        """Provide an unstarted trainer for Reef's optional model driver.

        Integrations using their own process entrypoint or an in-process runtime
        need not implement this method. Reef selects the inference definition
        separately and validates the pair before allocating either component.
        """
        raise NotImplementedError("this backend does not use Reef's model deployment driver")


class InProcessTrainingDeployment(TrainingDeployment):
    """An integration whose registered runtime owns local inference and training.

    Subclasses name their runtime_type. The runtime factory owns option parsing
    and imports its execution dependencies only when constructing the runtime.
    """

    runtime_type: str

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        unsupported = {
            "ray_address",
            "ray_namespace",
            "ray_actor_name",
            "inference_url",
            "inference_backend",
            "inference_handler_factory",
            "inference_handler_config",
            "tensor_parallel_size",
            "inference_num_gpus",
            "inference_options",
            "colocate",
        } & config["reef"].keys()
        if unsupported or {"training", "rollout"} & config.get("execution", {}).keys():
            raise DeployConfigError(
                "in-process training owns its execution and inference; remove Ray, executor and inference engine settings"
            )
        factory = runtime_factory_for(self.runtime_type)
        if factory is None:
            raise DeployConfigError(f"unknown training runtime type {self.runtime_type!r}")
        # Use the runtime's schema for both CLI and YAML options, without a model.
        factory.parse_config(
            self.runtime_config(settings, max_staleness=config["reef"].get("max_staleness", 0)), os.environ
        )
        return ()

    def runtime_config(
        self, settings: Mapping[str, Any], *, max_staleness: int, connector: Any = None
    ) -> dict[str, Any]:
        if connector is not None:
            raise ValueError("in-process training does not accept a remote connector")
        options = {
            key.replace("-", "_"): value
            for key, value in normalize_native_options(settings["training_backend_options"]).items()
        }
        managed = {"type", "model_path", "inference_timeout_s", "train_timeout_s", "max_staleness", "connect"}
        if managed & options.keys():
            raise DeployConfigError("training.options cannot override managed runtime fields")
        return {
            **options,
            "type": self.runtime_type,
            "inference_timeout_s": settings["inference_timeout_s"],
            "train_timeout_s": settings["train_timeout_s"],
            "max_staleness": max_staleness,
        }
