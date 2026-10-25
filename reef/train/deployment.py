"""Lightweight training deployment contracts, separate from model execution."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.errors import DeployConfigError
from reef.runtime.deployment import runtime_factory_for
from reef.runtime.executor.arguments import normalize_native_options
from reef.runtime.executor.config import role_executor_settings, select_executor

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


def refuse_serving_options(training_options: Mapping[str, Any]) -> None:
    """Refuse native training options that would configure serving engines or share GPUs with them.

    Serving engines are configured by the ``inference`` section alone, and
    training and serving never share GPUs. Raise ``DeployConfigError`` for an
    option (or an abbreviation of one) that does either.
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
    #: The serving of a deployment that names none and sets no engine setting: ``engines`` or ``trainer``.
    default_serving: str = "engines"

    @abstractmethod
    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        """Validate integration inputs and describe dependencies of the HTTP process.

        Bind derived component values in config, using existing process definitions
        for any children. Return an empty tuple for an in-process runtime.
        """

    @abstractmethod
    def runtime_config(self, settings: Mapping[str, Any], *, loss_family: str) -> dict[str, Any]:
        """Describe the training component for RuntimeRegistry, without constructing it.

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

    def trainer_serving_config(self, settings: Mapping[str, Any]) -> dict[str, Any] | None:
        """The runtime config of the backend's own serving (``inference.serving: trainer``), or None without one."""
        return None


class InProcessTrainingDeployment(TrainingDeployment):
    """An integration whose registered runtime factory builds its training component in process.

    Subclasses name their runtime_type. The runtime factory owns option parsing
    and imports its execution dependencies only when constructing the runtime.
    """

    runtime_type: str

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        if {"ray_address", "ray_namespace"} & config["reef"].keys() or "training" in config.get("execution", {}):
            raise DeployConfigError(
                "in-process training owns its execution; remove Ray and training executor settings"
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
