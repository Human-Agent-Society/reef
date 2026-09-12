"""Lightweight training deployment contracts, separate from model execution."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.errors import DeployConfigError
from reef.runtime.executor.arguments import normalize_native_options
from reef.runtime.registry import runtime_factory_for


class TrainingDeployment(ABC):
    """Lightweight integration definition; preparation never allocates model resources."""

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
            "inference_backend_factory",
            "inference_backend_config",
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
