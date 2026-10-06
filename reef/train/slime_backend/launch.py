"""Slime component definitions and runtime connection, without GPU imports."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from reef.core.config import config_value, interpolate_config
from reef.core.errors import DeployConfigError
from reef.runtime.executor.arguments import native_arguments, normalize_native_options
from reef.runtime.executor.connection import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.train.deployment import (
    TrainingDeployment,
    TrainingDeploymentPlan,
    driver_service,
    inference_handler_factory_for,
    prepare_inference_config,
    require_ray_roles,
)


def driver_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Set the CUDA/NCCL defaults used by the shipped Slime deployments."""
    return {
        "CUDA_DEVICE_MAX_CONNECTIONS": environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
        "NCCL_NVLS_ENABLE": environ.get("NCCL_NVLS_ENABLE", "0"),
    }


OptionValue = bool | int | float | str | list["OptionValue"] | dict[str, "OptionValue"] | None


def expand_option_references(
    config: Mapping[str, object], options: Mapping[str, OptionValue]
) -> dict[str, OptionValue]:
    """Resolve configuration references in nested native training and inference options."""

    def _expand(value: OptionValue) -> OptionValue:
        if isinstance(value, str):
            return interpolate_config(config, value)
        if isinstance(value, dict):
            return {key: _expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_expand(item) for item in value]
        return value

    return {name: _expand(value) for name, value in options.items()}


def driver_arguments(config: Mapping[str, Any]) -> list[str]:
    """Adapt resolved component config to the pinned Slime parser at launch.

    Keep generated inference flags out of training.options. Legacy explicit
    process stacks without inference_num_gpus retain their native argument path.
    """
    reef = config.get("reef", {})
    # The deploy layer leaves references such as ``${reef.model_path}`` in the
    # managed options so the model resolves once for the HTTP service and the
    # driver alike; they are expanded here, against the same resolved config.
    training_options = expand_option_references(config, reef.get("training_backend_options", {}))
    arguments = native_arguments(training_options)
    if reef.get("inference_num_gpus") is None:
        return arguments
    options = {
        "rollout-num-gpus": reef["inference_num_gpus"],
        "rollout-num-gpus-per-engine": reef["tensor_parallel_size"],
    }
    if reef.get("colocate"):
        # Slime's workers still read these flags; Reef derives them from one decision.
        options["colocate"] = True
        options["offload-rollout"] = True
        if "offload-train" not in training_options:
            options["offload-train"] = True
    for name, value in expand_option_references(config, reef.get("inference_options", {})).items():
        # Slime has dedicated router bind flags and passes other router flags
        # directly to RouterArgs. Engine flags are all prefixed by Slime.
        flag = (
            name
            if name.startswith("router-") and name not in {"router-ip", "router-port", "router-request-timeout-secs"}
            else "sglang-" + name
        )
        options[flag] = value
    return [*arguments, *native_arguments(options)]


class SlimeDeployment(TrainingDeployment):
    """Describe Slime components for the Reef driver and connect HTTP to their bridge."""

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        model = config_value(config, "reef", "model_path")
        if settings["inference_url"] or config.get("reef", {}).get("runtime"):
            raise DeployConfigError(
                "automatic weight training discovers its runtime and inference connection from the bridge"
            )
        if settings["inference_backend"] not in (None, "sglang"):
            raise DeployConfigError("Slime weight transfer currently requires inference.backend: sglang")

        require_ray_roles(config, "training", "rollout", backend="Slime")
        options = normalize_native_options(settings["training_backend_options"])
        native_arguments(options, reserved={"ready-file"})
        prepare_inference_config(config, settings, options)
        checkpoint = options.get("hf-checkpoint")
        if checkpoint is not None and (
            not isinstance(checkpoint, str)
            or os.path.expanduser(interpolate_config(config, checkpoint).strip()) != model
        ):
            raise DeployConfigError("training.options.hf-checkpoint must match inference.model-path")
        # Resolve/download the model once; both HTTP and Slime read that same path.
        options["hf-checkpoint"] = "${reef.model_path}"
        reef = config["reef"]
        reef.update(
            training_backend_options=options,
            ray_namespace=settings["ray_namespace"] or DEFAULT_NAMESPACE,
            ray_actor_name=settings["ray_actor_name"] or DEFAULT_ACTOR_NAME,
            inference_handler_factory=settings["inference_handler_factory"],
        )
        # Managed launches take native options from the resolved config.
        return (driver_service("slime-driver", settings, {**driver_environment(os.environ), "SLIME_ARGS_FILE": ""}),)

    def create_training_plan(self, config: Mapping[str, Any], *, loss_family: str) -> TrainingDeploymentPlan:
        from reef.train.slime_backend.driver import create_training_plan

        return create_training_plan(config, loss_family=loss_family)

    def runtime_config(
        self, settings: Mapping[str, Any], *, max_staleness: int, connector: Any = None
    ) -> dict[str, Any]:
        ray_address = settings["ray_address"]
        if not isinstance(ray_address, str) or not ray_address.strip():
            raise ValueError("reef.ray_address is required")
        inference_url = settings["inference_url"].strip() if isinstance(settings["inference_url"], str) else None
        if settings["inference_timeout_s"] <= 0:
            raise ValueError("reef.inference_timeout_s must be positive")
        if settings["train_timeout_s"] is not None and settings["train_timeout_s"] <= 0:
            raise ValueError("reef.train_timeout_s must be positive when set")
        runtime_config: dict[str, Any] = {
            "type": "slime_training",
            "inference_runtime": settings["inference_backend"] or "sglang",
            "inference_url": inference_url or None,
            "actor_name": settings["ray_actor_name"],
            "namespace": settings["ray_namespace"],
            "ray_address": ray_address,
            "inference_timeout_s": settings["inference_timeout_s"],
            "train_timeout_s": settings["train_timeout_s"],
        }
        if max_staleness:
            runtime_config["max_staleness"] = max_staleness
        inference_handler_factory = inference_handler_factory_for(settings["inference_handler_factory"])
        if inference_handler_factory is not None:
            runtime_config["inference_handler_factory"] = inference_handler_factory
        if not isinstance(settings["inference_handler_config"], Mapping):
            raise ValueError("reef.inference_handler_config must be an object")
        if settings["inference_handler_config"]:
            runtime_config["inference_handler_config"] = dict(settings["inference_handler_config"])
        if connector is not None:
            runtime_config["connect"] = connector
        return runtime_config
