"""Slime component definitions and runtime connection, without GPU imports."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

from reef.core.config import config_value, interpolate_config
from reef.core.errors import DeployConfigError
from reef.runtime.deployment import http_readiness_command
from reef.runtime.executor.arguments import native_arguments, normalize_native_options
from reef.runtime.executor.connection import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.train.deployment import (
    INFERENCE_RESERVED_OPTIONS,
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


def driver_arguments(config: Mapping[str, Any], *, loss_family: str = "") -> list[str]:
    """Adapt resolved component config to the pinned Slime parser at launch.

    Keep generated inference flags out of training.options. Legacy explicit
    process stacks without inference_num_gpus retain their native argument path.
    """
    reef = config.get("reef", {})
    # The deploy layer leaves references such as ``${reef.model_path}`` in the
    # managed options so the model resolves once for the HTTP service and the
    # driver alike; they are expanded here, against the same resolved config.
    training_options = expand_option_references(config, reef.get("training_backend_options", {}))
    if reef.get("teacher_model_path"):
        if not loss_family:
            raise DeployConfigError("a teacher engine requires a distillation loss family")
        managed = {
            f"{loss_family}-teacher-url": interpolate_config(config, "${endpoints.distill-teacher}"),
            f"{loss_family}-teacher-model-path": reef["teacher_model_path"],
            f"{loss_family}-teacher-timeout": reef.get("teacher_timeout", 300.0),
        }
        if any(name in training_options for name in (*managed, f"{loss_family}-teacher-checkpoint")):
            raise DeployConfigError("teacher section replaces native teacher URL, model path, timeout and checkpoint")
        training_options.update(managed)
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
    if reef.get("inference_backend") == "vllm":
        # vLLM engine options reach the driver through the resolved config;
        # Slime's parser only knows SGLang's names.
        return [*arguments, *native_arguments(options)]
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


def require_vllm_weight_transport(options: Mapping[str, Any]) -> None:
    """Admit only the Slime weight path a vLLM engine can receive: a full checkpoint reloaded from disk.

    Slime's NCCL and tensor transports, delta sync and LoRA publication call
    SGLang-only engine routes; ``check-weight-update-equal`` needs SGLang's
    weights checker.
    """
    if options.get("update-weight-transport") != "disk":
        raise DeployConfigError(
            "inference.backend: vllm requires training.options.update-weight-transport: disk; "
            "the NCCL and tensor transports reach SGLang engines only"
        )
    if options.get("update-weight-mode", "full") != "full":
        raise DeployConfigError("inference.backend: vllm requires training.options.update-weight-mode: full")
    if options.get("update-weight-local-checkpoint-dir"):
        raise DeployConfigError(
            "inference.backend: vllm cannot use training.options.update-weight-local-checkpoint-dir; "
            "vLLM engines reload the published checkpoint directly"
        )
    if int(options.get("megatron-lora-rank") or 0) > 0:
        raise DeployConfigError("inference.backend: vllm does not publish LoRA adapters yet; use full weights")
    if str(options.get("check-weight-update-equal", False)).lower() in ("true", "1"):
        raise DeployConfigError("inference.backend: vllm has no weights checker; drop check-weight-update-equal")


#: The teacher engine's service name; its endpoint reaches the driver as ``${endpoints.distill-teacher}``.
TEACHER_SERVICE = "distill-teacher"
#: SGLang options the teacher section sets itself or that would change which model it serves.
TEACHER_RESERVED_OPTIONS = frozenset(INFERENCE_RESERVED_OPTIONS) | {
    "tokenizer-path",
    "tokenizer-mode",
    "revision",
    "tokenizer-revision",
    "enable-lora",
    "lora-paths",
    "load-format",
    "enable-mis",
    "skip-tokenizer-init",
}


def teacher_service(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    """The teacher engine as a Ray service with its own GPUs, stopped with its deployment."""
    if not settings.get("teacher_model_path"):
        if settings.get("teacher_options"):
            raise DeployConfigError("teacher.options requires teacher.model-path")
        return None
    size = settings["teacher_num_gpus"]
    port = settings["teacher_port"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise DeployConfigError("teacher.num-gpus must be a positive integer")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 or port == settings["port"]:
        raise DeployConfigError("teacher.port must be valid and distinct from reef.port")
    options = normalize_native_options(settings["teacher_options"])
    arguments = native_arguments(options, reserved=set(TEACHER_RESERVED_OPTIONS))
    python = os.environ.get("REEF_PYTHON", sys.executable)
    return {
        "name": TEACHER_SERVICE,
        "executor": "ray",
        "resources": {"num_gpus": size, "num_cpus": 1},
        "endpoint": f"http://{{host}}:{port}",
        "command": [
            python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "${reef.teacher_model_path}",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tp",
            str(size),
            *arguments,
        ],
        "ready": http_readiness_command(python, f"${{endpoints.{TEACHER_SERVICE}}}/health"),
        "ready_timeout": settings["training_ready_timeout"],
    }


class SlimeDeployment(TrainingDeployment):
    """Describe Slime components for the Reef driver and connect HTTP to their bridge."""

    supports_teacher_engine = True

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        model = config_value(config, "reef", "model_path")
        if settings["inference_url"] or config.get("reef", {}).get("runtime"):
            raise DeployConfigError(
                "automatic weight training discovers its runtime and inference connection from the bridge"
            )
        if settings["inference_backend"] not in (None, "sglang", "vllm"):
            raise DeployConfigError("Slime weight transfer requires inference.backend: sglang or vllm")

        require_ray_roles(config, "training", "rollout", backend="Slime")
        options = normalize_native_options(settings["training_backend_options"])
        native_arguments(options, reserved={"ready-file"})
        if settings["inference_backend"] == "vllm":
            require_vllm_weight_transport(options)
        prepare_inference_config(config, settings, options)
        reef = config["reef"]
        if settings["inference_backend"] == "vllm" and reef["inference_num_gpus"] != reef["tensor_parallel_size"]:
            # Slime cannot name a router_url, and vLLM ships no router of its own.
            raise DeployConfigError(
                "inference.backend: vllm serves one engine per stack; "
                "set inference.num-gpus equal to inference.tensor-parallel-size"
            )
        checkpoint = options.get("hf-checkpoint")
        if checkpoint is not None and (
            not isinstance(checkpoint, str)
            or os.path.expanduser(interpolate_config(config, checkpoint).strip()) != model
        ):
            raise DeployConfigError("training.options.hf-checkpoint must match inference.model-path")
        # Resolve/download the model once; both HTTP and Slime read that same path.
        options["hf-checkpoint"] = "${reef.model_path}"
        reef.update(
            training_backend_options=options,
            ray_namespace=settings["ray_namespace"] or DEFAULT_NAMESPACE,
            ray_actor_name=settings["ray_actor_name"] or DEFAULT_ACTOR_NAME,
            inference_handler_factory=settings["inference_handler_factory"],
        )
        driver = driver_service("slime-driver", settings, driver_environment(os.environ))
        teacher = teacher_service(settings)
        if teacher is not None:
            driver["depends_on"] = [TEACHER_SERVICE]
            return (teacher, driver)
        return (driver,)

    def create_training_plan(self, config: Mapping[str, Any], *, loss_family: str) -> TrainingDeploymentPlan:
        from reef.train.slime_backend.driver import create_training_plan

        return create_training_plan(config, loss_family=loss_family)

    def runtime_config(self, settings: Mapping[str, Any], *, connector: Any = None) -> dict[str, Any]:
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
