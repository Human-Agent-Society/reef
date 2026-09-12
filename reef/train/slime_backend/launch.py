"""Slime-owned process topology and runtime connection, without GPU imports."""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Mapping
from typing import Any

from reef.core.config import config_value, interpolate_config
from reef.core.errors import DeployConfigError
from reef.runtime.executor.arguments import native_arguments, normalize_native_options
from reef.runtime.executor.config import role_executor_settings, select_executor
from reef.runtime.inference import InferenceBackendFactory
from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.train.deployment import TrainingDeployment

_NATIVE_INFERENCE = "reef.train.slime_backend.reef_adapters.sglang.chat.SGLangChatTrainingInferenceBackend"
_READY_PROBE = (
    "import os, pathlib, sys; "
    "p = pathlib.Path(os.environ['REEF_BRIDGE_READY_FILE']); "
    "sys.exit(0 if p.is_file() and p.read_text().strip() == 'reef-slime-bridge-ready' else 1)"
)


def driver_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Set the CUDA/NCCL defaults used by the shipped Slime deployments."""
    return {
        "CUDA_DEVICE_MAX_CONNECTIONS": environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
        "NCCL_NVLS_ENABLE": environ.get("NCCL_NVLS_ENABLE", "0"),
    }


def _configured_inference_backend_factory(path: str | None) -> InferenceBackendFactory | None:
    """Load an optional backend factory selected by deployment config."""

    if path is None:
        return None
    if not isinstance(path, str) or not path.strip():
        raise ValueError("reef.inference_backend_factory must be a non-empty dotted path")
    module_path, separator, attribute = path.strip().rpartition(".")
    if not separator or not module_path or not attribute:
        raise ValueError("reef.inference_backend_factory must be a dotted path")
    try:
        factory = getattr(importlib.import_module(module_path), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load reef.inference_backend_factory {path!r}") from exc
    if not callable(factory):
        raise ValueError(f"reef.inference_backend_factory {path!r} is not callable")
    return factory


# These alter engine creation or topology outside Reef's inference request.
# Reject argparse abbreviations as well as full names, even for omitted values.
_INFERENCE_LAUNCH_OPTIONS = {
    "rollout-num-gpus",
    "rollout-num-gpus-per-engine",
    "rollout-external",
    "rollout-external-engine-addrs",
    "prefill-num-servers",
}
_INFERENCE_RESERVED_OPTIONS = {
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
            or any(flag.startswith(name) for flag in _INFERENCE_LAUNCH_OPTIONS)
        ):
            raise DeployConfigError(
                f"training.options.{name} configures inference; use inference.num-gpus, "
                "inference.tensor-parallel-size or inference.options instead"
            )
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
    native_arguments(options, reserved=_INFERENCE_RESERVED_OPTIONS)
    if any(name.startswith("sglang-") for name in options):
        raise DeployConfigError("inference.options uses native SGLang names without the sglang- prefix")
    config["reef"].update(
        inference_backend="sglang",
        inference_num_gpus=num_gpus,
        tensor_parallel_size=parallel_size,
        inference_options=options,
    )


def driver_arguments(config: Mapping[str, Any]) -> list[str]:
    """Adapt resolved component config to the pinned Slime parser at launch.

    Keep generated inference flags out of training.options. Legacy explicit
    process stacks without inference_num_gpus retain their native argument path.
    """
    reef = config.get("reef", {})
    arguments = native_arguments(reef.get("training_backend_options", {}))
    if reef.get("inference_num_gpus") is None:
        return arguments
    options = {
        "rollout-num-gpus": reef["inference_num_gpus"],
        "rollout-num-gpus-per-engine": reef["tensor_parallel_size"],
    }
    for name, value in reef.get("inference_options", {}).items():
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
    """Launch the Slime controller and connect HTTP to its shared Ray bridge."""

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        model = config_value(config, "reef", "model_path")
        if settings["inference_url"] or config.get("reef", {}).get("runtime"):
            raise DeployConfigError(
                "automatic weight training discovers its runtime and inference connection from the bridge"
            )
        if settings["inference_backend"] not in (None, "sglang"):
            raise DeployConfigError("Slime-managed inference currently requires inference.backend: sglang")

        execution = config.setdefault("execution", {})
        for role in ("training", "rollout"):
            execution.setdefault(role, "ray")
            if select_executor(role_executor_settings(config, role), role=role).settings.backend != "ray":
                raise DeployConfigError(f"automatic Slime training requires execution.{role}.backend: ray")

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
            inference_backend_factory=settings["inference_backend_factory"] or _NATIVE_INFERENCE,
        )
        python = os.environ.get("REEF_PYTHON", sys.executable)
        driver = {
            "name": "slime-driver",
            "executor": "uni",
            "command": [python, "-m", "reef.service.slime_driver"],
            "ready": [python, "-c", _READY_PROBE],
            "ready_timeout": settings["training_ready_timeout"],
            "env": {
                **driver_environment(os.environ),
                "REEF_RAY_NAMESPACE": "${reef.ray_namespace}",
                "REEF_RAY_ACTOR_NAME": "${reef.ray_actor_name}",
                # Managed launches take native options from the resolved config.
                "SLIME_ARGS_FILE": "",
            },
        }
        return (driver,)

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
            "type": "ray_training",
            "inference_url": inference_url or None,
            "actor_name": settings["ray_actor_name"],
            "namespace": settings["ray_namespace"],
            "ray_address": ray_address,
            "inference_timeout_s": settings["inference_timeout_s"],
            "train_timeout_s": settings["train_timeout_s"],
        }
        if max_staleness:
            runtime_config["max_staleness"] = max_staleness
        inference_backend_factory = _configured_inference_backend_factory(settings["inference_backend_factory"])
        if inference_backend_factory is not None:
            runtime_config["inference_backend_factory"] = inference_backend_factory
        if not isinstance(settings["inference_backend_config"], Mapping):
            raise ValueError("reef.inference_backend_config must be an object")
        if settings["inference_backend_config"]:
            if inference_backend_factory is None:
                raise ValueError("reef.inference_backend_config requires reef.inference_backend_factory")
            runtime_config["inference_backend_config"] = dict(settings["inference_backend_config"])
        if connector is not None:
            runtime_config["connect"] = connector
        return runtime_config
