"""Assemble the native weight-training driver and HTTP service.

Slime owns its model workers and inference endpoint. The existing orchestrator
owns Ray attachment, driver readiness, HTTP startup and reverse-order cleanup.
Method-specific services are deliberately outside this assembly contract.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from reef.runtime.executor.config import role_executor_settings, select_executor
from reef.runtime.names import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.service.deploy.config import DeployConfigError, config_value, interpolate_config
from reef.service.deploy.options import native_arguments, normalize_native_options
from reef.service.deploy.provider import http_service
from reef.service.deploy.settings import service_settings_from_config

_NATIVE_INFERENCE = "reef.train.slime_backend.reef_adapters.sglang.chat.SGLangChatTrainingInferenceBackend"
_READY_PROBE = (
    "import os, pathlib, sys; "
    "p = pathlib.Path(os.environ['REEF_BRIDGE_READY_FILE']); "
    "sys.exit(0 if p.is_file() and p.read_text().strip() == 'reef-slime-bridge-ready' else 1)"
)


def assemble_training_services(config: dict[str, Any]) -> None:
    """Generate the supported Slime bridge topology without importing GPU packages."""
    settings = service_settings_from_config(config)
    if (settings.training_backend or "slime") != "slime":
        raise DeployConfigError("automatic weight training currently supports training.backend: slime")
    from reef.train.slime_backend.launch import driver_environment

    model = config_value(config, "reef", "model_path")
    if not isinstance(model, str) or not model:
        raise DeployConfigError("weight training requires --inference.model-path")
    if not settings.host.strip() or not 1 <= settings.port <= 65535:
        raise DeployConfigError("weight training requires a non-empty --reef.host and valid --reef.port")
    if settings.training_ready_timeout <= 0 or settings.inference_timeout_s <= 0:
        raise DeployConfigError("training.ready-timeout and inference.timeout-s must be positive")
    if settings.train_timeout_s is not None and settings.train_timeout_s <= 0:
        raise DeployConfigError("training.timeout-s must be positive")
    if (
        settings.upstream_url
        or settings.upstream_model
        or settings.upstream_api_key
        or settings.upstream_api != "openai"
    ):
        raise DeployConfigError(
            "automatic weight training uses training-owned inference; remove upstream provider settings"
        )
    if settings.inference_url or config.get("reef", {}).get("runtime"):
        raise DeployConfigError(
            "automatic weight training discovers its runtime and inference connection from the bridge"
        )
    if settings.inference_options or settings.tensor_parallel_size is not None:
        raise DeployConfigError("Slime owns inference workers; configure their native flags in training.options")
    if settings.inference_backend not in (None, "sglang"):
        raise DeployConfigError("Slime-managed inference currently requires inference.backend: sglang")

    execution = config.setdefault("execution", {})
    for role in ("training", "rollout"):
        execution.setdefault(role, "ray")
        if select_executor(role_executor_settings(config, role), role=role).settings.backend != "ray":
            raise DeployConfigError(f"automatic Slime training requires execution.{role}.backend: ray")

    options = normalize_native_options(settings.training_backend_options)
    native_arguments(options, reserved={"ready-file"})
    checkpoint = options.get("hf-checkpoint")
    if checkpoint is not None and (
        not isinstance(checkpoint, str) or os.path.expanduser(interpolate_config(config, checkpoint).strip()) != model
    ):
        raise DeployConfigError("training.options.hf-checkpoint must match inference.model-path")
    # Resolve/download the model once; both HTTP and Slime read that same path.
    options["hf-checkpoint"] = "${reef.model_path}"
    reef = config["reef"]
    reef.update(
        training_backend="slime",
        training_backend_options=options,
        ray_namespace=settings.ray_namespace or DEFAULT_NAMESPACE,
        ray_actor_name=settings.ray_actor_name or DEFAULT_ACTOR_NAME,
        inference_backend_factory=settings.inference_backend_factory or _NATIVE_INFERENCE,
    )
    python = os.environ.get("REEF_PYTHON", sys.executable)
    driver = {
        "name": "slime-driver",
        "executor": "uni",
        "command": [python, "-m", "reef.service.slime_driver"],
        "ready": [python, "-c", _READY_PROBE],
        "ready_timeout": settings.training_ready_timeout,
        "env": {
            **driver_environment(os.environ),
            "REEF_RAY_NAMESPACE": "${reef.ray_namespace}",
            "REEF_RAY_ACTOR_NAME": "${reef.ray_actor_name}",
            # Managed launches take native options from the resolved config.
            "SLIME_ARGS_FILE": "",
        },
    }
    http = http_service(config, settings)
    http["depends_on"] = [driver["name"]]
    config["services"] = [driver, http]
