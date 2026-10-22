"""Assemble selected training components using their integration's deployment definition."""

from __future__ import annotations

import importlib
from dataclasses import asdict
from importlib.metadata import entry_points
from typing import Any

from reef.inference.remote.config import RemoteInferenceSettings
from reef.service.deploy.config_utils import DeployConfigError, config_value
from reef.service.deploy.inference import http_service
from reef.service.deploy.service_config import service_config_from_mapping
from reef.train.deployment import TrainingDeployment


# Built-ins are references, so discovery does not import execution dependencies.
_BUILTINS = {
    "slime": "reef.train.slime_backend.launch:SlimeDeployment",
    "tinker": "reef.train.tinker_backend.launch:TinkerDeployment",
}


def training_deployment_for(name: str | None) -> TrainingDeployment:
    """Load one built-in, installed entry point, or module:class deployment definition."""
    selected = name or "slime"
    reference = _BUILTINS.get(selected, selected)
    if ":" not in reference:
        matches = tuple(entry_points(group="reef.training_backends", name=selected))
        if len(matches) != 1:
            reason = "unknown" if not matches else "ambiguous"
            raise DeployConfigError(
                f"{reason} training backend {selected!r}; install its integration or use module:class"
            )
        reference = matches[0].value
    module, _, attribute = reference.partition(":")
    try:
        definition = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as exc:
        raise DeployConfigError(f"cannot load training backend {selected!r}: {exc}") from exc
    if not isinstance(definition, type) or not issubclass(definition, TrainingDeployment):
        raise DeployConfigError("training backend must name a TrainingDeployment class")
    return definition()


def local_model_required(config: dict[str, Any]) -> bool:
    """Whether the selected training backend executes the model on this host.

    Provider deployments and local engines need the resolved snapshot on disk.
    A hosted backend without local engines, including one whose engines run on
    a remote component, keeps the model identifier as written.
    """
    reef = config.get("reef", {})
    backend = reef.get("training_backend")
    inference_backend = reef.get("inference_backend")
    if not backend or (inference_backend and not reef.get("inference_remote")):
        return True
    return training_deployment_for(backend).requires_local_model


def assemble_training_services(config: dict[str, Any]) -> None:
    """Validate common inputs; integrations own process topology and connections."""
    settings = service_config_from_mapping(config)
    backend = training_deployment_for(settings.training_backend)
    if settings.teacher_model_path and not backend.supports_teacher_engine:
        raise DeployConfigError("selected training backend does not support a teacher engine")
    if settings.inference_remote and not backend.supports_remote_inference:
        raise DeployConfigError("selected training backend does not support remote engines (inference.remote)")
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
            "automatic weight training requires managed inference; remove upstream provider settings"
        )
    if config.get("reef", {}).get("runtime"):
        raise DeployConfigError("weight training selects its runtime through training.backend; remove recipe.runtime")
    if settings.inference_remote:
        # Every later reader gets the validated section, with underscore keys and defaults filled in.
        config["reef"]["inference_remote"] = asdict(RemoteInferenceSettings.from_mapping(settings.inference_remote))
        settings = service_config_from_mapping(config)
    config["reef"]["training_backend"] = settings.training_backend or "slime"
    dependencies = backend.prepare(config, asdict(settings))
    http = http_service(config, service_config_from_mapping(config))
    if dependencies:
        http["depends_on"] = [process["name"] for process in dependencies]
    config["services"] = [*dependencies, http]
