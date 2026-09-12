"""Assemble selected training components using their integration's deployment definition."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from reef.service.deploy.config import DeployConfigError, config_value
from reef.service.deploy.provider import http_service
from reef.service.deploy.settings import service_settings_from_config
from reef.service.deploy.training_backend import training_deployment_for


def assemble_training_services(config: dict[str, Any]) -> None:
    """Validate common inputs; integrations own process topology and connections."""
    settings = service_settings_from_config(config)
    backend = training_deployment_for(settings.training_backend)
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
    if config.get("reef", {}).get("runtime"):
        raise DeployConfigError("weight training selects its runtime through training.backend; remove recipe.runtime")
    config["reef"]["training_backend"] = settings.training_backend or "slime"
    dependencies = backend.prepare(config, asdict(settings))
    http = http_service(config, service_settings_from_config(config))
    if dependencies:
        http["depends_on"] = [process["name"] for process in dependencies]
    config["services"] = [*dependencies, http]
