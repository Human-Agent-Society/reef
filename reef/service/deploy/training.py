"""Assemble selected training components using their integration's deployment definition."""

from __future__ import annotations

import importlib
from dataclasses import asdict, replace
from importlib.metadata import entry_points
from typing import Any

from reef.inference.remote.config import RemoteInferenceSettings
from reef.recipe.composite import CompositeRecipe
from reef.recipe.registry import recipe_class_for
from reef.service.deploy.config_utils import DeployConfigError, config_value
from reef.service.deploy.inference import http_service
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.train.deployment import NO_INFERENCE_BACKEND, TrainingDeployment

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


#: Settings that configure engines or need a served model, with their public names.
NO_INFERENCE_REFUSED_SETTINGS = (
    ("tensor_parallel_size", "inference.tensor-parallel-size"),
    ("inference_options", "inference.options"),
    ("inference_url", "inference.url"),
    ("inference_remote", "inference.remote"),
    ("inference_handler_factory", "inference.handler-factory"),
    ("inference_handler_config", "inference.handler-config"),
    ("colocate", "training.colocate"),
    ("teacher_model_path", "teacher.model-path"),
    ("evaluation_settings", "evaluation"),
    ("generator_settings", "generator"),
)


def validate_no_inference(settings: ServiceConfig) -> None:
    """Refuse what ``inference.backend: none`` cannot run: engines, colocation and anything that calls the model.

    The deployment trains on uploaded or imported data and serves no request.
    Engine settings and GPUs therefore have no engine to configure, and a
    teacher, candidate evaluation, task generation and the other components
    of a composite recipe would call a model that nothing serves.
    """
    values = asdict(settings)
    configured = [name for key, name in NO_INFERENCE_REFUSED_SETTINGS if values[key]]
    if settings.inference_num_gpus:
        configured.insert(0, "inference.num-gpus")
    if configured:
        raise DeployConfigError(
            f"inference.backend: none runs no inference engine and serves no requests; remove {', '.join(configured)}"
        )
    recipe = recipe_class_for(settings.recipe)
    if recipe is not None and issubclass(recipe, CompositeRecipe):
        raise DeployConfigError(
            "inference.backend: none cannot serve a composite recipe: its other components call the served model"
        )


def assemble_training_services(config: dict[str, Any]) -> None:
    """Validate common inputs; integrations own process topology and connections."""
    settings = service_config_from_mapping(config)
    backend = training_deployment_for(settings.training_backend)
    if settings.inference_backend == NO_INFERENCE_BACKEND:
        validate_no_inference(settings)
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
        # Every later reader gets the validated section, with underscore keys and defaults filled in. The
        # component loads the model as the file names it: a later download replaces reef.model_path on this host.
        remote = RemoteInferenceSettings.from_mapping(settings.inference_remote)
        if remote.model_path is None:
            remote = replace(remote, model_path=settings.model_path)
        config["reef"]["inference_remote"] = asdict(remote)
        settings = service_config_from_mapping(config)
    config["reef"]["training_backend"] = settings.training_backend or "slime"
    dependencies = backend.prepare(config, asdict(settings))
    http = http_service(config, service_config_from_mapping(config))
    if dependencies:
        http["depends_on"] = [process["name"] for process in dependencies]
    config["services"] = [*dependencies, http]
