"""Assemble a weight-training deployment: its training integration, and the serving its inference section selects."""

from __future__ import annotations

import importlib
from dataclasses import asdict, replace
from importlib.metadata import entry_points
from typing import Any

from reef.inference.remote.config import RemoteInferenceSettings
from reef.recipe.composite import CompositeRecipe
from reef.recipe.registry import recipe_class_for
from reef.service.deploy.config_utils import DeployConfigError, config_value
from reef.service.deploy.inference import http_service, local_inference_component
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.service.deploy.serving import ENGINES, NONE, prepare_engine_serving, serving_mode
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
    """Whether this host loads the model: for local serving engines, or for a training backend that runs it here.

    Provider deployments and local engines need the resolved snapshot on disk.
    A hosted backend whose releases engines on a remote component serve, or
    that serves them itself, keeps the model identifier as written.
    """
    backend_name = config.get("reef", {}).get("training_backend")
    if not backend_name:
        return True
    backend = training_deployment_for(backend_name)
    settings = service_config_from_mapping(config)
    if serving_mode(settings, backend) == ENGINES and not settings.inference_remote:
        return True
    return backend.requires_local_model


#: Settings that configure engines or need a served model, with their public names.
NO_INFERENCE_REFUSED_SETTINGS = (
    ("tensor_parallel_size", "inference.tensor-parallel-size"),
    ("inference_options", "inference.options"),
    ("inference_url", "inference.url"),
    ("inference_remote", "inference.remote"),
    ("inference_handler_factory", "inference.handler-factory"),
    ("inference_handler_config", "inference.handler-config"),
    ("teacher_model_path", "teacher.model-path"),
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
    """Validate common inputs, place the serving engines, and let the integration add its own processes.

    Serving follows the ``inference`` section (``serving_mode``): Reef's
    engines run on the inference component that ``inference.remote`` names,
    or on one that this host starts. The training integration owns only its
    own processes and connections.
    """
    settings = service_config_from_mapping(config)
    backend = training_deployment_for(settings.training_backend)
    mode = serving_mode(settings, backend)
    if mode == NONE:
        validate_no_inference(settings)
    if settings.teacher_model_path and not backend.supports_teacher_engine:
        raise DeployConfigError("selected training backend does not support a teacher engine")
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
        raise DeployConfigError("weight training selects its serving through inference.serving; remove recipe.runtime")
    if settings.inference_url:
        raise DeployConfigError("inference.url is reserved: Reef starts or names the serving engines itself")
    component = None
    if mode == ENGINES:
        prepare_engine_serving(config, settings)
        settings = service_config_from_mapping(config)
        if not settings.inference_remote:
            component = local_inference_component(config, settings)
            settings = service_config_from_mapping(config)
        # Every later reader gets the validated section, with underscore keys and defaults filled in. The
        # component loads the model as the file names it: a later download replaces reef.model_path on this host.
        remote = RemoteInferenceSettings.from_mapping(settings.inference_remote)
        if remote.model_path is None:
            remote = replace(remote, model_path=settings.model_path)
        config["reef"]["inference_remote"] = asdict(remote)
        settings = service_config_from_mapping(config)
        # The HTTP service starts the engines before it serves its first request.
        config["ready_timeout"] = max(int(config.get("ready_timeout", 30)), int(settings.training_ready_timeout))
    config["reef"]["training_backend"] = settings.training_backend or "slime"
    dependencies = backend.prepare(config, asdict(settings))
    if component is not None:
        dependencies = (component, *dependencies)
    http = http_service(config, service_config_from_mapping(config))
    environment = backend.http_environment(asdict(settings))
    if environment:
        http["env"] = {**http.get("env", {}), **environment}
    if dependencies:
        http["depends_on"] = [process["name"] for process in dependencies]
    config["services"] = [*dependencies, http]
