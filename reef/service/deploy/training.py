"""Assemble a weight-training deployment: its training integration, and the serving its inference section selects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

from reef.core.config import config_arguments
from reef.core.plugins import PluginConfigError, resolve_plugin_class
from reef.inference.remote.config import RemoteInferenceSettings
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.composite import CompositeRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import recipe_class_for
from reef.service.deploy.config_utils import DeployConfigError, config_value
from reef.service.deploy.inference import http_service, local_inference_component
from reef.service.deploy.service_config import ServiceSettings, service_config_from_mapping
from reef.service.deploy.serving import ENGINES, NONE, prepare_engine_serving, serving_mode
from reef.train.deployment import TrainingDeployment

#: The bundled training backends, as references so that reading the table imports no training dependency.
BUILTIN_DEPLOYMENTS = {
    "slime": "reef.train.slime_backend.launch:SlimeDeployment",
    "tinker": "reef.train.tinker_backend.launch:TinkerDeployment",
    "spindle_full": "reef.train.tinker_backend.launch:SpindleFullDeployment",
}


def training_deployment_for(name: str | None) -> TrainingDeployment:
    """The deployment definition of the ``training.backend`` value ``name`` (default ``slime``).

    ``name`` is a bundled backend, an installed ``reef.training_backends`` entry point, or a
    ``package.module:ClassName`` reference; each names a ``TrainingDeployment`` class, made with no arguments.
    Raise ``DeployConfigError`` for another name.
    """
    try:
        deployment_class = resolve_plugin_class(
            name or "slime",
            builtins=BUILTIN_DEPLOYMENTS,
            entry_point_group="reef.training_backends",
            base=TrainingDeployment,  # type: ignore[type-abstract]  # the base is a value; the result is concrete
            setting="training backend",
        )
    except PluginConfigError as exc:
        raise DeployConfigError(str(exc)) from exc
    return deployment_class()


def training_recipe_config(
    recipe_type: type[WeightTrainingRecipe], settings: Mapping[str, Any], *, model_path: str
) -> dict[str, Any]:
    """The flat recipe config of a weight training deployment, from the recipe's keys of its ``reef`` section.

    ``settings`` is the recipe-owned slice of the deployment's flat ``reef`` section: the caller removes the
    service's own keys first (``service_owned_keys``). Only keys the operator set are forwarded under ``data``, so
    every default lives with the config fields of ``recipe_type``, never in the service layer. This step only
    translates the layout; the loader's field resolution converts the types. ``model_path`` goes under
    ``model.path``. Raise ``RecipeConfigError`` for a key that ``recipe_type`` does not declare: the operator set
    a value that nothing would consume.
    """
    field_names = {argument.name for argument in config_arguments(recipe_type)}
    data: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in settings.items():
        if key in field_names:
            # A YAML key left empty (None) is unset, not a value to parse.
            if value is not None:
                data[key] = value
        else:
            unknown.append(key)
    if unknown:
        known = ", ".join(f"reef.{name}" for name in sorted(field_names))
        raise RecipeConfigError(
            f"{recipe_type.__name__} does not consume {', '.join(f'reef.{key}' for key in sorted(unknown))}; "
            f"the recipe would silently run with its defaults instead. "
            f"Settings {recipe_type.__name__} consumes: {known}. "
            f"A new setting is declared with config_field() on the recipe dataclass "
            f"(see reef.recipe.config_fields)."
        )
    return {"data": data, "model": {"path": model_path}}


def local_model_required(config: dict[str, Any]) -> bool:
    """Whether this host loads the model: for local serving engines, or for a training backend that runs it here.

    Provider deployments and local engines need the resolved snapshot on disk.
    A hosted backend whose snapshots engines on a remote component serve, or
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


def validate_no_inference(settings: ServiceSettings) -> None:
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
