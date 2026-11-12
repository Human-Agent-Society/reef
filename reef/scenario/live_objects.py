"""The live objects of one process, and what a recipe's configuration means with them.

A recipe is configuration (RFC #1390): it holds no runtime, no training component and no store. The process that
builds those objects holds them in one ``LiveObjects`` value. The service assembly (``reef.service.assembly``) makes
the value, and the dispatcher, the scenario registry and the scenario factory get it as an argument. The type is in
this package because the scenario factory is its lowest reader, and the service package imports this one.

The functions of this module answer the questions that need both a recipe and a live object: the components of a
scenario, the default start, the check of an item start, and the models of a harness. Each takes the recipe and the
``LiveObjects`` as arguments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from reef.artifact.components import ScenarioComponents
from reef.artifact.repository import DefaultStart
from reef.core.errors import ConfigError
from reef.harness.tasks.harness_models import HarnessModels
from reef.harness.tasks.model_binding import ModelBinding
from reef.inference.interfaces import InferenceRuntime
from reef.recipe.base import WEIGHTS_COMPONENT, Recipe, WeightTrainingRecipe
from reef.recipe.composite import CompositeRecipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.storage.data import DataStore
from reef.train.component import ScenarioRoot, TrainingArtifactLayout, TrainingComponent, component_root


@dataclass(frozen=True)
class ServedEndpoint:
    """Where this Reef answers inference itself: what a harness's own evaluation calls target.

    Values: ``url`` is the service's base URL, a non-empty string kept without surrounding spaces and without a
    trailing slash. ``token`` is a bearer token that the service accepts, or ``None`` for none.

    Purpose: the service assembly makes one for a deployment whose evaluation calls come back to the service;
    ``harness_models`` reads it.
    """

    url: str
    token: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("served endpoint requires a url")
        object.__setattr__(self, "url", self.url.strip().rstrip("/"))


@dataclass(frozen=True, kw_only=True)
class LiveObjects:
    """The objects of one process that run the deployment's recipe.

    Values:

    - ``runtime`` is the serving runtime, or ``None`` for a process with none.
    - ``training_component`` runs the jobs of the recipe that trains weights, or is ``None`` when no recipe of the
      deployment trains weights.
    - ``harness_component`` runs the jobs of the deployment's harness recipe, or is ``None`` when the deployment
      evolves no harness or the process built no component for it.
    - ``data_store`` keeps the data directory of each training job, or is ``None`` for no store.
    - ``served_endpoint`` is where this Reef answers inference itself, or ``None`` when a harness run calls the
      runtime's own endpoint.

    The two components, when both are set, have different names: a job names its component by that name.

    Purpose: the service assembly builds one value for each process (``reef.service.assembly``). The dispatcher
    gets it and gives the scenario factory a copy whose components are the jobs queue's views. The functions of
    this module read it together with the recipe. It holds exactly these objects; nothing else is looked up in it.
    """

    runtime: InferenceRuntime | None = None
    training_component: TrainingComponent | None = None
    harness_component: TrainingComponent | None = None
    data_store: DataStore | None = None
    served_endpoint: ServedEndpoint | None = None

    def __post_init__(self) -> None:
        if self.runtime is not None and not isinstance(self.runtime, InferenceRuntime):
            raise TypeError(f"runtime must be an InferenceRuntime, got {type(self.runtime).__name__}")
        for name, component in (
            ("training_component", self.training_component),
            ("harness_component", self.harness_component),
        ):
            if component is not None and not isinstance(component, TrainingComponent):
                raise TypeError(f"{name} must be a TrainingComponent, got {type(component).__name__}")
        if self.data_store is not None and not isinstance(self.data_store, DataStore):
            raise TypeError(f"data_store must be a DataStore, got {type(self.data_store).__name__}")
        if self.served_endpoint is not None and not isinstance(self.served_endpoint, ServedEndpoint):
            raise TypeError(f"served_endpoint must be a ServedEndpoint, got {type(self.served_endpoint).__name__}")
        if (
            self.training_component is not None
            and self.harness_component is not None
            and self.training_component.name == self.harness_component.name
        ):
            raise ValueError(f"two job components are named {self.training_component.name!r}")

    def job_components(self) -> tuple[TrainingComponent, ...]:
        """The components whose jobs the trainers submit, each under its own ``name``: the training component, then
        the harness component, without the ones that are ``None``."""
        return tuple(
            component for component in (self.training_component, self.harness_component) if component is not None
        )


def close_runtimes(
    training_components: Sequence[TrainingComponent], runtime: InferenceRuntime | None
) -> list[BaseException]:
    """Close training components and an inference runtime, and return their close errors in that order.

    Admission closes first, so that no new request starts. Then the training
    components close, and then the runtime shuts down. A failed close does not
    stop the next one. ``None`` stands for an absent runtime.
    """
    if runtime is not None:
        runtime.close_admission()
    errors: list[BaseException] = []
    for training_component in training_components:
        try:
            training_component.close()
        except BaseException as exc:
            errors.append(exc)
    if runtime is not None:
        try:
            runtime.shutdown()
        except BaseException as exc:
            errors.append(exc)
    return errors


def weight_component_name(recipe: Recipe) -> str | None:
    """The name of the scenario component whose weights ``recipe`` trains: ``weights`` for a weight recipe, the
    name of the one weight recipe among a composite's components, and ``None`` for a recipe that trains no weights."""
    if isinstance(recipe, WeightTrainingRecipe):
        return WEIGHTS_COMPONENT
    if isinstance(recipe, CompositeRecipe):
        # A composite binds at most one weight component.
        return next(
            (name for name, child in recipe.components.items() if isinstance(child, WeightTrainingRecipe)), None
        )
    return None


def weight_objects(recipe: Recipe, live: LiveObjects) -> tuple[TrainingComponent, InferenceRuntime]:
    """The training component and the runtime of ``live``, which a recipe that trains weights needs; raise
    ``RecipeConfigError`` when ``live`` has no training component or no runtime."""
    if live.training_component is None:
        raise RecipeConfigError(f"{type(recipe).__name__} requires a training component")
    if live.runtime is None:
        raise RecipeConfigError("weight recipes require an independent inference runtime")
    return live.training_component, live.runtime


def check_recipe_objects(recipe: Recipe, live: LiveObjects) -> None:
    """Check that ``live`` has the objects that ``recipe`` needs.

    A recipe that trains weights, alone or as a component of a composite, needs a training component and a runtime.
    A weight recipe whose jobs are task jobs (``WeightTrainingRecipe.trains_task_jobs``) needs a training component
    that takes the recipe's task parameters (``TrainingComponent.check_task_jobs``). Raise ``RecipeConfigError``
    otherwise.
    """
    recipes = recipe.components.values() if isinstance(recipe, CompositeRecipe) else (recipe,)
    for one in recipes:
        if not isinstance(one, WeightTrainingRecipe):
            continue
        training_component, _ = weight_objects(one, live)
        if type(one).trains_task_jobs:
            try:
                training_component.check_task_jobs(one.parameters_by_kind()["task"])
            except ValueError as exc:
                raise RecipeConfigError(
                    f"{type(one).__name__} trains task jobs, whose recipe.config.job_parameters name the "
                    f"objective, the rollouts of each task and the agent that runs them: {exc}"
                ) from exc


def scenario_components(recipe: Recipe, live: LiveObjects, scenario: str) -> ScenarioComponents:
    """The components that every snapshot of ``scenario`` binds in this deployment.

    The result is what the recipe declares from its configuration (``Recipe.build_components``), with the facts of
    the weight component that only live objects know: its kind is the runtime's ``served_weight_kind``, and its
    layout is the training component's (``TrainingArtifactLayout``). A recipe that trains no weights gets its
    declaration unchanged. Raise ``RecipeConfigError`` when the recipe trains weights and ``live`` has no training
    component or no runtime.
    """
    declared = recipe.build_components(scenario)
    name = weight_component_name(recipe)
    if name is None:
        return declared
    training_component, runtime = weight_objects(recipe, live)
    weights = replace(
        declared.components[name],
        kind=runtime.served_weight_kind,
        layout=TrainingArtifactLayout(training_component),
    )
    return ScenarioComponents({**declared.components, name: weights}, declared.validator)


def default_start(recipe: Recipe, live: LiveObjects) -> DefaultStart:
    """The content of a new scenario's root snapshot when no artifact item starts the scenario.

    - A weight recipe starts from the training component's untrained checkpoint
      (``TrainingComponent.default_start``).
    - A composite starts from each component's default start under that component's directory. The files and the
      model files of a component go under its name, and each input is named ``<component>.<input>``.
    - Any other recipe starts from what its configuration gives (``Recipe.default_start``).

    Raise ``RecipeConfigError`` when two components of a composite hold model files, and when the recipe trains
    weights and ``live`` has no training component.
    """
    if isinstance(recipe, WeightTrainingRecipe):
        return weight_objects(recipe, live)[0].default_start()
    if not isinstance(recipe, CompositeRecipe):
        return recipe.default_start()
    sources = {component: default_start(child, live) for component, child in recipe.components.items()}
    models = [component for component, source in sources.items() if source.model_files is not None]
    if len(models) > 1:
        raise RecipeConfigError(f"components {models} each hold a base model; a composite trains one model")
    return DefaultStart(
        files={
            f"{component}/{path}": text for component, source in sources.items() for path, text in source.files.items()
        },
        model_files=sources[models[0]].model_files if models else None,
        model_component=models[0] if models else None,
        inputs={
            f"{component}.{name}": value
            for component, source in sources.items()
            for name, value in source.inputs.items()
        },
    )


def check_root_start(recipe: Recipe, live: LiveObjects, root: ScenarioRoot) -> None:
    """Check that training can start from ``root`` before an item start is registered.

    The training component of ``live`` checks the root (``TrainingComponent.root_start``); without one, nothing is
    checked. For a composite, it checks each component's view of the root (``component_root``). Raise
    ``ValueError`` on refusal.
    """
    if live.training_component is None:
        return
    if isinstance(recipe, CompositeRecipe):
        for component in recipe.components:
            live.training_component.root_start(component_root(root, component))
    else:
        live.training_component.root_start(root)


def harness_models(
    recipe: HarnessEvolutionRecipe, live: LiveObjects, *, component: str | None = None
) -> HarnessModels:
    """The models of the runs of the harness recipe ``recipe``: the runtime of ``live`` as the model under test,
    behind the served endpoint of ``live`` when it has one, and the recipe's named models.

    ``component`` names the snapshot component that the harness is when the recipe is a component of a composite;
    the evaluation calls then name it, so that the snapshot's hooks for it stay out. ``None`` is a harness recipe
    on its own.
    """
    endpoint = live.served_endpoint
    return HarnessModels(
        runtime=live.runtime,
        adapter=recipe.evolution.adapter,
        model_name=recipe.model_name,
        model_metadata=recipe.model_metadata,
        named=recipe.models,
        service_url=None if endpoint is None else endpoint.url,
        service_token=None if endpoint is None else endpoint.token,
        service_component=None if endpoint is None else component,
    )


def harness_served_model(recipe: Recipe, live: LiveObjects, scenario: str, component: str) -> ModelBinding | None:
    """The model that ``component``'s harness in ``scenario`` calls: a client's layer asks Reef for this model by
    name and dialect (``HarnessModels.served``).

    ``None`` when the recipe evolves no harness in ``component``: a recipe other than a harness recipe, or a
    composite whose component of that name is none. Raise ``RecipeConfigError`` when ``live`` has no runtime.
    """
    if isinstance(recipe, CompositeRecipe):
        harness, named = recipe.components.get(component), component
    else:
        harness, named = recipe, None
    if not isinstance(harness, HarnessEvolutionRecipe):
        return None
    try:
        return harness_models(harness, live, component=named).served(scenario)
    except ConfigError as exc:
        raise RecipeConfigError(str(exc)) from exc


__all__ = [
    "LiveObjects",
    "ServedEndpoint",
    "check_recipe_objects",
    "check_root_start",
    "close_runtimes",
    "default_start",
    "harness_models",
    "harness_served_model",
    "scenario_components",
    "weight_component_name",
]
