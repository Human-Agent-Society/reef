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
from pathlib import Path

from reef.artifact.components import ComponentChecks, ComponentSpec, ScenarioComponents
from reef.artifact.repository import DefaultStart
from reef.core.errors import ConfigError
from reef.harness.tasks.harness_models import HarnessModels
from reef.harness.tasks.model_binding import ModelBinding
from reef.inference.interfaces import InferenceRuntime
from reef.recipe.base import WEIGHTS_COMPONENT, Recipe, WeightTrainingRecipe, every_check
from reef.recipe.composite import CompositeRecipe, component_recipes
from reef.recipe.deployment import harness_recipe
from reef.recipe.errors import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.storage.data import DataStore
from reef.train.component import (
    ScenarioRoot,
    TrainingArtifactLayout,
    TrainingBackendFacts,
    TrainingComponent,
    component_root,
)


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
    - ``training_facts`` are the facts of the training backend of the recipe that trains weights, as its
      configuration states them (``reef.train.component.training_backend_facts``), or ``None`` when no recipe of
      the deployment trains weights.
    - ``harness_facts`` are the facts of the deployment's harness recipe (``HarnessFacts``), or ``None`` when the
      deployment evolves no harness.
    - ``training_component`` and ``harness_component`` are the components of those two backends, or ``None``. In
      the value that the service assembly builds, a component is the one that runs the jobs, and only a process
      that runs the training service has it. In the value that the dispatcher gives the scenarios, a component is
      the jobs queue's view of the backend (``reef.train.job_queue.JobQueueComponent``), which every process that
      holds the facts has.
    - ``data_store`` keeps the data directory of each training job, or is ``None`` for no store.
    - ``served_endpoint`` is where this Reef answers inference itself, or ``None`` when a harness run calls the
      runtime's own endpoint.

    A value with a component has the facts of the same backend: the two have the same name. The facts write the
    default start of a new scenario and check a root and an import (``default_start``, ``check_root_start``,
    ``scenario_components``). The two components, when both are set, have different names: a job names its
    component by that name.

    Purpose: the service assembly builds one value for each process (``reef.service.assembly``). The dispatcher
    gets it, runs the jobs of its components when the process is a training service, and gives the scenario factory
    a copy whose components are the jobs queue's views. The functions of this module read it together with the
    recipe. It holds exactly these objects; nothing else is looked up in it.
    """

    runtime: InferenceRuntime | None = None
    training_component: TrainingComponent | None = None
    training_facts: TrainingBackendFacts | None = None
    harness_component: TrainingComponent | None = None
    harness_facts: TrainingBackendFacts | None = None
    data_store: DataStore | None = None
    served_endpoint: ServedEndpoint | None = None

    def __post_init__(self) -> None:
        if self.runtime is not None and not isinstance(self.runtime, InferenceRuntime):
            raise TypeError(f"runtime must be an InferenceRuntime, got {type(self.runtime).__name__}")
        for name, component, facts in (
            ("training", self.training_component, self.training_facts),
            ("harness", self.harness_component, self.harness_facts),
        ):
            if component is not None and not isinstance(component, TrainingComponent):
                raise TypeError(f"{name}_component must be a TrainingComponent, got {type(component).__name__}")
            if facts is not None and not isinstance(facts, TrainingBackendFacts):
                raise TypeError(f"{name}_facts must be TrainingBackendFacts, got {type(facts).__name__}")
            if component is None:
                continue
            if facts is None:
                raise ValueError(f"a process that holds the {name} component holds the facts of its backend too")
            if component.name != facts.name:
                raise ValueError(
                    f"the {name} component {component.name!r} and the facts of the backend {facts.name!r} are of "
                    "two backends"
                )
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
        """The components of the value, each under its own ``name``: the training component, then the harness
        component, without the ones that are ``None``. In the service assembly's value they are the components that
        the process built; in a dispatcher's value they are the jobs queue's views."""
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
    # A composite binds at most one weight component.
    for component, child in component_recipes(recipe):
        if isinstance(child, WeightTrainingRecipe):
            return WEIGHTS_COMPONENT if component is None else component
    return None


def check_recipe_objects(recipe: Recipe, live: LiveObjects) -> None:
    """Check that ``live`` has the objects that ``recipe`` needs.

    A recipe that trains weights, alone or as a component of a composite, needs a runtime and the facts of its
    training backend (``LiveObjects.training_facts``). A weight recipe whose jobs are task jobs
    (``WeightTrainingRecipe.trains_task_jobs``) needs a training component that takes the recipe's task parameters
    (``TrainingComponent.check_task_jobs``); a process that holds the facts and no component runs no job and leaves
    that check to the training service, which holds the component. Raise ``RecipeConfigError`` otherwise.
    """
    for _, one in component_recipes(recipe):
        if not isinstance(one, WeightTrainingRecipe):
            continue
        if live.training_facts is None:
            raise RecipeConfigError(f"{type(one).__name__} requires the facts of its training backend")
        if live.runtime is None:
            raise RecipeConfigError("weight recipes require an independent inference runtime")
        if live.training_component is not None and type(one).trains_task_jobs:
            try:
                live.training_component.check_task_jobs(one.parameters_by_kind()["task"])
            except ValueError as exc:
                raise RecipeConfigError(
                    f"{type(one).__name__} trains task jobs, whose recipe.config.job_parameters name the "
                    f"objective, the rollouts of each task and the agent that runs them: {exc}"
                ) from exc


def declared_components(recipe: Recipe, scenario: str) -> ScenarioComponents:
    """The components that every snapshot of ``scenario`` binds, as the configuration of ``recipe`` declares them.

    A recipe other than a composite gives its own declaration (``Recipe.build_components``). A composite gives
    each child's one declared component under the name that the composite binds the child to; a child that
    declares no component gives an empty ``ComponentSpec``. Raise ``RecipeConfigError`` for a child of a composite
    that declares several components.
    """
    if not isinstance(recipe, CompositeRecipe):
        return recipe.build_components(scenario)
    specs: dict[str, ComponentSpec] = {}
    for component, child in recipe.components.items():
        declared = child.build_components(scenario)
        if len(declared.components) > 1:
            raise RecipeConfigError(f"component {component!r} serves several components of its own")
        served = next(iter(declared.components.values()), ComponentSpec())
        # A recipe's own snapshot check (a recipe serving no component admits its whole snapshot with one) admits
        # this component here: inside the composite, the component is that recipe's snapshot.
        checks = every_check(declared.validator, served.validator)
        specs[component] = served if checks is served.validator else replace(served, validator=checks)
    return ScenarioComponents(specs)


@dataclass(frozen=True)
class RecipeChecks(ComponentChecks):
    """The admission checks that ``recipe`` declares for each scenario (``declared_components``).

    Values: ``recipe`` is the deployment's recipe. ``validate(scenario, component, artifact)`` runs
    ``declared_components(recipe, scenario).validate(directory, component)``.

    Purpose: the training service gives it to each job worker, which checks every checkpoint with it before it
    stores the checkpoint (``reef.train.job_queue.TrainingJobWorker``).
    """

    recipe: Recipe

    def __post_init__(self) -> None:
        if not isinstance(self.recipe, Recipe):
            raise TypeError(f"recipe must be a Recipe, got {type(self.recipe).__name__}")

    def validate(self, scenario: str, component: str | None, directory: Path) -> None:
        declared_components(self.recipe, scenario).validate(directory, component)


def scenario_components(recipe: Recipe, live: LiveObjects, scenario: str) -> ScenarioComponents:
    """The components that every snapshot of ``scenario`` binds in this deployment.

    The result is what the recipe declares from its configuration (``declared_components``), with the facts of
    the weight component that only live objects know: its kind is the runtime's ``served_weight_kind``, and its
    layout is that of the training backend (``TrainingArtifactLayout``): the facts check an import's files, and
    the training component of ``live`` answers what they leave to the backend's service. A recipe that
    trains no weights gets its declaration unchanged. Raise ``RecipeConfigError`` when the recipe trains weights
    and ``live`` has no runtime or no facts of the training backend.
    """
    declared = declared_components(recipe, scenario)
    name = weight_component_name(recipe)
    if name is None:
        return declared
    if live.training_facts is None:
        raise RecipeConfigError(f"{type(recipe).__name__} requires the facts of its training backend")
    layout = TrainingArtifactLayout(live.training_facts, live.training_component)
    if live.runtime is None:
        raise RecipeConfigError("weight recipes require an independent inference runtime")
    weights = replace(declared.components[name], kind=live.runtime.served_weight_kind, layout=layout)
    return ScenarioComponents({**declared.components, name: weights}, declared.validator)


def default_start(recipe: Recipe, live: LiveObjects) -> DefaultStart:
    """The content of a new scenario's root snapshot when no artifact item starts the scenario.

    - A weight recipe starts from what the training backend's facts write from the configuration
      (``TrainingBackendFacts.default_start``): a base root or a reference to the base model. No training
      component makes it.
    - A composite starts from each component's default start under that component's directory. The files of a
      component go under its name, and each input is named ``<component>.<input>``.
    - Any other recipe starts from what its configuration gives (``Recipe.default_start``).

    Raise ``RecipeConfigError`` when the recipe trains weights and ``live`` has no facts of the training backend.
    """
    if isinstance(recipe, WeightTrainingRecipe):
        if live.training_facts is None:
            raise RecipeConfigError(f"{type(recipe).__name__} requires the facts of its training backend")
        return live.training_facts.default_start()
    if not isinstance(recipe, CompositeRecipe):
        return recipe.default_start()
    sources = {component: default_start(child, live) for component, child in recipe.components.items()}
    return DefaultStart(
        files={
            f"{component}/{path}": text for component, source in sources.items() for path, text in source.files.items()
        },
        inputs={
            f"{component}.{name}": value
            for component, source in sources.items()
            for name, value in source.inputs.items()
        },
    )


def check_root_start(recipe: Recipe, live: LiveObjects, root: ScenarioRoot) -> None:
    """Check that training can start from ``root`` before an item start is registered.

    Each recipe of the deployment (``component_recipes``) has its root checked by the facts of its backend
    (``TrainingBackendFacts.check_root_start``), which write nothing: ``training_facts`` for a weight recipe and
    ``harness_facts`` for a harness recipe. A recipe of a composite has its own component's view of the root
    checked (``component_root``); a recipe on its own has ``root`` checked. Nothing is checked for a recipe that
    submits no job, and for one whose facts ``live`` does not hold. Raise ``ValueError`` on refusal.
    """
    for component, child in component_recipes(recipe):
        view = root if component is None else component_root(root, component)
        if isinstance(child, WeightTrainingRecipe) and live.training_facts is not None:
            live.training_facts.check_root_start(view)
        if isinstance(child, HarnessEvolutionRecipe) and live.harness_facts is not None:
            live.harness_facts.check_root_start(view)


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
    harness = harness_recipe(recipe)
    if harness is None or harness[0] not in (None, component):
        return None
    try:
        return harness_models(harness[1], live, component=harness[0]).served(scenario)
    except ConfigError as exc:
        raise RecipeConfigError(str(exc)) from exc


__all__ = [
    "LiveObjects",
    "RecipeChecks",
    "ServedEndpoint",
    "check_recipe_objects",
    "check_root_start",
    "close_runtimes",
    "declared_components",
    "default_start",
    "harness_models",
    "harness_served_model",
    "scenario_components",
    "weight_component_name",
]
