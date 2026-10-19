"""One scenario served and evolved by several recipes, one per release component.

A composite recipe binds a recipe to each component name. Its surface
serves every component's capabilities, its base artifact keeps each
component's seed under that component's directory, and its trainers are
the components' trainers, run as independent workers that meet at the
scenario's commit boundary (see ``docs/advanced_topics/state-model.rst``).

Configured by a dotted ``implementation`` whose config carries a
``components`` object. Each component has the shape of a file's ``recipe``
section, without ``runtime``: ``implementation`` names its recipe class and
``config`` holds its fields, sections and pipeline stages. Each component
uses the deployment's model:

.. code:: yaml

   recipe:
     implementation: reef.recipe.composite:CompositeRecipe
     config:
       components:
         weights:
           implementation: recipes.sao.recipe:SAORecipe
           config:
             pipeline: {task2batch: {config: {batch_size: 8}}}
         harness:
           implementation: reef.recipe.cordis:CordisRecipe
           config:
             evolution: {adapter: pi, ...}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from reef.core.components import validate_component_name
from reef.core.reports import ReportBase, ReportValidationError
from reef.inference.model_config import ModelConfig
from reef.observability import ExperimentLogger
from reef.recipe.base import Recipe, ServedEndpoint, WeightTrainingRecipe, every_check
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.recipe.config import recipe_config_from_mapping, recipe_config_from_section
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime
from reef.storage.records import RecordStore
from reef.storage.tasks import SplitSettings, TaskStorage, TaskStore
from reef.surface.base import ComponentSurface, HarnessInfo, Surface
from reef.train.trainer import ComponentTrainer

#: The checkpoint cadence key a composite refuses: every one of its steps checkpoints.
CHECKPOINT_CADENCE_KEY = "checkpoint_every_n_versions"
#: Metric prefixes experiment tracking keeps for itself; a component's metrics are prefixed with its name.
RESERVED_COMPONENT_NAMES = ("train", "step", "reef", "operations")
#: The sections a component keeps under its own ``config``; a file keeps them at its top level.
COMPONENT_FRAMEWORK_SECTIONS = ("pipeline", "evaluation")


class AnyComponentReport(ReportBase):
    """A report contract a payload meets when any of ``report_types`` parses it; the refusal names each."""

    report_types: ClassVar[tuple[type[ReportBase], ...]] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReportBase:
        refusals = []
        for report_type in cls.report_types:
            try:
                return report_type.from_dict(payload)
            except ReportValidationError as exc:
                refusals.append(f"{report_type.__name__}: {exc}")
        raise ReportValidationError("no component accepts this report: " + "; ".join(refusals))


def any_component_report(report_types: tuple[type[ReportBase], ...]) -> type[ReportBase]:
    """The contract a payload meets when any of ``report_types`` parses it, named after them."""
    name = "AnyOf" + "".join(item.__name__ for item in report_types)
    return type(name, (AnyComponentReport,), {"report_types": report_types})


def configured_training_mode(config: Mapping[str, Any]) -> str | None:
    """The ``data.training_mode`` a config names, in either spelling the field parser accepts; ``None`` unset."""
    data = config.get("data")
    if not isinstance(data, Mapping):
        return None
    for key in ("training_mode", "training-mode"):
        if data.get(key) is not None:
            return str(data[key])
    return None


def component_recipe_config(component: str, section: Any) -> dict[str, Any]:
    """The flat recipe config of ``components.<component>``, read the way a file's ``recipe`` section is read.

    A component has no runtime of its own: every component shares the deployment's runtime.
    """
    path = f"components.{component}"
    settings = recipe_config_from_section(section, path, COMPONENT_FRAMEWORK_SECTIONS)
    if "runtime" in settings:
        raise RecipeConfigError(
            f"{path}.runtime is not accepted: every component shares the deployment's runtime, set by recipe.runtime"
        )
    return settings


class ComponentTaskStorage(TaskStorage):
    """Opens the task stores of one component under ``<component>/<name>``, so components never share a store."""

    def __init__(self, storage: TaskStorage, component: str) -> None:
        self.storage = storage
        self.component = component

    def open(self, name: str, *, split: SplitSettings) -> TaskStore:
        return self.storage.open(f"{self.component}/{name}", split=split)


@dataclass(frozen=True, kw_only=True)
class CompositeRecipe(Recipe):
    """Bind one recipe per release component; the scenario runs their trainers side by side."""

    components: Mapping[str, Recipe] = field(default_factory=dict)
    name: str = "composite"
    #: A deployment file keeps ``components`` under ``recipe.config``.
    config_sections: ClassVar[tuple[str, ...]] = ("components",)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.components, Mapping) or len(self.components) < 2:
            raise RecipeConfigError("a composite recipe binds at least two components")
        for component, recipe in self.components.items():
            validate_component_name(component)
            if component in RESERVED_COMPONENT_NAMES:
                # Experiment tracking prefixes a component's metrics with its name; these prefixes are its own.
                raise RecipeConfigError(f"component {component!r} is a metric prefix experiment tracking reserves")
            if not isinstance(recipe, Recipe):
                raise RecipeConfigError(f"component {component!r} must be a Recipe")
            if isinstance(recipe, CompositeRecipe):
                raise RecipeConfigError(f"component {component!r} must not itself be a composite recipe")
            if recipe.training_mode != self.training_mode:
                raise RecipeConfigError(
                    f"component {component!r} runs training_mode {recipe.training_mode!r}, "
                    f"the composite recipe {self.training_mode!r}: every component shares one mode"
                )
        weight_components = [
            name for name, recipe in self.components.items() if isinstance(recipe, WeightTrainingRecipe)
        ]
        if len(weight_components) > 1:
            raise RecipeConfigError(f"components {weight_components} all train weights; a release loads one component")
        object.__setattr__(self, "components", dict(self.components))
        # A composed release names every component, so every step checkpoints.
        object.__setattr__(self, "checkpoint_strategy", EveryNVersions(1))

    @classmethod
    def from_resolved_config(
        cls,
        config: Mapping[str, Any],
        field_values: Mapping[str, Any],
        *,
        environ: Mapping[str, str],
        runtime: InferenceRuntime | None = None,
        training_runtime: TrainingRuntime | None = None,
    ) -> CompositeRecipe:
        raw = config.get("components")
        if not isinstance(raw, Mapping) or not raw:
            raise RecipeConfigError("a composite recipe config requires a non-empty 'components' object")
        # Every step of a composed scenario checkpoints, so a cadence setting is a mistake, not a choice.
        # A component's config refuses the key as an unknown field.
        cls.refuse_checkpoint_cadence(config, "recipe")
        for component in raw:
            if component in RESERVED_COMPONENT_NAMES:
                raise RecipeConfigError(
                    f"components.{component} is a metric prefix experiment tracking reserves "
                    f"({', '.join(RESERVED_COMPONENT_NAMES)}); rename the component"
                )
        # Resolve the deployment's runtime once, so the composite and every component share it.
        runtime = cls._resolve_runtime(environ, runtime)
        configured_mode = configured_training_mode(config)
        components: dict[str, Recipe] = {}
        for component, section in raw.items():
            settings = component_recipe_config(str(component), section)
            settings["model"] = dict(config.get("model", {}))
            if configured_mode is not None and configured_training_mode(settings) is None:
                # The composite's mode is every component's mode: a component sets its own only to disagree.
                settings["data"]["training_mode"] = configured_mode
            settings = recipe_config_from_mapping(settings)
            components[str(component)] = build_recipe(
                settings["implementation"],
                environ,
                config=settings,
                runtime=runtime,
                training_runtime=training_runtime,
            )
        modes = {recipe.training_mode for recipe in components.values()}
        if len(modes) != 1:
            raise RecipeConfigError(f"components run different training modes {sorted(modes)}; they share one")
        values = dict(field_values)
        if configured_mode is None:
            values["training_mode"] = next(iter(modes))
        try:
            return cls(components=components, runtime=runtime, training_runtime=training_runtime, **values)
        except ValueError as exc:
            raise RecipeConfigError(f"invalid {cls.__name__} configuration: {exc}") from exc

    @classmethod
    def select_weight_training(
        cls, config: Mapping[str, Any]
    ) -> tuple[type[WeightTrainingRecipe], Mapping[str, Any]] | None:
        """The one component that trains weights, so the deployment connects its training runtime.

        A release loads one component into a runtime, so two weight training
        components are refused here, before any runtime is connected. The
        selected component's config is returned flat, as its recipe reads it.
        """
        components = config.get("components")
        if not isinstance(components, Mapping):
            return None
        selected: list[tuple[str, tuple[type[WeightTrainingRecipe], Mapping[str, Any]]]] = []
        for name, section in components.items():
            settings = component_recipe_config(str(name), section)
            recipe_type = recipe_class_for(settings["implementation"])
            weight_training = None if recipe_type is None else recipe_type.select_weight_training(settings)
            if weight_training is not None:
                selected.append((str(name), weight_training))
        if len(selected) > 1:
            raise RecipeConfigError(
                f"components {[name for name, _ in selected]} all train weights; a release loads one component"
            )
        return selected[0][1] if selected else None

    @staticmethod
    def refuse_checkpoint_cadence(config: Mapping[str, Any], section: str) -> None:
        """Refuse the cadence key in every spelling it is written in: flat, under artifact, hyphenated."""
        artifact = config.get("artifact", {})
        candidates = [
            (section, key)
            for key in config
            if isinstance(key, str) and key.replace("-", "_") == CHECKPOINT_CADENCE_KEY
        ]
        if isinstance(artifact, Mapping):
            candidates.extend(
                (f"{section}.artifact", key)
                for key in artifact
                if isinstance(key, str) and key.replace("-", "_") == CHECKPOINT_CADENCE_KEY
            )
        if candidates:
            where, key = candidates[0]
            raise RecipeConfigError(f"{where}.{key} has no effect: every step of a composite recipe checkpoints")

    def with_served_endpoint(self, endpoint: ServedEndpoint) -> CompositeRecipe:
        # Each component's evaluation runs a candidate of that component alone: the endpoint names it.
        return replace(
            self,
            components={
                name: recipe.with_served_endpoint(replace(endpoint, component=name))
                for name, recipe in self.components.items()
            },
        )

    def with_model_config(self, config: ModelConfig) -> CompositeRecipe:
        super().with_model_config(config)
        return replace(
            self, components={name: recipe.with_model_config(config) for name, recipe in self.components.items()}
        )

    @property
    def report_type(self) -> type[ReportBase] | None:
        """The report contract ingress checks: a report any component accepts; ``None`` keeps ingress open.

        Each trainer parses reports with its own component's contract, so a
        weights recipe with a strict report (a rollout addressed into a grid)
        and a harness recipe with a plain scored one share a scenario: ingress
        refuses only a report no component would take.
        """
        return self.report_contract

    @cached_property
    def report_contract(self) -> type[ReportBase] | None:
        """The contract ``report_type`` answers, built once: the one the components share, or what any accepts."""
        declared: list[type[ReportBase]] = []
        for recipe in self.components.values():
            report_type = recipe.report_type
            if report_type is not None and report_type not in declared:
                declared.append(report_type)
        if not declared:
            return None
        if len(declared) == 1:
            return declared[0]
        return any_component_report(tuple(declared))

    def build_surface(self, scenario: str) -> Surface:
        """Every component's serving capabilities under its own name; one component may carry harness info."""
        components: dict[str, ComponentSurface] = {}
        harness: HarnessInfo | None = None
        for component, recipe in self.components.items():
            surface = recipe.serving_surface(scenario)
            if len(surface.components) > 1:
                raise RecipeConfigError(f"component {component!r} serves several components of its own")
            served = next(iter(surface.components.values()), ComponentSurface())
            # A recipe's own release check (a recipe serving no component admits its whole release with one) admits
            # this component here: inside the composite, the component is that recipe's release.
            checks = every_check(surface.validator, served.validator)
            components[component] = served if checks is served.validator else replace(served, validator=checks)
            if surface.harness is not None:
                if harness is not None:
                    raise RecipeConfigError("only one component may serve harness information")
                harness = surface.harness
        return Surface(components=components, harness=harness)

    def build_trainers(
        self,
        scenario: str,
        records: RecordStore,
        *,
        surface: Surface,
        algorithm_states: Mapping[str, Mapping[str, Any] | None],
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
    ) -> tuple[ComponentTrainer, ...]:
        trainers = []
        for component, recipe in self.components.items():
            if type(recipe).build is Recipe.build and recipe.training_mode != "auto":
                # A component that runs no step has no mode to run in: its bare processor takes records in auto.
                recipe = replace(recipe, training_mode="auto")
            trainer = recipe.build(
                scenario,
                records,
                algorithm_state=algorithm_states.get(component),
                experiment_logger=experiment_logger,
                task_storage=None if task_storage is None else ComponentTaskStorage(task_storage, component),
            )
            # Ingress admits what any component accepts; each trainer then releases the reports shaped for another.
            trainer.admit_reports_of(self.report_type)
            trainers.append(ComponentTrainer(component, trainer))
        return tuple(trainers)

    def base_artifact_files(self) -> Mapping[str, str] | None:
        """Each component's seed under that component's directory; ``None`` when no component seeds one."""
        files: dict[str, str] = {}
        for component, recipe in self.components.items():
            seed = recipe.base_artifact_files()
            if seed:
                files.update({f"{component}/{path}": text for path, text in seed.items()})
        return files or None

    def bootstrap_artifact_component(self) -> str | None:
        """The weight-training component: a bootstrap model snapshot is that component's base content."""
        return next(
            (name for name, recipe in self.components.items() if isinstance(recipe, WeightTrainingRecipe)), None
        )

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        return tuple(path for recipe in self.components.values() for path in recipe.scenario_state_dirs(scenario))

    def serving_status(self) -> Mapping[str, Any] | None:
        statuses = {component: recipe.serving_status() for component, recipe in self.components.items()}
        status = {component: value for component, value in statuses.items() if value is not None}
        return status or None


__all__ = ["CompositeRecipe"]
