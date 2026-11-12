"""One scenario served and evolved by several recipes, one per snapshot component.

A composite recipe binds a recipe to each component name. It declares
each recipe's component under that component's name, its default start keeps each
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
             pipeline: {sample2job: {config: {job_samples: 8, buffer_size: 32}}}
         harness:
           implementation: reef.recipe.harness:HarnessEvolutionRecipe
           config:
             evolution: {adapter: pi, ...}
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from reef.artifact.components import ComponentSpec, ScenarioComponents
from reef.artifact.repository import DefaultStart
from reef.core.components import validate_component_name
from reef.core.reports import ReportBase, ReportValidationError
from reef.harness.tasks.model_binding import ModelBinding
from reef.recipe.base import Recipe, ServedEndpoint, WeightTrainingRecipe, check_job_components, every_check
from reef.recipe.config import recipe_config_from_section
from reef.recipe.errors import RecipeConfigError
from reef.storage.data import DataStore
from reef.train.component import ScenarioRoot, TrainingComponent, component_root

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


@dataclass(frozen=True, kw_only=True)
class CompositeRecipe(Recipe):
    """Bind one recipe per snapshot component; the scenario runs their trainers side by side.

    The loader makes the component recipes from the ``components`` section
    (``reef.recipe.loading.load_composite_recipe``).
    """

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
            raise RecipeConfigError(
                f"components {weight_components} all train weights; a snapshot loads one component"
            )
        object.__setattr__(self, "components", dict(self.components))

    def with_served_endpoint(self, endpoint: ServedEndpoint) -> CompositeRecipe:
        # Each component's evaluation runs a candidate of that component alone: the endpoint names it.
        return replace(
            self,
            components={
                name: recipe.with_served_endpoint(replace(endpoint, component=name))
                for name, recipe in self.components.items()
            },
        )

    @property
    def harness_adapter(self) -> str | None:
        """The adapter of the component recipe that serves a harness tree; a composite serves at most one."""
        return next((recipe.harness_adapter for recipe in self.components.values() if recipe.harness_adapter), None)

    def harness_served_model(self, scenario: str, component: str) -> ModelBinding | None:
        recipe = self.components.get(component)
        return None if recipe is None else recipe.harness_served_model(scenario, component)

    def harness_state_dir(self, scenario: str, component: str) -> Path | None:
        recipe = self.components.get(component)
        return None if recipe is None else recipe.harness_state_dir(scenario, component)

    def job_components(self) -> tuple[TrainingComponent, ...]:
        """The job components of every component recipe, each once: the components share the composite's
        training component. Raise ``ValueError`` when two different components share a name."""
        found: dict[str, TrainingComponent] = {}
        for recipe in self.components.values():
            for component in recipe.job_components():
                known = found.setdefault(component.name, component)
                if known is not component:
                    raise ValueError(f"two job components of this composite are named {component.name!r}")
        return tuple(found.values())

    def with_job_components(
        self, components: Mapping[str, TrainingComponent], data_store: DataStore
    ) -> CompositeRecipe:
        check_job_components(self, components)
        return replace(
            self,
            training_component=(None if self.training_component is None else components[self.training_component.name]),
            data_store=data_store,
            components={
                name: recipe.with_job_components(
                    {component.name: components[component.name] for component in recipe.job_components()}, data_store
                )
                for name, recipe in self.components.items()
            },
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

    def build_components(self, scenario: str) -> ScenarioComponents:
        """Each recipe's one component under the name the composite binds it to."""
        specs: dict[str, ComponentSpec] = {}
        for component, recipe in self.components.items():
            declared = recipe.build_components(scenario)
            if len(declared.components) > 1:
                raise RecipeConfigError(f"component {component!r} serves several components of its own")
            served = next(iter(declared.components.values()), ComponentSpec())
            # A recipe's own snapshot check (a recipe serving no component admits its whole snapshot with one) admits
            # this component here: inside the composite, the component is that recipe's snapshot.
            checks = every_check(declared.validator, served.validator)
            specs[component] = served if checks is served.validator else replace(served, validator=checks)
        return ScenarioComponents(specs)

    def for_record2sample_configs(
        self, components: ScenarioComponents, configs: Mapping[str, Mapping[str, Any]]
    ) -> CompositeRecipe:
        """This composite with the recipe of each component that ``configs`` names given those keys over its
        record-to-sample config (``Recipe.with_record2sample_config``)."""
        if not configs:
            return self
        return replace(
            self,
            components={
                component: recipe.with_record2sample_config(configs.get(component, {}))
                for component, recipe in self.components.items()
            },
        )

    def check_root_start(self, root: ScenarioRoot) -> None:
        """Check every component's view of ``root`` with its recipe; propagate any ``ValueError`` refusal."""
        for component, recipe in self.components.items():
            recipe.check_root_start(component_root(root, component))

    def default_start(self) -> DefaultStart:
        """Each component's default start under that component's directory.

        The files and the model files of a component go under its name; each input is named ``<component>.<input>``.
        Raise ``RecipeConfigError`` when two components hold model files.
        """
        sources = {component: recipe.default_start() for component, recipe in self.components.items()}
        models = [component for component, source in sources.items() if source.model_files is not None]
        if len(models) > 1:
            raise RecipeConfigError(f"components {models} each hold a base model; a composite trains one model")
        return DefaultStart(
            files={
                f"{component}/{path}": text
                for component, source in sources.items()
                for path, text in source.files.items()
            },
            model_files=sources[models[0]].model_files if models else None,
            model_component=models[0] if models else None,
            inputs={
                f"{component}.{name}": value
                for component, source in sources.items()
                for name, value in source.inputs.items()
            },
        )

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        return tuple(path for recipe in self.components.values() for path in recipe.scenario_state_dirs(scenario))


__all__ = ["CompositeRecipe"]
