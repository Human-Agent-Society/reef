"""One scenario served and evolved by several recipes, one per snapshot component.

A composite recipe binds a recipe to each component name. It declares
each recipe's component under that component's name. The scenario's default
start keeps each component's seed under that component's directory
(``reef.scenario.live_objects.default_start``), and its trainers are
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
from reef.core.components import validate_component_name
from reef.core.reports import ReportBase, ReportValidationError
from reef.recipe.base import Recipe, WeightTrainingRecipe, every_check
from reef.recipe.config import recipe_config_from_section
from reef.recipe.errors import RecipeConfigError

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

    Values: ``components`` maps each component name to its recipe value; the composite holds child recipe values
    and no live object. The loader makes the component recipes from the ``components`` section
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

    @property
    def harness_adapter(self) -> str | None:
        """The adapter of the component recipe that serves a harness tree; a composite serves at most one."""
        return next((recipe.harness_adapter for recipe in self.components.values() if recipe.harness_adapter), None)

    def harness_state_dir(self, scenario: str, component: str) -> Path | None:
        recipe = self.components.get(component)
        return None if recipe is None else recipe.harness_state_dir(scenario, component)

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
        """Each recipe's one declared component under the name the composite binds it to."""
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

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        return tuple(path for recipe in self.components.values() for path in recipe.scenario_state_dirs(scenario))


__all__ = ["CompositeRecipe"]
