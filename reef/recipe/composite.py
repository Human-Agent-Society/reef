"""One scenario served and evolved by several recipes, one per release component.

A composite recipe binds a recipe to each component name. Its surface
serves every component's capabilities, its base artifact keeps each
component's seed under that component's directory, and its trainers are
the components' trainers, run as independent workers that meet at the
scenario's commit boundary (see ``docs/advanced_topics/state-model.rst``).

Configured by a dotted ``implementation`` whose config carries a
``components`` object, one recipe config per component name; each inherits
the deployment's ``model`` unless it names its own:

.. code:: yaml

   implementation: reef.recipe.composite:CompositeRecipe
   model:
     path: qwen3-8b
   components:
     weights:
       implementation: recipes.sao.recipe:SAORecipe
       data: {batch_size: 8}
     harness:
       implementation: reef.recipe.cordis:CordisRecipe
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
from reef.recipe.base import Recipe, ServedEndpoint, WeightTrainingRecipe
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.recipe.config import recipe_config_from_mapping
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import recipe_class_for
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime
from reef.storage.records import RecordStore
from reef.surface.base import ComponentSurface, HarnessInfo, Surface
from reef.train.trainer import ComponentTrainer

#: The checkpoint cadence key a composite refuses: every one of its steps checkpoints.
CHECKPOINT_CADENCE_KEY = "checkpoint_every_n_versions"
#: Metric prefixes experiment tracking keeps for itself; a component's metrics are prefixed with its name.
RESERVED_COMPONENT_NAMES = ("train", "step", "reef", "operations")


def any_component_report(report_types: tuple[type[ReportBase], ...]) -> type[ReportBase]:
    """A report contract a payload meets when any of ``report_types`` parses it; the refusal names each."""

    class ComponentReport(ReportBase):
        @classmethod
        def from_dict(cls, payload: Mapping[str, Any]) -> ReportBase:
            refusals = []
            for report_type in report_types:
                try:
                    return report_type.from_dict(payload)
                except ReportValidationError as exc:
                    refusals.append(f"{report_type.__name__}: {exc}")
            raise ReportValidationError("no component accepts this report: " + "; ".join(refusals))

    ComponentReport.__name__ = ComponentReport.__qualname__ = "AnyOf" + "".join(item.__name__ for item in report_types)
    return ComponentReport


@dataclass(frozen=True, kw_only=True)
class CompositeRecipe(Recipe):
    """Bind one recipe per release component; the scenario runs their trainers side by side."""

    components: Mapping[str, Recipe] = field(default_factory=dict)
    name: str = "composite"
    #: The versioned deployment layout keeps ``components`` under ``recipe.config``.
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
        cls._refuse_checkpoint_cadence(config, "recipe")
        for component in raw:
            if component in RESERVED_COMPONENT_NAMES:
                raise RecipeConfigError(
                    f"components.{component} is a metric prefix experiment tracking reserves "
                    f"({', '.join(RESERVED_COMPONENT_NAMES)}); rename the component"
                )
        # Resolve the deployment's runtime once, so the composite and every component share it.
        runtime = cls._resolve_runtime(environ, runtime)
        components: dict[str, Recipe] = {}
        for component, component_config in raw.items():
            if not isinstance(component_config, Mapping):
                raise RecipeConfigError(f"components.{component} must be an object")
            cls._refuse_checkpoint_cadence(component_config, f"components.{component}")
            merged = dict(component_config)
            merged.setdefault("model", dict(config.get("model", {})))
            settings = recipe_config_from_mapping(merged)
            recipe_class = recipe_class_for(settings["implementation"])
            if recipe_class is None:
                raise RecipeConfigError(
                    f"components.{component}.implementation must be 'recipe' or a dotted recipe class"
                )
            components[str(component)] = recipe_class.from_environment(
                environ, config=settings, runtime=runtime, training_runtime=training_runtime
            )
        modes = {recipe.training_mode for recipe in components.values()}
        if len(modes) != 1:
            raise RecipeConfigError(f"components run different training modes {sorted(modes)}; they share one")
        values = dict(field_values)
        configured_mode = config.get("data", {}).get("training_mode")
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
        components are refused here, before any runtime is connected.
        """
        components = config.get("components")
        if not isinstance(components, Mapping):
            return None
        selected: list[tuple[str, tuple[type[WeightTrainingRecipe], Mapping[str, Any]]]] = []
        for name, component_config in components.items():
            if not isinstance(component_config, Mapping):
                continue
            implementation = component_config.get("implementation")
            recipe_type = recipe_class_for(implementation) if isinstance(implementation, str) else None
            if recipe_type is None:
                continue
            weight_training = recipe_type.select_weight_training(component_config)
            if weight_training is not None:
                selected.append((str(name), weight_training))
        if len(selected) > 1:
            raise RecipeConfigError(
                f"components {[name for name, _ in selected]} all train weights; a release loads one component"
            )
        return selected[0][1] if selected else None

    @staticmethod
    def _refuse_checkpoint_cadence(config: Mapping[str, Any], section: str) -> None:
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
        return replace(
            self, components={name: recipe.with_served_endpoint(endpoint) for name, recipe in self.components.items()}
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
            surface = recipe.build_surface(scenario)
            if len(surface.components) > 1:
                raise RecipeConfigError(f"component {component!r} serves several components of its own")
            components[component] = next(iter(surface.components.values()), ComponentSurface())
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
    ) -> tuple[ComponentTrainer, ...]:
        return tuple(
            ComponentTrainer(
                component,
                recipe.build(
                    scenario,
                    records,
                    algorithm_state=algorithm_states.get(component),
                    experiment_logger=experiment_logger,
                ),
            )
            for component, recipe in self.components.items()
        )

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
        status = {
            component: recipe.serving_status()
            for component, recipe in self.components.items()
            if recipe.serving_status() is not None
        }
        return status or None


__all__ = ["CompositeRecipe"]
