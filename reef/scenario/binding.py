"""Capabilities bound immutably to one scenario."""

from __future__ import annotations

from dataclasses import dataclass

from reef.artifact.artifact import Artifact
from reef.artifact.components import ScenarioComponents
from reef.core.activation import ActivationRule
from reef.core.reports import ReportBase
from reef.inference.interfaces import InferenceRuntime
from reef.storage.served_models import ServedModels
from reef.train.component import TrainingComponent


@dataclass(frozen=True)
class ScenarioBinding:
    """Runtime-facing capabilities selected by a recipe at construction."""

    #: The components every release binds, with each one's admission check, run before that component is
    #: published or restored.
    components: ScenarioComponents
    runtime: InferenceRuntime | None
    #: The deployment's rule that decides whether an artifact that training publishes serves at once
    #: (``ActivationRule``). The scenario registry replaces it with the scenario's own rule when the scenario has one
    #: (``Scenario.set_activation_rule``).
    activation_rule: ActivationRule
    #: The report contract selected by the recipe while building its trainer;
    #: when set, every report on this scenario is parsed through it at
    #: ingress. ``None`` keeps open ingress.
    report_type: type[ReportBase] | None = None
    #: Whether the record import and the data items refuse image content (``Recipe.refuses_images``).
    refuses_images: bool = False
    #: The training component of a weight-training recipe, which its trainer submits jobs to.
    training_component: TrainingComponent | None = None
    #: Standalone scenarios activate their runtime directly. Service scenarios leave activation to inference.
    activates_runtime: bool = True
    #: The actual serving routes, or None for a standalone scenario that activates its own runtime.
    served_models: ServedModels | None = None


def activate_release(
    components: ScenarioComponents,
    artifact: Artifact,
    runtime: InferenceRuntime | None,
    scenario: str,
    *,
    previous: Artifact | None = None,
) -> None:
    """Load the release ``artifact`` of ``scenario`` into ``runtime`` before traffic routes to it.

    Only the loaded component (``ScenarioComponents.loaded_component``) loads. Nothing loads without a runtime or
    such a component, or when ``previous``, the release served before, already selects the same artifact for it.
    """
    name = components.loaded_component
    if name is None or runtime is None:
        return
    if previous is not None and not components.component_changed(artifact, previous, name):
        return
    runtime.activate_checkpoint(
        components.component_artifact(artifact, name),
        scenario=scenario if runtime.per_scenario_adapters else None,
    )
