"""The harness evolution recipe: one recipe for every harness evolution method.

``HarnessEvolutionRecipe`` reads the ``evolution`` config section and holds
it as configuration (RFC #1390). It builds nothing that runs: the service
builds the ``HarnessTrainingComponent``
(``reef.train.harness_backend.component``) from the recipe's settings, and
that component builds the method, the scorer and, per scenario component,
one ``HarnessBackend`` with the evaluator and the selection policy around
it. The scenario's trainer submits each job to the component, which runs it
in the training service. A method is a ``reef.train.harness_backend.Method``
that the deployment names in ``evolution.method``; a method needs no recipe
of its own. A recipe subclass is for a method whose deployment has other
defaults, such as its own seed tree::

    evolution:
      adapter: pi
      seed: seed/
      method: recipes.gepa:GEPA
      models: {reflection: {url: ..., model: ..., api_key_env: OPENAI_API_KEY}}
      evaluation:
        tasks: [...]
        scorer: harness.aime:AIMEAnswerScorer
        policy: compare
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

from reef.artifact.components import ArtifactLayout, ComponentSpec, ScenarioComponents
from reef.artifact.files import ArtifactFiles
from reef.artifact.repository import DefaultStart
from reef.core.activation import ActivationRule
from reef.core.config import Settings
from reef.core.errors import ConfigError
from reef.core.model_metadata import ModelMetadata
from reef.core.plugins import PluginReference
from reef.core.reports import ScoredRolloutReport
from reef.evaluation.activation import AlwaysActivate
from reef.evaluation.harness import EvaluationSettings
from reef.execution.executor.config import ExecutorSettings, role_executor_settings
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor, DescriptorError
from reef.harness.harbor.agent import can_run_in_task_container
from reef.harness.sync import HARNESS_COMPONENT, HarnessFiles
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tree import HarnessTree, validate_tree
from reef.recipe.base import Recipe
from reef.recipe.config_fields import ConfigReader, reader_field, section_field
from reef.recipe.errors import RecipeConfigError
from reef.record2sample.traces import TraceSamples
from reef.sample2job.traces import TraceJobData, TrainTaskJobData
from reef.train.harness_backend.settings import EvolutionSettings, HarnessSettings, configured_tasks
from reef.train.pipeline import PipelineSettings, PipelineSpec


@dataclass(frozen=True)
class HarnessLayout(ArtifactLayout):
    """The layout of a harness artifact for the adapter ``descriptor``, which an import must meet (RFC #980).

    A directory meets it when its files, read as UTF-8 text, form a
    ``HarnessTree`` that passes ``validate_tree`` for ``descriptor``: the
    files a snapshot of the harness component stores.
    Purpose: ``HarnessEvolutionRecipe.build_components`` binds it on the harness
    component; the committer checks an imported harness with it.
    """

    descriptor: AdapterDescriptor

    def check(self, files: ArtifactFiles) -> None:
        tree = HarnessTree({relative: files.read_bytes(relative).decode("utf-8") for relative in files.paths()})
        validation = validate_tree(tree, self.descriptor)
        if not validation.passed:
            raise ValueError(f"the harness tree fails validation: {validation.message()}")


def evolution_section(sections: Mapping[str, Settings]) -> EvolutionSettings:
    """The parsed ``evolution`` section among a harness recipe's ``sections``; ``TypeError`` without it."""
    evolution = sections.get("evolution")
    if not isinstance(evolution, EvolutionSettings):
        raise TypeError("a harness recipe's evolution field holds EvolutionSettings")
    return evolution


class SeedTree(ConfigReader):
    """Reads the seed tree: the files of the directory that ``evolution.seed`` names, the empty tree for none.

    Purpose: ``HarnessEvolutionRecipe.seed`` declares it.
    """

    def read(
        self, config: Mapping[str, Any], environ: Mapping[str, str], sections: Mapping[str, Settings]
    ) -> HarnessTree:
        seed = evolution_section(sections).seed
        try:
            return HarnessTree({}) if seed is None else HarnessTree.read(Path(seed).expanduser())
        except ValueError as exc:
            raise RecipeConfigError(f"evolution.seed: {exc}") from exc


class NamedModels(ConfigReader):
    """Reads the method's named models: one ``ModelBinding`` for each ``evolution.models`` entry, its key from the
    entry or from the environment variable that the entry names.

    Purpose: ``HarnessEvolutionRecipe.models`` declares it.
    """

    def read(
        self, config: Mapping[str, Any], environ: Mapping[str, str], sections: Mapping[str, Settings]
    ) -> dict[str, ModelBinding]:
        return {
            name: ModelBinding.from_config(section, environ, where=f"evolution.models.{name}")
            for name, section in evolution_section(sections).models.items()
        }


class NamedModelMetadata(ConfigReader):
    """Reads the model metadata: one ``ModelMetadata`` for each ``evolution.model_metadata`` entry, by model name.

    Purpose: ``HarnessEvolutionRecipe.model_metadata`` declares it.
    """

    def read(
        self, config: Mapping[str, Any], environ: Mapping[str, str], sections: Mapping[str, Settings]
    ) -> dict[str, ModelMetadata]:
        return {
            name: ModelMetadata.from_config(section)
            for name, section in evolution_section(sections).model_metadata.items()
        }


class RunWorkers(ConfigReader):
    """Reads the placement of the run workers: the executor that ``execution.evolution`` selects, by name from the
    ``executors`` profiles or inline.

    Purpose: ``HarnessEvolutionRecipe.worker_executor`` declares it.
    """

    def read(
        self, config: Mapping[str, Any], environ: Mapping[str, str], sections: Mapping[str, Settings]
    ) -> ExecutorSettings:
        return role_executor_settings(config, "evolution")


class ServedModelName(ConfigReader):
    """Reads the served model's name: ``model.path`` when it is a non-empty string, else ``None``.

    Purpose: ``HarnessEvolutionRecipe.model_name`` declares it.
    """

    def read(
        self, config: Mapping[str, Any], environ: Mapping[str, str], sections: Mapping[str, Settings]
    ) -> str | None:
        model = config.get("model")
        path = model.get("path") if isinstance(model, Mapping) else None
        return path if isinstance(path, str) and path else None


@dataclass(frozen=True)
class HarnessEvolutionRecipe(Recipe):
    """The configuration of harness evolution of one adapter's harness tree by one method.

    Values: ``evolution`` and ``evaluation`` are the parsed sections, which the loader reads (``section_field``);
    the loader reads the next five fields with their readers (``reader_field``).
    ``evolution`` and ``evaluation`` hold these settings:
    ``evolution.method`` names the deployment's ``Method``, and
    ``evaluation.scorer`` and ``evaluation.collector`` the scorer and the
    collector. ``seed`` is the first tree of a new scenario. ``models`` and
    ``model_metadata`` are the parsed ``evolution`` entries.
    ``worker_executor`` places the run workers (``execution.evolution``).
    ``model_name`` is ``model.path``, the served model's name when the
    runtime names none. The recipe builds no method, scorer, executor or
    agent: ``HarnessTrainingComponent`` builds them from ``harness_settings``.
    The recipe holds no runtime, no served endpoint and no training
    component: the process holds them (``reef.scenario.live_objects.LiveObjects``).
    The model under test is that runtime
    (``reef.scenario.live_objects.harness_models``), and the service builds
    the training component that runs the jobs
    (``reef.service.assembly.with_harness_components``).
    Purpose: a deployment names it as its recipe; the service loads it once
    from the deployment config, and the scenario factory's trainer builder
    makes each scenario's trainer, which submits each job to the process's
    harness component.
    """

    evolution: EvolutionSettings = section_field(
        "evolution", required="harness evolution requires an 'evolution' config section"
    )
    evaluation: EvaluationSettings = section_field(
        "evolution.evaluation", required="harness evolution requires an 'evolution' config section"
    )
    seed: HarnessTree = reader_field(SeedTree(), default_factory=lambda: HarnessTree({}))
    models: Mapping[str, ModelBinding] = reader_field(NamedModels(), default_factory=dict)
    model_metadata: Mapping[str, ModelMetadata] = reader_field(NamedModelMetadata(), default_factory=dict)
    worker_executor: ExecutorSettings = reader_field(RunWorkers(), default_factory=ExecutorSettings)
    model_name: str | None = reader_field(ServedModelName(), default=None)
    config_sections: ClassVar[tuple[str, ...]] = ("evolution",)

    name: str = field(default="harness_evolution", kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            descriptor = get_adapter(self.evolution.adapter)
        except DescriptorError as exc:
            raise RecipeConfigError(str(exc)) from exc
        # The activation section's review_kinds name parts of this tree; a name the tree lacks is refused here.
        self.activation_rule()
        seed_problems = validate_tree(self.seed, descriptor).problems
        if seed_problems:
            raise RecipeConfigError(f"evolution.seed fails validation: {'; '.join(seed_problems)}")
        offline = self.runs_offline()
        task_directories = self.task_directories(descriptor)
        if task_directories and not descriptor.takes_task_directory:
            # An adapter whose command takes text plays each Harbor task in the task's own container.
            if not can_run_in_task_container(descriptor):
                raise RecipeConfigError(
                    f"adapter {descriptor.name!r} cannot play Harbor task directories: Reef installs the agent in "
                    "each task's container with npm, uv or git, and the adapter declares no such install"
                )
            if self.evolution.executor != "local":
                raise RecipeConfigError(
                    "Harbor task directories run in their own containers, which are the runs' boundary: "
                    "set evolution.executor to local"
                )
            if offline and self.evaluation.tasks:
                raise RecipeConfigError(
                    "offline runs play task directories, so the evaluation tasks must be task directories "
                    "too: use evaluation.imported_tasks or evaluation.task_manifest instead of evaluation.tasks"
                )
        if self.evaluation.collect_failures and task_directories:
            raise RecipeConfigError("evaluation.collect_failures adds prompts, and these tasks are task directories")
        # A task manifest is read now, so a deployment with an unreadable one refuses to start.
        self.configured_tasks()

    def runs_offline(self) -> bool:
        """Whether the jobs roll out the imported train tasks (``TrainTaskJobData``)."""
        return issubclass(self.stages().sample2job.plugin_class, TrainTaskJobData)

    def harness_settings(self) -> HarnessSettings:
        """The configuration that the harness training component builds itself from: the two sections, the seed
        tree, the placement of the run workers, and whether the jobs are offline."""
        return HarnessSettings(
            evolution=self.evolution,
            evaluation=self.evaluation,
            seed=self.seed,
            worker_executor=self.worker_executor,
            runs_offline=self.runs_offline(),
        )

    def task_directories(self, descriptor: AdapterDescriptor) -> bool:
        """Whether the runs play Harbor task directories rather than prompts
        (``HarnessSettings.task_directories``)."""
        return self.harness_settings().task_directories(descriptor)

    def configured_tasks(self) -> tuple[str, ...]:
        """The configured evaluation tasks: ``evaluation.tasks``, or the eval tasks the task manifest names."""
        try:
            return configured_tasks(self.evaluation)
        except ConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

    @property
    def report_type(self) -> type[ScoredRolloutReport]:
        return ScoredRolloutReport

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2sample=TraceSamples, sample2job=TraceJobData)

    def stages(self) -> PipelineSettings:
        """The pipeline's stages, whose harness stage names the eval split in each job when the jobs evaluate on the
        imported tasks (``evolution.evaluation.imported_tasks``)."""
        stages = super().stages()
        sample2job = stages.sample2job
        if not self.evaluation.imported_tasks or not issubclass(sample2job.plugin_class, TraceJobData):
            return stages
        config = {**sample2job.config, "evaluation_split": True}
        return replace(stages, sample2job=replace(sample2job, config=config))

    @property
    def harness_adapter(self) -> str | None:
        """``evolution.adapter``."""
        return self.evolution.adapter

    def activation_rule(self, reference: PluginReference | None = None) -> ActivationRule:
        """The rule ``Recipe.activation_rule`` names, with the adapter's parts checked. It reads the recipe's
        configuration and the adapter's descriptor, and no live object.

        Under ``always``, every name in ``review_kinds`` must be a part of the adapter's tree, and a change to a
        part with ``review: true`` also waits for a person. ``RecipeConfigError`` for a name the tree does not have.
        """
        rule = super().activation_rule(reference)
        if not isinstance(rule, AlwaysActivate):
            return rule
        layout = get_adapter(self.evolution.adapter).layout
        unknown = sorted(rule.review_kinds - set(layout.parts))
        if unknown:
            raise RecipeConfigError(
                f"activation.config.review_kinds names {', '.join(unknown)}, not a part of the "
                f"{self.evolution.adapter} tree: {', '.join(layout.parts)}"
            )
        reviewed = frozenset(name for name, part in layout.parts.items() if part.review)
        return AlwaysActivate(rule.review_kinds | reviewed)

    def build_components(self, scenario: str) -> ScenarioComponents:
        descriptor = get_adapter(self.evolution.adapter)
        harness = ComponentSpec(
            files=HarnessFiles(descriptor), kind="harness_plugin", layout=HarnessLayout(descriptor)
        )
        return ScenarioComponents({HARNESS_COMPONENT: harness})

    def default_start(self) -> DefaultStart:
        """The seed tree, which a new scenario's root snapshot holds unless an artifact item starts it.

        The files are the seed tree's files; the empty seed makes a root with no files. The input is the seed
        tree's checksum.
        """
        return DefaultStart(files=dict(self.seed.files), inputs={"seed": self.seed.checksum()})

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        """The scenario's history and job records: what a delete archives beside the record store."""
        return (self.evolution.scenario_state_dir(scenario),)

    def harness_state_dir(self, scenario: str, component: str) -> Path | None:
        return self.evolution.component_state_dir(scenario, component)


__all__ = ["HarnessEvolutionRecipe", "HarnessLayout"]
