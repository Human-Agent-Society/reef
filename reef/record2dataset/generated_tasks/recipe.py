"""The generated-task recipe: train on the tasks that the deployment's task generator makes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.core.plugins import PluginConfigError, PluginReference
from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.core.tasks import TaskGenerator
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.record2dataset.generated_tasks.objective import TaskGroupObjective
from reef.record2dataset.generated_tasks.processor import (
    DEFAULT_BATCHES_PER_ROUND,
    DEFAULT_PLAYS_PER_TASK,
    DEFAULT_SCAFFOLD_TOLERANCE,
    DEFAULT_TASKS_PER_STEP,
    GeneratedTaskProcessor,
)
from reef.train.algos import StepScheduling
from reef.train.processors.base import DataProcessor
from reef.train.types import ProcessorContext


@dataclass(frozen=True)
class TaskGenerationConfig:
    """The ``task-generation`` section: the checked generator class, its config, and the deployment's environment."""

    plugin: PluginReference[TaskGenerator]
    environ: Mapping[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_setting(cls, value: object, *, environ: Mapping[str, str]) -> TaskGenerationConfig:
        return cls(TaskGenerator.reference(value, setting="task-generation"), dict(environ))

    def build(self, scenario: str) -> TaskGenerator:
        """Build the generator of one scenario."""
        return self.plugin.plugin_class.from_config(self.plugin.config, scenario=scenario, environ=self.environ)


@dataclass(frozen=True, kw_only=True)
class GeneratedTaskRecipe(WeightTrainingRecipe):
    """Rounds of generated tasks played by the served model; group relative advantages per task.

    The deployment's ``task-generation`` section selects the generator. With ``rounds`` above zero the
    processor runs that many rounds: ``generator_url`` names the generator service that ``reef serve``
    starts from the deployment's ``generator`` section, ``description`` says what the tasks are about,
    and ``state_dir`` keeps a summary of each round, in one directory for each scenario.
    """

    name: str = "generated-tasks"
    tasks_per_step: int = config_field(DEFAULT_TASKS_PER_STEP)
    plays_per_task: int = config_field(DEFAULT_PLAYS_PER_TASK)
    # A thinking template's generation prompt ends with a think scaffold the history drops; the assembly may
    # realign that many masked tokens ahead of the previous response.
    scaffold_tolerance: int = config_field(DEFAULT_SCAFFOLD_TOLERANCE)
    generator_url: str = config_field("")
    rounds: int = config_field(0)
    batches_per_round: int = config_field(DEFAULT_BATCHES_PER_ROUND)
    description: str = config_field("")
    state_dir: str = config_field("")
    served_model: str = config_field("")
    task_generation: TaskGenerationConfig | None = field(default=None, repr=False, compare=False)

    @property
    def report_type(self) -> type[ReportBase]:
        # The task player's report: the verifier reward as the score; the task under metadata.task.
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective=TaskGroupObjective.name,
            processor=GeneratedTaskProcessor,
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        kwargs = super()._recipe_kwargs(
            {key: value for key, value in settings.items() if key != "task_generation"}, values
        )
        setting = settings.get("task_generation")
        if setting is None:
            return kwargs
        try:
            return {**kwargs, "task_generation": TaskGenerationConfig.from_setting(setting, environ=values)}
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")
        if self.plays_per_task < 2:
            raise ValueError("plays_per_task must be at least two")
        if self.scaffold_tolerance < 0:
            raise ValueError("scaffold_tolerance must be non-negative")
        if self.rounds < 0 or self.batches_per_round < 1:
            raise ValueError("rounds must be non-negative and batches_per_round positive")
        if self.rounds > 0:
            if self.task_generation is None:
                raise ValueError("rounds need a task generator: add a task-generation section")
            if not self.generator_url or "${" in self.generator_url:
                raise ValueError(
                    "rounds need generator_url: ${endpoints.generator} when the deployment has a generator section"
                )
            if not self.description.strip() or not self.state_dir:
                raise ValueError("rounds need a description and a state_dir")

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        """The scenario's round summaries: what a delete archives beside the record store."""
        return (Path(self.state_dir).expanduser() / scenario,) if self.state_dir else ()

    def build_processor(self, context: ProcessorContext) -> DataProcessor:
        task_generator = None if self.task_generation is None else self.task_generation.build(context.scenario)
        return GeneratedTaskProcessor(context.with_config(self.processor_config()), task_generator=task_generator)
