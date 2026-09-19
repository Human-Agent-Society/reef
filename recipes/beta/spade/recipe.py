"""The SPADE recipes: the Reasoning Agent trains on the tasks the Designer wrote; the Designer trains on their regret."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.beta.spade.designer_processor import DEFAULT_GENERATIONS_PER_STEP, SpadeDesignerProcessor
from recipes.beta.spade.objective import SpadeObjective
from recipes.beta.spade.processor import (
    DEFAULT_BATCHES_PER_GENERATION,
    DEFAULT_COUNT,
    DEFAULT_EVAL_FRACTION,
    DEFAULT_HINT_PLAYS,
    DEFAULT_ROLLOUTS_PER_TASK,
    DEFAULT_SCAFFOLD_TOLERANCE,
    DEFAULT_TASKS_PER_STEP,
    SpadeProcessor,
)
from reef.core.batches import StepScheduling
from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.record2dataset.designer import DEFAULT_TURN_LIMIT, DIFFICULTIES


@dataclass(frozen=True, kw_only=True)
class SpadeRecipe(WeightTrainingRecipe):
    """Reports from the task player, grouped by task; group relative advantages on Tinker's importance sampling loss.

    With ``generations`` above zero the processor also runs the Designer: ``generator_url`` names the
    generator service ``reef serve`` starts from the deployment's ``generator`` section, ``description``
    and ``skills`` say what the tasks are about, and ``state_dir`` keeps each generation's report.
    """

    name: str = "spade"
    tasks_per_step: int = config_field(DEFAULT_TASKS_PER_STEP, env="REEF_SPADE_TASKS_PER_STEP")
    rollouts_per_task: int = config_field(DEFAULT_ROLLOUTS_PER_TASK, env="REEF_SPADE_ROLLOUTS_PER_TASK")
    # A thinking template's generation prompt ends with a think scaffold the history drops; the assembly may
    # realign that many masked tokens ahead of the previous response.
    scaffold_tolerance: int = config_field(DEFAULT_SCAFFOLD_TOLERANCE, env="REEF_SPADE_SCAFFOLD_TOLERANCE")
    # The Designer's generations, run by the processor through the generator service.
    generator_url: str = config_field("", env="REEF_SPADE_GENERATOR_URL")
    generations: int = config_field(0, env="REEF_SPADE_GENERATIONS")
    batches_per_generation: int = config_field(DEFAULT_BATCHES_PER_GENERATION)
    state_dir: str = config_field("", env="REEF_SPADE_GENERATION_DIR")
    description: str = config_field("")
    skills: tuple[str, ...] = config_field(())
    count: int = config_field(DEFAULT_COUNT)
    difficulty: str = config_field("medium")
    turn_limit: int = config_field(DEFAULT_TURN_LIMIT)
    grounding_path: str = config_field("")
    hint_plays: int = config_field(DEFAULT_HINT_PLAYS)
    eval_fraction: float = config_field(DEFAULT_EVAL_FRACTION)
    seed: int = config_field(0)
    designer_report: bool = config_field(True)
    # Hold every play until the generation lands; the generation then reports them all and trains as one batch.
    report_plays_after_generation: bool = config_field(False)
    served_model: str = config_field("")

    @property
    def report_type(self) -> type[ReportBase]:
        # The task player's report: the verifier reward as the score; the task under metadata.task.
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective=SpadeObjective.name,
            processor=SpadeProcessor,
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")
        if self.rollouts_per_task < 2:
            raise ValueError("rollouts_per_task must be at least two")
        if self.scaffold_tolerance < 0:
            raise ValueError("scaffold_tolerance must be non-negative")
        if self.generations < 0 or self.batches_per_generation < 1 or self.count < 1 or self.hint_plays < 0:
            raise ValueError(
                "generations must be non-negative, batches_per_generation and count positive, hint_plays non-negative"
            )
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        if self.turn_limit < 2:
            raise ValueError("turn_limit must be at least 2")
        if not 0 <= self.eval_fraction < 1:
            raise ValueError("eval_fraction must be in [0, 1)")
        if self.generations > 0:
            if not self.generator_url or "${" in self.generator_url:
                raise ValueError(
                    "generations need generator_url: ${endpoints.generator} when the deployment has a generator section"
                )
            if not self.description.strip() or not self.state_dir:
                raise ValueError("generations need a description and a state_dir")


@dataclass(frozen=True, kw_only=True)
class SpadeDesignerRecipe(WeightTrainingRecipe):
    """The generation's reports against Designer receipts, grouped by generation; regret as the reward on the same objective."""

    name: str = "spade_designer"
    generations_per_step: int = config_field(DEFAULT_GENERATIONS_PER_STEP, env="REEF_SPADE_GENERATIONS_PER_STEP")

    @property
    def report_type(self) -> type[ReportBase]:
        # One report per proposal: the regret as the score, the generation and its size under metadata.
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective=SpadeObjective.name,
            processor=SpadeDesignerProcessor,
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.generations_per_step <= 0:
            raise ValueError("generations_per_step must be positive")
