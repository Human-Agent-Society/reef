"""TTT-Discover backend recipe."""

from __future__ import annotations

from dataclasses import dataclass

from recipes.tttd.processor import TTTDBatches, TTTDRecord2Record, TTTDTasks
from recipes.tttd.report import TTTDGroupedRolloutReport
from reef.core.reports import ReportBase
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.train.algos import StepScheduling
from reef.train.pipeline import PipelineSpec


@dataclass(frozen=True, kw_only=True)
class TTTDRecipe(WeightTrainingRecipe):
    """TTT-Discover: one complete step grid, one optimizer step, on the adaptive-entropic objective.

    The recipe binds the objective, the step schedule, and the report
    contract. Its stages are ``TTTDRecord2Record``, ``TTTDTasks``, and
    ``TTTDBatches``. The deployment writes the step grid in
    ``pipeline.record2record.config``; the later stages read it from the
    record stage.
    """

    name: str = "tttd"

    @property
    def report_type(self) -> type[ReportBase]:
        # One rollout addressed into its exact step grid; the record-to-task
        # stage adds the check against the scenario's grid.
        return TTTDGroupedRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="tttd",
            # One optimizer step over the whole grid, as --global-batch-size does for Slime.
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2record=TTTDRecord2Record, record2task=TTTDTasks, task2batch=TTTDBatches)
