"""Reported-feedback processor for standard PPO/RLHF rollouts."""

from __future__ import annotations

from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem


class PpoRlhfProcessor(ReportedFeedbackProcessor):
    """Turn each scored rollout into one ordinary PPO training sample."""

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext) -> None:
        self._assembly = SampleAssembly.from_config(context)
        super().__init__(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        return self._assembly.build(context, context.require_score())

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:ppo_rlhf:{batch_number}", items)


__all__ = ["PpoRlhfProcessor"]
