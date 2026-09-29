"""SAO reported-feedback processor: one completed rollout, one training unit."""

from __future__ import annotations

from reef.core.records_types import AgentRecord
from reef.train.processors.common import make_policy_trajectory
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem


def make_sao_sample(item: AgentRecord, reward: float) -> TrajectoryItem:
    """Convert inference data and its evaluated reward into an SAO sample.

    ``make_policy_trajectory`` captures the shared ATIF training fields, so SAO and the
    group-relative processors resolve every shared field identically —
    including the ``runtime_load_id`` fallback chain the durable runtime needs
    to identify a training job's producing version. SAO then fills the two
    fields that path leaves at their defaults: ``action_mask`` (read from
    ``response.training`` first, the top-level payload second) and
    ``rollout_created_at``, for the backend's queue-age metric.
    """
    base = make_policy_trajectory(item, reward)
    payload = item.payload
    response = payload.get("response", {})
    training = response.get("training", {}) if isinstance(response, dict) else {}
    action_mask = training.get("action_mask", payload.get("action_mask", ())) if isinstance(training, dict) else ()
    return base.with_training(action_mask=[int(value) for value in action_mask], rollout_created_at=item.created_at)


class SAOProcessor(ReportedFeedbackProcessor):
    """Turn scored rollouts into independently-scheduled SAO samples.

    Single-Rollout Asynchronous Optimization ships each completed rollout on
    its own — no comparison group, no slowest-sample barrier. The dispatcher
    collects ``batch_size`` accepted rollouts (from as many prompts) into one
    training step; the recipe default is the paper's 128, and 1 trains once
    per accepted rollout for smoke runs.

    SAO reuses ``TrajectoryItem`` / ``TrainingBatch`` and fills ``action_mask``
    and ``rollout_created_at``. The training backend validates required
    tensors; malformed training input fails explicitly.
    """

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext) -> None:
        self._assembly = SampleAssembly.from_config(context, make_sample=make_sao_sample)
        super().__init__(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        sample = self._assembly.build(context, context.require_score())
        if not sample.training.get("action_mask", []):
            sample = sample.with_training(action_mask=sample.training.get("loss_mask", []))
        return sample

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:sao:{batch_number}", items)
