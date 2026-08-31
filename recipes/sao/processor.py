"""SAO reported-feedback processor: one completed rollout, one training unit."""

from __future__ import annotations

from dataclasses import replace

from reef.core.records_types import AgentRecord
from reef.train.processors.common import make_policy_sample
from reef.train.processors.reported import (
    NEVER,
    ReportContext,
    ReportDecision,
    ReportedFeedbackProcessor,
    SampleAssembly,
)
from reef.train.types import PolicyBatch, PolicySample, ProcessorContext, policy_row_violation


def make_sao_sample(item: AgentRecord, reward: float) -> PolicySample:
    """Convert inference data and its evaluated reward into an SAO sample.

    ``make_policy_sample`` builds the policy 5-tuple, so SAO and the
    group-relative processors resolve every shared field identically —
    including the ``weight_version`` fallback chain the durable runtime needs
    to identify a training job's producing version. SAO then fills the two
    fields that path leaves at their defaults: ``action_mask`` (read from
    ``response.training`` first, the top-level payload second) and
    ``rollout_created_at``, for the backend's queue-age metric.
    """
    base = make_policy_sample(item, reward)
    payload = item.payload
    response = payload.get("response", {})
    training = response.get("training", {}) if isinstance(response, dict) else {}
    action_mask = training.get("action_mask", payload.get("action_mask", ())) if isinstance(training, dict) else ()
    return replace(base, action_mask=tuple(int(value) for value in action_mask), rollout_created_at=item.created_at)


class SAOProcessor(ReportedFeedbackProcessor):
    """Turn scored rollouts into independently-scheduled SAO samples.

    Single-Rollout Asynchronous Optimization ships each completed rollout on
    its own — no comparison group, no slowest-sample barrier. With the recipe
    default ``batch_size=1`` the dispatcher trains once per accepted rollout, so
    a rollout enters training the moment its score arrives.

    SAO reuses ``PolicySample`` / ``PolicyBatch`` but fills the ``action_mask``
    (for skip-observation GAE) and ``rollout_created_at`` (for queue age) that
    the group-relative processors leave at their defaults. Nothing upstream
    validates a single-inference sample, so ``judge`` is where a malformed
    rollout is dropped as terminal rather than failing a training step: the
    Slime bridge's tensor contract, plus SAO's own rule that the action mask
    lines up with the loss mask.
    """

    output_schema = PolicyBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext) -> None:
        self._assembly = SampleAssembly.from_config(context, make_sample=make_sao_sample)
        super().__init__(context)

    def judge(self, context: ReportContext) -> ReportDecision:
        # 1. Shared gate: NEVER for reports that can never train, WAIT until
        #    every referenced inference has arrived.
        if (gate := context.eligibility()) is not None:
            return gate
        score = context.score
        if score is None:
            raise RuntimeError("eligible SAO report has no score")
        # 2. Assemble the policy sample, then apply SAO's documented default:
        #    with no observation spans marked, the action mask equals the loss
        #    mask. A harness that marks none and a multi-turn episode (which
        #    the shared assembly builds, filling no SAO field) both land here.
        sample = self._assembly.build(context, score)
        if sample is not None and not sample.action_mask:
            sample = replace(sample, action_mask=sample.loss_mask)
        # 3. Tensors the slime bridge would reject are terminal here, not at
        #    the training step.
        if sample is None or policy_row_violation(
            sample.tokens, sample.loss_mask, sample.rollout_log_probs, action_mask=sample.action_mask
        ):
            return NEVER
        # 4. The rollout trains on its own: one report, one candidate.
        return ReportDecision.train(sample)

    def make_batch(self, units, batch_number: int) -> PolicyBatch:
        return PolicyBatch(
            f"{self.scenario}:sao:{batch_number}",
            tuple(unit.candidates[0].value for unit in units),
        )
