"""SDPO's processor: the shared distillation processor with a whole sampling step as its batch unit.

The teacher's request is composed once the step's other rollouts are known
(:meth:`SDPOProcessor.make_batch`), with the reference's reprompt template
(lasgroup/SDPO at ``7c457fc1b1f6``, ``verl/trainer/config/sdpo.yaml``): the
original question followed by the first successful sibling's response and,
when enabled, the environment feedback the report carried.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from recipes.sdpo.report import SDPOReport
from reef.core.trajectories import trajectory_reward
from reef.train.processors.distill import DistillProcessor, normalize_messages_for_template
from reef.train.processors.reported import GroupDecision, ReportContext
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem, trajectories

#: The reference's reprompt strings (``sdpo.yaml`` at the pin), used verbatim.
REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question."
SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}"
FEEDBACK_TEMPLATE = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"
THINKING_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class SDPOProcessor(DistillProcessor):
    """One complete sampling step, one update; the teacher's context comes from the step's other rollouts.

    A report names its rollout's coordinates in the step's grid
    (``groups_per_step`` questions by ``rollouts_per_group`` attempts), and
    the coordinates are its retry slot: the first report at one wins. A step
    trains only complete, sampled from one policy release, with one request
    per question; anything else is discarded whole and listed in
    :meth:`status`. A rollout whose teacher reads no privileged information
    (no successful sibling, no feedback) keeps the plain request and a sample
    weight of 0, so an all-unsuccessful step still takes its optimizer step,
    as the reference does. A teacher sequence over ``max_teacher_tokens``
    fails the step instead of dropping part of the grid.
    """

    batch_label = "sdpo"
    ordered_groups = True

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.groups_per_step = int(config.get("groups_per_step", 32))
        self.rollouts_per_group = int(config.get("rollouts_per_group", 8))
        self.max_teacher_prompt_tokens = int(config.get("max_teacher_prompt_tokens", 10240))
        self.success_reward_threshold = float(config.get("success_reward_threshold", 0.5))
        self.allow_own_success_as_demonstration = bool(config.get("allow_own_success_as_demonstration", False))
        self.remove_thinking_from_demonstration = bool(config.get("remove_thinking_from_demonstration", True))
        self.include_environment_feedback = bool(config.get("include_environment_feedback", False))
        self.environment_feedback_only_without_solution = bool(
            config.get("environment_feedback_only_without_solution", True)
        )
        self.enable_thinking = bool(config.get("enable_thinking", False))
        if self.groups_per_step <= 0 or self.rollouts_per_group < 2:
            raise ValueError("SDPO needs positive groups_per_step and at least two rollouts_per_group")
        if not math.isfinite(self.success_reward_threshold):
            raise ValueError("success_reward_threshold must be finite")
        self.failed_steps: dict[int, str] = {}
        # A batch is one unit: the whole step.
        super().__init__(context.with_config({**config, "batch_size": 1}))
        if not 0 < self.max_teacher_prompt_tokens <= self._max_teacher_tokens:
            raise ValueError("SDPO needs 0 < max_teacher_prompt_tokens <= max_teacher_tokens")

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        parsed = context.parsed_report
        if not isinstance(parsed, SDPOReport):
            raise ValueError("SDPOProcessor requires the SDPOReport schema")
        if parsed.group >= self.groups_per_step or parsed.rollout >= self.rollouts_per_group:
            raise ValueError("SDPO report coordinates exceed the configured sampling grid")
        # The score picks the demonstrating rollouts; the teacher's distribution is the target.
        recorded = self.recorded_sample(context, context.require_score())
        messages = normalize_messages_for_template(recorded.messages)
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("SDPO's reprompt template needs a request that ends with the user's question")
        artifact_ref = context.inferences[0].artifact_ref
        return recorded.sample.with_metadata(
            sdpo={
                "step": parsed.step,
                "group": parsed.group,
                "rollout": parsed.rollout,
                "release_id": None if artifact_ref is None else artifact_ref.release_id,
                "messages": messages,
                "tools": recorded.tools,
                "response": recorded.response,
                "feedback": parsed.teacher_context,
            }
        )

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        parsed = context.parsed_report
        if not isinstance(parsed, SDPOReport):
            raise ValueError("SDPOProcessor requires the SDPOReport schema")
        return parsed.step, (parsed.group, parsed.rollout)

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        if len(items) != self.groups_per_step * self.rollouts_per_group:
            return GroupDecision.INCOMPLETE
        if not isinstance(key, int):
            raise TypeError("SDPO step must be an integer")
        releases = {item.metadata["sdpo"]["release_id"] for item in items}
        reason = ""
        if None in releases or len(releases) != 1:
            reason = "missing_or_mixed_release_ids"
        for group in range(self.groups_per_step):
            siblings = [item.metadata["sdpo"] for item in items if item.metadata["sdpo"]["group"] == group]
            if any((s["messages"], s["tools"]) != (siblings[0]["messages"], siblings[0]["tools"]) for s in siblings):
                reason = "different_requests_in_group"
        if reason:
            self.failed_steps[key] = reason
            return GroupDecision.DISCARD
        return GroupDecision.READY

    def status(self) -> Mapping[str, Any]:
        failed = [{"step": step, "reason": reason} for step, reason in sorted(self.failed_steps.items())]
        return {**super().status(), "failed_steps": failed}

    def demonstration(self, sample: TrajectoryItem, samples: Sequence[TrajectoryItem]) -> str | None:
        """The response of the first successful sibling in rollout order, or None when the question has none."""
        group = sample.metadata["sdpo"]["group"]
        for candidate in samples:
            if candidate.metadata["sdpo"]["group"] != group:
                continue
            if trajectory_reward(candidate) < self.success_reward_threshold:
                continue
            if candidate is sample and not self.allow_own_success_as_demonstration:
                continue
            response = candidate.metadata["sdpo"]["response"]
            return THINKING_BLOCK.sub("", response) if self.remove_thinking_from_demonstration else response
        return None

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        batch = TrainingBatch(f"{self.scenario}:{self.batch_label}:{batch_number}", items)
        samples = sorted(
            trajectories(batch), key=lambda s: (s.metadata["sdpo"]["group"], s.metadata["sdpo"]["rollout"])
        )
        result: list[TrajectoryItem] = []
        solutions_used = feedback_used = 0
        for sample in samples:
            data = sample.metadata["sdpo"]
            solution = self.demonstration(sample, samples)
            feedback = data["feedback"] if self.include_environment_feedback and data["feedback"] else None
            if solution is not None and self.environment_feedback_only_without_solution:
                feedback = None
            messages = data["messages"]
            active = solution is not None or feedback is not None
            if active:
                prompt = REPROMPT_TEMPLATE.format(
                    prompt=messages[-1]["content"],
                    solution=(
                        SOLUTION_TEMPLATE.format(successful_previous_attempt=solution) if solution is not None else ""
                    ),
                    feedback=FEEDBACK_TEMPLATE.format(feedback_raw=feedback) if feedback is not None else "",
                )
                messages = [*messages[:-1], {"role": "user", "content": prompt}]
            response_length = len(sample.training["loss_mask"])
            teacher_tokens = self.teacher_tokens(
                messages,
                data["tools"],
                sample.training["tokens"][-response_length:],
                max_prompt_tokens=self.max_teacher_prompt_tokens,
                enable_thinking=self.enable_thinking,
            )
            if len(teacher_tokens) > self._max_teacher_tokens:
                raise ValueError(
                    "SDPO teacher sequence exceeds max_teacher_tokens; increase the trainer and teacher windows"
                )
            result.append(sample.with_training(teacher_tokens=teacher_tokens, distill_sample_weight=float(active)))
            solutions_used += solution is not None
            feedback_used += feedback is not None
        self.experiment_logger.log(
            {
                "step": samples[0].metadata["sdpo"]["step"],
                "samples": len(samples),
                "solution_fraction": solutions_used / len(samples),
                "feedback_fraction": feedback_used / len(samples),
                "reprompt_fraction": sum(s.training["distill_sample_weight"] for s in result) / len(samples),
            },
            namespace="sdpo",
        )
        return TrainingBatch(batch.batch_id, tuple(result))


__all__ = ["FEEDBACK_TEMPLATE", "REPROMPT_TEMPLATE", "SOLUTION_TEMPLATE", "SDPOProcessor"]
