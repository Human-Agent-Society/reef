"""SDPO's synchronous sampling barrier and feedback-conditioned teacher prompts."""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping
from typing import Any

from recipes.sdpo.report import SDPOReport
from reef.core.trajectories import trajectory_reward
from reef.train.processors.common import recorded_request
from reef.train.processors.distill import normalize_messages_for_template
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem, trajectories

REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question."
SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}"
FEEDBACK_TEMPLATE = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"


class SDPOProcessor(ReportedFeedbackProcessor):
    """One complete sampling step, one update; demonstrations come only from its siblings.

    Coordinates are stable retry slots. Invalid steps are discarded as a whole:
    partial groups and different policy releases cannot supply comparable feedback.
    Rows with no privileged information remain in the update with zero distillation
    weight, including all-unsuccessful steps (AdamW still advances, as in the reference).
    """

    output_schema = TrainingBatch
    exclusive_sources = True
    ordered_groups = True

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.groups_per_step = int(config.get("groups_per_step", 32))
        self.rollouts_per_group = int(config.get("rollouts_per_group", 8))
        self.max_teacher_prompt_tokens = int(config.get("max_teacher_prompt_tokens", 10240))
        self.max_teacher_tokens = int(config.get("max_teacher_tokens", 18432))
        self.success_reward_threshold = float(config.get("success_reward_threshold", 0.5))
        self.dont_reprompt_on_self_success = bool(config.get("dont_reprompt_on_self_success", True))
        self.remove_thinking_from_demonstration = bool(config.get("remove_thinking_from_demonstration", True))
        self.include_environment_feedback = bool(config.get("include_environment_feedback", False))
        self.environment_feedback_only_without_solution = bool(
            config.get("environment_feedback_only_without_solution", True)
        )
        self.enable_thinking = bool(config.get("enable_thinking", False))
        if self.groups_per_step <= 0 or self.rollouts_per_group < 2:
            raise ValueError("SDPO needs positive groups_per_step and at least two rollouts_per_group")
        if self.max_teacher_prompt_tokens <= 0 or self.max_teacher_tokens < self.max_teacher_prompt_tokens:
            raise ValueError("SDPO needs 0 < max_teacher_prompt_tokens <= max_teacher_tokens")
        if not math.isfinite(self.success_reward_threshold):
            raise ValueError("success_reward_threshold must be finite")
        tokenizer_path = str(config.get("tokenizer_path", "")).strip()
        if not tokenizer_path:
            raise ValueError("tokenizer_path is required to render SDPO's teacher prompts")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.assembly = SampleAssembly.from_config(context)
        self.failed_steps: dict[int, str] = {}
        super().__init__(context.with_config({**config, "batch_size": 1}))

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        parsed = context.parsed_report
        if not isinstance(parsed, SDPOReport):
            raise ValueError("SDPOProcessor requires SDPOReport")
        if parsed.group >= self.groups_per_step or parsed.rollout >= self.rollouts_per_group:
            raise ValueError("SDPO report coordinates exceed the configured sampling grid")
        if len(context.inferences) != 1:
            raise ValueError("SDPO requires one recorded inference per rollout report")
        sample = self.assembly.build(context, context.require_score())
        inference = context.inferences[0]
        messages, tools = recorded_request(inference.payload)
        messages = normalize_messages_for_template(messages)
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("SDPO's paper prompt must end with the original user question")
        response_length = len(sample.training["loss_mask"])
        response_tokens = sample.training["tokens"][-response_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        release_id = inference.artifact_ref.release_id if inference.artifact_ref is not None else None
        return sample.with_metadata(
            sdpo={
                "step": parsed.step,
                "group": parsed.group,
                "rollout": parsed.rollout,
                "release_id": release_id,
                "messages": messages,
                "tools": tools,
                "response": response,
                "feedback": parsed.teacher_context,
            }
        )

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        parsed = context.parsed_report
        if not isinstance(parsed, SDPOReport):
            raise ValueError("SDPOProcessor requires SDPOReport")
        return parsed.step, (parsed.group, parsed.rollout)

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        if len(items) != self.groups_per_step * self.rollouts_per_group:
            return GroupDecision.INCOMPLETE
        if not isinstance(key, int):
            raise TypeError("SDPO step must be an integer")
        versions = {item.metadata["sdpo"]["release_id"] for item in items}
        reason = ""
        if None in versions or len(versions) != 1:
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
        return {
            "failed_steps": [{"step": step, "reason": reason} for step, reason in sorted(self.failed_steps.items())]
        }

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        batch = TrainingBatch(f"{self.scenario}:sdpo:{batch_number}", items)
        samples = sorted(
            trajectories(batch), key=lambda s: (s.metadata["sdpo"]["group"], s.metadata["sdpo"]["rollout"])
        )
        result = []
        solutions_used = feedback_used = 0
        for sample in samples:
            data = sample.metadata["sdpo"]
            successes = [
                s
                for s in samples
                if s.metadata["sdpo"]["group"] == data["group"]
                and trajectory_reward(s) >= self.success_reward_threshold
                and (not self.dont_reprompt_on_self_success or s is not sample)
            ]
            solution = successes[0].metadata["sdpo"]["response"] if successes else None
            if solution is not None and self.remove_thinking_from_demonstration:
                solution = re.sub(r"<think>.*?</think>\s*", "", solution, flags=re.DOTALL)
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
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                tools=data["tools"] or None,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                enable_thinking=self.enable_thinking,
            )[: self.max_teacher_prompt_tokens]
            response_length = len(sample.training["loss_mask"])
            teacher_tokens = [*prompt_ids, *sample.training["tokens"][-response_length:]]
            if len(teacher_tokens) > self.max_teacher_tokens:
                raise ValueError(
                    "SDPO teacher sequence exceeds max_teacher_tokens; increase the trainer and teacher windows"
                )
            result.append(sample.with_training(teacher_tokens=teacher_tokens, distill_sample_mask=int(active)))
            solutions_used += solution is not None
            feedback_used += feedback is not None
        self.experiment_logger.log(
            {
                "step": samples[0].metadata["sdpo"]["step"],
                "samples": len(samples),
                "solution_fraction": solutions_used / len(samples),
                "feedback_fraction": feedback_used / len(samples),
                "reprompt_fraction": sum(s.training["distill_sample_mask"] for s in result) / len(samples),
            },
            namespace="sdpo",
        )
        return TrainingBatch(batch.batch_id, tuple(result))
