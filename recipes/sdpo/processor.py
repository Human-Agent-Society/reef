"""SDPO's pipeline stages: a whole sampling step as one batch.

``SDPORecord2Record`` groups the reports of one sampling step;
``SDPOTasks`` turns each rollout into a task and owns the teacher's renderer
and windows; ``SDPOBatches`` batches one complete step. The teacher's request
is composed once the step's other rollouts are known
(:meth:`SDPOBatches.make_batch`), with the reference's reprompt template
(lasgroup/SDPO at ``7c457fc1b1f6``, ``verl/trainer/config/sdpo.yaml``): the
original question followed by the first successful sibling's response and,
when enabled, the environment feedback the report carried.

Each stage reads its own ``pipeline.<stage>.config``; a later stage reads
what an earlier stage owns from the stage object it receives.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from recipes.sdpo.report import SDPOReport
from reef.core.config import config_option
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.core.trajectories import trajectory_reward
from reef.pipeline_context import PipelineContext
from reef.record2record import GroupDecision, Record2Record, ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2task import Record2Task, ReportTasks
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2task.distill import TeacherRenderer, normalize_messages_for_template, recorded_sample
from reef.record2task.samples import recorded_request, recorded_response
from reef.task2batch import Task2Batch
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem, trajectories

#: The reference's reprompt strings (``sdpo.yaml`` at the pin), used verbatim.
REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question."
SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}"
FEEDBACK_TEMPLATE = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"
THINKING_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def sibling_request(context: ReportContext) -> tuple[list[Any], list[Any] | None]:
    """The question a rollout answered: its normalized recorded messages and its tools."""
    messages, tools = recorded_request(context.inferences[0].payload)
    return normalize_messages_for_template(messages), tools


@dataclass(frozen=True)
class SDPOGridSettings(PluginSettings):
    """The config of ``SDPORecord2Record``: the sampling grid of one step."""

    groups_per_step: int = config_option(32, help="The questions in one sampling grid.")
    rollouts_per_group: int = config_option(8, help="The attempts at each question; at least 2.")

    def __post_init__(self) -> None:
        if self.groups_per_step <= 0 or self.rollouts_per_group < 2:
            raise ValueError("SDPO needs positive groups_per_step and at least two rollouts_per_group")


class SDPOGrouping(ReportGrouping):
    """One sampling step is one group, and its (group, rollout) coordinates are the slots.

    A step is ready when its grid is complete, it was sampled from one policy
    release, and the rollouts of each question share one request. Otherwise the
    complete step is discarded and listed in ``failed_steps``.
    """

    def __init__(self, groups_per_step: int, rollouts_per_group: int) -> None:
        self.groups_per_step = groups_per_step
        self.rollouts_per_group = rollouts_per_group
        self.failed_steps: dict[int, str] = {}

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        parsed = context.parsed_report
        if not isinstance(parsed, SDPOReport):
            raise ValueError("SDPORecord2Record requires the SDPOReport schema")
        return parsed.step, (parsed.group, parsed.rollout)

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) != self.groups_per_step * self.rollouts_per_group:
            return GroupDecision.INCOMPLETE
        if not isinstance(key, int):
            raise TypeError("SDPO step must be an integer")
        releases = {
            None if member.inferences[0].artifact_ref is None else member.inferences[0].artifact_ref.release_id
            for member in members
        }
        reason = ""
        if None in releases or len(releases) != 1:
            reason = "missing_or_mixed_release_ids"
        reports = [(member, parsed) for member in members if isinstance(parsed := member.parsed_report, SDPOReport)]
        for group in range(self.groups_per_step):
            siblings = [sibling_request(member) for member, parsed in reports if parsed.group == group]
            if any(request != siblings[0] for request in siblings):
                reason = "different_requests_in_group"
        if reason:
            self.failed_steps[key] = reason
            return GroupDecision.DISCARD
        return GroupDecision.READY

    def status(self) -> Mapping[str, Any]:
        return {
            "failed_steps": [{"step": step, "reason": reason} for step, reason in sorted(self.failed_steps.items())]
        }


class SDPORecord2Record(ReportRecord2Record):
    """SDPO's record stage: one group per sampling step, in step order; a terminal report owns its source.

    Config: ``groups_per_step`` and ``rollouts_per_group``, the grid. The
    later stages read the grid from ``step_grouping``.
    """

    def __init__(self, context: PipelineContext, settings: SDPOGridSettings) -> None:
        #: The grouping of sampling steps, with the grid that the later stages read.
        self.step_grouping = SDPOGrouping(settings.groups_per_step, settings.rollouts_per_group)
        super().__init__(context, self.step_grouping, exclusive_sources=True, ordered_groups=True)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(context, SDPOGridSettings.parse(config, plugin_class=cls))


@dataclass(frozen=True)
class SDPOTeacherSettings(SampleAssemblySettings):
    """The config of ``SDPOTasks``: the teacher's tokenizer and windows, and the sample assembly."""

    tokenizer_path: str = config_option("", help="The served model's tokenizer, which renders the teacher prompt.")
    max_teacher_prompt_tokens: int = config_option(
        10240, help="The rendered teacher prompt is cut to this many tokens."
    )
    max_teacher_tokens: int = config_option(
        18432, help="The longest teacher sequence the trainer scores; a longer one fails the step."
    )
    enable_thinking: bool = config_option(
        False, help="The chat template's thinking switch, set as the engine sampled."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the teacher prompt")
        if not 0 < self.max_teacher_prompt_tokens <= self.max_teacher_tokens:
            raise ValueError("SDPO needs 0 < max_teacher_prompt_tokens <= max_teacher_tokens")


class SDPOTasks(ReportTasks):
    """One rollout as one task, with its grid coordinates, request, and response for the teacher prompt.

    The grid comes from the record stage. The stage owns the teacher's
    renderer and windows (config: ``tokenizer_path``,
    ``max_teacher_prompt_tokens``, ``max_teacher_tokens``,
    ``enable_thinking``, and the sample assembly's keys); ``SDPOBatches``
    renders the teacher sequences with them.
    """

    def __init__(self, settings: SDPOTeacherSettings, record2record: SDPORecord2Record) -> None:
        super().__init__()
        self.groups_per_step = record2record.step_grouping.groups_per_step
        self.rollouts_per_group = record2record.step_grouping.rollouts_per_group
        self.assembly = SampleAssembly(settings)
        self.renderer = TeacherRenderer(settings.tokenizer_path)
        self.max_teacher_prompt_tokens = settings.max_teacher_prompt_tokens
        self.max_teacher_tokens = settings.max_teacher_tokens
        self.enable_thinking = settings.enable_thinking

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        settings = SDPOTeacherSettings.parse(config, plugin_class=cls)
        if not isinstance(record2record, SDPORecord2Record):
            raise PluginConfigError(
                f"{cls.__name__} needs SDPORecord2Record as record2record, not {type(record2record).__name__}"
            )
        return cls(settings, record2record)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        parsed = context.parsed_report
        if not isinstance(parsed, SDPOReport):
            raise ValueError("SDPOTasks requires the SDPOReport schema")
        if parsed.group >= self.groups_per_step or parsed.rollout >= self.rollouts_per_group:
            raise ValueError("SDPO report coordinates exceed the configured sampling grid")
        # The score picks the demonstrating rollouts; the teacher's distribution is the target.
        sample = recorded_sample(self.assembly, context, context.require_score())
        payload = context.inferences[0].payload
        messages, tools = sibling_request(context)
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("SDPO's reprompt template needs a request that ends with the user's question")
        artifact_ref = context.inferences[0].artifact_ref
        return sample.with_metadata(
            sdpo={
                "step": parsed.step,
                "group": parsed.group,
                "rollout": parsed.rollout,
                "release_id": None if artifact_ref is None else artifact_ref.release_id,
                "messages": messages,
                "tools": tools,
                "response": recorded_response(payload),
                "feedback": parsed.teacher_context,
            }
        )


@dataclass(frozen=True)
class SDPODemonstrationSettings(PluginSettings):
    """The config of ``SDPOBatches``: which sibling the teacher reads, and whether it reads the feedback."""

    success_reward_threshold: float = config_option(0.5, help="An attempt at or above this score is correct.")
    allow_own_success_as_demonstration: bool = config_option(
        False, help="Let a correct attempt read its own response."
    )
    remove_thinking_from_demonstration: bool = config_option(True, help="Strip <think> blocks from the demonstration.")
    include_environment_feedback: bool = config_option(
        False, help="Add the report's teacher_context to the teacher's prompt."
    )
    environment_feedback_only_without_solution: bool = config_option(
        True, help="Use the feedback only when no correct sibling exists."
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.success_reward_threshold):
            raise ValueError("success_reward_threshold must be finite")


class SDPOBatches(Task2Batch):
    """One complete sampling step, one batch; the teacher's context comes from the step's other rollouts.

    A report names its rollout's coordinates in the step's grid
    (``groups_per_step`` questions by ``rollouts_per_group`` attempts), and
    the coordinates are its retry slot in ``SDPOGrouping``: the first report at one wins. A step
    trains only complete, sampled from one policy release, with one request
    per question; anything else is discarded whole and listed in
    ``status()``. A rollout whose teacher reads no privileged information
    (no successful sibling, no feedback) keeps the plain request and a sample
    weight of 0, so an all-unsuccessful step still takes its optimizer step,
    as the reference does. The teacher's renderer and windows come from
    ``SDPOTasks``. A teacher sequence over ``max_teacher_tokens`` fails the
    step instead of dropping part of the grid.
    """

    def __init__(self, context: PipelineContext, settings: SDPODemonstrationSettings, record2task: SDPOTasks) -> None:
        self.renderer = record2task.renderer
        self.max_teacher_prompt_tokens = record2task.max_teacher_prompt_tokens
        self.max_teacher_tokens = record2task.max_teacher_tokens
        self.enable_thinking = record2task.enable_thinking
        self.success_reward_threshold = settings.success_reward_threshold
        self.allow_own_success_as_demonstration = settings.allow_own_success_as_demonstration
        self.remove_thinking_from_demonstration = settings.remove_thinking_from_demonstration
        self.include_environment_feedback = settings.include_environment_feedback
        self.environment_feedback_only_without_solution = settings.environment_feedback_only_without_solution
        # A batch is one unit: the whole step.
        super().__init__(context, 1)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        settings = SDPODemonstrationSettings.parse(config, plugin_class=cls)
        if not isinstance(record2task, SDPOTasks):
            raise PluginConfigError(f"{cls.__name__} needs SDPOTasks as record2task, not {type(record2task).__name__}")
        return cls(context, settings, record2task)

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

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        batch = TrainingBatch(batch_id, items)
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
            teacher_tokens = self.renderer.teacher_tokens(
                messages,
                data["tools"],
                sample.training["tokens"][-response_length:],
                max_prompt_tokens=self.max_teacher_prompt_tokens,
                enable_thinking=self.enable_thinking,
            )
            if len(teacher_tokens) > self.max_teacher_tokens:
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


__all__ = [
    "FEEDBACK_TEMPLATE",
    "REPROMPT_TEMPLATE",
    "SOLUTION_TEMPLATE",
    "SDPOBatches",
    "SDPODemonstrationSettings",
    "SDPOGridSettings",
    "SDPOGrouping",
    "SDPORecord2Record",
    "SDPOTasks",
    "SDPOTeacherSettings",
]
