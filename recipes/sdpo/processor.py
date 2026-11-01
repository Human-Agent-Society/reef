"""SDPO's pipeline stages: a whole sampling step as one batch.

``SDPOSamples`` groups the reports of one sampling step, turns each rollout into
a sample, and owns the teacher's windows; ``SDPOJobData`` batches one complete
step and writes each rollout's teacher request, which the training component
renders. The teacher's request
is composed once the step's other rollouts are known
(:meth:`SDPOJobData.make_job_data`), with the reference's reprompt template
(lasgroup/SDPO at ``7c457fc1b1f6``, ``verl/trainer/config/sdpo.yaml``): the
original question followed by the first successful sibling's response and,
when enabled, the environment feedback the report carried.

Each stage reads its own ``pipeline.<stage>.config``; ``SDPOJobData`` reads
what ``SDPOSamples`` owns from the stage object it receives.
"""

from __future__ import annotations

import math
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from recipes.sdpo.report import SDPOReport
from reef.core.chat_rendering import TeacherRequest
from reef.core.config import config_option
from reef.core.plugins import PluginConfigError
from reef.core.trajectories import make_trajectory, trajectory_reward
from reef.pipeline_context import PipelineContext
from reef.record2sample import GroupDecision, Record2Sample, ReportContext, ReportGrouping, ReportSamples
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2sample.distill import student_call, student_sample
from reef.sample2job import Sample2Job, Sample2JobSettings
from reef.train.types import JobData, Sample, TrajectorySample, trajectories

#: The reference's reprompt strings (``sdpo.yaml`` at the pin), used verbatim.
REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question."
SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}"
FEEDBACK_TEMPLATE = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"
THINKING_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def sibling_request(context: ReportContext) -> tuple[list[Any], list[Any] | None]:
    """The question a rollout answered: the messages of its last call, normalized for a chat template, and its
    tools."""
    call = student_call(make_trajectory(context.ends))
    return call.messages, call.tools


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
            raise ValueError("SDPOSamples requires the SDPOReport schema")
        return parsed.step, (parsed.group, parsed.rollout)

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) != self.groups_per_step * self.rollouts_per_group:
            return GroupDecision.INCOMPLETE
        if not isinstance(key, int):
            raise TypeError("SDPO step must be an integer")
        releases = {
            None if member.ends[0].artifact_ref is None else member.ends[0].artifact_ref.release_id
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


@dataclass(frozen=True)
class SDPOSettings(SampleAssemblySettings):
    """The config of ``SDPOSamples``: the sampling grid of one step, the teacher's windows, the sample assembly, and
    the staleness limit.

    Values: ``groups_per_step`` is a positive int; ``rollouts_per_group`` is an int of at least two;
    ``0 < max_teacher_prompt_tokens <= max_teacher_tokens``; the other keys are as in ``SampleAssemblySettings``.
    The constructor refuses other values.
    Purpose: ``SDPOSamples.from_config`` parses it from ``pipeline.record2sample.config``.
    """

    groups_per_step: int = config_option(32, help="The questions in one sampling grid.")
    rollouts_per_group: int = config_option(8, help="The attempts at each question; at least 2.")
    max_teacher_prompt_tokens: int = config_option(
        10240, help="The rendered teacher prompt is cut to this many tokens."
    )
    max_teacher_tokens: int = config_option(
        18432, help="The longest teacher sequence the trainer scores; a longer sample does not train."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.groups_per_step <= 0 or self.rollouts_per_group < 2:
            raise ValueError("SDPO needs positive groups_per_step and at least two rollouts_per_group")
        if not 0 < self.max_teacher_prompt_tokens <= self.max_teacher_tokens:
            raise ValueError("SDPO needs 0 < max_teacher_prompt_tokens <= max_teacher_tokens")


class SDPOSamples(ReportSamples):
    """One rollout as one sample, with its grid coordinates, request, and response for the teacher prompt.

    One sampling step is one group (``SDPOGrouping``); a terminal report owns its source. The stage owns the grid and
    the teacher's windows (config: ``SDPOSettings``); ``SDPOJobData`` writes them into each sample's teacher request.
    """

    def __init__(self, context: PipelineContext, settings: SDPOSettings) -> None:
        #: The grouping of sampling steps.
        self.step_grouping = SDPOGrouping(settings.groups_per_step, settings.rollouts_per_group)
        super().__init__(context, self.step_grouping, exclusive_sources=True, max_staleness=settings.max_staleness)
        self.groups_per_step = settings.groups_per_step
        self.rollouts_per_group = settings.rollouts_per_group
        self.assembly = SampleAssembly(settings)
        self.max_teacher_prompt_tokens = settings.max_teacher_prompt_tokens
        self.max_teacher_tokens = settings.max_teacher_tokens

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, SDPOSettings.parse(config, plugin_class=cls))

    def make_sample(self, context: ReportContext) -> TrajectorySample:
        parsed = context.parsed_report
        if not isinstance(parsed, SDPOReport):
            raise ValueError("SDPOSamples requires the SDPOReport schema")
        if parsed.group >= self.groups_per_step or parsed.rollout >= self.rollouts_per_group:
            raise ValueError("SDPO report coordinates exceed the configured sampling grid")
        # The score picks the demonstrating rollouts; the teacher's distribution is the target.
        sample = student_sample(self.assembly, context, context.require_score())
        call = student_call(sample)
        messages, tools = call.messages, call.tools
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("SDPO's reprompt template needs a request that ends with the user's question")
        artifact_ref = context.ends[0].artifact_ref
        return sample.with_metadata(
            sdpo={
                "step": parsed.step,
                "group": parsed.group,
                "rollout": parsed.rollout,
                "release_id": None if artifact_ref is None else artifact_ref.release_id,
                "messages": messages,
                "tools": tools,
                "response": call.reply,
                "feedback": parsed.teacher_context,
            }
        )


@dataclass(frozen=True)
class SDPODemonstrationSettings(Sample2JobSettings):
    """The config of ``SDPOJobData``: which sibling the teacher reads, whether it reads the feedback, the buffer and the job size."""

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
        super().__post_init__()
        if not math.isfinite(self.success_reward_threshold):
            raise ValueError("success_reward_threshold must be finite")


class SDPOJobData(Sample2Job):
    """One complete sampling step, one job's data; the teacher's context comes from the step's other rollouts.

    A report names its rollout's coordinates in the step's grid
    (``groups_per_step`` questions by ``rollouts_per_group`` attempts), and
    the coordinates are its retry slot in ``SDPOGrouping``: the first report at one wins. A step
    trains only complete, sampled from one policy release, with one request
    per question; anything else is discarded whole and listed in
    ``status()``. A rollout whose teacher reads no privileged information
    (no successful sibling, no feedback) keeps the plain request and a sample
    weight of 0, so an all-unsuccessful step still takes its optimizer step,
    as the reference does. The teacher's windows come from ``SDPOSamples``; the
    training component renders each teacher sequence, and one over
    ``max_teacher_tokens`` does not train. ``max_job_samples`` is the
    grid, so the trainer refuses a larger ``job_samples``, and one job's data
    is exactly one grid. Set ``job_samples`` to the grid.
    """

    def __init__(
        self, context: PipelineContext, settings: SDPODemonstrationSettings, record2sample: SDPOSamples
    ) -> None:
        self.max_teacher_prompt_tokens = record2sample.max_teacher_prompt_tokens
        self.max_teacher_tokens = record2sample.max_teacher_tokens
        self.success_reward_threshold = settings.success_reward_threshold
        self.allow_own_success_as_demonstration = settings.allow_own_success_as_demonstration
        self.remove_thinking_from_demonstration = settings.remove_thinking_from_demonstration
        self.include_environment_feedback = settings.include_environment_feedback
        self.environment_feedback_only_without_solution = settings.environment_feedback_only_without_solution
        # One job's data is one grid: a larger job_samples would mix the coordinates of two grids.
        super().__init__(
            context,
            settings.buffer_size,
            settings.job_samples,
            max_job_samples=record2sample.groups_per_step * record2sample.rollouts_per_group,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2sample: Record2Sample,
        environ: Mapping[str, str],
    ) -> Self:
        settings = SDPODemonstrationSettings.parse(config, plugin_class=cls)
        if not isinstance(record2sample, SDPOSamples):
            raise PluginConfigError(
                f"{cls.__name__} needs SDPOSamples as record2sample, not {type(record2sample).__name__}"
            )
        return cls(context, settings, record2sample)

    def demonstration(self, sample: TrajectorySample, samples: Sequence[TrajectorySample]) -> str | None:
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

    def make_job_data(self, items: tuple[Sample, ...], data_id: str) -> JobData:
        job_data = JobData(data_id, items)
        samples = sorted(
            trajectories(job_data), key=lambda s: (s.metadata["sdpo"]["group"], s.metadata["sdpo"]["rollout"])
        )
        result: list[TrajectorySample] = []
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
            teacher = TeacherRequest(messages, data["tools"], self.max_teacher_prompt_tokens, self.max_teacher_tokens)
            result.append(
                sample.with_metadata(teacher=teacher.as_json()).with_training(distill_sample_weight=float(active))
            )
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
        return JobData(job_data.data_id, tuple(result))


__all__ = [
    "FEEDBACK_TEMPLATE",
    "REPROMPT_TEMPLATE",
    "SOLUTION_TEMPLATE",
    "SDPODemonstrationSettings",
    "SDPOGrouping",
    "SDPOJobData",
    "SDPOSamples",
    "SDPOSettings",
]
