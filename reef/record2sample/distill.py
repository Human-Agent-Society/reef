"""Distillation: the student's call, with what the teacher reads before the student's reply.

The distilling recipes make a teacher score the student's own reply. What the teacher reads beyond the student's
request is the recipe's policy (:meth:`DistillSamples.teacher_request`: a demonstration, environment feedback,
nothing). A sample is the student's call as an ATIF trajectory with the teacher's request in its metadata
(``reef.core.chat_rendering.TeacherRequest``); the training component renders both with the model's chat template:
the teacher sequence is the teacher's request followed by the student's reply tokens verbatim. A recipe whose
teacher context comes from other samples of the same job data, as SDPO's does, writes the request in
``make_job_data``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.chat_rendering import TeacherRequest, flatten_content, normalize_messages_for_template, trajectory_tools
from reef.core.config import config_option
from reef.core.job_data import TrajectorySample
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import atif_messages, make_trajectory
from reef.pipeline_context import PipelineContext
from reef.record2sample.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2sample.reports import ReportContext, ReportSamples


@dataclass(frozen=True)
class DistillSettings(SampleAssemblySettings):
    """The config of ``DistillSamples``: the teacher's window and the sample assembly.

    Values: ``max_teacher_tokens`` is a non-negative int, 0 for no limit; the other keys are as in
    ``SampleAssemblySettings``. The constructor refuses other values.
    Purpose: ``DistillSamples.from_config`` parses it from ``pipeline.record2sample.config``.
    """

    max_teacher_tokens: int = config_option(
        0, help="The longest teacher sequence the trainer scores; a longer sample does not train. 0 is no limit."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")


@dataclass(frozen=True)
class StudentCall:
    """The student's call of a distillation sample, as chat messages that a chat template takes.

    Values: ``messages`` is the request of the call, normalized for a chat template
    (``normalize_messages_for_template``); ``tools`` its tools, or None; ``reply`` the text of its reply.
    Purpose: ``student_call`` makes it from the sample's ATIF trajectory; a recipe composes the teacher's request
    from it.
    """

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    reply: str


def student_call(sample: TrajectorySample) -> StudentCall:
    """The student's call of ``sample``, a trajectory whose last step is the student's reply.

    ``ValueError``: the trajectory does not end with an agent reply.
    """
    conversation = normalize_messages_for_template(atif_messages(sample.trajectory))
    if not conversation or conversation[-1]["role"] != "assistant":
        raise ValueError("a distillation sample ends with the student's reply")
    return StudentCall(
        conversation[:-1], trajectory_tools(sample.trajectory), flatten_content(conversation[-1].get("content"))
    )


def student_sample(assembly: SampleAssembly, context: ReportContext, score: float) -> TrajectorySample:
    """The student's sample: the last call of the report's one trajectory, with ``score`` and the report's metadata.

    ``ValueError``: the report references more than one trajectory.
    """
    if len(context.ends) != 1:
        raise ValueError(
            f"a teacher sequence covers one recorded request per report; report "
            f"{context.report.agent_record_id} references {len(context.ends)}"
        )
    return assembly.with_report(make_trajectory(context.ends, score), context)


class DistillSamples(ReportSamples):
    """One report, one distillation sample: the student's call, and the request that its teacher reads.

    A report references one recorded request and carries the teacher's ``teacher_context``. The sample is the
    student's call as an ATIF trajectory, with the teacher's request (:meth:`teacher_request`) in its
    ``extra.reef.teacher`` (``TeacherRequest``). The training component renders the teacher sequence: the request with
    the chat template, then the student's reply tokens verbatim, so a teacher pass scores the student's own tokens. A
    sequence longer than ``max_teacher_tokens`` does not train. A recipe's subclass overrides :meth:`teacher_request`
    with its composition. Each report trains by itself and owns its inference records.
    """

    def __init__(self, context: PipelineContext, settings: DistillSettings) -> None:
        super().__init__(context, exclusive_sources=True, max_staleness=settings.max_staleness)
        self.assembly = SampleAssembly(settings)
        self.max_teacher_tokens = settings.max_teacher_tokens

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, DistillSettings.parse(config, plugin_class=cls))

    def teacher_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, response: str, teacher_context: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """The request the teacher reads, from the student's request, its response and the report's teacher context.

        The default is the request as recorded: the teacher reads no privileged text (on-policy distillation from a
        separate teacher).
        """
        return messages, tools

    def make_sample(self, context: ReportContext) -> TrajectorySample:
        parsed = context.parsed_report
        if not isinstance(parsed, TeacherContextReport):
            raise ValueError(f"{type(self).__name__} requires the TeacherContextReport schema")
        # The teacher's distribution is the signal; a reported score is metadata only.
        sample = student_sample(self.assembly, context, 0.0 if context.score is None else context.score)
        call = student_call(sample)
        messages, tools = self.teacher_request(call.messages, call.tools, call.reply, parsed.teacher_context)
        teacher = TeacherRequest(messages, tools, max_tokens=self.max_teacher_tokens)
        return sample.with_metadata(teacher=teacher.as_json())


__all__ = ["DistillSamples", "DistillSettings", "StudentCall", "student_call", "student_sample"]
