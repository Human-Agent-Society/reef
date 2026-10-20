"""Distillation: the student's request with the teacher's context, rendered as the teacher reads it.

The distilling recipes make a teacher score the student's own sample. What
the teacher reads beyond the student's request is the recipe's policy
(:meth:`DistillTasks.teacher_request`: a demonstration, environment feedback,
nothing); rendering it with the served model's chat template and shipping it
as ``teacher_tokens`` beside the student's policy tensors is the mechanism
they share, :func:`recorded_sample` and :meth:`TeacherRenderer.teacher_tokens`.
A recipe whose teacher context comes from other samples of the same batch, as
SDPO's does, composes the request in ``make_batch`` and renders it with the
same two.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self, cast

from reef.core.batches import TrajectoryItem
from reef.core.config import config_option
from reef.core.reports import TeacherContextReport
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record, ReportContext
from reef.record2task import ReportTasks, SkippedReport
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2task.samples import flatten_content, recorded_request, recorded_response

logger = logging.getLogger(__name__)


def normalize_messages_for_template(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Messages as a chat template expects them: text content, chat roles, tool arguments as objects."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        entry = dict(message)
        if entry.get("role") == "developer":
            entry["role"] = "system"
        content = entry.get("content")
        if content is not None and not isinstance(content, str):
            entry["content"] = flatten_content(content)
        if entry.get("tool_calls"):
            entry["tool_calls"] = [normalize_tool_call(call) for call in entry["tool_calls"]]
        normalized.append(entry)
    return normalized


def normalize_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    """A tool call with its function arguments as an object, as chat templates render them."""
    normalized = dict(call)
    function = normalized.get("function")
    if isinstance(function, Mapping):
        function = dict(function)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                function["arguments"] = {}
        normalized["function"] = function
    return normalized


@dataclass(frozen=True)
class DistillSettings(SampleAssemblySettings):
    """The config of ``DistillTasks``: the teacher's tokenizer and window, and the sample assembly."""

    tokenizer_path: str = config_option("", help="The served model's tokenizer, which renders the teacher prompt.")
    max_teacher_tokens: int = config_option(
        0, help="The longest teacher sequence the trainer scores; a longer report is skipped. 0 is no limit."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the teacher prompt")
        if self.max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")


class TeacherRenderer:
    """Render the teacher's request with the served model's chat template."""

    def __init__(self, tokenizer_path: str) -> None:
        # transformers belongs to the training environment; the service never renders a prompt.
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def teacher_tokens(
        self,
        messages: Sequence[Any],
        tools: Sequence[Any] | None,
        response_ids: Sequence[int],
        *,
        max_prompt_tokens: int = 0,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        """The teacher sequence: ``messages`` rendered with the chat template, then ``response_ids`` verbatim.

        ``max_prompt_tokens`` keeps only the first that many rendered prompt
        ids (0 keeps them all). ``enable_thinking`` is handed to the template
        when set, for templates that render a thinking switch into the
        generation prompt, so the teacher reads what the engine rendered.
        """
        template_options: dict[str, Any] = {}
        if enable_thinking is not None:
            template_options["enable_thinking"] = enable_thinking
        # tokenize=True and return_dict=False return the flat token list.
        prompt_ids = cast(
            list[int],
            self.tokenizer.apply_chat_template(
                list(messages),
                tools=list(tools) if tools else None,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **template_options,
            ),
        )
        if max_prompt_tokens > 0:
            prompt_ids = prompt_ids[:max_prompt_tokens]
        return [*prompt_ids, *(int(token) for token in response_ids)]


def recorded_sample(assembly: SampleAssembly, context: ReportContext, score: float) -> TrajectoryItem:
    """The student's policy sample from the report's one recorded inference.

    Checks what a teacher sequence needs: one inference, and its recorded
    prompt and response tokens.
    """
    if len(context.inferences) != 1:
        raise ValueError(
            f"a teacher sequence covers one recorded request per report; report "
            f"{context.report.agent_record_id} references {len(context.inferences)}"
        )
    sample = assembly.build(context, score)
    tokens = [int(token) for token in sample.training.get("tokens", [])]
    response_length = len(sample.training.get("loss_mask", []))
    if not 0 < response_length < len(tokens):
        raise ValueError("a teacher sequence requires the recorded prompt and response tokens of the inference")
    return sample


def teacher_overflow(context: ReportContext, teacher_tokens: Sequence[int], max_teacher_tokens: int) -> bool:
    """Return True when the teacher sequence is longer than ``max_teacher_tokens``, and log the report it skips.

    0 is no limit.
    """
    if not max_teacher_tokens or len(teacher_tokens) <= max_teacher_tokens:
        return False
    logger.warning(
        "report %s skipped: its teacher sequence is %d tokens, over max_teacher_tokens %d",
        context.report.agent_record_id,
        len(teacher_tokens),
        max_teacher_tokens,
    )
    return True


class DistillTasks(ReportTasks):
    """One report, one distillation task: the student's policy tensors plus its teacher sequence.

    A report references one recorded request and carries the teacher's
    ``teacher_context``. ``teacher_tokens`` is the teacher's request
    (:meth:`teacher_request`) rendered with the served model's chat template,
    followed by the student's response ids verbatim, so a teacher pass scores
    the student's own tokens. A sequence longer than ``max_teacher_tokens``
    cannot be scored by the trainer's window: the report is skipped and
    counted in ``teacher_overflow_reports``. A recipe's subclass overrides
    :meth:`teacher_request` with its composition.
    """

    def __init__(self, settings: DistillSettings) -> None:
        super().__init__()
        self.assembly = SampleAssembly(settings)
        self.max_teacher_tokens = settings.max_teacher_tokens
        self.renderer = TeacherRenderer(settings.tokenizer_path)
        self.overflow_count = 0

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(DistillSettings.parse(config, plugin_class=cls))

    def teacher_request(
        self, messages: list[Any], tools: list[Any] | None, response: str, teacher_context: str
    ) -> tuple[list[Any], list[Any] | None]:
        """The request the teacher reads, from the student's recorded request, its response and the report's teacher context.

        The default is the request as recorded: the teacher reads no
        privileged text (on-policy distillation from a separate teacher).
        """
        return messages, tools

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {"teacher_overflow_reports": self.overflow_count}

    def make_sample(self, context: ReportContext) -> TrajectoryItem | SkippedReport:
        parsed = context.parsed_report
        if not isinstance(parsed, TeacherContextReport):
            raise ValueError(f"{type(self).__name__} requires the TeacherContextReport schema")
        # The teacher's distribution is the signal; a reported score is metadata only.
        sample = recorded_sample(self.assembly, context, 0.0 if context.score is None else context.score)
        payload = context.inferences[0].payload
        messages, tools = recorded_request(payload)
        teacher_messages, teacher_tools = self.teacher_request(
            messages, tools, recorded_response(payload), parsed.teacher_context
        )
        response_length = len(sample.training["loss_mask"])
        teacher_tokens = self.renderer.teacher_tokens(
            teacher_messages, teacher_tools, sample.training["tokens"][-response_length:]
        )
        if teacher_overflow(context, teacher_tokens, self.max_teacher_tokens):
            self.overflow_count += 1
            return SkippedReport("teacher_overflow")
        return sample.with_training(teacher_tokens=teacher_tokens)


__all__ = [
    "DistillSettings",
    "DistillTasks",
    "TeacherRenderer",
    "normalize_messages_for_template",
    "normalize_tool_call",
    "recorded_sample",
    "teacher_overflow",
]
