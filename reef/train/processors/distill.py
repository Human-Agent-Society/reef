"""The distillation processor: the student's request with the teacher's context, rendered as the teacher reads it.

The distilling recipes make a teacher score the student's own sample. What
the teacher reads beyond the student's request is the recipe's policy
(:meth:`DistillProcessor.teacher_request`: a demonstration,
environment feedback, nothing); rendering it with the served model's chat
template and shipping it as ``teacher_tokens`` beside the student's policy
tensors is the mechanism they share, :meth:`DistillProcessor.recorded_sample`
and :meth:`DistillProcessor.teacher_tokens`. A recipe whose teacher context
comes from other samples of the same batch, as SDPO's does, composes the
request in ``make_batch`` and renders it with the same two methods.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.reports import TeacherContextReport
from reef.train.processors.common import flatten_content, recorded_request, recorded_response
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem

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
class RecordedSample:
    """The student's policy sample with the report it answers and the request and response text its inference recorded."""

    sample: TrajectoryItem
    report: TeacherContextReport
    messages: list[Any]
    tools: list[Any] | None
    response: str

    @property
    def response_ids(self) -> list[int]:
        """The student's response ids: the tail of its tokens that the loss mask covers."""
        response_length = len(self.sample.training["loss_mask"])
        return [int(token) for token in self.sample.training["tokens"][-response_length:]]


class DistillProcessor(ReportedFeedbackProcessor):
    """One report, one distillation sample: the student's policy tensors plus its teacher sequence.

    A report references one recorded request and carries the teacher's
    ``teacher_context``. ``teacher_tokens`` is the teacher's request
    (:meth:`teacher_request`) rendered with the served model's chat template
    (``tokenizer_path``), followed by the student's response ids verbatim,
    so a teacher pass scores the student's own tokens. A sequence longer
    than ``max_teacher_tokens`` cannot be scored by the trainer's window:
    its report is released with its inference record and counted in
    ``teacher_overflow_reports``. A recipe's subclass overrides
    :meth:`teacher_request` with its composition and sets ``batch_label``,
    its batches' name.
    """

    output_schema = TrainingBatch
    exclusive_sources = True
    batch_label = "teacher"

    def __init__(self, context: ProcessorContext) -> None:
        config = context.config
        self._assembly = SampleAssembly.from_config(context)
        self._max_teacher_tokens = int(config.get("max_teacher_tokens", 0))
        if self._max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")
        tokenizer_path = str(config.get("tokenizer_path", "")).strip()
        if not tokenizer_path:
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the teacher prompt")
        # transformers belongs to the training environment; the service never renders a prompt.
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self._overflow_reports: set[str] = set()
        self._overflow_count = 0
        super().__init__(context)

    def teacher_request(
        self, messages: list[Any], tools: list[Any] | None, response: str, teacher_context: str
    ) -> tuple[list[Any], list[Any] | None]:
        """The request the teacher reads, from the student's recorded request, its response and the report's teacher context.

        The default is the request as recorded: the teacher reads no
        privileged text (on-policy distillation from a separate teacher).
        """
        return messages, tools

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {**super().operational_metrics(), "teacher_overflow_reports": self._overflow_count}

    def recorded_sample(self, context: ReportContext, score: float) -> RecordedSample:
        """The student's policy sample from the report's one recorded inference, with its request and response text.

        Checks what a teacher sequence needs: the ``TeacherContextReport``
        schema, one inference, and its recorded prompt and response tokens.
        """
        parsed = context.parsed_report
        if not isinstance(parsed, TeacherContextReport):
            raise ValueError(f"{type(self).__name__} requires the TeacherContextReport schema")
        if len(context.inferences) != 1:
            raise ValueError(
                f"a teacher sequence covers one recorded request per report; report "
                f"{context.report.agent_record_id} references {len(context.inferences)}"
            )
        sample = self._assembly.build(context, score)
        tokens = [int(token) for token in sample.training.get("tokens", [])]
        response_length = len(sample.training.get("loss_mask", []))
        if not 0 < response_length < len(tokens):
            raise ValueError("a teacher sequence requires the recorded prompt and response tokens of the inference")
        payload = context.inferences[0].payload
        messages, tools = recorded_request(payload)
        return RecordedSample(sample, parsed, messages, tools, recorded_response(payload))

    def teacher_tokens(
        self,
        messages: Sequence[Any],
        tools: Sequence[Any] | None,
        response_ids: Sequence[int],
        *,
        max_prompt_tokens: int = 0,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        """The teacher sequence: ``messages`` rendered with the served model's chat template, then ``response_ids`` verbatim.

        ``max_prompt_tokens`` keeps only the first that many rendered prompt
        ids (0 keeps them all). ``enable_thinking`` is handed to the template
        when set, for templates that render a thinking switch into the
        generation prompt, so the teacher reads what the engine rendered.
        """
        template_options: dict[str, Any] = {}
        if enable_thinking is not None:
            template_options["enable_thinking"] = enable_thinking
        prompt_ids = self._tokenizer.apply_chat_template(
            list(messages),
            tools=list(tools) if tools else None,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            **template_options,
        )
        if max_prompt_tokens > 0:
            prompt_ids = prompt_ids[:max_prompt_tokens]
        return [*prompt_ids, *(int(token) for token in response_ids)]

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        # The teacher's distribution is the signal; a reported score is metadata only.
        recorded = self.recorded_sample(context, 0.0 if context.score is None else context.score)
        teacher_messages, teacher_tools = self.teacher_request(
            recorded.messages, recorded.tools, recorded.response, recorded.report.teacher_context
        )
        teacher_tokens = self.teacher_tokens(teacher_messages, teacher_tools, recorded.response_ids)
        if self._max_teacher_tokens and len(teacher_tokens) > self._max_teacher_tokens:
            self._overflow_reports.add(context.report.agent_record_id)
            logger.warning(
                "report %s skipped: its teacher sequence is %d tokens, over max_teacher_tokens %d",
                context.report.agent_record_id,
                len(teacher_tokens),
                self._max_teacher_tokens,
            )
        return recorded.sample.with_training(teacher_tokens=teacher_tokens)

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        # An overflowing report is its own group, so the group decision can release it.
        report_id = context.report.agent_record_id
        return (report_id if report_id in self._overflow_reports else None), None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        if key not in self._overflow_reports:
            raise ValueError(f"{type(self).__name__} groups only overflowing reports, got group key {key!r}")
        self._overflow_reports.discard(key)
        self._overflow_count += 1
        return GroupDecision.DISCARD

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:{self.batch_label}:{batch_number}", items)


__all__ = ["DistillProcessor", "RecordedSample", "normalize_messages_for_template", "normalize_tool_call"]
