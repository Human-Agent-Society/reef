"""Render SDPO's feedback-conditioned teacher prompt on the student's own response."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from reef.core.reports import TeacherContextReport
from reef.train.processors import DistillProcessor
from reef.train.processors.common import flatten_content
from reef.train.processors.reported import ReportContext
from reef.train.types import ProcessorContext, TrajectoryItem

DEFAULT_REPROMPT_TEMPLATE = "{prompt}{context}\n\nCorrectly solve the original question."


def reprompt_template(config: Mapping[str, Any]) -> str:
    template = str(config.get("reprompt_template", DEFAULT_REPROMPT_TEMPLATE))
    if "{prompt}" not in template or "{context}" not in template:
        raise ValueError("reprompt_template must contain {prompt} and {context}")
    try:
        template.format(prompt="", context="")
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("reprompt_template supports only {prompt} and {context}; escape literal braces") from error
    return template


def normalize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(message)
    if entry.get("role") == "developer":
        entry["role"] = "system"
    if entry.get("content") is not None and not isinstance(entry["content"], str):
        entry["content"] = flatten_content(entry["content"])
    calls = entry.get("tool_calls")
    if calls:
        normalized = []
        for call in calls:
            item = dict(call)
            function = item.get("function")
            if isinstance(function, Mapping):
                function = dict(function)
                if isinstance(function.get("arguments"), str):
                    try:
                        function["arguments"] = json.loads(function["arguments"])
                    except json.JSONDecodeError:
                        function["arguments"] = {}
                item["function"] = function
            normalized.append(item)
        entry["tool_calls"] = normalized
    return entry


class SDPOProcessor(DistillProcessor):
    """One graded rollout and its teacher context produce one training sample."""

    batch_label = "sdpo"

    def __init__(self, context: ProcessorContext) -> None:
        self.template = reprompt_template(context.config)
        super().__init__(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        sample = super().make_sample(context)
        parsed = context.parsed_report
        assert isinstance(parsed, TeacherContextReport)
        return sample.with_training(distill_sample_weight=float(bool(parsed.teacher_context.strip())))

    def teacher_overflow(self, report_id: str, token_count: int) -> None:
        raise ValueError(
            f"SDPO report {report_id} has {token_count} teacher tokens, over max_teacher_tokens; "
            "increase the teacher/trainer window or shorten feedback before retrying the complete group"
        )

    def teacher_request(
        self, messages: list[Any], tools: list[Any] | None, response: str, teacher_context: str
    ) -> tuple[list[Any], list[Any] | None]:
        rendered = [normalize_message(message) for message in messages]
        if not teacher_context:
            return rendered, tools
        if rendered and rendered[-1].get("role") == "user":
            last = dict(rendered[-1])
            last["content"] = self.template.format(prompt=str(last.get("content") or ""), context=teacher_context)
            return [*rendered[:-1], last], tools
        reprompt = self.template.format(prompt="", context=teacher_context)
        return [*rendered, {"role": "user", "content": reprompt}], tools


__all__ = ["DEFAULT_REPROMPT_TEMPLATE", "SDPOProcessor", "reprompt_template"]
