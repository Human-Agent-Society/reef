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
import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

from reef.core.artifact_ref import LiveWeightArtifactRef
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import exchange_messages
from reef.train.processors.common import flatten_content, make_policy_trajectory, recorded_request, recorded_response
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


class DistillProcessor(ReportedFeedbackProcessor):
    """One report, one distillation sample: the student's policy tensors plus its teacher sequence.

    A report carries ``teacher_context`` and references one request by default.
    ``accept_multi_turn_policy_samples`` enables one strictly aligned episode per report.
    Every turn must retain all response tokens and log probabilities from one policy release.
    The teacher reads the privileged initial request followed by the exact episode suffix.

    ``max_teacher_tokens`` limits the teacher sequence. Single-turn defaults release and count
    overflowing reports in ``teacher_overflow_reports``. Episode mode raises instead.
    Recipes compose the privileged request with :meth:`teacher_request` and set ``batch_label``.
    A recorded-token teacher sets ``renders_teacher_prompt=False`` and loads no tokenizer.
    """

    output_schema = TrainingBatch
    exclusive_sources = True
    batch_label = "teacher"
    renders_teacher_prompt = True

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        if config.get("accept_multi_turn_policy_samples") is True:
            if int(config.get("realign_threshold", 0)) != 0 or int(config.get("scaffold_tolerance", 0)) != 0:
                raise ValueError("episode distillation requires realign_threshold=0 and scaffold_tolerance=0")
            config = {**config, "realign_threshold": 0, "scaffold_tolerance": 0}
        context = context.with_config(config)
        self.assembly = SampleAssembly.from_config(context)
        self.max_teacher_tokens = int(config.get("max_teacher_tokens", 0))
        if self.max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")
        tokenizer_path = str(config.get("tokenizer_path", "")).strip()
        self.tokenizer = None
        if self.renders_teacher_prompt:
            if not tokenizer_path:
                raise ValueError("tokenizer_path is required: the served model's tokenizer renders the teacher prompt")
            # transformers belongs to the training environment; the service never renders a prompt.
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.overflow_reports: set[str] = set()
        self.overflow_count = 0
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
        return {**super().operational_metrics(), "teacher_overflow_reports": self.overflow_count}

    def recorded_sample(self, context: ReportContext, score: float) -> TrajectoryItem:
        """Assemble one report, retaining every sampled position in strict episode mode."""
        if len(context.inferences) != 1 and not self.assembly.accept_multi_turn:
            raise ValueError(
                f"a teacher sequence covers one recorded request per report; report "
                f"{context.report.agent_record_id} references {len(context.inferences)}"
            )
        selected_ids: list[int] = []
        selected_log_probs: list[float] = []
        if self.assembly.accept_multi_turn:
            releases: set[str] = set()
            runtime_load_ids: set[str] = set()
            previous_history: list[Mapping[str, Any]] | None = None
            for inference in context.inferences:
                request_messages, response_messages = exchange_messages(inference.payload)
                if previous_history is None and any(
                    message.get("role") == "assistant" for message in request_messages
                ):
                    raise ValueError("episode references must start before the first assistant turn")
                if previous_history is not None and (
                    request_messages[: len(previous_history)] != previous_history
                    or any(message.get("role") == "assistant" for message in request_messages[len(previous_history) :])
                ):
                    raise ValueError(
                        "episode references must include every assistant turn in one exact message history"
                    )
                previous_history = [*request_messages, *response_messages]
                turn = make_policy_trajectory(inference, score)
                turn_tokens = turn.training["tokens"]
                turn_mask = turn.training["loss_mask"]
                turn_log_probs = turn.training["rollout_log_probs"]
                if not 0 < len(turn_mask) < len(turn_tokens) or any(value != 1 for value in turn_mask):
                    raise ValueError("episode distillation requires every assistant response token to remain selected")
                if len(turn_log_probs) != len(turn_mask) or any(not math.isfinite(value) for value in turn_log_probs):
                    raise ValueError("episode distillation requires complete finite rollout_log_probs on every turn")
                artifact_ref = inference.artifact_ref
                runtime_load_id = turn.training["runtime_load_id"]
                if artifact_ref is None or not isinstance(runtime_load_id, str) or not runtime_load_id:
                    raise ValueError(
                        "episode distillation requires a known policy release and runtime load on every turn"
                    )
                if isinstance(artifact_ref, LiveWeightArtifactRef) and artifact_ref.runtime_load_id != runtime_load_id:
                    raise ValueError("episode runtime load disagrees with its inference receipt")
                releases.add(artifact_ref.release_id)
                runtime_load_ids.add(runtime_load_id)
                selected_ids.extend(turn_tokens[-len(turn_mask) :])
                selected_log_probs.extend(turn_log_probs)
            if len(releases) != 1 or len(runtime_load_ids) != 1:
                raise ValueError("episode distillation requires one policy release and runtime load across all turns")
        sample = self.assembly.build(context, score)
        tokens = sample.training["tokens"]
        loss_mask = sample.training["loss_mask"]
        response_length = len(loss_mask)
        if not 0 < response_length < len(tokens):
            raise ValueError("a teacher sequence requires the recorded prompt and response tokens of the inference")
        if self.assembly.accept_multi_turn:
            assembled_ids = [
                token for token, selected in zip(tokens[-response_length:], loss_mask, strict=True) if selected
            ]
            assembled_log_probs = [
                value
                for value, selected in zip(sample.training["rollout_log_probs"], loss_mask, strict=True)
                if selected
            ]
            if assembled_ids != selected_ids or assembled_log_probs != selected_log_probs:
                raise ValueError("episode assembly changed selected response tokens or rollout_log_probs")
        return sample

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
        if self.tokenizer is None:
            raise RuntimeError(f"{type(self).__name__} reads recorded ids and renders no teacher prompt")
        template_options: dict[str, Any] = {}
        if enable_thinking is not None:
            template_options["enable_thinking"] = enable_thinking
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
            if self.assembly.accept_multi_turn and len(prompt_ids) > max_prompt_tokens:
                raise ValueError("episode teacher prompt exceeds max_teacher_prompt_tokens; truncation is not allowed")
            prompt_ids = prompt_ids[:max_prompt_tokens]
        return [*prompt_ids, *(int(token) for token in response_ids)]

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        parsed = context.parsed_report
        if not isinstance(parsed, TeacherContextReport):
            raise ValueError(f"{type(self).__name__} requires the TeacherContextReport schema")
        # The teacher's distribution is the signal; a reported score is metadata only.
        sample = self.recorded_sample(context, 0.0 if context.score is None else context.score)
        payload = context.inferences[0].payload
        messages, tools = recorded_request(payload)
        teacher_messages, teacher_tools = self.teacher_request(
            messages, tools, recorded_response(context.inferences[-1].payload), parsed.teacher_context
        )
        response_length = len(sample.training["loss_mask"])
        teacher_tokens = self.teacher_tokens(
            teacher_messages, teacher_tools, sample.training["tokens"][-response_length:]
        )
        self.check_teacher_length(context, teacher_tokens)
        return sample.with_training(teacher_tokens=teacher_tokens)

    def check_teacher_length(self, context: ReportContext, teacher_tokens: Sequence[int]) -> None:
        """Reject episode overflow; release and count overflowing single-turn reports."""
        if self.max_teacher_tokens and len(teacher_tokens) > self.max_teacher_tokens:
            if self.assembly.accept_multi_turn:
                raise ValueError("episode teacher sequence exceeds max_teacher_tokens")
            self.overflow_reports.add(context.report.agent_record_id)
            logger.warning(
                "report %s skipped: its teacher sequence is %d tokens, over max_teacher_tokens %d",
                context.report.agent_record_id,
                len(teacher_tokens),
                self.max_teacher_tokens,
            )

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        # An overflowing report is its own group, so the group decision can release it.
        report_id = context.report.agent_record_id
        return (report_id if report_id in self.overflow_reports else None), None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        if key not in self.overflow_reports:
            raise ValueError(f"{type(self).__name__} groups only overflowing reports, got group key {key!r}")
        self.overflow_reports.discard(key)
        self.overflow_count += 1
        return GroupDecision.DISCARD

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:{self.batch_label}:{batch_number}", items)


__all__ = ["DistillProcessor", "normalize_messages_for_template", "normalize_tool_call"]
