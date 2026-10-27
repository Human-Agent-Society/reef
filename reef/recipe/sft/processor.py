"""The SFT stages: each report becomes one sample of a recorded request and its target response.

``SFTRecord2Record`` makes each report its own record set. ``SFTTasks``
renders the request and the target with the served model's chat template.
The loss mask selects the target tokens. ``SFTJobData`` puts the tasks into
job data, together with uploaded ATIF trajectories, which ``SFTTasks``
renders with every agent turn trained.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self, cast

from reef.core.config import config_option
from reef.core.job_data import TrajectoryItem
from reef.core.plugins import NoSettings, PluginSettings
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import atif_messages, make_trajectory
from reef.inference.chat import template_kwargs_of
from reef.pipeline_context import PipelineContext
from reef.record2record import ExclusiveReportRecord2Record, Record2Record, ReportContext
from reef.record2task import ReportTasks, SkippedReport
from reef.record2task.distill import normalize_messages_for_template
from reef.record2task.samples import recorded_request, recorded_response_message
from reef.task2job import DistillJobData

logger = logging.getLogger(__name__)

#: ``demonstration`` trains the report's ``teacher_context``. ``response`` trains the recorded response.
SFT_TARGETS = ("demonstration", "response")


class SFTRecord2Record(ExclusiveReportRecord2Record):
    """Make each valid report its own record set, and never drop a unit as stale. No config.

    The staleness check drops tokens that an older model version sampled. An
    SFT target is fixed text, not sampled tokens. Thus the stage keeps each
    held unit until job data uses it, and it refuses ``max_staleness``.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context)

    def drop_stale_units(self) -> None:
        return


@dataclass(frozen=True)
class SFTSettings(PluginSettings):
    """The config of ``SFTTasks``: the tokenizer, the target, and the maximum sample length."""

    tokenizer_path: str = config_option(
        "", help="The served model's tokenizer. It renders the request and the target."
    )
    target: str = config_option(
        "demonstration",
        help="The response to train: demonstration (the report's teacher_context) or response (the recorded response).",
    )
    max_tokens: int = config_option(
        0, help="The maximum tokens in a sample. Reef skips a longer report. 0 disables the limit."
    )
    chat_template_kwargs: Mapping[str, Any] | None = config_option(
        None,
        help=(
            "Chat template arguments for the render, for example {enable_thinking: false}. "
            "Set the inference handler's chat_template_kwargs to the same value."
        ),
    )

    def __post_init__(self) -> None:
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the sample")
        if self.target not in SFT_TARGETS:
            raise ValueError(f"target must be one of: {', '.join(SFT_TARGETS)}")
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative (0 disables the limit)")
        template_kwargs_of(self.chat_template_kwargs, "chat_template_kwargs")


class SFTTasks(ReportTasks):
    """Make one supervised sample from one report and its recorded request.

    The sample tokens are the recorded request with the generation prompt,
    then the target as the assistant turn. The loss mask selects the target
    tokens. With ``target: demonstration``, the target is the report's
    ``teacher_context`` and the stage ignores the recorded response. With
    ``target: response``, the target is the recorded response message with
    its tool calls, for example an imported dataset example. The stage skips
    a sample longer than ``max_tokens`` and counts it in ``sft_overflow_reports``.
    """

    def __init__(self, settings: SFTSettings) -> None:
        # transformers is part of the training environment. The service never renders a sample.
        super().__init__()
        from transformers import AutoTokenizer

        self.target = settings.target
        self.max_tokens = settings.max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_path, trust_remote_code=True)
        #: The chat template arguments of every render; a task-to-job stage that renders reuses them.
        self.chat_template_kwargs = template_kwargs_of(settings.chat_template_kwargs, "chat_template_kwargs")
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
        return cls(SFTSettings.parse(config, plugin_class=cls))

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {"sft_overflow_reports": self.overflow_count}

    def target_message(self, payload: Mapping[str, Any], report: TeacherContextReport) -> Mapping[str, Any]:
        """Return the assistant message that the sample trains."""
        if self.target == "demonstration":
            if not report.teacher_context:
                raise ValueError("target demonstration requires the report's teacher_context")
            return {"role": "assistant", "content": report.teacher_context}
        message = recorded_response_message(payload)
        if message is None:
            raise ValueError("target response requires the recorded response message of the inference")
        return {**message, "role": "assistant"}

    def render(
        self, conversation: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None, *, add_generation_prompt: bool
    ) -> list[int]:
        """The token IDs of ``conversation`` in the chat template, with the stage's template arguments."""
        # tokenize=True and return_dict=False return a flat list of token IDs.
        ids = self.tokenizer.apply_chat_template(
            list(conversation),
            tools=list(tools) if tools else None,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        return [int(token) for token in cast(list[int], ids)]

    def sequence_ids(
        self, conversation: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None
    ) -> tuple[list[int], int]:
        """Render a conversation that ends with the target turn. Return its token IDs and the target token count."""
        prompt_ids = self.render(conversation[:-1], tools, add_generation_prompt=True)
        sequence_ids = self.render(conversation, tools, add_generation_prompt=False)
        if sequence_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "the chat template does not render the generation prompt as a prefix of the assistant turn"
            )
        if len(sequence_ids) == len(prompt_ids):
            raise ValueError("the target response renders to no tokens")
        return sequence_ids, len(sequence_ids) - len(prompt_ids)

    def uploaded_sample(self, trajectory: TrajectoryItem) -> TrajectoryItem:
        """Render an uploaded ATIF conversation with the chat template, with every generated agent turn trained.

        An agent step with ``is_copied_context: true`` (ATIF's mark for a step copied from earlier context) is
        context: it renders, and it trains nothing. Every other agent step trains. The tokens are the conversation
        up to its last trained turn; the steps after it train nothing and are left out. The loss mask covers the
        tokens after the first trained turn's generation prompt: 1 on each trained turn's tokens, 0 on everything
        between them. ``ValueError``: ``is_copied_context`` is not a boolean; the trajectory has no agent step
        that trains, or its first trained step has no step before it; the chat template does not render each
        trained turn as a prefix of the whole conversation (a template that drops earlier turns' reasoning does
        not); or the sample is longer than ``max_tokens``.
        """
        steps = trajectory.trajectory.get("steps") or []
        conversation = normalize_messages_for_template(atif_messages(trajectory.trajectory))
        # atif_messages turns each agent step into exactly one assistant message, in step order.
        copied = [step.get("is_copied_context", False) for step in steps if step["source"] == "agent"]
        if any(not isinstance(value, bool) for value in copied):
            raise ValueError("ATIF is_copied_context must be a boolean")
        agent_turns = [index for index, message in enumerate(conversation) if message["role"] == "assistant"]
        trained = [index for index, is_copied in zip(agent_turns, copied, strict=True) if not is_copied]
        if not trained:
            raise ValueError(
                "an uploaded trajectory needs one or more agent steps to train; copied context steps train nothing"
            )
        if trained[0] == 0:
            raise ValueError("an uploaded trajectory needs a system or user step before its first agent step")
        conversation = conversation[: trained[-1] + 1]
        tokens = self.render(conversation, None, add_generation_prompt=False)
        spans: list[tuple[int, int]] = []
        for index in trained:
            # The turn's tokens are those between its generation prompt and its end, both prefixes of the whole.
            prompt_ids = self.render(conversation[:index], None, add_generation_prompt=True)
            turn_ids = self.render(conversation[: index + 1], None, add_generation_prompt=False)
            prefixes = tokens[: len(prompt_ids)] == prompt_ids and tokens[: len(turn_ids)] == turn_ids
            if not prefixes or len(turn_ids) <= len(prompt_ids):
                raise ValueError(
                    "the chat template does not render each turn as a prefix of the whole conversation, "
                    "so its agent turns cannot all train"
                )
            spans.append((len(prompt_ids), len(turn_ids)))
        if self.max_tokens and len(tokens) > self.max_tokens:
            raise ValueError(f"the sample has {len(tokens)} tokens, more than max_tokens {self.max_tokens}")
        prompt_length = spans[0][0]
        loss_mask = [0] * (len(tokens) - prompt_length)
        for start, end in spans:
            loss_mask[start - prompt_length : end - prompt_length] = [1] * (end - start)
        return trajectory.with_training(tokens=tokens, loss_mask=loss_mask, rollout_log_probs=[])

    def make_sample(self, context: ReportContext) -> TrajectoryItem | SkippedReport:
        parsed = context.parsed_report
        if not isinstance(parsed, TeacherContextReport):
            raise ValueError(f"{type(self).__name__} requires the TeacherContextReport schema")
        if len(context.inferences) != 1:
            raise ValueError(
                f"an SFT report must reference one recorded request; report "
                f"{context.report.agent_record_id} references {len(context.inferences)}"
            )
        inference = context.inferences[0]
        messages, tools = recorded_request(inference.payload)
        conversation = normalize_messages_for_template([*messages, self.target_message(inference.payload, parsed)])
        tokens, target_length = self.sequence_ids(conversation, tools)
        if self.max_tokens and len(tokens) > self.max_tokens:
            logger.warning(
                "report %s skipped: its SFT sample has %d tokens, more than max_tokens %d",
                context.report.agent_record_id,
                len(tokens),
                self.max_tokens,
            )
            self.overflow_count += 1
            return SkippedReport("sft_overflow")
        # A reported score is only metadata. All samples train with the same weight.
        sample = make_trajectory((inference,), 0.0 if context.score is None else context.score)
        return sample.with_training(tokens=tokens, loss_mask=[1] * target_length, rollout_log_probs=[])


class SFTJobData(DistillJobData):
    """``DistillJobData`` that also takes uploaded ATIF trajectories, which ``SFTTasks.uploaded_sample`` renders."""

    accepts_trajectories = True


__all__ = ["SFT_TARGETS", "SFTJobData", "SFTRecord2Record", "SFTSettings", "SFTTasks"]
