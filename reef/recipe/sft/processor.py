"""The SFT stages: each report becomes one sample of a recorded request and its target response.

``SFTRecord2Record`` makes each report its own record set. ``SFTTasks``
renders the request and the target with the served model's chat template.
The loss mask selects the target tokens. The shared ``DistillBatches``
stage puts the tasks into batches.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self, cast

from reef.core.batches import TrajectoryItem
from reef.core.config import config_option
from reef.core.plugins import PluginSettings
from reef.core.reports import TeacherContextReport
from reef.core.trajectories import make_trajectory
from reef.pipeline_context import PipelineContext
from reef.record2record import ExclusiveReportRecord2Record, Record2Record, ReportContext
from reef.record2task import ReportTasks, SkippedReport
from reef.record2task.distill import normalize_messages_for_template
from reef.record2task.samples import recorded_request, recorded_response_message

logger = logging.getLogger(__name__)

#: ``demonstration`` trains the report's ``teacher_context``. ``response`` trains the recorded response.
SFT_TARGETS = ("demonstration", "response")


class SFTRecord2Record(ExclusiveReportRecord2Record):
    """Make each valid report its own record set, and never drop a set as stale. No config.

    The staleness check drops tokens that an older model version sampled. An
    SFT target is fixed text, not sampled tokens. Thus the stage keeps each
    held set until a batch uses it.
    """

    def drop_stale_sets(self, max_staleness: int) -> None:
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

    def __post_init__(self) -> None:
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the sample")
        if self.target not in SFT_TARGETS:
            raise ValueError(f"target must be one of: {', '.join(SFT_TARGETS)}")
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative (0 disables the limit)")


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
        from transformers import AutoTokenizer

        self.target = settings.target
        self.max_tokens = settings.max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_path, trust_remote_code=True)
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

    def sequence_ids(
        self, conversation: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None
    ) -> tuple[list[int], int]:
        """Render a conversation that ends with the target turn. Return its token IDs and the target token count."""
        template_tools = list(tools) if tools else None
        # tokenize=True and return_dict=False return a flat list of token IDs.
        prompt_ids = cast(
            list[int],
            self.tokenizer.apply_chat_template(
                list(conversation[:-1]),
                tools=template_tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            ),
        )
        sequence_ids = cast(
            list[int],
            self.tokenizer.apply_chat_template(
                list(conversation), tools=template_tools, tokenize=True, add_generation_prompt=False, return_dict=False
            ),
        )
        if sequence_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "the chat template does not render the generation prompt as a prefix of the assistant turn"
            )
        if len(sequence_ids) == len(prompt_ids):
            raise ValueError("the target response renders to no tokens")
        return [int(token) for token in sequence_ids], len(sequence_ids) - len(prompt_ids)

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


__all__ = ["SFT_TARGETS", "SFTRecord2Record", "SFTSettings", "SFTTasks"]
