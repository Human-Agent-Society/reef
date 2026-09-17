"""SFT reported-feedback processor: one report, one supervised sample.

The demonstration becomes the assistant turn of the recorded request, rendered
with the served model's chat template so the trained tokens are exactly the
model's own rendering of that answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.chat_request import normalize_messages_for_template, recorded_request
from reef.core.reports import TeacherContextReport
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem


class DemonstrationTokenizer(ABC):
    """Render a request and its demonstration into prompt ids and the demonstration's response ids."""

    @abstractmethod
    def sequence_ids(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None, demonstration: str
    ) -> tuple[list[int], list[int]]:
        """``(prompt_ids, response_ids)``: the request with the generation prompt, then the assistant turn."""


class ChatTemplateDemonstrationTokenizer(DemonstrationTokenizer):
    """The served model's Hugging Face tokenizer applying its own chat template.

    The demonstration becomes the assistant message of the conversation; its
    response ids are what the template adds after the generation prompt (the
    content, the end-of-turn token and the template's turn separator), so
    the trained sequence is exactly the model's own rendering of that answer.
    """

    def __init__(self, tokenizer_path: str) -> None:
        # transformers belongs to the training environment; importing it here
        # keeps ``import recipes.sft`` light for the service.
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def sequence_ids(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None, demonstration: str
    ) -> tuple[list[int], list[int]]:
        chat = normalize_messages_for_template(messages)
        template_tools = list(tools) if tools else None
        prompt = self._tokenizer.apply_chat_template(
            chat, tools=template_tools, tokenize=False, add_generation_prompt=True
        )
        conversation = self._tokenizer.apply_chat_template(
            [*chat, {"role": "assistant", "content": demonstration}],
            tools=template_tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not conversation.startswith(prompt):
            raise ValueError(
                "the chat template does not render the generation prompt as a prefix of the assistant turn"
            )
        prompt_ids = [int(token) for token in self._tokenizer(prompt, add_special_tokens=False)["input_ids"]]
        response_ids = [
            int(token) for token in self._tokenizer(conversation[len(prompt) :], add_special_tokens=False)["input_ids"]
        ]
        return prompt_ids, response_ids


class SFTProcessor(ReportedFeedbackProcessor):
    """The recorded request followed by its demonstration as the assistant turn.

    The student's own response is recorded and ignored: the sample the trainer
    sees is the request rendered with the served model's chat template and the
    demonstration's tokens, and only those tokens carry loss.
    """

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext, tokenizer: DemonstrationTokenizer | None = None) -> None:
        self._assembly = SampleAssembly.from_config(context)
        if tokenizer is None:
            tokenizer_path = str(context.config.get("tokenizer_path", "")).strip()
            if not tokenizer_path:
                raise ValueError("SFT requires tokenizer_path: the served model's tokenizer renders the sample")
            tokenizer = ChatTemplateDemonstrationTokenizer(tokenizer_path)
        self._tokenizer = tokenizer
        super().__init__(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        parsed = context.parsed_report
        if not isinstance(parsed, TeacherContextReport):
            raise ValueError("SFTProcessor requires the TeacherContextReport schema")
        if len(context.inferences) != 1:
            raise ValueError("SFT trains one recorded request per report")
        sample = self._assembly.build(context, 0.0 if context.score is None else context.score)
        messages, tools = recorded_request(context.inferences[0].payload)
        prompt_ids, response_ids = self._tokenizer.sequence_ids(messages, tools, parsed.context)
        if not response_ids:
            raise ValueError("the demonstration rendered to no tokens")
        # The student's sample gives way to the demonstration; the token
        # spans that described the student's response no longer apply.
        return sample.with_training(
            tokens=[*prompt_ids, *response_ids],
            loss_mask=[1] * len(response_ids),
            rollout_log_probs=[],
            runtime_load_spans=[],
        )

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:sft:{batch_number}", items)
