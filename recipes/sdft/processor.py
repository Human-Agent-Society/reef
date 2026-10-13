"""SDFT's stages: the shared distillation stages with the demonstration appended to the request."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.config import config_option
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2task.distill import DistillSettings, DistillTasks, normalize_messages_for_template
from reef.task2batch import DistillBatches

#: The reference implementation's demonstration block (idanshen/Self-Distillation, ``main.py``).
DEFAULT_CONTEXT_TEMPLATE = (
    "This is an example for a response to the question:\n{context}\n\n"
    "Now answer with a response of your own, including the thinking process."
)
CONTEXT_PLACEHOLDER = "{context}"


@dataclass(frozen=True)
class SDFTSettings(DistillSettings):
    """The config of ``SDFTTasks``: the shared distillation settings and the demonstration block's template."""

    context_template: str = config_option(
        DEFAULT_CONTEXT_TEMPLATE, help="The text added to the final user message; {context} marks the demonstration."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if CONTEXT_PLACEHOLDER not in self.context_template:
            raise ValueError(f"context_template must contain {CONTEXT_PLACEHOLDER}")


class SDFTTasks(DistillTasks):
    """One rollout and its demonstration, one training task.

    The teacher reads the student's request with the demonstration block
    appended to its final user message, as ``main.py`` of the reference
    implementation rewrites the question; an agent request that ends in a
    tool result gets the block as a new user message instead, so the
    demonstration still sits right before the response it conditions. The
    block is ``context_template`` with ``{context}`` replaced.
    """

    def __init__(self, settings: SDFTSettings) -> None:
        # The template is read before the tokenizer loads.
        self.template = settings.context_template
        super().__init__(settings)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(SDFTSettings.parse(config, plugin_class=cls))

    def teacher_request(
        self, messages: list[Any], tools: list[Any] | None, response: str, teacher_context: str
    ) -> tuple[list[Any], list[Any] | None]:
        block = self.template.replace(CONTEXT_PLACEHOLDER, teacher_context)
        rendered = normalize_messages_for_template(messages)
        if rendered and rendered[-1].get("role") == "user":
            last = dict(rendered[-1])
            content = str(last.get("content") or "")
            last["content"] = f"{content}\n\n{block}" if content else block
            return [*rendered[:-1], last], tools
        return [*rendered, {"role": "user", "content": block}], tools


class SDFTBatches(DistillBatches):
    """SDFT's batches: distillation tasks as they come, ``batch_size`` reports to a batch."""

    batch_label = "sdft"


__all__ = ["CONTEXT_PLACEHOLDER", "DEFAULT_CONTEXT_TEMPLATE", "SDFTBatches", "SDFTSettings", "SDFTTasks"]
