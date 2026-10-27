"""GEPA's reflective mutation prompt, its record rendering, and its extractor.

This is the one place the method reproduces upstream text rather than
upstream behaviour, so the search this method runs is the published one.
``reflection_prompt.md`` holds ``InstructionProposalSignature.default_prompt_template``
copied verbatim from GEPA 0.1.2, and ``PROMPT_TEMPLATE`` is its text;
``render_prompt`` reproduces that class's ``format_samples`` markdown;
``extract_new_text`` reproduces its ``output_extractor``. Nothing here imports
``gepa``: the method carries no runtime dependency on the package it
reimplements.

    Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
    https://github.com/gepa-ai/gepa - MIT License

The record shape the renderer is given is GEPA's reflective dataset: one
mapping per minibatch example, conventionally ``Inputs`` / ``Generated
Outputs`` / ``Feedback``. Keys are rendered in insertion order, so the
caller's ordering is what the reflection model reads.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: GEPA 0.1.2 ``InstructionProposalSignature.default_prompt_template``, verbatim. The file ends with a newline that
#: the template does not have.
PROMPT_TEMPLATE = Path(__file__).with_name("reflection_prompt.md").read_text(encoding="utf-8").removesuffix("\n")

#: Markdown headers deepen with nesting and stop at ``######``: the same
#: cap GEPA applies, so deeply nested feedback values stay renderable.
MAX_HEADER_LEVEL = 6


def render_value(value: Any, level: int = 3) -> str:
    """One record value as markdown: a mapping or a list as headers at ``level``, anything else as its text."""
    if isinstance(value, Mapping):
        rendered = "".join(
            f"{'#' * level} {key}\n{render_value(item, min(level + 1, MAX_HEADER_LEVEL))}"
            for key, item in value.items()
        )
        return rendered or "\n"
    if isinstance(value, (list, tuple)):
        rendered = "".join(
            f"{'#' * level} Item {index + 1}\n{render_value(item, min(level + 1, MAX_HEADER_LEVEL))}"
            for index, item in enumerate(value)
        )
        return rendered or "\n"
    return f"{str(value).strip()}\n\n"


def render_prompt(current_text: str, records: Sequence[Mapping[str, Any]]) -> str:
    """The reflection prompt: the component's current text, then the records as numbered examples.

    The records go in as they are, without ``untrusted_text``, so that the prompt stays upstream's word for
    word. Run GEPA on traffic that you trust, or with ``evolution.publish: review``.
    """
    side_info = "\n\n".join(
        f"# Example {number}\n" + "".join(f"## {key}\n{render_value(value)}" for key, value in record.items())
        for number, record in enumerate(records, start=1)
    )
    return PROMPT_TEMPLATE.replace("<curr_param>", current_text).replace("<side_info>", side_info)


def extract_new_text(reply: str) -> str:
    """The instruction inside the reply's fences, GEPA's ``output_extractor``.

    The reply is asked for a fenced block, so the text between the first and
    last fence wins, with the leading language tag dropped. Models that open
    a fence and never close it, or close one they never opened, are common
    enough that both halves have their own fallback; a reply with no fence at
    all is taken whole.
    """
    start = reply.find("```") + 3
    end = reply.rfind("```")
    if start >= end:
        stripped = reply.strip()
        if stripped.startswith("```"):
            opening = re.match(r"^```\S*\n?", reply)
            if opening is not None:
                return reply[opening.end() :].strip()
        elif stripped.endswith("```"):
            return stripped[:-3].strip()
        return stripped
    content = reply[start:end]
    language = re.match(r"^\S*\n", content)
    if language is not None:
        content = content[language.end() :]
    return content.strip()


__all__ = ["PROMPT_TEMPLATE", "extract_new_text", "render_prompt"]
