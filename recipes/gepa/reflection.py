"""GEPA's reflective mutation prompt, its record rendering, and its extractor.

This is the one place the method reproduces upstream text rather than
upstream behaviour, so the search this recipe runs is the published one.
``PROMPT_TEMPLATE`` is ``InstructionProposalSignature.default_prompt_template``
copied verbatim from GEPA 0.1.2; ``render_prompt`` reproduces that class's
``format_samples`` markdown; ``extract_new_text`` reproduces its
``output_extractor``. Nothing here imports ``gepa``: the method carries no
runtime dependency on the package it reimplements.

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
from typing import Any

#: GEPA 0.1.2 ``InstructionProposalSignature.default_prompt_template``, verbatim.
PROMPT_TEMPLATE = """I provided an assistant with the following instructions to perform a task for me:
```
<curr_param>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<side_info>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."""

#: The two slots the template fills. A custom template missing either one
#: would silently drop the current instruction or the whole reflective
#: dataset, so a template is checked before it is ever sent to a model.
PLACEHOLDERS = ("<curr_param>", "<side_info>")

#: Markdown headers deepen with nesting and stop at ``######``: the same
#: cap GEPA applies, so deeply nested feedback values stay renderable.
_MAX_HEADER_LEVEL = 6


def _render_value(value: Any, level: int = 3) -> str:
    if isinstance(value, Mapping):
        rendered = "".join(
            f"{'#' * level} {key}\n{_render_value(item, min(level + 1, _MAX_HEADER_LEVEL))}"
            for key, item in value.items()
        )
        return rendered or "\n"
    if isinstance(value, (list, tuple)):
        rendered = "".join(
            f"{'#' * level} Item {index + 1}\n{_render_value(item, min(level + 1, _MAX_HEADER_LEVEL))}"
            for index, item in enumerate(value)
        )
        return rendered or "\n"
    return f"{str(value).strip()}\n\n"


def _render_record(record: Mapping[str, Any], number: int) -> str:
    body = "".join(f"## {key}\n{_render_value(value)}" for key, value in record.items())
    return f"# Example {number}\n{body}"


def render_prompt(
    current_text: str,
    records: Sequence[Mapping[str, Any]],
    template: str = PROMPT_TEMPLATE,
) -> str:
    """The reflection prompt: the component's current text plus the records."""
    missing = [placeholder for placeholder in PLACEHOLDERS if placeholder not in template]
    if missing:
        raise ValueError(f"reflection template is missing placeholder(s): {', '.join(missing)}")
    side_info = "\n\n".join(_render_record(record, index + 1) for index, record in enumerate(records))
    return template.replace("<curr_param>", current_text).replace("<side_info>", side_info)


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
