"""Terminus adapter quirks: constructor knobs, skill frontmatter, one context module.

Terminus 2 takes its behavior from constructor arguments rather than a config
file it discovers, so ``terminus/config.json`` is a flat object whose keys must
each name a real Terminus 2 argument. An unknown key is a defect in the tree,
and failing at render is what keeps a gated change meaning what it says instead
of silently dropping a knob.

Skills carry the ``name`` and ``description`` frontmatter the instruction
builder reads, synthesized when an evolved node ships bare text, under both
skill roots. Terminus 2 has no slash-command surface, so ``agent_command``
renders under the second root and the runner names those skills as
user-invocable when it joins them.

``code_extension`` is the context seam: one module defining
``assemble(state, request, files)``, loaded in the runner's own process and
called before every model call the main loop makes. It is limited to one
module because the seam is a single call, and two modules would leave the
order between them undefined.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from reef.harness.render import RenderError

_CONFIG = "terminus/config.json"
_CONTEXT = "terminus/context/"
_SKILL_ROOTS = ("terminus/skills/", "terminus-commands/")

#: Terminus 2 constructor arguments a tree may set. Verified against harbor
#: 0.22.0; a harbor bump should re-check the signature.
_ALLOWED_KNOBS = {
    "enable_summarize",
    "interleaved_thinking",
    "llm_call_kwargs",
    "max_thinking_tokens",
    "max_turns",
    "parser_name",
    "proactive_summarization_threshold",
    "reasoning_effort",
    "temperature",
}
#: Set by Reef's model binding, which renders after the tree and therefore
#: wins the merge. Admitted here so the merged config validates.
_BINDING_KNOBS = {"api_base", "llm_kwargs", "model_name"}


def _with_frontmatter(path: str, text: str) -> str:
    if text.startswith("---\n"):
        return text
    name = path.split("/")[-2]
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header: dict[str, Any] = {"name": name, "description": first[:200] or name}
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


def _validate_config(config: dict[str, Any]) -> None:
    unknown = sorted(set(config) - _ALLOWED_KNOBS - _BINDING_KNOBS)
    if unknown:
        raise RenderError(f"terminus config sets keys that are not Terminus 2 arguments: {', '.join(unknown)}")
    turns = config.get("max_turns")
    if turns is not None and (isinstance(turns, bool) or not isinstance(turns, int) or turns < 1):
        raise RenderError("terminus config max_turns must be a positive integer")


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    try:
        config = json.loads(files[_CONFIG])
    except (KeyError, json.JSONDecodeError) as exc:
        raise RenderError("terminus primary config must be a JSON object") from exc
    if not isinstance(config, dict):
        raise RenderError("terminus primary config must be an object")
    _validate_config(config)

    modules = sorted(path for path in files if path.startswith(_CONTEXT) and path.endswith(".py"))
    if len(modules) > 1:
        raise RenderError(f"terminus admits one context module, the assemble seam; got {', '.join(modules)}")

    for path, text in list(files.items()):
        if path.startswith(_SKILL_ROOTS) and path.endswith("/SKILL.md"):
            files[path] = _with_frontmatter(path, text)

    return files
