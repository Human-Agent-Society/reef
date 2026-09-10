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

One ``code_extension`` can define an Agent subclass of Harbor's Terminus2.
Rendering only checks its syntax. Execution requires Reef's sandbox around
the runner and Harbor's remote E2B environment for the terminal task: the
task container alone does not isolate evolved Python in the outer runner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import yaml

from reef.harness.episodes.executor import EpisodeExecutor, EpisodeLaunchError, SandboxExecutor
from reef.harness.runners.terminus.tree import ENVIRONMENT_ENV, TerminusTreeError, extension_source
from reef.harness.tree.render import RenderError

_CONFIG = "terminus/config.json"
_SKILL_ROOTS = ("terminus/skills/", "terminus-commands/")

#: Terminus 2 constructor arguments a tree may set. Verified against harbor
#: 0.20.0, which reef-eval 0.1.1 resolves; a bump should re-check the signature.
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

    try:
        extension_source(files)
    except TerminusTreeError as exc:
        raise RenderError(str(exc)) from exc

    for path, text in list(files.items()):
        if path.startswith(_SKILL_ROOTS) and path.endswith("/SKILL.md"):
            files[path] = _with_frontmatter(path, text)

    return files


def validate_execution(files: Mapping[str, str], executor: EpisodeExecutor) -> None:
    """Refuse unisolated Python and local Docker nested in Reef's jail."""
    try:
        extension = extension_source(files)
    except TerminusTreeError as exc:
        raise EpisodeLaunchError(str(exc)) from exc
    if isinstance(executor, SandboxExecutor):
        if executor.env.get(ENVIRONMENT_ENV) != "e2b":
            raise EpisodeLaunchError(
                "terminus Docker cannot run under evolution.executor: sandbox; "
                "set REEF_TERMINUS_ENVIRONMENT=e2b and include it in evolution.sandbox.env_from"
            )
        if not executor.egress_hosts or not executor.env.get("E2B_API_KEY"):
            raise EpisodeLaunchError("sandboxed terminus requires egress_hosts and E2B_API_KEY in sandbox.env_from")
    elif extension is not None:
        raise EpisodeLaunchError("terminus code_extension requires evolution.executor: sandbox with remote E2B tasks")
