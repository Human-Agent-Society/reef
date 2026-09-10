"""Translate a rendered terminus tree into Harbor's native agent inputs.

Stdlib only, and free of any Harbor import: this is the half of the runner
that loads anywhere, including CI without Docker, so the mapping from node
kind to agent input is testable without the benchmark.

Harbor accepts the declarative nodes as native inputs:

- ``config`` becomes Terminus 2 constructor arguments.
- ``rules`` becomes a trial ``extra_instruction_paths`` entry.
- ``skill`` and ``agent_command`` become ``AgentConfig.skills`` roots, which
  keeps Harbor's progressive skill loading instead of pasting every skill
  body into the prompt.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.errors import ReefError

CONFIG_PATH = "terminus/config.json"
RULES_PATH = "terminus/AGENTS.md"
SKILL_ROOT = "terminus/skills"
COMMAND_ROOT = "terminus-commands"
CONTEXT_ROOT = "terminus/context/"
ENVIRONMENT_ENV = "REEF_TERMINUS_ENVIRONMENT"
#: Where the adapter renders. ``terminus-commands`` is a sibling of
#: ``terminus``, not a child, so the tree is read from the episode root.
TREE_PREFIXES = ("terminus/", "terminus-commands/")
#: Written during the run, under the tree but not part of it.
RUNTIME_PREFIXES = ("terminus/sessions/", "terminus/trials/")


class TerminusTreeError(ReefError):
    """The rendered tree cannot drive Terminus 2."""


def extension_source(files: Mapping[str, str]) -> tuple[str, str] | None:
    """Validate one self-contained Agent module without importing or executing it."""
    modules = [(path, code) for path, code in files.items() if path.startswith(CONTEXT_ROOT)]
    if not modules:
        return None
    if len(modules) != 1:
        raise TerminusTreeError("terminus accepts exactly one code_extension defining class Agent")
    path, code = modules[0]
    name = path.removeprefix(CONTEXT_ROOT)
    if "/" in name or not name.endswith(".py") or not name[:-3].isidentifier():
        raise TerminusTreeError("terminus code_extension must have a Python identifier name")
    try:
        module = ast.parse(code, filename=path)
        compile(module, path, "exec")
    except (SyntaxError, ValueError) as exc:
        raise TerminusTreeError(f"invalid terminus code_extension: {exc}") from exc
    if not any(isinstance(node, ast.ClassDef) and node.name == "Agent" for node in module.body):
        raise TerminusTreeError("terminus code_extension must define class Agent inheriting Harbor's Terminus2")
    return path, code


def load_tree(root: Path | str) -> dict[str, str]:
    """Read a rendered tree from the episode root, keyed as ``render_composition`` keys it.

    ``root`` is the episode root rather than the ``terminus`` directory,
    because ``terminus-commands`` sits beside it; reading one level down would
    drop every ``agent_command`` and rekey the rest.

    Only rendered paths are read back. The episode root also holds the task
    workspace and, once the run starts, the session and trial directories,
    none of which are the composition under evaluation.
    """
    base = Path(root)
    if not base.is_dir():
        raise TerminusTreeError(f"terminus tree root {base} is not a directory")
    tree = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        key = path.relative_to(base).as_posix()
        if key.startswith(TREE_PREFIXES) and not key.startswith(RUNTIME_PREFIXES):
            tree[key] = path.read_text(encoding="utf-8")
    return tree


def terminus_kwargs(files: Mapping[str, str]) -> dict[str, Any]:
    """Constructor arguments from ``terminus/config.json``.

    The render quirk already rejected keys that are not Terminus 2 arguments,
    so this only has to parse; a tree that reaches here with a broken config
    was not rendered by Reef.
    """
    raw = files.get(CONFIG_PATH)
    if raw is None:
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TerminusTreeError(f"{CONFIG_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise TerminusTreeError(f"{CONFIG_PATH} must be an object")
    return config


def instruction_paths(root: Path | str, files: Mapping[str, str]) -> list[Path]:
    """The rules file Harbor should append to the task instruction, if any."""
    return [Path(root) / RULES_PATH] if (files.get(RULES_PATH) or "").strip() else []


def skill_roots(root: Path | str, files: Mapping[str, str]) -> list[Path]:
    """The skill roots Harbor should load, skipping any the tree left empty.

    Harbor rejects an empty root, and a composition that carries no skill and
    no command is the stock agent rather than an error.
    """
    base = Path(root)
    return [
        base / prefix
        for prefix in (SKILL_ROOT, COMMAND_ROOT)
        if any(key.startswith(f"{prefix}/") and key.endswith("/SKILL.md") for key in files)
    ]
