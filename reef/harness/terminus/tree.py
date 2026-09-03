"""Translate a rendered terminus tree into Terminus 2 behavior inputs.

Stdlib only, and free of any harbor import: this is the half of the runner
that loads anywhere, including CI without Docker, so the mapping from node
kind to agent seam is testable without the benchmark.

One node kind, one seam:

- ``config`` becomes Terminus 2 constructor arguments.
- ``rules``, ``skill`` and ``agent_command`` become instruction text, the
  commands named as user-invocable because Terminus 2 has no other command
  surface.
- ``code_extension`` becomes the :class:`ContextPolicy`, called before every
  model call the main loop makes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.errors import ReefError

CONFIG_PATH = "terminus/config.json"
RULES_PATH = "terminus/AGENTS.md"
CONTEXT_PREFIX = "terminus/context/"
SKILL_PREFIX = "terminus/skills/"
COMMAND_PREFIX = "terminus-commands/"
#: Where the adapter renders. ``terminus-commands`` is a sibling of
#: ``terminus``, not a child, so the tree is read from the episode root.
TREE_PREFIXES = ("terminus/", "terminus-commands/")
#: Written during the run, under the tree but not part of it.
RUNTIME_PREFIXES = ("terminus/sessions/", "terminus/trials/")


class TerminusTreeError(ReefError):
    """The rendered tree cannot drive Terminus 2."""


def load_tree(root: Path | str) -> dict[str, str]:
    """Read a rendered tree from the episode root, keyed as ``render_composition`` keys it.

    ``root`` is the episode root rather than the ``terminus`` directory,
    because ``terminus-commands`` sits beside it; reading one level down would
    drop every ``agent_command`` and rekey the rest.

    Only rendered paths are read back. The episode root also holds the task
    workspace and, once the run starts, the session and trial directories,
    none of which are the composition under evaluation.

    The runner and the agent both read the tree here rather than passing it
    through the agent's kwargs, so the agent behaves the same whether it is
    constructed in this process or another.
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


def _without_frontmatter(text: str) -> str:
    """A SKILL.md body without its YAML header.

    The render quirk writes ``name`` and ``description`` frontmatter so the
    file is a well-formed skill on disk. Terminus 2 takes this text as
    instruction rather than parsing it, so the delimiters would reach the
    model as literal noise.
    """
    if not text.startswith("---\n"):
        return text.strip()
    _, _, rest = text.partition("---\n")
    body, delimiter, tail = rest.partition("\n---\n")
    return tail.strip() if delimiter else body.strip()


def instruction_text(files: Mapping[str, str]) -> str:
    """The tree's rules, skills, and commands as one instruction supplement.

    Empty when the tree carries none of them, which is the stock Terminus 2
    instruction and therefore the measured baseline.
    """
    sections: list[str] = []
    rules = (files.get(RULES_PATH) or "").strip()
    if rules:
        sections.append(rules)
    for prefix, label in ((SKILL_PREFIX, ""), (COMMAND_PREFIX, "User-invocable. ")):
        for path in sorted(files):
            if path.startswith(prefix) and path.endswith("/SKILL.md"):
                body = _without_frontmatter(files[path])
                if body:
                    sections.append(f"{label}{body}" if label else body)
    return "\n\n".join(sections)


class ContextPolicy:
    """The tree's ``assemble`` seam: one module, compiled once, called per turn.

    The module defines ``assemble(state, request, files) -> messages``.
    ``state`` is this policy's own dict and persists across calls, so a policy
    can carry notes between turns; ``request`` is the pending call; ``files``
    is the whole tree, so a policy can read its own siblings.
    """

    def __init__(self, path: str, source: str, files: Mapping[str, str]) -> None:
        namespace: dict[str, Any] = {}
        try:
            exec(compile(source, path, "exec"), namespace)  # the tree is the program
        except Exception as exc:  # any import-time failure is a tree defect, not a Reef bug
            raise TerminusTreeError(f"{path} raised while loading: {exc}") from exc
        if not callable(namespace.get("assemble")):
            raise TerminusTreeError(f"{path} must define a callable assemble(state, request, files)")
        self.path = path
        self._namespace = namespace
        self._files = dict(files)
        self._state: dict[str, Any] = {}

    def assemble(self, request: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        """Rebuilt messages, or ``None`` to leave the stock assembly alone.

        A policy that raises returns ``None``: a defective candidate degrades
        to stock behavior and still earns a score, instead of killing the
        trial and starving the gate of evidence.
        """
        messages = self._namespace["assemble"](self._state, dict(request), self._files)
        if not isinstance(messages, list) or not messages:
            return None
        if not all(isinstance(message, dict) and "role" in message for message in messages):
            return None
        return messages


def context_policy(files: Mapping[str, str]) -> ContextPolicy | None:
    """The tree's context module, or ``None`` when it ships none."""
    modules = sorted(path for path in files if path.startswith(CONTEXT_PREFIX) and path.endswith(".py"))
    if not modules:
        return None
    if len(modules) > 1:
        raise TerminusTreeError(f"terminus admits one context module; got {', '.join(modules)}")
    return ContextPolicy(modules[0], files[modules[0]], files)
