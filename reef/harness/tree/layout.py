"""The source tree layout: which paths a harness tree may hold, and what each one is.

Every file of a source tree has one kind. The kind decides where an adapter
renders the file (the descriptor's ``files:`` section names one path template
per kind) and which checks its text must pass::

    rules.md                      rules            the rules every session reads
    config/<target>.json          config           a JSON object merged into one config target
    skills/<name>/SKILL.md        skill            an Agent Skill
    commands/<name>.md            agent_command    a prompt template a person types as a command
    extensions/<name>.<ext>       code_extension   code the harness loads in process; ext follows the adapter
    native/tools/<name>.py        native_tool      a tool of the native loop
    native/hooks/<name>.py        native_hook      a listener at one event of the native loop
    native/graphs/<name>.json     native_graph     the native loop's control flow
    native/agents/<name>.json     native_agent     one agent the native loop delegates to
    native/loops/<name>.py        native_loop      the native loop itself as code

Model endpoints and credentials are never in a tree: the model binding writes
them when a tree is rendered for an episode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath


class TreeKind(StrEnum):
    """The kinds of file a source tree holds.

    Values: exactly the ten members below. Each value equals the key of the
    adapter descriptor's ``files:`` section that renders it, except
    ``config``, which renders through the descriptor's config targets.
    Purpose: ``locate`` gives each tree path one kind; validation and
    rendering choose their checks and target paths by kind.
    """

    RULES = "rules"
    CONFIG = "config"
    SKILL = "skill"
    AGENT_COMMAND = "agent_command"
    CODE_EXTENSION = "code_extension"
    NATIVE_TOOL = "native_tool"
    NATIVE_HOOK = "native_hook"
    NATIVE_GRAPH = "native_graph"
    NATIVE_AGENT = "native_agent"
    NATIVE_LOOP = "native_loop"


#: The names a skill, command, extension, config target or native file may take: they become path segments.
NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"
#: The file suffixes a code extension may have, one per adapter template.
EXTENSION_SUFFIXES = (".ts", ".mjs", ".js", ".py")

LAYOUT = {
    TreeKind.RULES: re.compile(r"rules\.md"),
    TreeKind.CONFIG: re.compile(rf"config/(?P<name>{NAME_PATTERN})\.json"),
    TreeKind.SKILL: re.compile(rf"skills/(?P<name>{NAME_PATTERN})/SKILL\.md"),
    TreeKind.AGENT_COMMAND: re.compile(rf"commands/(?P<name>{NAME_PATTERN})\.md"),
    # The name is matched lazily, so the suffix is the last one: "a.b.ts" is the extension "a.b".
    TreeKind.CODE_EXTENSION: re.compile(r"extensions/(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*?)(?P<suffix>\.ts|\.mjs|\.js|\.py)"),
    TreeKind.NATIVE_TOOL: re.compile(rf"native/tools/(?P<name>{NAME_PATTERN})\.py"),
    TreeKind.NATIVE_HOOK: re.compile(rf"native/hooks/(?P<name>{NAME_PATTERN})\.py"),
    TreeKind.NATIVE_GRAPH: re.compile(rf"native/graphs/(?P<name>{NAME_PATTERN})\.json"),
    TreeKind.NATIVE_AGENT: re.compile(rf"native/agents/(?P<name>{NAME_PATTERN})\.json"),
    TreeKind.NATIVE_LOOP: re.compile(rf"native/loops/(?P<name>{NAME_PATTERN})\.py"),
}
"""One full-match pattern per kind; a tree path is on the layout if and only if exactly one matches it."""

#: Kinds whose change always waits for a person: loop code runs in the loop process with its privileges.
ALWAYS_REVIEWED_KINDS = frozenset({TreeKind.NATIVE_LOOP})


@dataclass(frozen=True)
class TreeLocation:
    """What one tree path is on the layout.

    Values: ``kind`` is the path's kind. ``name`` is the skill, command,
    extension or native file name, or the config target, matching
    ``NAME_PATTERN``; it is ``""`` for ``rules``, and only then. ``suffix``
    is one of ``EXTENSION_SUFFIXES`` for a ``code_extension`` and ``""`` for
    every other kind.
    Purpose: ``locate`` makes one from a path; validation and rendering read
    it to choose the checks and the adapter path.
    """

    kind: TreeKind
    name: str
    suffix: str = ""

    def __post_init__(self) -> None:
        if (self.name == "") != (self.kind is TreeKind.RULES):
            raise ValueError(f"a {self.kind} location needs a name exactly when it is not rules")
        if self.name and not re.fullmatch(NAME_PATTERN, self.name):
            raise ValueError(f"tree name {self.name!r} must match {NAME_PATTERN}")
        if (self.suffix != "") != (self.kind is TreeKind.CODE_EXTENSION):
            raise ValueError("a location carries a suffix exactly when it is a code_extension")
        if self.suffix and self.suffix not in EXTENSION_SUFFIXES:
            raise ValueError(f"code extension suffix {self.suffix!r} must be one of {', '.join(EXTENSION_SUFFIXES)}")


def locate(path: str) -> TreeLocation | None:
    """What ``path`` is on the layout, or ``None`` when no kind's pattern matches it."""
    for kind, pattern in LAYOUT.items():
        match = pattern.fullmatch(path)
        if match is None:
            continue
        groups = match.groupdict()
        return TreeLocation(kind, groups.get("name") or "", groups.get("suffix") or "")
    return None


def source_path(location: TreeLocation) -> str:
    """The tree path of ``location``: ``locate(source_path(location)) == location``."""
    templates = {
        TreeKind.RULES: "rules.md",
        TreeKind.CONFIG: "config/{name}.json",
        TreeKind.SKILL: "skills/{name}/SKILL.md",
        TreeKind.AGENT_COMMAND: "commands/{name}.md",
        TreeKind.CODE_EXTENSION: "extensions/{name}{suffix}",
        TreeKind.NATIVE_TOOL: "native/tools/{name}.py",
        TreeKind.NATIVE_HOOK: "native/hooks/{name}.py",
        TreeKind.NATIVE_GRAPH: "native/graphs/{name}.json",
        TreeKind.NATIVE_AGENT: "native/agents/{name}.json",
        TreeKind.NATIVE_LOOP: "native/loops/{name}.py",
    }
    return templates[location.kind].format(name=location.name, suffix=location.suffix)


def extension_suffix(template: str) -> str:
    """The code extension suffix an adapter's ``code_extension`` template renders, such as ``.ts``."""
    suffix = PurePosixPath(template).suffix
    if suffix not in EXTENSION_SUFFIXES:
        raise ValueError(f"code_extension template {template!r} must end in one of {', '.join(EXTENSION_SUFFIXES)}")
    return suffix


__all__ = [
    "ALWAYS_REVIEWED_KINDS",
    "EXTENSION_SUFFIXES",
    "LAYOUT",
    "NAME_PATTERN",
    "TreeKind",
    "TreeLocation",
    "extension_suffix",
    "locate",
    "source_path",
]
