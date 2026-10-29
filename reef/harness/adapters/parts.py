"""The parts of a harness tree: the paths a tree may hold for one adapter, and what each one is.

An adapter declares its tree in the ``parts`` section of its ``adapter.yaml``. Each
part names the tree path a method writes (``path``), what Reef checks the
file as (``format``), where the adapter's harness reads it (``renders``),
what the file holds (``holds``), whether a change to it always waits for
a person (``review``), and how a person's change to it merges with Reef's
(``merge``). A ``{name}`` in a path is one name that matches
``NAME_PATTERN``, such as a skill's name; a config part's name is one of the
descriptor's config targets, and its file merges into that target. A tree
path that no part matches is refused.

Model endpoints and credentials are never in a tree: the model binding writes
them when a tree is rendered for an episode.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any


class Format(StrEnum):
    """What Reef checks a part's file as, and which render step processes it.

    Values: ``text`` is any text. ``skill`` and ``command`` are text that the
    adapter's ``process_skill`` and ``process_command`` render steps
    process. ``code`` is code the harness loads in its process. ``config``
    is a JSON object merged into one of the adapter's config files. The
    ``native_*`` formats are the native loop's tools, hooks, graphs, agents
    and loop code.
    Purpose: validation chooses a file's checks by its part's format, and
    rendering chooses the file's render step by it.
    """

    TEXT = "text"
    SKILL = "skill"
    COMMAND = "command"
    CODE = "code"
    CONFIG = "config"
    NATIVE_TOOL = "native_tool"
    NATIVE_HOOK = "native_hook"
    NATIVE_GRAPH = "native_graph"
    NATIVE_AGENT = "native_agent"
    NATIVE_LOOP = "native_loop"


#: The names a ``{name}`` in a path takes, such as a skill's or a config target's: they become path segments.
NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"
#: The names of parts.
PART_NAME_PATTERN = r"[a-z][a-z0-9_]*"
PLACEHOLDER = "{name}"
PART_KEYS = frozenset({"path", "format", "renders", "holds", "review", "merge"})
#: How a file that a person and Reef both changed merges (``reef.harness.sync.merge_trees``): ``file`` refuses, and
#: ``lines`` merges the lines each side changed, refusing only where both changed the same lines.
MERGE_MODES = ("file", "lines")


class TreeLayoutError(ValueError):
    """An adapter's ``parts`` section is missing, malformed, or declares parts that conflict."""


@dataclass(frozen=True)
class Part:
    """One part of an adapter's harness tree.

    Values: ``name`` matches ``PART_NAME_PATTERN``. ``path`` is the tree path
    a method writes, relative, with ``{name}`` at most once, inside one path
    segment. ``format`` is what Reef checks the file as. ``renders`` is where
    the adapter's harness reads the file, relative to its root, with
    ``{name}`` exactly when ``path`` has it; it is ``None`` exactly for a
    ``config`` part, whose file merges into the config target its name
    names. ``holds`` says in one line what the file holds, for methods and
    people. ``review`` is True when a change to the part always waits for a
    person, whatever the deployment's ``evolution.review_parts`` says.
    ``merge`` is one of ``MERGE_MODES``.
    Purpose: ``parse_tree_layout`` makes one per entry of an adapter's ``parts``;
    validation, rendering and the host facts a method reads use it.
    """

    name: str
    path: str
    format: Format
    renders: str | None
    holds: str
    review: bool = False
    merge: str = "file"
    #: The full match pattern of ``path``.
    matcher: re.Pattern[str] = field(init=False, repr=False, compare=False)
    #: The full match pattern of ``renders``; ``None`` for a config part.
    render_matcher: re.Pattern[str] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(PART_NAME_PATTERN, self.name):
            raise TreeLayoutError(f"part name {self.name!r} must match {PART_NAME_PATTERN}")
        named = check_template(self.path, f"part {self.name!r} path")
        if (self.format is Format.CONFIG) != (self.renders is None):
            raise TreeLayoutError(f"part {self.name!r} needs renders exactly when its format is not config")
        if self.format is Format.CONFIG and not named:
            raise TreeLayoutError(f"config part {self.name!r} needs {PLACEHOLDER} in its path: the config target")
        if self.renders is not None and check_template(self.renders, f"part {self.name!r} renders") != named:
            raise TreeLayoutError(f"part {self.name!r} renders {PLACEHOLDER} exactly when its path has it")
        if not isinstance(self.holds, str) or not self.holds.strip():
            raise TreeLayoutError(f"part {self.name!r} needs holds: what its file holds, in one line")
        if self.merge not in MERGE_MODES:
            raise TreeLayoutError(f"part {self.name!r} merge must be one of {', '.join(MERGE_MODES)}")
        object.__setattr__(self, "matcher", template_pattern(self.path))
        object.__setattr__(self, "render_matcher", None if self.renders is None else template_pattern(self.renders))

    def match(self, path: str) -> str | None:
        """The name ``path`` holds in this part's ``{name}``, ``""`` for a part without one, or ``None`` when
        ``path`` is not a path of this part."""
        found = self.matcher.fullmatch(path)
        if found is None:
            return None
        return found.groupdict().get("name") or ""

    def match_render(self, path: str) -> str | None:
        """The name a rendered ``path`` holds in this part's ``{name}``, ``""`` for a part without one, or
        ``None`` when ``path`` is not where this part renders (always ``None`` for a config part)."""
        found = None if self.render_matcher is None else self.render_matcher.fullmatch(path)
        if found is None:
            return None
        return found.groupdict().get("name") or ""

    def tree_path(self, name: str = "") -> str:
        """The tree path of the file named ``name``: ``match(tree_path(name)) == name``."""
        return self.path.replace(PLACEHOLDER, name)

    def render_path(self, name: str = "") -> str:
        """Where the adapter's harness reads the file named ``name``; ValueError for a config part."""
        if self.renders is None:
            raise ValueError(f"config part {self.name!r} merges into a config target and renders no path")
        return self.renders.replace(PLACEHOLDER, name)

    def to_json(self) -> dict[str, Any]:
        """The part as a method reads it in ``material/host.json``."""
        return {
            "path": self.path,
            "format": self.format.value,
            "renders": self.renders,
            "holds": self.holds,
            "review": self.review,
        }


@dataclass(frozen=True)
class TreeLocation:
    """What one tree path is in an adapter's tree.

    Values: ``part`` is the part whose path the tree path matches, and
    ``name`` the name it holds in the part's ``{name}``, ``""`` for a part
    without one.
    Purpose: ``TreeLayout.locate`` makes one; validation and rendering read
    it to choose the checks and the rendered path.
    """

    part: Part
    name: str


@dataclass(frozen=True)
class TreeLayout:
    """The parts of one adapter's harness tree.

    Values: ``parts`` maps each part's name to the part, in declaration
    order. No tree path is a path of two parts, and at most one part has
    the ``config`` format.
    Purpose: an ``AdapterDescriptor`` holds its adapter's layout; ``locate``
    gives each tree path its part, and ``locate_render`` each rendered path.
    """

    parts: Mapping[str, Part]

    def __post_init__(self) -> None:
        if any(name != part.name for name, part in self.parts.items()):
            raise TreeLayoutError("a layout maps each part's name to the part")
        if sum(part.format is Format.CONFIG for part in self.parts.values()) > 1:
            raise TreeLayoutError("a tree takes at most one config part")
        for part in self.parts.values():
            example = part.tree_path("x")
            claimed = [other.name for other in self.parts.values() if other.match(example) is not None]
            if claimed != [part.name]:
                raise TreeLayoutError(f"parts {', '.join(claimed)} both take the path {example!r}")

    def locate(self, path: str) -> TreeLocation | None:
        """What ``path`` is in this tree, or ``None`` when no part takes it."""
        for part in self.parts.values():
            name = part.match(path)
            if name is not None:
                return TreeLocation(part, name)
        return None

    def part_names(self, paths: Iterable[str]) -> dict[str, str]:
        """The name of the part each of ``paths`` is, for each path a part takes; the step pages show it."""
        located = {path: self.locate(path) for path in paths}
        return {path: location.part.name for path, location in located.items() if location is not None}

    def locate_render(self, path: str) -> TreeLocation | None:
        """The part that renders to ``path`` and the name it holds, or ``None`` when no part renders there.

        A config part renders no path of its own. When two parts render to one path, such as Codex's skills and
        commands, the part declared first takes it: ``locate(part.tree_path(name))`` of the result renders to
        ``path`` again.
        """
        for part in self.parts.values():
            name = part.match_render(path)
            if name is not None:
                return TreeLocation(part, name)
        return None


def check_template(template: str, where: str) -> bool:
    """Whether ``template`` holds ``{name}``; TreeLayoutError when it is not a relative path with at most one
    ``{name}`` inside one segment and no other brace."""
    if not isinstance(template, str) or not template:
        raise TreeLayoutError(f"{where} must be a relative path")
    pure = PurePosixPath(template)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != template:
        raise TreeLayoutError(f"{where} {template!r} must be a normal relative path")
    count = template.count(PLACEHOLDER)
    if count > 1 or template.replace(PLACEHOLDER, "").count("{") or template.replace(PLACEHOLDER, "").count("}"):
        raise TreeLayoutError(f"{where} {template!r} may hold {PLACEHOLDER} once and no other brace")
    return count == 1


def template_pattern(template: str) -> re.Pattern[str]:
    """The full match pattern of a path template: ``{name}`` is one name of ``NAME_PATTERN``."""
    before, _, after = template.partition(PLACEHOLDER)
    if template.count(PLACEHOLDER) == 0:
        return re.compile(re.escape(template))
    return re.compile(re.escape(before) + f"(?P<name>{NAME_PATTERN})" + re.escape(after))


def parse_tree_layout(raw: object) -> TreeLayout:
    """Check one ``parts`` section: a mapping of part names to their ``path``, ``format``, ``renders`` (absent for
    a config part), ``holds``, and optional ``review`` and ``merge``."""
    if not isinstance(raw, Mapping) or not raw:
        raise TreeLayoutError("parts must be a mapping of part names to parts")
    parts: dict[str, Part] = {}
    for name, entry in raw.items():
        if not isinstance(entry, Mapping) or not {"path", "format", "holds"} <= set(entry) <= PART_KEYS:
            raise TreeLayoutError(
                f"part {name!r} needs path, format and holds, and takes renders, review and merge besides"
            )
        try:
            format = Format(entry["format"])
        except ValueError as exc:
            known = ", ".join(item.value for item in Format)
            raise TreeLayoutError(f"part {name!r} format must be one of {known}") from exc
        review = entry.get("review", False)
        if not isinstance(review, bool):
            raise TreeLayoutError(f"part {name!r} review must be true or false")
        parts[str(name)] = Part(
            str(name), entry["path"], format, entry.get("renders"), entry["holds"], review, entry.get("merge", "file")
        )
    return TreeLayout(parts)


__all__ = [
    "MERGE_MODES",
    "NAME_PATTERN",
    "Format",
    "Part",
    "TreeLayout",
    "TreeLayoutError",
    "TreeLocation",
    "parse_tree_layout",
]
