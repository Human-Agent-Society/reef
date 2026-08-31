"""SkillClaw's delivery layer: the pool catalog injected into every request.

``SkillCatalogModule`` is ported from the sealed campaign's harness surface
(reef/surface/harness/modules.py on the #223 branch, commit 0c81580e): the
paper-pinned OpenClaw catalog format, the eligibility rules, and the
``catalog_names`` inverse are unchanged. It is method-specific delivery, so
it lives in the method package, not in reef.

The one adaptation is the layer: the artifact this rebuild serves is the
rendered pi composition tree, so the module owns the ``pi-agent`` layer and
reads the pool from its ``skills/<name>/SKILL.md`` entries (the render path
of ``skill`` nodes). The injected section is byte-identical to the sealed
one: OpenClaw's ``## Skills (mandatory)`` system-prompt section
(``system-prompt.ts``: ``buildSkillsSection`` wrapping
``formatSkillsForPrompt`` or ``formatSkillsCompact``), listing each eligible
skill's frontmatter name and description plus the location where the agent's
read tool finds the body under ``public_root``. Bodies are never injected;
the consumer materializes the served tree on the agent's filesystem and the
model lazily reads at most one SKILL.md per task. A skill without
frontmatter, without a name, or without a description is not listed; neither
is one whose frontmatter sets ``disable-model-invocation``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import yaml

from reef.core.errors import ReefError
from reef.surface.skills import SkillLayer


class SkillCatalogModule(SkillLayer):
    """The LLM-read layer for a skill pool: many skills, catalog-injected."""

    layer = "pi-agent"
    skills_dir = "skills"
    skill_filename = "SKILL.md"

    _NAME_ENTRY = re.compile(r"<skill>\s*<name>(.*?)</name>", re.DOTALL)

    def __init__(self, *, public_root: str, max_chars: int = 30_000) -> None:
        if not public_root.strip():
            raise ValueError("public_root must name the directory where the agent reads skill bodies")
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self._public_root = public_root.rstrip("/")
        self._max_chars = max_chars

    def validate(self, files: Mapping[str, str]) -> None:
        """Every ``skills/`` entry must live in a skill directory with a
        SKILL.md; the layer's other rendered files (settings.json,
        models.json, ...) are the adapter's own and pass through."""
        directories = set()
        prefix = f"{self.skills_dir}/"
        for path in files:
            if not path.startswith(prefix):
                continue
            name, separator, rest = path[len(prefix) :].partition("/")
            if not separator or not name or not rest:
                raise ReefError(f"{self.layer}/{path} is not under a skill directory")
            directories.add(name)
        for name in sorted(directories):
            if f"{prefix}{name}/{self.skill_filename}" not in files:
                raise ReefError(f"{self.layer}/{prefix}{name}/ must contain {self.skill_filename}")

    def prepare_request(self, files: Mapping[str, str], path: str, payload: dict[str, Any]) -> dict[str, Any]:
        text = self.served_text(files)
        if text is None:
            return payload
        if path == "/v1/messages":
            system = payload.get("system")
            if system is None:
                return {**payload, "system": text}
            if isinstance(system, str):
                return {**payload, "system": f"{system}\n\n{text}"}
            return {**payload, "system": [*system, {"type": "text", "text": text}]}
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload
        for index, message in enumerate(messages):
            if isinstance(message, Mapping) and message.get("role") == "system":
                existing = _flatten_content(message.get("content"))
                patched = {**message, "content": f"{existing}\n\n{text}"}
                return {**payload, "messages": [*messages[:index], patched, *messages[index + 1 :]]}
        return {**payload, "messages": [{"role": "system", "content": text}, *messages]}

    def served_text(self, files: Mapping[str, str]) -> str | None:
        """The complete injected section for this tree, or None when empty."""
        skills = self._eligible_skills(files)
        if not skills:
            return None
        full = self._format_catalog(skills, describe=True)
        catalog = full if len(full) <= self._max_chars else self._format_catalog(skills, describe=False)
        return self._skills_section(catalog)

    def _eligible_skills(self, files: Mapping[str, str]) -> list[dict[str, str]]:
        """Catalog rows in ``skills/<name>/SKILL.md`` path order, ineligible skills dropped."""
        skills = []
        for path in sorted(files):
            parts = path.split("/")
            if len(parts) != 3 or parts[0] != self.skills_dir or parts[2] != self.skill_filename:
                continue
            front = _frontmatter(files[path])
            if front is None:
                continue
            name = str(front.get("name", "")).strip()
            description = str(front.get("description", "")).strip()
            if not name or not description or front.get("disable-model-invocation", False):
                continue
            skills.append({"name": name, "description": description})
        return skills

    def _format_catalog(self, skills: list[dict[str, str]], *, describe: bool) -> str:
        matches = "description" if describe else "name"
        lines = [
            "\n\nThe following skills provide specialized instructions for specific tasks.",
            f"Use the read tool to load a skill's file when the task matches its {matches}.",
            "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
            "",
            "<available_skills>",
        ]
        for skill in skills:
            location = f"{self._public_root}/{skill['name']}/{self.skill_filename}"
            lines.append("  <skill>")
            lines.append(f"    <name>{_escape_xml(skill['name'])}</name>")
            if describe:
                lines.append(f"    <description>{_escape_xml(skill['description'])}</description>")
            lines.append(f"    <location>{_escape_xml(location)}</location>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    @staticmethod
    def _skills_section(catalog: str) -> str:
        return "\n".join(
            [
                "## Skills (mandatory)",
                "Before replying: scan <available_skills> <description> entries.",
                "- If exactly one skill clearly applies: read its SKILL.md at <location> with `read`, then follow it.",
                "- If multiple could apply: choose the most specific one, then read/follow it.",
                "- If none clearly apply: do not read any SKILL.md.",
                "Constraints: never read more than one skill up front; only read after selecting.",
                "- When a skill drives external API writes, assume rate limits: prefer fewer larger writes, avoid tight one-item loops, serialize bursts when possible, and respect 429/Retry-After.",
                catalog.strip(),
                "",
            ]
        )

    @classmethod
    def catalog_names(cls, payload: Mapping[str, Any]) -> tuple[str, ...]:
        """Skill names listed in a post-transform recorded request.

        The inverse of prepare_request() for consumers of recorded traffic:
        recording happens after injection, so the names come from the
        ``<available_skills>`` entries in the payload's system content.
        """
        texts = []
        system = payload.get("system")
        if isinstance(system, str):
            texts.append(system)
        elif isinstance(system, list):
            texts.extend(
                block.get("text", "") for block in system if isinstance(block, Mapping) and block.get("type") == "text"
            )
        messages = payload.get("messages")
        if isinstance(messages, list):
            texts.extend(
                _flatten_content(message.get("content"))
                for message in messages
                if isinstance(message, Mapping) and message.get("role") == "system"
            )
        return tuple(_unescape_xml(match) for text in texts for match in cls._NAME_ENTRY.findall(text))


def _flatten_content(content: Any) -> str:
    """A message's text content as one string, list-of-parts form included."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, Mapping) and part.get("type") == "text"]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _frontmatter(text: str) -> Mapping[str, Any] | None:
    """A SKILL.md's YAML frontmatter mapping, or None when it has none."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    try:
        front = yaml.safe_load(text[3:end].strip()) or {}
    except yaml.YAMLError:
        front = {}
    return front if isinstance(front, Mapping) else None


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _unescape_xml(text: str) -> str:
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )
