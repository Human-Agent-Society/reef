"""Antigravity adapter render steps: ensure SKILL.md has valid YAML frontmatter."""

from __future__ import annotations

from typing import Any
import yaml

from reef.harness.adapters.descriptor import AdapterRenderer


def _with_frontmatter(path: str, text: str) -> str:
    """Ensure every skill has valid YAML frontmatter for Antigravity."""
    if text.startswith("---\n"):
        return text
    name = path.split("/")[-2] if "/" in path else "skill"
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header: dict[str, Any] = {"name": name, "description": first[:200] or name}
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


class AntigravityAdapterRenderer(AdapterRenderer):
    """Antigravity render steps."""

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        """Inject required YAML frontmatter if missing from SKILL.md."""
        return _with_frontmatter(path, text)
