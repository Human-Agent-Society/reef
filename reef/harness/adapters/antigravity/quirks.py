"""Antigravity adapter quirks: configuration invariants and skill frontmatter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import yaml

cleanup_whitelist = (
    "antigravity/**",
)

SETTINGS_PATH = "antigravity/settings.json"


def _with_frontmatter(path: str, text: str) -> str:
    """Ensure every skill has valid YAML frontmatter for Antigravity."""
    if text.startswith("---\n"):
        return text
    name = path.split("/")[-2] if "/" in path else "skill"
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header: dict[str, Any] = {"name": name, "description": first[:200] or name}
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


try:
    from reef.harness.adapters.descriptor import AdapterRenderer

    class AntigravityAdapterRenderer(AdapterRenderer):
        @staticmethod
        def process_skill(path: str, text: str) -> str:
            """Give every skill the YAML frontmatter required by Antigravity."""
            return _with_frontmatter(path, text)

except ImportError:
    def finalize_render(files: Mapping[str, str]) -> Mapping[str, str]:
        """Fallback finalize_render for Reef versions prior to AdapterRenderer."""
        result = dict(files)
        for path, text in list(result.items()):
            if "skills/" in path and path.endswith("/SKILL.md"):
                result[path] = _with_frontmatter(path, text)
        return result
