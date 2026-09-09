"""pi adapter quirks: what the declarative descriptor cannot state.

pi is the friendlier of the two bundled harnesses - one environment variable
relocates its whole composition and it never mutates rendered files at boot -
so the quirks reduce to the files it may create beside the composition:
``trust.json`` (trust decisions) and ``auth.json`` (provider auth caches) can
appear in the agent directory even for an offline run.

pi follows the Agent Skills spec: a ``SKILL.md`` needs ``name`` and
``description`` frontmatter or the skill is reported as a conflict at startup
and never offered. A skill node whose text carries none gets both, the
description being the text's first line, the same synthesis codex and dsh do.
"""

from __future__ import annotations

from typing import Any

import yaml

cleanup_whitelist = (
    "pi-agent/trust.json",
    "pi-agent/auth.json",
)

_SKILLS = "pi-agent/skills/"


def _with_frontmatter(path: str, text: str) -> str:
    if text.startswith("---\n"):
        return text
    name = path.split("/")[-2]
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header: dict[str, Any] = {"name": name, "description": first[:200] or name}
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    """Give every skill the frontmatter pi requires when its text has none."""
    for path, text in list(files.items()):
        if path.startswith(_SKILLS) and path.endswith("/SKILL.md"):
            files[path] = _with_frontmatter(path, text)
    return files
