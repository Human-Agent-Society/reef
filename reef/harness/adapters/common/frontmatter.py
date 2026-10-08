"""The ``name`` and ``description`` frontmatter that most harnesses need to list a ``SKILL.md`` file."""

from __future__ import annotations

import yaml


def skill_frontmatter(path: str, text: str) -> dict[str, str]:
    """The ``name`` and ``description`` for a skill whose text has no frontmatter.

    ``path`` ends in ``<name>/SKILL.md``.

    - ``name`` is that directory's name.
    - ``description`` is the first non-empty line of ``text``, without its Markdown heading marks and cut to 200
      characters. When ``text`` is empty, it is the name.

    An adapter whose harness needs another header form uses these values and writes the header itself.
    """
    name = path.split("/")[-2]
    first_line = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    return {"name": name, "description": first_line[:200] or name}


def with_skill_frontmatter(path: str, text: str) -> str:
    """``text`` with the :func:`skill_frontmatter` header written as YAML between two ``---`` lines in front of it.

    Text that already starts with a ``---`` line is returned unchanged.
    """
    if text.startswith("---\n"):
        return text
    header = yaml.dump(skill_frontmatter(path, text), sort_keys=False, default_flow_style=False, allow_unicode=True)
    return "---\n" + header + "---\n" + text
