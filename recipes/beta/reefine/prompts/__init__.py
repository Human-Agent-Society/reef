"""Reefine's prompts: each ``<name>.md`` file in this directory is one ``str.format`` template.

A template names its fields in single braces; doubled braces stand for
literal ones. The final newline of a file is not part of its text: the code
that joins templates adds the space or the blank line between them. A
``<name>.yaml`` file holds word sets that templates take in their fields:
each top level key names one set, which maps a field to its text.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def prompt(name: str) -> str:
    """The template in ``<name>.md``, without the file's final newline."""
    return Path(__file__).with_name(f"{name}.md").read_text(encoding="utf-8").removesuffix("\n")


def prompt_words(name: str) -> dict[str, dict[str, str]]:
    """The word sets in ``<name>.yaml``, by set name; each set maps a field to its text."""
    return yaml.safe_load(Path(__file__).with_name(f"{name}.yaml").read_text(encoding="utf-8"))


__all__ = ["prompt", "prompt_words"]
