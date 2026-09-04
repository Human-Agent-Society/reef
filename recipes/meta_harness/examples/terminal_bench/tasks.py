"""Read a Terminal-Bench task list.

Task ids arrive either on a command line (``--tasks a,b,c``) or in a file that
also carries provenance, so accept commas, newlines, and ``#`` comments from
one parser rather than letting each entry point disagree about the format.
"""

from __future__ import annotations

from pathlib import Path


def parse_tasks(text: str) -> tuple[str, ...]:
    """Split task ids from comma- or newline-separated text, dropping comments."""
    tasks: list[str] = []
    for line in text.splitlines():
        body = line.split("#", 1)[0]
        tasks.extend(task.strip() for task in body.split(",") if task.strip())
    return tuple(tasks)


def read_tasks(path: str | Path) -> tuple[str, ...]:
    """Read task ids from a file."""
    return parse_tasks(Path(path).read_text(encoding="utf-8"))
