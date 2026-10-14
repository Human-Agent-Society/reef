"""What AutoEnvScaling rebuilds tasks from: recorded coding-agent sessions on Reef, and the repository each one changed.

A sources file is a JSON list; each entry names one session record and the repository it worked on::

    [{"record_id": "rec-1", "scenario": "sessions",
      "repository": {"url": "https://github.com/org/repo", "commit": "3f2c1ab", "path": "/app"}}]

``commit`` is the commit the session started from, ``path`` where the task's environment checks the
repository out. :func:`render_session` turns the record's payload into the text the agent reads.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")
# The agent reads the session from a file, so it gets nearly all of it; the budget only bounds a runaway record.
SESSION_BUDGET_CHARS = 5_000_000
RESULT_CHARS = 4_000
# A file the session read is the repository's content the change starts from: kept longer than other results.
READ_RESULT_CHARS = 12_000
CUT_CHARS = 200
CUT_MARKER = "\n[... cut to fit the session budget]"
# A session over budget is cut in this order; the user's words are never cut.
CUT_ORDER = ("[tool result]", "[tool call", "[assistant]")


@dataclass(frozen=True)
class Repository:
    """Where a session's repository lives, the commit it started from, and its checkout path in the environment."""

    url: str
    commit: str
    path: str

    def __post_init__(self) -> None:
        # The commit goes on a git command line; a hash never reads as an option.
        if not COMMIT_PATTERN.fullmatch(self.commit):
            raise ValueError(f"a repository's commit must match {COMMIT_PATTERN.pattern}")
        if not self.path.startswith("/"):
            raise ValueError("a repository's path must be an absolute path in the environment")


@dataclass(frozen=True)
class SessionSource:
    """One recorded session to rebuild: its inference record on Reef and the repository it changed."""

    record_id: str
    scenario: str
    repository: Repository


def load_sources(path: Path) -> tuple[SessionSource, ...]:
    """The sources a sources file lists, in order."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list) or not document:
        raise ValueError(f"{path} must hold a non-empty JSON list of sources")
    sources = []
    for position, entry in enumerate(document):
        where = f"{path}: source {position}"
        repository = entry.get("repository") if isinstance(entry, Mapping) else None
        if not isinstance(repository, Mapping):
            raise ValueError(f"{where} must be an object with a repository object")
        sources.append(
            SessionSource(
                record_id=string_field(entry, "record_id", where),
                scenario=string_field(entry, "scenario", where),
                repository=Repository(
                    url=string_field(repository, "url", where),
                    commit=string_field(repository, "commit", where),
                    path=string_field(repository, "path", where),
                ),
            )
        )
    return tuple(sources)


def string_field(document: Mapping[str, object], key: str, where: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} needs a non-empty string {key!r}")
    return value


def render_session(payload: Mapping[str, Any], *, budget_chars: int = SESSION_BUDGET_CHARS) -> str:
    """The session as text, cut to ``budget_chars``.

    The payload is an inference record's: its ``messages`` then its ``response``, each a turn whose content is
    text or Anthropic content blocks (``text``, ``tool_use``, ``tool_result``). A tool call's input keeps its
    first 4000 characters, and a tool result too (a file ``Read``, 12000). A session still over budget has its
    longest tool results cut to 200 characters, then its longest tool calls, then the agent's longest turns. The
    user's words are never cut, and a line already that short is left alone, so the cutting ends even when the
    budget cannot be met.
    """
    lines: list[str] = []
    tool_names: dict[str, str] = {}
    turns = [*payload["messages"], *([payload["response"]] if payload.get("response") else [])]
    for turn in turns:
        role = turn.get("role", "assistant")
        content = turn.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for block in blocks:
            kind = block.get("type")
            if kind == "text" and block["text"].strip():
                lines.append(f"[{role}] {block['text'].strip()}")
            elif kind == "tool_use":
                tool_names[block["id"]] = block["name"]
                tool_input = json.dumps(block["input"], ensure_ascii=False)
                lines.append(f"[tool call {block['name']}] {tool_input[:RESULT_CHARS]}")
            elif kind == "tool_result":
                result = block.get("content") or ""
                text = result if isinstance(result, str) else "\n".join(part.get("text", "") for part in result)
                limit = READ_RESULT_CHARS if tool_names.get(block["tool_use_id"]) == "Read" else RESULT_CHARS
                cut = f"\n[... {len(text) - limit} characters cut]" if len(text) > limit else ""
                lines.append(f"[tool result] {text[:limit]}{cut}")
    total_chars = sum(len(line) for line in lines)
    for prefix in CUT_ORDER:
        while total_chars > budget_chars:
            candidates = [index for index, line in enumerate(lines) if line.startswith(prefix)]
            longest = max(candidates, key=lambda index: len(lines[index]), default=None)
            if longest is None or len(lines[longest]) <= CUT_CHARS + len(CUT_MARKER):
                break
            shortened = lines[longest][:CUT_CHARS] + CUT_MARKER
            total_chars -= len(lines[longest]) - len(shortened)
            lines[longest] = shortened
    return "\n\n".join(lines)
