"""Session-log readers: one per harness trajectory format.

Both surveyed harnesses persist trajectories as plain files, so a reader is
a pure function from the descriptor's trajectory directory to an ordered
tuple of event objects. A missing directory reads as an empty trajectory (a
crashed episode still yields a result); corruption raises, except for one
torn final JSONL line per file, which a mid-write crash legitimately leaves
behind (the same tolerance the commit log applies).

Subclass :class:`TrajectoryReader`, set the class attribute ``format``, implement
:meth:`TrajectoryReader.__call__`, and decorate with ``@register_trajectory_reader``::

    @register_trajectory_reader
    class MyReader(TrajectoryReader):
        format = "my-format"

        def __call__(self, path):
            ...

External readers are either registered at runtime via
:func:`register_reader` or named as a dotted ``"module:callable"`` reference
resolved by :func:`reader_for`; plain callables from a dotted reference are
wrapped by :class:`_CallableTrajectoryReader`.
"""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reef.core.errors import ReefError


class TrajectoryError(ReefError):
    """A session log exists but cannot be decoded as its declared format."""


class TrajectoryReader(ABC):
    """Base class for harness trajectory readers.

    Required (subclass must set):
        ``format`` — canonical format name matched against the descriptor's
        ``trajectory_format``.

    Implement:
        ``__call__`` — read the trajectory directory and return an ordered
        tuple of event objects.
    """

    format: str

    @abstractmethod
    def __call__(self, path: Path) -> tuple[dict[str, Any], ...]:
        """Read the trajectory directory and return ordered event objects."""


class _CallableTrajectoryReader(TrajectoryReader):
    """Adapter wrapping a plain callable as a :class:`TrajectoryReader` instance.

    Used when a dotted ``"module:callable"`` reference resolves to a function
    rather than a ``TrajectoryReader`` subclass instance.
    """

    format = ""

    def __init__(self, fn: Callable[[Path], tuple[dict[str, Any], ...]]) -> None:
        self._fn = fn

    def __call__(self, path: Path) -> tuple[dict[str, Any], ...]:
        return self._fn(path)


_readers: dict[str, TrajectoryReader] = {}
"""Bundled trajectory readers registered at import time by
:func:`register_trajectory_reader`.

Read by :func:`reader_for`; external readers are registered at runtime
through :func:`register_reader`.
"""


def register_trajectory_reader(cls: type[TrajectoryReader]) -> type[TrajectoryReader]:
    """Class decorator: instantiate and register a bundled trajectory reader."""
    instance = cls()
    fmt = instance.format
    if fmt in _readers:
        raise ValueError(f"trajectory reader {fmt!r} is already registered")
    _readers[fmt] = instance
    return cls


def register_reader(reader: TrajectoryReader) -> TrajectoryReader:
    """Register an external trajectory reader (module-level convenience)."""
    if not isinstance(reader, TrajectoryReader):
        raise TypeError(f"a trajectory reader registers a TrajectoryReader, got {type(reader).__name__}")
    fmt = reader.format
    if fmt in _readers:
        raise ValueError(f"trajectory reader {fmt!r} is already registered")
    _readers[fmt] = reader
    return reader


def reader_for(format_: str) -> TrajectoryReader:
    """Resolve a registered reader name or a dotted ``"module:callable"`` reference."""
    reader = _readers.get(format_)
    if reader is not None:
        return reader
    if ":" in format_:
        return _resolve_dotted(format_)
    available = ", ".join(sorted(_readers))
    raise TrajectoryError(f"unknown trajectory format {format_!r}; known: {available}")


def _resolve_dotted(reference: str) -> TrajectoryReader:
    """Import a dotted reader reference, wrapping plain callables."""
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        available = ", ".join(sorted(_readers))
        raise TrajectoryError(f"unknown trajectory format {reference!r}; known: {available}")
    candidate = getattr(importlib.import_module(module_name), attribute)
    if isinstance(candidate, TrajectoryReader):
        return candidate
    if callable(candidate):
        return _CallableTrajectoryReader(candidate)
    raise TrajectoryError(f"trajectory reader {reference!r} is not callable")


@register_trajectory_reader
class PiSessionReader(TrajectoryReader):
    """Read pi session JSONL trees: every ``*.jsonl`` under ``path``, in order."""

    format = "pi-session-jsonl"

    def __call__(self, path: Path) -> tuple[dict[str, Any], ...]:
        events: list[dict[str, Any]] = []
        for file in sorted(Path(path).rglob("*.jsonl")):
            lines = file.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    if index == len(lines) - 1:
                        break  # torn tail from a crash mid-write; the event never landed
                    raise TrajectoryError(f"pi session {file} has a corrupt event at line {index + 1}") from exc
                if not isinstance(event, dict):
                    raise TrajectoryError(f"pi session {file} line {index + 1} is not an event object")
                events.append(event)
        return tuple(events)


@register_trajectory_reader
class ClaudeSessionReader(TrajectoryReader):
    """Read Claude Code session JSONL trees: every ``*.jsonl`` under ``path``.

    Claude Code writes one transcript per session under
    ``CLAUDE_CONFIG_DIR/projects/<cwd-slug>/<session-id>.jsonl``, one event
    object per line. Sorting by relative path orders sessions by their
    project slug and id; the reader tolerates one torn final line per file,
    the residue a crash mid-write leaves behind.
    """

    format = "claude-session-jsonl"

    def __call__(self, path: Path) -> tuple[dict[str, Any], ...]:
        events: list[dict[str, Any]] = []
        for file in sorted(Path(path).rglob("*.jsonl")):
            lines = file.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    if index == len(lines) - 1:
                        break  # torn tail from a crash mid-write; the event never landed
                    raise TrajectoryError(f"claude session {file} has a corrupt event at line {index + 1}") from exc
                if not isinstance(event, dict):
                    raise TrajectoryError(f"claude session {file} line {index + 1} is not an event object")
                events.append(event)
        return tuple(events)


@register_trajectory_reader
class OpencodeStorageReader(TrajectoryReader):
    """Read opencode storage JSON trees: every ``*.json`` under ``path``.

    Sessions, messages, and parts are one JSON object per file under the
    storage directory; ids are lexicographically ordered, so sorting by
    relative path yields creation order within each subtree.
    """

    format = "opencode-storage-json"

    def __call__(self, path: Path) -> tuple[dict[str, Any], ...]:
        events: list[dict[str, Any]] = []
        for file in sorted(Path(path).rglob("*.json")):
            try:
                event = json.loads(file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise TrajectoryError(f"opencode storage file {file} is not valid JSON") from exc
            if not isinstance(event, dict):
                raise TrajectoryError(f"opencode storage file {file} is not an event object")
            events.append(event)
        return tuple(events)


# Module-level instances for backward-compatible call syntax.
read_pi_session = PiSessionReader()
read_claude_session = ClaudeSessionReader()
read_opencode_storage = OpencodeStorageReader()
