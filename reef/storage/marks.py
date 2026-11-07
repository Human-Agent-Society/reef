"""Marks: named positions that a service keeps in the shared store, such as how far it has copied a record store."""

from __future__ import annotations

from abc import ABC, abstractmethod


def check_mark_name(name: object) -> str:
    """``name`` as a mark name; raise ``ValueError`` unless it is a non-empty string."""
    if not isinstance(name, str) or not name:
        raise ValueError("a mark name is a non-empty string")
    return name


class Marks(ABC):
    """Named positions in the shared store.

    Values: a mapping from non-empty names to non-negative ints; a name that was never advanced reads 0.
    ``advance(name, value)`` sets the mark to the larger of its value and ``value``, so a mark never decreases and
    an advance that runs twice changes nothing.

    Purpose: a service keeps a durable position in the coordination store. The record copy of a version before
    RFC #1219 kept the mark ``record_copy:<scenario>``; nothing in Reef writes a mark now.
    """

    @abstractmethod
    def read(self, name: str) -> int:
        """The value of the mark ``name``; 0 when it was never advanced."""

    @abstractmethod
    def advance(self, name: str, value: int) -> None:
        """Set the mark ``name`` to the larger of its value and ``value``; ``ValueError`` for a negative value."""


__all__ = ["Marks", "check_mark_name"]
