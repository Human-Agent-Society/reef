"""Benchmarks and collections of task items (RFC #1312).

A task item names its benchmarks in its metadata, under ``benchmarks``. A benchmark is not stored: it is every task
item of an owner whose ``benchmarks`` has the name. A collection is a stored value of one owner that names benchmarks
and other collections of that owner. This module defines the collection value and the rules that the names follow.
``DataStore`` keeps the collections.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.errors import ReefError
from reef.core.evaluation import COLLECTION_AVERAGES, CollectionAverage


class CollectionNotFoundError(ReefError):
    """A collection name that the owner does not store."""


def check_names(value: object, name: str) -> tuple[str, ...]:
    """Return ``value``, a list or tuple of distinct non-empty strings, as a tuple; ``ValueError`` for another value.
    ``name`` names the value in the error."""
    if (
        not isinstance(value, list | tuple)
        or any(not isinstance(entry, str) or not entry for entry in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a list of distinct non-empty strings")
    return tuple(value)


@dataclass(frozen=True)
class TaskCollection:
    """A named set of benchmarks of one owner.

    Values: ``name`` is a non-empty string without ``/``, other than ``.`` and ``..``. ``benchmarks`` and
    ``collections`` are tuples of distinct non-empty strings. Each of them can be empty: a collection with neither
    has no benchmark, and a client can add them later. ``collections`` names other collections of the same owner,
    and does not have ``name``. ``average`` is ``task`` or ``benchmark``: how an evaluation on the collection
    computes its mean score (``EvaluationCollection``).
    Purpose: a client stores it (``DataStore.put_collection``); the collection routes and an evaluation on the
    collection read its benchmarks (``collection_benchmarks``).
    """

    name: str
    benchmarks: tuple[str, ...] = ()
    collections: tuple[str, ...] = ()
    average: CollectionAverage = "task"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "/" in self.name or self.name in (".", ".."):
            raise ValueError(f"a collection name must be a non-empty name without '/', got {self.name!r}")
        object.__setattr__(self, "benchmarks", check_names(self.benchmarks, "a collection's benchmarks"))
        object.__setattr__(self, "collections", check_names(self.collections, "a collection's collections"))
        if self.name in self.collections:
            raise ValueError(f"the collection {self.name!r} cannot name itself")
        if self.average not in COLLECTION_AVERAGES:
            raise ValueError(f"a collection's average is one of {', '.join(COLLECTION_AVERAGES)}")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "benchmarks": list(self.benchmarks),
            "collections": list(self.collections),
            "average": self.average,
        }

    @classmethod
    def from_json(cls, value: object) -> TaskCollection:
        """The collection of a JSON object with ``name``, and with ``benchmarks``, ``collections`` and ``average`` or
        their defaults; ``ValueError`` for another value."""
        known = {"name", "benchmarks", "collections", "average"}
        if not isinstance(value, Mapping) or "name" not in value or set(value) - known:
            raise ValueError(f"a collection is an object of {sorted(known)}")
        return cls(
            value["name"], value.get("benchmarks", ()), value.get("collections", ()), value.get("average", "task")
        )


def collection_benchmarks(stored: Mapping[str, TaskCollection], name: str) -> tuple[str, ...]:
    """The distinct benchmarks of the collection ``name`` of ``stored``, which maps each collection of one owner by
    its name: its own benchmarks first, then those of each of its collections, in order.

    Raise ``ValueError`` when ``name`` or a collection under it is not in ``stored``, or when a collection under
    ``name`` names a collection that it is under: a cycle.
    """
    found: dict[str, None] = {}
    done: set[str] = set()

    def _add(current: str, above: tuple[str, ...]) -> None:
        if current in above:
            raise ValueError(f"the collections {' -> '.join((*above, current))} make a cycle")
        if current not in stored:
            raise ValueError(f"the owner has no collection {current!r}")
        if current in done:
            return
        done.add(current)
        found.update(dict.fromkeys(stored[current].benchmarks))
        for inner in stored[current].collections:
            _add(inner, (*above, current))

    _add(name, ())
    return tuple(found)


__all__ = ["CollectionNotFoundError", "TaskCollection", "check_names", "collection_benchmarks"]
