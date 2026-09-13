"""Track acknowledged inference memory transitions independently of a backend."""

from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import Protocol


class InferenceMemoryOperations(Protocol):
    """Synchronous engine operations; return only after every region is changed."""

    def release(self, regions: Sequence[str]) -> None: ...

    def resume(self, regions: Sequence[str]) -> None: ...


class InferenceMemory:
    """Pair releases and resumes, including cold startup and partial failures.

    Repeating an acknowledged operation is safe. A failed operation can have
    changed only part of an engine, so its state remains uncertain until the
    owning deployment replaces that engine.
    """

    def __init__(self, operations: InferenceMemoryOperations, regions: Sequence[str]) -> None:
        self._operations = operations
        self._regions = tuple(regions)
        self._released: set[str] = set()
        self._uncertain = False
        self._lock = RLock()

    def _selection(self, regions: Sequence[str] | None) -> tuple[str, ...]:
        if self._uncertain:
            raise RuntimeError("inference memory state is uncertain; replace the engine before reuse")
        selected = self._regions if regions is None else tuple(dict.fromkeys(regions))
        if unknown := set(selected).difference(self._regions):
            raise ValueError(f"unknown inference memory regions: {sorted(unknown)}")
        return selected

    def release(self, regions: Sequence[str] | None = None) -> None:
        with self._lock:
            selected = tuple(region for region in self._selection(regions) if region not in self._released)
            if not selected:
                return
            try:
                self._operations.release(selected)
            except BaseException:
                self._uncertain = True
                raise
            self._released.update(selected)

    def resume(self, regions: Sequence[str] | None = None) -> None:
        with self._lock:
            selected = tuple(region for region in self._selection(regions) if region in self._released)
            if not selected:
                return
            try:
                self._operations.resume(selected)
            except BaseException:
                self._uncertain = True
                raise
            self._released.difference_update(selected)
