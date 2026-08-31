"""DisposableList: the ordered accumulator behind every fiber's effect state.

A fiber must replay its collected disposers in reverse collection order and
also support removal by token or by value, because a nested effect handle
re-parents itself out of its fiber's list when a parent effect collects it.
A plain list gives order but O(n) tokenless removal; this is the cordis
utils.ts structure: a monotonic serial number keyed into an insertion-ordered
dict, with a reverse index for removal by value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


class DisposableList:
    """Insertion-ordered values with removal tokens and reversed drain."""

    def __init__(self) -> None:
        self._serial = 0
        self._entries: dict[int, Any] = {}
        # id() keys are safe: every value is strongly referenced by _entries
        # for exactly as long as its index entry exists, so ids cannot be
        # reused while an entry is live.
        self._index: dict[int, int] = {}

    def push(self, value: Any) -> Callable[[], None]:
        """Append a value; the returned token removes exactly this entry."""
        self._serial += 1
        serial = self._serial

        self._entries[serial] = value
        self._index[id(value)] = serial

        def remove() -> None:
            self._entries.pop(serial, None)
            if self._index.get(id(value)) == serial:
                self._index.pop(id(value))

        return remove

    def delete(self, value: Any) -> bool:
        """Remove by value: the re-parenting path of a collected child handle."""
        serial = self._index.pop(id(value), None)
        if serial is None:
            return False
        return self._entries.pop(serial, None) is not None

    def clear(self) -> list[Any]:
        """Empty the list and return the values in reverse collection order."""
        values = list(self._entries.values())
        self._entries.clear()
        self._index.clear()
        values.reverse()
        return values

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Any]:
        return iter(list(self._entries.values()))
