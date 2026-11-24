"""Inference failure recovery.

An engine integration implements :class:`InferenceEngineGroup`.
:class:`InferenceControl` drives it to pause and recover the engines around
weight loads, without importing the integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class InferenceEngineGroup(ABC):
    """Concrete operations on the group of serving engines of one inference service."""

    @abstractmethod
    def pause(self) -> object: ...

    @abstractmethod
    def resume(self) -> object: ...

    @abstractmethod
    def recover(self) -> int:
        """Replace dead engines and keep the live ones; return the number of engine slots it refilled.

        The result is 0 when no engine was dead. A replacement engine serves
        the base model under the engine's own default runtime load ID.
        """

    @abstractmethod
    def terminate(self) -> int:
        """Retire the engines after an uncertain update; return the number of engines it retired."""


class InferenceControl:
    """Coordinate pause and recovery of the engines.

    An engine integration's worker makes one per engine group and calls it
    from its control actor, one call at a time. ``paused`` is True from a
    ``pause``, a ``terminate`` or a failed ``recover`` until the next
    successful ``resume``, and starts False. The operations:

    - ``pause`` sets ``paused`` and returns the engine group's answer to its
      pause, unchecked.
    - ``resume`` returns the engine group's answer to its resume and clears
      ``paused``.
    - ``terminate`` sets ``paused``, retires the engines, and returns how many
      it retired.
    - ``recover`` replaces dead engines and returns how many slots it
      refilled. When ``paused`` is True, it pauses the engines again, so the
      replacements stay paused.
    """

    def __init__(self, engines: InferenceEngineGroup) -> None:
        self._engines = engines
        self.paused = False

    def pause(self) -> object:
        # Preserve pause intent even if only some engines cross the barrier.
        self.paused = True
        return self._engines.pause()

    def resume(self) -> object:
        result = self._engines.resume()
        self.paused = False
        return result

    def terminate(self) -> int:
        self.paused = True
        return self._engines.terminate()

    def recover(self) -> int:
        """Replace dead engines (``InferenceEngineGroup.recover``) and return the number of slots it refilled."""
        try:
            replaced = self._engines.recover()
            if self.paused:
                self._engines.pause()
            return replaced
        except BaseException:
            # A failed recovery leaves the engines in an unknown state behind an unfinished weight load.
            self.paused = True
            raise
