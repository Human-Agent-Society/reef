"""Inference pause and recovery ordering, independent of engine and transport types."""

from __future__ import annotations

from typing import Protocol


class InferenceEngines(Protocol):
    """Concrete operations on the inference engines attached to a trainer."""

    @property
    def owned(self) -> bool:
        """Whether engine replacement is controlled by this deployment."""

    def pause(self) -> object: ...

    def resume(self) -> object: ...

    def recover(self) -> None:
        """Replace dead engines; preserve healthy engines and initial attachment."""

    def terminate(self) -> int:
        """Retire owned engines after an uncertain update; never kill borrowed engines."""


class InferenceMonitor(Protocol):
    """Background engine monitoring must respect publication/recovery barriers."""

    def pause(self) -> None:
        """Drain active checks and retirement before engine mutation; raise on timeout."""

    def resume(self) -> None: ...


class WeightUpdateConnection(Protocol):
    """Connection fencing for direct worker-to-engine weight transport."""

    def is_usable(self) -> bool:
        """True only when the update lock is known to be idle and unpoisoned."""

    def replace(self) -> None:
        """Replace the uncertain update lock; existing worker connections become stale."""


class InferenceControl:
    """Coordinate pause, recovery and reconnect without routing weight tensors.

    Calls must be serialized by the owning actor. A successful recovery of a
    paused publication leaves both generation and monitoring paused. Only the
    training publication coordinator may resume after the durable commit gate.
    """

    def __init__(
        self,
        engines: InferenceEngines,
        connection: WeightUpdateConnection,
        monitor: InferenceMonitor,
    ) -> None:
        self._engines = engines
        self._connection = connection
        self._monitor = monitor
        self.paused = False
        self.reconnect_required = False

    def pause(self) -> object:
        # Preserve pause intent even if only some engines cross the barrier.
        self.paused = True
        self._monitor.pause()
        return self._engines.pause()

    def resume(self) -> object:
        result = self._engines.resume()
        self.paused = False
        self._monitor.resume()
        return result

    def terminate(self) -> int:
        self.paused = True
        self._monitor.pause()
        return self._engines.terminate() if self._engines.owned else 0

    def recover(self) -> None:
        try:
            self._monitor.pause()
            try:
                usable = self._connection.is_usable()
            except Exception:
                usable = False
            if not usable:
                if not self._engines.owned:
                    raise RuntimeError("uncertain external weight update requires restarting the external deployment")
                self._connection.replace()
                self.reconnect_required = True
            self._engines.recover()
            if self.paused:
                self._engines.pause()
            else:
                self._monitor.resume()
        except BaseException:
            # A failed recovery cannot authorize a later legacy monitor-resume
            # call to replace engines behind an unfinished publication.
            self.paused = True
            raise

    def prepare_training_connection(self) -> None:
        """Fence a new trainer attachment even when all engines are healthy.

        The deployment owner must retire the previous training workers first.
        This is a serialized attachment handshake, not leader election. Engine
        and lock recovery keep their existing ownership rules.
        """
        self.paused = True
        self.reconnect_required = True
        self.recover()

    def acknowledge_reconnect(self) -> None:
        """Called only after training workers have attached to the current targets."""
        self.reconnect_required = False
