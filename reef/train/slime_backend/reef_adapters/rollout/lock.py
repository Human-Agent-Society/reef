"""Failure-aware coordination actor for Slime weight transports."""

from __future__ import annotations

import ray

from slime.ray.ray_actor import RayActor


@ray.remote
class ReefRolloutLock(RayActor):
    """Serialize fan-out and permanently fence an uncertain transport."""

    def __init__(self) -> None:
        self._locked = False
        self._poisoned = False
        self._completed_phases: dict[str, dict[str, str | None]] = {}

    def acquire(self) -> bool:
        if self._poisoned:
            raise RuntimeError("rollout engine lock is poisoned; reconnect weight-update transports")
        if self._locked:
            return False
        self._locked = True
        return True

    def release(self) -> None:
        if self._poisoned:
            raise RuntimeError("poisoned rollout engine lock cannot be released")
        if not self._locked:
            raise RuntimeError("rollout engine lock is not acquired")
        self._locked = False

    def poison(self) -> None:
        self._locked = True
        self._poisoned = True

    def status(self) -> dict[str, bool]:
        return {"locked": self._locked, "poisoned": self._poisoned}

    def complete_phase(self, phase_id: str, error: str | None = None) -> None:
        if not isinstance(phase_id, str) or not phase_id:
            raise ValueError("phase_id must be a non-empty string")
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError("phase error must be a non-empty string or None")
        self._completed_phases[phase_id] = {"error": error}

    def phase_status(self, phase_id: str) -> dict[str, str | None] | None:
        return self._completed_phases.get(phase_id)

    def clear_phases(self) -> None:
        if self._poisoned or self._locked:
            raise RuntimeError("cannot clear phases while the rollout transport is active")
        self._completed_phases.clear()


__all__ = ["ReefRolloutLock"]
