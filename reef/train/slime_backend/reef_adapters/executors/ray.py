"""Slime's GPU placement and rendezvous behind Reef's Executor contract."""

from __future__ import annotations

from contextlib import suppress

from reef.runtime.executor.delegating import DelegatingExecutor
from reef.runtime.executor.ray import RayExecutor


class SlimeRayExecutor(DelegatingExecutor):
    """Reuse Slime's launcher and route worker operations through RayExecutor.

    Placement groups are borrowed: actor, critic and rollout can share one
    reservation. Shutdown releases only the training workers. Worker-level
    tensor transport and Megatron process groups stay owned by Slime.
    """

    def _init_executor(self) -> None:
        from slime.ray.actor_group import RayTrainGroup

        self._launcher = RayTrainGroup(**dict(self.config.options))
        try:
            self._launcher._allocate_gpus_for_actor(self._launcher._pg, self._launcher._num_gpus_per_actor)
        except BaseException:
            with suppress(Exception):
                self._launcher.release()
            raise
        self._rpc = RayExecutor.from_workers(self._launcher._actor_handlers)
        self._closed = False

    def shutdown(self) -> None:
        if self._closed:
            return
        self._rpc.shutdown()
        # Preserve Slime's release ordering and wait before GPU reuse.
        self._launcher.release()
        self._closed = True
