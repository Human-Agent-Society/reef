"""Default SGLang rollout executor, with all Slime/Ray serving ownership here."""

from __future__ import annotations

from reef.runtime.executor.delegating import DelegatingExecutor
from reef.runtime.executor.local import LocalExecutor
from reef.train.slime_backend.reef_adapters.executors.config import (
    DEFAULT_ROLLOUT_EXECUTOR as DEFAULT_ROLLOUT_EXECUTOR,
)
from reef.train.slime_backend.reef_adapters.executors.config import slime_executor_class


def rollout_executor_class(args):
    return slime_executor_class(getattr(args, "reef_rollout_executor_backend", "auto"), role="rollout")


class SlimeRayRolloutExecutor(DelegatingExecutor):
    """One control rank manages Slime's multi-node SGLang engine groups.

    The control object lives in the rollout manager; engine processes are Ray
    actors allocated by Slime. Alternative executors implement the same control
    RPC vocabulary without inheriting Slime's server or actor classes.
    """

    def _init_executor(self) -> None:
        from reef.train.slime_backend.reef_adapters.executors.rollout_worker import SlimeRayRolloutWorker

        self._worker = SlimeRayRolloutWorker(**dict(self.config.options))
        self._rpc = LocalExecutor.from_workers([self._worker], owned=True)

    def check_health(self, timeout: float | None = None) -> None:
        self._rpc.rpc(0, "check_health", timeout=timeout)
