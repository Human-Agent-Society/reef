"""Deployment-owned inference and coordinated Slime model reservations.

The Reef process entrypoint owns this object. Training receives borrowed
inference controls and placement groups; disposing a training manager never
releases these resources. Slime's placement/launch helpers remain an internal
implementation detail during the first non-colocated migration.
"""

from __future__ import annotations

import logging
from typing import Any

from reef.runtime.executor import ExecutorConfig, WorkerSpec
from reef.runtime.executor.ray import RayExecutor

_logger = logging.getLogger(__name__)


class SlimeInferenceResources:
    """Own inference controls and the shared allocation until training exits."""

    def __init__(self) -> None:
        self.placement_groups: dict[str, Any] = {}
        self._inference: RayExecutor | None = None
        self._started = False
        self._closed = False

    def start(self, args: Any) -> None:
        if self._started or self._closed:
            raise RuntimeError("inference resources can only be started once")
        if (
            getattr(args, "colocate", False)
            or getattr(args, "rollout_external", False)
            or int(getattr(args, "megatron_lora_rank", 0) or 0) > 0
        ):
            raise ValueError(
                "separate inference ownership currently requires managed non-colocated full-weight engines"
            )
        self._started = True
        from slime.ray.placement_group import create_placement_groups
        from slime.ray.utils import add_default_ray_env_vars

        from reef.train.slime_backend.reef_adapters.worker_hooks import reef_rollout_env_vars

        try:
            self.placement_groups = create_placement_groups(args)
            self._inference = RayExecutor(
                ExecutorConfig(
                    backend=RayExecutor,
                    workers=(
                        WorkerSpec(
                            worker_cls="reef.train.slime_backend.reef_adapters.inference:SlimeInferenceWorker",
                            args=(args, self.placement_groups["rollout"]),
                        ),
                    ),
                    options={
                        "num_cpus": 1,
                        "num_gpus": 0,
                        "runtime_env": {"env_vars": add_default_ray_env_vars(reef_rollout_env_vars())},
                    },
                    launch_timeout_s=14_400,
                )
            )
            self._inference.rpc(0, "check_health", timeout=30)
        except BaseException:
            try:
                self.close()
            except Exception:
                _logger.exception("Failed to clean up inference resources after startup failure")
            raise

    @property
    def serving(self) -> RayExecutor:
        """A serializable borrowed connection; it cannot destroy the owner actor."""
        if self._inference is None or self._closed:
            raise RuntimeError("inference resources are not running")
        return RayExecutor.from_workers(self._inference.workers)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        if self._inference is not None:
            try:
                self._inference.rpc(0, "shutdown", timeout=90)
            except Exception as exc:
                errors.append(exc)
            try:
                self._inference.shutdown()
            except Exception as exc:
                errors.append(exc)
        from ray.util.placement_group import remove_placement_group

        released = set()
        for placement in self.placement_groups.values():
            if placement is None or placement[0] is None or placement[0].id in released:
                continue
            released.add(placement[0].id)
            try:
                remove_placement_group(placement[0])
            except Exception as exc:
                errors.append(exc)
        self.placement_groups.clear()
        if errors:
            raise errors[0]
