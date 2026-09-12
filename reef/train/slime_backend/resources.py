"""Slime resource and inference components instantiated by Reef's model driver.

Slime launch helpers stay private to the integration. Resource allocation,
inference startup and training startup are separate lifecycle operations.
"""

from __future__ import annotations

from typing import Any

import ray

from reef.runtime.deployment import DeploymentResources, InferenceConnection
from reef.runtime.executor import ExecutorConfig, WorkerSpec
from reef.runtime.executor.ray import RayExecutor

INFERENCE_PROTOCOL = "slime-sglang-control-v1"


class SlimeDeploymentResources:
    """Own one Ray client session and the coordinated model reservations."""

    def __init__(
        self,
        args: Any,
        *,
        ray_address: str,
        namespace: str,
        runtime_env: dict[str, Any] | None = None,
        allocate_models: bool = True,
    ) -> None:
        self.args = args
        self.ray_address = ray_address
        self.namespace = namespace
        self.runtime_env = runtime_env
        self.allocate_models = allocate_models
        self.placement_groups: dict[str, Any] = {}
        self._started = False
        self._closed = False
        self._owns_session = False

    def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("deployment resources can only be started once")
        if ray.is_initialized():
            raise RuntimeError("the Slime model driver requires its own Ray client session")
        self._started = True
        self._owns_session = True
        ray.init(address=self.ray_address, namespace=self.namespace, runtime_env=self.runtime_env)
        if self.allocate_models:
            from slime.ray.placement_group import create_placement_groups

            self.placement_groups = create_placement_groups(self.args)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        from ray.util.placement_group import remove_placement_group

        errors = []
        released = set()
        try:
            for placement in self.placement_groups.values():
                if placement is None or placement[0] is None or placement[0].id in released:
                    continue
                released.add(placement[0].id)
                try:
                    remove_placement_group(placement[0])
                except Exception as exc:
                    errors.append(exc)
        finally:
            self.placement_groups.clear()
            if self._owns_session:
                # Disconnect this job; never stop an externally owned cluster.
                # Job-scoped reservations also expire if a launch helper failed
                # before returning its placement handles.
                ray.shutdown()
        if errors:
            raise errors[0]


class SlimeInferenceService:
    """Own engines and the control actor; borrow the deployment allocation."""

    connection_protocol = INFERENCE_PROTOCOL

    def __init__(self, args: Any) -> None:
        self.args = args
        self._inference: RayExecutor | None = None
        self._started = False
        self._closed = False

    def start(self, resources: DeploymentResources) -> InferenceConnection:
        if self._started or self._closed:
            raise RuntimeError("inference service can only be started once")
        if not isinstance(resources, SlimeDeploymentResources) or "rollout" not in resources.placement_groups:
            raise ValueError("Slime inference requires its supplied model reservations")
        self._started = True
        from slime.ray.utils import add_default_ray_env_vars
        from reef.train.slime_backend.reef_adapters.worker_hooks import reef_rollout_env_vars

        self._inference = RayExecutor(
            ExecutorConfig(
                backend=RayExecutor,
                workers=(
                    WorkerSpec(
                        worker_cls="reef.train.slime_backend.reef_adapters.inference:SlimeInferenceWorker",
                        args=(self.args, resources.placement_groups["rollout"]),
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
        return InferenceConnection(self.connection_protocol, RayExecutor.from_workers(self._inference.workers))

    def check_health(self) -> None:
        if self._inference is None or self._closed:
            raise RuntimeError("inference service is not running")
        self._inference.rpc(0, "check_health", timeout=30)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._inference is not None:
            try:
                self._inference.rpc(0, "shutdown", timeout=90)
            finally:
                self._inference.shutdown()
