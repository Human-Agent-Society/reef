"""Slime placement integration instantiated by Reef's model driver.

The native Slime allocator reserves inference and training GPUs together.
Reef owns the allocation; the separate components borrow their reservations.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import ray

from reef.runtime.deployment import DeploymentHealth
from reef.runtime.executor.process_guard import DEPLOYMENT_ENV, retire


class SlimeDeploymentHealth:
    """Poll both components without blocking behind long model operations."""

    def __init__(self, *components: DeploymentHealth) -> None:
        self.components = components

    def poll(self) -> None:
        for component in self.components:
            component.poll()


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
        self._process_lease = uuid4().hex if allocate_models else None
        self._nodes: list[str] = []

    @property
    def inference_placement(self) -> Any:
        return self.placement_groups.get("rollout")

    def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("deployment resources can only be started once")
        if ray.is_initialized():
            raise RuntimeError("the Slime model driver requires its own Ray client session")
        self._started = True
        self._owns_session = True
        runtime_env = dict(self.runtime_env or {})
        if self._process_lease is not None:
            runtime_env["env_vars"] = {**runtime_env.get("env_vars", {}), DEPLOYMENT_ENV: self._process_lease}
            runtime_env["worker_process_setup_hook"] = "reef.runtime.executor.process_guard.install"
        ray.init(address=self.ray_address, namespace=self.namespace, runtime_env=runtime_env or None)
        if self._process_lease is not None:
            self._nodes = [node["NodeID"] for node in ray.nodes() if node["Alive"]]
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
                try:
                    if self._process_lease is not None and self._nodes:
                        self._retire_processes()
                except Exception as exc:
                    errors.append(exc)
                finally:
                    ray.shutdown()
        if errors:
            raise errors[0]

    def _retire_processes(self) -> None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        lease = self._process_lease
        if lease is None:
            return
        nodes = set(self._nodes)
        nodes.update(node["NodeID"] for node in ray.nodes() if node["Alive"])
        cleanup = ray.remote(num_cpus=0, max_retries=0)(retire)
        pending = [
            cleanup.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
                runtime_env={"env_vars": {DEPLOYMENT_ENV: ""}},
            ).remote(lease)
            for node in nodes
        ]
        # A lost node cannot confirm process retirement. Do not reuse its GPU
        # reservations automatically or mutate an externally owned cluster.
        ray.get(pending, timeout=45)
