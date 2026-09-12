"""Slime resource and inference components instantiated by Reef's model driver.

Slime launch helpers stay private to the integration. Resource allocation,
inference startup and training startup are separate lifecycle operations.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import ray

from reef.runtime.deployment import DeploymentHealth, DeploymentResources, InferenceConnection
from reef.runtime.executor import ExecutorConfig, WorkerSpec
from reef.runtime.executor.process_guard import DEPLOYMENT_ENV, retire
from reef.runtime.executor.ray import RayExecutor

INFERENCE_PROTOCOL = "slime-sglang-control-v2"


class SlimeDeploymentHealth:
    """Poll both components without blocking behind long model operations."""

    def __init__(self, *components: DeploymentHealth) -> None:
        self.components = components

    def poll(self) -> None:
        for component in self.components:
            component.poll()


class RayHealthProbe:
    """Keep one outstanding RPC; a busy actor is not presumed dead."""

    def __init__(self) -> None:
        self.pending: Any = None

    def poll(self, actor: Any, method: str) -> None:
        if self.pending is None:
            self.pending = getattr(actor, method).remote()
        ready, _ = ray.wait([self.pending], timeout=0)
        if not ready:
            return
        pending, self.pending = self.pending, None
        result = ray.get(pending)
        if isinstance(result, dict) and result.get("ok") is False and result.get("recoverable") is not True:
            raise RuntimeError(f"model component failed its health check: {result!r}")


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

        nodes = set(self._nodes)
        nodes.update(node["NodeID"] for node in ray.nodes() if node["Alive"])
        cleanup = ray.remote(num_cpus=0, max_retries=0)(retire)
        pending = [
            cleanup.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
                runtime_env={"env_vars": {DEPLOYMENT_ENV: ""}},
            ).remote(self._process_lease)
            for node in nodes
        ]
        # A lost node cannot confirm process retirement. Do not reuse its GPU
        # reservations automatically or mutate an externally owned cluster.
        ray.get(pending, timeout=45)


class SlimeInferenceService:
    """Own engines and the control actor; borrow the deployment allocation."""

    connection_protocol = INFERENCE_PROTOCOL

    def __init__(self, args: Any) -> None:
        self.args = args
        self._inference: RayExecutor | None = None
        self._started = False
        self._closed = False
        self._probe = RayHealthProbe()

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

    def poll(self) -> None:
        if self._inference is None or self._closed:
            raise RuntimeError("inference service is not running")
        self._probe.poll(self._inference.workers[0], "check_health")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._inference is not None:
            try:
                self._inference.rpc(0, "shutdown", timeout=90)
            except Exception:
                # The resource owner confirms retirement of native children.
                logging.getLogger(__name__).exception("Inference shutdown failed; retiring its owned process groups")
            finally:
                self._inference.shutdown()
