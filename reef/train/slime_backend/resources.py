"""Reef-owned Ray session and training GPU reservation for the Slime training component.

Reef reserves the training GPUs itself from the parsed Slime topology; the
actor and critic groups borrow their bundles from that one reservation. The
serving engines reserve their own GPUs on the inference component.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import ray

from reef.runtime.deployment import DeploymentResources
from reef.runtime.executor.placement import GpuBundles, ModelGpuLayout, ModelGpuReservation, reserve_model_gpus
from reef.runtime.executor.process_guard import DEPLOYMENT_ENV, retire
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments


def model_gpu_layout(args: SlimeArguments) -> ModelGpuLayout:
    """The training GPUs from Slime's parsed topology flags: actor nodes times GPUs per node."""
    return ModelGpuLayout(
        training_gpus=int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node), inference_gpus=0
    )


class SlimeDeploymentResources(DeploymentResources):
    """Own one Ray client session and the training GPU reservation."""

    def __init__(
        self,
        args: SlimeArguments,
        *,
        ray_address: str,
        namespace: str,
        runtime_env: dict[str, Any] | None = None,
    ) -> None:
        self.args = args
        self.ray_address = ray_address
        self.namespace = namespace
        self.runtime_env = runtime_env
        #: Slime's view of the reservation: the actor and critic groups share the training bundles.
        self.placement_groups: dict[str, GpuBundles | None] = {}
        self._reservation: ModelGpuReservation | None = None
        self._started = False
        self._closed = False
        self._owns_session = False
        self._process_lease = uuid4().hex
        self._nodes: list[str] = []

    def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("deployment resources can only be started once")
        if ray.is_initialized():
            raise RuntimeError("the Slime training component requires its own Ray client session")
        self._started = True
        self._owns_session = True
        runtime_env = dict(self.runtime_env or {})
        runtime_env["env_vars"] = {**runtime_env.get("env_vars", {}), DEPLOYMENT_ENV: self._process_lease}
        runtime_env["worker_process_setup_hook"] = "reef.runtime.executor.process_guard.install"
        ray.init(address=self.ray_address, namespace=self.namespace, runtime_env=runtime_env)
        self._nodes = [node["NodeID"] for node in ray.nodes() if node["Alive"]]
        self._reservation = reserve_model_gpus(model_gpu_layout(self.args))
        self.placement_groups = {
            "actor": self._reservation.training,
            "critic": self._reservation.training if self.args.use_critic else None,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        try:
            if self._reservation is not None:
                try:
                    self._reservation.release()
                except Exception as exc:
                    errors.append(exc)
        finally:
            self.placement_groups = {}
            if self._owns_session:
                # Disconnect this job; never stop an externally owned cluster.
                # Job-scoped reservations also expire if a launch helper failed
                # before returning its placement handles.
                try:
                    if self._nodes:
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
