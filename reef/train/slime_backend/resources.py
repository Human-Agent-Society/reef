"""Reef-owned Ray session and model GPU reservations for a Slime deployment.

Reef reserves the model GPUs itself from the parsed Slime topology; the
separate components borrow their bundles from that one reservation.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import ray

from reef.runtime.deployment import InferenceResources
from reef.runtime.executor.placement import GpuBundles, ModelGpuLayout, ModelGpuReservation, reserve_model_gpus
from reef.runtime.executor.process_guard import DEPLOYMENT_ENV, retire


def model_gpu_layout(args: Any) -> ModelGpuLayout:
    """The deployment's GPU layout from Slime's parsed topology flags."""
    return ModelGpuLayout(
        training_gpus=int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node),
        inference_gpus=int(getattr(args, "rollout_num_gpus", 0) or 0),
        colocate=bool(getattr(args, "colocate", False)),
        external_inference=bool(getattr(args, "rollout_external", False)),
    )


class SlimeDeploymentResources(InferenceResources):
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
        #: Slime's view of the reservation: the actor and critic groups share
        #: the training bundles, rollout engines take the inference bundles.
        self.placement_groups: dict[str, GpuBundles | None] = {}
        self._reservation: ModelGpuReservation | None = None
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
            self._reservation = reserve_model_gpus(model_gpu_layout(self.args))
            self.placement_groups = {
                "actor": self._reservation.training,
                "critic": self._reservation.training if getattr(self.args, "use_critic", False) else None,
                "rollout": self._reservation.inference,
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
