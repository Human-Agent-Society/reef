"""The Ray session and GPU reservation of one Slime job's driver.

Slime places the job itself (``create_placement_groups``): one placement group
of the GPUs its arguments name, which the actor, the critic and Slime's
rollout engines share as Slime splits them (with ``--colocate`` the engines
take training GPUs). A driver without Slime's serving session reserves the
GPUs itself, for its own lifetime; with the session, the session holds them
across jobs (``reef.train.slime_backend.rollout_group``) and the driver trains
on the session's placements. Reef's serving engines reserve their own GPUs on
the inference component.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import ray

from reef.runtime.deployment import DeploymentResources
from reef.runtime.executor.placement import GpuBundles
from reef.runtime.executor.process_guard import DEPLOYMENT_ENV, retire
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments


class SlimeDeploymentResources(DeploymentResources):
    """Own one Ray client session and, unless ``place`` is False, the job's placement group, which Slime makes and splits.

    With ``place`` False the driver trains on the placements of Slime's serving session (``adopt``), which this
    object neither makes nor removes.
    """

    def __init__(
        self,
        args: SlimeArguments,
        *,
        ray_address: str,
        namespace: str,
        runtime_env: dict[str, Any] | None = None,
        place: bool = True,
    ) -> None:
        self.args = args
        self.place = place
        self.ray_address = ray_address
        self.namespace = namespace
        self.runtime_env = runtime_env
        #: Slime's split of the placement group: ``actor``, ``critic`` (None without a critic) and ``rollout``
        #: (no bundles without rollout engines).
        self.placement_groups: dict[str, GpuBundles | None] = {}
        self._group: Any = None
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
        if not self.place:
            return
        from slime.ray.placement_group import create_placement_groups

        groups = create_placement_groups(self.args)
        self._group = groups["actor"][0]
        self.placement_groups = {
            role: None if bundles is None else GpuBundles(*bundles) for role, bundles in groups.items()
        }

    def adopt(self, placement_groups: dict[str, GpuBundles | None]) -> None:
        """Train on ``placement_groups``, the placements of Slime's serving session, which the session removes."""
        if self.place or not self._started or self._closed:
            raise RuntimeError("only started resources that place nothing adopt the serving session's placements")
        self.placement_groups = dict(placement_groups)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        try:
            if self._group is not None:
                try:
                    ray.util.remove_placement_group(self._group)
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
        retire_lease(self._process_lease, self._nodes)


def retire_lease(lease: str, nodes: list[str]) -> None:
    """Retire the processes of ``lease`` on ``nodes`` and on every live node of the connected cluster."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    targets = set(nodes)
    targets.update(node["NodeID"] for node in ray.nodes() if node["Alive"])
    cleanup = ray.remote(num_cpus=0, max_retries=0)(retire)
    pending = [
        cleanup.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node, soft=False),
            runtime_env={"env_vars": {DEPLOYMENT_ENV: ""}},
        ).remote(lease)
        for node in targets
    ]
    # A lost node cannot confirm process retirement. Do not reuse its GPU
    # reservations automatically or mutate an externally owned cluster.
    ray.get(pending, timeout=45)
