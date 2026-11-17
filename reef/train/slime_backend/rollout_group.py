"""Slime's own rollout group: the SGLang engines of Slime's serving session, for the component's rollout jobs.

The engines belong to the training component, not to Reef's serving: they answer only the rollouts of a dataset job
or a task job (``reef.train.slime_backend.harbor_rollouts``), at the weights that the job trains. Slime starts, syncs
and offloads them as in its own training loop. With ``--colocate`` they share the training GPUs, and Slime moves
the GPU memory between the engines and the training workers; otherwise they run on GPUs of their own.

Slime's ``ServingCluster`` owns the engines, their router and the GPUs of the job's placement. It is a detached
Ray actor, named after the save root, so it outlives the job's driver: the next job's driver attaches to it and
finds the engines started. The component disposes it when it closes (``dispose_serving``). The session's actors
run under a process lease of their own (``SERVING_FILE``), so a driver's process cleanup leaves them running.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import ray

from reef.execution.executor.placement import GpuBundles
from reef.execution.executor.process_guard import DEPLOYMENT_ENV
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.train_groups import SlimeTrainGroup

#: The variable that loads Reef's SGLang plugin into an engine (``reef.inference.sglang.plugin``). The image's
#: SGLang takes a weight update only inside a session that Slime does not open; the plugin opens it.
SGLANG_PLUGIN_ENV = "SGLANG_REEF_PLUGIN"


#: The component's Slime serving session, in its save root: ``session``, the name of Slime's serving actors, and
#: ``lease``, the process lease (``reef.execution.executor.process_guard``) that they and their engines run under.
SERVING_FILE = "reef-serving.json"


@contextmanager
def serving_lease(lease: str) -> Iterator[None]:
    """Give the Ray actors that Slime creates meanwhile ``lease``, instead of the driver's process lease.

    Slime creates its serving owner and rollout manager with its default Ray environment, and the engines and
    routers they start inherit theirs; the training workers, which Slime creates later with the same defaults,
    keep the driver's lease, which the driver retires when it exits.
    """
    from slime.ray import utils

    utils.RAY_DEFAULT_ENV_VARS[DEPLOYMENT_ENV] = lease
    try:
        yield
    finally:
        utils.RAY_DEFAULT_ENV_VARS.pop(DEPLOYMENT_ENV)


def serving_session(args: SlimeArguments) -> str:
    """The name of the Slime serving session that a driver with ``args`` creates or attaches to."""
    from slime.ray.training_recovery import training_session_name

    return training_session_name(args)


def serving_exists(session: str) -> bool:
    """Whether Slime's serving owner of ``session`` runs on the connected Ray cluster."""
    from slime.ray.training_recovery import RECOVERY_NAMESPACE

    try:
        ray.get_actor(f"{session}:serving", namespace=RECOVERY_NAMESPACE)
    except ValueError:
        return False
    return True


def dispose_serving(session: str, timeout_s: float) -> None:
    """Stop the rollout manager and the serving owner of ``session``: its engines, routers and GPUs."""
    from slime.ray.training_recovery import RECOVERY_NAMESPACE

    # The manager first: it borrows the engines that the serving owner holds.
    for name in (session, f"{session}:serving"):
        try:
            actor = ray.get_actor(name, namespace=RECOVERY_NAMESPACE)
        except ValueError:
            continue
        try:
            ray.get(actor.dispose.remote(timeout_s), timeout=timeout_s)
        finally:
            ray.kill(actor, no_restart=True)


class SlimeRolloutGroup:
    """The rollout engines of one Slime training component, behind Slime's ``RolloutManager``.

    Values: ``startup`` is Slime's ``RolloutStartup`` of this driver: the manager's Ray handle, the serving
    session's placements and the arguments with the router's address. ``colocated`` is True when the engines
    share the training GPUs, so they hold GPU memory only between ``sync`` and ``release``. ``router_url`` is
    the non-empty base URL of the engines' router.
    Purpose: ``start`` creates or attaches to the component's serving session; the component syncs the engines
    to the actor's weights before each rollout and releases them before it trains, and ``close`` detaches the
    driver, leaving the engines to the next job.
    """

    def __init__(self, startup: Any, *, colocated: bool, router_url: str) -> None:
        if not router_url:
            raise ValueError("SlimeRolloutGroup.router_url must be non-empty")
        self.startup = startup
        self.manager = startup.manager
        self.colocated = colocated
        self.router_url = router_url

    @classmethod
    def start(cls, args: SlimeArguments, lease: str) -> SlimeRolloutGroup:
        """Create or attach to the serving session of ``args``, whose actors run under ``lease``.

        A new session places the GPUs and starts the engines; colocated engines start offloaded.
        """
        from slime.ray.placement_group import create_rollout_manager

        with serving_lease(lease):
            startup = create_rollout_manager(args)
        router = f"http://{startup.args.sglang_router_ip}:{startup.args.sglang_router_port}"
        return cls(startup, colocated=bool(args.colocate), router_url=router)

    @property
    def placements(self) -> dict[str, GpuBundles | None]:
        """The serving session's GPUs, by role (``actor``, ``critic``, ``rollout``), which the workers train on."""
        return {role: None if value is None else GpuBundles(*value) for role, value in self.startup.placements.items()}

    def sync(self, actor_group: SlimeTrainGroup) -> None:
        """Load the actor's weights into the engines, which then hold GPU memory and serve."""
        if self.colocated:
            ray.get(self.manager.onload_weights.remote())
        actor_group.update_weights()
        if self.colocated:
            ray.get(self.manager.onload_kv.remote())

    def release(self) -> None:
        """Free the colocated engines' GPU memory for training; engines on their own GPUs keep it."""
        if self.colocated:
            ray.get(self.manager.offload.remote())

    def rollouts_per_epoch(self) -> int:
        """The rollouts of one pass over the data source's dataset, in ``--rollout-batch-size`` prompts."""
        return int(ray.get(self.manager.get_num_rollout_per_epoch.remote()))

    def generate(self, rollout_id: int) -> Any:
        """Generate and reward rollout ``rollout_id`` with Slime's rollout function: its training data per DP rank."""
        return ray.get(self.manager.generate.remote(rollout_id))

    def close(self) -> None:
        """Detach this driver from the session: the engines, their router and the GPUs stay for the next job."""
        self.startup.close(failed=True)
