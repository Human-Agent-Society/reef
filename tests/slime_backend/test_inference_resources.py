"""A Slime driver's Ray session and its placement group, which Slime makes and splits."""

import sys
import types

import pytest

from reef.runtime.executor.placement import GpuBundles
from reef.train.slime_backend import resources
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments


@pytest.fixture
def ray_session(monkeypatch):
    """Ray's session and the GPU reservation, recorded in one ordered event list."""

    class Session:
        def __init__(self):
            self.events = []
            self.failure = None
            self.initialized = False
            self.options = None
            self.args = None

        def event(self, name):
            self.events.append(name)
            if name == self.failure:
                raise RuntimeError(f"failed {name}")

    session = Session()

    def connect(**kwargs):
        session.initialized = True
        session.options = kwargs
        session.event("connect")

    def disconnect():
        session.initialized = False
        session.event("disconnect")

    def create_placement_groups(args):
        session.args = args
        session.event("reserve")
        # Slime's split of one group: the critic shares the actor's bundles, the engines take the ones after.
        actor = ("group", [0, 1], [0, 1])
        return {"actor": actor, "critic": actor if args.use_critic else None, "rollout": ("group", [], [])}

    placement = types.ModuleType("slime.ray.placement_group")
    placement.create_placement_groups = create_placement_groups  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.ray.placement_group", placement)
    monkeypatch.setattr(resources.ray.util, "remove_placement_group", lambda group: session.event("release"))
    monkeypatch.setattr(resources.ray, "is_initialized", lambda: session.initialized)
    monkeypatch.setattr(resources.ray, "init", connect)
    monkeypatch.setattr(resources.ray, "shutdown", disconnect)
    monkeypatch.setattr(resources.ray, "nodes", list)
    return session


def component_resources(*, use_critic=False):
    args = SlimeArguments(
        actor_num_nodes=1,
        actor_num_gpus_per_node=2,
        use_critic=use_critic,
    )
    return resources.SlimeDeploymentResources(
        args, ray_address="external", namespace="test", runtime_env={"env_vars": {"PYTHONPATH": "/repo"}}
    )


@pytest.mark.parametrize("use_critic", [False, True])
def test_the_driver_connects_once_reserves_slimes_placement_group_and_releases_it_before_disconnecting(
    ray_session, use_critic
):
    owned = component_resources(use_critic=use_critic)
    owned.start()
    # Slime makes and splits the job's placement group from its own arguments.
    assert ray_session.args is owned.args
    actor = GpuBundles("group", [0, 1], [0, 1])
    assert owned.placement_groups == {
        "actor": actor,
        "critic": actor if use_critic else None,
        "rollout": GpuBundles("group", [], []),
    }
    assert ray_session.options == {
        "address": "external",
        "namespace": "test",
        "runtime_env": {
            "env_vars": {"PYTHONPATH": "/repo", resources.DEPLOYMENT_ENV: owned._process_lease},
            "worker_process_setup_hook": "reef.runtime.executor.process_guard.install",
        },
    }
    with pytest.raises(RuntimeError, match="only be started once"):
        owned.start()
    owned.close()
    owned.close()
    assert ray_session.events == ["connect", "reserve", "release", "disconnect"]
    assert owned.placement_groups == {}


@pytest.mark.parametrize("failure", ["reserve", "release"])
def test_a_failed_reservation_or_release_still_disconnects_the_owned_session(ray_session, failure):
    ray_session.failure = failure
    owned = component_resources()
    with pytest.raises(RuntimeError, match=failure):
        owned.start()
        owned.close()
    owned.close()
    assert ray_session.events[-1] == "disconnect"
    assert ray_session.events.count("disconnect") == 1


def test_an_existing_client_session_is_not_disconnected(ray_session):
    ray_session.initialized = True
    owned = component_resources()
    with pytest.raises(RuntimeError, match="own Ray client session"):
        owned.start()
    owned.close()
    assert ray_session.events == []
    assert ray_session.initialized is True


@pytest.mark.parametrize("confirmed", [True, False])
def test_closing_retires_the_deployments_processes_before_disconnecting(ray_session, monkeypatch, confirmed):
    owned = component_resources()
    owned.start()
    owned._nodes = ["node"]

    def retire():
        ray_session.events.append("retire-processes")
        if not confirmed:
            raise RuntimeError("process cleanup unconfirmed")

    monkeypatch.setattr(owned, "_retire_processes", retire)
    if confirmed:
        owned.close()
    else:
        with pytest.raises(RuntimeError, match="unconfirmed"):
            owned.close()
    assert ray_session.events[-3:] == ["release", "retire-processes", "disconnect"]
