"""The Slime component's Ray session and its one reservation of the training GPUs."""

import pytest

from reef.runtime.executor.placement import ModelGpuLayout
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
            self.layout = None

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

    class Reservation:
        training = "training-bundles"

        def release(self):
            session.event("release")

    def reserve(layout):
        session.layout = layout
        session.event("reserve")
        return Reservation()

    monkeypatch.setattr(resources, "reserve_model_gpus", reserve)
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
def test_the_component_connects_once_reserves_its_training_gpus_and_releases_them_before_disconnecting(
    ray_session, use_critic
):
    owned = component_resources(use_critic=use_critic)
    owned.start()
    # The serving engines reserve their own GPUs on the inference component: the component reserves training GPUs.
    assert ray_session.layout == ModelGpuLayout(training_gpus=2, inference_gpus=0)
    # The actor and the critic share the training bundles.
    assert owned.placement_groups == {
        "actor": "training-bundles",
        "critic": "training-bundles" if use_critic else None,
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


@pytest.mark.parametrize(("rollout_num_gpus", "rollout_external"), [(4, False), (0, True), (0, False)])
def test_the_reservation_holds_only_the_training_gpus(rollout_num_gpus, rollout_external):
    layout = resources.model_gpu_layout(
        SlimeArguments(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            rollout_num_gpus=rollout_num_gpus,
            rollout_external=rollout_external,
        )
    )
    assert layout == ModelGpuLayout(16, 0)
    assert layout.total == 16
