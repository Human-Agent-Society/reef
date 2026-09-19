"""Training allocation and native sender attachment have separate ownership."""

from types import SimpleNamespace

import pytest

pytest.importorskip("ray", reason="requires the optional Ray runtime")

from reef.runtime.deployment import WeightTransferSession
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.train.slime_backend import training
from reef.train.slime_backend.resources import SlimeDeploymentResources


class Group:
    def __init__(self, role, events, *, release_error=False):
        self.role = role
        self.events = events
        self.release_error = release_error
        self.executor = UniProcExecutor.from_workers([object()])
        self.train_parallel_config = {}

    def set_rollout_manager(self, receiver):
        self.events.append(("attach", self.role, receiver))
        self.train_parallel_config = {"attached": True}

    def check_health(self):
        self.events.append(("health", self.role))

    def release(self):
        self.events.append(("release", self.role))
        if self.release_error:
            raise RuntimeError("release failed")
        self.executor.shutdown()


def service_and_resources(monkeypatch, *, critic=False, release_error=False):
    events = []
    actor = Group("actor", events, release_error=release_error)
    critic_group = Group("critic", events) if critic else None
    args = SimpleNamespace()
    preparation = object()
    service = training.SlimeTrainingService(args, preparation=preparation, loss_family_config="recipe-settings")
    resources = SlimeDeploymentResources(args, ray_address="unused", namespace="test")
    resources.placement_groups = {"actor": "actor-pg", "critic": "critic-pg"}

    def create(received, placements, rollout_manager):
        assert received is args
        assert placements is resources.placement_groups
        assert rollout_manager is None
        events.append(("allocate",))
        return actor, critic_group

    monkeypatch.setattr(training, "create_train_groups", create)
    return service, resources, events


@pytest.mark.parametrize("critic", [False, True])
def test_training_initializes_without_receiver_then_attaches_sender(monkeypatch, critic):
    service, resources, events = service_and_resources(monkeypatch, critic=critic)
    service.start(resources)
    assert events == [("allocate",)]
    assert service._session_id is None
    assert not hasattr(service, "_bridge")
    receiver = object()
    control = RayExecutor.from_workers([receiver])
    session = WeightTransferSession(service.weight_transfer_protocol, control, "session-1")
    service.attach_weight_transport(session)
    service.attach_weight_transport(session)
    expected = [("allocate",), ("attach", "actor", receiver)]
    if critic:
        expected.append(("attach", "critic", receiver))
    assert events == expected
    service.check_health()
    service.close()
    service.close()
    assert [event for event in events if event[0] == "release"] == (
        [("release", "critic"), ("release", "actor")] if critic else [("release", "actor")]
    )
    assert not control._closed
    assert resources.placement_groups == {"actor": "actor-pg", "critic": "critic-pg"}


def test_training_operations_receive_groups_and_policy_without_inference(monkeypatch):
    service, resources, _events = service_and_resources(monkeypatch, critic=True)
    service.start(resources)
    with pytest.raises(RuntimeError, match="attached weight transport"):
        service.backend()
    control = RayExecutor.from_workers([object()])
    service.attach_weight_transport(WeightTransferSession(service.weight_transfer_protocol, control, "session-1"))
    captured = []

    def create(args, actor, critic, *, preparation, loss_family_config):
        assert actor.train_parallel_config == {"attached": True}
        assert critic.train_parallel_config == {"attached": True}
        captured.append((args, preparation, loss_family_config))
        return object()

    monkeypatch.setattr(training, "create_training_backend", create)
    assert service.backend() is service.backend()
    assert captured == [(service.args, service.preparation, "recipe-settings")]
    service.close()


def test_native_transport_rejects_reuse_across_receiver_incarnations(monkeypatch):
    service, resources, events = service_and_resources(monkeypatch)
    receiver = RayExecutor.from_workers([object()])
    session = WeightTransferSession(service.weight_transfer_protocol, receiver, "first")
    with pytest.raises(RuntimeError, match="start training workers"):
        service.attach_weight_transport(session)
    service.start(resources)
    with pytest.raises(ValueError, match="incompatible weight transfer"):
        service.attach_weight_transport(WeightTransferSession("different", receiver, "first"))
    service.attach_weight_transport(session)
    with pytest.raises(RuntimeError, match="different deployment"):
        service.attach_weight_transport(WeightTransferSession(session.protocol, receiver, "second"))
    assert len([event for event in events if event[0] == "attach"]) == 1
    service.close()


def test_service_does_not_observe_original_workers_after_operation_handoff(monkeypatch):
    service, resources, _events = service_and_resources(monkeypatch)
    service.start(resources)
    # The coordinator may intentionally retire and recreate these workers.
    # Its operations observe the replacements; the original service must not
    # turn a known retirement into a cold deployment restart.
    service._actor_group.executor._fail("intentionally retired after handoff")
    service.poll()
    service.close()


def test_training_cleanup_failure_preserves_borrowed_receiver_and_reservations(monkeypatch):
    service, resources, events = service_and_resources(monkeypatch, critic=True, release_error=True)
    service.start(resources)
    control = RayExecutor.from_workers([object()])
    service.attach_weight_transport(WeightTransferSession(service.weight_transfer_protocol, control, "first"))
    with pytest.raises(RuntimeError, match="release failed"):
        service.close()
    service.close()
    assert events[-2:] == [("release", "critic"), ("release", "actor")]
    assert not control._closed
    assert resources.placement_groups
