"""Weight publication lifecycle checks for the Megatron runtime adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def actor_module():
    pytest.importorskip("slime.backends.megatron_utils.actor")
    from reef.train.slime_backend.reef_adapters.megatron import train_actor

    return train_actor


@pytest.mark.parametrize("colocate,offload_train", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("lora_rank", [0, 8])
def test_weight_publication_restores_only_colocated_full_weights(
    actor_module, monkeypatch: pytest.MonkeyPatch, colocate: bool, offload_train: bool, lora_rank: int
) -> None:
    actor = object.__new__(actor_module.ReefMegatronTrainRayActor)
    actor.args = SimpleNamespace(
        colocate=colocate, offload_train=offload_train, megatron_lora_rank=lora_rank, use_fault_tolerance=True
    )
    events: list[str] = []
    resident = not (colocate and offload_train and lora_rank == 0)

    def wake_up() -> None:
        nonlocal resident
        resident = True
        events.append("wake")

    def sleep() -> None:
        nonlocal resident
        resident = False
        events.append("sleep")

    def publish(*, manage_generation: bool, force_full: bool) -> str:
        assert resident
        assert manage_generation is False
        assert force_full is True
        assert actor.args.use_fault_tolerance is False
        events.append("publish")
        return "published"

    original_update = publish
    actor.weight_updater = SimpleNamespace(update_weights=original_update)
    actor.wake_up = wake_up
    actor.sleep = sleep

    def native_update(self) -> str:
        events.append("native")
        return self.weight_updater.update_weights()

    monkeypatch.setattr(actor_module.MegatronTrainRayActor, "update_weights", native_update)
    actor._update_lora_weights = Mock(return_value="lora-published")

    result = actor.update_weights(manage_generation=False, force_full=True)

    if lora_rank:
        assert result == "lora-published"
        assert events == []
        actor._update_lora_weights.assert_called_once_with()
    else:
        assert result == "published"
        if colocate and offload_train:
            assert events == ["wake", "native", "publish", "sleep"]
            assert resident is False
        else:
            assert events == ["native", "publish"]
    assert actor.weight_updater.update_weights is original_update
    assert actor.args.use_fault_tolerance is True


@pytest.mark.parametrize("manage_generation", [True, False])
@pytest.mark.parametrize("fail_at", ["wake", "publish"])
def test_failed_publication_restores_updater_and_generation_policy(
    actor_module, monkeypatch: pytest.MonkeyPatch, manage_generation: bool, fail_at: str
) -> None:
    actor = object.__new__(actor_module.ReefMegatronTrainRayActor)
    actor.args = SimpleNamespace(colocate=True, offload_train=True, megatron_lora_rank=0, use_fault_tolerance=True)
    events: list[str] = []

    def wake_up() -> None:
        events.append("wake")
        if fail_at == "wake":
            raise RuntimeError("restore failed")

    def publish(*, manage_generation: bool, force_full: bool) -> None:
        events.append("publish")
        assert actor.args.use_fault_tolerance is manage_generation
        raise RuntimeError("transport failed")

    original_update = publish
    actor.weight_updater = SimpleNamespace(update_weights=original_update)
    actor.wake_up = wake_up
    actor.sleep = lambda: events.append("sleep")
    monkeypatch.setattr(
        actor_module.MegatronTrainRayActor, "update_weights", lambda self: self.weight_updater.update_weights()
    )

    with pytest.raises(RuntimeError, match="restore failed|transport failed"):
        actor.update_weights(manage_generation=manage_generation)

    if fail_at == "publish":
        assert events == ["wake", "publish", "sleep"]
    else:
        assert events == ["wake"]
    assert actor.weight_updater.update_weights is original_update
    assert actor.args.use_fault_tolerance is True
