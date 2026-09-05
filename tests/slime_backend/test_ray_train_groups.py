from __future__ import annotations

from types import SimpleNamespace

import pytest
from slime.ray import actor_group

from reef.runtime.executor import ExecutorFuture, resolve
from reef.train.slime_backend.reef_adapters import ray_train_groups
from reef.train.slime_backend.reef_adapters.executors.ray import SlimeRayExecutor
from reef.train.slime_backend.reef_adapters.train_groups import SlimeTrainGroup


class _RemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _worker(rank):
    return SimpleNamespace(
        init=_RemoteMethod(0),
        restore_runtime_load_id_for_republication=_RemoteMethod(f"restore-{rank}"),
        pop_metrics=_RemoteMethod({"rank": rank}),
        get_runtime_load_id=_RemoteMethod("incarnation:4"),
        update_weights=_RemoteMethod(f"updated-{rank}"),
    )


@pytest.fixture
def ray_group(monkeypatch: pytest.MonkeyPatch):
    workers = [_worker(0), _worker(1)]
    allocations = []
    killed = []
    reservation = object()
    placement = (reservation, [3, 1], [7, 5])

    def allocate(launcher, pg, num_gpus_per_actor):
        allocations.append((pg, num_gpus_per_actor, launcher.role))
        launcher._actor_handlers = list(workers)

    monkeypatch.setattr(actor_group.RayTrainGroup, "_allocate_gpus_for_actor", allocate)
    monkeypatch.setattr(actor_group.ray, "get", lambda value, timeout=None: value)
    monkeypatch.setattr(actor_group.ray, "kill", lambda actor, *, no_restart: killed.append((actor, no_restart)))
    monkeypatch.setattr(actor_group.time, "sleep", lambda duration: None)
    group = ray_train_groups.ReefRayTrainGroup(
        args=SimpleNamespace(update_weight_mode="full", update_weight_transport="nccl"),
        num_nodes=1,
        num_gpus_per_node=2,
        pg=placement,
        num_gpus_per_actor=0.4,
    )
    assert group.create() == [0, 0]
    yield group, workers, allocations, killed, placement
    group.release()


@pytest.mark.unit
def test_bridge_facing_train_group_methods_fan_out(ray_group) -> None:
    group, workers, *_ = ray_group

    assert ray_train_groups.ReefRayTrainGroup is SlimeTrainGroup
    assert isinstance(group.executor, SlimeRayExecutor)
    assert group.restore_runtime_load_id_for_republication("incarnation:4") == ["restore-0", "restore-1"]
    metrics = group.async_pop_rank0_metrics()
    version = group.async_get_rank0_runtime_load_id()
    assert isinstance(metrics, ExecutorFuture)
    assert resolve(metrics) == {"rank": 0}
    assert resolve(version) == "incarnation:4"
    assert workers[0].restore_runtime_load_id_for_republication.calls == [(("incarnation:4",), {})]
    assert workers[1].restore_runtime_load_id_for_republication.calls == [(("incarnation:4",), {})]
    assert workers[1].pop_metrics.calls == []
    assert workers[1].get_runtime_load_id.calls == []


@pytest.mark.unit
def test_non_disk_weight_update_preserves_default_and_explicit_worker_flags(ray_group) -> None:
    group, workers, *_ = ray_group

    assert group.update_weights() == ["updated-0", "updated-1"]
    assert group.update_weights(manage_generation=False, force_full=True) == ["updated-0", "updated-1"]
    for worker in workers:
        assert worker.update_weights.calls == [
            ((), {}),
            ((), {"manage_generation": False, "force_full": True}),
        ]


@pytest.mark.unit
def test_ray_launcher_preserves_placement_and_releases_only_owned_workers(ray_group) -> None:
    group, workers, allocations, killed, placement = ray_group

    assert allocations == [(placement, 0.4, "actor")]
    assert group.executor._launcher._pg is placement
    group.release()
    group.release()

    # The generic RPC wrapper borrows these handles. Slime's launcher alone
    # owns their teardown; the reservation can still be shared with rollout.
    assert killed == [(worker, True) for worker in workers]
    assert all(target is not placement[0] for target, _ in killed)
    with pytest.raises(RuntimeError, match="not running"):
        _ = group.executor
