from __future__ import annotations

from types import SimpleNamespace

import pytest
from slime.ray import actor_group

from reef.runtime.executor import ExecutorFuture, resolve
from reef.train.slime_backend.reef_adapters.executors.ray import SlimeRayExecutor
from reef.train.slime_backend.reef_adapters.train_groups import SlimeTrainGroup, merge_rank_metrics


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
        pop_metrics=_RemoteMethod({"rank": rank}),
        save_model=_RemoteMethod(f"saved-{rank}"),
        parallel_layout=_RemoteMethod({"dp_size": 2, "cp_size": 1}),
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
    group = SlimeTrainGroup(
        args=SimpleNamespace(reef_executor_backend="auto", reef_train_executor_options={}),
        num_nodes=1,
        num_gpus_per_node=2,
        pg=placement,
        num_gpus_per_actor=0.4,
    )
    assert group.create() == [0, 0]
    yield group, workers, allocations, killed, placement
    group.release()


@pytest.mark.unit
def test_train_group_methods_fan_out(ray_group) -> None:
    group, workers, *_ = ray_group

    assert isinstance(group.executor, SlimeRayExecutor)
    # The start read the workers' parallel layout, which the batch split reads.
    assert group.train_parallel_config == {"dp_size": 2, "cp_size": 1}
    assert group.save_model(4, force_sync=True) == ["saved-0", "saved-1"]
    for worker in workers:
        assert worker.parallel_layout.calls == [((), {})]
        assert worker.save_model.calls == [((4,), {"force_sync": True})]
    metrics = group.async_pop_metrics()
    assert isinstance(metrics, ExecutorFuture)
    assert resolve(metrics) == [{"rank": 0}, {"rank": 1}]  # every rank drains: the last pipeline stage logs a step


@pytest.mark.unit
def test_workers_that_disagree_on_the_parallel_layout_are_refused(ray_group) -> None:
    group, workers, *_ = ray_group
    group.release()
    workers[1].parallel_layout.result = {"dp_size": 1, "cp_size": 2}

    with pytest.raises(RuntimeError, match="disagree on training parallel config"):
        group.create()
    with pytest.raises(RuntimeError, match="not running"):
        _ = group.executor


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


@pytest.mark.unit
def test_rank_metrics_merge_keeps_the_last_stage_step_and_rank0_on_a_shared_key() -> None:
    first_stage = {"train/step": 3, "perf/update_weights_time": 1.5}
    last_stage = {"train/step": 3, "train/loss": 1.25, "train/grad_norm": 2.0, "perf/update_weights_time": 9.0}

    assert merge_rank_metrics([first_stage, {}, last_stage]) == {
        "train/step": 3,
        "train/loss": 1.25,
        "train/grad_norm": 2.0,
        "perf/update_weights_time": 1.5,
    }
