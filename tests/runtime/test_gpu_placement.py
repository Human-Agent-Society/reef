"""Reef reserves GPUs in one placement group and slices them into training and inference bundles."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from reef.runtime.deployment import BorrowedGpuResources, select_inference_resources
from reef.runtime.executor.placement import GpuBundles, ModelGpuLayout, ModelGpuReservation, reserve_model_gpus


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layout", "total", "training", "inference"),
    [
        (ModelGpuLayout(2, 2), 4, [0, 1], [2, 3]),
        (ModelGpuLayout(2, 0), 2, [0, 1], []),
        (ModelGpuLayout(0, 2), 2, [], [0, 1]),
    ],
)
def test_layout_slices_one_reservation_into_disjoint_bundles(layout, total, training, inference):
    assert layout.total == total
    bundles = GpuBundles(SimpleNamespace(id="pg"), list(range(total)), [10 + index for index in range(total)])
    reservation = ModelGpuReservation(bundles, layout)
    assert reservation.training.bundle_indices == training
    assert reservation.inference.bundle_indices == inference
    assert reservation.inference.gpu_ids == [10 + index for index in inference]
    assert reservation.training.group is reservation.inference.group is bundles.group


@pytest.mark.unit
def test_layout_rejects_impossible_shapes():
    with pytest.raises(ValueError, match="training_gpus must be a non-negative int"):
        ModelGpuLayout(-1, 2)
    with pytest.raises(ValueError, match="inference_gpus must be a non-negative int"):
        ModelGpuLayout(2, True)
    with pytest.raises(ValueError, match="at least one GPU"):
        ModelGpuLayout(0, 0)
    with pytest.raises(ValueError, match="do not match"):
        ModelGpuReservation(GpuBundles(None, [0], [0]), ModelGpuLayout(2, 2))


def test_inference_can_borrow_a_subset_without_releasing_the_training_reservation():
    group = SimpleNamespace(id="training")
    owner = BorrowedGpuResources(GpuBundles(group, [5, 2, 4, 3], [0, 1, 2, 3]))
    selected = select_inference_resources(owner, 2)
    selected.start()
    assert selected.inference_placement == GpuBundles(group, [5, 2], [0, 1])
    selected.close()
    assert owner.inference_placement == GpuBundles(group, [5, 2, 4, 3], [0, 1, 2, 3])
    for size in (0, -1, True, 1.5, 5):
        with pytest.raises(ValueError):
            select_inference_resources(owner, size)


@pytest.mark.parametrize(
    ("nodes", "devices", "tp", "width"),
    [
        (["a", "a", "b", "b"], [2, 3, 0, 1], 1, 1),
        (["a", "a", "b", "b"], [2, 3, 0, 1], 2, 2),
        (["a", "a", "b", "b"], [2, 3, 0, 1], 4, 2),
        (["a", "b"], [0, 0], 2, 1),
        (["a", "a", "a", "b"], [0, 1, 2, 0], 4, None),
        (["a", "b", "a", "b"], [0, 0, 1, 1], 4, None),
        (["a", "a"], [0, 2], 2, None),
    ],
)
def test_shared_engine_topology_uses_real_node_boundaries(monkeypatch, nodes, devices, tp, width):
    table = {"state": "CREATED", "bundles_to_node_id": dict(enumerate(nodes))}
    monkeypatch.setattr(
        importlib.import_module("ray.util.placement_group"), "placement_group_table", lambda group: table
    )
    resources = BorrowedGpuResources(GpuBundles(object(), list(range(len(nodes))), devices))
    if width is None:
        with pytest.raises(ValueError):
            resources.node_width(tp)
    else:
        assert resources.node_width(tp) == width
    table["state"] = "REMOVED"
    with pytest.raises(ValueError, match="not live"):
        resources.node_width(1)


@pytest.mark.unit
def test_release_returns_the_group_once(monkeypatch):
    released = []
    # Release imports the module lazily; a stand-in keeps this test free of Ray.
    placement_module = ModuleType("ray.util.placement_group")
    placement_module.remove_placement_group = released.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", placement_module)
    group = SimpleNamespace(id="pg")
    reservation = ModelGpuReservation(GpuBundles(group, [0, 1], [0, 1]), ModelGpuLayout(1, 1))
    reservation.release()
    reservation.release()
    assert released == [group]


def test_real_ray_reservation_orders_bundles_by_node_and_device(monkeypatch):
    """A local Ray cluster with pretend GPUs exercises placement without CUDA."""
    ray = pytest.importorskip("ray")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    ray.init(address="local", num_cpus=4, num_gpus=4, include_dashboard=False)
    try:
        reservation = reserve_model_gpus(ModelGpuLayout(training_gpus=2, inference_gpus=2), wait_log_interval_s=1)
        try:
            training, inference = reservation.training, reservation.inference
            assert sorted(training.gpu_ids + inference.gpu_ids) == [0, 1, 2, 3]
            assert training.gpu_ids == [0, 1] and inference.gpu_ids == [2, 3]
            assert set(training.bundle_indices).isdisjoint(inference.bundle_indices)
            assert training.group is inference.group
            borrowed = BorrowedGpuResources(training)
            assert borrowed.node_width(2) == 2
            attached = BorrowedGpuResources.from_config(borrowed.to_config())
            assert attached.bundles.group.id == training.group.id
            assert attached.bundles.bundle_indices == training.bundle_indices
            attached.close()
            assert borrowed.node_width(2) == 2
        finally:
            reservation.release()
        serving = reserve_model_gpus(ModelGpuLayout(training_gpus=0, inference_gpus=2), wait_log_interval_s=1)
        try:
            assert serving.training.bundle_indices == [] and len(serving.inference.gpu_ids) == 2
        finally:
            serving.release()
        assert not ray.util.placement_group_table() or all(
            entry["state"] == "REMOVED" for entry in ray.util.placement_group_table().values()
        )
    finally:
        ray.shutdown()
