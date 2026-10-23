"""Reef reserves one model deployment's GPUs and slices them per component."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from reef.runtime.executor.placement import GpuBundles, ModelGpuLayout, ModelGpuReservation, reserve_model_gpus


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layout", "total", "training", "inference"),
    [
        (ModelGpuLayout(2, 2), 4, [0, 1], [2, 3]),
        (ModelGpuLayout(2, 4, colocate=True), 4, [0, 1], [0, 1, 2, 3]),
        (ModelGpuLayout(4, 2, colocate=True), 4, [0, 1, 2, 3], [0, 1, 2, 3]),
        (ModelGpuLayout(2, 0, external_inference=True), 2, [0, 1], []),
        (ModelGpuLayout(0, 2), 2, [], [0, 1]),
    ],
)
def test_layout_slices_one_reservation_for_both_components(layout, total, training, inference):
    assert layout.total == total
    bundles = GpuBundles(SimpleNamespace(id="pg"), list(range(total)), [10 + index for index in range(total)])
    reservation = ModelGpuReservation(bundles, layout)
    assert reservation.training.bundle_indices == training
    assert reservation.inference.bundle_indices == inference
    assert reservation.inference.gpu_ids == [10 + index for index in inference]
    assert reservation.training.group is reservation.inference.group is bundles.group


@pytest.mark.unit
def test_layout_rejects_impossible_shapes():
    with pytest.raises(ValueError, match="training GPUs"):
        ModelGpuLayout(-1, 2)
    with pytest.raises(ValueError, match="hosted trainer"):
        ModelGpuLayout(0, 2, colocate=True)
    with pytest.raises(ValueError, match="inference GPUs"):
        ModelGpuLayout(2, 0)
    with pytest.raises(ValueError, match="do not match"):
        ModelGpuReservation(GpuBundles(None, [0], [0]), ModelGpuLayout(2, 2))


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
            assert reservation.training_node_id == ray.get_runtime_context().get_node_id()
        finally:
            reservation.release()
        hosted = reserve_model_gpus(ModelGpuLayout(training_gpus=0, inference_gpus=2), wait_log_interval_s=1)
        try:
            assert hosted.training_node_id is None
        finally:
            hosted.release()
        assert not ray.util.placement_group_table() or all(
            entry["state"] == "REMOVED" for entry in ray.util.placement_group_table().values()
        )
    finally:
        ray.shutdown()
