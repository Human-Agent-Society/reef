from __future__ import annotations

from types import SimpleNamespace

import pytest
from slime.ray.actor_group import RayTrainGroup

from reef.train.slime_backend.reef_adapters import ray_train_groups


class _RemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def _group(*handlers):
    group = object.__new__(ray_train_groups.ReefRayTrainGroup)
    group._actor_handlers = list(handlers)
    return group


@pytest.mark.unit
def test_bridge_facing_train_group_methods_fan_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ray_train_groups.ray, "get", lambda value: value)
    first = SimpleNamespace(
        restore_weight_version_for_republication=_RemoteMethod("restore-0"),
        pop_metrics=_RemoteMethod("metrics"),
        get_weight_version=_RemoteMethod("incarnation:4"),
    )
    second = SimpleNamespace(
        restore_weight_version_for_republication=_RemoteMethod("restore-1"),
    )
    group = _group(first, second)

    assert group.restore_weight_version_for_republication("incarnation:4") == ["restore-0", "restore-1"]
    assert group.async_pop_rank0_metrics() == "metrics"
    assert group.async_get_rank0_weight_version() == "incarnation:4"


@pytest.mark.unit
def test_train_group_uses_public_update_path_until_full_disk_reload(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(ray_train_groups.ray, "get", lambda value: value)
    monkeypatch.setattr(RayTrainGroup, "update_weights", lambda self: "public-update")
    group = _group(SimpleNamespace())
    group._full_disk_weight_update_enabled = lambda: False

    assert group.update_weights() == "public-update"

    update = _RemoteMethod("updated")
    version = _RemoteMethod("deployment:6")
    group._actor_handlers = [SimpleNamespace(update_weights=update, get_weight_version=version)]
    group._full_disk_weight_update_enabled = lambda: True
    group._release_train_enabled = lambda: True
    group.args = SimpleNamespace(update_weight_disk_dir=str(tmp_path))
    group._disk_weight_version = 5
    released: list[bool] = []
    reloaded: list[tuple[object, int, str, bool]] = []
    group.release = lambda: released.append(True)
    group._reload_rollout_weights_from_disk = lambda path, sequence, exact, *, manage_generation: reloaded.append(
        (path, sequence, exact, manage_generation)
    )

    assert group.update_weights() is None
    assert group._disk_weight_version == 6
    assert update.calls == [((), {})]
    assert released == [True]
    assert reloaded == [(tmp_path / "weight_v000006", 6, "deployment:6", True)]
