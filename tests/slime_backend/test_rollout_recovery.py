from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_rollout_module(monkeypatch: pytest.MonkeyPatch):
    ray = types.ModuleType("ray")
    ray.get = lambda handles, **_kwargs: handles  # type: ignore[attr-defined]
    ray.kill = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    ray.remote = lambda value: value  # type: ignore[attr-defined]
    ray.__path__ = []  # type: ignore[attr-defined]
    ray_util = types.ModuleType("ray.util")
    ray_util.__path__ = []  # type: ignore[attr-defined]
    scheduling = types.ModuleType("ray.util.scheduling_strategies")
    scheduling.PlacementGroupSchedulingStrategy = type(  # type: ignore[attr-defined]
        "PlacementGroupSchedulingStrategy", (), {}
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", ray_util)
    monkeypatch.setitem(sys.modules, "ray.util.scheduling_strategies", scheduling)

    sglang = types.ModuleType("sglang")
    sglang.__path__ = []  # type: ignore[attr-defined]
    srt = types.ModuleType("sglang.srt")
    srt.__path__ = []  # type: ignore[attr-defined]
    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"  # type: ignore[attr-defined]
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"  # type: ignore[attr-defined]
    constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"  # type: ignore[attr-defined]
    sglang_engine = types.ModuleType("slime.backends.sglang_utils.sglang_engine")
    sglang_engine.SGLangEngine = type("SGLangEngine", (), {})  # type: ignore[attr-defined]
    sglang_engine.resolve_sglang_engine_class = lambda _args: sglang_engine.SGLangEngine  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants)

    def stub(name: str, **attributes: object) -> None:
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    stub(
        "slime.backends.sglang_utils.external",
        start_external_rollout_servers=lambda *_args, **_kwargs: None,
    )
    stub(
        "slime.backends.sglang_utils.sglang_config",
        ModelConfig=type("ModelConfig", (), {}),
        ServerGroupConfig=type("ServerGroupConfig", (), {}),
        SglangConfig=type("SglangConfig", (), {}),
    )
    monkeypatch.setitem(
        sys.modules,
        "slime.backends.sglang_utils.sglang_engine",
        sglang_engine,
    )
    stub(
        "slime.utils.dp_schedule",
        build_dp_schedule=lambda *_args, **_kwargs: None,
    )
    stub(
        "slime.utils.health_monitor",
        RolloutHealthMonitor=type("RolloutHealthMonitor", (), {}),
    )
    stub(
        "slime.utils.http_utils",
        _wrap_ipv6=lambda host: host,
        find_available_port=lambda *_args, **_kwargs: 0,
        get_host_info=lambda: ("localhost", "127.0.0.1"),
        init_http_client=lambda *_args, **_kwargs: None,
    )
    stub(
        "slime.utils.logging_utils",
        configure_logger=lambda *_args, **_kwargs: None,
        init_tracking=lambda *_args, **_kwargs: None,
    )
    stub("slime.rollout.base_types", call_rollout_fn=lambda *_args, **_kwargs: None)
    stub("slime.rollout.sample_hooks", set_current_rollout_id=lambda _value: None)
    stub("slime.utils.data", get_source=lambda _sample: "test")
    stub(
        "slime.utils.metric_utils",
        compute_pass_rate=lambda *_args, **_kwargs: {},
        compute_rollout_step=lambda *_args, **_kwargs: 0,
        compute_statistics=lambda *_args, **_kwargs: {},
        dict_add_prefix=lambda values, _prefix: values,
        has_repetition=lambda _value: False,
    )
    stub(
        "slime.utils.misc",
        Box=type("Box", (), {}),
        decode_int32_meta_array=lambda _value: [],
        group_by=lambda values, _key: {"test": values},
        load_function=lambda _path: None,
    )
    stub("slime.utils.types", Sample=type("Sample", (), {}))
    stub(
        "slime.ray.rollout_validation",
        validate_server_group_gpu_indices=lambda **_kwargs: None,
    )
    stub("slime.ray.ray_actor", RayActor=type("RayActor", (), {}))
    stub(
        "slime.ray.utils",
        NOSET_VISIBLE_DEVICES_ENV_VARS_LIST=[],
        Lock=type("Lock", (), {}),
        add_default_ray_env_vars=lambda env_vars=None: env_vars or {},
    )

    module_name = "slime.ray._rollout_recovery_test"
    installed = importlib.util.find_spec("slime.ray.rollout")
    assert installed is not None and installed.origin is not None
    path = Path(installed.origin)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _load_manager_module(monkeypatch: pytest.MonkeyPatch):
    raw_rollout = _load_rollout_module(monkeypatch)
    path = Path(__file__).parents[2] / "reef" / "train" / "slime_backend" / "reef_adapters" / "rollout" / "manager.py"
    name = "reef.train.slime_backend.reef_adapters._rollout_manager_recovery_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return raw_rollout, module


class _ServerGroup:
    def __init__(self, engines: list[object | None], *, num_new_engines: int) -> None:
        self.all_engines = engines
        self.num_new_engines = num_new_engines
        self.worker_type = "regular"
        self.needs_offload = False
        self.num_gpus_per_engine = 1
        self.gpu_offset = 0
        self.start_calls = 0

    @property
    def engines(self):
        return self.all_engines

    def parallel_config(self):
        return {"tp_size": 1, "pp_size": 1, "ep_size": 1, "moe_dp_size": 1}

    def start_engines(self, port_cursors):
        self.start_calls += 1
        dead_indices = [i for i, engine in enumerate(self.all_engines) if engine is None]
        for index in dead_indices:
            self.all_engines[index] = object()
        self.num_new_engines = len(dead_indices)
        return [], port_cursors


class _RemoteMethod:
    def __init__(self, result: object) -> None:
        self.result = result

    def remote(self, *_args, **_kwargs):
        return self.result


@pytest.mark.unit
def test_healthy_recovery_preserves_pending_initial_engine_count(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_rollout_module(monkeypatch)
    from reef.train.slime_backend.reef_adapters.rollout.manager import recover_server

    group = _ServerGroup([object(), object()], num_new_engines=2)
    server = module.RolloutServer(server_groups=[group])

    recover_server(server)

    assert group.start_calls == 0
    assert group.num_new_engines == 2


@pytest.mark.unit
def test_recovery_still_starts_dead_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_rollout_module(monkeypatch)
    from reef.train.slime_backend.reef_adapters.rollout.manager import recover_server

    group = _ServerGroup([object(), None], num_new_engines=0)
    server = module.RolloutServer(server_groups=[group])

    recover_server(server)

    assert group.start_calls == 1
    assert group.num_new_engines == 1
    assert all(engine is not None for engine in group.all_engines)


@pytest.mark.unit
def test_reef_external_fields_are_tensorized_without_patching_slime(monkeypatch: pytest.MonkeyPatch) -> None:
    _load_rollout_module(monkeypatch)
    from reef.train.slime_backend.reef_adapters.rollout.manager import tensorize_external_fields

    data = {
        "action_masks": [[1, 0]],
        "advantages": [[0.5, -0.25]],
        "topk_indices": [[[3, 7], [4, 8]]],
        "prm_teacher_topk_log_probs_cand": [[[[-0.1, -0.2], [-0.3, -0.4]]]],
    }

    from reef.train.slime_backend.loss_families import LOSS_FAMILIES

    # Each family declares its own columns; the manager applies whatever the
    # active family declared. Both sets here, so one call covers the table.
    declared = {
        **LOSS_FAMILIES.resolve("sao").rollout_tensor_dtypes,
        **LOSS_FAMILIES.resolve("openclawrl").rollout_tensor_dtypes,
    }
    tensorize_external_fields(data, declared)

    assert data["action_masks"][0].dtype == torch.int32
    assert data["advantages"][0].dtype == torch.float32
    assert data["topk_indices"][0].dtype == torch.int64
    assert data["prm_teacher_topk_log_probs_cand"][0].is_contiguous()


@pytest.mark.unit
def test_retracting_pause_clears_the_shared_cache_before_a_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing the previous weights built may survive into the next ones."""
    raw_rollout, module = _load_manager_module(monkeypatch)

    class _Recorder:
        def __init__(self, calls: list[tuple[str, object]], name: str) -> None:
            self.calls = calls
            self.name = name

        def remote(self, *args, **_kwargs):
            self.calls.append((self.name, args[0] if args else None))
            return self.name

    def _paused(mode: str) -> list[tuple[str, object]]:
        calls: list[tuple[str, object]] = []
        engine = types.SimpleNamespace(
            pause_generation=_Recorder(calls, "pause"),
            flush_cache=_Recorder(calls, "flush"),
        )
        group = _ServerGroup([engine], num_new_engines=0)
        manager = object.__new__(module.ReefRolloutManagerImpl)
        manager.servers = {"actor": raw_rollout.RolloutServer(server_groups=[group])}
        manager.args = types.SimpleNamespace(weight_update_pause_mode=mode)
        manager.pause_generation_for_update()
        return calls

    assert _paused("retract") == [("pause", "retract"), ("flush", None)]
    assert _paused("in_place") == [("pause", "in_place")], "an engine that keeps in-flight KV keeps its cache too"


@pytest.mark.unit
@pytest.mark.parametrize("failed_phase", ["pause", "flush"])
def test_failed_pause_barrier_retires_engines(monkeypatch: pytest.MonkeyPatch, failed_phase: str) -> None:
    raw_rollout, module = _load_manager_module(monkeypatch)

    class _Remote:
        def __init__(self, phase):
            self.phase = phase

        def remote(self, *args, **kwargs):
            return self.phase

    def resolve(handles, **kwargs):
        if failed_phase in handles:
            raise TimeoutError(failed_phase)
        return handles

    monkeypatch.setattr(module.ray, "get", resolve)

    engines = [
        types.SimpleNamespace(
            pause_generation=_Remote("pause"), flush_cache=_Remote("flush"), shutdown=_Remote("shutdown")
        )
        for _ in range(2)
    ]
    group = _ServerGroup(engines, num_new_engines=0)
    manager = object.__new__(module.ReefRolloutManagerImpl)
    manager.args = types.SimpleNamespace(weight_update_pause_mode="retract")
    manager.servers = {"actor": raw_rollout.RolloutServer(server_groups=[group])}
    manager._health_monitors = []
    manager._generation_paused_for_update = False
    with pytest.raises(TimeoutError, match=failed_phase):
        manager.pause_generation_for_update()
    assert group.all_engines == [None, None]
    assert manager._generation_paused_for_update is (failed_phase == "flush")


@pytest.mark.unit
def test_uncertain_weight_update_synchronously_retires_managed_engine_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_rollout, module = _load_manager_module(monkeypatch)
    engines = [
        types.SimpleNamespace(shutdown=_RemoteMethod("shutdown-0")),
        types.SimpleNamespace(shutdown=_RemoteMethod("shutdown-1")),
    ]
    group = _ServerGroup(list(engines), num_new_engines=0)
    server = raw_rollout.RolloutServer(server_groups=[group])
    pauses: list[str] = []
    manager = object.__new__(module.ReefRolloutManagerImpl)
    manager.servers = {"actor": server}
    manager._health_monitors = [types.SimpleNamespace(pause=lambda: pauses.append("pause"))]
    killed = []
    monkeypatch.setattr(module.ray, "kill", lambda engine, **kwargs: killed.append((engine, kwargs)))

    assert manager.terminate_updatable_engines() == 2
    assert group.all_engines == [None, None]
    assert [engine for engine, _ in killed] == engines
    assert all(kwargs == {"no_restart": True} for _, kwargs in killed)
    assert pauses == ["pause"]


@pytest.mark.unit
def test_uncertain_weight_update_keeps_deployment_owned_external_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, module = _load_manager_module(monkeypatch)
    engine = types.SimpleNamespace(shutdown=_RemoteMethod("shutdown"))
    server = types.SimpleNamespace(update_weights=True, server_groups=[], engines=[engine])
    manager = object.__new__(module.ReefRolloutManagerImpl)
    manager.servers = {"actor": server}
    manager._health_monitors = []
    killed = []
    monkeypatch.setattr(module.ray, "kill", lambda actor, **kwargs: killed.append((actor, kwargs)))

    assert manager.terminate_updatable_engines() == 0
    assert server.engines == [engine]
    assert killed == []


@pytest.mark.unit
def test_recovered_engine_is_paused_before_an_in_place_update_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_rollout, module = _load_manager_module(monkeypatch)
    lifecycle: list[str] = []

    class RecordingRemote:
        def __init__(self, event: str) -> None:
            self.event = event

        def remote(self, *args):
            lifecycle.append(f"{self.event}:{','.join(map(str, args))}")
            return self.event

    engine = types.SimpleNamespace(pause_generation=RecordingRemote("pause"))
    group = _ServerGroup([engine], num_new_engines=1)
    server = raw_rollout.RolloutServer(server_groups=[group])
    monitor = types.SimpleNamespace(
        pause=lambda: lifecycle.append("monitor_pause"),
        resume=lambda: lifecycle.append("monitor_resume"),
    )
    manager = object.__new__(module.ReefRolloutManagerImpl)
    manager.args = types.SimpleNamespace(weight_update_pause_mode="in_place", rollout_external=False)
    manager.servers = {"actor": server}
    manager.rollout_engine_lock = types.SimpleNamespace(status=_RemoteMethod({"locked": False, "poisoned": False}))
    manager._health_monitors = [monitor]
    manager._generation_paused_for_update = True
    manager._weight_update_reconnect_required = False

    manager.recover_updatable_engines()

    assert lifecycle == ["monitor_pause", "pause:in_place", "monitor_resume"]


@pytest.mark.unit
def test_recovery_replaces_a_poisoned_update_lock_and_forces_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_rollout, module = _load_manager_module(monkeypatch)
    lifecycle: list[str] = []
    old_lock = types.SimpleNamespace(status=_RemoteMethod({"locked": True, "poisoned": True}))
    new_lock = object()
    server = raw_rollout.RolloutServer(server_groups=[_ServerGroup([object()], num_new_engines=0)])
    manager = object.__new__(module.ReefRolloutManagerImpl)
    manager.args = types.SimpleNamespace(rollout_external=False)
    manager.servers = {"actor": server}
    manager.rollout_engine_lock = old_lock
    manager._health_monitors = [
        types.SimpleNamespace(
            pause=lambda: lifecycle.append("monitor_pause"),
            resume=lambda: lifecycle.append("monitor_resume"),
        )
    ]
    manager._generation_paused_for_update = False
    manager._weight_update_reconnect_required = False
    manager._new_rollout_engine_lock = lambda: new_lock
    killed = []
    monkeypatch.setattr(module.ray, "kill", lambda actor, **kwargs: killed.append((actor, kwargs)))

    result = manager.recover_updatable_engines()

    assert result[1] is new_lock
    assert result[2] == 1
    assert manager._weight_update_reconnect_required is True
    assert killed == [(old_lock, {"no_restart": True})]
    assert lifecycle == ["monitor_pause", "monitor_resume"]


@pytest.mark.unit
def test_poisoned_external_update_requires_deployment_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_rollout, module = _load_manager_module(monkeypatch)
    server = raw_rollout.RolloutServer(server_groups=[_ServerGroup([object()], num_new_engines=0)])
    manager = object.__new__(module.ReefRolloutManagerImpl)
    manager.args = types.SimpleNamespace(rollout_external=True)
    manager.servers = {"actor": server}
    manager.rollout_engine_lock = types.SimpleNamespace(status=_RemoteMethod({"locked": True, "poisoned": True}))
    manager._health_monitors = [types.SimpleNamespace(pause=lambda: None, resume=lambda: None)]
    manager._new_rollout_engine_lock = lambda: pytest.fail("external lock must stay fenced")

    with pytest.raises(RuntimeError, match="restarting the external deployment"):
        manager.recover_updatable_engines()


@pytest.mark.unit
def test_recovery_replaces_an_orphaned_locked_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_rollout, module = _load_manager_module(monkeypatch)
    old_lock = types.SimpleNamespace(status=_RemoteMethod({"locked": True, "poisoned": False}))
    new_lock = object()
    server = raw_rollout.RolloutServer(server_groups=[_ServerGroup([object()], num_new_engines=0)])
    manager = object.__new__(module.ReefRolloutManagerImpl)
    manager.args = types.SimpleNamespace(rollout_external=False)
    manager.servers = {"actor": server}
    manager.rollout_engine_lock = old_lock
    manager._health_monitors = []
    manager._generation_paused_for_update = False
    manager._weight_update_reconnect_required = False
    manager._new_rollout_engine_lock = lambda: new_lock
    monkeypatch.setattr(module.ray, "kill", lambda *_args, **_kwargs: None)

    result = manager.recover_updatable_engines()

    assert result[1] is new_lock
    assert result[2] == 1


@pytest.mark.unit
def test_dp_packaging_preserves_runtime_load_ids_per_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, module = _load_manager_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_dp_schedule",
        lambda *_args, **_kwargs: (
            [[0], [1]],
            [[[0]], [[0]]],
            [1],
            [2],
        ),
    )
    monkeypatch.setattr(module.ray, "put", lambda value, **_kwargs: value, raising=False)
    monkeypatch.setattr(module, "Box", lambda value: value)

    manager = object.__new__(module.ReefRolloutManagerImpl)
    manager.args = types.SimpleNamespace(
        global_batch_size=2,
        rollout_data_transport="object-store",
        custom_rollout_data_keys=(
            "producing_runtime_load_spans",
            "producing_runtime_load_ids",
        ),
    )
    manager.train_parallel_config = {"dp_size": 2}
    spans = [
        [{"start": 0, "end": 1, "runtime_load_id": "engine:1"}],
        [{"start": 0, "end": 1, "runtime_load_id": "engine:2"}],
    ]
    packed = manager._split_train_data_by_dp(
        {
            "tokens": [[10], [20]],
            "rollout_ids": [0, 1],
            "producing_runtime_load_ids": ["engine:1", "engine:2"],
            "producing_runtime_load_spans": spans,
        }
    )

    assert [rank["producing_runtime_load_ids"] for rank in packed] == [["engine:1"], ["engine:2"]]
    assert [rank["producing_runtime_load_spans"] for rank in packed] == [[spans[0]], [spans[1]]]


def _round_robin_dp_schedule(_args, config, total_lengths, *, global_batch_size, rollout_indices):
    """Stand-in packer: one micro-batch per rank per step, samples dealt round-robin."""
    dp_size = config["dp_size"]
    partitions = [list(range(rank, len(total_lengths), dp_size)) for rank in range(dp_size)]
    micro_batches = [[list(range(len(partitions[rank])))] for rank in range(dp_size)]
    return partitions, micro_batches, [1], [global_batch_size]


def _manager_for_schedule(monkeypatch: pytest.MonkeyPatch, *, global_batch_size: int):
    _, module = _load_manager_module(monkeypatch)
    monkeypatch.setattr(module, "build_dp_schedule", _round_robin_dp_schedule)
    monkeypatch.setattr(module.ray, "put", lambda value, **_kwargs: value, raising=False)
    monkeypatch.setattr(module, "Box", lambda value: value)
    manager = object.__new__(module.ReefRolloutManagerImpl)
    manager.args = types.SimpleNamespace(global_batch_size=global_batch_size, rollout_data_transport="object-store")
    manager.train_parallel_config = {"dp_size": 2}
    return manager


@pytest.mark.unit
def test_dp_packaging_splices_reef_step_sizes_with_a_partial_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _manager_for_schedule(monkeypatch, global_batch_size=99)

    packed = manager._split_train_data_by_dp(
        {
            "tokens": [[10], [11], [20], [30], [31], [40]],
            "rollout_ids": [0, 0, 1, 2, 2, 3],
            "external_step_sizes": [2, 2],
        }
    )

    # step 0 = rollouts {0, 1} = samples [0, 1, 2]; step 1 = rollouts {2, 3} = samples [3, 4, 5]
    assert [rank["partition"] for rank in packed] == [[0, 2, 3, 5], [1, 4]]
    assert [[row.tolist() for row in rank["tokens"]] for rank in packed] == [[[10], [20], [30], [40]], [[11], [31]]]
    assert [rank["micro_batch_indices"] for rank in packed] == [[[0, 1], [2, 3]], [[0], [1]]]
    assert packed[0]["num_microbatches"] == [1, 1]
    assert packed[0]["global_batch_sizes"] == [2, 2]

    partial = manager._split_train_data_by_dp(
        {"tokens": [[1], [2], [3], [4], [5]], "rollout_ids": [0, 1, 2, 3, 4], "external_step_sizes": [3, 2]}
    )
    assert partial[0]["global_batch_sizes"] == [3, 2]
    assert [rank["partition"] for rank in partial] == [[0, 2, 3], [1, 4]]


@pytest.mark.unit
def test_dp_packaging_configured_size_honors_remainder_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _manager_for_schedule(monkeypatch, global_batch_size=2)
    data = {"tokens": [[1], [2], [3], [4], [5]], "rollout_ids": [0, 1, 2, 3, 4]}

    with pytest.raises(ValueError, match="do not form complete global batches of 2"):
        manager._split_train_data_by_dp(dict(data))
    with pytest.raises(ValueError, match="do not form complete global batches of 2"):
        manager._split_train_data_by_dp({**data, "external_remainder": "error"})

    packed = manager._split_train_data_by_dp({**data, "external_remainder": "partial"})
    assert packed[0]["global_batch_sizes"] == [2, 2, 1]
    assert packed[0]["num_microbatches"] == [1, 1, 1]
    assert [rank["partition"] for rank in packed] == [[0, 2, 4], [1, 3]]
