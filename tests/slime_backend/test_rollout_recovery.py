from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from reef.execution.executor.delegating import DelegatingExecutor
from reef.execution.executor.uniproc import UniProcExecutor
from reef.inference.sglang.config import SGLangSettings

torch = pytest.importorskip("torch")


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for Ray, SGLang and the Slime modules the batch processor imports."""
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
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants)

    def stub(name: str, **attributes: object) -> None:
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)

    # The batch processor's Slime imports.
    stub("slime.utils.dp_schedule", build_dp_schedule=lambda *_args, **_kwargs: None)
    stub("slime.utils.misc", Box=type("Box", (), {}))
    stub("slime.utils.types", Sample=type("Sample", (), {}))


def _load_manager_module(monkeypatch: pytest.MonkeyPatch, *, serving: bool = False):
    _install_stubs(monkeypatch)
    path = Path(__file__).parents[2] / "reef" / "train" / "slime_backend" / "reef_adapters" / "batches.py"
    if serving:
        path = Path(__file__).parents[2] / "reef" / "inference" / "sglang" / "worker.py"
    name = "reef.train.slime_backend.reef_adapters._rollout_manager_recovery_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class _RecordingRolloutExecutor(DelegatingExecutor):
    def _init_executor(self):
        self.calls = []
        self.closed = False
        executor = self

        class Worker:
            def __getattr__(self, name):
                def invoke(*args, **kwargs):
                    executor.calls.append((name, args, kwargs))
                    return name

                return invoke

            def shutdown(self):
                executor.closed = True

        self._rpc = UniProcExecutor.from_workers([Worker()], owned=True)


def test_rollout_manager_routes_entire_serving_lifecycle_through_custom_executor(monkeypatch):
    from reef.inference.sglang.control import SGLangControl

    args = SGLangSettings("model", 1, 1, 1, executor=_RecordingRolloutExecutor)
    manager = SGLangControl(args, "placement")
    executor = manager._serving
    assert executor.config.options == {"config": args, "pg": "placement"}
    methods = [
        "inference_url",
        "get_runtime_load_ids",
        "pause_generation_for_update",
        "continue_generation_after_update",
        "terminate_updatable_engines",
        "updatable_engines",
        "recover_updatable_engines",
    ]
    for method in methods:
        assert getattr(manager, method)() == method
    assert manager.load_adapter_from_disk("adapter", "/release/adapter", "deployment:3") == "load_adapter_from_disk"
    assert manager.load_weights_from_disk("/release", "deployment:4") == "load_weights_from_disk"
    assert executor.calls[-2:] == [
        ("load_adapter_from_disk", ("adapter", "/release/adapter", "deployment:3"), {}),
        ("load_weights_from_disk", ("/release", "deployment:4"), {}),
    ]
    manager.shutdown()
    assert executor.closed


def test_rollout_init_failure_releases_already_launched_engines(monkeypatch):
    module = _load_manager_module(monkeypatch, serving=True)
    engine = types.SimpleNamespace(shutdown=_RemoteMethod("shutdown"))

    class Cluster:
        def __init__(self, config, pg):
            self.group = types.SimpleNamespace(all_engines=[engine])
            self.routers = []

        def start(self):
            raise RuntimeError("failed init")

    monkeypatch.setattr(module, "SGLangCluster", Cluster)
    monkeypatch.setattr(
        module.ray, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failed init"))
    )
    killed = []
    monkeypatch.setattr(module.ray, "kill", lambda worker, **kwargs: killed.append(worker))
    args = SGLangSettings("model", 1, 1, 1)
    with pytest.raises(RuntimeError, match="failed init"):
        module.SGLangWorker(args, "borrowed-placement")
    assert killed == [engine]


def _worker(module, group, config=None):
    """A bare serving worker over ``group``, with the inference control its constructor builds."""
    from reef.inference.recovery import InferenceControl

    worker = object.__new__(module.SGLangWorker)
    worker.config = SGLangSettings("model", 1, 1, 1) if config is None else config
    worker.cluster = types.SimpleNamespace(group=group, routers=[])
    worker._closed = False
    worker._control = InferenceControl(module._SGLangEngineGroup(worker))
    return worker


class _ServerGroup:
    def __init__(self, engines: list[object | None]) -> None:
        self.all_engines = engines
        self.recover_calls = 0

    def recover(self):
        self.recover_calls += 1
        self.all_engines[:] = [engine or object() for engine in self.all_engines]


class _RemoteMethod:
    def __init__(self, result: object) -> None:
        self.result = result

    def remote(self, *_args, **_kwargs):
        return self.result


@pytest.mark.unit
def test_recovery_starts_only_dead_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_manager_module(monkeypatch, serving=True)
    healthy = _ServerGroup([object(), object()])
    assert _worker(module, healthy).recover_updatable_engines() == 0
    assert healthy.recover_calls == 0

    group = _ServerGroup([object(), None])
    assert _worker(module, group).recover_updatable_engines() == 1
    assert group.recover_calls == 1
    assert all(engine is not None for engine in group.all_engines)


@pytest.mark.unit
def test_reef_external_fields_are_tensorized_without_patching_slime(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stubs(monkeypatch)
    from reef.train.slime_backend.reef_adapters.batches import tensorize_external_fields

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
def test_uncertain_weight_update_synchronously_retires_engine_handles(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_manager_module(monkeypatch, serving=True)
    engines = [
        types.SimpleNamespace(shutdown=_RemoteMethod("shutdown-0")),
        types.SimpleNamespace(shutdown=_RemoteMethod("shutdown-1")),
    ]
    group = _ServerGroup(list(engines))
    manager = _worker(module, group, SGLangSettings("model", 1, 1, 1, pause_mode="retract"))
    killed = []
    monkeypatch.setattr(module.ray, "kill", lambda engine, **kwargs: killed.append((engine, kwargs)))

    assert manager.terminate_updatable_engines() == 2
    assert group.all_engines == [None, None]
    assert [engine for engine, _ in killed] == engines
    assert all(kwargs == {"no_restart": True} for _, kwargs in killed)


@pytest.mark.unit
def test_recovered_engine_is_paused_before_an_in_place_update_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_manager_module(monkeypatch, serving=True)
    lifecycle: list[str] = []

    class RecordingRemote:
        def __init__(self, event: str) -> None:
            self.event = event

        def remote(self, *args):
            lifecycle.append(f"{self.event}:{','.join(map(str, args))}")
            return self.event

    engine = types.SimpleNamespace(pause_generation=RecordingRemote("pause"))
    manager = _worker(module, _ServerGroup([engine]))
    manager._control.paused = True

    manager.recover_updatable_engines()

    assert lifecycle == ["pause:in_place"]


@pytest.mark.unit
@pytest.mark.parametrize("pause_mode", ["retract", "in_place"])
def test_retracting_pause_clears_the_shared_cache_before_a_publication(
    monkeypatch: pytest.MonkeyPatch, pause_mode: str
) -> None:
    """Nothing the previous weights built may survive into the next ones."""
    module = _load_manager_module(monkeypatch, serving=True)
    calls: list[str] = []

    class RecordingRemote:
        def __init__(self, event: str) -> None:
            self.event = event

        def remote(self, *args):
            calls.append(":".join((self.event, *map(str, args))))
            return self.event

    engines = [
        types.SimpleNamespace(pause_generation=RecordingRemote("pause"), flush_cache=RecordingRemote("flush"))
        for _ in range(2)
    ]
    # The dead slot recovers later; the pause fences the engines that survive.
    group = _ServerGroup([*engines, None])
    manager = _worker(module, group, SGLangSettings("model", 1, 1, 1, pause_mode=pause_mode))

    manager.pause_generation_for_update()

    if pause_mode == "retract":
        assert calls == ["pause:retract", "pause:retract", "flush", "flush"]
    else:
        assert calls == ["pause:in_place", "pause:in_place"], "an engine that keeps in-flight KV keeps its cache"


@pytest.mark.unit
def test_dp_packaging_preserves_producing_artifact_ids_per_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_manager_module(monkeypatch)
    monkeypatch.setattr(
        sys.modules["slime.utils.dp_schedule"],
        "build_dp_schedule",
        lambda *_args, **_kwargs: (
            [[0], [1]],
            [[[0]], [[0]]],
            [1],
            [2],
        ),
    )
    monkeypatch.setattr(module.ray, "put", lambda value, **_kwargs: value, raising=False)
    monkeypatch.setattr(sys.modules["slime.utils.misc"], "Box", lambda value: value)

    args = types.SimpleNamespace(
        global_batch_size=2,
        rollout_data_transport="object-store",
        custom_rollout_data_keys=("producing_artifact_ids",),
        reef_rollout_tensor_dtypes={},
    )
    manager = module.TrainingBatchProcessor(args, {"dp_size": 2})
    packed = manager._split_train_data_by_dp(
        {
            "tokens": [[10], [20]],
            "rollout_ids": [0, 1],
            "external_step_sizes": [2],
            "producing_artifact_ids": ["artifact-1", "artifact-2"],
        }
    )

    assert [rank["producing_artifact_ids"] for rank in packed] == [["artifact-1"], ["artifact-2"]]


def _round_robin_dp_schedule(_args, config, total_lengths, *, global_batch_size, rollout_indices):
    """Stand-in packer: one micro-batch per rank per step, samples dealt round-robin."""
    dp_size = config["dp_size"]
    partitions = [list(range(rank, len(total_lengths), dp_size)) for rank in range(dp_size)]
    micro_batches = [[list(range(len(partitions[rank])))] for rank in range(dp_size)]
    return partitions, micro_batches, [1], [global_batch_size]


def _manager_for_schedule(monkeypatch: pytest.MonkeyPatch, *, global_batch_size: int):
    module = _load_manager_module(monkeypatch)
    monkeypatch.setattr(sys.modules["slime.utils.dp_schedule"], "build_dp_schedule", _round_robin_dp_schedule)
    monkeypatch.setattr(module.ray, "put", lambda value, **_kwargs: value, raising=False)
    monkeypatch.setattr(sys.modules["slime.utils.misc"], "Box", lambda value: value)
    args = types.SimpleNamespace(
        global_batch_size=global_batch_size,
        rollout_data_transport="object-store",
        custom_rollout_data_keys=(),
        reef_rollout_tensor_dtypes={},
    )
    return module.TrainingBatchProcessor(args, {"dp_size": 2})


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
def test_dp_packaging_trains_a_partial_last_step(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _manager_for_schedule(monkeypatch, global_batch_size=2)
    data = {"tokens": [[1], [2], [3], [4], [5]], "rollout_ids": [0, 1, 2, 3, 4]}

    packed = manager._split_train_data_by_dp({**data, "external_step_sizes": [2, 2, 1]})
    assert packed[0]["global_batch_sizes"] == [2, 2, 1]
    assert packed[0]["num_microbatches"] == [1, 1, 1]
    assert [rank["partition"] for rank in packed] == [[0, 2, 4], [1, 3]]


def test_batch_processor_has_no_actor_or_inference_lifecycle(monkeypatch):
    module = _load_manager_module(monkeypatch)
    processor = module.TrainingBatchProcessor(
        types.SimpleNamespace(custom_rollout_data_keys=(), reef_rollout_tensor_dtypes={}), {"dp_size": 1}
    )
    assert processor.train_parallel_config == {"dp_size": 1}
    for name in ("inference_url", "pause_generation_for_update", "dispose", "shutdown"):
        assert not hasattr(processor, name)


def test_shutdown_retires_the_engines_and_the_router_once(monkeypatch):
    module = _load_manager_module(monkeypatch, serving=True)
    engine = types.SimpleNamespace(shutdown=_RemoteMethod("shutdown"))
    group = _ServerGroup([engine])
    worker = _worker(module, group, SGLangSettings("model", 1, 1, 1, pause_mode="retract"))
    stopped = []
    router = types.SimpleNamespace(
        is_alive=lambda: False,
        terminate=lambda: stopped.append("terminate"),
        join=lambda timeout: stopped.append("join"),
    )
    worker.cluster.routers.append(router)
    killed = []
    monkeypatch.setattr(module.ray, "kill", lambda target, **kwargs: killed.append(target))
    worker.shutdown()
    worker.shutdown()
    assert killed == [engine]
    assert group.all_engines == [None]
    assert stopped == ["join"] and worker.cluster.routers == []
