from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from reef.train.slime_backend.reef_adapters.preflight import configure_sglang_runtime
from reef.train.slime_backend.reef_adapters.sglang import plugin as sglang_plugin
from reef.train.slime_backend.reef_adapters.sglang.lora_schema import (
    require_lora_distributed_request_schema,
    require_lora_tensor_request_schema,
)
from reef.train.slime_backend.reef_adapters.sglang.plugin import (
    REEF_SGLANG_PLUGIN_ENV,
    install_colocated_retract_offload,
    install_scheduler_runtime_load_id_tracking,
)
from reef.train.slime_backend.reef_adapters.worker_hooks import reef_rollout_env_vars


def _load_sglang_engine_module(monkeypatch: pytest.MonkeyPatch):
    requests = types.ModuleType("requests")
    requests.get = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    requests.post = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    requests.exceptions = types.SimpleNamespace(  # type: ignore[attr-defined]
        HTTPError=type("HTTPError", (Exception,), {}),
    )
    urllib3 = types.ModuleType("urllib3")
    urllib3.__path__ = []  # type: ignore[attr-defined]
    urllib3_exceptions = types.ModuleType("urllib3.exceptions")
    urllib3_exceptions.NewConnectionError = type("NewConnectionError", (Exception,), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setitem(sys.modules, "urllib3", urllib3)
    monkeypatch.setitem(sys.modules, "urllib3.exceptions", urllib3_exceptions)

    sglang_router = types.ModuleType("sglang_router")
    sglang_router.__version__ = "0.3.0"  # type: ignore[attr-defined]
    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = type("ServerArgs", (), {})
    sglang_utils = types.ModuleType("sglang.srt.utils")
    sglang_utils.kill_process_tree = lambda _pid: None  # type: ignore[attr-defined]
    sglang = types.ModuleType("sglang")
    sglang.__path__ = []  # type: ignore[attr-defined]
    sglang_srt = types.ModuleType("sglang.srt")
    sglang_srt.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", sglang_srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", sglang_utils)
    monkeypatch.setitem(sys.modules, "sglang_router", sglang_router)

    lora = types.ModuleType("reef.train.slime_backend.reef_adapters.megatron.lora")
    lora.LORA_ADAPTER_NAME = "reef_lora"  # type: ignore[attr-defined]
    lora.megatron_lora_enabled = lambda _args: False  # type: ignore[attr-defined]
    lora.sglang_lora_target_modules = lambda _args: []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "reef.train.slime_backend.reef_adapters.megatron.lora", lora)

    external = types.ModuleType("slime.backends.sglang_utils.external")
    external.get_server_info = lambda _url: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.external", external)

    class BaseEngine:
        def __init__(
            self,
            args,
            rank,
            worker_type="regular",
            base_gpu_id=None,
            sglang_overrides=None,
            num_gpus_per_engine=None,
        ):
            self.args = args
            self.rank = rank
            self.worker_type = worker_type
            self.base_gpu_id = base_gpu_id
            self.sglang_overrides = sglang_overrides
            self.num_gpus_per_engine = num_gpus_per_engine

        def _register_to_router(self, server_args_dict):
            self.registered = server_args_dict
            return "registered"

    raw_engine = types.ModuleType("slime.backends.sglang_utils.sglang_engine")
    raw_engine.SGLangEngine = BaseEngine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.sglang_engine", raw_engine)

    path = Path(__file__).parents[2] / "reef" / "train" / "slime_backend" / "reef_adapters" / "sglang" / "engine.py"
    spec = importlib.util.spec_from_file_location("_reef_test_sglang_engine", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_lora_request_schema(
    monkeypatch: pytest.MonkeyPatch,
    distributed_fields: tuple[str, ...],
    tensor_fields: tuple[str, ...] = (
        "lora_name",
        "config_dict",
        "serialized_named_tensors",
        "load_format",
        "expected_checksums",
    ),
) -> None:
    managers = types.ModuleType("sglang.srt.managers")
    managers.__path__ = []  # type: ignore[attr-defined]
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")
    io_struct.LoadLoRAAdapterFromDistributedReqInput = type(  # type: ignore[attr-defined]
        "LoadLoRAAdapterFromDistributedReqInput",
        (),
        {"__struct_fields__": distributed_fields},
    )
    io_struct.LoadLoRAAdapterFromTensorsReqInput = type(  # type: ignore[attr-defined]
        "LoadLoRAAdapterFromTensorsReqInput",
        (),
        {"__struct_fields__": tensor_fields},
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.managers", managers)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)


@pytest.mark.unit
def test_lora_schema_preflight_accepts_distributed_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_lora_request_schema(
        monkeypatch,
        ("lora_name", "config_dict", "names", "dtypes", "shapes", "group_name", "upsert"),
    )

    require_lora_distributed_request_schema()
    require_lora_tensor_request_schema()


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["config_dict", "names", "dtypes", "shapes", "group_name", "upsert"])
def test_lora_schema_preflight_rejects_incomplete_distributed_receiver(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    fields = {"lora_name", "config_dict", "names", "dtypes", "shapes", "group_name", "upsert"} - {missing}
    _install_lora_request_schema(monkeypatch, tuple(sorted(fields)))

    with pytest.raises(RuntimeError, match=missing):
        require_lora_distributed_request_schema()


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing",
    ["config_dict", "serialized_named_tensors", "load_format", "expected_checksums"],
)
def test_lora_schema_preflight_rejects_incomplete_colocated_receiver(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    distributed_fields = ("lora_name", "config_dict", "names", "dtypes", "shapes", "group_name", "upsert")
    tensor_fields = {
        "lora_name",
        "config_dict",
        "serialized_named_tensors",
        "load_format",
        "expected_checksums",
    } - {missing}
    _install_lora_request_schema(monkeypatch, distributed_fields, tuple(sorted(tensor_fields)))

    with pytest.raises(RuntimeError, match=missing):
        require_lora_tensor_request_schema()


@pytest.mark.unit
def test_engine_extension_preserves_version_and_lora_request_schemas(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    calls: list[tuple[str, dict[str, object]]] = []
    engine = object.__new__(module.ReefSGLangEngine)

    def request(route, payload=None, **_kwargs):
        calls.append((route, payload or {}))
        return [True] if route == "set_internal_state" else {"success": True}

    engine.node_rank = 0
    engine._make_request = request

    assert engine.set_runtime_load_id("deployment:2") == {"success": True}
    assert engine.load_lora_adapter_from_tensors(
        "adapter",
        {"r": 8},
        ["rank-0"],
        load_format="flattened_bucket",
        pinned=True,
        expected_checksums={"a": "digest"},
    ) == {"success": True}
    assert engine.load_lora_adapter_from_distributed(
        "adapter",
        {"r": 8},
        ["a", "b"],
        ["torch.float16", "bfloat16"],
        [[2, 4], [4, 2]],
        "reef-lora",
        pinned=True,
    ) == {"success": True}
    assert engine.unload_lora_adapter("adapter") == {"success": True}
    assert calls == [
        (
            "update_weight_version",
            {"new_version": "deployment:2", "abort_all_requests": False},
        ),
        (
            "set_internal_state",
            {"server_args": {"weight_version": "deployment:2"}},
        ),
        (
            "load_lora_adapter_from_tensors",
            {
                "lora_name": "adapter",
                "config_dict": {"r": 8},
                "serialized_named_tensors": ["rank-0"],
                "pinned": True,
                "load_format": "flattened_bucket",
                "expected_checksums": {"a": "digest"},
            },
        ),
        (
            "load_lora_adapter_from_distributed",
            {
                "lora_name": "adapter",
                "config_dict": {"r": 8},
                "names": ["a", "b"],
                "dtypes": ["float16", "bfloat16"],
                "shapes": [[2, 4], [4, 2]],
                "group_name": "reef-lora",
                "pinned": True,
                "upsert": True,
            },
        ),
        ("unload_lora_adapter", {"lora_name": "adapter"}),
    ]


@pytest.mark.unit
def test_get_runtime_load_id_uses_model_info(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    requested: list[tuple[str, float]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"weight_version": "training-incarnation:3"}

    def get(url: str, *, timeout: float):
        requested.append((url, timeout))
        return Response()

    monkeypatch.setattr(module.requests, "get", get)
    engine = object.__new__(module.ReefSGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    engine.args = types.SimpleNamespace(distributed_timeout_minutes=10)

    assert engine.get_runtime_load_id() == "training-incarnation:3"
    assert requested == [("http://127.0.0.1:30000/model_info", 30.0)]


@pytest.mark.unit
def test_engine_preflights_scheduler_runtime_load_id_tracking_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    events = []
    engine = object.__new__(module.ReefSGLangEngine)
    engine.node_rank = 0
    engine.worker_type = "regular"
    engine.get_runtime_load_id = lambda: "engine:3"
    engine._make_request = lambda endpoint, payload, **_kwargs: events.append((endpoint, payload)) or [True]

    assert engine._register_to_router({"model_path": "model"}) == "registered"
    assert events == [("set_internal_state", {"server_args": {"weight_version": "engine:3"}})]
    assert engine.registered == {"model_path": "model"}


@pytest.mark.unit
def test_pause_generation_forwards_in_place_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    calls = []
    engine = object.__new__(module.ReefSGLangEngine)
    engine._make_request = lambda endpoint, payload=None: calls.append((endpoint, payload))

    engine.pause_generation("in_place")
    engine.continue_generation()

    assert calls == [("pause_generation", {"mode": "in_place"}), ("continue_generation", None)]


@pytest.mark.unit
def test_disk_weight_update_does_not_implicitly_flush_kv_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    requested = []
    engine = object.__new__(module.ReefSGLangEngine)
    engine._make_request = lambda endpoint, payload: requested.append((endpoint, payload)) or {"success": True}

    engine.update_weights_from_disk("/weights/v2", runtime_load_id="engine:2")

    assert requested == [
        (
            "update_weights_from_disk",
            {"model_path": "/weights/v2", "flush_cache": False, "weight_version": "engine:2"},
        )
    ]


@pytest.mark.unit
def test_install_sglang_extensions_selects_reef_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    rollout = types.ModuleType("slime.ray.rollout")
    rollout.SGLangEngine = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "slime.ray.rollout", rollout)
    monkeypatch.setattr(importlib.import_module("slime.ray"), "rollout", rollout, raising=False)
    ray_utils = importlib.import_module("slime.ray.utils")
    monkeypatch.setenv("SLIME_HOST_IP", "127.0.0.1")
    for name in ("SLIME_HOST_IP", "NO_PROXY", "no_proxy"):
        monkeypatch.delitem(ray_utils.RAY_DEFAULT_ENV_VARS, name, raising=False)

    module.install_sglang_extensions()

    assert rollout.SGLangEngine is module.ReefSGLangEngine
    assert ray_utils.RAY_DEFAULT_ENV_VARS["SLIME_HOST_IP"] == "127.0.0.1"
    assert "127.0.0.1" in ray_utils.RAY_DEFAULT_ENV_VARS["NO_PROXY"].split(",")


@pytest.mark.unit
def test_reef_engine_carries_lora_server_args_across_ray_process(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    monkeypatch.setattr(module, "megatron_lora_enabled", lambda _args: True)
    monkeypatch.setattr(module, "sglang_lora_target_modules", lambda _args: ["q_proj"])
    checked: list[bool] = []
    monkeypatch.setattr(module, "require_lora_tensor_request_schema", lambda: checked.append(True))
    monkeypatch.setattr(module, "require_lora_distributed_request_schema", lambda: checked.append(True))

    engine = module.ReefSGLangEngine(
        types.SimpleNamespace(megatron_lora_rank=8),
        0,
        sglang_overrides={"mem_fraction_static": 0.5},
    )

    assert checked == [True, True]
    assert engine.sglang_overrides == {
        "mem_fraction_static": 0.5,
        "enable_lora": True,
        "max_loras_per_batch": 1,
        "max_loaded_loras": 1,
        "max_lora_rank": 8,
        "lora_target_modules": ["q_proj"],
        "enable_weights_cpu_backup": True,
        "tokenizer_worker_num": 1,
    }


@pytest.mark.unit
def test_reef_engine_sizes_lora_slots_from_max_loaded_loras(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    monkeypatch.setattr(module, "megatron_lora_enabled", lambda _args: True)
    monkeypatch.setattr(module, "require_lora_tensor_request_schema", lambda: None)
    monkeypatch.setattr(module, "require_lora_distributed_request_schema", lambda: None)
    monkeypatch.setattr(module, "sglang_lora_target_modules", lambda _args: ["q_proj"])

    engine = module.ReefSGLangEngine(types.SimpleNamespace(megatron_lora_rank=8, max_loaded_loras=4), 0)
    assert engine.sglang_overrides["max_loaded_loras"] == 4
    assert engine.sglang_overrides["max_loras_per_batch"] == 4

    with pytest.raises(ValueError, match="max-loaded-loras"):
        module.ReefSGLangEngine(types.SimpleNamespace(megatron_lora_rank=8, max_loaded_loras=0), 0)


@pytest.mark.unit
def test_reef_engine_rejects_conflicting_lora_override(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    monkeypatch.setattr(module, "megatron_lora_enabled", lambda _args: True)
    monkeypatch.setattr(module, "require_lora_tensor_request_schema", lambda: None)
    monkeypatch.setattr(module, "require_lora_distributed_request_schema", lambda: None)

    with pytest.raises(ValueError, match="conflict"):
        module.ReefSGLangEngine(
            types.SimpleNamespace(megatron_lora_rank=8),
            0,
            sglang_overrides={"enable-lora": False},
        )


@pytest.mark.unit
def test_reef_config_selects_pause_mode_and_disables_shared_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_PLUGINS", "telemetry")
    disjoint = types.SimpleNamespace(colocate=False, megatron_lora_rank=0, sglang_config=None)
    configure_sglang_runtime(disjoint)

    assert disjoint.sglang_disable_radix_cache is True
    assert disjoint.sglang_incremental_streaming_output is True
    assert disjoint.weight_update_pause_mode == "in_place"
    assert os.environ[REEF_SGLANG_PLUGIN_ENV] == "1"
    assert os.environ["SGLANG_PLUGINS"] == "telemetry,reef"

    colocated = types.SimpleNamespace(
        colocate=True,
        megatron_lora_rank=0,
        prefill_num_servers=0,
        sglang_config=None,
    )
    configure_sglang_runtime(colocated)
    assert colocated.weight_update_pause_mode == "retract"
    assert colocated.sglang_incremental_streaming_output is True


@pytest.mark.unit
def test_reef_config_enables_sglang_plugin_without_preexisting_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SGLANG_PLUGINS", raising=False)
    args = types.SimpleNamespace(colocate=False, megatron_lora_rank=0, sglang_config=None)

    configure_sglang_runtime(args)

    assert os.environ["SGLANG_PLUGINS"] == "reef"


@pytest.mark.unit
def test_sglang_plugin_environment_crosses_ray_actor_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {
        "SGLANG_PLUGINS": "telemetry,reef",
        REEF_SGLANG_PLUGIN_ENV: "1",
        "UNRELATED": "ignored",
    }

    assert reef_rollout_env_vars(env) == {
        "SGLANG_PLUGINS": "telemetry,reef",
        REEF_SGLANG_PLUGIN_ENV: "1",
    }


@pytest.mark.unit
def test_native_sglang_plugin_is_explicitly_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = []
    monkeypatch.setattr(
        sglang_plugin,
        "install_scheduler_runtime_load_id_tracking",
        lambda: installed.append("runtime_load_ids"),
    )
    monkeypatch.setattr(
        sglang_plugin,
        "install_colocated_retract_offload",
        lambda: installed.append("colocated"),
    )
    monkeypatch.delenv(REEF_SGLANG_PLUGIN_ENV, raising=False)

    sglang_plugin.install_sglang_plugin()
    assert installed == []

    monkeypatch.setenv(REEF_SGLANG_PLUGIN_ENV, "1")
    sglang_plugin.install_sglang_plugin()
    assert installed == ["runtime_load_ids", "colocated"]


@pytest.mark.unit
def test_reef_config_rejects_yaml_that_reenables_shared_prefixes(tmp_path: Path) -> None:
    config = tmp_path / "sglang.yaml"
    config.write_text(
        """\
sglang:
  - name: actor
    server_groups:
      - worker_type: regular
        num_gpus: 1
        overrides:
          disable-radix-cache: false
""",
        encoding="utf-8",
    )
    args = types.SimpleNamespace(colocate=False, megatron_lora_rank=0, sglang_config=str(config))

    with pytest.raises(ValueError, match="disable_radix_cache=true"):
        configure_sglang_runtime(args)


@pytest.mark.unit
def test_colocated_reef_config_rejects_pd_disaggregation(tmp_path: Path) -> None:
    config = tmp_path / "sglang.yaml"
    config.write_text(
        """\
sglang:
  - name: actor
    server_groups:
      - worker_type: prefill
        num_gpus: 1
      - worker_type: decode
        num_gpus: 1
""",
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        colocate=True,
        megatron_lora_rank=0,
        prefill_num_servers=0,
        sglang_config=str(config),
    )

    with pytest.raises(ValueError, match="regular SGLang engine"):
        configure_sglang_runtime(args)


@pytest.mark.unit
def test_scheduler_stamps_tokens_before_cross_process_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    current_args = types.SimpleNamespace(weight_version="engine:6")
    server_args = types.ModuleType("sglang.srt.server_args")
    server_args.get_global_server_args = lambda: current_args  # type: ignore[attr-defined]

    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")

    class Scheduler:
        def run_batch(self, batch):
            return batch.result

        def process_batch_result(self, batch, result):
            assert result is batch.result
            Accumulator().accept(req=batch.req)

        def set_internal_state(self, request):
            return types.SimpleNamespace(updated=True, server_args=dict(request.server_args))

    scheduler_module.Scheduler = Scheduler  # type: ignore[attr-defined]
    components = types.ModuleType("sglang.srt.managers.scheduler_components")
    components.__path__ = []  # type: ignore[attr-defined]
    output_streamer = types.ModuleType("sglang.srt.managers.scheduler_components.output_streamer")

    class Accumulator:
        def accept(self, *, req):
            req.accepted = True

    output_streamer._GenerationStreamAccumulator = Accumulator  # type: ignore[attr-defined]
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class UpdateResult:
        def __init__(self, *, success=True, message="updated", num_paused_requests=0):
            self.success = success
            self.message = message
            self.num_paused_requests = num_paused_requests

    io_struct.UpdateWeightFromDiskReqOutput = UpdateResult  # type: ignore[attr-defined]

    class BeginWeightUpdateReqInput:
        def __init__(self, *, selector="all"):
            self.selector = selector

    class EndWeightUpdateReqInput:
        pass

    io_struct.BeginWeightUpdateReqInput = BeginWeightUpdateReqInput  # type: ignore[attr-defined]
    io_struct.EndWeightUpdateReqInput = EndWeightUpdateReqInput  # type: ignore[attr-defined]
    weight_updater = types.ModuleType("sglang.srt.managers.scheduler_components.weight_updater")

    class SchedulerWeightUpdaterManager:
        """The session semantics SGLang's RL weight path enforces."""

        calls = []
        _weight_update_in_progress = False

        def begin_weight_update(self, recv_req):
            assert not self._weight_update_in_progress, "session already open"
            self._weight_update_in_progress = True
            self.calls.append(("begin", recv_req.selector))
            return UpdateResult()

        def end_weight_update(self, recv_req):
            assert self._weight_update_in_progress, "end without begin"
            self._weight_update_in_progress = False
            self.calls.append(("end", None))
            return UpdateResult()

        def update_weights_from_disk(self, recv_req):
            self.calls.append(("disk", recv_req.weight_version))
            return UpdateResult()

        def update_weights_from_distributed(self, recv_req):
            assert self._weight_update_in_progress, "requires an open begin_weight_update session"
            self.calls.append(("distributed", recv_req.weight_version))
            return UpdateResult()

        def update_weights_from_tensor(self, recv_req):
            assert self._weight_update_in_progress, "requires an open begin_weight_update session"
            self.calls.append(("tensor", recv_req.weight_version))
            return UpdateResult()

    weight_updater.SchedulerWeightUpdaterManager = SchedulerWeightUpdaterManager  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler_components", components)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler_components.output_streamer", output_streamer)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler_components.weight_updater", weight_updater)

    install_scheduler_runtime_load_id_tracking()
    request = types.SimpleNamespace(output_ids_through_stop=[20, 21], customized_info=None, accepted=False)
    Accumulator().accept(req=request)
    current_args.weight_version = "engine:7"
    request.output_ids_through_stop.append(22)
    Accumulator().accept(req=request)
    assert request.customized_info["_reef_token_runtime_load_ids"] == ["engine:6", "engine:6", "engine:7"]

    scheduler = Scheduler()
    queued_request = types.SimpleNamespace(output_ids_through_stop=[40], customized_info=None, accepted=False)
    queued_batch = types.SimpleNamespace(req=queued_request, result=object())
    current_args.weight_version = "engine:9"
    queued_result = scheduler.run_batch(queued_batch)
    current_args.weight_version = "engine:10"
    scheduler.process_batch_result(queued_batch, queued_result)
    assert queued_request.customized_info["_reef_token_runtime_load_ids"] == ["engine:9"]

    result = scheduler.set_internal_state(types.SimpleNamespace(server_args={"weight_version": "engine:8"}))
    assert result.updated is True
    assert current_args.weight_version == "engine:8"

    updater = SchedulerWeightUpdaterManager()
    current_args.weight_version = "engine:7"
    stale = updater.update_weights_from_disk(types.SimpleNamespace(load_format="delta", weight_version="engine:6"))
    cross_incarnation = updater.update_weights_from_disk(
        types.SimpleNamespace(load_format="delta", weight_version="previous:99")
    )
    assert stale.success is False
    assert "refusing stale delta" in stale.message
    assert cross_incarnation.success is False
    assert SchedulerWeightUpdaterManager.calls == []

    current_batch = updater.update_weights_from_disk(
        types.SimpleNamespace(load_format="delta", weight_version="engine:7")
    )
    next_tensor = updater.update_weights_from_tensor(types.SimpleNamespace(weight_version="engine:8"))
    synced = updater.update_weights_from_distributed(types.SimpleNamespace(weight_version="engine:9"))
    assert current_batch.success is True
    assert next_tensor.success is True
    assert synced.success is True
    # The disk load needs no session. The two updates that assert one are
    # wrapped in begin -> update -> end, the sequence Slime's single-call form
    # omits and SGLang refuses without.
    assert SchedulerWeightUpdaterManager.calls == [
        ("disk", "engine:7"),
        ("begin", "all"),
        ("tensor", "engine:8"),
        ("end", None),
        ("begin", "all"),
        ("distributed", "engine:9"),
        ("end", None),
    ]
    assert updater._weight_update_in_progress is False
    assert current_args.weight_version == "engine:9"


@pytest.mark.unit
def test_weight_update_session_defers_to_the_caller_and_to_older_sglang(monkeypatch: pytest.MonkeyPatch) -> None:
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class BeginWeightUpdateReqInput:
        def __init__(self, *, selector="all"):
            self.selector = selector

    io_struct.BeginWeightUpdateReqInput = BeginWeightUpdateReqInput  # type: ignore[attr-defined]
    io_struct.EndWeightUpdateReqInput = type("EndWeightUpdateReqInput", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)
    calls: list[str] = []

    class SessionUpdater:
        _weight_update_in_progress = False

        def begin_weight_update(self, recv_req):
            calls.append("begin")
            self._weight_update_in_progress = True

        def end_weight_update(self, recv_req):
            calls.append("end")
            self._weight_update_in_progress = False

    # A caller that opened its own session keeps it: opening a second one is
    # what SGLang asserts against, and closing someone else's would end it early.
    owned = SessionUpdater()
    owned._weight_update_in_progress = True
    with sglang_plugin._weight_update_session(owned):
        calls.append("update")
    assert calls == ["update"]
    assert owned._weight_update_in_progress is True

    # An SGLang predating the two-phase protocol has neither method.
    calls.clear()
    with sglang_plugin._weight_update_session(object()):
        calls.append("update")
    assert calls == ["update"]

    # A failing update still closes the session it opened.
    calls.clear()
    updater = SessionUpdater()
    with pytest.raises(RuntimeError, match="broadcast failed"), sglang_plugin._weight_update_session(updater):
        raise RuntimeError("broadcast failed")
    assert calls == ["begin", "end"]
    assert updater._weight_update_in_progress is False


@pytest.mark.unit
def test_colocated_retract_queue_is_idle_only_for_paused_gpu_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")

    class Scheduler:
        def pause_generation(self, recv_req):
            if getattr(self, "pause_error", False):
                raise RuntimeError("pause failed")

        def continue_generation(self, recv_req):
            return None

        def is_fully_idle(self, for_health_check=False):
            del for_health_check
            return not self.waiting_queue and not self.gpu_busy

    scheduler_module.Scheduler = Scheduler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    install_colocated_retract_offload()

    scheduler = Scheduler()
    suspended = [object(), object()]
    scheduler.waiting_queue = suspended
    scheduler.gpu_busy = False

    scheduler.pause_generation(types.SimpleNamespace(mode="in_place"))
    assert scheduler.is_fully_idle() is False

    scheduler.pause_generation(types.SimpleNamespace(mode="retract"))
    assert scheduler.is_fully_idle() is True
    assert scheduler.waiting_queue is suspended
    assert scheduler.is_fully_idle(for_health_check=True) is False

    scheduler.continue_generation(types.SimpleNamespace())
    assert scheduler.is_fully_idle() is False

    scheduler.pause_error = True
    with pytest.raises(RuntimeError, match="pause failed"):
        scheduler.pause_generation(types.SimpleNamespace(mode="retract"))
    assert scheduler.is_fully_idle() is False

    scheduler.pause_error = False
    scheduler.pause_generation(types.SimpleNamespace(mode="retract"))
    scheduler.gpu_busy = True
    assert scheduler.is_fully_idle() is False
    assert scheduler.waiting_queue is suspended


@pytest.mark.unit
def test_colocated_retract_uses_native_ignore_waiting_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")

    class Scheduler:
        def __init__(self):
            self.calls = []

        def pause_generation(self, recv_req):
            return None

        def continue_generation(self, recv_req):
            return None

        def is_fully_idle(self, for_health_check=False, ignore_waiting=False):
            self.calls.append((for_health_check, ignore_waiting))
            return ignore_waiting and not for_health_check

    scheduler_module.Scheduler = Scheduler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    install_colocated_retract_offload()

    scheduler = Scheduler()
    scheduler.pause_generation(types.SimpleNamespace(mode="retract"))

    assert scheduler.is_fully_idle(ignore_waiting=False) is True
    assert scheduler.calls[-1] == (False, True)
    assert scheduler.is_fully_idle(for_health_check=True) is False
    assert scheduler.calls[-1] == (True, False)
