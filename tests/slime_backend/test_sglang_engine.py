from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from concurrent.futures import Future
from pathlib import Path

import pytest

from reef.inference.sglang import config as sglang_config
from reef.inference.sglang import deployment as sglang_deployment
from reef.inference.sglang import plugin as sglang_plugin
from reef.inference.sglang.config import SGLangConfig
from reef.inference.sglang.deployment import create_inference
from reef.inference.sglang.launch import engine_environment
from reef.inference.sglang.lora_schema import (
    adapter_scoped_prefix_cache_supported,
    require_lora_distributed_request_schema,
    require_lora_tensor_request_schema,
)
from reef.inference.sglang.plugin import (
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

    path = Path(__file__).parents[2] / "reef" / "inference" / "sglang" / "engine.py"
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
    engine.config = SGLangConfig("model", 1, 1, 1)

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

    engine.router_ip = None
    assert engine._register_to_router({"model_path": "model"}) is None
    assert events == [("set_internal_state", {"server_args": {"weight_version": "engine:3"}})]


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


def _training_inference_config(**options):
    return SGLangConfig(**_training_inference_values(**options))


def _training_inference_values(**options):
    from reef.train.slime_backend.inference import inference_config

    return inference_config(
        types.SimpleNamespace(
            **{
                "hf_checkpoint": "model",
                "seed": 1,
                "offload_rollout": False,
                "fp16": False,
                "use_rollout_routing_replay": False,
                "megatron_lora_rank": 0,
                "rollout_num_gpus": 1,
                "rollout_num_gpus_per_engine": 1,
                "num_gpus_per_node": 1,
                "actor_num_nodes": 1,
                "actor_num_gpus_per_node": 1,
                "colocate": False,
                "disjoint_prefix_sharing": False,
                **options,
            },
        )
    )


def test_training_inference_forces_the_flags_reef_serving_relies_on():
    # Slime's parser defaults --sglang-disable-radix-cache to off; the config
    # validation rejects that, so the translation must set the flags itself.
    config = _training_inference_config(sglang_disable_radix_cache=False)
    assert config.options["disable_radix_cache"] is True
    assert config.options["incremental_streaming_output"] is True
    opted_out = _training_inference_config(colocate=True, offload_rollout=True, sglang_disable_radix_cache=True)
    assert opted_out.options["disable_radix_cache"] is True, "a launch may always opt out of sharing"


def test_inference_engine_does_not_inherit_or_patch_slime(monkeypatch):
    module = _load_sglang_engine_module(monkeypatch)
    assert module.ReefSGLangEngine.__bases__ == (object,)
    assert not hasattr(module, "install_sglang_extensions")


def test_reef_engine_carries_lora_server_args_across_ray_process(monkeypatch):
    module = _load_sglang_engine_module(monkeypatch)
    import pickle

    lora = sys.modules["reef.train.slime_backend.reef_adapters.megatron.lora"]
    monkeypatch.setattr(lora, "sglang_lora_target_modules", lambda args: ["q_proj"])
    checked = []
    monkeypatch.setattr(module, "require_lora_tensor_request_schema", lambda: checked.append(True))
    monkeypatch.setattr(module, "require_lora_distributed_request_schema", lambda: checked.append(True))
    config = pickle.loads(pickle.dumps(_training_inference_config(megatron_lora_rank=8)))
    engine = module.ReefSGLangEngine(config, 0, sglang_overrides={"mem_fraction_static": 0.5})
    assert checked == [True, True]
    assert engine.config.options["enable_lora"] is True
    assert engine.config.options["lora_target_modules"] == ["q_proj"]
    assert engine.config.options["max_lora_rank"] == 8
    assert engine.config.options["max_loaded_loras"] == 1
    assert engine.config.options["enable_weights_cpu_backup"] is True
    assert engine.config.options["tokenizer_worker_num"] == 1
    assert engine.sglang_overrides == {"mem_fraction_static": 0.5}


def test_reef_engine_sizes_lora_slots_from_max_loaded_loras(monkeypatch):
    _load_sglang_engine_module(monkeypatch)
    config = _training_inference_config(megatron_lora_rank=8, max_loaded_loras=4)
    assert config.options["max_loaded_loras"] == 4
    assert config.options["max_loras_per_batch"] == 4
    with pytest.raises(ValueError, match="max-loaded-loras"):
        _training_inference_config(megatron_lora_rank=8, max_loaded_loras=0)


def test_reef_engine_rejects_conflicting_lora_override(monkeypatch):
    module = _load_sglang_engine_module(monkeypatch)
    monkeypatch.setattr(module, "require_lora_tensor_request_schema", lambda: None)
    monkeypatch.setattr(module, "require_lora_distributed_request_schema", lambda: None)
    with pytest.raises(ValueError, match="conflict"):
        module.ReefSGLangEngine(
            SGLangConfig("model", 1, 1, 1, options={"enable_lora": True}), 0, sglang_overrides={"enable-lora": False}
        )


@pytest.mark.unit
def test_reef_config_selects_pause_mode_and_scopes_shared_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SGLANG_PLUGINS", "telemetry")
    disjoint = create_inference(_training_inference_values()).config
    assert disjoint.options["disable_radix_cache"] is True
    assert disjoint.options["incremental_streaming_output"] is True
    assert disjoint.pause_mode == "in_place"
    assert engine_environment(disjoint)[REEF_SGLANG_PLUGIN_ENV] == "1"
    assert engine_environment(disjoint)["SGLANG_PLUGINS"] == "telemetry,reef"
    assert os.environ["SGLANG_PLUGINS"] == "telemetry"

    colocated = create_inference(_training_inference_values(colocate=True, offload_rollout=True)).config
    assert colocated.pause_mode == "retract"
    assert colocated.options["incremental_streaming_output"] is True
    # A colocated step releases the KV cache before every publication, so a
    # shared entry cannot outlive the weights that built it.
    assert colocated.options["disable_radix_cache"] is False


@pytest.mark.unit
def test_disjoint_shares_prefixes_only_by_opting_into_retraction() -> None:
    """Sharing entries and retracting in-flight KV are one decision."""
    default = _training_inference_config()
    assert default.options["disable_radix_cache"] is True
    assert default.pause_mode == "in_place"

    sharing = _training_inference_config(disjoint_prefix_sharing=True)
    assert sharing.options["disable_radix_cache"] is False
    assert sharing.pause_mode == "retract", "the cache can only be cleared once no request holds KV"

    with pytest.raises(ValueError, match="disable_radix_cache=true"):
        SGLangConfig("model", 1, 1, 1, options={"disable_radix_cache": False})


@pytest.mark.unit
@pytest.mark.parametrize("adapter_scoped", [True, False])
def test_colocated_lora_shares_prefixes_only_when_sglang_keys_them_by_adapter(
    monkeypatch: pytest.MonkeyPatch, adapter_scoped: bool
) -> None:
    """One engine holds several scenarios' adapters; the engine must keep them apart."""
    _load_sglang_engine_module(monkeypatch)
    monkeypatch.setattr(sglang_config, "adapter_scoped_prefix_cache_supported", lambda: adapter_scoped)

    config = _training_inference_config(colocate=True, offload_rollout=True, megatron_lora_rank=8)

    assert config.options["enable_lora"] is True
    assert (
        config.options["disable_radix_cache"] is not adapter_scoped
    ), "an engine that cannot isolate adapters shares nothing"


@pytest.mark.unit
@pytest.mark.parametrize("broken", [None, "request", "radix", "request_id", "unsupported", "unforeseen"])
def test_adapter_prefix_check_exercises_actual_key_isolation(
    monkeypatch: pytest.MonkeyPatch, broken: str | None
) -> None:
    requests = types.ModuleType("sglang.srt.managers.schedule_batch")
    radix = types.ModuleType("sglang.srt.mem_cache.radix_cache")
    sampling = types.ModuleType("sglang.srt.sampling.sampling_params")

    class Req:
        def __init__(self, *, rid, origin_input_text, origin_input_ids, sampling_params, lora_id):
            if broken == "unsupported":
                raise TypeError("unsupported request API")
            if broken == "unforeseen":
                raise ValueError("a later pin rejects this request shape")
            self.extra_key = None if broken == "request" else lora_id
            if broken == "request_id":
                self.extra_key = rid

    class RadixKey:
        def __init__(self, *, token_ids, extra_key):
            self.token_ids = token_ids
            self.extra_key = extra_key

        def child_key(self):
            if broken == "radix":
                return tuple(self.token_ids)
            return self.extra_key, tuple(self.token_ids)

    requests.Req = Req  # type: ignore[attr-defined]
    radix.RadixKey = RadixKey  # type: ignore[attr-defined]
    sampling.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    for module in (requests, radix, sampling):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter_scoped_prefix_cache_supported.cache_clear()

    try:
        assert adapter_scoped_prefix_cache_supported() is (broken is None)
    finally:
        adapter_scoped_prefix_cache_supported.cache_clear()


@pytest.mark.unit
def test_reef_config_enables_sglang_plugin_without_preexisting_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SGLANG_PLUGINS", raising=False)
    config = create_inference(_training_inference_values()).config
    assert engine_environment(config)["SGLANG_PLUGINS"] == "reef"
    assert "SGLANG_PLUGINS" not in os.environ


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
def test_reef_config_rejects_disjoint_yaml_that_reenables_shared_prefixes(tmp_path: Path) -> None:
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
    with pytest.raises(ValueError, match="disable_radix_cache=true"):
        create_inference(_training_inference_values(sglang_config=str(config)))

    colocated = create_inference(
        _training_inference_values(colocate=True, offload_rollout=True, sglang_config=str(config))
    ).config
    assert colocated.models[0].groups[0].options["disable_radix_cache"] is False, "a colocated group may share"


@pytest.mark.unit
@pytest.mark.parametrize("via_yaml", [True, False])
def test_disjoint_prefix_sharing_rejects_pd_disaggregation(tmp_path: Path, via_yaml: bool) -> None:
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
    topology = {"sglang_config": str(config)} if via_yaml else {"prefill_num_servers": 1}
    with pytest.raises(ValueError, match="regular SGLang engines"):
        create_inference(_training_inference_values(disjoint_prefix_sharing=True, rollout_num_gpus=2, **topology))
    # Without the opt-in, a disjoint PD deployment keeps in-flight KV and remains valid.
    create_inference(_training_inference_values(rollout_num_gpus=2, **topology))


@pytest.mark.unit
@pytest.mark.parametrize("colocate", [True, False])
@pytest.mark.parametrize("adapter_scoped", [True, False])
@pytest.mark.parametrize("disabled", [True, False])
def test_lora_yaml_cannot_override_required_cache_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, colocate: bool, adapter_scoped: bool, disabled: bool
) -> None:
    _load_sglang_engine_module(monkeypatch)
    monkeypatch.setattr(sglang_config, "adapter_scoped_prefix_cache_supported", lambda: adapter_scoped)
    config = tmp_path / "sglang.yaml"
    config.write_text(
        "sglang:\n  - name: actor\n    server_groups:\n      - worker_type: regular\n"
        f"        num_gpus: 1\n        overrides:\n          disable-radix-cache: {str(disabled).lower()}\n",
        encoding="utf-8",
    )
    values = _training_inference_values(
        colocate=colocate,
        offload_rollout=colocate,
        disjoint_prefix_sharing=not colocate,
        megatron_lora_rank=8,
        sglang_config=str(config),
    )
    monkeypatch.setattr(sglang_deployment, "require_lora_tensor_request_schema", lambda: None)
    monkeypatch.setattr(sglang_deployment, "require_lora_distributed_request_schema", lambda: None)
    if not adapter_scoped and not disabled:
        with pytest.raises(ValueError, match="disable_radix_cache=true"):
            create_inference(values)
    else:
        create_inference(values)


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
    with pytest.raises(ValueError, match="regular engines"):
        create_inference(
            _training_inference_values(
                colocate=True, offload_rollout=True, rollout_num_gpus=2, sglang_config=str(config)
            )
        )


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

        def is_fully_idle(self, for_health_check=False, ignore_waiting=False):
            del for_health_check
            return (ignore_waiting or not self.waiting_queue) and not self.gpu_busy

    scheduler_module.Scheduler = Scheduler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    install_colocated_retract_offload()

    scheduler = Scheduler()
    suspended = [object(), object()]
    scheduler.waiting_queue = suspended
    scheduler.grammar_manager = types.SimpleNamespace(grammar_queue=[])
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
def test_colocated_retract_uses_pinned_ignore_waiting_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")

    class Scheduler:
        def __init__(self):
            self.calls = []
            self.waiting_queue = []
            self.grammar_manager = types.SimpleNamespace(grammar_queue=[])

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
    assert scheduler.is_fully_idle(True, False) is False
    assert scheduler.calls[-1] == (True, False)


@pytest.mark.unit
def test_retract_flush_preserves_pending_grammar_and_gpu_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler_module = types.ModuleType("sglang.srt.managers.scheduler")
    future: Future[str] = Future()
    request = types.SimpleNamespace(grammar=future)
    grammar_queue = [request]

    class Scheduler:
        def __init__(self):
            self.waiting_queue = []
            self.grammar_manager = types.SimpleNamespace(grammar_queue=grammar_queue)
            self.gpu_busy = False
            self.idle_error = False
            self.cache = {"old-prefix": "old-kv"}

        def pause_generation(self, recv_req):
            return None

        def continue_generation(self, recv_req):
            return None

        def is_fully_idle(self, for_health_check=False, ignore_waiting=False):
            if self.idle_error:
                raise RuntimeError("idle check failed")
            return (
                not self.gpu_busy
                and (ignore_waiting or not self.waiting_queue)
                and (for_health_check or not self.grammar_manager.grammar_queue)
            )

        def flush_cache(self):
            if not self.is_fully_idle():
                return False
            self.cache.clear()
            return True

    scheduler_module.Scheduler = Scheduler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.scheduler", scheduler_module)
    install_colocated_retract_offload()
    scheduler = Scheduler()
    scheduler.pause_generation(types.SimpleNamespace(mode="in_place"))
    assert scheduler.flush_cache() is False
    scheduler.pause_generation(types.SimpleNamespace(mode="retract"))
    scheduler.gpu_busy = True
    assert scheduler.flush_cache() is False
    assert scheduler.cache
    scheduler.gpu_busy = False
    assert scheduler.flush_cache() is True
    assert not scheduler.cache
    assert scheduler.grammar_manager.grammar_queue is grammar_queue
    assert not future.done()

    scheduler.idle_error = True
    with pytest.raises(RuntimeError, match="idle check failed"):
        scheduler.is_fully_idle()
    assert scheduler.grammar_manager.grammar_queue is grammar_queue
    scheduler.idle_error = False
    scheduler.continue_generation(types.SimpleNamespace())
    assert scheduler.is_fully_idle() is False
    future.set_result("compiled grammar")
    assert scheduler.grammar_manager.grammar_queue.pop().grammar.result() == "compiled grammar"


@pytest.mark.unit
def test_memory_transitions_pair_cold_startup_with_keep_base_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_sglang_engine_module(monkeypatch)
    calls: list[tuple[str, object]] = []
    engine = object.__new__(module.ReefSGLangEngine)
    engine.node_rank = 0
    engine._make_request = lambda route, payload=None, **_kw: calls.append((route, payload))
    engine.flush_cache = lambda: calls.append(("flush_cache", None))

    engine.release_memory_occupation()
    engine.release_memory_occupation(["kv_cache", "cuda_graph"])
    engine.resume_memory_occupation(["weights"])
    engine.resume_memory_occupation(["weights"])
    engine.resume_memory_occupation()
    engine.release_memory_occupation(["kv_cache", "cuda_graph"])
    engine.resume_memory_occupation()

    assert calls == [
        ("flush_cache", None),
        ("release_memory_occupation", {"tags": ["weights", "kv_cache", "cuda_graph"]}),
        ("resume_memory_occupation", {"tags": ["weights"]}),
        ("resume_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]}),
        ("flush_cache", None),
        ("release_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]}),
        ("resume_memory_occupation", {"tags": ["kv_cache", "cuda_graph"]}),
    ]


@pytest.mark.parametrize("failure", ["release_memory_occupation", "resume_memory_occupation"])
def test_uncertain_memory_transition_requires_engine_replacement(monkeypatch, failure):
    module = _load_sglang_engine_module(monkeypatch)
    engine = object.__new__(module.ReefSGLangEngine)
    calls = []

    def request(route, payload):
        calls.append(route)
        return {"success": route != failure}

    engine._make_request = request
    engine.flush_cache = lambda: None
    with pytest.raises(RuntimeError, match=failure):
        engine.release_memory_occupation()
        engine.resume_memory_occupation()
    before = list(calls)
    with pytest.raises(RuntimeError, match="uncertain"):
        engine.resume_memory_occupation()
    with pytest.raises(RuntimeError, match="uncertain"):
        engine.release_memory_occupation()
    assert calls == before
