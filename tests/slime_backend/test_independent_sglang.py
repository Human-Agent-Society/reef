"""The inference backend launches and exposes control without importing Slime."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest

from reef.runtime.sglang.config import SGLangConfig, SGLangGroupConfig
from reef.runtime.sglang.engine import ReefSGLangEngine
from reef.runtime.sglang.launch import SGLangEngineGroup, SGLangModel


def test_sglang_imports_with_training_packages_blocked():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules.update(dict.fromkeys(('slime', 'megatron', 'reef.train.slime_backend')))
from reef.runtime.sglang import chat, config, control, engine, executor, health, launch, plugin, service
assert engine.ReefSGLangEngine.__bases__ == (object,)
assert not any(name.startswith('slime.') or name.startswith('reef.train.slime_backend.') for name in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def native_engine(monkeypatch):
    from reef.runtime.sglang import engine as module

    fields = [
        "model_path",
        "host",
        "port",
        "nccl_port",
        "node_rank",
        "nnodes",
        "dist_init_addr",
        "tp_size",
        "pp_size",
        "base_gpu_id",
        "gpu_id_step",
        "enable_memory_saver",
        "cuda_graph_backend_prefill",
        "enable_lora",
        "max_lora_rank",
        "disable_radix_cache",
        "incremental_streaming_output",
    ]
    monkeypatch.setitem(
        sys.modules, "sglang.srt.server_args", SimpleNamespace(ServerArgs=make_dataclass("ServerArgs", fields))
    )
    launched, events = [], []
    process = SimpleNamespace()
    monkeypatch.setattr(module, "launch_engine", lambda options: launched.append(options) or process)
    monkeypatch.setattr(module, "wait_ready", lambda *args: events.append("ready"))
    monkeypatch.setattr(ReefSGLangEngine, "get_runtime_load_id", lambda self: "engine:1")
    monkeypatch.setattr(
        ReefSGLangEngine, "_sync_scheduler_runtime_load_id", lambda self, version: events.append(version)
    )
    monkeypatch.setattr(
        module.requests,
        "post",
        lambda *args, **kwargs: events.append("register") or SimpleNamespace(raise_for_status=lambda: None),
    )
    return module, launched, events


def test_native_launch_uses_supplied_placement_and_versions_before_registration(native_engine):
    _, launched, events = native_engine
    config = SGLangConfig("model", 8, 8, 4, options={"pp_size": 2, "enable_memory_saver": True})
    worker = ReefSGLangEngine(config, 0, base_gpu_id=4)
    worker.init("10.0.0.1:18000", 16000, 17000, "10.0.0.1", router_ip="10.0.0.2", router_port=3000)
    assert launched[0]["tp_size"] == 4
    assert launched[0]["pp_size"] == 2
    assert launched[0]["nnodes"] == 2
    assert launched[0]["node_rank"] == 0
    assert launched[0]["base_gpu_id"] == 4
    assert launched[0]["cuda_graph_backend_prefill"] == "disabled"
    assert events == ["ready", "engine:1", "register"]


def test_nonzero_node_does_not_register_or_read_http(native_engine):
    _, launched, events = native_engine
    worker = ReefSGLangEngine(SGLangConfig("model", 8, 8, 4), 1)
    worker.init("10.0.0.1:18000", 16000, 17000, "10.0.0.2", router_ip="10.0.0.3", router_port=3000)
    assert launched[0]["node_rank"] == 1
    assert events == []


def test_external_engine_is_validated_and_never_launched_or_terminated(native_engine, monkeypatch):
    module, launched, _ = native_engine
    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"disable_radix_cache": True, "incremental_streaming_output": True},
        ),
    )
    worker = ReefSGLangEngine(
        SGLangConfig("model", 1, 1, 1, options={"disable_radix_cache": True}), 0, external_url="http://provider:8000"
    )
    worker.init("provider:8000", 8000, None, "provider")
    worker.shutdown()
    assert launched == []


def test_external_engine_refuses_incompatible_capture_settings(native_engine, monkeypatch):
    module, launched, events = native_engine
    monkeypatch.setattr(
        module.requests,
        "get",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: {"disable_radix_cache": False}
        ),
    )
    worker = ReefSGLangEngine(
        SGLangConfig("model", 1, 1, 1, options={"disable_radix_cache": True}), 0, external_url="http://provider:8000"
    )
    with pytest.raises(ValueError, match="disable_radix_cache"):
        worker.init("provider:8000", 8000, None, "provider")
    assert not launched and not events


def test_group_preserves_shared_gpu_placement_and_multinode_rendezvous(monkeypatch):
    from reef.runtime.sglang import launch

    calls, initialized = [], []

    class Remote:
        def __init__(self, fn):
            self.remote = fn

    class Actor:
        def __init__(self, rank):
            self._get_current_node_ip_and_free_port = Remote(
                lambda start_port=15000, consecutive=1: (f"node-{rank}", start_port)
            )
            self.init = Remote(lambda **kwargs: initialized.append(kwargs))

    class ActorClass:
        def options(self, **kwargs):
            calls.append(kwargs)
            return self

        def remote(self, config, **kwargs):
            return Actor(kwargs["rank"])

    monkeypatch.setattr(launch.ray, "remote", lambda cls: ActorClass())
    monkeypatch.setattr(launch.ray, "get", lambda value: value)
    config = SGLangConfig("model", 8, 8, 4, offload=True, shared_gpus=8)
    group = SGLangEngineGroup(
        config,
        SGLangGroupConfig("regular", 8, 8),
        (object(), list(range(8)), [0, 1, 2, 3, 0, 1, 2, 3]),
        0,
        ("router", 3000),
    )
    group.start_engines({})
    assert len(calls) == 2
    assert [c["scheduling_strategy"].placement_group_bundle_index for c in calls] == [0, 4]
    assert initialized[0]["dist_init_addr"] == initialized[1]["dist_init_addr"] == "node-0:15003"
    assert group.needs_offload
    assert len(group.engines) == 1
    model = SGLangModel([group])
    assert model.engine_gpu_counts == [8]
    assert model.engine_gpu_offsets == [0]
    assert model.engine_parallel_configs == [{"tp_size": 8, "pp_size": 1, "ep_size": 1, "moe_dp_size": 1}]


def test_group_rejects_noncontiguous_gpu_reservations_before_launch(monkeypatch):
    config = SGLangConfig("model", 2, 2, 4)
    group = SGLangEngineGroup(
        config, SGLangGroupConfig("regular", 2, 2), (object(), [0, 1], [0, 2]), 0, ("router", 3000)
    )
    with pytest.raises(ValueError, match="contiguous"):
        group.start_engines({})


def test_native_control_preserves_weight_checker_and_checkpoint_pull_wire(monkeypatch):
    from reef.runtime.sglang import engine as module

    calls = []

    def post(url, *, json, timeout):
        calls.append((url, json))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"success": True})

    monkeypatch.setattr(module.requests, "post", post)
    engine = ReefSGLangEngine(SGLangConfig("model", 1, 1, 1), 0)
    engine.server_host, engine.server_port = "127.0.0.1", 8000
    engine.check_weights("checksum")
    engine.pull_weights(3, source_dir="/published", local_checkpoint_dir="/local")
    assert calls == [
        ("http://127.0.0.1:8000/weights_checker", {"action": "checksum"}),
        (
            "http://127.0.0.1:8000/pull_weights",
            {"target_version": 3, "source_dir": "/published", "local_checkpoint_dir": "/local"},
        ),
    ]


@pytest.mark.parametrize("visible,physical,expected", [("4,5,6,7", 6, 2), ("4,5,6,7", 2, 2), ("", 3, 3)])
def test_native_launch_maps_physical_placement_to_visible_gpu(monkeypatch, visible, physical, expected):
    from reef.runtime.sglang.process import local_gpu_id

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert local_gpu_id(physical) == expected


def test_sglang_enables_capture_plugin_without_training_preflight(monkeypatch):
    from reef.runtime.sglang.launch import engine_environment

    monkeypatch.setenv("SGLANG_PLUGINS", "telemetry")
    config = SGLangConfig("model", 1, 1, 1)
    env = engine_environment(config)
    assert env["SGLANG_PLUGINS"] == "telemetry,reef"
    assert env["SGLANG_REEF_PLUGIN"] == "1"
    assert config.options["disable_radix_cache"] is True
    assert config.options["incremental_streaming_output"] is True
