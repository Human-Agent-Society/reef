"""The inference backend launches and exposes control without importing Slime."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest

from reef.inference.sglang.config import SGLangConfig
from reef.inference.sglang.engine import ReefSGLangEngine
from reef.inference.sglang.launch import SGLangEngineGroup


def test_sglang_imports_with_training_packages_blocked():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules.update(dict.fromkeys(('slime', 'megatron', 'reef.train.slime_backend')))
from reef.inference.sglang import chat, config, control, engine, executor, launch, plugin, service
from reef.inference.interfaces import InferenceEngine
assert engine.ReefSGLangEngine.__bases__ == (InferenceEngine,)
assert not any(name.startswith('slime.') or name.startswith('reef.train.slime_backend.') for name in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def native_engine(monkeypatch):
    from reef.inference.sglang import engine as module

    fields = [
        "model_path",
        "host",
        "port",
        "nccl_port",
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
    process = SimpleNamespace(pid=4321)
    monkeypatch.setattr(module, "launch_engine", lambda options: launched.append(options) or process)
    monkeypatch.setattr(module, "wait_ready", lambda *args: events.append("ready"))
    monkeypatch.setattr(module, "check_server_owner", lambda *args: events.append(("owner", *args)))
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
    config = SGLangConfig("model", 8, 8, 8, options={"pp_size": 2, "enable_memory_saver": True})
    worker = ReefSGLangEngine(config, 0, base_gpu_id=4)
    worker.init("10.0.0.1:18000", 16000, 17000, "10.0.0.1", router_ip="10.0.0.2", router_port=3000)
    assert launched[0]["tp_size"] == 4
    assert launched[0]["pp_size"] == 2
    assert launched[0]["dist_init_addr"] == "10.0.0.1:18000"
    assert launched[0]["base_gpu_id"] == 4
    assert launched[0]["cuda_graph_backend_prefill"] == "disabled"
    assert events == ["ready", ("owner", "10.0.0.1", 16000, 4321), "engine:1", "register"]


def test_group_places_each_engine_on_its_reserved_gpus_and_its_own_ports(monkeypatch):
    from reef.inference.sglang import launch

    calls, constructed, initialized = [], [], []

    class Remote:
        def __init__(self, fn):
            self.remote = fn

    class Actor:
        def __init__(self, rank):
            self._get_current_node_ip_and_free_port = Remote(
                lambda start_port=15000, consecutive=1: ("node", start_port)
            )
            self.init = Remote(lambda **kwargs: initialized.append(kwargs))

    class ActorClass:
        def options(self, **kwargs):
            calls.append(kwargs)
            return self

        def remote(self, config, **kwargs):
            constructed.append(kwargs)
            return Actor(kwargs["rank"])

    monkeypatch.setattr(launch.ray, "remote", lambda cls: ActorClass())
    monkeypatch.setattr(launch.ray, "get", lambda value: value)
    config = SGLangConfig("model", 8, 4, 8)
    group = SGLangEngineGroup(config, (object(), list(range(8)), [0, 1, 2, 3, 4, 5, 6, 7]), ("router", 3000))
    group.start_engines({})
    assert [c["scheduling_strategy"].placement_group_bundle_index for c in calls] == [0, 4]
    assert constructed == [{"rank": 0, "base_gpu_id": 0}, {"rank": 1, "base_gpu_id": 4}]
    # Each engine has a range of 35 ports; port + 3 is its rendezvous port.
    assert [(call["port"], call["dist_init_addr"]) for call in initialized] == [
        (15000, "node:15003"),
        (15035, "node:15038"),
    ]
    assert all((call["router_ip"], call["router_port"]) == ("router", 3000) for call in initialized)
    assert len(group.engines) == 2


def test_group_rejects_noncontiguous_gpu_reservations_before_launch(monkeypatch):
    config = SGLangConfig("model", 2, 2, 4)
    group = SGLangEngineGroup(config, (object(), [0, 1], [0, 2]), ("router", 3000))
    with pytest.raises(ValueError, match="contiguous"):
        group.start_engines({})


@pytest.mark.parametrize("visible,physical,expected", [("4,5,6,7", 6, 2), ("4,5,6,7", 2, 2), ("", 3, 3)])
def test_native_launch_maps_physical_placement_to_visible_gpu(monkeypatch, visible, physical, expected):
    from reef.inference.sglang.process import local_gpu_id

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert local_gpu_id(physical) == expected


def test_engine_process_starts_the_http_server_without_the_training_allocator_settings(monkeypatch):
    from reef.inference.sglang import process as module

    started = []
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints.http_server", SimpleNamespace(launch_server=started.append)
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", SimpleNamespace(ServerArgs=lambda **options: options))
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    options = {"model_path": "model"}
    module._run_engine(options)
    assert started == [options]
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert "PYTORCH_ALLOC_CONF" not in os.environ


def test_engine_starts_in_its_own_spawned_process(monkeypatch):
    from reef.inference.sglang import process as module

    events = []

    class Process:
        def __init__(self, target, args):
            events.append((target, args))

        def start(self):
            events.append("start")

    monkeypatch.setattr(
        module.multiprocessing, "get_context", lambda method: events.append(method) or SimpleNamespace(Process=Process)
    )
    options = {"model_path": "model"}
    assert isinstance(module.launch_engine(options), Process)
    assert events == ["spawn", (module._run_engine, (options,)), "start"]


def test_sglang_enables_reef_plugin_without_training_preflight(monkeypatch):
    from reef.inference.sglang.launch import engine_environment

    monkeypatch.setenv("SGLANG_PLUGINS", "telemetry")
    config = SGLangConfig("model", 1, 1, 1)
    env = engine_environment(config)
    assert env["SGLANG_PLUGINS"] == "telemetry,reef"
    assert env["SGLANG_REEF_PLUGIN"] == "1"
    assert config.options["disable_radix_cache"] is True
    assert config.options["incremental_streaming_output"] is True
