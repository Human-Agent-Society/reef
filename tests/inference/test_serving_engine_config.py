"""The serving engines' input that Reef builds from the inference settings and the training component's release format."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from reef.inference.remote.runtime import serving_engine_config
from reef.inference.sglang import config as sglang_config
from reef.inference.sglang import deployment as sglang_deployment
from reef.inference.sglang.deployment import create_inference
from reef.inference.sglang.launch import engine_environment
from reef.inference.sglang.plugin import REEF_SGLANG_PLUGIN_ENV
from reef.inference.vllm.config import VLLMConfig
from reef.runtime.jobs import ReleaseFormat

FULL = ReleaseFormat("full")
ADAPTER = ReleaseFormat("adapter", 8, ("q_proj", "v_proj"))


def engine_input(
    engine: str = "sglang",
    *,
    release_format: ReleaseFormat = FULL,
    options: dict | None = None,
    prefix_sharing: bool = False,
    adapter_capacity: int = 1,
    num_gpus: int = 1,
    tensor_parallel_size: int = 1,
    environ: dict | None = None,
) -> dict:
    return serving_engine_config(
        engine,
        model_path="Qwen/Qwen3-8B",
        num_gpus=num_gpus,
        tensor_parallel_size=tensor_parallel_size,
        options=options or {},
        release_format=release_format,
        adapter_capacity=adapter_capacity,
        prefix_sharing=prefix_sharing,
        executor="auto",
        executor_options={},
        environ=environ or {},
    )


@pytest.fixture
def lora_receiver(monkeypatch):
    """An SGLang whose LoRA request schema Reef accepts, without importing SGLang."""
    monkeypatch.setattr(sglang_deployment, "require_lora_tensor_request_schema", lambda: None)
    monkeypatch.setattr(sglang_deployment, "require_lora_distributed_request_schema", lambda: None)


def test_sglang_input_loads_the_model_with_the_flags_reef_serving_relies_on(monkeypatch):
    monkeypatch.setenv("SGLANG_PLUGINS", "telemetry")
    values = engine_input(options={"mem-fraction-static": 0.7}, num_gpus=4, tensor_parallel_size=2)
    config = create_inference(values).config
    assert (config.model_path, config.num_gpus, config.gpus_per_engine, config.gpus_per_node) == (
        "Qwen/Qwen3-8B",
        4,
        2,
        4,
    )
    assert config.options["mem_fraction_static"] == 0.7
    for flag in ("trust_remote_code", "skip_server_warmup", "enable_metrics", "incremental_streaming_output"):
        assert config.options[flag] is True
    assert "enable_lora" not in config.options
    # An in-place pause keeps in-flight KV across a load, so no prefix is shared.
    assert config.pause_mode == "in_place" and config.options["disable_radix_cache"] is True
    environment = engine_environment(config)
    assert environment[REEF_SGLANG_PLUGIN_ENV] == "1"
    assert environment["SGLANG_PLUGINS"] == "telemetry,reef"
    assert os.environ["SGLANG_PLUGINS"] == "telemetry"


def test_sglang_input_enables_reefs_plugin_without_other_plugins(monkeypatch):
    monkeypatch.delenv("SGLANG_PLUGINS", raising=False)
    assert engine_environment(create_inference(engine_input()).config)["SGLANG_PLUGINS"] == "reef"
    assert "SGLANG_PLUGINS" not in os.environ


def test_sglang_shares_prefixes_only_when_each_load_retracts_the_requests_in_flight():
    sharing = create_inference(engine_input(prefix_sharing=True)).config
    assert sharing.pause_mode == "retract" and sharing.options["disable_radix_cache"] is False
    opted_out = create_inference(engine_input(prefix_sharing=True, options={"disable-radix-cache": True})).config
    assert opted_out.options["disable_radix_cache"] is True, "a deployment may always opt out of sharing"
    with pytest.raises(ValueError, match="disable_radix_cache=true"):
        create_inference(engine_input(options={"disable-radix-cache": False}))


def test_adapter_releases_size_the_engines_adapter_table_from_the_release_format_and_capacity(lora_receiver):
    config = create_inference(engine_input(release_format=ADAPTER, adapter_capacity=4)).config
    assert {
        name: config.options[name]
        for name in (
            "enable_lora",
            "max_lora_rank",
            "max_loaded_loras",
            "max_loras_per_batch",
            "lora_target_modules",
            "tokenizer_worker_num",
        )
    } == {
        "enable_lora": True,
        "max_lora_rank": 8,
        "max_loaded_loras": 4,
        "max_loras_per_batch": 4,
        "lora_target_modules": ["q_proj", "v_proj"],
        "tokenizer_worker_num": 1,
    }


@pytest.mark.parametrize("adapter_scoped", [True, False])
def test_retracting_adapter_engines_share_prefixes_only_when_sglang_keys_them_by_adapter(
    monkeypatch, lora_receiver, adapter_scoped
):
    """One engine holds several scenarios' adapters; the engine must keep their prefix-cache entries apart."""
    monkeypatch.setattr(sglang_config, "adapter_scoped_prefix_cache_supported", lambda: adapter_scoped)
    config = create_inference(engine_input(release_format=ADAPTER, prefix_sharing=True)).config
    assert config.options["disable_radix_cache"] is not adapter_scoped
    values = engine_input(release_format=ADAPTER, prefix_sharing=True, options={"disable-radix-cache": False})
    if adapter_scoped:
        create_inference(values)
    else:
        with pytest.raises(ValueError, match="does not key prefix-cache entries by adapter"):
            create_inference(values)


def test_only_sglang_variables_reach_the_engines_on_the_component_host():
    environ = {"SGLANG_ENABLE_X": "1", "SLIME_HOST_IP": "10.0.0.5", "HTTP_PROXY": "http://proxy"}
    assert engine_input(environ=environ)["env_vars"] == {"SGLANG_ENABLE_X": "1"}
    assert engine_input("vllm", environ=environ)["env_vars"] == {}


def test_router_options_configure_the_router_and_router_port_binds_it():
    values = engine_input(options={"router-policy": "cache_aware", "router-port": 30000, "context-length": 4096})
    config = create_inference(values).config
    assert config.router_options == {"policy": "cache_aware"}
    assert config.router_port == 30000
    assert config.options["context_length"] == 4096
    assert not [name for name in config.options if name.startswith("router")]
    assert engine_input()["router_port"] is None
    with pytest.raises(ValueError, match="router-ip names an external router"):
        engine_input(options={"router-ip": "10.0.0.1"})


def test_another_engine_name_gets_the_sglang_input():
    values = engine_input("my.engines:create", release_format=ADAPTER)
    assert values["options"]["enable_lora"] is True
    assert values["env_vars"] == {} and values["router_options"] == {}


def test_vllm_input_keeps_vllm_option_names_and_the_pause_mode():
    values = engine_input("vllm", options={"max-model-len": 4096, "gpu-memory-utilization": 0.5})
    assert values["options"] == {"max_model_len": 4096, "gpu_memory_utilization": 0.5, "trust_remote_code": True}
    config = VLLMConfig(**values)
    assert config.pause_mode == "in_place"
    assert VLLMConfig(**engine_input("vllm", prefix_sharing=True)).pause_mode == "retract"
    assert engine_input("vllm", options={"trust-remote-code": False})["options"]["trust_remote_code"] is False


def test_vllm_refuses_adapter_releases():
    # The serving assembly refuses more than one vLLM engine (tests/reef_service/test_training_startup.py).
    with pytest.raises(ValueError, match="vllm serves full weights only"):
        engine_input("vllm", release_format=ADAPTER)


def test_the_engine_input_imports_no_engine_implementation_or_training_framework():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
sys.modules.update(dict.fromkeys(('sglang', 'torch', 'megatron', 'slime', 'reef.train.slime_backend')))
from reef.inference.remote.runtime import serving_engine_config
from reef.runtime.jobs import ReleaseFormat
values = serving_engine_config('sglang', model_path='model', num_gpus=1, tensor_parallel_size=1, options={},
    release_format=ReleaseFormat('adapter', 8, ('all',)), adapter_capacity=1, prefix_sharing=False,
    executor='auto', executor_options={}, environ={})
json.dumps(values)
assert 'ray' not in sys.modules
assert {name for name in sys.modules if name.startswith('reef.inference.sglang.')} <= {'reef.inference.sglang.chat'}
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
