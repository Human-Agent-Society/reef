"""Slime's serving engines on vLLM or SGLang: the engines' input that Reef builds from Slime's flags."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from reef.inference.vllm.config import VLLMConfig
from reef.train.slime_backend.inference import inference_config, remote_inference_config


def slime_args(**options):
    return SimpleNamespace(
        **{
            "hf_checkpoint": "model",
            "seed": 1,
            "fp16": False,
            "use_rollout_routing_replay": False,
            "megatron_lora_rank": 0,
            "rollout_num_gpus": 1,
            "rollout_num_gpus_per_engine": 1,
            "num_gpus_per_node": 1,
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 1,
            "disjoint_prefix_sharing": False,
            "reef_rollout_executor_backend": "auto",
            "reef_rollout_executor_options": {},
            **options,
        }
    )


def test_vllm_inference_config_uses_reef_option_names_and_placement_decisions(monkeypatch):
    # SGLang images set SGLANG_* variables; a vLLM engine receives none of them.
    monkeypatch.setenv("SGLANG_IMAGE_TAG", "lmsysorg/sglang:test")
    values = inference_config(slime_args(), "vllm", {"max-model-len": 4096, "gpu-memory-utilization": 0.5})
    config = VLLMConfig(**values)
    assert config.options["max_model_len"] == 4096
    assert config.options["gpu_memory_utilization"] == 0.5
    assert config.options["trust_remote_code"] is True
    assert config.options["seed"] == 1
    assert "sglang" not in str(values) and not any(name.startswith("SGLANG_") for name in values["env_vars"])
    assert config.pause_mode == "in_place"
    assert VLLMConfig(**inference_config(slime_args(disjoint_prefix_sharing=True), "vllm")).pause_mode == "retract"
    assert VLLMConfig(**inference_config(slime_args(fp16=True), "vllm")).options["dtype"] == "float16"
    with pytest.raises(ValueError, match="supports inference backends"):
        inference_config(slime_args(), "tinker")


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_remote_engine_input_loads_the_components_model_without_this_hosts_addresses(monkeypatch, backend):
    # An SGLang image sets SGLANG_* variables of its own; only the one set here may reach the engine.
    for name in [name for name in os.environ if name.startswith("SGLANG_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("SLIME_HOST_IP", "10.0.0.5")
    monkeypatch.setenv("SGLANG_ENABLE_X", "1")
    args = slime_args(hf_checkpoint="/root/.cache/huggingface/hub/snapshot", rollout_num_gpus=2, num_gpus_per_node=8)
    local = inference_config(args, backend)
    remote = remote_inference_config(args, backend, None, model_path="Qwen/Qwen3-8B")
    assert local["model_path"] == "/root/.cache/huggingface/hub/snapshot"
    assert remote["model_path"] == "Qwen/Qwen3-8B"
    if "model_path" in local["options"]:
        assert remote["options"]["model_path"] == "Qwen/Qwen3-8B"
    # SGLang's variables reach only an SGLang engine.
    assert "REEF_INFERENCE_HOST" in local["env_vars"]
    assert remote["env_vars"] == ({"SGLANG_ENABLE_X": "1"} if backend == "sglang" else {})
    assert remote["gpus_per_node"] == 2
    if backend == "sglang":
        assert (remote["external_engines"], remote["router_host"], remote["router_port"]) == ((), None, None)
    assert {
        key: value
        for key, value in remote.items()
        if key
        not in {"model_path", "options", "env_vars", "gpus_per_node", "external_engines", "router_host", "router_port"}
    } == {
        key: value
        for key, value in local.items()
        if key
        not in {"model_path", "options", "env_vars", "gpus_per_node", "external_engines", "router_host", "router_port"}
    }


def test_sglang_mapping_is_unchanged_by_the_backend_argument():
    legacy = inference_config(slime_args())
    assert inference_config(slime_args(), "sglang") == legacy
    assert legacy["options"]["incremental_streaming_output"] is True
    assert "router_options" in legacy
