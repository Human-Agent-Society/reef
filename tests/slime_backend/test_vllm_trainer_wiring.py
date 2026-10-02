"""Slime pairs with a vLLM receiver through the disk weight path: config mapping, protocol and driver flags."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.inference.vllm.config import VLLMConfig
from reef.inference.vllm.service import INFERENCE_PROTOCOL as VLLM_PROTOCOL
from reef.inference.vllm.service import VLLMInferenceService
from reef.runtime.deployment import DeploymentResources, ModelDeploymentPlan
from reef.train.slime_backend.inference import inference_config
from reef.train.slime_backend.reef_adapters.bridge import BridgePreparation, RetentionConfig
from reef.train.slime_backend.training import WEIGHT_TRANSFER_PROTOCOLS, SlimeTrainingService


def slime_args(**options):
    return SimpleNamespace(
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
        }
    )


def test_vllm_inference_config_uses_reef_option_names_and_placement_decisions():
    values = inference_config(
        slime_args(colocate=True, offload_rollout=True), "vllm", {"max-model-len": 4096, "gpu-memory-utilization": 0.5}
    )
    config = VLLMConfig(**values)
    assert config.options["max_model_len"] == 4096
    assert config.options["gpu_memory_utilization"] == 0.5
    assert config.options["trust_remote_code"] is True
    assert config.options["seed"] == 1
    assert "sglang" not in str(values)
    assert config.offload is True and config.shared_gpus == 1 and config.pause_mode == "retract"
    disjoint = VLLMConfig(**inference_config(slime_args(), "vllm"))
    assert disjoint.offload is False and disjoint.shared_gpus == 0 and disjoint.pause_mode == "in_place"
    assert VLLMConfig(**inference_config(slime_args(disjoint_prefix_sharing=True), "vllm")).pause_mode == "retract"
    assert VLLMConfig(**inference_config(slime_args(fp16=True), "vllm")).options["dtype"] == "float16"
    with pytest.raises(ValueError, match="supports inference backends"):
        inference_config(slime_args(), "tinker")


def test_sglang_mapping_is_unchanged_by_the_backend_argument():
    legacy = inference_config(slime_args())
    assert inference_config(slime_args(), "sglang") == legacy
    assert legacy["options"]["incremental_streaming_output"] is True
    assert "router_options" in legacy


class _Resources(DeploymentResources):
    def training_node_id(self):
        return None

    def start(self):
        return None

    def close(self):
        return None


def test_training_service_declares_the_receiver_protocol_for_its_backend():
    preparation = BridgePreparation(retention=RetentionConfig(), loss_family=None, lora=False)

    def service(**kwargs):
        return SlimeTrainingService(SimpleNamespace(), preparation=preparation, loss_family_config=None, **kwargs)

    assert service().weight_transfer_protocol == WEIGHT_TRANSFER_PROTOCOLS["sglang"] == "slime-sglang-control-v2"
    vllm_trainer = service(inference_backend="vllm")
    assert vllm_trainer.weight_transfer_protocol == VLLM_PROTOCOL
    with pytest.raises(ValueError, match="no receiver protocol"):
        service(inference_backend="tinker")
    plan = ModelDeploymentPlan(
        resources=_Resources(), inference=VLLMInferenceService(VLLMConfig("model", 1, 1, 1)), training=vllm_trainer
    )
    plan.validate()
    with pytest.raises(ValueError, match="incompatible inference control protocol"):
        ModelDeploymentPlan(
            resources=_Resources(), inference=VLLMInferenceService(VLLMConfig("model", 1, 1, 1)), training=service()
        ).validate()
