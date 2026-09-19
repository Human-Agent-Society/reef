"""Inference assembly owns native settings and validates them before allocation."""

from __future__ import annotations

import subprocess
import sys

import pytest

from reef.inference.sglang import deployment
from reef.inference.sglang.executor import SGLangExecutor


@pytest.mark.parametrize("executor", [None, "auto", "ray"])
def test_native_inference_factory_accepts_plain_input_without_starting_workers(monkeypatch, executor):
    from reef.inference.sglang.service import RayExecutor

    def fail_allocation(*args, **kwargs):
        raise AssertionError("factory must not allocate workers")

    monkeypatch.setattr(RayExecutor, "__init__", fail_allocation)
    options = {
        "model_path": "model",
        "num_gpus": 2,
        "gpus_per_engine": 1,
        "gpus_per_node": 2,
        "models": (
            {
                "name": "actor",
                "groups": ({"worker_type": "regular", "num_gpus": 2, "gpus_per_engine": 1},),
            },
        ),
        "executor": executor,
        "executor_options": {"custom": "value"},
    }
    service = deployment.create_inference(options)
    assert service.config.executor is SGLangExecutor
    assert service.config.models[0].name == "actor"
    assert service.config.models[0].groups[0].num_gpus == 2
    assert service.config.executor_options == {"custom": "value"}
    assert options["executor"] == executor
    service.close()


@pytest.mark.parametrize("executor", ["mp", "uni"])
def test_native_inference_factory_rejects_non_native_executor_aliases(executor):
    with pytest.raises(ValueError, match="SGLang inference requires ray"):
        deployment.create_inference(
            {"model_path": "model", "num_gpus": 1, "gpus_per_engine": 1, "gpus_per_node": 1, "executor": executor}
        )


def test_native_factory_validates_lora_receiver_schema_before_allocation(monkeypatch):
    checks = []
    monkeypatch.setattr(deployment, "require_lora_tensor_request_schema", lambda: checks.append("tensor"))

    def incompatible_schema():
        raise RuntimeError("receiver cannot load distributed LoRA")

    monkeypatch.setattr(deployment, "require_lora_distributed_request_schema", incompatible_schema)
    with pytest.raises(RuntimeError, match="cannot load distributed LoRA"):
        deployment.create_inference(
            {
                "model_path": "model",
                "num_gpus": 1,
                "gpus_per_engine": 1,
                "gpus_per_node": 1,
                "options": {"enable_lora": True},
            }
        )
    assert checks == ["tensor"]


def test_native_factory_imports_and_builds_with_training_stack_blocked():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules.update(dict.fromkeys(('slime', 'megatron', 'torch', 'reef.train.slime_backend')))
from reef.inference.sglang.deployment import create_inference
service = create_inference(dict(model_path='model', num_gpus=1, gpus_per_engine=1, gpus_per_node=1))
assert service.config.options['disable_radix_cache'] is True
service.close()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_slime_input_mapping_imports_no_receiver_implementation_or_framework():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
from types import SimpleNamespace
sys.modules.update(dict.fromkeys(('reef.inference.sglang', 'sglang', 'torch', 'megatron')))
from reef.train.slime_backend.inference import inference_config
from reef.train.slime_backend.reef_adapters.preflight import validate_bridge_args
values = inference_config(SimpleNamespace(hf_checkpoint='model', rollout_num_gpus=1, rollout_num_gpus_per_engine=1, num_gpus_per_node=1, seed=1, offload_rollout=False, fp16=False, use_rollout_routing_replay=False, megatron_lora_rank=0, disjoint_prefix_sharing=False))
assert values['model_path'] == 'model'
assert values['executor'] == 'auto'
json.dumps(values)
assert not any(name.startswith('reef.inference.sglang.') for name in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
