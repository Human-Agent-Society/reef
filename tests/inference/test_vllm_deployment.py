"""vLLM inference assembly owns native settings and validates them before allocation."""

from __future__ import annotations

import subprocess
import sys

import pytest

from reef.inference import deployment as registry
from reef.inference.vllm.config import VLLMSettings
from reef.inference.vllm.control import VLLMExecutor
from reef.inference.vllm.service import VLLMInferenceService


def base_values(**overrides):
    return {"model_path": "model", "num_gpus": 2, "gpus_per_engine": 2, "gpus_per_node": 8, **overrides}


@pytest.mark.parametrize("executor", [None, "auto", "ray"])
def test_factory_accepts_plain_input_without_starting_workers(monkeypatch, executor):
    from reef.execution.executor.ray import RayExecutor

    def fail_allocation(*args, **kwargs):
        raise AssertionError("factory must not allocate workers")

    monkeypatch.setattr(RayExecutor, "__init__", fail_allocation)
    options = base_values(executor=executor, executor_options={"custom": "value"}, options={"max-model-len": 4096})
    service = registry.inference_service_for("vllm", options)
    assert isinstance(service, VLLMInferenceService)
    assert service.config.executor is VLLMExecutor
    assert service.config.executor_options == {"custom": "value"}
    assert service.config.options["max_model_len"] == 4096
    assert options["executor"] == executor
    service.close()


@pytest.mark.parametrize("executor", ["mp", "uni"])
def test_factory_rejects_non_native_executor_aliases(executor):
    with pytest.raises(ValueError, match="vLLM inference requires ray"):
        VLLMInferenceService.from_config(base_values(executor=executor))


def test_config_derives_reef_serving_options():
    config = VLLMSettings(**base_values())
    assert config.options["logprobs_mode"] == "processed_logprobs"
    assert config.options["generation_config"] == "vllm"
    explicit = VLLMSettings(**base_values(options={"generation_config": "/models/qwen/generation"}))
    assert explicit.options["generation_config"] == "/models/qwen/generation"
    assert config.options["enable_prefix_caching"] is False
    assert config.engine_count == 1
    retracting = VLLMSettings(**base_values(pause_mode="retract"))
    assert retracting.options["enable_prefix_caching"] is True


@pytest.mark.parametrize(
    "values, message",
    [
        (base_values(num_gpus=3), "whole number of engines"),
        (base_values(gpus_per_engine=2, gpus_per_node=1), "fit on one node"),
        (base_values(num_gpus=4), "requires router_url"),
        (base_values(pause_mode="drain"), "unknown vLLM pause mode"),
        (base_values(options={"tensor_parallel_size": 2}), "derives these vLLM options"),
        (base_values(options={"enable_prefix_caching": True}), "unless publication retracts"),
        (base_values(options={"enable_prefix_caching": "yes"}, pause_mode="retract"), "must be a boolean"),
        (base_values(request_timeout=0), "timeouts must be positive"),
        (base_values(engine_port_base=0), "engine_port_base"),
        (base_values(engine_port_base=65501), "engine_port_base"),
    ],
)
def test_config_rejects_settings_reef_cannot_serve(values, message):
    with pytest.raises(ValueError, match=message):
        VLLMSettings(**values)


def test_two_engines_serve_behind_a_configured_router():
    config = VLLMSettings(**base_values(num_gpus=4, router_url="http://router:8000"))
    assert config.engine_count == 2


def test_factory_imports_and_builds_with_training_stack_and_vllm_blocked():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules.update(dict.fromkeys(('slime', 'megatron', 'torch', 'vllm', 'reef.train.slime_backend')))
from reef.inference.vllm.service import VLLMInferenceService
service = VLLMInferenceService.from_config(dict(model_path='model', num_gpus=1, gpus_per_engine=1, gpus_per_node=1))
assert service.config.options['logprobs_mode'] == 'processed_logprobs'
from reef.inference.vllm import backend, control, engine, launch, service as service_module, worker
from reef.inference.interfaces import InferenceEngine
assert engine.ReefVLLMEngine.__bases__ == (InferenceEngine,)
assert not any(name.startswith(('slime.', 'reef.train.slime_backend.', 'reef.inference.sglang')) for name in sys.modules)
service.close()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
