"""vLLM inference assembly owns native settings and validates them before allocation."""

from __future__ import annotations

import subprocess
import sys

import pytest

from reef.inference import deployment as registry
from reef.inference.vllm import deployment
from reef.inference.vllm.config import REEF_CONNECTOR_CONFIG, VLLMConfig, kv_transfer_config
from reef.inference.vllm.control import VLLMExecutor
from reef.inference.vllm.service import INFERENCE_PROTOCOL, VLLMInferenceService


def base_values(**overrides):
    return {"model_path": "model", "num_gpus": 2, "gpus_per_engine": 2, "gpus_per_node": 8, **overrides}


@pytest.mark.parametrize("executor", [None, "auto", "ray"])
def test_factory_accepts_plain_input_without_starting_workers(monkeypatch, executor):
    from reef.inference.vllm.service import RayExecutor

    def fail_allocation(*args, **kwargs):
        raise AssertionError("factory must not allocate workers")

    monkeypatch.setattr(RayExecutor, "__init__", fail_allocation)
    options = base_values(executor=executor, executor_options={"custom": "value"}, options={"max-model-len": 4096})
    service = registry.inference_service_for("vllm", options)
    assert isinstance(service, VLLMInferenceService)
    assert service.connection_protocol == INFERENCE_PROTOCOL
    assert service.config.executor is VLLMExecutor
    assert service.config.executor_options == {"custom": "value"}
    assert service.config.options["max_model_len"] == 4096
    assert options["executor"] == executor
    service.close()


@pytest.mark.parametrize("executor", ["mp", "uni"])
def test_factory_rejects_non_native_executor_aliases(executor):
    with pytest.raises(ValueError, match="vLLM inference requires ray"):
        deployment.create_inference(base_values(executor=executor))


def test_config_derives_reef_serving_options():
    config = VLLMConfig(**base_values())
    assert config.options["logprobs_mode"] == "processed_logprobs"
    assert config.options["enable_prefix_caching"] is False
    assert config.options["kv_transfer_config"] == REEF_CONNECTOR_CONFIG
    assert config.engine_count == 1
    retracting = VLLMConfig(**base_values(pause_mode="retract"))
    assert retracting.options["enable_prefix_caching"] is True


@pytest.mark.parametrize(
    "values, message",
    [
        (base_values(num_gpus=3), "whole number of engines"),
        (base_values(gpus_per_engine=2, gpus_per_node=1), "fit on one node"),
        (base_values(num_gpus=4), "requires router_url"),
        (base_values(check_weights=True), "no weights checker"),
        (base_values(pause_mode="drain"), "unknown vLLM pause mode"),
        (base_values(options={"tensor_parallel_size": 2}), "derives these vLLM options"),
        (base_values(options={"kv-offloading-size": 4}), "replaces the configured KV connector"),
        (base_values(options={"enable_prefix_caching": True}), "unless publication retracts"),
        (base_values(options={"enable_prefix_caching": "yes"}, pause_mode="retract"), "must be a boolean"),
        (base_values(options={"kv_transfer_config": {"kv_role": "kv_both"}}), "naming kv_connector"),
        (base_values(request_timeout=0), "timeouts must be positive"),
    ],
)
def test_config_rejects_settings_reef_cannot_serve(values, message):
    with pytest.raises(ValueError, match=message):
        VLLMConfig(**values)


def test_two_engines_serve_behind_a_configured_router():
    config = VLLMConfig(**base_values(num_gpus=4, router_url="http://router:8000"))
    assert config.engine_count == 2


def test_configured_connectors_compose_with_reef_under_multi_connector():
    offloading = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": 1},
    }
    composed = kv_transfer_config(offloading)
    assert composed["kv_connector"] == "MultiConnector"
    assert composed["kv_connector_extra_config"]["connectors"] == [offloading, REEF_CONNECTOR_CONFIG]
    # A MultiConnector the operator wrote gains Reef's connector once and keeps its other settings.
    assert kv_transfer_config(composed) == composed
    multi = {"kv_connector": "MultiConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"connectors": []}}
    assert kv_transfer_config(multi)["kv_connector_extra_config"]["connectors"] == [REEF_CONNECTOR_CONFIG]
    assert kv_transfer_config('{"kv_connector": "ReefConnector", "kv_role": "kv_both"}')["kv_connector"] == (
        "ReefConnector"
    )


def test_factory_imports_and_builds_with_training_stack_and_vllm_blocked():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules.update(dict.fromkeys(('slime', 'megatron', 'torch', 'vllm', 'reef.train.slime_backend')))
from reef.inference.vllm.deployment import create_inference
service = create_inference(dict(model_path='model', num_gpus=1, gpus_per_engine=1, gpus_per_node=1))
assert service.config.options['kv_transfer_config']['kv_connector'] == 'ReefConnector'
from reef.inference.vllm import backend, control, engine, launch, service as service_module, worker
assert engine.ReefVLLMEngine.__bases__ == (object,)
assert not any(name.startswith(('slime.', 'reef.train.slime_backend.', 'reef.inference.sglang')) for name in sys.modules)
service.close()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
