"""Inference assembly owns native settings and validates them before allocation."""

from __future__ import annotations

import subprocess
import sys

import pytest

from reef.inference.sglang import deployment
from reef.inference.sglang.executor import SGLangExecutor


@pytest.mark.parametrize("executor", [None, "auto", "ray"])
def test_native_inference_factory_accepts_plain_input_without_starting_workers(monkeypatch, executor):
    from reef.execution.executor.ray import RayExecutor

    def fail_allocation(*args, **kwargs):
        raise AssertionError("factory must not allocate workers")

    monkeypatch.setattr(RayExecutor, "__init__", fail_allocation)
    options = {
        "model_path": "model",
        "num_gpus": 2,
        "gpus_per_engine": 1,
        "gpus_per_node": 2,
        "executor": executor,
        "executor_options": {"custom": "value"},
    }
    service = deployment.create_inference(options)
    assert service.config.executor is SGLangExecutor
    assert service.config.engine_count == 2
    assert service.config.executor_options == {"custom": "value"}
    assert options["executor"] == executor
    service.close()


@pytest.mark.parametrize("executor", ["mp", "uni"])
def test_native_inference_factory_rejects_non_native_executor_aliases(executor):
    with pytest.raises(ValueError, match="SGLang inference requires ray"):
        deployment.create_inference(
            {"model_path": "model", "num_gpus": 1, "gpus_per_engine": 1, "gpus_per_node": 1, "executor": executor}
        )


@pytest.mark.parametrize(
    ("capacities", "message"),
    [
        ({"num_gpus": 3, "gpus_per_engine": 2, "gpus_per_node": 4}, "whole number of engines"),
        ({"num_gpus": 4, "gpus_per_engine": 4, "gpus_per_node": 2}, "must fit on one node"),
    ],
)
def test_native_factory_refuses_engines_that_do_not_fit_the_reservation(capacities, message):
    with pytest.raises(ValueError, match=message):
        deployment.create_inference({"model_path": "model", **capacities})


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
