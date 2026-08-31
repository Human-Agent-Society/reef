"""RayRuntime reads the bridge's per-scenario adapter mode."""

from __future__ import annotations

import pytest

from recipes.sao import SAORecipe
from reef.runtime.adapters.ray_runtime import RayRuntime, RayRuntimeError

from .test_ray_runtime import DeferredWeightUpdateTrainGroupHandle


class ScenarioLoraHandle(DeferredWeightUpdateTrainGroupHandle):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lora_mode = "scenario"
        self.lora_adapters = {"math": {"weight_version": "inc:4", "adapter": "reef-adapter-bWF0aA.inc:4"}}
        self.adapter_residency = {
            "capacity": 2,
            "resident": 1,
            "scenarios": {"math": {"current": {"version": "inc:4", "adapter": "reef-adapter-bWF0aA.inc:4"}}},
        }
        self.job_scenario = "math"
        self.acknowledged: list[str] = []

    def health(self):
        health = dict(super().health())
        health["lora_mode"] = self.lora_mode
        health["lora_adapters"] = self.lora_adapters
        health["adapter_residency"] = self.adapter_residency
        health["training_job"] = {**health["training_job"], "scenario": self.job_scenario}
        return health

    def acknowledge_training_commit(self, training_job_id: str) -> None:
        self.acknowledged.append(training_job_id)
        self.status = "COMPLETE"


def test_scenario_mode_enables_concurrent_training_scenarios() -> None:
    runtime = RayRuntime(train_group_handle=ScenarioLoraHandle(), inference_url="http://router")
    assert runtime.concurrent_training_scenarios is True
    assert runtime.serving_adapter_name() is None
    assert runtime.serving_adapter_version("math") == "inc:4"
    assert runtime.serving_adapter_version("code") is None


def test_shared_mode_is_the_default() -> None:
    runtime = RayRuntime(train_group_handle=DeferredWeightUpdateTrainGroupHandle(), inference_url="http://router")
    assert runtime.concurrent_training_scenarios is False
    assert runtime.serving_adapter_version("math") is None
    assert runtime.adapter_residency_status() is None
    assert SAORecipe(runtime).serving_status() is None


def test_recipe_reports_the_bridges_adapter_residency() -> None:
    handle = ScenarioLoraHandle()
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")
    assert runtime.adapter_residency_status() == handle.adapter_residency
    # The bridge's residency is the runtime-wide serving state /reef/status shows.
    assert SAORecipe(runtime).serving_status() == {"adapters": handle.adapter_residency}
    handle.adapter_residency = None
    assert runtime.adapter_residency_status() is None


def test_malformed_lora_mode_is_rejected() -> None:
    handle = ScenarioLoraHandle()
    handle.lora_mode = "weird"
    with pytest.raises(RayRuntimeError, match="unknown LoRA mode"):
        RayRuntime(train_group_handle=handle, inference_url="http://router")


def test_reconcile_leaves_another_scenarios_pending_job_alone() -> None:
    handle = ScenarioLoraHandle(status="READY_TO_COMMIT", rollout_id=3)
    runtime = RayRuntime(train_group_handle=handle, inference_url="http://router")
    # code's backend sees math's job awaiting commit: admission stays closed, no ack.
    runtime.reconcile_training_job(4, committed_training_job_id="job-3", scenario="code")
    assert handle.acknowledged == []
    assert runtime.inference_admission_status["open"] is False
    runtime.reconcile_training_job(4, committed_training_job_id="job-3", scenario="math")
    assert handle.acknowledged == ["job-3"]
    assert runtime.inference_admission_status["open"] is True
