"""Native runtime construction respects the separately selected inference implementation."""

from __future__ import annotations

import pytest

from reef.inference.http import InferenceProxyRuntime
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.inference.sglang.runtime import SGLangInferenceRuntime
from reef.runtime.deployment import RuntimeConfigError, RuntimeFactory, RuntimeRegistry
from reef.service.assembly import _connect_training_runtime
from reef.service.deploy.service_config import ServiceConfig
from reef.train.runtime import ExecutorTrainingRuntime
from reef.train.slime_backend.runtime import SlimeTrainingRuntime

from .test_executor_runtime import Coordinator


def test_slime_runtime_uses_the_selected_inference_factory(monkeypatch):
    coordinator = Coordinator()
    connections = []

    def connect(**kwargs):
        connections.append(kwargs)
        return coordinator

    monkeypatch.setattr("reef.train.slime_backend.runtime.connect_ray_coordinator", connect)
    selected = InferenceProxyRuntime(base_url="http://other-inference")
    received = []

    class OtherInference(RuntimeFactory):
        kind = "other"

        def __call__(self, config, model_path, recipe_config, environ):
            received.append((config, model_path, recipe_config))
            return selected

    monkeypatch.setattr("reef.runtime.deployment._runtime_kinds", {"other": OtherInference()})
    training, inference = RuntimeRegistry().build(
        {
            "type": "slime_training",
            "inference_runtime": "other",
            "actor_name": "separate-trainer",
            "namespace": "training-cluster",
            "train_timeout_s": 14400,
            "max_staleness": 2,
        },
        model_path="model",
        recipe_config={"method": "custom"},
        environ={},
    )
    assert isinstance(training, SlimeTrainingRuntime)
    assert inference is selected
    assert training.max_staleness == 2
    assert received[0][0]["control"] is coordinator
    assert received[0][1:] == ("model", {"method": "custom"})
    assert connections[0]["actor_name"] == "separate-trainer"
    assert connections[0]["namespace"] == "training-cluster"
    assert connections[0]["train_timeout_s"] == 14400
    assert "inference_handler_factory" not in received[0][0]
    inference.shutdown()
    assert coordinator.shutdown_events == []
    training.shutdown()
    assert coordinator.shutdown_events == ["shutdown"]


def test_sglang_runtime_selects_its_native_request_handler(monkeypatch):
    coordinator = Coordinator()
    requests = []
    handler = InferenceProxyRuntime(base_url="http://router").inference_handler

    def from_config(cls, upstream_url, *, model_path, timeout_s, **config):
        requests.append((upstream_url, model_path, timeout_s, config))
        return handler

    monkeypatch.setattr(SGLangInferenceHandler, "from_config", classmethod(from_config))
    runtime = RuntimeRegistry().build(
        {
            "type": "sglang",
            "control": coordinator,
            "inference_timeout_s": 42,
            "inference_handler_config": {"trust_remote_code": True},
        },
        model_path="selected-model",
        environ={},
    )
    assert isinstance(runtime, SGLangInferenceRuntime)
    assert runtime.inference_handler is handler
    assert requests == [("http://router", "selected-model", 42, {"trust_remote_code": True})]
    assert not runtime.inference_admission_status["open"]
    runtime.shutdown()
    assert coordinator.shutdown_events == []


def test_automatic_service_assembly_uses_both_native_runtime_implementations(monkeypatch):
    coordinator = Coordinator()
    monkeypatch.setattr("reef.train.slime_backend.runtime.connect_ray_coordinator", lambda **kwargs: coordinator)
    handler = InferenceProxyRuntime(base_url="http://router").inference_handler
    received = []

    def from_config(cls, upstream_url, *, model_path, timeout_s, **config):
        received.append(config)
        return handler

    monkeypatch.setattr(SGLangInferenceHandler, "from_config", classmethod(from_config))
    settings = ServiceConfig(
        recipe="recipe",
        training_backend="slime",
        inference_backend="sglang",
        ray_address="auto",
        inference_handler_config={"trust_remote_code": True},
    )
    training, inference = _connect_training_runtime(settings, model_path="model", max_staleness=0)
    try:
        assert isinstance(training, SlimeTrainingRuntime)
        assert isinstance(inference, SGLangInferenceRuntime)
        assert inference.inference_handler is handler
        assert received == [{"trust_remote_code": True}]
    finally:
        inference.shutdown()
        training.shutdown()


def test_slime_runtime_requires_explicit_inference_selection_before_connecting(monkeypatch):
    def connect(**kwargs):
        pytest.fail("invalid runtime configuration must not connect to Ray")

    monkeypatch.setattr("reef.train.slime_backend.runtime.connect_ray_coordinator", connect)
    with pytest.raises(RuntimeConfigError, match="inference_runtime"):
        RuntimeRegistry().build({"type": "slime_training"}, model_path="model", environ={})


def test_failed_inference_construction_releases_the_training_connection(monkeypatch):
    coordinator = Coordinator()
    monkeypatch.setattr("reef.train.slime_backend.runtime.connect_ray_coordinator", lambda **kwargs: coordinator)
    with pytest.raises(RuntimeConfigError, match="unknown runtime type"):
        RuntimeRegistry().build(
            {"type": "slime_training", "inference_runtime": "unavailable"}, model_path="model", environ={}
        )
    assert coordinator.shutdown_events == ["shutdown"]


def test_an_inference_factory_returning_a_pair_releases_all_connections(monkeypatch):
    coordinator = Coordinator()
    other_coordinator = Coordinator()
    stopped = []
    monkeypatch.setattr("reef.train.slime_backend.runtime.connect_ray_coordinator", lambda **kwargs: coordinator)

    class OtherInference(InferenceProxyRuntime):
        def shutdown(self):
            stopped.append("inference")
            super().shutdown()

    class WrongInferenceFactory(RuntimeFactory):
        kind = "wrong-pair"

        def __call__(self, config, model_path, recipe_config, environ):
            return ExecutorTrainingRuntime(other_coordinator), OtherInference(base_url="http://other")

    monkeypatch.setattr("reef.runtime.deployment._runtime_kinds", {"wrong-pair": WrongInferenceFactory()})
    with pytest.raises(RuntimeConfigError, match="must return an InferenceRuntime"):
        RuntimeRegistry().build(
            {"type": "slime_training", "inference_runtime": "wrong-pair"}, model_path="model", environ={}
        )
    assert coordinator.shutdown_events == ["shutdown"]
    assert other_coordinator.shutdown_events == ["shutdown"]
    assert stopped == ["inference"]
