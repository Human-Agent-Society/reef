"""``inference.backend: none``: a runtime that serves nothing and refuses every inference request."""

from __future__ import annotations

import asyncio

import pytest

from reef.artifact import Artifact
from reef.core.errors import InferenceUnavailable
from reef.inference.null import NULL_INFERENCE_URL, NullInferenceHandler, NullInferenceRuntime
from reef.runtime.deployment import RuntimeRegistry


def test_the_factory_builds_a_runtime_that_serves_nothing_and_loads_nothing(tmp_path):
    runtime = RuntimeRegistry().build({"type": "none", "inference_timeout_s": 12.0}, model_path="/models/qwen")
    assert isinstance(runtime, NullInferenceRuntime)
    assert (runtime.base_url, runtime.model_path, runtime.inference_timeout_s) == (
        NULL_INFERENCE_URL,
        "/models/qwen",
        12.0,
    )
    # Nothing is served, so several training scenarios can share the runtime.
    assert runtime.shared_by_training_scenarios is True
    assert runtime.per_scenario_adapters is False
    release = Artifact.local(tmp_path)
    assert runtime.activate_checkpoint(release, scenario="math") == release.ref.release_id
    assert runtime.current_runtime_load_id() is None
    runtime.shutdown()


def test_admission_and_the_handler_refuse_every_request(tmp_path):
    # The HTTP contract (503 on every inference route) is in tests/reef_service/test_runtime_registry.py.
    runtime = NullInferenceRuntime()
    with pytest.raises(InferenceUnavailable, match=r"inference\.backend: none"):
        asyncio.run(runtime.acquire_inference())
    handler = runtime.inference_handler
    assert isinstance(handler, NullInferenceHandler)
    with pytest.raises(InferenceUnavailable, match=r"inference\.backend: none"):
        asyncio.run(handler.inference(Artifact.local(tmp_path), "/v1/chat/completions", {}))
