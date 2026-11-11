"""Control and runtime integrations must implement the complete inherited interface."""

import pytest

from reef.core.served_models import ServedArtifact
from reef.execution.executor.failure import ExecutorFailureListener
from reef.inference.http import InferenceProxyRuntime
from reef.inference.interfaces import AdapterEngine, InferenceBackend, InferenceEngine, InferenceRuntime
from reef.inference.null import NullInferenceRuntime
from reef.inference.recovery import InferenceEngineGroup
from reef.inference.serving import AdapterServingRuntime, FullWeightServingRuntime, SavedWeightServingRuntime
from reef.inference.sglang.engine import ReefSGLangEngine
from reef.inference.tinker import TinkerInferenceRuntime
from reef.inference.vllm.engine import ReefVLLMEngine


@pytest.mark.parametrize(
    "interface",
    [
        InferenceEngineGroup,
        InferenceBackend,
        InferenceEngine,
        AdapterEngine,
        ExecutorFailureListener,
    ],
)
def test_incomplete_control_integration_cannot_be_constructed(interface):
    class IncompleteIntegration(interface):
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteIntegration()


@pytest.mark.parametrize("engine_type", [ReefSGLangEngine, ReefVLLMEngine])
def test_native_engines_implement_the_whole_engine_vocabulary(engine_type):
    assert issubclass(engine_type, InferenceEngine)
    assert not engine_type.__abstractmethods__


def test_a_runtime_that_gives_no_served_artifact_cannot_be_constructed():
    class UnboundRuntime(InferenceRuntime):
        @property
        def inference_handler(self):
            raise AssertionError("never called")

        def activate_checkpoint(self, artifact, *, scenario=None):
            raise AssertionError("never called")

    with pytest.raises(TypeError, match="served_artifact"):
        UnboundRuntime(base_url="http://engine.invalid")
    # The shared saved-weight base gives no route either: each runtime reports its own binding.
    assert "served_artifact" in SavedWeightServingRuntime.__abstractmethods__


def test_a_runtime_serves_snapshots_without_weights_only_when_it_names_its_endpoint_model():
    class WeightRuntime(InferenceRuntime):
        @property
        def inference_handler(self):
            raise AssertionError("never called")

        def activate_checkpoint(self, artifact, *, scenario=None):
            raise AssertionError("never called")

        def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
            return None

    # The default refuses a snapshot without weights instead of naming a model for it.
    assert WeightRuntime(base_url="http://engine.invalid").endpoint_model_name("math") is None


@pytest.mark.parametrize(
    "runtime_type",
    [
        InferenceProxyRuntime,
        NullInferenceRuntime,
        TinkerInferenceRuntime,
        AdapterServingRuntime,
        FullWeightServingRuntime,
    ],
)
def test_bundled_runtimes_implement_the_whole_runtime_interface(runtime_type):
    assert issubclass(runtime_type, InferenceRuntime)
    assert not runtime_type.__abstractmethods__
