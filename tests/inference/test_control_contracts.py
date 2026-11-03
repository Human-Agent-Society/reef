"""Control integrations must implement the complete inherited interface."""

import pytest

from reef.execution.executor.failure import ExecutorFailureListener
from reef.inference.interfaces import AdapterEngine, InferenceBackend, InferenceEngine
from reef.inference.recovery import EngineHealthChecks, EngineHealthTarget, InferenceEngineGroup, InferenceMonitor
from reef.inference.sglang.engine import ReefSGLangEngine
from reef.inference.vllm.engine import ReefVLLMEngine


@pytest.mark.parametrize(
    "interface",
    [
        InferenceEngineGroup,
        InferenceMonitor,
        EngineHealthChecks,
        EngineHealthTarget,
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
