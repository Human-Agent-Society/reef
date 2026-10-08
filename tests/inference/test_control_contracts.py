"""Control integrations must implement the complete inherited interface."""

import pytest

from reef.inference.sglang.engine import ReefSGLangEngine
from reef.inference.vllm.engine import ReefVLLMEngine
from reef.runtime.executor.failure import ExecutorFailureListener
from reef.runtime.interfaces import AdapterEngine, InferenceEngine, InferenceMemoryOperations
from reef.runtime.recovery import (
    EngineHealthChecks,
    EngineHealthTarget,
    InferenceEngineGroup,
    InferenceMonitor,
    WeightUpdateConnection,
)


@pytest.mark.parametrize(
    "interface",
    [
        InferenceEngineGroup,
        InferenceMonitor,
        WeightUpdateConnection,
        EngineHealthChecks,
        EngineHealthTarget,
        InferenceMemoryOperations,
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
