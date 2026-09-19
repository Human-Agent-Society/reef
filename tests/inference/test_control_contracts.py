"""Control integrations must implement the complete inherited interface."""

import pytest

from reef.runtime.executor.failure import ExecutorFailureListener
from reef.runtime.interfaces import AdapterEngine, InferenceMemoryOperations
from reef.runtime.recovery import (
    EngineHealthChecks,
    EngineHealthTarget,
    InferenceEngines,
    InferenceMonitor,
    WeightUpdateConnection,
)


@pytest.mark.parametrize(
    "interface",
    [
        InferenceEngines,
        InferenceMonitor,
        WeightUpdateConnection,
        EngineHealthChecks,
        EngineHealthTarget,
        InferenceMemoryOperations,
        AdapterEngine,
        ExecutorFailureListener,
    ],
)
def test_incomplete_control_integration_cannot_be_constructed(interface):
    class IncompleteIntegration(interface):
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteIntegration()
