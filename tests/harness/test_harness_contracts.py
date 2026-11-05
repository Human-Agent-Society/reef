"""Incomplete harness integrations fail when constructed."""

import pytest

from reef.execution.sandbox.process import ProgramExecutor
from reef.harness.adapters.descriptor import ExecutionValidator
from reef.harness.runners.native import HookListener, Next, ToolRunner
from reef.harness.runners.native.artifact_client import ArtifactUpdateListener, EventWriter
from reef.harness.runners.native.enforce import Enforcer, Tool
from reef.harness.runners.native.graph import Host, TurnLoop
from reef.harness.runners.native.selftools import ServeState
from reef.harness.runners.native.serve import EventSink


@pytest.mark.parametrize(
    "interface",
    [
        ExecutionValidator,
        ProgramExecutor,
        HookListener,
        Next,
        ToolRunner,
        Enforcer,
        Tool,
        Host,
        TurnLoop,
        EventWriter,
        ArtifactUpdateListener,
        ServeState,
        EventSink,
    ],
)
def test_incomplete_harness_integration_cannot_be_constructed(interface):
    class IncompleteIntegration(interface):
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteIntegration()
