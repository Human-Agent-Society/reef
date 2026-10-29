"""Incomplete harness integrations fail when constructed."""

import pytest

from reef.execution.sandbox.process import ProgramExecutor
from reef.harness.adapters.descriptor import ExecutionValidator
from reef.harness.capture import ReleaseObserver
from reef.harness.rollouts.model_binding import ModelBindingsResolver
from reef.harness.runners.native import HookListener, Next, ToolRunner
from reef.harness.runners.native.enforce import Enforcer, Tool
from reef.harness.runners.native.graph import Host, TurnLoop
from reef.harness.runners.native.release_client import EventWriter, ReleaseUpdateListener
from reef.harness.runners.native.selftools import ServeState
from reef.harness.runners.native.serve import EventSink


@pytest.mark.parametrize(
    "interface",
    [
        ExecutionValidator,
        ReleaseObserver,
        ProgramExecutor,
        ModelBindingsResolver,
        HookListener,
        Next,
        ToolRunner,
        Enforcer,
        Tool,
        Host,
        TurnLoop,
        EventWriter,
        ReleaseUpdateListener,
        ServeState,
        EventSink,
    ],
)
def test_incomplete_harness_integration_cannot_be_constructed(interface):
    class IncompleteIntegration(interface):
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteIntegration()
