"""Incomplete harness integrations fail when constructed."""

import pytest

from reef.harness.adapters.descriptor import ExecutionValidator
from reef.harness.client.wrapper import ReleaseObserver
from reef.harness.compose.registry import ObjectPlugin
from reef.harness.episodes.executor import EpisodeExecutor
from reef.harness.episodes.model_binding import ModelBindingsResolver
from reef.harness.runners.native import HookListener, Next, ToolRunner
from reef.harness.runners.native.enforce import Enforcer, Tool
from reef.harness.runners.native.graph import Host, TurnLoop
from reef.harness.runners.native.host import TreeOrder
from reef.harness.runners.native.release_client import EventWriter, ReleaseUpdateListener
from reef.harness.runners.native.selftools import ServeState
from reef.harness.runners.native.serve import EventSink


@pytest.mark.parametrize(
    "interface",
    [
        ExecutionValidator,
        ReleaseObserver,
        ObjectPlugin,
        EpisodeExecutor,
        ModelBindingsResolver,
        HookListener,
        Next,
        ToolRunner,
        Enforcer,
        Tool,
        Host,
        TurnLoop,
        TreeOrder,
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
