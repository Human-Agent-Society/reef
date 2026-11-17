"""Loading a scenario's snapshot into the inference runtime."""

from __future__ import annotations

from pathlib import Path

from tests.support.runtime_stubs import StubInferenceRuntime

from reef.artifact import Artifact
from reef.artifact.components import ComponentSpec, ScenarioComponents
from reef.scenario.binding import activate_snapshot

WEIGHTS = ScenarioComponents({"weights": ComponentSpec(kind="model_adapter")})


def checkpoint(tmp_path: Path) -> Artifact:
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    return Artifact.local(root)


def test_a_snapshot_loads_for_its_scenario_only_on_a_runtime_of_per_scenario_adapters(tmp_path: Path) -> None:
    saved = checkpoint(tmp_path)
    adapters, shared = StubInferenceRuntime(per_scenario_adapters=True), StubInferenceRuntime()
    activate_snapshot(WEIGHTS, saved, adapters, "math")
    activate_snapshot(WEIGHTS, saved, shared, "math")
    assert adapters.activated_scenarios == ["math"]
    assert shared.activated_scenarios == [None]


def test_nothing_loads_without_a_loaded_component_or_a_runtime(tmp_path: Path) -> None:
    saved = checkpoint(tmp_path)
    runtime = StubInferenceRuntime()
    activate_snapshot(ScenarioComponents({"harness": ComponentSpec(kind="harness_plugin")}), saved, runtime, "s")
    activate_snapshot(WEIGHTS, saved, None, "s")
    assert runtime.activated_scenarios == []
