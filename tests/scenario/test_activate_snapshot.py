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


class RecordingRuntime(StubInferenceRuntime):
    """Records the scenario each activation names."""

    def __init__(self, *, per_scenario: bool) -> None:
        super().__init__()
        self.per_scenario = per_scenario
        self.scenarios: list[str | None] = []

    @property
    def per_scenario_adapters(self) -> bool:
        return self.per_scenario

    def activate_checkpoint(self, artifact, *, scenario=None):
        self.scenarios.append(scenario)
        return "inc:3"


def test_a_release_loads_for_its_scenario_only_on_a_runtime_of_per_scenario_adapters(tmp_path: Path) -> None:
    saved = checkpoint(tmp_path)
    adapters, shared = RecordingRuntime(per_scenario=True), RecordingRuntime(per_scenario=False)
    activate_snapshot(WEIGHTS, saved, adapters, "math")
    activate_snapshot(WEIGHTS, saved, shared, "math")
    assert adapters.scenarios == ["math"]
    assert shared.scenarios == [None]


def test_nothing_loads_without_a_loaded_component_or_a_runtime(tmp_path: Path) -> None:
    saved = checkpoint(tmp_path)
    runtime = RecordingRuntime(per_scenario=False)
    activate_snapshot(ScenarioComponents({"harness": ComponentSpec(kind="harness_plugin")}), saved, runtime, "s")
    activate_snapshot(WEIGHTS, saved, None, "s")
    assert runtime.scenarios == []
