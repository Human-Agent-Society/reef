"""The weight loader on a runtime that serves each scenario its own adapter."""

from __future__ import annotations

from pathlib import Path

from tests.support.runtime_stubs import StubInferenceRuntime

from reef.artifact import Artifact
from reef.surface.weights import WeightLoader


def checkpoint(tmp_path: Path) -> Artifact:
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    return Artifact.local(root)


class RecordingRuntime(StubInferenceRuntime):
    """Records the scenario each activation names."""

    def __init__(self) -> None:
        super().__init__()
        self.scenarios: list[str | None] = []

    def activate_checkpoint(self, artifact, *, scenario=None):
        self.scenarios.append(scenario)
        return "inc:3"


def test_the_loader_activates_releases_through_the_runtime_for_its_scenario(tmp_path: Path) -> None:
    runtime = RecordingRuntime()
    saved = checkpoint(tmp_path)
    assert WeightLoader("math").activate(saved, runtime) == "inc:3"
    assert WeightLoader().activate(saved, runtime) == "inc:3"
    assert runtime.scenarios == ["math", None]
    # A runtime that serves no weights binds nothing: the release names what serves.
    assert WeightLoader("math").activate(saved, None) == saved.ref.release_id
