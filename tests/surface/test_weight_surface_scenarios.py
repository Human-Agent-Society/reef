"""Per-scenario adapter routing on a runtime that serves each scenario its own adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.runtime_stubs import StubInferenceRuntime

from reef.artifact import Artifact
from reef.core.errors import ReefError
from reef.surface import adapter_name
from reef.surface.weights import WeightInferenceHooks, WeightLoader


def checkpoint(tmp_path: Path) -> Artifact:
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    return Artifact.local(root)


class AdapterRuntime(StubInferenceRuntime):
    """Serves one adapter version per scenario, from ``adapters``, and ``engine`` as the last load's version."""

    def __init__(self, adapters: dict[str, str], engine: str = "inc:9") -> None:
        super().__init__()
        self.adapters = adapters
        self.engine = engine

    def current_runtime_load_id(self):
        return self.engine

    def serving_adapter_runtime_load_id(self, scenario):
        return self.adapters.get(scenario)


def hooks_for(scenario: str, adapters: dict[str, str]) -> WeightInferenceHooks:
    return WeightInferenceHooks(scenario=scenario, runtime=AdapterRuntime(adapters))


def test_checkpoint_requests_route_to_the_adapter_the_engine_serves(tmp_path: Path) -> None:
    out = hooks_for("math", {"math": "inc:3"}).prepare_request(checkpoint(tmp_path), "/v1/chat/completions", {})
    assert out["lora_path"] == adapter_name("math", "inc:3")
    # A restart serves the checkpoint again under a new version.
    out = hooks_for("math", {"math": "new:1"}).prepare_request(checkpoint(tmp_path), "/v1/chat/completions", {})
    assert out["lora_path"] == adapter_name("math", "new:1")
    with pytest.raises(ReefError, match="serves adapter"):
        hooks_for("math", {"math": "new:1"}).prepare_request(
            checkpoint(tmp_path), "/v1/chat/completions", {"lora_path": adapter_name("math", "inc:3")}
        )


def test_a_scenario_that_serves_no_adapter_samples_the_base(tmp_path: Path) -> None:
    hooks = hooks_for("math", {"code": "inc:1"})
    out = hooks.prepare_request(checkpoint(tmp_path), "/v1/chat/completions", {"messages": []})
    assert "lora_path" not in out
    with pytest.raises(ReefError, match="published no adapter yet"):
        hooks.prepare_request(checkpoint(tmp_path), "/v1/chat/completions", {"lora_path": "x"})


def test_a_restart_serves_the_checkpoint_under_the_version_it_was_loaded_again(tmp_path: Path) -> None:
    # The restart loaded math's checkpoint again as new:1.
    runtime = AdapterRuntime({"math": "new:1"})
    loader = WeightLoader("math")
    saved = checkpoint(tmp_path)
    # Requests reach the adapter that holds the checkpoint, and records carry its version.
    out = WeightInferenceHooks(scenario="math", runtime=runtime).prepare_request(saved, "/v1/chat/completions", {})
    assert out["lora_path"] == adapter_name("math", "new:1")
    assert loader.served_runtime_load_id(saved, runtime) == "new:1"
    assert WeightLoader("code").served_runtime_load_id(saved, runtime) is None


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
