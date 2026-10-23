"""Per-scenario adapter routing on a shared-base training runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.runtime_stubs import StubTrainingRuntime

from reef.artifact import Artifact, ArtifactRef, LiveWeightArtifactRef
from reef.core.errors import ReefError
from reef.surface import adapter_name, create_weight_surface
from reef.surface.weights import WeightInferenceHooks, WeightLoader, artifact_runtime_load_id


def live(scenario_version: str) -> Artifact:
    return Artifact(
        LiveWeightArtifactRef(
            content_id="live:x", release_id="live:p:1", parent_release_id=None, runtime_load_id=scenario_version
        ),
        None,
    )


def checkpoint(tmp_path: Path, runtime_load_id: str | None) -> Artifact:
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    return Artifact.local(root, metadata={} if runtime_load_id is None else {"runtime_load_id": runtime_load_id})


class AdapterRuntime(StubTrainingRuntime):
    """Serves one adapter version per scenario, from ``adapters``, and ``engine`` as the engine version."""

    def __init__(self, adapters: dict[str, str], engine: str = "inc:9") -> None:
        super().__init__()
        self.adapters = adapters
        self.engine = engine

    def serving_runtime_load_id(self):
        return self.engine

    def serving_adapter_runtime_load_id(self, scenario):
        return self.adapters.get(scenario)


def hooks_for(scenario: str, adapters: dict[str, str]) -> WeightInferenceHooks:
    return WeightInferenceHooks(scenario=scenario, runtime=AdapterRuntime(adapters).inference)


def test_live_requests_route_to_the_scenario_publication() -> None:
    hooks = hooks_for("math", {"math": "inc:9"})
    out = hooks.prepare_request(live("inc:7"), "/v1/chat/completions", {"messages": []})
    assert out["lora_path"] == adapter_name("math", "inc:7")
    assert out["return_meta_info"] is True
    with pytest.raises(ReefError, match="asked for lora_path"):
        hooks.prepare_request(live("inc:7"), "/v1/chat/completions", {"lora_path": adapter_name("code", "inc:7")})


def test_checkpoint_requests_route_to_the_adapter_the_engine_serves(tmp_path: Path) -> None:
    # Before a restart, the engine serves the checkpoint under the version that published it.
    out = hooks_for("math", {"math": "inc:3"}).prepare_request(
        checkpoint(tmp_path, "inc:3"), "/v1/chat/completions", {}
    )
    assert out["lora_path"] == adapter_name("math", "inc:3")
    # A restart publishes it again under a new version.
    out = hooks_for("math", {"math": "new:1"}).prepare_request(
        checkpoint(tmp_path, "inc:3"), "/v1/chat/completions", {}
    )
    assert out["lora_path"] == adapter_name("math", "new:1")
    with pytest.raises(ReefError, match="holds no adapter for scenario 'math'"):
        hooks_for("math", {}).prepare_request(checkpoint(tmp_path, "inc:3"), "/v1/chat/completions", {})


def test_unpublished_scenarios_sample_the_base(tmp_path: Path) -> None:
    hooks = hooks_for("math", {})
    out = hooks.prepare_request(checkpoint(tmp_path, None), "/v1/chat/completions", {"messages": []})
    assert "lora_path" not in out
    with pytest.raises(ReefError, match="published no adapter yet"):
        hooks.prepare_request(checkpoint(tmp_path, None), "/v1/chat/completions", {"lora_path": "x"})


def test_shared_and_per_scenario_modes_are_exclusive() -> None:
    with pytest.raises(ValueError, match="either"):
        WeightInferenceHooks("reef_lora", scenario="math", runtime=AdapterRuntime({}).inference)
    assert artifact_runtime_load_id(ArtifactRef("id", "v", None)) is None


def test_recovery_checks_the_scenario_adapter_not_the_global_version() -> None:
    class Runtime(StubTrainingRuntime):
        def serving_runtime_load_id(self):
            return "inc:9"  # another scenario published since

        def serving_adapter_runtime_load_id(self, scenario):
            return {"math": "inc:4"}.get(scenario)

    current = LiveWeightArtifactRef(
        content_id="live:x", release_id="live:p:4", parent_release_id=None, runtime_load_id="inc:4"
    )
    checkpoint_ref = ArtifactRef("ckpt", "c0", None)
    runtime = Runtime().inference
    assert WeightLoader("math").recover(current, checkpoint_ref, runtime) == current
    assert WeightLoader().recover(current, checkpoint_ref, runtime) == checkpoint_ref
    # The engine holds no adapter for code: its live head is unservable, so
    # serving falls back to the exact checkpoint instead of routing to a
    # name the engine would reject.
    assert WeightLoader("code").recover(current, checkpoint_ref, runtime) == checkpoint_ref
    surface = create_weight_surface(scenario="math", runtime=runtime)
    assert isinstance(surface.loader, WeightLoader) and isinstance(surface.inference, WeightInferenceHooks)


def test_a_restart_serves_the_checkpoint_under_the_version_it_was_published_again(tmp_path: Path) -> None:
    # The restart published math's checkpoint again as new:1. No committed version names new:1.
    runtime = AdapterRuntime({"math": "new:1"}).inference
    loader = WeightLoader("math")
    saved = checkpoint(tmp_path, "inc:2")
    live = LiveWeightArtifactRef(
        content_id="live:x", release_id="live:p:3", parent_release_id=None, runtime_load_id="inc:3"
    )
    # A newer live head that saved no checkpoint falls back to the checkpoint; a checkpoint head stays.
    assert loader.recover(live, saved.ref, runtime) == saved.ref
    assert loader.recover(saved.ref, saved.ref, runtime) == saved.ref
    # Requests reach the adapter that holds the checkpoint, and records carry the engine's version.
    out = WeightInferenceHooks(scenario="math", runtime=runtime).prepare_request(saved, "/v1/chat/completions", {})
    assert out["lora_path"] == adapter_name("math", "new:1")
    assert loader.served_runtime_load_id(saved, runtime) == "new:1"
