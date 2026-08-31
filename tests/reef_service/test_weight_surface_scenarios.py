"""Per-scenario adapter routing on a shared-base training runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime

from reef.artifact import Artifact, ArtifactRef, LiveWeightArtifactRef
from reef.core.errors import ReefError
from reef.surface import adapter_name, create_weight_surface
from reef.surface.weights import WeightInferenceHooks, WeightLoader, artifact_weight_version


def live(scenario_version: str) -> Artifact:
    return Artifact(
        LiveWeightArtifactRef(
            artifact_id="live:x", version="live:p:1", parent_version=None, weight_version=scenario_version
        ),
        None,
    )


def checkpoint(tmp_path: Path, weight_version: str | None) -> Artifact:
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    return Artifact.local(root, metadata={} if weight_version is None else {"weight_version": weight_version})


def test_live_requests_route_to_the_scenario_publication() -> None:
    hooks = WeightInferenceHooks(scenario="math")
    out = hooks.prepare_request(live("inc:7"), "/v1/chat/completions", {"messages": []})
    assert out["lora_path"] == adapter_name("math", "inc:7")
    assert out["return_meta_info"] is True
    with pytest.raises(ReefError, match="asked for lora_path"):
        hooks.prepare_request(live("inc:7"), "/v1/chat/completions", {"lora_path": adapter_name("code", "inc:7")})


def test_checkpoint_requests_route_by_recorded_weight_version(tmp_path: Path) -> None:
    hooks = WeightInferenceHooks(scenario="math")
    out = hooks.prepare_request(checkpoint(tmp_path, "inc:3"), "/v1/chat/completions", {})
    assert out["lora_path"] == adapter_name("math", "inc:3")


def test_unpublished_scenarios_sample_the_base(tmp_path: Path) -> None:
    hooks = WeightInferenceHooks(scenario="math")
    out = hooks.prepare_request(checkpoint(tmp_path, None), "/v1/chat/completions", {"messages": []})
    assert "lora_path" not in out
    with pytest.raises(ReefError, match="published no adapter yet"):
        hooks.prepare_request(checkpoint(tmp_path, None), "/v1/chat/completions", {"lora_path": "x"})


def test_shared_and_per_scenario_modes_are_exclusive() -> None:
    with pytest.raises(ValueError, match="either"):
        WeightInferenceHooks("reef_lora", scenario="math")
    assert artifact_weight_version(ArtifactRef("id", "v", None)) is None


def test_recovery_checks_the_scenario_adapter_not_the_global_version() -> None:
    class Runtime(StubTrainingRuntime):
        def serving_weight_version(self):
            return "inc:9"  # another scenario published since

        def serving_adapter_version(self, scenario):
            return {"math": "inc:4"}.get(scenario)

    current = LiveWeightArtifactRef(
        artifact_id="live:x", version="live:p:4", parent_version=None, weight_version="inc:4"
    )
    checkpoint_ref = ArtifactRef("ckpt", "c0", None)
    assert WeightLoader("math").recover(current, checkpoint_ref, Runtime()) == current
    assert WeightLoader().recover(current, checkpoint_ref, Runtime()) == checkpoint_ref
    # The engine holds no adapter for code: its live head is unservable, so
    # serving falls back to the exact checkpoint instead of routing to a
    # name the engine would reject.
    assert WeightLoader("code").recover(current, checkpoint_ref, Runtime()) == checkpoint_ref
    surface = create_weight_surface(scenario="math")
    assert isinstance(surface.loader, WeightLoader) and isinstance(surface.inference, WeightInferenceHooks)
