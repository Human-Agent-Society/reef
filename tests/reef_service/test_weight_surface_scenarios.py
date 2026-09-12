"""Per-scenario adapter routing on a shared-base training runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime

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


def test_live_requests_route_to_the_scenario_publication() -> None:
    hooks = WeightInferenceHooks(scenario="math")
    out = hooks.prepare_request(live("inc:7"), "/v1/chat/completions", {"messages": []})
    assert out["lora_path"] == adapter_name("math", "inc:7")
    assert out["return_meta_info"] is True
    with pytest.raises(ReefError, match="asked for lora_path"):
        hooks.prepare_request(live("inc:7"), "/v1/chat/completions", {"lora_path": adapter_name("code", "inc:7")})


def test_checkpoint_requests_route_by_recorded_runtime_load_id(tmp_path: Path) -> None:
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
    assert WeightLoader("math").recover(current, checkpoint_ref, Runtime()) == current
    assert WeightLoader().recover(current, checkpoint_ref, Runtime()) == checkpoint_ref
    # The engine holds no adapter for code: its live head is unservable, so
    # serving falls back to the exact checkpoint instead of routing to a
    # name the engine would reject.
    assert WeightLoader("code").recover(current, checkpoint_ref, Runtime()) == checkpoint_ref
    surface = create_weight_surface(scenario="math")
    assert isinstance(surface.loader, WeightLoader) and isinstance(surface.inference, WeightInferenceHooks)


def test_a_restarted_engine_gets_the_recovered_head_loaded_back(tmp_path: Path) -> None:
    """Recovery decides; activation has to act on the decision.

    A runtime that keeps its weights inside the Reef process loses them when
    that process exits. `recover` already spots that — the engine reports a
    runtime load ID from a new incarnation — and falls serving back to the
    checkpoint. Before this, nothing then put the checkpoint into the engine:
    the scenario reported its full step count and kept training from the bare
    base model, with no record anywhere that it had.
    """
    restored: list[str] = []

    class Runtime(StubTrainingRuntime):
        def serving_runtime_load_id(self):
            return "mlx-222-1"  # a fresh process: counter back at one

        def restore_checkpoint(self, artifact):
            restored.append(str(artifact.local_path))
            return "mlx-222-2"

    current = LiveWeightArtifactRef(
        content_id="live:x", release_id="live:p:40", parent_release_id=None, runtime_load_id="mlx-111-40"
    )
    ckpt = checkpoint(tmp_path, "mlx-111-40")
    loader = WeightLoader()

    assert loader.recover(current, ckpt.ref, Runtime()) == ckpt.ref
    assert loader.restore_recovered(ckpt, Runtime()) == "mlx-222-2"
    assert restored == [str(ckpt.local_path)]

    # A runtime that now serves it needs nothing more.
    class Loaded(Runtime):
        def serving_runtime_load_id(self):
            return "mlx-111-40"

    assert loader.restore_recovered(ckpt, Loaded()) is None
    assert len(restored) == 1


def test_an_artifact_with_no_recorded_version_is_left_alone(tmp_path: Path) -> None:
    # An unknown published version is not a sign of a stale engine, and a
    # runtime that reports none of its own cannot be compared against.
    class Runtime(StubTrainingRuntime):
        def serving_runtime_load_id(self):
            return "mlx-111-40"

        def restore_checkpoint(self, artifact):  # pragma: no cover - must not run
            raise AssertionError("a publication must not reload weights from disk")

    assert WeightLoader().restore_recovered(checkpoint(tmp_path, None), Runtime()) is None


def test_a_matching_engine_keeps_serving_the_live_head(tmp_path: Path) -> None:
    # Same process, same weights: recovery leaves the live head in place and
    # activation stays out of the way.
    class Runtime(StubTrainingRuntime):
        def serving_runtime_load_id(self):
            return "mlx-111-40"

        def restore_checkpoint(self, artifact):  # pragma: no cover - must not run
            raise AssertionError("an unchanged engine must not be reloaded")

    current = LiveWeightArtifactRef(
        content_id="live:x", release_id="live:p:40", parent_release_id=None, runtime_load_id="mlx-111-40"
    )
    loader = WeightLoader()
    assert loader.recover(current, ArtifactRef("ckpt", "c0", None), Runtime()) == current
    assert loader.restore_recovered(checkpoint(tmp_path, "mlx-111-40"), Runtime()) is None


def test_a_head_that_is_its_own_checkpoint_still_gets_restored(tmp_path: Path) -> None:
    """`checkpoint_every_n_versions: 1` makes every publication a checkpoint.

    The head's release id then equals the checkpoint's, and recovery used to
    return on that alone — before asking whether the engine still held those
    weights. That is not an edge case: for any deployment checkpointing every
    version it is the only case, and it silently served the bare base model
    after every restart while reporting the full step count.
    """
    restored: list[str] = []

    class Runtime(StubTrainingRuntime):
        def serving_runtime_load_id(self):
            return "mlx-222-1"

        def restore_checkpoint(self, artifact):
            restored.append(str(artifact.local_path))
            return "mlx-222-2"

    same_release = "live:p:40"
    current = LiveWeightArtifactRef(
        content_id="live:x", release_id=same_release, parent_release_id=None, runtime_load_id="mlx-111-40"
    )
    ckpt = checkpoint(tmp_path, "mlx-111-40")
    loader = WeightLoader()

    # Head and checkpoint are one release, so `recover` short-circuits...
    assert loader.recover(current, ArtifactRef("ckpt", same_release, None), Runtime()).release_id == same_release
    # ...but the question it short-circuits past has already been answered.
    assert loader.restore_recovered(ckpt, Runtime()) == "mlx-222-2"
    assert restored == [str(ckpt.local_path)]


def test_a_materialized_artifact_carries_the_version_it_was_published_under(tmp_path: Path) -> None:
    """Materializing checks out the manifest beside the bytes; returning only
    the bytes made every caller that needed the record see an empty mapping.

    Restoration after a restart depends entirely on this: with no recorded
    version there is nothing to compare the engine against, so the check
    passes vacuously and the stale engine keeps serving.
    """
    import json

    from reef.artifact.git_lfs import _cached_metadata

    destination = tmp_path / "cached"
    destination.mkdir()
    (destination / "reef-artifact.json").write_text(
        json.dumps({"content_id": "c", "metadata": {"runtime_load_id": "mlx-111-40"}})
    )
    assert _cached_metadata(destination) == {"runtime_load_id": "mlx-111-40"}

    # A directory with no manifest, or an unreadable one, is not an error:
    # older caches predate it and the caller treats absence as "unknown".
    assert _cached_metadata(tmp_path / "missing") == {}
    (destination / "reef-artifact.json").write_text("{not json")
    assert _cached_metadata(destination) == {}
