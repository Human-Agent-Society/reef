"""Per-scenario adapter routing on a shared-base training runtime."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingRuntime

from reef.artifact import Artifact, ArtifactRef, InMemoryRepositoryBackend, LiveWeightArtifactRef
from reef.core.errors import ReefError
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import adapter_name, create_weight_surface
from reef.surface.base import ComponentSurface, Surface, WeightRuntime
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
    runtime = Runtime().inference
    assert WeightLoader("math").recover(current, checkpoint_ref, runtime) == current
    assert WeightLoader().recover(current, checkpoint_ref, runtime) == checkpoint_ref
    # The engine holds no adapter for code: its live head is unservable, so
    # serving falls back to the exact checkpoint instead of routing to a
    # name the engine would reject.
    assert WeightLoader("code").recover(current, checkpoint_ref, runtime) == checkpoint_ref
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

    class Runtime(StubTrainingRuntime, WeightRuntime):
        def restore_recovered_checkpoint(self, artifact):
            return self.restore_checkpoint(artifact.materialize())

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

    class Runtime(StubTrainingRuntime, WeightRuntime):
        def restore_recovered_checkpoint(self, artifact):
            return self.restore_checkpoint(artifact.materialize())

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

    from reef.artifact.artifact import ArtifactMaterializationError
    from reef.artifact.git_lfs import materialized_metadata

    destination = tmp_path / "cached"
    destination.mkdir()
    (destination / "reef-artifact.json").write_text(
        json.dumps({"content_id": "c", "metadata": {"runtime_load_id": "mlx-111-40"}})
    )
    assert materialized_metadata(destination) == {"runtime_load_id": "mlx-111-40"}

    # A directory with no manifest is a bootstrap tree, which records nothing;
    # an unreadable manifest is an error, not an unknown version.
    assert materialized_metadata(tmp_path / "missing") == {}
    (destination / "reef-artifact.json").write_text("{not json")
    with pytest.raises(ArtifactMaterializationError, match="invalid artifact manifest"):
        materialized_metadata(destination)


@pytest.mark.parametrize("stale_head", [False, True], ids=["current-head", "repair-head-mirror"])
def test_scenario_startup_restores_materialized_checkpoint_into_fresh_runtime(
    tmp_path: Path, stale_head: bool
) -> None:
    from reef.storage.commits import CommitRecord

    class MaterializingBackend(InMemoryRepositoryBackend):
        def materialize(self, ref):
            artifact = super().materialize(ref)
            return Artifact(ref, None, local_path=artifact.local_path, metadata=self._storage.metadata[ref.release_id])

    class WeightSurfaceRecipe(Recipe):
        def build_surface(self, scenario):
            return create_weight_surface()

    restored: list[tuple[Path, str]] = []

    class FreshRuntime(StubInferenceRuntime, WeightRuntime):
        def restore_recovered_checkpoint(self, artifact):
            return self.restore_checkpoint(artifact.materialize())

        def serving_runtime_load_id(self):
            return "mlx-fresh-1"

        def restore_checkpoint(self, artifact):
            restored.append((artifact.local_path, artifact.metadata["runtime_load_id"]))
            return "mlx-fresh-2"

    initial = tmp_path / "initial"
    initial.mkdir()
    backend_factory = MaterializingBackend.factory(initial, root=tmp_path / "repository")
    first = Dispatcher(
        WeightSurfaceRecipe(), backend_factory, scenario_storage=SQLiteScenarioStorage(tmp_path / "records")
    )
    first.get_or_create_scenario("math")
    first.close()

    backend = backend_factory("math")
    staged = checkpoint(tmp_path, "mlx-previous-40")
    checkpoint_artifact = Artifact.local(
        staged.local_path,
        metadata={**backend.metadata(), "runtime_load_id": "mlx-previous-40"},
    )
    previous = backend.current()
    published = backend.publish(checkpoint_artifact, expected_parent=previous, advance_head=not stale_head)
    if stale_head:
        storage = SQLiteScenarioStorage(tmp_path / "records")
        store = storage.open("math")
        try:
            store.commit_step(
                expected_step=0,
                commit=CommitRecord(
                    scenario="math",
                    step=1,
                    artifact_ref=published,
                    checkpoint=True,
                    algorithm_state=None,
                    high_water_sequence=0,
                    high_water_offset=0,
                ),
            )
        finally:
            store.close()
            storage.close()
        assert backend.current() == previous
        assert "runtime_load_id" not in backend.metadata()

    runtime = FreshRuntime(StubTrainingRuntime(), base_url="http://fresh-runtime")
    second = Dispatcher(
        WeightSurfaceRecipe(runtime=runtime),
        backend_factory,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )
    try:
        scenario = second.get_or_create_scenario("math")
        assert scenario.repository.require_current_artifact() == published
        assert backend.current() == published
        assert restored == [(backend.materialize(published).local_path, "mlx-previous-40")]
    finally:
        second.close()


@pytest.mark.parametrize("served", ["engine:0", None])
def test_recovery_does_not_reload_a_runtime_using_default_startup_hook(tmp_path: Path, served: str | None) -> None:
    class Runtime(StubInferenceRuntime):
        def serving_runtime_load_id(self):
            return served

        def restore_checkpoint(self, artifact):
            raise AssertionError("startup recovery must not invoke ordinary rollback")

    runtime = Runtime(StubTrainingRuntime(), base_url="http://runtime")
    assert WeightLoader().restore_recovered(checkpoint(tmp_path, "old-incarnation:4"), runtime) is None


def test_recovery_reloads_an_opted_in_runtime_with_unknown_version(tmp_path: Path) -> None:
    restored: list[Path] = []

    class Runtime(StubInferenceRuntime):
        def serving_runtime_load_id(self):
            return None

        def restore_recovered_checkpoint(self, artifact):
            loaded = artifact.materialize()
            restored.append(loaded.local_path)
            return "fresh:1"

    runtime = Runtime(StubTrainingRuntime(), base_url="http://runtime")
    artifact = checkpoint(tmp_path, "old-incarnation:4")
    assert WeightLoader().restore_recovered(artifact, runtime) == "fresh:1"
    assert restored == [artifact.local_path]


@pytest.mark.parametrize("restored_version", ["", 3])
def test_recovery_refuses_an_invalid_runtime_load_id(tmp_path: Path, restored_version) -> None:
    class Runtime(StubInferenceRuntime, WeightRuntime):
        def serving_runtime_load_id(self):
            return "new-incarnation:0"

        def restore_recovered_checkpoint(self, artifact):
            return restored_version

        def restore_checkpoint(self, artifact):
            raise AssertionError("startup recovery must use its startup hook")

    runtime = Runtime(StubTrainingRuntime(), base_url="http://runtime")
    with pytest.raises(TypeError, match="non-empty runtime load ID"):
        WeightLoader().restore_recovered(checkpoint(tmp_path, "old-incarnation:4"), runtime)


@pytest.mark.parametrize(
    ("opted_in", "recorded", "served", "recovery_needed"),
    [
        (False, "previous:4", "fresh:0", False),
        (True, "previous:4", "previous:4", False),
        (False, "previous:4", None, False),
        (True, "previous:4", None, True),
        (True, None, "fresh:0", False),
    ],
    ids=[
        "opted-out",
        "matching-version",
        "opted-out-unknown-version",
        "unknown-runtime-needs-recovery",
        "unknown-recorded-version",
    ],
)
def test_startup_downloads_checkpoint_only_when_runtime_needs_recovery(
    tmp_path: Path, opted_in: bool, recorded: str | None, served: str | None, recovery_needed: bool
) -> None:
    from reef.artifact.artifact import ArtifactMaterializationError

    class UnavailableBackend(InMemoryRepositoryBackend):
        checkpoint_unavailable = False

        def materialize(self, ref):
            if self.checkpoint_unavailable:
                raise ArtifactMaterializationError("checkpoint download is unavailable")
            artifact = super().materialize(ref)
            return Artifact(ref, None, local_path=artifact.local_path, metadata=self._storage.metadata[ref.release_id])

    class WeightSurfaceRecipe(Recipe):
        def build_surface(self, scenario):
            return create_weight_surface()

    class Runtime(StubInferenceRuntime):
        def serving_runtime_load_id(self):
            return served

        def restore_checkpoint(self, artifact):
            raise AssertionError("unnecessary startup recovery must not load weights")

    class RecoveringRuntime(Runtime, WeightRuntime):
        def restore_recovered_checkpoint(self, artifact):
            return self.restore_checkpoint(artifact.materialize())

    initial = tmp_path / "initial"
    initial.mkdir()
    factory = UnavailableBackend.factory(initial, root=tmp_path / "repository")
    first = Dispatcher(WeightSurfaceRecipe(), factory, scenario_storage=SQLiteScenarioStorage())
    first.get_or_create_scenario("math")
    first.close()
    backend = factory("math")
    staged = checkpoint(tmp_path, recorded)
    published = backend.publish(
        Artifact.local(staged.local_path, metadata={**backend.metadata(), **staged.metadata}),
        expected_parent=backend.current(),
    )
    backend.checkpoint_unavailable = True
    runtime_type = RecoveringRuntime if opted_in else Runtime
    runtime = runtime_type(StubTrainingRuntime(), base_url="http://runtime")
    second = Dispatcher(WeightSurfaceRecipe(runtime=runtime), factory, scenario_storage=SQLiteScenarioStorage())
    try:
        if recovery_needed:
            with pytest.raises(ArtifactMaterializationError, match="download is unavailable"):
                second.get_or_create_scenario("math")
            assert second.loaded_scenario("math") is None
        else:
            scenario = second.get_or_create_scenario("math")
            assert scenario.repository.require_current_artifact() == published
            assert runtime.serving_runtime_load_id() == served
    finally:
        second.close()


@pytest.mark.parametrize("composite", [False, True], ids=["flat", "composed"])
@pytest.mark.parametrize("mode", ["default", "matching", "unknown", "restore"])
def test_surface_startup_keeps_component_bytes_lazy(tmp_path: Path, composite: bool, mode: str) -> None:
    from reef.artifact.artifact import ArtifactRepository
    from reef.core.components import ComponentEntry, ReleaseComponents

    root = tmp_path / "release"
    weights = root / "weights" if composite else root
    weights.mkdir(parents=True)
    (weights / "adapter.txt").write_text("trained weights")
    entries = {"weights": ComponentEntry("weights:1", {"runtime_load_id": "previous:4"})}
    if composite:
        entries["harness"] = ComponentEntry("harness:1")
    manifest = ReleaseComponents(entries)
    metadata = {
        "components": manifest.to_dict(),
        "runtime_load_id": "outer:99" if composite else "previous:4",
    }
    ref = ArtifactRef(manifest.content_id, "release:1", None)
    materialized: list[ArtifactRef] = []
    restored: list[Path] = []

    class Backend(ArtifactRepository):
        def materialize(self, requested):
            materialized.append(requested)
            if mode not in {"restore", "unknown"}:
                raise AssertionError("unnecessary recovery downloaded component bytes")
            return Artifact(requested, None, local_path=root, metadata=metadata)

    class Runtime(StubInferenceRuntime):
        def serving_runtime_load_id(self):
            return "previous:4" if mode == "matching" else None if mode == "unknown" else "fresh:0"

        def restore_checkpoint(self, artifact):
            raise AssertionError("startup must not invoke ordinary rollback")

    class RecoveringRuntime(Runtime):
        def restore_recovered_checkpoint(self, artifact):
            loaded = artifact.materialize()
            assert loaded.metadata["runtime_load_id"] == "previous:4"
            assert (loaded.local_path / "adapter.txt").read_text() == "trained weights"
            restored.append(loaded.local_path)
            return "fresh:1"

    components = {"weights": ComponentSurface(loader=WeightLoader())}
    if composite:
        components["harness"] = ComponentSurface()
    surface = Surface(components=components)
    runtime_type = Runtime if mode == "default" else RecoveringRuntime
    runtime = runtime_type(StubTrainingRuntime(), base_url="http://runtime")
    artifact = Artifact(ref, Backend(), metadata=metadata)

    surface.restore_recovered(artifact, runtime)
    surface.activate(artifact, runtime)

    assert restored == ([weights] if mode in {"restore", "unknown"} else [])
    assert materialized == ([ref] if mode in {"restore", "unknown"} else [])


def test_startup_refuses_metadata_from_a_concurrently_advanced_head(tmp_path: Path) -> None:
    from reef.artifact.artifact import ArtifactConflict
    from reef.artifact.repository import RepositoryBackendFactory
    from reef.storage.commits import CommitRecord

    class AdvancingBackend(InMemoryRepositoryBackend):
        metadata_reads = 0
        next_artifact = None

        def metadata(self):
            if self.next_artifact is not None:
                self.metadata_reads += 1
                if self.metadata_reads == 2:
                    self.publish(self.next_artifact, expected_parent=self.current())
                    self.next_artifact = None
            return super().metadata()

    class WeightRecipe(Recipe):
        def build_surface(self, scenario):
            return create_weight_surface()

    calls: list[str] = []

    class Runtime(StubInferenceRuntime):
        def serving_runtime_load_id(self):
            return "engine:2"

        def restore_recovered_checkpoint(self, artifact):
            calls.append("restore")
            return "engine:loaded"

        def activate_checkpoint(self, artifact):
            calls.append("activate")
            return artifact.ref.release_id

    initial = tmp_path / "initial"
    initial.mkdir()
    factory = AdvancingBackend.factory(initial, root=tmp_path / "repository")
    records = tmp_path / "records"
    first = Dispatcher(WeightRecipe(), factory, scenario_storage=SQLiteScenarioStorage(records))
    first.get_or_create_scenario("math")
    first.close()
    backend = factory("math")
    artifact = Artifact.local(initial, metadata={**backend.metadata(), "runtime_load_id": "engine:1"})
    published = backend.publish(artifact, expected_parent=backend.current())
    storage = SQLiteScenarioStorage(records)
    store = storage.open("math")
    store.commit_step(
        expected_step=0,
        commit=CommitRecord(
            scenario="math",
            step=1,
            artifact_ref=published,
            checkpoint=True,
            algorithm_state=None,
            high_water_sequence=0,
            high_water_offset=0,
        ),
    )
    store.close()
    storage.close()
    backend.next_artifact = Artifact.local(initial, metadata={**backend.metadata(), "runtime_load_id": "engine:2"})

    class ExplicitFactory(RepositoryBackendFactory):
        def __call__(self, scenario):
            return factory(scenario)

    runtime = Runtime(StubTrainingRuntime(), base_url="http://runtime")
    second = Dispatcher(
        WeightRecipe(runtime=runtime), ExplicitFactory(), scenario_storage=SQLiteScenarioStorage(records)
    )
    try:
        with pytest.raises(ArtifactConflict, match="metadata"):
            second.get_or_create_scenario("math")
        assert second.loaded_scenario("math") is None
        assert calls == []
    finally:
        second.close()


@pytest.mark.parametrize("checkpoint_has_manifest", [True, False], ids=["missing-weights", "missing-manifest"])
def test_startup_refuses_checkpoint_missing_the_weight_component(
    tmp_path: Path, checkpoint_has_manifest: bool
) -> None:
    from reef.artifact.artifact import ArtifactNotFound
    from reef.artifact.repository import RepositoryBackendFactory
    from reef.core.components import ComponentEntry, ReleaseComponents
    from reef.storage.commits import SCENARIO_METADATA_KEY, CommitRecord, scenario_metadata_for

    class WeightRecipe(Recipe):
        def build_surface(self, scenario):
            return Surface(
                components={
                    "weights": ComponentSurface(loader=WeightLoader()),
                    "harness": ComponentSurface(),
                }
            )

    calls: list[str] = []

    class Runtime(StubInferenceRuntime):
        def serving_runtime_load_id(self):
            return "fresh:0"

        def restore_recovered_checkpoint(self, artifact):
            calls.append("restore")
            return "fresh:1"

        def activate_checkpoint(self, artifact):
            calls.append("activate")
            return artifact.ref.release_id

    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    backend = factory("math")
    base = backend.resolve_release()
    manifest = ReleaseComponents({"weights": ComponentEntry("weights:1"), "harness": ComponentEntry("harness:1")})
    previous = backend.fork(
        base.release_id,
        metadata={
            SCENARIO_METADATA_KEY: scenario_metadata_for(name="math", base_artifact=base),
            "components": manifest.to_dict(),
        },
    )
    checkpoint_metadata = dict(backend.metadata())
    if checkpoint_has_manifest:
        broken = ReleaseComponents({"harness": ComponentEntry("harness:2")})
        checkpoint_metadata["components"] = broken.to_dict()
    else:
        checkpoint_metadata.pop("components")
    published = backend.publish(
        Artifact.local(initial, metadata=checkpoint_metadata),
        expected_parent=previous,
        advance_head=False,
    )
    records = tmp_path / "records"
    storage = SQLiteScenarioStorage(records)
    store = storage.open("math")
    store.commit_step(
        expected_step=0,
        commit=CommitRecord(
            scenario="math",
            step=1,
            artifact_ref=published,
            checkpoint=True,
            algorithm_state=None,
            high_water_sequence=0,
            high_water_offset=0,
        ),
    )
    store.close()
    storage.close()
    assert backend.current() == previous

    class ExplicitFactory(RepositoryBackendFactory):
        def __call__(self, scenario):
            return factory(scenario)

    runtime = Runtime(StubTrainingRuntime(), base_url="http://runtime")
    dispatcher = Dispatcher(
        WeightRecipe(runtime=runtime), ExplicitFactory(), scenario_storage=SQLiteScenarioStorage(records)
    )
    try:
        with pytest.raises(ArtifactNotFound, match="no component 'weights'"):
            dispatcher.get_or_create_scenario("math")
        assert backend.current() == published
        assert dispatcher.loaded_scenario("math") is None
        assert calls == []
    finally:
        dispatcher.close()
