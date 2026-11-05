"""Multi-component releases: the manifest, component views, per-component specs, and commits."""

from __future__ import annotations

import shutil
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from tests.support.runtime_stubs import StubInferenceRuntime
from tests.support.serving import wait_for_serving

from reef.artifact import (
    Artifact,
    ArtifactNotFound,
    ArtifactPublicationError,
    DefaultStart,
    InMemoryRepositoryBackend,
    Repository,
)
from reef.artifact.artifact import ArtifactValidator, SnapshotRef
from reef.artifact.components import AcceptAnyArtifact, ComponentSpec, ScenarioComponents, TextFileTree
from reef.artifact.composite import compose_snapshot
from reef.core.activation import ActivationRule
from reef.core.components import COMPONENTS_METADATA_KEY, ComponentEntry, SnapshotComponents, snapshot_components
from reef.core.errors import ReefError
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import AlwaysActivate
from reef.observability import NullExperimentTracker
from reef.recipe.base import Recipe
from reef.scenario.binding import activate_snapshot
from reef.scenario.factory import ScenarioFactory
from reef.service.request_service import RequestService
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import ComponentTrainer
from reef.train.types import TrainingCommit

WEIGHTS = "weights"
HARNESS = "harness"
#: The changed kinds of a step held for review: the test recipe's activation rule does not activate it.
HELD = frozenset({"held"})


def _tree(root: Path, files: Mapping[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text)
    return root


class RecordingRuntime(StubInferenceRuntime):
    """Record the artifact ID of the component content that each activation received."""

    def __init__(self) -> None:
        super().__init__()
        self.activated_ids: list[str] = []

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        self.activated_ids.append(artifact.artifact_id)
        return super().activate_checkpoint(artifact, scenario=scenario)


class _RecordingValidator(ArtifactValidator):
    def __init__(self) -> None:
        self.seen: list[str] = []

    def validate(self, artifact: Artifact) -> None:
        self.seen.append(artifact.artifact_id)


@pytest.mark.unit
def test_manifest_preserves_component_ids_without_a_combination_identity() -> None:
    single = SnapshotComponents({WEIGHTS: ComponentEntry("w1")})
    assert single.single
    assert single.artifact_ids("snapshot") == {WEIGHTS: "w1"}
    assert single.relative_path(WEIGHTS) == Path(".")

    both = SnapshotComponents({WEIGHTS: ComponentEntry("w1"), HARNESS: ComponentEntry("h1", {"note": "seed"})})
    assert not both.single
    assert both.names == (WEIGHTS, HARNESS)
    assert both.artifact_ids("snapshot") == {WEIGHTS: "w1", HARNESS: "h1"}
    reordered = SnapshotComponents({HARNESS: ComponentEntry("h1"), WEIGHTS: ComponentEntry("w1")})
    assert reordered.artifact_ids("other-snapshot") == both.artifact_ids("snapshot")
    assert both.relative_path(HARNESS) == Path(HARNESS)
    assert SnapshotComponents.from_dict(both.to_dict()) == both
    assert snapshot_components({COMPONENTS_METADATA_KEY: both.to_dict()}) == both
    assert snapshot_components(None) is None
    assert snapshot_components({"runtime_load_id": "inc:1"}) is None

    with pytest.raises(KeyError):
        both.relative_path("config")
    with pytest.raises(ValueError, match="directory name"):
        SnapshotComponents({"../escape": ComponentEntry("x")})
    with pytest.raises(ValueError, match="at least one"):
        SnapshotComponents({})


@pytest.mark.unit
def test_component_views_of_a_composed_release(tmp_path: Path) -> None:
    weights = Artifact.local(
        _tree(tmp_path / "w", {"adapter_config.json": "{}"}), metadata={"runtime_load_id": "inc:3"}
    )
    harness = Artifact.local(_tree(tmp_path / "h", {"AGENTS.md": "rules"}))
    composed = compose_snapshot({WEIGHTS: weights, HARNESS: harness}, directory=tmp_path / "release")

    assert composed.components is not None
    assert composed.components.names == (WEIGHTS, HARNESS)
    with pytest.raises(ArtifactNotFound, match="holds components"):
        _ = composed.artifact_id
    view = composed.component(WEIGHTS)
    assert view.local_path == tmp_path / "release" / WEIGHTS
    assert view.artifact_id == weights.artifact_id
    assert view.ref.revision == composed.ref.revision
    assert dict(view.metadata) == {"runtime_load_id": "inc:3"}
    assert TextFileTree().read_files(composed.component(HARNESS)) == {"AGENTS.md": "rules"}
    with pytest.raises(ArtifactNotFound):
        composed.component("config")

    # A snapshot without a manifest is its own only component, whatever the caller names.
    assert harness.component(HARNESS) is harness
    with pytest.raises(ArtifactPublicationError, match="at least two"):
        compose_snapshot({HARNESS: harness}, directory=tmp_path / "single")


@pytest.mark.unit
def test_materialized_releases_keep_their_manifest_and_composed_staging_carries_components(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _tree(root / WEIGHTS, {"adapter_config.json": "{}"})
    _tree(root / HARNESS, {"AGENTS.md": "seed"})
    backend = InMemoryRepositoryBackend("agent", directory=tmp_path / "repository")
    manifest = SnapshotComponents({WEIGHTS: ComponentEntry("w0"), HARNESS: ComponentEntry("h0")})
    root_ref = backend.register(Artifact.local(root), metadata={COMPONENTS_METADATA_KEY: manifest.to_dict()})
    repository = Repository(
        backend, root_ref, current_artifact=root_ref, checkpoint_artifact=root_ref, local_dir=tmp_path / "local"
    )

    materialized = repository.materialize(root_ref)
    assert materialized.components == manifest
    assert TextFileTree().read_files(materialized.component(HARNESS)) == {"AGENTS.md": "seed"}

    evolved = Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "evolved"}))
    staged = repository.stage_composed(
        1, {WEIGHTS: materialized.component(WEIGHTS), HARNESS: evolved}, parent=root_ref
    )
    published = repository.publish(staged, expected_parent=root_ref, metadata=staged.metadata)
    again = repository.materialize(published)
    assert again.components is not None
    assert again.components.entries[WEIGHTS].artifact_id == "w0"
    assert again.components.entries[HARNESS].artifact_id == evolved.artifact_id
    assert again.components.artifact_ids(again.ref.revision) == staged.components.artifact_ids(staged.ref.revision)
    assert TextFileTree().read_files(again.component(HARNESS)) == {"AGENTS.md": "evolved"}
    assert TextFileTree().read_files(again.component(WEIGHTS)) == {"adapter_config.json": "{}"}


@pytest.mark.unit
def test_components_route_each_capability_to_its_component(tmp_path: Path) -> None:
    runtime = RecordingRuntime()
    validator = _RecordingValidator()
    components = ScenarioComponents(
        components={
            WEIGHTS: ComponentSpec(validator=validator, kind="model_adapter"),
            HARNESS: ComponentSpec(files=TextFileTree()),
        }
    )
    assert components.names == (WEIGHTS, HARNESS)
    assert not components.single
    assert components.loaded_component == WEIGHTS
    assert components.files_component == HARNESS

    weights = Artifact.local(_tree(tmp_path / "w", {"adapter_config.json": "{}"}))
    harness = Artifact.local(_tree(tmp_path / "h", {"AGENTS.md": "rules"}))
    first = compose_snapshot({WEIGHTS: weights, HARNESS: harness}, directory=tmp_path / "r1")
    assert components.files_component is not None
    assert components.read_files(first) == {"AGENTS.md": "rules"}

    components.validate(first)
    assert validator.seen == [weights.artifact_id]
    activate_snapshot(components, first, runtime, "agent")
    assert runtime.activated_ids == [weights.artifact_id]


@pytest.mark.unit
def test_components_reject_ambiguous_component_sets() -> None:
    with pytest.raises(ValueError, match="at most one component loads"):
        ScenarioComponents(components={"a": ComponentSpec(kind="model"), "b": ComponentSpec(kind="model_adapter")})
    with pytest.raises(ValueError, match="file tree"):
        ScenarioComponents(
            components={"a": ComponentSpec(files=TextFileTree()), "b": ComponentSpec(files=TextFileTree())}
        )
    with pytest.raises(ValueError, match="directory name"):
        ScenarioComponents(components={"a/b": ComponentSpec()})


class _TwoComponentRecipe(Recipe):
    """A record-only recipe whose scenario serves weights and a harness tree through one trainer."""

    def activation_rule(self) -> ActivationRule:
        return AlwaysActivate(HELD)

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents(
            components={
                WEIGHTS: ComponentSpec(kind="model_adapter"),
                HARNESS: ComponentSpec(files=TextFileTree()),
            }
        )

    def default_start(self) -> DefaultStart:
        return DefaultStart(files={f"{WEIGHTS}/adapter_config.json": "{}", f"{HARNESS}/AGENTS.md": "seed"})

    def build_trainers(
        self,
        scenario,
        records,
        *,
        components,
        algorithm_states,
        experiment_logger=None,
        task_storage=None,
        artifacts=None,
        samples=None,
        root=None,
    ):
        # One trainer, bound to the harness, publishes whichever component a step names.
        return (
            ComponentTrainer(
                HARNESS,
                self.build(
                    scenario,
                    records,
                    algorithm_state=algorithm_states.get(HARNESS),
                    experiment_logger=experiment_logger,
                    root=root,
                ),
            ),
        )


@pytest.mark.unit
def test_multi_component_scenario_commits_one_component_and_carries_the_rest(tmp_path: Path) -> None:
    runtime = RecordingRuntime()
    recipe = _TwoComponentRecipe(runtime=runtime)
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        components = scenario.components
        base = scenario.repository.materialize(scenario.current_snapshot())
        assert base.components is not None
        assert base.components.names == (WEIGHTS, HARNESS)
        base_weights = base.components.entries[WEIGHTS].artifact_id
        wait_for_serving(dispatcher, "agent")
        assert runtime.activated_ids == [base_weights]
        assert components.files_component is not None
        assert components.read_files(base) == {"AGENTS.md": "seed"}

        # A harness step carries the weights forward and does not reload them.
        evolved = Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "evolved"}))
        scenario.commit(TrainingCommit(state={}, artifact=evolved, component=HARNESS))
        after_harness = scenario.repository.materialize(scenario.current_snapshot())
        assert after_harness.components is not None
        assert after_harness.components.entries[WEIGHTS].artifact_id == base_weights
        assert after_harness.components.entries[HARNESS].artifact_id == evolved.artifact_id
        assert components.read_files(after_harness) == {"AGENTS.md": "evolved"}
        assert TextFileTree().read_files(after_harness.component(WEIGHTS)) == {"adapter_config.json": "{}"}
        wait_for_serving(dispatcher, "agent")
        assert runtime.activated_ids == [base_weights]

        # A weights step activates the new weights and keeps the evolved harness.
        trained = Artifact.local(
            _tree(tmp_path / "w1", {"adapter_config.json": '{"r": 8}'}), metadata={"runtime_load_id": "inc:1"}
        )
        scenario.commit(TrainingCommit(state={}, artifact=trained, component=WEIGHTS))
        after_weights = scenario.repository.materialize(scenario.current_snapshot())
        assert after_weights.components is not None
        assert after_weights.components.entries[WEIGHTS].artifact_id == trained.artifact_id
        assert dict(after_weights.component(WEIGHTS).metadata) == {"runtime_load_id": "inc:1"}
        assert components.read_files(after_weights) == {"AGENTS.md": "evolved"}
        wait_for_serving(dispatcher, "agent")
        assert runtime.activated_ids[0] == base_weights
        assert set(runtime.activated_ids[1:]) == {trained.artifact_id}
        # The record names the harness trainer, but the tree did not change: no new harness head.
        service = RequestService(dispatcher)
        headers = {"x-reef-scenario": "agent"}
        assert service.harness_manifest(headers)["artifact_id"] == after_harness.component(HARNESS).artifact_id
        assert [row["artifact_id"] for row in service.harness_artifacts(headers)["artifacts"]] == [
            base.component(HARNESS).artifact_id,
            after_harness.component(HARNESS).artifact_id,
        ]

        # A publication naming no component replaces the committing trainer's own; unknown names are refused.
        scenario.commit(TrainingCommit(state={}, artifact=Artifact.local(_tree(tmp_path / "h2", {"AGENTS.md": "x"}))))
        assert components.read_files(scenario.repository.materialize(scenario.current_snapshot())) == {
            "AGENTS.md": "x"
        }
        with pytest.raises(ReefError, match="serves no component"):
            scenario.commit(TrainingCommit(state={}, artifact=evolved, component="config"))

        # Activating the harness-only step's harness binds it to the weights served now: no weights reload.
        served_before = len(runtime.activated_ids)
        tip = scenario.repository.checkpoint_artifact
        assert tip is not None
        activated = scenario.activate(evolved.artifact_id, HARNESS)
        assert activated.artifact_id == evolved.artifact_id
        served = scenario.repository.materialize(scenario.current_snapshot())
        assert served.component(HARNESS).artifact_id == activated.artifact_id
        assert served.components is not None
        assert served.components.entries[WEIGHTS].artifact_id == trained.artifact_id
        assert components.read_files(served) == {"AGENTS.md": "evolved"}
        wait_for_serving(dispatcher, "agent")
        assert runtime.activated_ids[served_before:] == []
        assert service.harness_manifest(headers)["artifact_id"] == after_harness.component(HARNESS).artifact_id

        # Activating the root's weights loads them and keeps the served harness; the chain tip stays. The root
        # release made its artifacts, so their artifact_id is the root snapshot's.
        base_weights_id = base.components.artifact_ids(base.ref.revision)[WEIGHTS]
        assert base_weights_id == base.component(WEIGHTS).artifact_id
        assert base_weights_id != scenario.repository.start_artifact.revision
        trained_weights = scenario.artifact_status()[WEIGHTS]["training_start"]
        scenario.activate(base_weights_id, WEIGHTS)
        # The activation moves the weights' active artifact; their training start stays on the trained weights.
        assert scenario.artifact_status()[WEIGHTS] == {"active": base_weights_id, "training_start": trained_weights}
        assert trained_weights != base_weights_id
        served = scenario.repository.materialize(scenario.current_snapshot())
        assert served.components is not None
        assert served.components.entries[WEIGHTS].artifact_id == base_weights
        assert components.read_files(served) == {"AGENTS.md": "evolved"}
        wait_for_serving(dispatcher, "agent")
        assert runtime.activated_ids[served_before:] == [base_weights]
        assert scenario.repository.checkpoint_artifact == tip

        # A harness step held for review is made on the chain tip; activating its harness binds it to the weights
        # served now.
        held = Artifact.local(_tree(tmp_path / "h3", {"AGENTS.md": "held"}))
        scenario.commit(TrainingCommit(state={}, artifact=held, component=HARNESS, changed_kinds=HELD))
        assert components.read_files(scenario.repository.materialize(scenario.current_snapshot())) == {
            "AGENTS.md": "evolved"
        }
        held_record = scenario.store.history()[-1]
        held_ref = held_record.snapshot
        assert held_ref.parent_revision == tip.revision
        # The held artifact is the chain tip, the parent of the next publication, but does not serve.
        assert scenario.repository.checkpoint_artifact == held_ref and not held_record.activated
        scenario.activate(held.artifact_id, HARNESS)
        served = scenario.repository.materialize(scenario.current_snapshot())
        assert served.components is not None
        assert served.components.entries[WEIGHTS].artifact_id == base_weights
        assert components.read_files(served) == {"AGENTS.md": "held"}
        assert service.harness_manifest(headers)["artifact_id"] == held.artifact_id

        # Weights held for review are activated as weights, under the tree served by then.
        held_weights = Artifact.local(
            _tree(tmp_path / "w3", {"adapter_config.json": '{"r": 32}'}), metadata={"runtime_load_id": "inc:3"}
        )
        scenario.commit(TrainingCommit(state={}, artifact=held_weights, component=WEIGHTS, changed_kinds=HELD))
        held_weights_snapshot = scenario.store.history()[-1].snapshot.revision
        newest = Artifact.local(_tree(tmp_path / "h4", {"AGENTS.md": "newest"}))
        scenario.commit(TrainingCommit(state={}, artifact=newest, component=HARNESS))
        scenario.activate(held_weights.artifact_id, WEIGHTS)
        served = scenario.repository.materialize(scenario.current_snapshot())
        assert served.components is not None
        assert served.components.entries[WEIGHTS].artifact_id == held_weights.artifact_id
        assert components.read_files(served) == {"AGENTS.md": "newest"}
        wait_for_serving(dispatcher, "agent")
        assert runtime.activated_ids[-1] == held_weights.artifact_id
        # That activation changed no tree: the harness head and catalog stay at the newest harness step.
        assert service.harness_manifest(headers)["artifact_id"] == newest.artifact_id
        catalog = service.harness_artifacts(headers)["artifacts"]
        assert catalog[-1]["artifact_id"] == newest.artifact_id
        assert f"{held_weights_snapshot}:{HARNESS}" not in {row["artifact_id"] for row in catalog}
        # The artifact listing marks each component's active artifact.
        active = {row["component"]: row["artifact_id"] for row in scenario.artifacts() if row["active"]}
        assert active == {WEIGHTS: held_weights.artifact_id, HARNESS: newest.artifact_id}
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_selections_restore_and_recover_without_publishing_artifacts(tmp_path: Path, monkeypatch: Any) -> None:
    repositories = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    with closing(SQLiteScenarioStorage(tmp_path / "store")) as storage:
        factory = ScenarioFactory(
            _TwoComponentRecipe(),
            repositories,
            local_artifact_dir=tmp_path / "staged",
            scenario_storage=storage,
            experiment_tracker=NullExperimentTracker(),
        )
        with closing(factory.load_or_create("agent")) as scenario:
            initial = {name: state["active"] for name, state in scenario.artifact_status().items()}
            scenario.commit(
                TrainingCommit(
                    state={}, artifact=Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "one"})), component=HARNESS
                )
            )
            after_first = {name: state["active"] for name, state in scenario.artifact_status().items()}
            scenario.commit(
                TrainingCommit(
                    state={},
                    artifact=Artifact.local(_tree(tmp_path / "w1", {"adapter_config.json": '{"r":8}'})),
                    component=WEIGHTS,
                )
            )
            weights_id = scenario.artifact_status()[WEIGHTS]["active"]
            scenario.commit(
                TrainingCommit(
                    state={},
                    artifact=Artifact.local(_tree(tmp_path / "h2", {"AGENTS.md": "held"})),
                    component=HARNESS,
                    changed_kinds=HELD,
                )
            )
            starts = {name: state["training_start"] for name, state in scenario.artifact_status().items()}
            checkpoint = scenario.repository.require_checkpoint_artifact()
            artifacts = {row["artifact_id"] for row in scenario.artifacts()}

            def refuse_publication(*args: Any, **kwargs: Any) -> None:
                pytest.fail("selecting existing artifacts must not publish storage")

            publish = scenario.repository.backend.publish
            monkeypatch.setattr(scenario.repository.backend, "publish", refuse_publication)
            scenario.activate(initial[HARNESS], HARNESS)
            assert scenario.artifact_status()[WEIGHTS]["active"] == weights_id
            scenario.serve(after_first[HARNESS], HARNESS)
            assert scenario.restore(1) == after_first
            assert {name: state["training_start"] for name, state in scenario.artifact_status().items()} == starts
            assert {row["artifact_id"] for row in scenario.artifacts()} == artifacts
            assert scenario.repository.require_checkpoint_artifact() == checkpoint
            scenario.commit(TrainingCommit(state={}))
            active = scenario.current_snapshot()
            assert active.selection_view

        # The next process has no materialized views. Only operation records and original artifact bytes survive.
        shutil.rmtree(tmp_path / "staged")
        with closing(factory.load_or_create("agent")) as recovered:
            assert recovered.current_snapshot() == active
            assert recovered.repository.require_checkpoint_artifact() == checkpoint
            assert {name: state["active"] for name, state in recovered.artifact_status().items()} == after_first
            assert {name: state["training_start"] for name, state in recovered.artifact_status().items()} == starts
            served = recovered.served_beside()[after_first[HARNESS]]
            beside = recovered.repository.select(served.selection)
            assert beside.component(WEIGHTS).artifact_id == weights_id
            assert beside.component(HARNESS).artifact_id == after_first[HARNESS]
            assert [record.operation for record in recovered.store.history()][-4:] == [
                "activate",
                "serve",
                "restore",
                "training",
            ]
            # A later held checkpoint stores the current selection independently of the operation history.
            monkeypatch.setattr(recovered.repository.backend, "publish", publish)
            recovered.commit(
                TrainingCommit(
                    state={},
                    artifact=Artifact.local(_tree(tmp_path / "h3", {"AGENTS.md": "also held"})),
                    component=HARNESS,
                    changed_kinds=HELD,
                )
            )
        shutil.rmtree(tmp_path / "staged")
        with closing(SQLiteScenarioStorage(tmp_path / "empty-store")) as empty_storage:
            checkpoint_factory = ScenarioFactory(
                _TwoComponentRecipe(),
                repositories,
                scenario_storage=empty_storage,
                experiment_tracker=NullExperimentTracker(),
            )
            with closing(checkpoint_factory.load_or_create("agent")) as recovered:
                assert recovered.current_snapshot() == active
                assert {name: state["active"] for name, state in recovered.artifact_status().items()} == after_first
                view = recovered.repository.resolve(active)
                assert view.component(WEIGHTS).artifact_id == after_first[WEIGHTS]
                assert view.component(HARNESS).artifact_id == after_first[HARNESS]
                with pytest.raises(ValueError, match="earlier operation history is unavailable"):
                    recovered.restore(recovered.scenario_step)
                # The creation remains a known selection even when the intervening operations were lost.
                assert recovered.restore(0) == initial


@pytest.mark.unit
def test_concurrent_first_readers_share_one_selection_view(tmp_path: Path, monkeypatch: Any) -> None:
    root = compose_snapshot(
        {
            WEIGHTS: Artifact.local(_tree(tmp_path / "weights", {"adapter_config.json": "{}"})),
            HARNESS: Artifact.local(_tree(tmp_path / "harness", {"AGENTS.md": "seed"})),
        },
        directory=tmp_path / "composed",
    )
    backend = InMemoryRepositoryBackend("agent", directory=tmp_path / "repository")
    initial = backend.register(root, metadata=root.metadata)
    repository = Repository(backend, initial, current_artifact=initial)
    selection = repository.selection(initial, (WEIGHTS, HARNESS))
    reference = repository.select(selection).ref
    recovered = Repository(backend, initial, current_artifact=initial)
    recovered.remember_selection(reference, selection)
    entered, duplicate, resume, second_started = (threading.Event() for _ in range(4))

    def compose_once(components: Mapping[str, Artifact], *, directory: Path) -> Artifact:
        if entered.is_set():
            duplicate.set()
        entered.set()
        if not resume.wait(timeout=5):
            raise TimeoutError("test did not release composition")
        return compose_snapshot(components, directory=directory)

    def second_reader() -> Artifact:
        second_started.set()
        return recovered.resolve(reference)

    monkeypatch.setattr("reef.artifact.repository.compose_snapshot", compose_once)
    with ThreadPoolExecutor(max_workers=2) as readers:
        first = readers.submit(recovered.resolve, reference)
        try:
            assert entered.wait(timeout=5)
            second = readers.submit(second_reader)
            assert second_started.wait(timeout=5)
            assert not duplicate.wait(timeout=0.05)
            # Head compare-and-swap is independent of a reader's blocked filesystem work.
            recovered.activate(initial, expected=initial)
        finally:
            resume.set()
        first_view, second_view = first.result(timeout=5), second.result(timeout=5)
    assert first_view is second_view
    assert first_view.component(HARNESS).local_path.joinpath("AGENTS.md").read_text() == "seed"


@pytest.mark.unit
def test_restoring_a_selection_does_not_readmit_unchanged_component_artifacts(tmp_path: Path) -> None:
    class TrainedWeights(ArtifactValidator):
        """A training admission check that refuses the scenario's default seed."""

        def validate(self, artifact: Artifact) -> None:
            if artifact.materialize().local_path.joinpath("adapter_config.json").read_text() == "{}":
                raise ReefError("default weights are not a trained candidate")

    class RestoreRecipe(_TwoComponentRecipe):
        """A harness trainer whose unchanged default weights do not pass training admission."""

        def build_components(self, scenario: str) -> ScenarioComponents:
            return ScenarioComponents(
                components={
                    WEIGHTS: ComponentSpec(validator=TrainedWeights()),
                    HARNESS: ComponentSpec(files=TextFileTree()),
                }
            )

    with closing(SQLiteScenarioStorage(tmp_path / "store")) as storage:
        factory = ScenarioFactory(
            RestoreRecipe(),
            InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
            scenario_storage=storage,
            experiment_tracker=NullExperimentTracker(),
        )
        with closing(factory.load_or_create("agent")) as scenario:
            for step in (1, 2):
                scenario.commit(
                    TrainingCommit(
                        state={},
                        component=HARNESS,
                        artifact=Artifact.local(_tree(tmp_path / f"h{step}", {"AGENTS.md": f"harness {step}"})),
                    )
                )
            selected = scenario.restore(1)
            view = scenario.repository.resolve(scenario.current_snapshot())
            assert view.component(HARNESS).local_path.joinpath("AGENTS.md").read_text() == "harness 1"
            assert selected[WEIGHTS] == scenario.artifact_status()[WEIGHTS]["active"]


@pytest.mark.unit
def test_selection_recovers_after_record_append_before_head_movement(tmp_path: Path, monkeypatch: Any) -> None:
    repositories = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    with closing(SQLiteScenarioStorage(tmp_path / "store")) as storage:
        factory = ScenarioFactory(
            _TwoComponentRecipe(),
            repositories,
            scenario_storage=storage,
            experiment_tracker=NullExperimentTracker(),
        )
        with closing(factory.load_or_create("agent")) as scenario:
            scenario.commit(
                TrainingCommit(
                    state={},
                    artifact=Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "held"})),
                    component=HARNESS,
                    changed_kinds=HELD,
                )
            )
            checkpoint = scenario.repository.require_checkpoint_artifact()
            selected_id = scenario.repository.resolve(checkpoint).component(HARNESS).artifact_id
            previous = scenario.current_snapshot()

            def interrupted(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("process stopped after append")

            monkeypatch.setattr(scenario.repository, "activate", interrupted)
            with pytest.raises(RuntimeError, match="process stopped"):
                scenario.activate(selected_id, HARNESS)
            assert scenario.current_snapshot() == previous
            assert scenario.store.history()[-1].selection is not None
        with closing(factory.load_or_create("agent")) as recovered:
            assert recovered.scenario_step == 2
            assert recovered.artifact_status()[HARNESS]["active"] == selected_id
            assert recovered.repository.require_checkpoint_artifact() == checkpoint
            assert (
                recovered.repository.resolve(recovered.current_snapshot()).component(HARNESS).artifact_id
                == selected_id
            )


@pytest.mark.unit
def test_a_held_release_activates_without_the_manifest_of_its_parent(tmp_path: Path) -> None:
    """An activation reads the active selection and the artifact's release only: the creation's bytes may be gone."""
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        components = scenario.components
        assert components.files_component is not None
        creation = scenario.repository.materialize(scenario.current_snapshot())
        held = Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "held"}))
        scenario.commit(TrainingCommit(state={}, artifact=held, component=HARNESS, changed_kinds=HELD))
        later = Artifact.local(
            _tree(tmp_path / "w1", {"adapter_config.json": '{"r": 8}'}), metadata={"runtime_load_id": "inc:1"}
        )
        scenario.commit(TrainingCommit(state={}, artifact=later, component=WEIGHTS))
        assert creation.local_path is not None
        shutil.rmtree(creation.local_path)
        scenario.activate(held.artifact_id, HARNESS)
        served = scenario.repository.materialize(scenario.current_snapshot())
        assert served.components is not None
        assert served.components.entries[WEIGHTS].artifact_id == later.artifact_id
        assert components.read_files(served) == {"AGENTS.md": "held"}
        assert scenario.scenario_step == 3
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_an_activation_after_a_rejected_harness_step_adds_no_harness_release(tmp_path: Path) -> None:
    """The tree a rejected step served is the last one named; activating the served artifact changes nothing."""
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        scenario.commit(
            TrainingCommit(
                state={}, artifact=Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "one"})), component=HARNESS
            )
        )
        first_harness_id = scenario.repository.resolve(scenario.current_snapshot()).component(HARNESS).artifact_id
        weights = Artifact.local(
            _tree(tmp_path / "w1", {"adapter_config.json": '{"r": 8}'}), metadata={"runtime_load_id": "inc:1"}
        )
        scenario.commit(TrainingCommit(state={}, artifact=weights, component=WEIGHTS))
        scenario.commit(TrainingCommit(state={}, metrics={"selected": False}), component=HARNESS)
        steps = scenario.scenario_step
        # The first harness artifact serves already: activating it changes nothing.
        assert scenario.activate(first_harness_id, HARNESS).artifact_id == first_harness_id
        assert scenario.scenario_step == steps
        rows = scenario.harness_rows()
        assert next(row["artifact_id"] for row in rows if row["active"]) == first_harness_id
        assert all("release_id" not in row and "snapshot" not in row for row in rows)
        # Newest first: the rejected step's row (named by the first harness artifact), that snapshot, the creation.
        assert [row["operation"] for row in rows] == ["training", "training", "creation"]
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_an_activation_refused_by_validation_changes_nothing(tmp_path: Path, monkeypatch: Any) -> None:
    staged_root = tmp_path / "staged"
    dispatcher = Dispatcher(
        _TwoComponentRecipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=staged_root,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "store"),
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        assert scenario is not None
        held = Artifact.local(_tree(tmp_path / "h1", {"AGENTS.md": "held"}))
        scenario.commit(TrainingCommit(state={}, artifact=held, component=HARNESS, changed_kinds=HELD))
        served = scenario.current_snapshot()
        before = {path for path in staged_root.rglob("*") if path.is_dir()}

        def refuse(self: ArtifactValidator, artifact: Artifact) -> None:
            raise ReefError("refused by the validator")

        # An activation runs the snapshot's own check, as its publication did.
        monkeypatch.setattr(AcceptAnyArtifact, "validate", refuse)
        with pytest.raises(ReefError, match="refused by the validator"):
            scenario.activate(held.artifact_id, HARNESS)
        assert {path for path in staged_root.rglob("*") if path.is_dir()} == before
        assert scenario.scenario_step == 1
        assert scenario.current_snapshot() == served
    finally:
        dispatcher.close()


def _seedless_snapshot(tmp_path: Path) -> Artifact:
    release = tmp_path / "release"
    (release / "harness").mkdir(parents=True)
    (release / "harness" / "AGENTS.md").write_text("seed", encoding="utf-8")
    manifest = {
        "harness": {"artifact_id": "artifact:harness", "metadata": {}},
        "weights": {"artifact_id": "artifact:weights", "metadata": {}},
    }
    return Artifact(
        SnapshotRef("rel-1", None),
        None,
        local_path=release,
        metadata={COMPONENTS_METADATA_KEY: manifest},
    )


@pytest.mark.unit
def test_two_readers_of_a_seedless_component_both_get_its_directory(tmp_path: Path, monkeypatch: Any) -> None:
    """Two inference threads read one snapshot after a harness commit: both find the weights leaf missing and
    the second must not fail on the directory the first made."""
    artifact = _seedless_snapshot(tmp_path)
    barrier = threading.Barrier(2)
    original = Path.mkdir

    def paced(self: Path, *args: Any, **kwargs: Any) -> None:
        # Both threads are past the existence check when they get here.
        barrier.wait(timeout=5)
        original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", paced)
    errors: list[BaseException] = []

    def read() -> None:
        try:
            artifact.component("weights")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert [repr(error) for error in errors] == []
    assert artifact.component("weights").local_path == tmp_path / "release" / "weights"
