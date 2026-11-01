from __future__ import annotations

import importlib.util
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent, runtime_bindings

from reef.artifact import Artifact, DefaultStart, DirectorySource, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.artifact_ref import RootSource
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.sync import create_harness_surface
from reef.recipe import CompositeRecipe, Recipe
from reef.recipe.harness import HarnessLayout
from reef.runtime.jobs import ScenarioRoot, TrainingComponent
from reef.scenario.factory import ItemStart
from reef.scenario.registry import InstanceState
from reef.storage.commits import SCENARIO_METADATA_KEY
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import Surface
from reef.train.jobs import JobLoopState


def test_dispatcher_constructor_has_no_redundant_scenario_binding_stores() -> None:
    parameters = inspect.signature(Dispatcher).parameters

    assert "recipe_store" not in parameters
    assert "release_id_store" not in parameters
    assert "recipe_resolver" not in parameters
    assert "recipe_validator" not in parameters
    assert "checkpoint_strategy" not in parameters


def test_artifact_state_and_metadata_are_not_parallel_public_types() -> None:
    assert importlib.util.find_spec("reef.scenarios") is None


def test_scenario_owns_recipe_derived_policy_and_its_root_release() -> None:
    scenario = build_default_dispatcher(
        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    ).get_or_create_scenario("math")

    assert not hasattr(scenario, "recipe")
    # A new scenario is at its root release: a release with no parent, made from the recipe's default start.
    root = scenario.repository.start_artifact
    assert root.parent_release_id is None and root.content_id == DefaultStart().content_id
    assert scenario.repository.current_artifact == scenario.repository.checkpoint_artifact == root


def test_scenario_step_is_owned_by_scenario() -> None:
    scenario = build_default_dispatcher(
        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    ).get_or_create_scenario("math")

    assert not hasattr(scenario.repository.start_artifact, "scenario")
    assert scenario.scenario_step == 0
    assert scenario.repository.current_artifact.release_id
    assert scenario.repository.checkpoint_artifact.release_id
    assert not hasattr(scenario, "release_id")
    assert not hasattr(scenario, "selected_release_id")


def test_scenario_owns_scenario_scoped_repository() -> None:
    scenario = build_default_dispatcher(
        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    ).get_or_create_scenario("math")

    repository = scenario.repository

    assert not hasattr(repository, "scenario")
    assert not hasattr(repository.start_artifact, "scenario")
    assert scenario.scenario_step == 0
    assert repository.checkpoint_artifact == repository.current_artifact
    assert not hasattr(scenario, "base_artifact")
    assert not hasattr(scenario, "current_artifact")
    assert not hasattr(scenario, "checkpoint_artifact")


def test_scenario_metadata_round_trips() -> None:
    from reef.storage.commits import parse_scenario_metadata, scenario_metadata_for

    scenario = build_default_dispatcher(
        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    ).get_or_create_scenario("math")
    source = RootSource("default", inputs={"seed": "s"})
    metadata = scenario_metadata_for(name=scenario.name, source=source, scenario_step=scenario.scenario_step)

    assert metadata["format"] == "reef-scenario/5"
    # The registration is carried by the root release, so it names no start_artifact.
    assert "start_artifact" not in metadata
    assert metadata["source"] == {"kind": "default", "inputs": {"seed": "s"}}
    assert metadata["scenario_step"] == 0
    assert metadata["operation"] == "training"
    assert "recipe" not in metadata
    checkpoint_head = scenario.repository.require_current_artifact()
    assert parse_scenario_metadata(metadata, checkpoint_head=checkpoint_head) == (
        "math",
        checkpoint_head,
        source,
        None,
    )

    # Metadata written before recipe identity was removed remain readable;
    # the deployment recipe now supplies those capabilities during recovery.
    assert parse_scenario_metadata(
        {**metadata, "recipe": "legacy-name"}, checkpoint_head=checkpoint_head
    ) == parse_scenario_metadata(metadata, checkpoint_head=checkpoint_head)


@pytest.mark.parametrize("operation", ["activate", "train_from", "rollback"])
def test_checkpoint_metadata_names_only_a_training_step(operation) -> None:
    """An artifact is made only by a training step; its metadata names no other operation."""
    from reef.core.artifact_ref import ArtifactRef
    from reef.storage.commits import parse_scenario_metadata, scenario_metadata_for

    base = ArtifactRef("base", "base", None)
    metadata = scenario_metadata_for(name="math", source=RootSource("default"), start_artifact=base, scenario_step=1)
    metadata["operation"] = operation
    metadata["record_progress"] = {"high_water_sequence": 0, "high_water_offset": 0, "consumed_ids": []}
    with pytest.raises(ValueError, match="operation must be 'training'"):
        parse_scenario_metadata(metadata, checkpoint_head=ArtifactRef("head", "head", base.release_id))


def test_dispatcher_restores_agent_record_from_configured_directory(tmp_path) -> None:
    backend = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"
    first = build_default_dispatcher(
        backend_factory=backend,
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
        coordination=memory_coordination(),
    )
    record = AgentRecord.create(
        agent_record_id="persisted",
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": 1.0},
        created_at=1.0,
    )
    first.get_or_create_scenario("math").records.append(record)

    second = build_default_dispatcher(
        backend_factory=backend,
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
        coordination=memory_coordination(),
    )

    assert second.get_or_create_scenario("math").records.replay("math") == (record,)


def test_dispatcher_restores_algorithm_state_from_artifact_metadata(tmp_path) -> None:
    backend = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"
    first = build_default_dispatcher(
        backend_factory=backend,
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
        coordination=memory_coordination(),
    )
    first.accept_record(
        AgentRecord.create(
            agent_record_id="first",
            scenario="math",
            request_type=RequestType.REPORT,
            payload={"score": 1.0},
            created_at=1.0,
        ),
    )
    # recipe never trains, so step stays at 0 and records accumulate.
    assert first.get_or_create_scenario("math").trainer.state == {}

    second = build_default_dispatcher(
        backend_factory=backend,
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
        coordination=memory_coordination(),
    )
    recovered = second.get_or_create_scenario("math")

    # recipe publishes no artifact, so algorithm_state is not recovered
    # from artifact metadata. The trainer restarts from step 0.
    assert recovered.trainer.state == {}
    second.accept_record(
        AgentRecord.create(
            agent_record_id="second",
            scenario="math",
            request_type=RequestType.REPORT,
            payload={"score": 1.0},
            created_at=2.0,
        )
    )
    assert recovered.trainer.state == {}


def test_scenario_close_closes_processor_before_records() -> None:
    scenario = build_default_dispatcher(
        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    ).get_or_create_scenario("math")
    calls: list[str] = []

    def _spy(name: str, original):
        def wrapped() -> None:
            calls.append(name)
            original()

        return wrapped

    scenario.trainer._pipeline.close = _spy("processor", scenario.trainer._pipeline.close)
    scenario.records.close = _spy("records", scenario.records.close)

    scenario.close()
    scenario.close()

    # Processor teardown strictly precedes the store closing (a processor
    # worker must never observe a closed store), and close is idempotent.
    assert calls[:2] == ["processor", "records"]


def test_dispatcher_close_tears_down_scenarios_through_close() -> None:
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    scenario = dispatcher.get_or_create_scenario("math")
    closed: list[str] = []
    original = scenario.close
    scenario.close = lambda: (closed.append("scenario"), original())[-1]

    dispatcher.close()

    assert closed == ["scenario"]


def test_dispatcher_reload_closes_the_dropped_scenario_instance() -> None:
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    dropped = dispatcher.get_or_create_scenario("math")
    closed: list[str] = []
    original = dropped.close
    dropped.close = lambda: (closed.append("dropped"), original())[-1]

    recovered = dispatcher.registry.reload("math")

    assert closed == ["dropped"]
    assert recovered is not dropped
    assert dispatcher.get_or_create_scenario("math") is recovered


def test_lock_instance_tells_a_replaced_instance_from_a_removed_one() -> None:
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    registry = dispatcher.registry
    try:
        dropped = dispatcher.get_or_create_scenario("math")
        with registry.lock_instance(dropped) as state:
            assert state is InstanceState.CURRENT

        recovered = registry.reload("math")
        with registry.lock_instance(dropped) as state:
            assert state is InstanceState.REPLACED

        with registry.lock_for("math"):
            registry.unload("math")
        recovered.close()
        with registry.lock_instance(recovered) as state:
            assert state is InstanceState.REMOVED
    finally:
        dispatcher.close()


@pytest.mark.parametrize("operation", [None, "training"])
def test_checkpoint_metadata_decodes_directly_to_commit_record(operation) -> None:
    from reef.core.artifact_ref import ArtifactRef, encode_artifact_ref
    from reef.storage.commits import CommitRecord, RecordProgress, parse_scenario_metadata, scenario_metadata_for

    base = ArtifactRef("base-content", "base-release", None)
    head = ArtifactRef("checkpoint-content", "checkpoint-release", base.release_id)
    # Persisted reef-scenario/5 metadata contains registration and recovery
    # fields; the containing artifact supplies the checkpoint release identity.
    metadata = {
        "format": "reef-scenario/5",
        "scenario": "math",
        "source": {"kind": "item", "item": "upload-1", "uri": "s3://bucket/upload-1/", "version": ""},
        "start_artifact": encode_artifact_ref(base),
        "scenario_step": 3,
        "algorithm_state": {"steps": 3},
        "record_progress": {
            "high_water_sequence": 7,
            "high_water_offset": 5,
            "compacted_ids": ["old-record"],
            "consumed_ids": ["trained-record"],
        },
        "metrics": {"loss": 0.25},
    }
    if operation is not None:
        metadata["operation"] = operation
    metadata["training_job_id"] = "job-3"

    name, registered_base, source, checkpoint = parse_scenario_metadata(metadata, checkpoint_head=head)

    assert name == "math"
    assert registered_base == base
    assert source == RootSource("item", item="upload-1", uri="s3://bucket/upload-1/", version="")
    assert isinstance(checkpoint, CommitRecord)
    assert checkpoint.artifact_ref == head
    assert checkpoint.step == 3
    assert checkpoint.checkpoint is True
    # Metadata that names no active_artifact belongs to a checkpoint the activation rule activated.
    assert checkpoint.activated is True
    assert checkpoint.algorithm_state == {"steps": 3}
    assert checkpoint.high_water_sequence == 7
    assert checkpoint.high_water_offset == 5
    assert checkpoint.consumed_ids == frozenset({"trained-record", "old-record"})
    assert checkpoint.operation == "training"
    assert checkpoint.operation_verified is (operation is not None)
    assert checkpoint.training_job_id == metadata.get("training_job_id")
    assert checkpoint.metrics == {"loss": 0.25}

    encoded = scenario_metadata_for(
        name=name,
        source=source,
        start_artifact=registered_base,
        scenario_step=checkpoint.step,
        algorithm_state=checkpoint.algorithm_state,
        record_progress=RecordProgress(
            checkpoint.high_water_sequence,
            checkpoint.high_water_offset,
            checkpoint.consumed_ids,
        ),
        metrics=checkpoint.metrics,
        training_job_id=checkpoint.training_job_id,
    )
    assert encoded == {
        **metadata,
        "operation": "training",
        "record_progress": {
            "high_water_sequence": 7,
            "high_water_offset": 5,
            "consumed_ids": ["old-record", "trained-record"],
        },
    }

    metadata["algorithm_state"]["steps"] = 99
    metadata["metrics"]["loss"] = 99
    assert checkpoint.algorithm_state == {"steps": 3}
    assert checkpoint.metrics == {"loss": 0.25}


def test_checkpoint_metadata_names_the_artifact_that_serves_after_a_held_step() -> None:
    """A recovery without the commit history reads from the artifact head which artifact serves."""
    from reef.core.artifact_ref import ArtifactRef
    from reef.storage.commits import (
        RecordProgress,
        active_artifact_ref,
        parse_scenario_metadata,
        scenario_metadata_for,
        served_artifact_ref,
    )

    base = ArtifactRef("base-content", "base-release", None)
    served = ArtifactRef("served-content", "served-release", base.release_id)
    head = ArtifactRef("held-content", "held-release", served.release_id)
    progress = RecordProgress(0, 0, frozenset())
    source = RootSource("default")
    held = scenario_metadata_for(
        name="math",
        source=source,
        start_artifact=base,
        scenario_step=2,
        record_progress=progress,
        active_artifact=served,
    )
    _, _, _, checkpoint = parse_scenario_metadata(held, checkpoint_head=head)
    assert checkpoint is not None and checkpoint.activated is False
    assert served_artifact_ref(held, checkpoint_head=head) == served
    assert active_artifact_ref((checkpoint,), served_artifact_ref(held, checkpoint_head=head)) == served

    activated = scenario_metadata_for(
        name="math", source=source, start_artifact=base, scenario_step=2, record_progress=progress
    )
    assert served_artifact_ref(activated, checkpoint_head=head) == head
    with pytest.raises(ValueError, match="active_artifact must be an object"):
        served_artifact_ref({**held, "active_artifact": "served-release"}, checkpoint_head=head)
    with pytest.raises(ValueError, match="names no active_artifact"):
        scenario_metadata_for(name="math", source=source, active_artifact=served)
    with pytest.raises(ValueError, match="start_artifact exactly after step zero"):
        scenario_metadata_for(name="math", source=source, start_artifact=base)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"record_progress": None}, "requires record_progress"),
        ({"scenario_step": True}, "scenario_step must be non-negative"),
        ({"operation": "promote"}, "operation must be 'training'"),
        ({"start_artifact": None}, "after step zero requires start_artifact"),
        ({"source": {"kind": "base"}}, "a root source is"),
        # A scenario that an earlier Reef stored is refused, not read: it is created again.
        ({"format": "reef-scenario/4"}, r"format 'reef-scenario/4'; this Reef reads only reef-scenario/5"),
    ],
)
def test_checkpoint_metadata_rejects_invalid_recovery_state(changes, message) -> None:
    from reef.core.artifact_ref import ArtifactRef, encode_artifact_ref
    from reef.storage.commits import parse_scenario_metadata

    base = ArtifactRef("base", "base", None)
    metadata = {
        "format": "reef-scenario/5",
        "scenario": "math",
        "source": {"kind": "default", "inputs": {}},
        "start_artifact": encode_artifact_ref(base),
        "scenario_step": 1,
        "record_progress": {
            "high_water_sequence": 0,
            "high_water_offset": 0,
            "consumed_ids": [],
        },
        **changes,
    }
    with pytest.raises(ValueError, match=message):
        parse_scenario_metadata(metadata, checkpoint_head=base)


class _RootImportingComponent(StubTrainingComponent):
    """A stub training component whose default start holds ``model.txt`` = ``model``, and whose trainer starts from
    an import of its scenario's root, as every component that imports releases does."""

    def __init__(self, model: str) -> None:
        super().__init__(shares_scenarios=True)
        self.model = model

    def default_start(self) -> DefaultStart:
        return DefaultStart(files={"model.txt": self.model}, inputs={"model": self.model})

    def root_start(self, root: ScenarioRoot) -> str:
        return TrainingComponent.root_start(self, root)


class _SharedRuntime(StubInferenceRuntime):
    """A stub runtime that serves each training scenario its own weights, so scenarios share one deployment."""

    @property
    def shared_by_training_scenarios(self) -> bool:
        return True


def _weights_dispatcher(tmp_path, backend_factory, model: str) -> Dispatcher:
    """A dispatcher of the test policy recipe whose training component's default start is the model ``model``."""
    component = _RootImportingComponent(model)
    return Dispatcher(
        TestPolicyRecipe(**runtime_bindings(component, _SharedRuntime())),
        backend_factory,
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )


@pytest.mark.unit
def test_two_scenarios_own_distinct_root_releases(tmp_path) -> None:
    dispatcher = _weights_dispatcher(tmp_path, InMemoryRepositoryBackend.factory(), "base-model")
    try:
        first, second = (dispatcher.get_or_create_scenario(name) for name in ("first", "second"))
        roots = [scenario.repository.start_artifact for scenario in (first, second)]
        assert roots[0].release_id != roots[1].release_id
        # Equal default starts make equal content, so content-addressed serving names do not change.
        assert roots[0].content_id == roots[1].content_id == DefaultStart(inputs={"model": "base-model"}).content_id
        assert all(root.parent_release_id is None for root in roots)
        (creation,) = first.artifacts()
        assert creation["operation"] == "creation" and creation["artifact_id"] == roots[0].release_id
        assert creation["source"] == {"kind": "default", "inputs": {"model": "base-model"}}
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_changed_default_start_neither_blocks_nor_changes_an_existing_scenario(tmp_path) -> None:
    """The old scenario keeps serving and training its own root; a new scenario gets the new default start."""
    backend_factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    first = _weights_dispatcher(tmp_path, backend_factory, "old-model")
    old_root = first.get_or_create_scenario("old").repository.start_artifact
    first.close()

    second = _weights_dispatcher(tmp_path, backend_factory, "new-model")
    try:
        old = second.get_or_create_scenario("old")
        assert old.repository.start_artifact == old.current_artifact_ref() == old_root
        # The runtime serves the old scenario's own root.
        assert second.recipe.runtime.served == old_root.release_id
        old_start = Path(JobLoopState.from_dict(old.trainer.state).start)
        assert (old_start / "model.txt").read_text() == "old-model"
        assert old.artifacts()[0]["source"] == {"kind": "default", "inputs": {"model": "old-model"}}

        new = second.get_or_create_scenario("new")
        new_root = new.repository.materialize(new.repository.start_artifact)
        assert (new_root.local_path / "model.txt").read_text() == "new-model"
        assert (Path(JobLoopState.from_dict(new.trainer.state).start) / "model.txt").read_text() == "new-model"
    finally:
        second.close()


@pytest.mark.unit
def test_concurrent_creations_of_one_name_register_one_root(tmp_path) -> None:
    dispatcher = _weights_dispatcher(tmp_path, InMemoryRepositoryBackend.factory(), "base-model")
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: dispatcher.create_scenario("shared"), range(8)))
        assert sum(created for _, created in results) == 1
        assert len({scenario.repository.start_artifact for scenario, _ in results}) == 1
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_scenario_stored_by_an_earlier_reef_is_refused_not_read(tmp_path) -> None:
    """A registration in ``reef-scenario/4`` named a shared base release; this Reef refuses it with a clear error."""
    backend_factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    (tmp_path / "old-root").mkdir()
    stored = {"format": "reef-scenario/4", "scenario": "math", "base_artifact": {}, "scenario_step": 0}
    backend_factory("math").register(Artifact.local(tmp_path / "old-root"), metadata={SCENARIO_METADATA_KEY: stored})
    dispatcher = build_default_dispatcher(
        backend_factory=backend_factory, scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    )
    try:
        with pytest.raises(ValueError, match=r"'reef-scenario/4'; this Reef reads only reef-scenario/5, so create"):
            dispatcher.get_or_create_scenario("math")
    finally:
        dispatcher.close()


class _NoReleaseStarts(_RootImportingComponent):
    """A component that, as Slime, starts only from its own default start."""

    def root_start(self, root: ScenarioRoot) -> str:
        if root.source.kind != "default":
            raise ValueError("this component trains only its configured weights")
        return self.base_start


class ItemTreeRecipe(Recipe):
    """A recipe with Recipe's fields that serves an importable pi harness and runs no training jobs.

    Scenario creation tests use its fixed default tree and harness layout as a composite's second component.
    """

    def build_surface(self, scenario: str) -> Surface:
        descriptor = get_adapter("pi")
        return create_harness_surface(descriptor, layout=HarnessLayout(descriptor))

    def default_start(self) -> DefaultStart:
        return DefaultStart(files={"rules.md": "default rules"})


@pytest.mark.unit
@pytest.mark.parametrize("component", ["weights", "harness"])
@pytest.mark.parametrize("accepts_item", [False, True])
def test_composite_item_roots_are_checked_before_registration(
    tmp_path: Path, component: str, accepts_item: bool
) -> None:
    training = _RootImportingComponent("base-model") if accepts_item else _NoReleaseStarts("base-model")
    weights = TestPolicyRecipe(**runtime_bindings(training, _SharedRuntime()))
    recipe = CompositeRecipe(components={"weights": weights, "harness": ItemTreeRecipe()})
    backend_factory = InMemoryRepositoryBackend.factory()
    directory = tmp_path / "item"
    directory.mkdir()
    if component == "weights":
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_text("imported weights")
    else:
        (directory / "rules.md").write_text("imported rules")
    item = ItemStart(
        "upload-1", directory.as_uri(), DirectorySource(directory, source_version="v1", link=False), component
    )
    dispatcher = Dispatcher(
        recipe,
        backend_factory,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )
    try:
        if accepts_item:
            scenario, created = dispatcher.create_scenario("retry", item)
            root = scenario.repository.materialize(scenario.repository.start_artifact)
            expected_file = "model.safetensors" if component == "weights" else "rules.md"
            assert root.local_path is not None
            expected_text = "imported weights" if component == "weights" else "imported rules"
            assert (root.local_path / component / expected_file).read_text() == expected_text
        else:
            with pytest.raises(ValueError, match="trains only its configured weights"):
                dispatcher.create_scenario("retry", item)
            assert backend_factory("retry").metadata() is None
            assert not dispatcher.has_scenario("retry")
            scenario, created = dispatcher.create_scenario("retry")
        assert created
        assert scenario.repository.start_artifact.parent_release_id is None
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_an_item_root_is_checked_by_the_layout_and_the_training_start_before_it_is_registered(tmp_path) -> None:
    """A creation from an item writes the item as the root release; an item that the layout or the training
    component refuses registers nothing."""
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    (weights / "model.safetensors").write_text("w9")
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "notes.txt").write_text("no weights")

    def item(directory: Path) -> ItemStart:
        return ItemStart("upload-1", directory.as_uri(), DirectorySource(directory, source_version="v1", link=False))

    dispatcher = _weights_dispatcher(tmp_path, InMemoryRepositoryBackend.factory(), "base-model")
    try:
        with pytest.raises(ValueError, match=r"config\.json and weight files"):
            dispatcher.create_scenario("refused", item(notes))
        assert not dispatcher.has_scenario("refused")
        scenario, created = dispatcher.create_scenario("started", item(weights))
        assert created
        root = scenario.repository.start_artifact
        assert scenario.current_artifact_ref() == root and root.parent_release_id is None
        assert (scenario.repository.materialize(root).local_path / "model.safetensors").read_text() == "w9"
        (row,) = scenario.artifacts()
        assert row["source"] == {"kind": "item", "item": "upload-1", "uri": weights.as_uri(), "version": "v1"}
    finally:
        dispatcher.close()

    component = _NoReleaseStarts("base-model")
    pinned = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(component, _SharedRuntime())),
        InMemoryRepositoryBackend.factory(),
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )
    try:
        with pytest.raises(ValueError, match="trains only its configured weights"):
            pinned.create_scenario("refused", item(weights))
        assert not pinned.has_scenario("refused")
    finally:
        pinned.close()
