from __future__ import annotations

import importlib.util
import inspect

import pytest

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.recipe import Recipe
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.storage.sqlite import SQLiteScenarioStorage


def test_dispatcher_constructor_has_no_redundant_scenario_binding_stores() -> None:
    parameters = inspect.signature(Dispatcher).parameters

    assert "recipe_store" not in parameters
    assert "release_id_store" not in parameters
    assert "recipe_resolver" not in parameters
    assert "recipe_validator" not in parameters
    assert "checkpoint_strategy" not in parameters


def test_artifact_state_and_metadata_are_not_parallel_public_types() -> None:
    assert importlib.util.find_spec("reef.scenarios") is None


def test_scenario_owns_recipe_derived_policy_and_base_artifact() -> None:
    scenario = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage()).get_or_create_scenario("math")

    assert not hasattr(scenario, "recipe")
    assert scenario.repository.base_artifact.release_id
    assert scenario._committer.checkpoint_strategy is not None
    assert scenario.repository.current_artifact == scenario.repository.checkpoint_artifact


def test_scenario_step_is_owned_by_scenario() -> None:
    scenario = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage()).get_or_create_scenario("math")

    assert not hasattr(scenario.repository.base_artifact, "scenario")
    assert scenario.scenario_step == 0
    assert scenario.repository.current_artifact.release_id
    assert scenario.repository.checkpoint_artifact.release_id
    assert not hasattr(scenario, "release_id")
    assert not hasattr(scenario, "selected_release_id")


def test_scenario_owns_scenario_scoped_repository() -> None:
    scenario = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage()).get_or_create_scenario("math")

    repository = scenario.repository

    assert not hasattr(repository, "scenario")
    assert not hasattr(repository.base_artifact, "scenario")
    assert scenario.scenario_step == 0
    assert repository.checkpoint_artifact == repository.current_artifact
    assert not hasattr(scenario, "base_artifact")
    assert not hasattr(scenario, "current_artifact")
    assert not hasattr(scenario, "checkpoint_artifact")


def test_each_scenario_keeps_recipe_derived_checkpoint_policy(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend_factory = InMemoryRepositoryBackend.factory(initial)
    fast_dispatcher = Dispatcher(
        Recipe(name="fast", checkpoint_strategy=EveryNVersions(1)),
        backend_factory,
        scenario_storage=SQLiteScenarioStorage(),
    )
    slow_dispatcher = Dispatcher(
        Recipe(name="slow", checkpoint_strategy=EveryNVersions(3)),
        backend_factory,
        scenario_storage=SQLiteScenarioStorage(),
    )

    fast = fast_dispatcher.get_or_create_scenario("fast")
    slow = slow_dispatcher.get_or_create_scenario("slow")

    assert fast._committer.checkpoint_strategy.n == 1
    assert slow._committer.checkpoint_strategy.n == 3


def test_scenario_metadata_round_trips() -> None:
    from reef.storage.commits import parse_scenario_metadata

    scenario = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage()).get_or_create_scenario("math")
    metadata = scenario.to_metadata()

    assert metadata["format"] == "reef-scenario/4"
    assert "scenario" not in metadata["base_artifact"]
    assert metadata["scenario_step"] == 0
    assert metadata["operation"] == "training"
    assert "recipe" not in metadata
    checkpoint_head = scenario.repository.require_current_artifact()
    assert parse_scenario_metadata(metadata, checkpoint_head=checkpoint_head) == (
        "math",
        scenario.repository.base_artifact,
        None,
    )

    # Metadata written before recipe identity was removed remain readable;
    # the deployment recipe now supplies those capabilities during recovery.
    assert parse_scenario_metadata(
        {**metadata, "recipe": "legacy-name"}, checkpoint_head=checkpoint_head
    ) == parse_scenario_metadata(metadata, checkpoint_head=checkpoint_head)


def test_rollback_metadata_preserves_its_operation() -> None:
    from reef.storage.commits import parse_scenario_metadata

    scenario = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage()).get_or_create_scenario("math")
    assert scenario is not None
    metadata = scenario.to_metadata()
    metadata["operation"] = "rollback"
    metadata["rollback_target_release_id"] = "checkpoint-v1"

    metadata["scenario_step"] = 1
    metadata["record_progress"] = {
        "high_water_sequence": 0,
        "high_water_offset": 0,
        "compacted_ids": [],
        "consumed_ids": [],
    }
    _, _, checkpoint = parse_scenario_metadata(
        metadata, checkpoint_head=scenario.repository.require_current_artifact()
    )

    assert checkpoint is not None
    assert checkpoint.operation == "rollback"
    assert checkpoint.rollback_target_release_id == "checkpoint-v1"


def test_dispatcher_restores_agent_record_from_configured_directory(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"
    first = build_default_dispatcher(
        backend_factory=backend,
        agent_record_dir=agent_record_dir,
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
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
        agent_record_dir=agent_record_dir,
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
    )

    assert second.get_or_create_scenario("math").records.replay("math") == (record,)


def test_dispatcher_restores_algorithm_state_from_artifact_metadata(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"
    first = build_default_dispatcher(
        backend_factory=backend,
        agent_record_dir=agent_record_dir,
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
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
        agent_record_dir=agent_record_dir,
        scenario_storage=SQLiteScenarioStorage(agent_record_dir),
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
    scenario = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage()).get_or_create_scenario("math")
    calls: list[str] = []

    def _spy(name: str, original):
        def wrapped() -> None:
            calls.append(name)
            original()

        return wrapped

    scenario.trainer._processor.close = _spy("processor", scenario.trainer._processor.close)
    scenario.records.close = _spy("records", scenario.records.close)

    scenario.close()
    scenario.close()

    # Processor teardown strictly precedes the store closing (a processor
    # worker must never observe a closed store), and close is idempotent.
    assert calls[:2] == ["processor", "records"]


def test_dispatcher_close_tears_down_scenarios_through_close() -> None:
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage())
    scenario = dispatcher.get_or_create_scenario("math")
    closed: list[str] = []
    original = scenario.close
    scenario.close = lambda: (closed.append("scenario"), original())[-1]

    dispatcher.close()

    assert closed == ["scenario"]


def test_dispatcher_reload_closes_the_dropped_scenario_instance() -> None:
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage())
    dropped = dispatcher.get_or_create_scenario("math")
    closed: list[str] = []
    original = dropped.close
    dropped.close = lambda: (closed.append("dropped"), original())[-1]

    recovered = dispatcher._registry.reload("math")

    assert closed == ["dropped"]
    assert recovered is not dropped
    assert dispatcher.get_or_create_scenario("math") is recovered


@pytest.mark.parametrize("operation", [None, "training", "rollback", "promote"])
def test_checkpoint_metadata_decodes_directly_to_commit_record(operation) -> None:
    from reef.core.artifact_ref import ArtifactRef, encode_artifact_ref
    from reef.storage.commits import CommitRecord, RecordProgress, parse_scenario_metadata, scenario_metadata_for

    base = ArtifactRef("base-content", "base-release", None)
    head = ArtifactRef("checkpoint-content", "checkpoint-release", base.release_id)
    # Persisted reef-scenario/4 metadata contains registration and recovery
    # fields; the containing artifact supplies the checkpoint release identity.
    metadata = {
        "format": "reef-scenario/4",
        "scenario": "math",
        "base_artifact": encode_artifact_ref(base),
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
    if operation in ("rollback", "promote"):
        metadata["rollback_target_release_id"] = base.release_id
    else:
        metadata["training_job_id"] = "job-3"

    name, registered_base, checkpoint = parse_scenario_metadata(metadata, checkpoint_head=head)

    assert name == "math"
    assert registered_base == base
    assert isinstance(checkpoint, CommitRecord)
    assert checkpoint.artifact_ref == head
    assert checkpoint.step == 3
    assert checkpoint.checkpoint is True
    assert checkpoint.pending is False
    assert checkpoint.algorithm_state == {"steps": 3}
    assert checkpoint.high_water_sequence == 7
    assert checkpoint.high_water_offset == 5
    assert checkpoint.compacted_ids == frozenset({"old-record"})
    assert checkpoint.consumed_ids == frozenset({"trained-record"})
    assert checkpoint.operation == (operation or "training")
    assert checkpoint.operation_verified is (operation is not None)
    assert checkpoint.rollback_target_release_id == metadata.get("rollback_target_release_id")
    assert checkpoint.training_job_id == metadata.get("training_job_id")
    assert checkpoint.metrics == {"loss": 0.25}

    encoded = scenario_metadata_for(
        name=name,
        base_artifact=registered_base,
        scenario_step=checkpoint.step,
        algorithm_state=checkpoint.algorithm_state,
        record_progress=RecordProgress(
            checkpoint.high_water_sequence,
            checkpoint.high_water_offset,
            checkpoint.compacted_ids,
            checkpoint.consumed_ids,
        ),
        operation=checkpoint.operation,
        rollback_target_release_id=checkpoint.rollback_target_release_id,
        metrics=checkpoint.metrics,
        training_job_id=checkpoint.training_job_id,
    )
    assert encoded == {**metadata, "operation": operation or "training"}

    metadata["algorithm_state"]["steps"] = 99
    metadata["metrics"]["loss"] = 99
    assert checkpoint.algorithm_state == {"steps": 3}
    assert checkpoint.metrics == {"loss": 0.25}


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"record_progress": None}, "requires record_progress"),
        ({"scenario_step": True}, "scenario_step must be non-negative"),
        (
            {"operation": "rollback", "rollback_target_release_id": "base", "training_job_id": "job"},
            "cannot carry training_job_id",
        ),
    ],
)
def test_checkpoint_metadata_rejects_invalid_recovery_state(changes, message) -> None:
    from reef.core.artifact_ref import ArtifactRef, encode_artifact_ref
    from reef.storage.commits import parse_scenario_metadata

    base = ArtifactRef("base", "base", None)
    metadata = {
        "format": "reef-scenario/4",
        "scenario": "math",
        "base_artifact": encode_artifact_ref(base),
        "scenario_step": 1,
        "record_progress": {
            "high_water_sequence": 0,
            "high_water_offset": 0,
            "compacted_ids": [],
            "consumed_ids": [],
        },
        **changes,
    }
    with pytest.raises(ValueError, match=message):
        parse_scenario_metadata(metadata, checkpoint_head=base)
