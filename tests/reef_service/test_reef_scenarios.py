from __future__ import annotations

import importlib.util
import inspect

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.recipe import Recipe, RecipeRegistry
from reef.scenario.checkpoint_strategy import EveryNVersions


def test_dispatcher_constructor_has_no_redundant_scenario_binding_stores() -> None:
    parameters = inspect.signature(Dispatcher).parameters

    assert "recipe_store" not in parameters
    assert "artifact_version_store" not in parameters
    assert "recipe_resolver" not in parameters
    assert "recipe_validator" not in parameters
    assert "checkpoint_strategy" not in parameters


def test_artifact_state_and_snapshot_are_not_parallel_public_types() -> None:
    assert importlib.util.find_spec("reef.scenarios") is None


def test_scenario_owns_recipe_and_base_artifact() -> None:
    scenario = build_default_dispatcher().get_or_create_scenario("math", "recipe")

    assert scenario.recipe == "recipe"
    assert scenario.repository.base_artifact.version
    assert scenario._commit_protocol.checkpoint_strategy is not None
    assert scenario.repository.current_artifact == scenario.repository.checkpoint_artifact


def test_scenario_step_is_owned_by_scenario() -> None:
    scenario = build_default_dispatcher().get_or_create_scenario("math", "recipe")

    assert not hasattr(scenario.repository.base_artifact, "scenario")
    assert scenario.scenario_step == 0
    assert scenario.repository.current_artifact.version
    assert scenario.repository.checkpoint_artifact.version
    assert not hasattr(scenario, "artifact_version")
    assert not hasattr(scenario, "selected_artifact_version")


def test_scenario_owns_scenario_scoped_repository() -> None:
    scenario = build_default_dispatcher().get_or_create_scenario("math", "recipe")

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
    dispatcher = Dispatcher(
        RecipeRegistry(
            recipes={
                "fast": Recipe(name="fast", checkpoint_strategy=EveryNVersions(1)),
                "slow": Recipe(name="slow", checkpoint_strategy=EveryNVersions(3)),
            },
        ),
        InMemoryRepositoryBackend.factory(initial),
    )

    fast = dispatcher.get_or_create_scenario("fast", "fast")
    slow = dispatcher.get_or_create_scenario("slow", "slow")

    assert fast._commit_protocol.checkpoint_strategy.n == 1
    assert slow._commit_protocol.checkpoint_strategy.n == 3


def test_scenario_snapshot_round_trips() -> None:
    from reef.scenario.snapshot import ScenarioSnapshot, parse_snapshot_metadata

    scenario = build_default_dispatcher().get_or_create_scenario("math", "recipe")
    metadata = scenario.to_snapshot_metadata()

    assert metadata["format"] == "reef-scenario/3"
    assert "scenario" not in metadata["base_artifact"]
    assert metadata["scenario_step"] == 0
    assert metadata["operation"] == "training"
    assert parse_snapshot_metadata(metadata) == ScenarioSnapshot(
        scenario="math",
        recipe="recipe",
        base_artifact=scenario.repository.base_artifact,
        scenario_step=0,
        algorithm_state=None,
        record_progress=None,
        training_job_id=None,
        operation="training",
        rollback_target_artifact_version=None,
    )


def test_legacy_rollback_snapshot_preserves_its_operation() -> None:
    from reef.scenario.snapshot import parse_snapshot_metadata

    scenario = build_default_dispatcher().get_or_create_scenario("math", "recipe")
    assert scenario is not None
    metadata = scenario.to_snapshot_metadata()
    metadata.pop("operation")
    metadata["rollback"] = {"target_artifact_version": "checkpoint-v1"}

    snapshot = parse_snapshot_metadata(metadata)

    assert snapshot.operation == "rollback"
    assert snapshot.rollback_target_artifact_version == "checkpoint-v1"


def test_dispatcher_restores_agent_record_from_configured_directory(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"
    first = build_default_dispatcher(backend_factory=backend, agent_record_dir=agent_record_dir)
    record = AgentRecord.create(
        agent_record_id="persisted",
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": 1.0},
        created_at=1.0,
    )
    first.get_or_create_scenario("math", "recipe").records.append(record)

    second = build_default_dispatcher(backend_factory=backend, agent_record_dir=agent_record_dir)

    assert second.get_or_create_scenario("math").records.replay("math") == (record,)


def test_dispatcher_restores_algorithm_state_from_artifact_metadata(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"
    first = build_default_dispatcher(backend_factory=backend, agent_record_dir=agent_record_dir)
    first.accept_record(
        AgentRecord.create(
            agent_record_id="first",
            scenario="math",
            request_type=RequestType.REPORT,
            payload={"score": 1.0},
            created_at=1.0,
        ),
        recipe="recipe",
    )
    # recipe never trains, so step stays at 0 and records accumulate.
    assert first.get_or_create_scenario("math").trainer.state == {}

    second = build_default_dispatcher(backend_factory=backend, agent_record_dir=agent_record_dir)
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
    scenario = build_default_dispatcher().get_or_create_scenario("math", "recipe")
    calls: list[str] = []

    def _spy(name: str, original):
        def wrapped() -> None:
            calls.append(name)
            original()

        return wrapped

    scenario.trainer._processor.close = _spy("processor", scenario.trainer._processor.close)
    scenario._records.close = _spy("records", scenario._records.close)

    scenario.close()
    scenario.close()

    # Processor teardown strictly precedes the store closing (a processor
    # worker must never observe a closed store), and close is idempotent.
    assert calls[:2] == ["processor", "records"]


def test_dispatcher_close_tears_down_scenarios_through_close() -> None:
    dispatcher = build_default_dispatcher()
    scenario = dispatcher.get_or_create_scenario("math", "recipe")
    closed: list[str] = []
    original = scenario.close
    scenario.close = lambda: (closed.append("scenario"), original())[-1]

    dispatcher.close()

    assert closed == ["scenario"]


def test_dispatcher_reload_closes_the_dropped_scenario_instance() -> None:
    dispatcher = build_default_dispatcher()
    dropped = dispatcher.get_or_create_scenario("math", "recipe")
    closed: list[str] = []
    original = dropped.close
    dropped.close = lambda: (closed.append("dropped"), original())[-1]

    recovered = dispatcher._registry.reload("math")

    assert closed == ["dropped"]
    assert recovered is not dropped
    assert dispatcher.get_or_create_scenario("math") is recovered
