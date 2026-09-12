"""Scenario lifecycle uses the injected store contract without local path access."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from reef.artifact import ArtifactRef, InMemoryRepositoryBackend
from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.observability import ExperimentLogger
from reef.recipe.base import Recipe
from reef.records import RecordRetention, RecordStore
from reef.scenario.state import CommitRecord, ScenarioSnapshot
from reef.scenario.store import ScenarioStore, ScenarioStoreFactory
from reef.storage.factory import SQLiteScenarioStoreFactory
from reef.train.trainer import Trainer
from reef.train.types import TrainStepResult


class OpaqueStore(ScenarioStore):
    """Delegate storage while exposing no SQLite paths or commit-log attribute."""

    def __init__(self, scenario: str, inner: ScenarioStore, events: list[tuple[str, str]], *, fail_recovery: bool):
        self.scenario = scenario
        self._inner = inner
        self._events = events
        self._fail_recovery = fail_recovery
        self.close_count = 0

    @property
    def records(self) -> RecordStore:
        return self._inner.records

    @property
    def durable(self) -> bool:
        return self._inner.durable

    def history(self) -> tuple[CommitRecord, ...]:
        return self._inner.history()

    def training_run_position(self) -> tuple[int, int]:
        return self._inner.training_run_position()

    def commit_step(self, *, expected_step: int, commit: CommitRecord) -> CommitRecord:
        self._events.append(("commit", self.scenario))
        return self._inner.commit_step(expected_step=expected_step, commit=commit)

    def recover(self, *, snapshot: ScenarioSnapshot, checkpoint_head: ArtifactRef) -> CommitRecord | None:
        self._events.append(("recover", self.scenario))
        if self._fail_recovery:
            raise RuntimeError("injected recovery failure")
        return self._inner.recover(snapshot=snapshot, checkpoint_head=checkpoint_head)

    def close(self) -> None:
        self.close_count += 1
        self._events.append(("close", self.scenario))
        self._inner.close()


class OpaqueStoreFactory(ScenarioStoreFactory):
    """The adapter owns its directory; Dispatcher sees only domain operations."""

    def __init__(self, directory: Path, *, fail_recovery: bool = False):
        self._inner: ScenarioStoreFactory = SQLiteScenarioStoreFactory(directory)
        self._fail_recovery = fail_recovery
        self.sessions: list[OpaqueStore] = []
        self.events: list[tuple[str, str]] = []
        self.prune_calls: list[tuple[float, int]] = []
        self.close_count = 0

    @property
    def durable(self) -> bool:
        return self._inner.durable

    def open(self, scenario: str) -> ScenarioStore:
        store = OpaqueStore(scenario, self._inner.open(scenario), self.events, fail_recovery=self._fail_recovery)
        self.sessions.append(store)
        return store

    def archive(self, scenario: str) -> tuple[str, ...]:
        self.events.append(("archive", scenario))
        self._inner.archive(scenario)
        return (f"archive://{scenario}",)

    def prune(self, *, days: float, max_bytes: int) -> int:
        self.prune_calls.append((days, max_bytes))
        return self._inner.prune(days=days, max_bytes=max_bytes)

    def close(self) -> None:
        self.close_count += 1
        self.events.append(("close_factory", ""))
        self._inner.close()


class FailingRecipe(Recipe):
    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        raise RuntimeError("injected recipe failure")


class IncompleteRecords(RecordStore):
    def close(self) -> None:
        pass


class IncompleteStore(ScenarioStore):
    def close(self) -> None:
        pass


class IncompleteStoreFactory(ScenarioStoreFactory):
    def close(self) -> None:
        pass


@pytest.mark.parametrize("implementation", (IncompleteRecords, IncompleteStore, IncompleteStoreFactory))
def test_incomplete_storage_subclasses_cannot_be_instantiated(implementation: type) -> None:
    with pytest.raises(TypeError, match="abstract"):
        implementation()


def test_sqlite_implementations_inherit_storage_bases(tmp_path: Path) -> None:
    with closing(SQLiteScenarioStoreFactory(tmp_path)) as factory, closing(factory.open("math")) as store:
        assert isinstance(factory, ScenarioStoreFactory)
        assert isinstance(store, ScenarioStore)
        assert isinstance(store.records, RecordStore)


def backend_factory(tmp_path: Path):
    initial = tmp_path / "initial"
    initial.mkdir()
    return InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")


def trace(record_id: str, scenario: str = "math") -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=record_id,
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"text": record_id},
    )


def test_injected_store_commits_reads_history_and_recovers(tmp_path: Path) -> None:
    repository_factory = backend_factory(tmp_path)
    directory = tmp_path / "private-store"
    factory = OpaqueStoreFactory(directory)
    dispatcher = Dispatcher(Recipe(), repository_factory, scenario_store_factory=factory)
    try:
        scenario = dispatcher.get_or_create_scenario("math")
        assert scenario is not None
        assert scenario.store is factory.sessions[0]
        assert scenario.store.durable
        scenario.records.append(trace("first"))
        scenario.commit(TrainStepResult(state={}))
        assert scenario.scenario_step == 1
        assert ("commit", "math") in factory.events
        assert [row["step"] for row in dispatcher.read_commits("math")["commits"]] == [1]
        stored = dispatcher.read_record("math", "first")
        assert stored is not None
        assert stored["payload"] == {"text": "first"}
        assert dispatcher.read_records("math")["records"][0]["agent_record_id"] == "first"
        assert scenario.store.training_run_position() == (0, 1)
    finally:
        dispatcher.close()
        dispatcher.close()
    assert factory.sessions[0].close_count == 1
    assert factory.close_count == 1

    reopened_factory = OpaqueStoreFactory(directory)
    reopened = Dispatcher(Recipe(), repository_factory, scenario_store_factory=reopened_factory)
    try:
        recovered = reopened.get_or_create_scenario("math")
        assert recovered is not None
        assert recovered.scenario_step == 1
        assert recovered.records.get("math", "first") is not None
        assert [row["step"] for row in reopened.read_commits("math")["commits"]] == [1]
        assert ("recover", "math") in reopened_factory.events
    finally:
        reopened.close()


def test_injected_factory_owns_retention_archive_and_same_name_recreation(tmp_path: Path) -> None:
    factory = OpaqueStoreFactory(tmp_path / "private-store")
    model_directory = tmp_path / "model-settings"
    dispatcher = Dispatcher(
        Recipe(), backend_factory(tmp_path), agent_record_dir=model_directory, scenario_store_factory=factory
    )
    try:
        math = dispatcher.get_or_create_scenario("math")
        code = dispatcher.get_or_create_scenario("code")
        assert math is not None and code is not None
        math.model_config.save(None)
        model_path = next(model_directory.glob("*-model.json"))
        model_content = model_path.read_bytes()
        math.records.append(trace("first"))
        code.records.append(trace("other", "code"))
        math.records.compact("math", frozenset({"first"}))
        retention = RecordRetention(days=7, max_bytes=1)
        assert dispatcher.prune_record_archives(retention) == 1
        assert factory.prune_calls == [(retention.days, retention.max_bytes)]
        assert dispatcher.read_record("math", "first") is None
        assert dispatcher.read_record("code", "other") is not None

        math.records.append(trace("archived"))
        math.records.compact("math", frozenset({"archived"}))
        result = dispatcher.delete_scenario("math")
        assert "archive://math" in result["archived"]
        assert not model_path.exists()
        [archived_model] = (model_directory / "archived").glob("*/*-model.json")
        assert archived_model.read_bytes() == model_content
        assert str(archived_model) in result["archived"]
        assert factory.events.index(("close", "math")) < factory.events.index(("archive", "math"))
        assert dispatcher.prune_record_archives(retention) == 1
        replacement = dispatcher.get_or_create_scenario("math")
        assert replacement is not None
        assert replacement is not math
        assert replacement.records.count("math") == 0
        assert replacement.store.history() == ()
        assert dispatcher.read_record("code", "other") is not None
    finally:
        dispatcher.close()
    assert all(session.close_count == 1 for session in factory.sessions)
    assert factory.events[-1] == ("close_factory", "")


@pytest.mark.parametrize("failure_point", ("recovery", "recipe"))
def test_failed_construction_closes_injected_store(tmp_path: Path, failure_point: str) -> None:
    factory = OpaqueStoreFactory(tmp_path / "private-store", fail_recovery=failure_point == "recovery")
    recipe = FailingRecipe() if failure_point == "recipe" else Recipe()
    dispatcher = Dispatcher(recipe, backend_factory(tmp_path), scenario_store_factory=factory)
    try:
        with pytest.raises(RuntimeError, match=f"injected {failure_point} failure"):
            dispatcher.get_or_create_scenario("math")
        assert not dispatcher.has_loaded("math")
        assert len(factory.sessions) == 1
        assert factory.sessions[0].close_count == 1
    finally:
        dispatcher.close()
    assert factory.close_count == 1


def test_trainer_teardown_failure_still_closes_store_and_factory(tmp_path: Path, monkeypatch) -> None:
    factory = OpaqueStoreFactory(tmp_path / "private-store")
    dispatcher = Dispatcher(Recipe(), backend_factory(tmp_path), scenario_store_factory=factory)
    scenario = dispatcher.get_or_create_scenario("math")
    assert scenario is not None
    original_close = scenario.trainer.close

    def fail_close() -> None:
        original_close()
        factory.events.append(("close_trainer", "math"))
        raise RuntimeError("injected trainer close failure")

    monkeypatch.setattr(scenario.trainer, "close", fail_close)
    with pytest.raises(RuntimeError, match="injected trainer close failure"):
        dispatcher.close()
    dispatcher.close()
    scenario.close()
    assert factory.sessions[0].close_count == 1
    assert factory.close_count == 1
    assert factory.events[-3:] == [("close_trainer", "math"), ("close", "math"), ("close_factory", "")]
