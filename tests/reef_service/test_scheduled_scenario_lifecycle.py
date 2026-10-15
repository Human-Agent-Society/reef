"""Scheduler assignments guard construction and acknowledge completed local cleanup."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import closing
from pathlib import Path

import pytest

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.repository import RepositoryBackendFactory
from reef.core.errors import ReefError, ScenarioBusy
from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.inference.model_config import ModelConfig
from reef.recipe import Recipe
from reef.scenario.admission import (
    ScenarioAdmissionError,
    ScenarioAdmissions,
    ScenarioControlRecord,
    ScenarioControlResult,
)
from reef.scenario.scenario import Scenario
from reef.storage.commit_log import CommitLogScenarioStore
from reef.storage.json_admissions import JSONScenarioAdmissionStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.trainer import Trainer


class ScheduledDeployment:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        bootstrap = directory / "bootstrap"
        bootstrap.mkdir()
        self.backend_factory: RepositoryBackendFactory = InMemoryRepositoryBackend.factory(
            bootstrap, root=directory / "repository"
        )
        self.dispatchers: list[Dispatcher] = []

    def open(self, *, scheduled: bool = True) -> Dispatcher:
        dispatcher = Dispatcher(
            Recipe(),
            self.backend_factory,
            agent_record_dir=self.directory / "records",
            scenario_storage=SQLiteScenarioStorage(self.directory / "records"),
            require_scenario_admission=scheduled,
            scenario_admission_store=(
                JSONScenarioAdmissionStore(self.directory / "assignments") if scheduled else None
            ),
        )
        self.dispatchers.append(dispatcher)
        return dispatcher

    def close(self) -> None:
        for dispatcher in self.dispatchers:
            dispatcher.close()


@pytest.fixture
def deployment(tmp_path: Path) -> Iterator[ScheduledDeployment]:
    with closing(ScheduledDeployment(tmp_path)) as value:
        yield value


@pytest.mark.parametrize("entry", ("create", "model", "require", "reload", "record", "preload"))
def test_unapproved_entries_never_construct_scenarios(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    dispatcher = deployment.open()
    constructed: list[str] = []

    def forbidden(scenario: str, release_id: str | None = None, *, model_config: ModelConfig) -> Scenario:
        constructed.append(scenario)
        raise RuntimeError("unapproved construction")

    monkeypatch.setattr(dispatcher._registry._scenario_factory, "load_or_create", forbidden)
    if entry == "preload":
        dispatcher._preload_scenarios(("waiting",))
    else:
        with pytest.raises(ReefError):
            if entry == "create":
                dispatcher.get_or_create_scenario("waiting")
            elif entry == "model":
                dispatcher.configure_scenario_model("waiting", None, create=True)
            elif entry == "require":
                dispatcher._registry.require("waiting")
            elif entry == "reload":
                dispatcher._registry.reload("waiting")
            else:
                dispatcher.accept_record(
                    AgentRecord.create(scenario="waiting", request_type=RequestType.INFERENCE, payload={})
                )
    assert constructed == []
    assert dispatcher._training.local_workers == {}
    assert list((deployment.directory / "records").glob("*-model.json")) == []


def test_start_reload_delete_and_recreate_keep_lifecycle_identity(deployment: ScheduledDeployment) -> None:
    dispatcher = deployment.open()
    started = dispatcher.start_scheduled_scenario(1, 1, "work")
    scenario = dispatcher.get_or_create_scenario("work")
    assert started.state == "admitted" and scenario is not None
    assert dispatcher.start_scheduled_scenario(1, 1, "work") == started
    assert dispatcher.get_or_create_scenario("work") is scenario
    with pytest.raises(ScenarioAdmissionError, match="tenant"):
        dispatcher.start_scheduled_scenario(1, 2, "work")
    with pytest.raises(ScenarioAdmissionError, match="another lifecycle"):
        dispatcher.start_scheduled_scenario(2, 1, "work")
    rebuilt = dispatcher._registry.reload("work")
    assert rebuilt is not scenario
    assert dispatcher.scheduled_scenario_status("work").state == started.state
    with pytest.raises(ScenarioAdmissionError, match="scheduler cleanup"):
        dispatcher.delete_scenario("work")
    deleted = dispatcher.cleanup_scheduled_scenario(1, "work", tenant_id=1)
    assert deleted.state == "deleted" and deleted.cleanup_complete
    assert dispatcher.cleanup_scheduled_scenario(1, "work", tenant_id=1) == deleted
    assert not dispatcher.has_scenario("work")
    with pytest.raises(ScenarioAdmissionError, match="completed"):
        dispatcher.start_scheduled_scenario(1, 1, "work")
    assert dispatcher.start_scheduled_scenario(2, 2, "work").state == "admitted"
    with pytest.raises(ScenarioAdmissionError, match="does not match"):
        dispatcher.cleanup_scheduled_scenario(1, "work", tenant_id=1)


def test_restart_requires_control_recovery_before_business_resolution(deployment: ScheduledDeployment) -> None:
    first = deployment.open()
    first.start_scheduled_scenario(7, 1, "work")
    first.close()
    restarted = deployment.open()
    status = restarted.scheduled_scenario_status("work")
    assert status is not None and status.state == "starting" and status.scenario_id == 7
    assert restarted._lifecycle.preload_thread is None
    assert not restarted.has_loaded("work")
    with pytest.raises(ScenarioAdmissionError, match="starting"):
        restarted.get_or_create_scenario("work")
    assert restarted.start_scheduled_scenario(7, 1, "work").state == "admitted"
    assert restarted.has_loaded("work")


def test_restart_cleanup_uses_saved_owner_without_constructing_resources(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = deployment.open()
    first.start_scheduled_scenario(7, 1, "work")
    first.close()
    restarted = deployment.open()

    def forbidden(scenario: str, release_id: str | None = None, *, model_config: ModelConfig) -> Scenario:
        raise RuntimeError("cleanup must not construct a scenario")

    monkeypatch.setattr(restarted._registry._scenario_factory, "load_or_create", forbidden)
    result = restarted.cleanup_scheduled_scenario(7, "work", tenant_id=1)
    assert result.cleanup_complete and not restarted.has_scenario("work")
    restarted.close()
    again = deployment.open()
    assert again.scheduled_scenario_status("work") == result
    with pytest.raises(ScenarioAdmissionError, match="completed"):
        again.start_scheduled_scenario(7, 1, "work")


def test_cleanup_before_start_persists_a_tombstone(deployment: ScheduledDeployment) -> None:
    first = deployment.open()
    result = first.cleanup_scheduled_scenario(5, "never-started", tenant_id=1)
    assert result.cleanup_complete and not first.has_scenario("never-started")
    first.close()
    restarted = deployment.open()
    with pytest.raises(ScenarioAdmissionError, match="completed"):
        restarted.start_scheduled_scenario(5, 1, "never-started")
    assert restarted.scenario_admissions is not None
    assert restarted.scenario_admissions.admitted_by_name == {}
    assert restarted.scheduled_operations == {}


def test_unowned_registration_cannot_be_claimed_or_deleted(deployment: ScheduledDeployment) -> None:
    previous = deployment.open(scheduled=False)
    previous.get_or_create_scenario("legacy")
    previous.close()
    dispatcher = deployment.open()
    with pytest.raises(ScenarioAdmissionError, match="ownership migration"):
        dispatcher.start_scheduled_scenario(1, 1, "legacy")
    with pytest.raises(ScenarioAdmissionError, match="unassigned"):
        dispatcher.cleanup_scheduled_scenario(1, "legacy", tenant_id=1)
    assert dispatcher.has_scenario("legacy")
    assert dispatcher.scheduled_scenario_status("legacy") is None


def test_failed_start_archives_owned_registration_and_allows_name_reuse(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = deployment.open()

    def fail_wake(scenario: Scenario) -> None:
        raise RuntimeError("worker startup failed")

    with monkeypatch.context() as patch:
        patch.setattr(dispatcher, "_wake_training", fail_wake)
        status = dispatcher.start_scheduled_scenario(1, 1, "work")
    assert status.state == "cleanup_pending" and not status.cleanup_complete
    result = dispatcher.cleanup_scheduled_scenario(1, "work", tenant_id=1)
    assert result.state == "failed" and result.cleanup_complete
    assert not dispatcher.has_scenario("work")
    assert dispatcher.start_scheduled_scenario(2, 1, "work").state == "admitted"


def test_failed_construction_is_archived_without_reconstructing(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = deployment.open()
    factory = dispatcher._registry._scenario_factory
    original_recover = factory._recover

    def fail_recovery(*args: object, **kwargs: object) -> Scenario:
        raise RuntimeError("scenario recovery unavailable")

    monkeypatch.setattr(factory, "_recover", fail_recovery)
    status = dispatcher.start_scheduled_scenario(1, 1, "work")
    assert status.startup_failed and not status.cleanup_complete
    assert not dispatcher.has_loaded("work") and dispatcher.has_scenario("work")
    assert dispatcher.cleanup_scheduled_scenario(1, "work").cleanup_complete
    assert not dispatcher.has_scenario("work")
    assert dispatcher._registry._model_configs == {} and dispatcher._registry._scenario_locks == {}
    monkeypatch.setattr(factory, "_recover", original_recover)
    assert dispatcher.start_scheduled_scenario(2, 1, "work").state == "admitted"


def test_cleanup_cannot_acknowledge_before_in_flight_start_finishes(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = deployment.open()
    factory = dispatcher._registry._scenario_factory
    original_create = factory.load_or_create
    creating = threading.Event()
    continue_start = threading.Event()
    completed: list[ScenarioControlResult] = []

    def create(scenario: str, release_id: str | None = None, *, model_config: ModelConfig) -> Scenario:
        creating.set()
        continue_start.wait(5)
        return original_create(scenario, release_id, model_config=model_config)

    monkeypatch.setattr(factory, "load_or_create", create)

    def start() -> None:
        completed.append(dispatcher.start_scheduled_scenario(1, 1, "work"))

    starting = threading.Thread(target=start)
    starting.start()
    try:
        assert creating.wait(5)
        assert dispatcher.scenario_admissions is not None
        assert dispatcher.scenario_admissions.admitted_by_name == {}
        with pytest.raises(ScenarioBusy, match="lifecycle operation"):
            dispatcher.cleanup_scheduled_scenario(1, "work", tenant_id=1)
        with pytest.raises(ScenarioAdmissionError, match="starting"):
            dispatcher.get_or_create_scenario("work")
        status = dispatcher.scheduled_scenario_status("work")
        assert status is not None and status.state == "starting" and not status.cleanup_complete
    finally:
        continue_start.set()
        starting.join(5)
    assert not starting.is_alive() and completed[0].state == "admitted"
    assert dispatcher.cleanup_scheduled_scenario(1, "work").cleanup_complete


def test_idle_reclaim_rechecks_requests_cycles_and_new_activity(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = deployment.open()
    dispatcher.start_scheduled_scenario(1, 1, "work")
    admissions = dispatcher.scenario_admissions
    assert admissions is not None
    clock = [100.0]
    monkeypatch.setattr("reef.scenario.admission.time.time", lambda: clock[0])
    idle = dispatcher.scheduled_scenario_status("work")
    assert idle is not None and idle.idle_since == 100
    lease = admissions.acquire_request("work", 1)
    with pytest.raises(ScenarioBusy, match="request"):
        dispatcher.cleanup_scheduled_scenario(1, "work")
    busy = dispatcher.reclaim_scheduled_scenario(1, "work", generation=2, tenant_id=1, inactive_before=101)
    assert busy.state == "admitted" and busy.idle_since is None and busy.generation == 2
    admissions.release_request("work", lease)
    with dispatcher.local_cycle_lock("work"):
        busy = dispatcher.reclaim_scheduled_scenario(1, "work", generation=3, tenant_id=1, inactive_before=101)
        assert busy.state == "admitted" and busy.idle_since is None
    clock[0] = 102
    dispatcher.accept_record(AgentRecord.create(scenario="work", request_type=RequestType.INFERENCE, payload={}))
    refreshed = dispatcher.scheduled_scenario_status("work")
    assert refreshed is not None and refreshed.idle_since == 102
    refused = dispatcher.reclaim_scheduled_scenario(1, "work", generation=4, tenant_id=1, inactive_before=101)
    assert refused.state == "admitted" and not refused.reclaim_complete
    reclaimed = dispatcher.reclaim_scheduled_scenario(1, "work", generation=5, tenant_id=1, inactive_before=102)
    assert reclaimed.state == "reclaimed" and reclaimed.reclaim_complete and not reclaimed.cleanup_complete


@pytest.mark.parametrize("reclaim", [False, True])
def test_cleanup_joins_workers_outside_registry_locks_and_retries_do_not_wait(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch, reclaim: bool
) -> None:
    dispatcher = deployment.open()
    dispatcher.start_scheduled_scenario(1, 1, "work")
    began = threading.Event()
    finishing = threading.Event()
    release_worker = threading.Event()
    completed: list[ScenarioControlResult] = []

    def worker(scenario: str, component: str | None, ready: threading.Event) -> None:
        ready.wait(5)
        began.set()
        release_worker.wait(5)
        with dispatcher._registry.lock_for(scenario):
            finishing.set()

    monkeypatch.setattr(dispatcher, "_run_local_backend_worker", worker)
    dispatcher._start_local_backend_worker("work")
    assert began.wait(5)

    def cleanup() -> None:
        if reclaim:
            idle = dispatcher.scheduled_scenario_status("work")
            assert idle is not None and idle.idle_since is not None
            completed.append(
                dispatcher.reclaim_scheduled_scenario(
                    1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since
                )
            )
        else:
            completed.append(dispatcher.cleanup_scheduled_scenario(1, "work"))

    cleaning = threading.Thread(target=cleanup)
    # The thread's join is observed after cleanup has atomically changed the state.
    original_stop = dispatcher._stop_local_backend_worker
    detached = threading.Event()

    def stop(scenario: str):
        workers = original_stop(scenario)
        detached.set()
        return workers

    monkeypatch.setattr(dispatcher, "_stop_local_backend_worker", stop)
    cleaning.start()
    try:
        assert detached.wait(5)
        assert completed == []
        status = dispatcher.scheduled_scenario_status("work")
        assert status is not None and status.state == ("reclaim_pending" if reclaim else "cleanup_pending")
        assert not status.cleanup_complete and not status.reclaim_complete
        assert dispatcher.scenario_admissions is not None
        assert dispatcher.scenario_admissions.get("work") is None
        with pytest.raises(ScenarioAdmissionError, match="no admitted scenario"):
            dispatcher.scenario_admissions.acquire_request("work", 1)
        with pytest.raises(ScenarioBusy, match="lifecycle operation"):
            dispatcher.cleanup_scheduled_scenario(1, "work")
        with pytest.raises(ScenarioBusy, match="lifecycle operation"):
            dispatcher.start_scheduled_scenario(1, 1, "work")
    finally:
        release_worker.set()
        cleaning.join(5)
    assert finishing.is_set() and not cleaning.is_alive()
    assert len(completed) == 1
    assert completed[0].cleanup_complete is not reclaim
    assert completed[0].reclaim_complete is reclaim
    assert dispatcher.scheduled_operations == {}


@pytest.mark.parametrize("failing_resource", ("trainer", "store", "records"))
def test_cleanup_failure_keeps_quota_until_resources_close(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch, failing_resource: str
) -> None:
    dispatcher = deployment.open()
    dispatcher.start_scheduled_scenario(1, 1, "work")
    current = dispatcher.get_or_create_scenario("work")
    assert current is not None
    original_trainer_close = current.trainer.close
    original_store_close = current.store.close
    original_records_close = current.records.close
    attempts = {"trainer": 0, "store": 0, "records": 0}

    def close_trainer() -> None:
        attempts["trainer"] += 1
        if failing_resource == "trainer" and attempts["trainer"] == 1:
            raise RuntimeError("backend teardown unavailable")
        original_trainer_close()

    def close_store() -> None:
        attempts["store"] += 1
        if failing_resource == "store" and attempts["store"] == 1:
            raise RuntimeError("store teardown unavailable")
        original_store_close()

    def close_records() -> None:
        attempts["records"] += 1
        if failing_resource == "records" and attempts["records"] == 1:
            raise RuntimeError("record store teardown unavailable")
        original_records_close()

    monkeypatch.setattr(current.trainer, "close", close_trainer)
    monkeypatch.setattr(current.store, "close", close_store)
    monkeypatch.setattr(current.records, "close", close_records)
    pending = dispatcher.cleanup_scheduled_scenario(1, "work")
    assert pending.state == "cleanup_pending" and not pending.cleanup_complete
    assert not current._closed
    completed = dispatcher.cleanup_scheduled_scenario(1, "work")
    assert completed.cleanup_complete and current._closed
    assert attempts == {
        "trainer": 2 if failing_resource == "trainer" else 1,
        "store": 1 if failing_resource == "trainer" else 2,
        "records": 2 if failing_resource == "records" else 1,
    }
    current.close()
    assert dispatcher.cleanup_scheduled_scenario(1, "work") == completed


@pytest.mark.parametrize("creation_phase", ("binding", "publication"))
@pytest.mark.parametrize("failing_resource", ("trainer", "store"))
def test_construction_and_cleanup_double_failure_retains_real_resources(
    deployment: ScheduledDeployment,
    monkeypatch: pytest.MonkeyPatch,
    creation_phase: str,
    failing_resource: str,
) -> None:
    dispatcher = deployment.open()
    available = threading.Event()
    close_attempts: list[bool] = []
    original_trainer_close = Trainer.close
    original_store_close = CommitLogScenarioStore.close

    def fail_binding(recipe: Recipe) -> None:
        raise RuntimeError("private-model-detail: construction failed")

    def fail_publication(scenario: Scenario) -> None:
        raise RuntimeError("private-model-detail: publication failed")

    def close_trainer(trainer: Trainer) -> None:
        close_attempts.append(True)
        if not available.is_set():
            raise RuntimeError("private-model-detail: trainer unavailable")
        original_trainer_close(trainer)

    def close_store(store: CommitLogScenarioStore) -> None:
        close_attempts.append(True)
        if not available.is_set():
            raise RuntimeError("private-model-detail: store unavailable")
        original_store_close(store)

    with monkeypatch.context() as patch:
        if creation_phase == "binding":
            patch.setattr(Recipe, "inference_handler", property(fail_binding))
        else:
            patch.setattr(Scenario, "publish_shipped_content", fail_publication)
        if failing_resource == "trainer":
            patch.setattr(Trainer, "close", close_trainer)
        else:
            patch.setattr(CommitLogScenarioStore, "close", close_store)
        started = dispatcher.start_scheduled_scenario(1, 1, "work")
        assert started.state == "cleanup_pending" and not started.cleanup_complete
        assert not dispatcher.has_loaded("work")
        [resources] = dispatcher._registry.failed_constructions["work"]
        pending = dispatcher.cleanup_scheduled_scenario(1, "work")
        assert pending.state == "cleanup_pending" and not pending.cleanup_complete
        assert dispatcher.has_scenario("work")
        assert len(close_attempts) == 2
        available.set()
        completed = dispatcher.cleanup_scheduled_scenario(1, "work")
    assert len(close_attempts) == 3
    assert resources.trainers == () and resources.store is None
    assert dispatcher._registry.failed_constructions == {}
    assert completed.state == "failed" and completed.cleanup_complete
    assert not dispatcher.has_scenario("work")
    for path in (deployment.directory / "assignments").glob("*.json"):
        assert "private-model-detail" not in path.read_text()
    assert dispatcher.start_scheduled_scenario(2, 1, "work").state == "admitted"


def test_shutdown_closes_failed_construction_resources_without_confirming_quota_release(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = deployment.open()
    available = threading.Event()
    original_close = Trainer.close

    def fail_binding(recipe: Recipe) -> None:
        raise RuntimeError("construction failed")

    def close_trainer(trainer: Trainer) -> None:
        if not available.is_set():
            raise RuntimeError("trainer unavailable")
        original_close(trainer)

    monkeypatch.setattr(Recipe, "inference_handler", property(fail_binding))
    monkeypatch.setattr(Trainer, "close", close_trainer)
    dispatcher.start_scheduled_scenario(1, 1, "work")
    [resources] = dispatcher._registry.failed_constructions["work"]
    available.set()
    dispatcher.close()
    assert resources.trainers == () and resources.store is None
    assert dispatcher._registry.failed_constructions == {}
    status = dispatcher.scheduled_scenario_status("work")
    assert status is not None and status.state == "cleanup_pending" and not status.cleanup_complete


def test_model_validation_has_no_construction_or_storage_side_effects(deployment: ScheduledDeployment) -> None:
    dispatcher = deployment.open()
    dispatcher.validate_scheduled_model(None)
    with pytest.raises(ValueError):
        dispatcher.validate_scheduled_model(123)
    assert dispatcher.list_scenarios() == ()
    assert list((deployment.directory / "records").glob("*-model.json")) == []


def test_reclaim_preserves_commits_records_identity_and_resume_after_restart(deployment: ScheduledDeployment) -> None:
    from reef.train.types import TrainStepResult

    first = deployment.open()
    first.start_scheduled_scenario(1, 1, "work", generation=1)
    instance = first.get_or_create_scenario("work")
    assert instance is not None
    record = AgentRecord.create(scenario="work", request_type=RequestType.INFERENCE, payload={"text": "keep me"})
    first.accept_record(record)
    instance.trainer.apply_committed_state({"progress": 3})
    instance.commit(TrainStepResult(state={"progress": 3}))
    release = instance.current_artifact_ref()
    idle = first.scheduled_scenario_status("work")
    assert idle is not None and idle.idle_since is not None
    reclaimed = first.reclaim_scheduled_scenario(1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since)
    assert reclaimed.reclaim_complete and reclaimed.generation == 2
    assert not first.has_loaded("work") and first.has_scenario("work")
    assert instance._closed
    assert not list((deployment.directory / "records").glob("archived/*"))
    with pytest.raises(ScenarioAdmissionError, match="reclaimed"):
        first.get_or_create_scenario("work")
    first.close()
    resumed = deployment.open()
    restored = resumed.start_scheduled_scenario(1, 1, "work", generation=3)
    assert restored.state == "admitted" and restored.scenario_id == 1 and restored.generation == 3
    recovered = resumed.get_or_create_scenario("work")
    assert recovered is not None and recovered.scenario_step == 1
    assert recovered.current_artifact_ref() == release
    assert recovered.trainer.state == {"progress": 3}
    assert recovered.records.get("work", record.agent_record_id) is not None
    with pytest.raises(ScenarioAdmissionError, match="stale"):
        resumed.reclaim_scheduled_scenario(1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since)
    with pytest.raises(ScenarioAdmissionError, match="stale"):
        resumed.cleanup_scheduled_scenario(1, "work", generation=2, tenant_id=1)
    assert resumed.has_loaded("work")
    assert resumed.cleanup_scheduled_scenario(1, "work", generation=4, tenant_id=1).cleanup_complete


def test_explicit_delete_of_reclaimed_scenario_archives_state(deployment: ScheduledDeployment) -> None:
    dispatcher = deployment.open()
    dispatcher.start_scheduled_scenario(1, 1, "work")
    idle = dispatcher.scheduled_scenario_status("work")
    assert idle is not None and idle.idle_since is not None
    dispatcher.reclaim_scheduled_scenario(1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since)
    completed = dispatcher.cleanup_scheduled_scenario(1, "work", generation=3, tenant_id=1)
    assert completed.cleanup_complete and not completed.reclaim_complete
    assert not dispatcher.has_scenario("work")
    with pytest.raises(ScenarioAdmissionError, match="completed"):
        dispatcher.start_scheduled_scenario(1, 1, "work", generation=4)


@pytest.mark.parametrize("busy_reason", ("batch", "derivation", "error", "cycle"))
def test_non_idle_work_is_never_reclaimed(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch, busy_reason: str
) -> None:
    dispatcher = deployment.open()
    dispatcher.start_scheduled_scenario(1, 1, "work")
    current = dispatcher.get_or_create_scenario("work")
    assert current is not None
    idle = dispatcher.scheduled_scenario_status("work")
    assert idle is not None and idle.idle_since is not None
    if busy_reason == "batch":
        monkeypatch.setattr(current.trainer, "batch_ready", lambda: True)
    elif busy_reason == "derivation":
        monkeypatch.setattr(current.trainer.processor, "derivation_pending", lambda: True)
    elif busy_reason == "error":
        dispatcher._record_training_error("work", "training needs recovery")
    else:
        dispatcher.local_cycle_lock("work").acquire()
    try:
        result = dispatcher.reclaim_scheduled_scenario(
            1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since
        )
        assert result.state == "admitted" and result.idle_since is None
        assert dispatcher.get_or_create_scenario("work") is current and not current._closed
    finally:
        if busy_reason == "cycle":
            dispatcher.local_cycle_lock("work").release()


def test_reclaim_failure_remains_pending_until_resource_close_succeeds(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = deployment.open()
    dispatcher.start_scheduled_scenario(1, 1, "work")
    current = dispatcher.get_or_create_scenario("work")
    assert current is not None
    close = current.trainer.close
    available = threading.Event()

    def close_trainer() -> None:
        if not available.is_set():
            raise RuntimeError("temporary teardown error")
        close()

    monkeypatch.setattr(current.trainer, "close", close_trainer)
    idle = dispatcher.scheduled_scenario_status("work")
    assert idle is not None and idle.idle_since is not None
    pending = dispatcher.reclaim_scheduled_scenario(
        1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since
    )
    assert pending.state == "reclaim_pending" and not pending.reclaim_complete
    assert dispatcher.scenario_admissions is not None
    assert dispatcher.scenario_admissions.get("work") is None
    with pytest.raises(ScenarioAdmissionError, match="no admitted scenario"):
        dispatcher.scenario_admissions.acquire_request("work", 1)
    assert dispatcher.has_scenario("work")
    with pytest.raises(ScenarioAdmissionError, match="execution round"):
        dispatcher.start_scheduled_scenario(1, 1, "work", generation=3)
    available.set()
    result = dispatcher.reclaim_scheduled_scenario(
        1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since
    )
    assert result.reclaim_complete and current._closed
    assert dispatcher.scenario_admissions.get("work") is None
    assert (
        dispatcher.reclaim_scheduled_scenario(1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since)
        == result
    )
    assert dispatcher.start_scheduled_scenario(1, 1, "work", generation=2) == result
    assert dispatcher.scenario_admissions.get("work") is None
    assert dispatcher.start_scheduled_scenario(1, 1, "work", generation=3).state == "admitted"


@pytest.mark.parametrize("legacy_activity", [False, True])
def test_legacy_assignment_migration_does_not_restore_admission(tmp_path: Path, legacy_activity: bool) -> None:
    store = JSONScenarioAdmissionStore(tmp_path / "assignments")
    value = {
        "scenario_id": 1,
        "tenant_id": 1,
        "scenario_name": "work",
        "state": "admitted",
        "generation": 3,
        "cleanup_complete": False,
        "idle_since": 100.0,
    }
    if legacy_activity:
        value["last_activity"] = 90.0
    store.write("work", value)
    admissions = ScenarioAdmissions(store)
    assert admissions.get("work") is None
    restored = admissions.status("work")
    assert restored is not None and restored.state == "starting"
    assert restored.generation == 3 and restored.idle_since is None
    saved = store.read("work")
    assert isinstance(saved, dict)
    assert "state" not in saved and "last_activity" not in saved and "idle_since" not in saved
    assert ScenarioControlRecord.from_value(saved, "work").operation == "start"
    admissions.begin(1, 1, "work", generation=3)
    assert admissions.get("work") is None
    admissions.admit("work", 1)
    assert admissions.get("work") is not None


def test_legacy_tenant_mapping_is_explicit_and_durable(tmp_path: Path) -> None:
    from dataclasses import asdict

    directory = tmp_path / "assignments"
    store = JSONScenarioAdmissionStore(directory)
    value = asdict(ScenarioControlRecord(1, 1, "work", complete=True))
    value["tenant_id"] = "alice"
    value.pop("generation")
    store.write("work", value)
    with pytest.raises(ValueError, match="explicit tenant_id_map"):
        store.read("work")
    migrated = JSONScenarioAdmissionStore(directory, tenant_id_map={"alice": 7}).read("work")
    status = ScenarioControlRecord.from_value(migrated, "work")
    assert status.tenant_id == 7 and status.generation == 1
    assert store.read("work") == migrated


@pytest.mark.parametrize(
    "legacy_state,operation,complete",
    [
        ("starting", "start", False),
        ("admitted", "start", True),
        ("reclaim_pending", "reclaim", False),
        ("reclaimed", "reclaim", True),
        ("cleanup_pending", "cleanup", False),
        ("deleted", "cleanup", True),
        ("failed", "cleanup", True),
    ],
)
def test_legacy_lifecycle_records_migrate_without_admitting_instances(
    tmp_path: Path, legacy_state: str, operation: str, complete: bool
) -> None:
    store = JSONScenarioAdmissionStore(tmp_path / "assignments")
    store.write(
        "work",
        {
            "scenario_id": 1,
            "tenant_id": 1,
            "scenario_name": "work",
            "state": legacy_state,
            "generation": 4,
            "cleanup_complete": legacy_state in {"deleted", "failed"},
            "reclaim_complete": legacy_state == "reclaimed",
        },
    )
    admissions = ScenarioAdmissions(store)
    result = admissions.status("work")
    assert result is not None and result.scenario_id == 1 and result.generation == 4
    assert result.state == ("starting" if legacy_state == "admitted" else legacy_state)
    assert admissions.admitted_by_name == {}
    persisted = store.read("work")
    assert isinstance(persisted, dict)
    assert "state" not in persisted
    assert persisted["operation"] == operation and persisted["complete"] is complete


def test_reclaim_intent_write_failure_keeps_admission_and_resources(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = deployment.open()
    dispatcher.start_scheduled_scenario(1, 1, "work")
    instance = dispatcher.get_or_create_scenario("work")
    admissions = dispatcher.scenario_admissions
    assert instance is not None and admissions is not None
    idle = dispatcher.scheduled_scenario_status("work")
    assert idle is not None and idle.idle_since is not None
    write = admissions.storage.write

    def fail_reclaim_write(scenario_name: str, value: Mapping[str, object]) -> None:
        if value["operation"] == "reclaim":
            raise OSError("assignment storage unavailable")
        write(scenario_name, value)

    with monkeypatch.context() as patch:
        patch.setattr(admissions.storage, "write", fail_reclaim_write)
        with pytest.raises(OSError, match="assignment storage"):
            dispatcher.reclaim_scheduled_scenario(
                1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since
            )
    assert admissions.get("work") is not None
    assert dispatcher.get_or_create_scenario("work") is instance and not instance._closed
    assert dispatcher.scheduled_cleanups == {}
    assert dispatcher.reclaim_scheduled_scenario(
        1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since
    ).reclaim_complete


def test_reclaim_restart_preserves_model_and_selected_training_mode(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reef.train.processors.base import DataProcessor

    monkeypatch.setattr(DataProcessor, "supported_training_modes", frozenset({"auto", "manual", "hybrid"}))
    first = deployment.open()
    model = {"url": "http://provider.invalid/v1", "model": "test-model", "api": "openai"}
    first.start_scheduled_scenario(1, 1, "work", model=model, has_model=True)
    first.set_training_mode("work", "manual")
    idle = first.scheduled_scenario_status("work")
    assert idle is not None and idle.idle_since is not None
    result = first.reclaim_scheduled_scenario(1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since)
    assert result.reclaim_complete and result.training_mode == "manual"
    first.close()
    resumed = deployment.open()
    assert resumed.start_scheduled_scenario(1, 1, "work", generation=3).state == "admitted"
    current = resumed.get_or_create_scenario("work")
    assert current is not None and current.training_mode == "manual"
    assert current.model_config.view() == {**model, "has_api_key": False}
    assert resumed._registry.reload("work").training_mode == "manual"


def test_restart_completes_pending_reclaim_without_constructing_resources(
    deployment: ScheduledDeployment, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = deployment.open()
    first.start_scheduled_scenario(1, 1, "work")
    admissions = first.scenario_admissions
    assert admissions is not None
    admissions.advance_control("work", 1, 1, 2)
    idle = first.scheduled_scenario_status("work")
    assert idle is not None and idle.idle_since is not None
    assert admissions.begin_cleanup("work", 1, inactive_before=idle.idle_since, reclaim=True)
    first.close()
    restarted = deployment.open()

    def forbidden(scenario: str, release_id: str | None = None, *, model_config: ModelConfig) -> Scenario:
        raise RuntimeError("reclaim recovery must not construct resources")

    with monkeypatch.context() as patch:
        patch.setattr(restarted._registry._scenario_factory, "load_or_create", forbidden)
        result = restarted.reclaim_scheduled_scenario(1, "work", generation=2, tenant_id=1, inactive_before=100)
        assert result.reclaim_complete and restarted.has_scenario("work")
    assert restarted.start_scheduled_scenario(1, 1, "work", generation=3).state == "admitted"


def test_unread_input_blocks_reclaim_until_processed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from .test_component_trainers import HARNESS, WEIGHTS, _local_pair, _seeded_repository, _TwoTrainerRecipe

    with closing(
        Dispatcher(
            _TwoTrainerRecipe(backends=_local_pair(tmp_path)),
            _seeded_repository(tmp_path),
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=tmp_path / "records",
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
            require_scenario_admission=True,
            scenario_admission_store=JSONScenarioAdmissionStore(tmp_path / "assignments"),
        )
    ) as dispatcher:
        # Hold worker dispatch while keeping input acceptance and the readiness query real.
        monkeypatch.setattr(dispatcher, "_wake_training", lambda current: None)
        dispatcher.start_scheduled_scenario(1, 1, "work")
        current = dispatcher.get_or_create_scenario("work")
        assert current is not None
        current.records.append(AgentRecord.create(scenario="work", request_type=RequestType.INFERENCE, payload={}))
        assert current.trainer_for(HARNESS).has_unread_records()
        pending = dispatcher.scheduled_scenario_status("work")
        assert pending is not None and pending.idle_since is None
        for component in (HARNESS, WEIGHTS):
            assert current.prepare_training_step(component) is None
            assert not current.trainer_for(component).has_unread_records()
        idle = dispatcher.scheduled_scenario_status("work")
        assert idle is not None and idle.idle_since is not None
        result = dispatcher.reclaim_scheduled_scenario(
            1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since
        )
        assert result.reclaim_complete


def test_long_evaluation_is_not_idle_without_new_events(tmp_path: Path) -> None:
    from .test_component_trainers import (
        HARNESS,
        WEIGHTS,
        _ComponentBackend,
        _records,
        _seeded_repository,
        _SlowBackend,
        _TwoTrainerRecipe,
    )

    slow = _SlowBackend(HARNESS, tmp_path / "candidates")
    recipe = _TwoTrainerRecipe(backends={HARNESS: slow, WEIGHTS: _ComponentBackend(WEIGHTS, tmp_path / "candidates")})
    dispatcher = Dispatcher(
        recipe,
        _seeded_repository(tmp_path),
        local_artifact_dir=tmp_path / "staged",
        agent_record_dir=tmp_path / "records",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        require_scenario_admission=True,
        scenario_admission_store=JSONScenarioAdmissionStore(tmp_path / "assignments"),
    )
    try:
        dispatcher.start_scheduled_scenario(1, 1, "work")
        for record in _records(1, "work"):
            dispatcher.accept_record(record)
        assert slow.evaluating.wait(5)
        result = dispatcher.reclaim_scheduled_scenario(1, "work", generation=2, tenant_id=1, inactive_before=10**12)
        assert result.state == "admitted" and result.idle_since is None
        assert not result.reclaim_complete and dispatcher.has_loaded("work")
    finally:
        slow.unblock()
        dispatcher.close()


def test_manual_buffered_records_survive_reclaim_until_work_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .test_component_trainers import HARNESS, WEIGHTS, _local_pair, _records, _seeded_repository, _TwoTrainerRecipe

    recipe = _TwoTrainerRecipe(backends=_local_pair(tmp_path), hybrid_components=frozenset({HARNESS, WEIGHTS}))
    with closing(
        Dispatcher(
            recipe,
            _seeded_repository(tmp_path),
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=tmp_path / "records",
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
            require_scenario_admission=True,
            scenario_admission_store=JSONScenarioAdmissionStore(tmp_path / "assignments"),
        )
    ) as dispatcher:
        monkeypatch.setattr(dispatcher, "_wake_training", lambda current: None)
        dispatcher.start_scheduled_scenario(1, 1, "work")
        dispatcher.set_training_mode("work", "manual")
        current = dispatcher.get_or_create_scenario("work")
        assert current is not None
        records = _records(1, "work")
        for record in records:
            current.records.append(record)
        for component in (HARNESS, WEIGHTS):
            assert current.prepare_training_step(component) is None
            assert current.trainer_for(component).processor_status()["ready_units"] == 1
        idle = dispatcher.scheduled_scenario_status("work")
        assert idle is not None and idle.idle_since is not None
        assert dispatcher.reclaim_scheduled_scenario(
            1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since
        ).reclaim_complete
        assert dispatcher.start_scheduled_scenario(1, 1, "work", generation=3).state == "admitted"
        recovered = dispatcher.get_or_create_scenario("work")
        assert recovered is not None and recovered is not current and recovered.training_mode == "manual"
        assert recovered.prepare_training_step(HARNESS) is None
        assert recovered.trainer_for(HARNESS).processor_status()["ready_units"] == 1
        dispatcher.set_training_mode("work", "auto")
        result = recovered.prepare_training_step(HARNESS)
        assert result is not None
        batch = recovered.trainer_for(HARNESS).pending_batch
        assert batch is not None
        assert {record.agent_record_id for record in records} <= {
            record_id for item in batch.items for record_id in item.source_agent_record_ids
        }


def test_legacy_assignment_binds_central_generation_once_after_restart(deployment: ScheduledDeployment) -> None:
    from dataclasses import asdict

    first = deployment.open()
    started = first.start_scheduled_scenario(1, 1, "work")
    first.close()
    store = JSONScenarioAdmissionStore(deployment.directory / "assignments")
    legacy = asdict(started)
    legacy.pop("generation")
    legacy.pop("generation_bound")
    store.write("work", legacy)
    restored = deployment.open()
    status = restored.start_scheduled_scenario(1, 1, "work", generation=3)
    assert status.state == "admitted" and status.generation == 3 and status.generation_bound
    with pytest.raises(ScenarioAdmissionError, match="stale"):
        restored.cleanup_scheduled_scenario(1, "work", generation=2, tenant_id=1)
    restored.close()
    again = deployment.open()
    with pytest.raises(ScenarioAdmissionError, match="execution round"):
        again.start_scheduled_scenario(1, 1, "work", generation=4)
    assert again.start_scheduled_scenario(1, 1, "work", generation=3).state == "admitted"


def test_resume_does_not_resolve_original_moving_release_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .test_component_trainers import HARNESS, WEIGHTS, _local_pair, _records, _seeded_repository, _TwoTrainerRecipe

    with closing(
        Dispatcher(
            _TwoTrainerRecipe(backends=_local_pair(tmp_path)),
            _seeded_repository(tmp_path),
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=tmp_path / "records",
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
            require_scenario_admission=True,
            scenario_admission_store=JSONScenarioAdmissionStore(tmp_path / "assignments"),
        )
    ) as dispatcher:
        monkeypatch.setattr(dispatcher, "_wake_training", lambda current: None)
        assert dispatcher.start_scheduled_scenario(1, 1, "work", release_id="head").state == "admitted"
        current = dispatcher.get_or_create_scenario("work")
        assert current is not None
        base_release = current.current_artifact_ref()
        for record in _records(1, "work"):
            current.records.append(record)
        for component in (HARNESS, WEIGHTS):
            result = current.prepare_training_step(component)
            assert result is not None
            current.commit(result, component=component)
        committed_release = current.current_artifact_ref()
        assert committed_release != base_release
        idle = dispatcher.scheduled_scenario_status("work")
        assert idle is not None and idle.idle_since is not None
        assert dispatcher.reclaim_scheduled_scenario(
            1, "work", generation=2, tenant_id=1, inactive_before=idle.idle_since
        ).reclaim_complete
        resumed = dispatcher.start_scheduled_scenario(1, 1, "work", generation=3, release_id="head")
        assert resumed.state == "admitted"
        recovered = dispatcher.get_or_create_scenario("work")
        assert recovered is not None and recovered.current_artifact_ref() == committed_release
        assert recovered.scenario_step == 2
