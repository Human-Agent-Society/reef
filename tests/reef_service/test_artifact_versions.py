"""Release listing and rollback tests (issue #78, phase 4)."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.artifact_versions import (
    RecordingExperimentTracker,
    RollbackRuntime,
    dispatcher,
    inference,
    report,
    train,
)
from tests.support.runtime_stubs import runtime_bindings

from reef.dispatcher import Dispatcher
from reef.observability import TrainingExperimentContext, TrainingExperimentEvent
from reef.recipe.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.runtime.interfaces import InferenceHandler
from reef.scenario import ReleaseNotRestorable
from reef.service.app import RequestService, create_app
from reef.storage.commit_log import CommitHistoryScenarioStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.types import TrainStepResult


class BlockingBackend(InferenceHandler):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.release_id: str | None = None

    async def inference(self, artifact, path, payload):
        del path, payload
        self.release_id = artifact.ref.release_id
        self.started.set()
        await self.finish.wait()
        return {"choices": [{"message": {"content": "ok"}}], "metadata": {"runtime_load_id": "w1"}}


@pytest.mark.unit
def test_versions_are_wal_backed_and_rollback_appends_a_new_commit(tmp_path) -> None:
    value, runtime, _ = dispatcher(tmp_path)
    created = value.get_or_create_scenario("math").releases()[0]["release_id"]
    train(value, 1)
    train(value, 2)
    train(value, 3)

    before = value.list_releases("math")
    assert len(before) == 4
    assert all(version["restorable"] for version in before)
    assert before[-1]["release_id"] == created
    target_version = before[-2]["release_id"]
    last_training_step = value.build_training_status()["scenarios"]["math"]["last_committed_step"]

    published = value.rollback("math", target_version)

    assert published.release_id != target_version
    assert (
        value.get_or_create_scenario("math").repository.resolve(published).local_path.joinpath("model.txt").read_text()
        == "w1"
    )
    assert runtime.restored == ["w1"]
    assert runtime.restore_order == ["training", "inference"]
    assert runtime.inference.inference_admission_status["open"]
    record = value.get_or_create_scenario("math").store.history()[-1]
    assert record.operation == "rollback"
    assert record.rollback_target_release_id == target_version
    assert record.artifact_ref == published
    status = value.build_training_status()["scenarios"]["math"]
    assert status["scenario_step"] == 4
    assert status["last_committed_step"] == last_training_step


@pytest.mark.unit
def test_rollback_retry_settles_the_recorded_release_without_republishing(tmp_path, monkeypatch) -> None:
    tracker = RecordingExperimentTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    created = value.get_or_create_scenario("math").releases()[0]["release_id"]
    train(value, 1)
    scenario = value.get_or_create_scenario("math")
    original = scenario.trainer.commit_applied
    should_fail = True

    def fail_after_effect(state):
        nonlocal should_fail
        original(state)
        if should_fail:
            should_fail = False
            raise RuntimeError("injected settlement failure")

    monkeypatch.setattr(scenario.trainer, "commit_applied", fail_after_effect)

    with pytest.raises(RuntimeError, match="injected settlement failure"):
        value.rollback("math", created)

    assert scenario.scenario_step == 1
    assert scenario.store.durable
    recorded = scenario.store.history()[-1]
    assert recorded.step == 2
    assert recorded.operation == "rollback"
    assert not scenario.runtime.inference_admission_status["open"]

    published = value.rollback("math", created)

    assert scenario.runtime.inference_admission_status["open"]
    assert published == recorded.artifact_ref
    assert scenario.scenario_step == 2
    assert scenario.store.history()[-1] == recorded
    assert len(scenario.store.history()) == 2
    # The tracker hears about the rollback on the report thread; wait for it rather than race it.
    value.registry.experiment_reports.flush()
    assert len(tracker.rollbacks) == 1
    assert tracker.rollbacks[0].produced_artifact_ref == recorded.artifact_ref


@pytest.mark.unit
def test_rollback_starts_a_new_experiment_run_segment(tmp_path) -> None:
    tracker = RecordingExperimentTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    train(value, 1)
    train(value, 2)
    target_version = value.list_releases("math")[-2]["release_id"]

    value.rollback("math", target_version)
    # The engine serves the restored checkpoint under the version that the restore returned.
    train(value, 4, "restored:w1")
    # The step is visible before the tracker hears about the commit; wait for the event rather than race it.
    for _ in range(1000):
        if len(tracker.events) == 3:
            break
        time.sleep(0.001)

    assert [(event.context.run_segment, event.context.run_step) for event in tracker.events] == [
        (0, 0),
        (0, 1),
        (3, 0),
    ]
    assert len(tracker.rollbacks) == 1
    rollback = tracker.rollbacks[0]
    assert rollback.step == 3
    assert rollback.run_segment == 0
    assert rollback.target_release_id == target_version


@pytest.mark.unit
def test_recovery_adopts_a_lost_rollback_record_without_treating_it_as_training(tmp_path) -> None:
    value, _, backend_factory = dispatcher(tmp_path)
    train(value, 1)
    train(value, 2)
    target_version = value.list_releases("math")[-2]["release_id"]
    value.rollback("math", target_version)

    scenario = value.get_or_create_scenario("math")
    assert scenario is not None
    assert isinstance(scenario.store, CommitHistoryScenarioStore)
    journal = scenario.store.commit_history
    assert journal is not None
    path = journal.path
    records = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")

    restarted = Dispatcher(
        TestPolicyRecipe(
            **runtime_bindings(RollbackRuntime(tmp_path / "restarted-export")),
            checkpoint_strategy=EveryNVersions(1),
        ),
        backend_factory,
        local_artifact_dir=tmp_path / "restarted-staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )
    recovered = restarted.get_or_create_scenario("math")
    assert recovered is not None
    adopted = recovered.store.history()[-1]

    assert adopted.operation == "rollback"
    assert adopted.operation_verified is True
    assert adopted.rollback_target_release_id == target_version
    assert recovered.committed_training_without_job_id is False


@pytest.mark.unit
def test_older_versions_and_version_catalog_survive_restart(tmp_path) -> None:
    value, _, backend_factory = dispatcher(tmp_path)
    for index in (1, 2, 3):
        train(value, index)
    scenario = value.get_or_create_scenario("math")

    version_one = scenario.releases()[-2]
    version_one_ref = scenario.repository.backend.resolve_release(version_one["release_id"])
    assert scenario.repository.resolve(version_one_ref).local_path.joinpath("model.txt").read_text() == "w1"

    restarted_runtime = RollbackRuntime(tmp_path / "restarted-export")
    restarted = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(restarted_runtime), checkpoint_strategy=EveryNVersions(1)),
        backend_factory,
        local_artifact_dir=tmp_path / "restarted-staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )
    recovered = restarted.get_or_create_scenario("math")
    assert [version["operation"] for version in recovered.releases()] == [
        "training",
        "training",
        "training",
        "creation",
    ]


@pytest.mark.unit
def test_live_version_is_listed_but_cannot_be_a_rollback_target(tmp_path) -> None:
    value, _, _ = dispatcher(tmp_path, checkpoint_every=2)
    train(value, 1)
    train(value, 2)

    versions = value.list_releases("math")
    live = next(version for version in versions if not version["checkpoint"])
    assert not live["restorable"]
    with pytest.raises(ReleaseNotRestorable, match="no durable checkpoint bytes"):
        value.rollback("math", live["release_id"])


@pytest.mark.unit
def test_request_records_the_artifact_ref_captured_before_inference(tmp_path) -> None:
    async def run() -> None:
        value, _, _ = dispatcher(tmp_path)
        train(value, 1)
        selected_version = value.get_or_create_scenario("math").current_artifact_ref().release_id
        backend = BlockingBackend()
        service = RequestService(value)
        request = asyncio.create_task(
            service.infer_with_data(
                {"x-reef-scenario": "math"},
                {"model": "reef", "messages": [{"role": "user", "content": "hi"}]},
                "/v1/chat/completions",
                backend,
            )
        )
        await backend.started.wait()

        train(value, 2)

        backend.finish.set()
        _, item = await request
        assert backend.release_id == selected_version
        assert item.artifact_ref is not None
        assert item.artifact_ref.release_id == selected_version

    asyncio.run(run())


@pytest.mark.unit
def test_version_and_rollback_http_api(tmp_path) -> None:
    async def run() -> None:
        value, _, _ = dispatcher(tmp_path)
        train(value, 1)
        train(value, 2)
        client = TestClient(TestServer(create_app(value)))
        await client.start_server()
        try:
            listed = await client.get("/reef/scenarios/math/releases")
            assert listed.status == 200
            listed_body = await listed.json()
            assert len(listed_body["releases"]) == 3
            target_version = listed_body["releases"][-2]["release_id"]

            rolled_back = await client.post(
                "/reef/scenarios/math/rollback",
                json={"release_id": target_version},
            )
            assert rolled_back.status == 200
            assert (await rolled_back.json())["release_id"] != target_version

            missing = await client.post(
                "/reef/scenarios/math/rollback",
                json={"release_id": "missing"},
            )
            assert missing.status == 404
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("component", ["training", "inference"])
def test_rollback_restore_failure_keeps_requests_closed_and_head_unchanged(tmp_path, monkeypatch, component):
    value, training, _ = dispatcher(tmp_path)
    created = value.get_or_create_scenario("math").releases()[0]["release_id"]
    train(value, 1)
    scenario = value.get_or_create_scenario("math")
    head = scenario.current_artifact_ref()

    def fail_restore(artifact):
        raise RuntimeError("restore failed")

    target = training if component == "training" else training.inference
    monkeypatch.setattr(target, "restore_checkpoint", fail_restore)
    with pytest.raises(RuntimeError, match="restore failed"):
        value.rollback("math", created)
    assert not training.inference.inference_admission_status["open"]
    assert scenario.current_artifact_ref() == head


@pytest.mark.unit
@pytest.mark.parametrize("tracker_fails", [False, True])
def test_commit_preserves_experiment_metadata_and_isolates_tracker_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracker_fails: bool
) -> None:
    tracker = RecordingExperimentTracker()

    def correlation(context: TrainingExperimentContext) -> dict[str, str]:
        if tracker_fails:
            raise ConnectionError("tracker unavailable")
        return {"experiment/run_id": f"run-{context.run_segment}"}

    record_event = tracker.record
    recorded = Event()

    def record(event: TrainingExperimentEvent) -> None:
        record_event(event)
        recorded.set()
        if tracker_fails:
            raise ConnectionError("tracker unavailable")

    monkeypatch.setattr(tracker, "correlation_metrics", correlation)
    monkeypatch.setattr(tracker, "record", record)
    value, runtime, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    try:
        train(value, 1)
        scenario = value.get_or_create_scenario("math")
        assert recorded.wait(5), "commit event was not reported"
        (event,) = tracker.events
        (commit,) = scenario.store.history()
        assert event.metrics == commit.metrics
        assert event.context.backend == "RuntimeCandidateBackend"
        assert event.context.backend_config["runtime"] == "RollbackRuntime"
        assert event.training_job_id == "job-0"
        assert event.produced_runtime_load_id == "w1"
        assert event.checkpoint_path == str(runtime.checkpoint_dir / "0")
        if tracker_fails:
            assert "experiment/run_id" not in commit.metrics
        else:
            assert commit.metrics["experiment/run_id"] == "run-0"
    finally:
        value.close()


@pytest.mark.unit
def test_experiment_events_follow_settled_state_and_resumed_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = RecordingExperimentTracker()
    value, runtime, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = value.get_or_create_scenario("math")
    initial = scenario.current_artifact_ref()
    observed = []

    def record(event) -> None:
        observed.append((event.context.step, scenario.scenario_step, scenario.current_artifact_ref()))

    def rollback(event) -> None:
        observed.append((event.step, scenario.scenario_step, scenario.current_artifact_ref()))
        observed.append(runtime.inference.inference_admission_status["open"])

    monkeypatch.setattr(tracker, "record", record)
    monkeypatch.setattr(tracker, "record_rollback", rollback)
    try:
        train(value, 1)
        for _ in range(1000):
            if observed:
                break
            time.sleep(0.001)
        trained = scenario.current_artifact_ref()
        restored = value.rollback("math", initial.release_id)
        value.registry.experiment_reports.flush()
        assert observed == [(1, 1, trained), (2, 2, restored), True]
    finally:
        value.close()


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["rollback", "promote"])
def test_a_slow_rollback_tracker_holds_up_neither_the_rollback_nor_scenario_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    tracker = RecordingExperimentTracker()
    value, runtime, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = value.get_or_create_scenario("math")
    initial = scenario.current_artifact_ref()
    entered, release = Event(), Event()
    record_rollback = tracker.record_rollback

    def record(event) -> None:
        entered.set()
        release.wait(10)
        record_rollback(event)

    try:
        train(value, 1)
        monkeypatch.setattr(tracker, "record_rollback", record)
        restored = value.rollback("math", initial.release_id, operation=operation)
        assert entered.wait(5), "rollback tracker was not called"
        assert runtime.inference.inference_admission_status["open"]
        assert value.get_or_create_scenario("math") is scenario
        release.set()
        value.registry.experiment_reports.flush()
        assert tracker.rollbacks[0].produced_artifact_ref == restored
        assert scenario.store.history()[-1].operation == operation
    finally:
        release.set()
        value.close()


@pytest.mark.unit
def test_slow_correlation_tracker_does_not_block_published_artifact_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = RecordingExperimentTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = value.get_or_create_scenario("math")
    initial = scenario.current_artifact_ref()
    entered, release, recorded = Event(), Event(), Event()
    record_event = tracker.record

    def correlation(context: TrainingExperimentContext) -> dict[str, str]:
        entered.set()
        release.wait(10)
        return {"experiment/run_id": f"run-{context.run_segment}"}

    def record(event: TrainingExperimentEvent) -> None:
        record_event(event)
        recorded.set()

    monkeypatch.setattr(tracker, "correlation_metrics", correlation)
    monkeypatch.setattr(tracker, "record", record)
    try:
        value.accept_record(inference(1))
        value.accept_record(report(1))
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                assert entered.wait(5), "correlation tracker was not called"
                read = executor.submit(scenario.release_queries.artifact_with_metrics)
                artifact, _ = read.result(timeout=2)
                assert artifact.ref == initial
                assert not recorded.is_set(), "commit must wait for correlation metadata"
            finally:
                release.set()
        assert recorded.wait(5), "commit event was not reported"
        (event,) = tracker.events
        (commit,) = scenario.store.history()
        assert commit.metrics["experiment/run_id"] == "run-0"
        assert event.metrics == commit.metrics
        assert event.context.step == commit.step == 1
        assert event.context.source_artifact_ref == initial
    finally:
        release.set()
        value.close()


@pytest.mark.unit
def test_slow_training_tracker_does_not_block_scenario_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = RecordingExperimentTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = value.get_or_create_scenario("math")
    entered, release, recorded = Event(), Event(), Event()
    record_event = tracker.record

    def record(event: TrainingExperimentEvent) -> None:
        entered.set()
        release.wait(10)
        record_event(event)
        recorded.set()

    monkeypatch.setattr(tracker, "record", record)
    try:
        value.accept_record(inference(1))
        value.accept_record(report(1))
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                assert entered.wait(5), "training tracker was not called"
                lookup = executor.submit(value.get_or_create_scenario, "math")
                assert lookup.result(timeout=2) is scenario
                assert not recorded.is_set(), "tracker must remain blocked during the read"
            finally:
                release.set()
        assert recorded.wait(5)
        assert len(tracker.events) == 1
    finally:
        release.set()
        value.close()


class CountingCheckpoints(CheckpointStrategy):
    """Record each policy decision while selecting the configured cadence."""

    def __init__(self, every: int) -> None:
        self.every = every
        self.versions: list[int] = []

    def should_checkpoint(self, scenario: str, version: int, result: TrainStepResult) -> bool:
        self.versions.append(version)
        return version % self.every == 0


@pytest.mark.unit
@pytest.mark.parametrize("checkpoint_every", [1, 2])
def test_exported_weights_use_one_checkpoint_decision_per_step(tmp_path, monkeypatch, checkpoint_every: int) -> None:
    value, runtime, _ = dispatcher(tmp_path, checkpoint_every=checkpoint_every)
    try:
        scenario = value.get_or_create_scenario("math")
        strategy = CountingCheckpoints(checkpoint_every)
        monkeypatch.setattr(scenario._committer, "_checkpoint_strategy", strategy)
        train(value, 1)
        train(value, 2)
        assert strategy.versions == [1, 2]
        assert [record.checkpoint for record in scenario.store.history()] == [checkpoint_every == 1, True]
        assert scenario.trainer.state["steps"] == 2
        assert runtime.trained == 2
    finally:
        value.close()
