"""Release listing, activation and new training start tests (RFC #965)."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.artifact_versions import (
    ImportingComponent,
    RecordingExperimentTracker,
    dispatcher,
    inference,
    report,
    train,
)
from tests.support.runtime_stubs import runtime_bindings

from reef.dispatcher import Dispatcher
from reef.observability import TrainingExperimentContext, TrainingExperimentEvent
from reef.runtime.interfaces import InferenceHandler
from reef.runtime.jobs import SavedCheckpoint
from reef.service.app import RequestService, create_app
from reef.storage.commit_history import CommitHistoryScenarioStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.jobs import JobLoopState
from reef.train.trainer import Trainer


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


def stored_paths(tmp_path: Path) -> set[Path]:
    """Every file and directory the repository backend stores: a new release adds some."""
    return set((tmp_path / "repository").rglob("*"))


def restart(tmp_path: Path, backend_factory) -> tuple[Dispatcher, ImportingComponent]:
    """A new process on the same repository and commit storage."""
    component = ImportingComponent(tmp_path / "restarted-export")
    value = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(component, component.inference)),
        backend_factory,
        local_artifact_dir=tmp_path / "restarted-staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
    )
    return value, component


@pytest.mark.unit
def test_activate_serves_an_existing_release_and_writes_no_artifact(tmp_path) -> None:
    value, runtime, _ = dispatcher(tmp_path)
    train(value, 1)
    train(value, 2)
    train(value, 3)
    scenario = value.get_or_create_scenario("math")

    before = value.list_artifacts("math")
    # Each job commits its submission, its checkpoint and its end; only the checkpoint is an artifact.
    assert [row["operation"] for row in before] == ["training", "training", "training", "creation"]
    assert [row["active"] for row in before] == [True, False, False, False]
    assert all(row["artifact_id"] == row["release_id"] and row["component"] for row in before)
    target = before[-2]
    last_training_step = value.build_training_status()["scenarios"]["math"]["last_committed_step"]
    checkpoint = scenario.repository.checkpoint_artifact
    stored = stored_paths(tmp_path)
    trainer_state = dict(scenario.trainer.state)
    runtime.restore_order.clear()
    served_before = len(runtime.activated)

    activated = value.activate("math", target["artifact_id"])

    # The release serves as it was made: its own id and bytes, no copy, no new release.
    assert (activated.artifact_id, activated.release_id, activated.content_id) == (
        target["artifact_id"],
        target["release_id"],
        target["content_id"],
    )
    assert activated.component == target["component"]
    assert scenario.current_artifact_ref().release_id == target["release_id"]
    assert scenario.repository.checkpoint_artifact == checkpoint
    assert scenario.repository.backend.current() == checkpoint
    assert stored_paths(tmp_path) == stored
    assert runtime.activated[served_before:] == ["w1"]
    # Activation leaves every trainer's start alone: the component imports nothing.
    assert runtime.restore_order == ["inference"]
    assert scenario.trainer.state == trainer_state
    record = scenario.store.history()[-1]
    assert record.operation == "activate"
    assert record.artifact_ref == scenario.current_artifact_ref()
    assert record.checkpoint is False and record.algorithm_state is None and record.component is None
    after = value.list_artifacts("math")
    assert len(after) == len(before)
    assert [row["artifact_id"] for row in after if row["active"]] == [target["artifact_id"]]
    status = value.build_training_status()["scenarios"]["math"]
    assert status["scenario_step"] == 10
    assert status["last_committed_step"] == last_training_step
    # Activating the active artifact changes nothing.
    assert value.activate("math", target["artifact_id"]) == activated
    assert len(scenario.store.history()) == 10
    value.close()


@pytest.mark.unit
def test_train_from_sets_the_next_job_start_and_leaves_serving_alone(tmp_path) -> None:
    value, runtime, _ = dispatcher(tmp_path)
    train(value, 1)
    train(value, 2)
    scenario = value.get_or_create_scenario("math")
    served = scenario.current_artifact_ref()
    target = value.list_artifacts("math")[-2]
    runtime.restore_order.clear()
    served_before = len(runtime.activated)

    start = value.train_from("math", target["artifact_id"])

    assert (start.artifact_id, start.release_id) == (target["artifact_id"], target["release_id"])
    assert scenario.current_artifact_ref() == served
    assert runtime.activated[served_before:] == []
    assert runtime.restore_order == ["training"]
    record = scenario.store.history()[-1]
    assert (record.operation, record.artifact_ref.release_id, record.component) == (
        "train_from",
        start.release_id,
        None,
    )
    assert [row["artifact_id"] for row in value.list_artifacts("math") if row["active"]] == [served.release_id]
    imported = str(runtime.imported[-1])
    assert JobLoopState.from_dict(scenario.trainer.state).start == imported
    # The next job starts from the imported release; the engine still serves w2.
    train(value, 3, "w2")
    assert runtime.submitted[-1].start == imported
    value.close()


@pytest.mark.unit
def test_activation_and_training_start_survive_a_restart(tmp_path) -> None:
    value, _, backend_factory = dispatcher(tmp_path)
    created = value.get_or_create_scenario("math").artifacts()[0]
    train(value, 1)
    train(value, 2)
    first = value.list_artifacts("math")[-2]
    activated = value.activate("math", first["artifact_id"])
    value.train_from("math", created["artifact_id"])
    value.close()

    restarted, component = restart(tmp_path, backend_factory)
    recovered = restarted.get_or_create_scenario("math")

    # The active artifact is the last activation in the log, not the newest checkpoint.
    assert recovered.current_artifact_ref().release_id == activated.release_id
    assert component.activated == ["w1"]
    assert [row["artifact_id"] for row in recovered.artifacts() if row["active"]] == [first["artifact_id"]]
    # The training start is taken again: the trainer's own commits do not record it yet.
    assert JobLoopState.from_dict(recovered.trainer.state).start == str(component.imported[-1])
    train(restarted, 3, "w1")
    assert component.submitted[-1].start == str(component.imported[-1])
    restarted.close()


@pytest.mark.unit
def test_a_lost_activation_record_leaves_the_release_the_log_names(tmp_path) -> None:
    value, _, backend_factory = dispatcher(tmp_path)
    train(value, 1)
    train(value, 2)
    scenario = value.get_or_create_scenario("math")
    newest = scenario.current_artifact_ref()
    value.activate("math", value.list_artifacts("math")[-2]["artifact_id"])
    assert isinstance(scenario.store, CommitHistoryScenarioStore)
    last_step = scenario.store.history()[-1].step
    value.close()
    # Lose the activation behind the store's back, as a crash before its append would.
    database = tmp_path / "agent-record" / f"{hashlib.sha256(b'math').hexdigest()}.sqlite3"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("DELETE FROM scenario_commit WHERE step = ?", (last_step,))

    restarted, _ = restart(tmp_path, backend_factory)
    recovered = restarted.get_or_create_scenario("math")

    assert recovered.current_artifact_ref() == newest
    assert recovered.store.history()[-1].operation == "training"
    restarted.close()


@pytest.mark.unit
def test_train_from_starts_a_new_experiment_run_segment_and_activate_does_not(tmp_path) -> None:
    tracker = RecordingExperimentTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    train(value, 1)
    train(value, 2)
    created, target_version = (row["artifact_id"] for row in value.list_artifacts("math")[:-3:-1])

    value.activate("math", target_version)
    value.train_from("math", created)
    # The engine serves the activated checkpoint under the version that its weights name.
    train(value, 3, "w1")
    value.registry.experiment_reports.flush()

    # Each job reports three commits; the activation at step 7 reports nothing, the new start at step 8 opens the
    # next segment.
    assert [(event.context.run_segment, event.context.run_step) for event in tracker.events] == [
        *((0, run_step) for run_step in range(6)),
        *((8, run_step) for run_step in range(3)),
    ]
    (start,) = tracker.starts
    assert (start.step, start.run_segment) == (8, 0)
    assert start.start_artifact_ref.release_id == created
    # The steps after the new start name the training start as their source, not the active release.
    assert {event.context.source_artifact_ref.release_id for event in tracker.events[6:7]} == {created}
    value.close()


@pytest.mark.unit
def test_older_versions_and_version_catalog_survive_restart(tmp_path) -> None:
    value, _, backend_factory = dispatcher(tmp_path)
    for index in (1, 2, 3):
        train(value, index)
    scenario = value.get_or_create_scenario("math")

    version_one = scenario.artifacts()[-2]
    version_one_ref = scenario.repository.backend.resolve_release(version_one["release_id"])
    assert scenario.repository.resolve(version_one_ref).local_path.joinpath("model.txt").read_text() == "w1"

    restarted, _ = restart(tmp_path, backend_factory)
    recovered = restarted.get_or_create_scenario("math")
    assert [version["operation"] for version in recovered.releases()] == [*(["training"] * 9), "creation"]
    assert recovered.artifacts() == scenario.artifacts()


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
def test_activate_and_train_from_http_api(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        value, runtime, _ = dispatcher(tmp_path)
        train(value, 1)
        train(value, 2)
        client = TestClient(TestServer(create_app(value)))
        await client.start_server()
        try:
            listed = await client.get("/reef/scenarios/math/artifacts")
            assert listed.status == 200
            artifacts = (await listed.json())["artifacts"]
            assert [row["operation"] for row in artifacts] == ["training", "training", "creation"]
            target = artifacts[-2]
            component = target["component"]

            activated = await client.post("/reef/scenarios/math/activate", json={"artifact_id": target["artifact_id"]})
            assert activated.status == 200
            assert await activated.json() == {
                "scenario": "math",
                "component": component,
                "artifact_id": target["artifact_id"],
                "release_id": target["release_id"],
                "content_id": target["content_id"],
            }
            body = {"artifact_id": target["artifact_id"], "component": component}
            started = await client.post("/reef/scenarios/math/train-from", json=body)
            assert started.status == 200
            assert await started.json() == {
                "scenario": "math",
                "component": component,
                "artifact_id": target["artifact_id"],
            }

            for path in ("activate", "train-from"):
                missing = await client.post(f"/reef/scenarios/math/{path}", json={"artifact_id": "missing"})
                assert missing.status == 404
                unknown = await client.post(f"/reef/scenarios/other/{path}", json={"artifact_id": "missing"})
                assert unknown.status == 404
                empty = await client.post(f"/reef/scenarios/math/{path}", json={"artifact_id": " "})
                assert empty.status == 400
                other = await client.post(
                    f"/reef/scenarios/math/{path}", json={"artifact_id": target["artifact_id"], "component": "other"}
                )
                assert other.status == 400

            def refuse(release):
                raise ValueError("these weights cannot train")

            monkeypatch.setattr(runtime, "import_start", refuse)
            refused = await client.post("/reef/scenarios/math/train-from", json={"artifact_id": target["artifact_id"]})
            assert refused.status == 400
            assert "cannot train" in await refused.text()

            monkeypatch.setattr(Trainer, "has_pending_commit", property(lambda self: True))
            created = artifacts[-1]["artifact_id"]
            for path in ("activate", "train-from"):
                waiting = await client.post(f"/reef/scenarios/math/{path}", json={"artifact_id": created})
                assert waiting.status == 409
        finally:
            await client.close()
            value.close()

    asyncio.run(run())


@pytest.mark.parametrize("component", ["training", "inference"])
def test_a_failed_activation_or_training_start_changes_nothing(tmp_path, monkeypatch, component):
    value, training, _ = dispatcher(tmp_path)
    created = value.get_or_create_scenario("math").artifacts()[0]["artifact_id"]
    train(value, 1)
    scenario = value.get_or_create_scenario("math")
    head = scenario.current_artifact_ref()
    trainer_state = dict(scenario.trainer.state)

    def fail_restore(*args, **kwargs):
        raise RuntimeError("restore failed")

    if component == "training":
        monkeypatch.setattr(training, "import_start", fail_restore)
        operation = value.train_from
    else:
        monkeypatch.setattr(training.inference, "activate_checkpoint", fail_restore)
        operation = value.activate
    try:
        with pytest.raises(RuntimeError, match="restore failed"):
            operation("math", created)
        assert scenario.current_artifact_ref() == head
        assert scenario.trainer.state == trainer_state
        assert len(scenario.store.history()) == 3
    finally:
        value.close()


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
        value.registry.experiment_reports.flush()
        commits = scenario.store.history()
        (job,) = runtime.submitted
        # The job's submission, its checkpoint and its end each report one event.
        assert [event.metrics for event in tracker.events] == [commit.metrics for commit in commits]
        assert [event.training_job_id for event in tracker.events] == [job.job_id] * 3
        assert [event.checkpoint_path for event in tracker.events] == [None, str(runtime.checkpoint_root / "1"), None]
        for event in tracker.events:
            assert event.context.backend == "stub"
            assert event.context.backend_config == {
                "job_plan": "fixed",
                "parameters": {"method": {"objective": "sft"}},
            }
        for commit in commits:
            if tracker_fails:
                assert "experiment/run_id" not in commit.metrics
            else:
                assert commit.metrics["experiment/run_id"] == "run-0"
    finally:
        value.close()


@pytest.mark.unit
def test_experiment_events_follow_settled_state_and_a_new_training_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = RecordingExperimentTracker()
    value, runtime, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = value.get_or_create_scenario("math")
    initial = scenario.current_artifact_ref()
    observed = []

    def record(event) -> None:
        observed.append((event.context.step, scenario.scenario_step, scenario.current_artifact_ref()))

    def record_start(event) -> None:
        observed.append((event.step, scenario.scenario_step, scenario.current_artifact_ref()))
        observed.append(runtime.inference.inference_admission_status["open"])

    monkeypatch.setattr(tracker, "record", record)
    monkeypatch.setattr(tracker, "record_train_from", record_start)
    try:
        train(value, 1)
        value.registry.experiment_reports.flush()
        trained = scenario.current_artifact_ref()
        # Each event follows its own settled commit: the scenario is at that step or later.
        assert [event_step for event_step, _, _ in observed] == [1, 2, 3]
        assert all(scenario_step >= event_step for event_step, scenario_step, _ in observed)
        assert observed[-1] == (3, 3, trained)
        observed.clear()
        value.train_from("math", initial.release_id)
        value.registry.experiment_reports.flush()
        # A new start leaves serving alone: the trained release still serves.
        assert observed == [(4, 4, trained), True]
    finally:
        value.close()


@pytest.mark.unit
def test_a_slow_tracker_holds_up_neither_the_training_start_nor_scenario_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = RecordingExperimentTracker()
    value, runtime, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = value.get_or_create_scenario("math")
    initial = scenario.current_artifact_ref()
    entered, release = Event(), Event()
    record_train_from = tracker.record_train_from

    def record(event) -> None:
        entered.set()
        release.wait(10)
        record_train_from(event)

    try:
        train(value, 1)
        monkeypatch.setattr(tracker, "record_train_from", record)
        start = value.train_from("math", initial.release_id)
        assert entered.wait(5), "the tracker was not called"
        assert runtime.inference.inference_admission_status["open"]
        assert value.get_or_create_scenario("math") is scenario
        release.set()
        value.registry.experiment_reports.flush()
        assert tracker.starts[0].start_artifact_ref.release_id == start.release_id
        assert scenario.store.history()[-1].operation == "train_from"
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
        value.registry.experiment_reports.flush()
        event = tracker.events[0]
        commit = scenario.store.history()[0]
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
        # The job's submission, its checkpoint and its end.
        for _ in range(5000):
            if len(tracker.events) == 3:
                break
            time.sleep(0.001)
        assert len(tracker.events) == 3
    finally:
        release.set()
        value.close()


@pytest.mark.unit
@pytest.mark.parametrize("checkpoints", [0, 2])
def test_each_saved_checkpoint_becomes_an_artifact_version(tmp_path, monkeypatch, checkpoints: int) -> None:
    value, runtime, _ = dispatcher(tmp_path)

    def save_checkpoints(job):
        saved = []
        for step in range(checkpoints):
            path = runtime.checkpoint_root / f"{len(runtime.submitted)}-{step}"
            path.mkdir(parents=True)
            (path / "model.txt").write_text(path.name)
            saved.append(SavedCheckpoint(step, str(path), str(path)))
        return tuple(saved)

    monkeypatch.setattr(runtime, "train", save_checkpoints)
    try:
        scenario = value.get_or_create_scenario("math")
        train(value, 1, scenario.serving_runtime_load_id())
        train(value, 2, scenario.serving_runtime_load_id())
        assert [record.checkpoint for record in scenario.store.history()] == [False, *[True] * checkpoints, False] * 2
        state = JobLoopState.from_dict(scenario.trainer.state)
        assert state.finished_jobs == runtime.trained == 2
        # Training is linear: the second job starts from the first job's last checkpoint.
        expected_start = str(runtime.checkpoint_root / f"1-{checkpoints - 1}") if checkpoints else None
        assert runtime.submitted[1].start == expected_start
    finally:
        value.close()
