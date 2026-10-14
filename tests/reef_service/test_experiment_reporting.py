"""Tracker latency must not extend serving locks, including nested scenario loads."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from threading import Event

import pytest
from tests.support.artifact_versions import RecordingExperimentTracker, dispatcher, train
from tests.support.component_trainers import HARNESS, _dispatcher, _RecordingTracker, _scenario

from reef.artifact import Artifact
from reef.train import TrainStepResult


def test_queued_rollbacks_do_not_hold_registry_while_waiting_for_tracker(tmp_path, monkeypatch):
    tracker = RecordingExperimentTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = value.get_or_create_scenario("math")
    initial = scenario.current_artifact_ref()
    entered, release, waiting = Event(), Event(), Event()
    record_rollback = tracker.record_rollback

    def blocked(event):
        entered.set()
        release.wait(10)
        record_rollback(event)

    reporting = scenario.experiment_reporting

    @contextmanager
    def wait_for_reporting():
        waiting.set()
        with reporting():
            yield

    try:
        train(value, 1)
        monkeypatch.setattr(tracker, "record_rollback", blocked)
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(value.rollback, "math", initial.release_id)
            try:
                assert entered.wait(5)
                monkeypatch.setattr(scenario, "experiment_reporting", wait_for_reporting)
                second = pool.submit(value.rollback, "math", initial.release_id)
                assert waiting.wait(5)
                lookup = pool.submit(value.get_or_create_scenario, "math")
                assert lookup.result(timeout=2) is scenario
                assert not first.done() and not second.done()
            finally:
                release.set()
            first.result(timeout=5)
            second.result(timeout=5)
        assert [event.step for event in tracker.rollbacks] == [2, 3]
    finally:
        release.set()
        value.close()


@pytest.mark.parametrize("callback", ["record", "correlation_metrics"])
@pytest.mark.parametrize("failing", [False, True])
def test_local_tracker_read_locks(tmp_path, monkeypatch, callback, failing):
    tracker = _RecordingTracker()
    value, _ = _dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = _scenario(value, 1)
    entered, release = Event(), Event()

    def blocked(event):
        entered.set()
        release.wait(10)
        if failing:
            raise ConnectionError("tracker unavailable")
        if callback == "record":
            tracker.events.append(event)
        return {}

    monkeypatch.setattr(tracker, callback, blocked)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            operation = pool.submit(value._process_local_backend_step, "agent", HARNESS)
            try:
                assert entered.wait(5)
                if callback == "record":
                    read = pool.submit(value.get_or_create_scenario, "agent")
                else:
                    read = pool.submit(scenario.artifact_with_metrics)
                read.result(timeout=2)
                assert not operation.done()
            finally:
                release.set()
            assert operation.result(timeout=5)
        assert scenario.scenario_step == 1
    finally:
        release.set()
        value.close()


@pytest.mark.parametrize("operation", ["load", "reload"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("tracker_fails", [False, True])
def test_shipped_content_tracker_runs_after_registration(tmp_path, monkeypatch, operation, nested, tracker_fails):
    tracker = _RecordingTracker()
    value, backends = _dispatcher(tmp_path, experiment_tracker=tracker)
    original = _scenario(value)
    if operation == "load":
        with value._registry.lock_for("agent"):
            value._registry.remove("agent")
        original.close()
    update = tmp_path / "updated-harness"
    update.mkdir()
    (update / "harness.txt").write_text("updated shipped content")
    entered, release = Event(), Event()

    def shipped(state, tree):
        return TrainStepResult(state, artifact=Artifact.local(update), metrics={"shipped_content_update": True})

    def blocked(event):
        entered.set()
        release.wait(10)
        tracker.events.append(event)
        if tracker_fails:
            raise ConnectionError("tracker unavailable")

    def load():
        with value._registry.lock_for("agent") if nested else nullcontext():
            if operation == "reload":
                return value._registry.reload("agent")
            return value.get_or_create_scenario("agent")

    monkeypatch.setattr(backends[HARNESS], "shipped_content_update", shipped)
    monkeypatch.setattr(tracker, "record", blocked)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reload = pool.submit(load)
            try:
                assert entered.wait(5), "shipped-content update did not reach tracker"
                read = pool.submit(value.get_or_create_scenario, "agent")
                loaded = read.result(timeout=2)
                assert loaded is not original
                assert loaded.scenario_step == 1
                assert not reload.done(), "tracker must remain blocked during the read"
            finally:
                release.set()
            assert reload.result(timeout=5) is loaded
        assert len(tracker.events) == 1
        assert tracker.events[0].metrics == loaded.store.history()[0].metrics
    finally:
        release.set()
        value.close()


def test_failed_load_still_reports_landed_commit_outside_registry_lock(tmp_path, monkeypatch):
    from reef.scenario.scenario import Scenario

    tracker = _RecordingTracker()
    value, backends = _dispatcher(tmp_path, experiment_tracker=tracker)
    original = _scenario(value)
    update = tmp_path / "updated-harness"
    update.mkdir()
    (update / "harness.txt").write_text("updated shipped content")
    entered, release = Event(), Event()
    publish = Scenario.publish_shipped_content

    def shipped(state, tree):
        return TrainStepResult(state, artifact=Artifact.local(update), metrics={"shipped_content_update": True})

    def fail_after_publication(scenario):
        publish(scenario)
        raise RuntimeError("load failed after publication")

    def blocked(event):
        entered.set()
        release.wait(10)
        tracker.events.append(event)
        raise ConnectionError("tracker unavailable")

    monkeypatch.setattr(backends[HARNESS], "shipped_content_update", shipped)
    monkeypatch.setattr(Scenario, "publish_shipped_content", fail_after_publication)
    monkeypatch.setattr(tracker, "record", blocked)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reload = pool.submit(value._registry.reload, "agent")
            try:
                assert entered.wait(5), "landed event was lost during failed load"
                read = pool.submit(value.get_or_create_scenario, "agent")
                assert read.result(timeout=2) is original
            finally:
                release.set()
            with pytest.raises(RuntimeError, match="load failed after publication"):
                reload.result(timeout=5)
        assert len(tracker.events) == 1
        assert tracker.events[0].context.step == 1
        assert original.store.history()[0].step == 1
    finally:
        release.set()
        value.close()
