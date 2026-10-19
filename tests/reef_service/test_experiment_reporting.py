"""A slow or failing tracker blocks no serving operation: its calls run on the report thread."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event

import pytest
from tests.support.artifact_versions import RecordingExperimentTracker, dispatcher, train
from tests.support.component_trainers import HARNESS, _dispatcher, _RecordingTracker, _scenario

from reef.artifact import Artifact
from reef.train import TrainStepResult


def test_rollbacks_and_reads_finish_while_the_tracker_blocks(tmp_path, monkeypatch):
    tracker = RecordingExperimentTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = value.get_or_create_scenario("math")
    initial = scenario.current_artifact_ref()
    entered, release = Event(), Event()
    record_rollback = tracker.record_rollback

    def blocked(event):
        entered.set()
        release.wait(10)
        record_rollback(event)

    try:
        train(value, 1)
        monkeypatch.setattr(tracker, "record_rollback", blocked)
        promoted = scenario.current_artifact_ref()
        value.rollback("math", initial.release_id)
        assert entered.wait(5)
        value.rollback("math", promoted.release_id)
        assert value.get_or_create_scenario("math") is scenario
        assert tracker.rollbacks == []
        release.set()
        value.registry.experiment_reports.flush()
        assert [event.step for event in tracker.rollbacks] == [2, 3]
    finally:
        release.set()
        value.close()


@pytest.mark.parametrize("callback", ["record", "correlation_metrics"])
@pytest.mark.parametrize("failing", [False, True])
def test_local_tracker_read_locks(tmp_path, monkeypatch, callback, failing):
    """A blocked ``record`` holds up nothing. ``correlation_metrics`` runs before the commit record is written,
    under the operation lock only, so release reads still answer."""
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
            operation = pool.submit(value.local_workers.process_step, "agent", HARNESS)
            try:
                assert entered.wait(5)
                if callback == "record":
                    assert operation.result(timeout=2)
                    assert value.get_or_create_scenario("agent") is scenario
                else:
                    pool.submit(scenario.release_queries.artifact_with_metrics).result(timeout=2)
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
def test_a_load_that_publishes_shipped_content_does_not_wait_for_the_tracker(
    tmp_path, monkeypatch, operation, nested, tracker_fails
):
    tracker = _RecordingTracker()
    value, backends = _dispatcher(tmp_path, experiment_tracker=tracker)
    original = _scenario(value)
    if operation == "load":
        with value.registry.lock_for("agent"):
            value.registry.remove("agent")
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
        with value.registry.lock_for("agent") if nested else nullcontext():
            if operation == "reload":
                return value.registry.reload("agent")
            return value.get_or_create_scenario("agent")

    monkeypatch.setattr(backends[HARNESS], "shipped_content_update", shipped)
    monkeypatch.setattr(tracker, "record", blocked)
    try:
        loaded = load()
        assert entered.wait(5), "shipped-content update did not reach tracker"
        assert loaded is not original and loaded.scenario_step == 1
        assert value.get_or_create_scenario("agent") is loaded
        release.set()
        value.registry.experiment_reports.flush()
        assert len(tracker.events) == 1
        assert tracker.events[0].metrics == loaded.store.history()[0].metrics
    finally:
        release.set()
        value.close()


def test_a_failed_load_still_reports_the_commit_that_landed(tmp_path, monkeypatch):
    from reef.scenario.scenario import Scenario

    tracker = _RecordingTracker()
    value, backends = _dispatcher(tmp_path, experiment_tracker=tracker)
    original = _scenario(value)
    update = tmp_path / "updated-harness"
    update.mkdir()
    (update / "harness.txt").write_text("updated shipped content")
    publish = Scenario.publish_shipped_content

    def shipped(state, tree):
        return TrainStepResult(state, artifact=Artifact.local(update), metrics={"shipped_content_update": True})

    def fail_after_publication(scenario):
        publish(scenario)
        raise RuntimeError("load failed after publication")

    def failing_record(event):
        tracker.events.append(event)
        raise ConnectionError("tracker unavailable")

    monkeypatch.setattr(backends[HARNESS], "shipped_content_update", shipped)
    monkeypatch.setattr(Scenario, "publish_shipped_content", fail_after_publication)
    monkeypatch.setattr(tracker, "record", failing_record)
    try:
        with pytest.raises(RuntimeError, match="load failed after publication"):
            value.registry.reload("agent")
        assert value.get_or_create_scenario("agent") is original
        value.registry.experiment_reports.flush()
        assert len(tracker.events) == 1
        assert tracker.events[0].context.step == 1
        assert original.store.history()[0].step == 1
    finally:
        value.close()
