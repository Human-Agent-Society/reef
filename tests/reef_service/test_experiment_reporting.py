"""A slow or failing tracker blocks no serving operation: its calls run on the report thread."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from tests.support.artifact_versions import RecordingExperimentTracker, dispatcher, train
from tests.support.component_trainers import HARNESS, _dispatcher, _RecordingTracker, _scenario


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
        # The job's submission, checkpoint and end are steps 1 to 3.
        assert [event.step for event in tracker.rollbacks] == [4, 5]
    finally:
        release.set()
        value.close()


@pytest.mark.parametrize("callback", ["record", "correlation_metrics"])
@pytest.mark.parametrize("failing", [False, True])
def test_local_tracker_read_locks(tmp_path, monkeypatch, callback, failing):
    """A blocked ``record`` holds up nothing. ``correlation_metrics`` runs before the commit record is written,
    under the operation lock only, so release reads still answer."""
    tracker = _RecordingTracker()
    # Loading a scenario wakes its workers: hold them, so that this test runs the one cycle.
    value, _ = _dispatcher(tmp_path, experiment_tracker=tracker, hold_local_cycles=True)
    scenario = _scenario(value, 1)
    value.local_workers.open()
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
