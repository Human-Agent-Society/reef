"""A slow or failing tracker blocks no serving operation: its calls run on the report thread."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from tests.support.artifact_versions import RecordingExperimentTracker, dispatcher, train
from tests.support.component_trainers import HARNESS, _dispatcher, _RecordingTracker, _scenario
from tests.support.samples_service import make_samples

from reef.observability import ExperimentRun
from reef.train.job_queue import scenario_jobs


class RunRecordingTracker(RecordingExperimentTracker):
    """Has one open run per scenario and training segment, as the W&B tracker does."""

    def experiment_run(self, context):
        run_id = f"{context.scenario}-segment-{context.run_segment}"
        return ExperimentRun("wandb", "reef", run_id, "team", f"https://wandb.ai/team/reef/runs/{run_id}")


def test_the_job_listing_names_the_run_that_records_each_job(tmp_path):
    tracker = RunRecordingTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    try:
        train(value, 1)
        value.registry.experiment_reports.flush()
        (job,) = scenario_jobs(value.queue, "math")
        assert any(event.training_job_id == job["id"] for event in tracker.events)
        assert job["experiment"] == {
            "tracker": "wandb",
            "project": "reef",
            "entity": "team",
            "run_id": "math-segment-0",
            "url": "https://wandb.ai/team/reef/runs/math-segment-0",
        }
    finally:
        value.close()


def test_a_job_has_no_experiment_without_a_tracker(tmp_path):
    value, _, _ = dispatcher(tmp_path)
    try:
        train(value, 1)
        (job,) = scenario_jobs(value.queue, "math")
        assert job["state"] == "finished" and job["experiment"] is None
    finally:
        value.close()


def test_training_starts_and_reads_finish_while_the_tracker_blocks(tmp_path, monkeypatch):
    tracker = RecordingExperimentTracker()
    value, _, _ = dispatcher(tmp_path, experiment_tracker=tracker)
    scenario = value.get_or_create_scenario("math")
    # The creation's artifact is the scenario's root release.
    base_artifact = scenario.repository.start_artifact.release_id
    entered, release = Event(), Event()
    record_train_from = tracker.record_train_from

    def blocked(event):
        entered.set()
        release.wait(10)
        record_train_from(event)

    try:
        train(value, 1)
        monkeypatch.setattr(tracker, "record_train_from", blocked)
        trained = scenario.current_artifact_ref()
        value.train_from("math", base_artifact)
        assert entered.wait(5)
        value.train_from("math", trained.release_id)
        value.activate("math", base_artifact)
        assert value.get_or_create_scenario("math") is scenario
        assert tracker.starts == []
        release.set()
        value.registry.experiment_reports.flush()
        # The job's submission, checkpoint and end are steps 1 to 3.
        assert [event.step for event in tracker.starts] == [4, 5]
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
    # The rows went to the store directly: make their samples, as the samples service does.
    make_samples(scenario)
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
