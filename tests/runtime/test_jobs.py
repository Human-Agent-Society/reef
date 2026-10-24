"""The training-job values: what a job, a saved checkpoint and a job status may hold."""

from __future__ import annotations

import pytest

from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingJob, job_id_for
from reef.train.algos.signals import PreparedJob
from reef.train.types import JobData


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"job_id": ""}, ValueError, "job_id must be a non-empty string"),
        ({"backend": ""}, ValueError, "backend must be a non-empty string"),
        ({"start": ""}, ValueError, "start must be a non-empty string or None"),
        ({"data": ()}, TypeError, "data must be a JobData"),
        ({"parameters": [("lr", 1.0)]}, TypeError, "parameters must be a mapping"),
        ({"parameters": {"lr": float("nan")}}, ValueError, "Out of range float"),
    ],
)
def test_a_training_job_checks_its_values_at_construction(changes, error, message):
    values = {"job_id": "job", "backend": "tinker", "start": None, "data": JobData("b"), "parameters": {}}
    TrainingJob(**values)
    with pytest.raises(error, match=message):
        TrainingJob(**{**values, **changes})


@pytest.mark.unit
def test_a_job_id_names_the_backend_start_data_and_parameters():
    job_id = job_id_for("tinker", None, ["row"], {"lr": 1.0})
    assert job_id == job_id_for("tinker", None, ["row"], {"lr": 1.0})
    assert job_id != job_id_for("slime", None, ["row"], {"lr": 1.0})
    assert job_id != job_id_for("tinker", "/weights", ["row"], {"lr": 1.0})
    assert job_id != job_id_for("tinker", None, ["other"], {"lr": 1.0})
    assert job_id != job_id_for("tinker", None, ["row"], {"lr": 2.0})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"state": "paused"}, "running, finished or failed"),
        ({"state": "failed"}, "error must be set exactly when the job failed"),
        ({"state": "running", "error": "boom"}, "error must be set exactly when the job failed"),
        (
            {"state": "running", "checkpoints": (SavedCheckpoint(2, "/b", "/b"), SavedCheckpoint(1, "/a", "/a"))},
            "increasing step order",
        ),
        ({"state": "failed", "error": ""}, "error must be a non-empty string"),
        ({"state": "finished", "metrics": {"loss": float("nan")}}, "Out of range float"),
    ],
)
def test_a_job_status_is_consistent(values, message):
    with pytest.raises(ValueError, match=message):
        JobStatus("job", **values)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("step", "path", "release"), [(-1, "/a", "/a"), (True, "/a", "/a"), (0, "", "/a"), (0, "/a", "")]
)
def test_a_saved_checkpoint_has_a_step_a_path_and_a_release(step, path, release):
    with pytest.raises(ValueError):
        SavedCheckpoint(step, path, release)


@pytest.mark.unit
def test_a_job_status_names_its_job_and_checkpoint_metrics_encode_as_json():
    with pytest.raises(ValueError, match="job_id must be a non-empty string"):
        JobStatus("", "running")
    with pytest.raises(ValueError, match="Out of range float"):
        SavedCheckpoint(0, "/a", "/a", {"loss": float("inf")})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ({"action": "retry"}, ValueError, "must be 'train' or 'skip'"),
        ({"action": "skip", "payload": {"rows": []}}, ValueError, "only a train job"),
        ({"action": "train", "payload": None}, ValueError, "only a train job"),
        ({"metrics": None}, TypeError, "must be mappings"),
    ],
)
def test_a_prepared_training_step_is_a_train_step_with_a_payload_or_a_skip(values, error, message):
    defaults = {"action": "train", "next_algorithm_state": {}, "metrics": {}, "payload": {"rows": []}}
    PreparedJob(**defaults)
    PreparedJob("skip", {}, {})
    with pytest.raises(error, match=message):
        PreparedJob(**{**defaults, **values})
