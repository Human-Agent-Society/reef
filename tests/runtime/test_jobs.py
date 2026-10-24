"""The training-job values: what a job, a saved checkpoint and a job status may hold."""

from __future__ import annotations

import pytest

from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingJob, job_id_for
from reef.train.types import TrainingBatch


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"job_id": ""}, ValueError, "job_id must be a non-empty string"),
        ({"backend": ""}, ValueError, "backend must be a non-empty string"),
        ({"start": ""}, ValueError, "start must be a non-empty string or None"),
        ({"data": ()}, TypeError, "data must be a TrainingBatch"),
        ({"parameters": [("lr", 1.0)]}, TypeError, "parameters must be a mapping"),
        ({"parameters": {"lr": float("nan")}}, ValueError, "Out of range float"),
    ],
)
def test_a_training_job_checks_its_values_at_construction(changes, error, message):
    values = {"job_id": "job", "backend": "tinker", "start": None, "data": TrainingBatch("b"), "parameters": {}}
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
        ({"state": "running", "algorithm_state": {}}, "algorithm_state is set only when the job finished"),
        (
            {"state": "running", "checkpoints": (SavedCheckpoint(2, "/b"), SavedCheckpoint(1, "/a"))},
            "increasing step order",
        ),
    ],
)
def test_a_job_status_is_consistent(values, message):
    with pytest.raises(ValueError, match=message):
        JobStatus("job", **values)


@pytest.mark.unit
@pytest.mark.parametrize(("step", "path"), [(-1, "/a"), (True, "/a"), (0, "")])
def test_a_saved_checkpoint_has_a_step_and_a_path(step, path):
    with pytest.raises(ValueError):
        SavedCheckpoint(step, path)
