"""Delayed samples in job data: the exact-sample checks and the JSON round trip."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from reef.core.job_data import DelayedSample, JobData, TaskSample, TrajectorySample

DISTRIBUTION = {"temperature": 0.6, "top_p": 1.0, "top_k": -1}


def sample(**training: object) -> TrajectorySample:
    fields: dict[str, object] = {
        "tokens": [1, 2, 3, 4, 5],
        "loss_mask": [1, 1],
        "rollout_log_probs": [-0.5, -1.25],
        "artifact_id": "artifact:a",
        "distribution": dict(DISTRIBUTION),
        **training,
    }
    document = {
        "schema_version": "ATIF-v1.6",
        "agent": {"name": "designer", "version": "1"},
        "steps": [{"step_id": 1, "source": "agent", "message": "env"}],
        "extra": {"reef": {"training": {key: value for key, value in fields.items() if value is not None}}},
    }
    return TrajectorySample(document, None, ("record-1",))


def test_round_trip_keeps_the_delayed_samples_after_the_tasks(tmp_path: Path) -> None:
    tasks = (TaskSample(tmp_path / "task-a"), TaskSample(tmp_path / "task-b"))
    delayed = (DelayedSample(sample(), 1.5), DelayedSample(sample(rollout_log_probs=[-0.1, -0.2]), -0.75))
    data = JobData("s:data:1", tasks, delayed=delayed)

    value = data.to_dict()

    assert [entry["advantage"] for entry in value["delayed"]] == [1.5, -0.75]
    assert JobData.from_dict(value) == data


def test_data_without_delayed_samples_keeps_its_json_and_reads_old_values(tmp_path: Path) -> None:
    data = JobData("s:data:1", (TaskSample(tmp_path / "task"),))

    value = data.to_dict()

    assert "delayed" not in value
    assert JobData.from_dict(value) == data


@pytest.mark.parametrize(
    ("training", "message"),
    [
        ({"tokens": []}, "sampled tokens"),
        ({"loss_mask": [1, 1, 1, 1, 1]}, "loss mask"),
        ({"loss_mask": [0, 0]}, "trained token"),
        ({"rollout_log_probs": [-0.5]}, "log-prob"),
        ({"rollout_log_probs": [-0.5, math.nan]}, "log-prob"),
        ({"rollout_log_probs": None}, "log-prob"),
        ({"artifact_id": None}, "artifact"),
        ({"distribution": None}, "distribution"),
        ({"distribution": {"temperature": 0.0, "top_p": 1.0, "top_k": -1}}, "temperature"),
        ({"distribution": {"temperature": 1.0, "top_p": 1.0, "top_k": 0}}, "top_k"),
    ],
)
def test_a_delayed_sample_needs_its_exact_sample(training: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DelayedSample(sample(**training), 1.0)


def test_a_delayed_sample_needs_a_finite_advantage() -> None:
    with pytest.raises(ValueError, match="finite"):
        DelayedSample(sample(), math.inf)
    with pytest.raises(TypeError, match="number"):
        DelayedSample(sample(), True)


def test_delayed_samples_ride_only_with_tasks(tmp_path: Path) -> None:
    delayed = (DelayedSample(sample(), 1.0),)
    with pytest.raises(ValueError, match="task samples"):
        JobData("s:data:1", (), delayed=delayed)
    with pytest.raises(ValueError, match="task samples"):
        JobData("s:data:1", (sample(), TaskSample(tmp_path / "task")), delayed=delayed)
