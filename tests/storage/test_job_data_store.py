from __future__ import annotations

import json

import pytest
from tests.support._trajectories import policy_trajectory

from reef.core.job_data import JobData, TaskSample
from reef.core.training_request import TrainingRequest
from reef.dispatcher.work_retention import EndedWorkSweeper
from reef.storage.job_data_store import JOB_FILE, LocalJobDataStore, read_job_data, write_job_data
from reef.storage.sql_coordination import memory_coordination
from reef.storage.work_queue import NewWork
from reef.train.job_queue import JOBS_QUEUE, job_from_item, job_kind


def task_dir(root, name):
    directory = root / name
    (directory / "tests").mkdir(parents=True)
    (directory / "instruction.md").write_text(f"Solve {name}.", encoding="utf-8")
    (directory / "tests" / "test.sh").write_text("exit 0", encoding="utf-8")
    return directory


def test_a_directory_keeps_the_samples_in_order_and_copies_each_task(tmp_path):
    trajectories = JobData("s:data:1", (policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0),) * 2)
    write_job_data(trajectories, tmp_path / "trajectories")
    assert read_job_data(tmp_path / "trajectories") == trajectories
    assert json.loads((tmp_path / "trajectories" / JOB_FILE).read_text()) == {
        "format": 1,
        "data_id": "s:data:1",
        "sample_count": 2,
        "kind": "trajectory",
    }

    tasks = JobData("s:data:2", (TaskSample(task_dir(tmp_path, "add")), TaskSample(task_dir(tmp_path / "b", "add"))))
    write_job_data(tasks, tmp_path / "tasks")
    read = read_job_data(tmp_path / "tasks")
    # Two tasks with one directory name keep apart: each has its own number.
    assert [sample.task_path for sample in read.samples] == [
        tmp_path / "tasks" / "tasks" / "0" / "add",
        tmp_path / "tasks" / "tasks" / "1" / "add",
    ]
    assert (read.samples[1].task_path / "tests" / "test.sh").read_text() == "exit 0"

    with pytest.raises(ValueError, match="both trajectories and tasks"):
        write_job_data(JobData("mixed", (*trajectories.samples, *tasks.samples)), tmp_path / "mixed")
    assert not (tmp_path / "mixed").exists()


def test_the_local_store_writes_a_job_once_and_deletes_it(tmp_path):
    store = LocalJobDataStore(tmp_path / "job-data")
    first = JobData("s:data:1", (policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0),))
    ref = store.ref("job-1", first)
    assert (ref.uri, ref.path) == ((tmp_path / "job-data" / "job-1").resolve().as_uri(), store.open(ref.uri))
    assert (ref.data_id, ref.sample_count, ref.kind) == ("s:data:1", 1, "trajectory")
    # A second write of the job keeps the first directory.
    assert store.write("job-1", JobData("other")) == ref.uri
    assert read_job_data(ref.path) == first
    store.delete("job-1")
    with pytest.raises(FileNotFoundError):
        store.open(ref.uri)
    with pytest.raises(ValueError, match="no safe directory name"):
        store.write("../job", first)


def test_a_queued_job_with_inline_data_gets_a_directory_and_its_directory_goes_with_its_item(tmp_path):
    store = LocalJobDataStore(tmp_path / "job-data")
    queue = memory_coordination(lease_seconds=300).work_queue
    request = TrainingRequest("prefer tests", "session-1", "release-1", id="r1")
    data = JobData("s:request:r1", (policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0),), request=request)
    # The payload of a job that an earlier Reef queued: the data is inline, with its request.
    payload = {"backend": "steps", "start": "steps:base", "data": data.to_dict(), "parameters": {}}
    queue.add(NewWork(JOBS_QUEUE, "j1", "s", job_kind("steps"), payload, serial_key="s"), now=0)

    job = job_from_item(queue.get(JOBS_QUEUE, "j1"), store)
    assert (job.request_id, job.data_id, job.kind) == ("r1", "s:request:r1", "trajectory")
    assert read_job_data(job.data_dir).samples == data.samples

    queue.take(JOBS_QUEUE, "training-1", [job_kind("steps")], now=1)
    queue.finish(JOBS_QUEUE, "j1", "training-1", {"state": "finished", "metrics": {}}, now=2)
    assert EndedWorkSweeper(queue, retention_seconds=1, job_data=store).sweep() == 1
    assert not job.data_dir.exists()
