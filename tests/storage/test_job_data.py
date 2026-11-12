from __future__ import annotations

import json
from dataclasses import replace

import pytest
from tests.support.trajectories import policy_trajectory

from reef.core.job_data import JobData, TaskFiles, TaskSample
from reef.storage.data import MemoryDataStore
from reef.storage.job_data import JOB_FILE, SAMPLES_FILE, job_data_key, job_task_files, read_job_data, write_job_data
from reef.storage.tasks import LocalTaskFileStore, TaskStoreError


def task_dir(root, name, text="exit 0"):
    directory = root / name
    (directory / "tests").mkdir(parents=True)
    (directory / "tests" / "test.sh").write_text(text, encoding="utf-8")
    return directory


def stored_task(task_files, partition, directory):
    """The sample of the task directory ``directory``, as the task file store keeps it in ``partition``."""
    digest = task_files.write(partition, directory)
    return TaskSample(task_files.path(partition, digest), files=TaskFiles(partition, digest))


def test_a_directory_keeps_the_samples_in_order_and_names_each_task_in_the_task_file_store(tmp_path):
    trajectories = JobData("s:data:1", (policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0),) * 2)
    write_job_data(trajectories, tmp_path / "trajectories")
    assert read_job_data(tmp_path / "trajectories", (), "s:data:1") == trajectories
    assert job_task_files(tmp_path / "trajectories") == ()
    assert json.loads((tmp_path / "trajectories" / JOB_FILE).read_text()) == {
        "format": 2,
        "key": job_data_key(trajectories),
        "sample_count": 2,
        "kind": "trajectory",
    }

    task_files = LocalTaskFileStore(tmp_path / "task-files")
    # Two tasks with one directory name, in two partitions.
    first = stored_task(task_files, "one", task_dir(tmp_path / "a", "add"))
    second = stored_task(task_files, "two", task_dir(tmp_path / "b", "add", "exit 1"))
    tasks = JobData("s:data:2", (first, second))
    write_job_data(tasks, tmp_path / "tasks")
    # The directory holds the two lines and the header, and no file of a task.
    assert sorted(path.name for path in (tmp_path / "tasks").iterdir()) == [JOB_FILE, SAMPLES_FILE]
    lines = [json.loads(line) for line in (tmp_path / "tasks" / SAMPLES_FILE).read_text().splitlines()]
    assert [line["files"] for line in lines] == [
        {"partition": "one", "digest": first.files.digest, "name": "add"},
        {"partition": "two", "digest": second.files.digest, "name": "add"},
    ]
    assert all("task_path" not in line for line in lines)
    assert job_task_files(tmp_path / "tasks") == (first.files, second.files)
    paths = tuple(task_files.path(files.partition, files.digest) for files in job_task_files(tmp_path / "tasks"))
    assert read_job_data(tmp_path / "tasks", paths, "s:data:2") == tasks
    assert (
        read_job_data(tmp_path / "tasks", paths).samples[1].task_path / "tests" / "test.sh"
    ).read_text() == "exit 1"
    for wrong in ((), paths[:1], (*paths, paths[0])):
        with pytest.raises(ValueError, match="tasks than the"):
            read_job_data(tmp_path / "tasks", wrong)

    with pytest.raises(ValueError, match="both trajectories and tasks"):
        write_job_data(JobData("mixed", (*trajectories.samples, *tasks.samples)), tmp_path / "mixed")
    assert not (tmp_path / "mixed").exists()
    # A task directory that the task file store does not keep cannot go into a job data directory.
    with pytest.raises(ValueError, match="the task file store does not keep the task directory"):
        write_job_data(JobData("loose", (TaskSample(task_dir(tmp_path / "c", "add")),)), tmp_path / "loose")
    assert not (tmp_path / "loose").exists()


def test_the_key_names_the_samples_and_the_digest_of_each_task_without_a_read_of_its_files(tmp_path):
    sample = policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0)
    assert job_data_key(JobData("one", (sample,))) == job_data_key(JobData("two", (sample,)))
    assert job_data_key(JobData("one", (sample,))) != job_data_key(JobData("one", (sample, sample)))
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    first = stored_task(task_files, "one", task_dir(tmp_path / "1", "add"))
    same = stored_task(task_files, "one", task_dir(tmp_path / "2", "add"))
    changed = stored_task(task_files, "one", task_dir(tmp_path / "3", "add", "exit 1"))
    assert job_data_key(JobData("t", (first,))) == job_data_key(JobData("t", (same,)))
    assert job_data_key(JobData("t", (first,))) != job_data_key(JobData("t", (changed,)))
    # The key reads the sample, not the directory: a task whose directory is gone still has its key.
    gone = replace(first, task_path=tmp_path / "no-such-directory" / "add")
    assert job_data_key(JobData("t", (gone,))) == job_data_key(JobData("t", (first,)))


def test_the_data_store_gives_each_task_of_a_job_its_directory_from_the_task_file_store(tmp_path):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    store = MemoryDataStore(task_files)
    task = stored_task(task_files, "one", task_dir(tmp_path / "a", "add"))
    other = stored_task(task_files, "one", task_dir(tmp_path / "b", "sub"))
    first = store.write_job_directory(JobData("s:data:1", (task, other)))
    assert first.task_paths == (task.task_path, other.task_path)
    assert first.read() == JobData("s:data:1", (task, other))
    # A job that shares a task writes no task file: its directory names the same digest.
    second = store.write_job_directory(JobData("s:data:2", (task,)))
    assert second.uri != first.uri and second.task_paths == (task.task_path,)
    assert not any(path.is_dir() for path in second.path.iterdir())
    # A reference from the jobs queue opens with its tasks' directories again.
    opened = store.open_job_directory(first.to_dict())
    assert opened.task_paths == first.task_paths and opened.read() == first.read()
    # A task whose files the store does not hold fails the open, not the component that reads the data later.
    missing = store.write_job_directory(JobData("s:data:3", (replace(task, files=TaskFiles("one", "0" * 64)),)))
    with pytest.raises(TaskStoreError, match="holds no files with digest"):
        store.open_job_directory(missing.to_dict())
