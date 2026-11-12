from __future__ import annotations

import hashlib
import json

import pytest
from tests.support._trajectories import policy_trajectory
from tests.support.job_data import delayed_sample

from reef.core.job_data import JobData, TaskSample
from reef.storage.job_data import DELAYED_FILE, JOB_FILE, job_data_key, read_job_data, write_job_data


def task_dir(root, name, text="exit 0"):
    directory = root / name
    (directory / "tests").mkdir(parents=True)
    (directory / "tests" / "test.sh").write_text(text, encoding="utf-8")
    return directory


def test_a_directory_keeps_the_samples_in_order_and_copies_each_task(tmp_path):
    trajectories = JobData("s:data:1", (policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0),) * 2)
    write_job_data(trajectories, tmp_path / "trajectories")
    assert read_job_data(tmp_path / "trajectories", "s:data:1") == trajectories
    assert json.loads((tmp_path / "trajectories" / JOB_FILE).read_text()) == {
        "format": 1,
        "key": job_data_key(trajectories),
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


def test_the_key_names_the_samples_and_the_files_of_their_tasks(tmp_path):
    sample = policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0)
    assert job_data_key(JobData("one", (sample,))) == job_data_key(JobData("two", (sample,)))
    assert job_data_key(JobData("one", (sample,))) != job_data_key(JobData("one", (sample, sample)))
    first = JobData("t", (TaskSample(task_dir(tmp_path / "1", "add")),))
    same = JobData("t", (TaskSample(task_dir(tmp_path / "2", "add")),))
    changed = JobData("t", (TaskSample(task_dir(tmp_path / "3", "add", "exit 1")),))
    assert job_data_key(first) == job_data_key(same) != job_data_key(changed)


def test_a_directory_keeps_the_delayed_samples_of_a_task_job(tmp_path):
    tasks = (TaskSample(task_dir(tmp_path, "add")),)
    data = JobData("s:data:3", tasks, delayed=(delayed_sample(1.5), delayed_sample(-0.75)))
    write_job_data(data, tmp_path / "carried")
    read = read_job_data(tmp_path / "carried", "s:data:3")
    assert read.delayed == data.delayed
    assert [entry.advantage for entry in read.delayed] == [1.5, -0.75]
    assert len((tmp_path / "carried" / DELAYED_FILE).read_text().splitlines()) == 2
    # Data without delayed samples writes no file for them.
    write_job_data(JobData("s:data:4", tasks), tmp_path / "plain")
    assert not (tmp_path / "plain" / DELAYED_FILE).exists()
    assert read_job_data(tmp_path / "plain").delayed == ()


def test_the_key_names_the_delayed_samples_and_keeps_the_key_of_data_without_them(tmp_path):
    tasks = (TaskSample(task_dir(tmp_path, "add")),)
    plain = JobData("t", tasks)
    carried = JobData("t", tasks, delayed=(delayed_sample(1.0),))
    other = JobData("t", tasks, delayed=(delayed_sample(-1.0),))
    assert len({job_data_key(plain), job_data_key(carried), job_data_key(other)}) == 3
    assert job_data_key(carried) == job_data_key(JobData("u", tasks, delayed=(delayed_sample(1.0),)))
    # The key of data without delayed samples is the key of its samples alone, as before.
    digest = hashlib.sha256()
    for path in sorted(item for item in tasks[0].task_path.rglob("*") if item.is_file()):
        digest.update(path.relative_to(tasks[0].task_path).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    entry = {**plain.to_dict()["samples"][0], "task_path": "add"}
    digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
    assert job_data_key(plain) == digest.hexdigest()
