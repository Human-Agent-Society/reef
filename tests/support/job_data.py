"""Job data directories for tests that build a ``TrainingJob`` without a job loop, and delayed samples of job data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from tests.support.trajectories import policy_trajectory

from reef.core.job_data import DelayedSample, JobData, TaskFiles, TaskSample
from reef.storage.data.job_data import JobDataRef
from reef.storage.data.store import MemoryDataStore
from reef.storage.data.tasks import LocalTaskFileStore

#: The partition of the task file store that keeps the tasks of a test's job data.
PARTITION = "test-tasks"
#: The stores of the references that this module gave: a store deletes its files when it is collected.
STORES: list[MemoryDataStore] = []

#: The sampling distribution of the samples that ``delayed_sample`` makes.
DELAYED_DISTRIBUTION = {"temperature": 0.6, "top_p": 1.0, "top_k": -1}


def job_data_ref(data: JobData) -> JobDataRef:
    """``data`` written as a job data directory by a new in-memory data store. The store's task file store first
    takes the directory of each task that no store keeps, as an import of the task does."""
    task_files = LocalTaskFileStore()
    samples = tuple(
        (
            replace(sample, files=TaskFiles(PARTITION, task_files.write(PARTITION, sample.task_path)))
            if isinstance(sample, TaskSample) and sample.files is None
            else sample
        )
        for sample in data.samples
    )
    store = MemoryDataStore(task_files)
    STORES.append(store)
    return store.write_job_directory(replace(data, samples=samples))


def delayed_sample(
    advantage: float = 1.0, *, tokens: int = 5, distribution: Mapping[str, object] = DELAYED_DISTRIBUTION
) -> DelayedSample:
    """A recorded reply of ``tokens`` tokens, the last two its response, with its exact sample and ``advantage``."""
    trajectory = policy_trajectory(
        "designer", range(1, tokens + 1), (1, 1), (-0.5, -1.25), 0.0, artifact_id="artifact:a"
    )
    return DelayedSample(trajectory.with_training(distribution=dict(distribution)), advantage)
