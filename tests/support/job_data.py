"""Job data directories for tests that build a ``TrainingJob`` without a job loop."""

from __future__ import annotations

from dataclasses import replace

from reef.core.job_data import JobData, TaskFiles, TaskSample
from reef.storage.data import MemoryDataStore
from reef.storage.job_data import JobDataRef
from reef.storage.tasks import LocalTaskFileStore

#: The partition of the task file store that keeps the tasks of a test's job data.
PARTITION = "test-tasks"
#: The stores of the references that this module gave: a store deletes its files when it is collected.
STORES: list[MemoryDataStore] = []


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
