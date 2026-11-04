"""Job data directories for tests that build a ``TrainingJob`` without a job loop."""

from __future__ import annotations

from reef.core.job_data import JobData
from reef.storage.data import MemoryDataStore
from reef.storage.job_data import JobDataRef
from reef.storage.tasks import LocalTaskFileStore


def job_data_ref(data: JobData) -> JobDataRef:
    """``data`` written as a job data directory by a new in-memory data store."""
    return MemoryDataStore(LocalTaskFileStore()).write_job_directory(data)
