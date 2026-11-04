"""Job data directories for tests that build a ``TrainingJob`` without a job loop."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from reef.core.job_data import JobData
from reef.storage.job_data_store import JobDataRef, LocalJobDataStore


def job_data_ref(data: JobData) -> JobDataRef:
    """``data`` written as a job data directory in a new temporary directory."""
    store = LocalJobDataStore(Path(tempfile.mkdtemp(prefix="reef-test-job-data-")))
    return store.ref(f"job-{uuid.uuid4().hex}", data)
