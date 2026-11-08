"""Job data directories for tests that build a ``TrainingJob`` without a job loop, and the delayed samples of job data."""

from __future__ import annotations

from collections.abc import Mapping

from tests.support._trajectories import policy_trajectory

from reef.core.job_data import DelayedSample, JobData
from reef.storage.data import MemoryDataStore
from reef.storage.job_data import JobDataRef
from reef.storage.tasks import LocalTaskFileStore

#: The sampling distribution of the samples that ``delayed_sample`` makes.
DELAYED_DISTRIBUTION = {"temperature": 0.6, "top_p": 1.0, "top_k": -1}


def job_data_ref(data: JobData) -> JobDataRef:
    """``data`` written as a job data directory by a new in-memory data store."""
    return MemoryDataStore(LocalTaskFileStore()).write_job_directory(data)


def delayed_sample(
    advantage: float = 1.0, *, tokens: int = 5, distribution: Mapping[str, object] = DELAYED_DISTRIBUTION
) -> DelayedSample:
    """A recorded reply of ``tokens`` tokens, the last two its response, with its exact sample and ``advantage``."""
    trajectory = policy_trajectory(
        "designer", range(1, tokens + 1), (1, 1), (-0.5, -1.25), 0.0, artifact_id="artifact:a"
    )
    return DelayedSample(trajectory.with_training(distribution=dict(distribution)), advantage)
