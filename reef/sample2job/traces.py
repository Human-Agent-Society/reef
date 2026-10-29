"""Job data of harness evolution: trace samples as they come, and offline runs of the imported train tasks.

``TraceJobData`` puts the trace samples of the buffer into one job's data.
``TrainTaskRollouts`` also trains the scenario's imported train tasks: they
are buffer samples, and the harness backend rolls a job's tasks out on the
head tree as the step's material. That is offline harness evolution: a step
needs no traffic, so uploaded train tasks start a run.
"""

from __future__ import annotations

from reef.core import RequestType
from reef.sample2job.distill import DistillJobData


class TraceJobData(DistillJobData):
    """Put trace samples into job data unmodified; the backends consume them without more processing.

    In ``manual`` a queued instruction takes the samples of the manual
    buffer, possibly none, so the method reads the request beside them.
    """

    reads_instructions = True
    required_request_types = frozenset(RequestType)


class TrainTaskRollouts(TraceJobData):
    """``TraceJobData`` that also trains the scenario's imported train tasks.

    Each imported ``train`` task is a buffer sample, in upload order, and
    trains once. One job's data holds samples of one kind, so a step either
    reads traces or rolls out up to ``buffer_size`` train tasks on the head
    tree. A deployment names the class in ``pipeline.sample2job`` for offline
    harness evolution.
    """

    accepts_imported_tasks = True


__all__ = ["TraceJobData", "TrainTaskRollouts"]
