"""Trace job data for harness evolution: trace samples put into job data unmodified."""

from __future__ import annotations

from reef.core import RequestType
from reef.sample2job.distill import DistillJobData


class TraceJobData(DistillJobData):
    """Put trace samples into job data unmodified; the backends consume them without more processing.

    In ``hybrid`` a queued instruction takes the traces that automatic job
    data would take next, up to ``job_samples``, so the proposer
    reads the request beside them; in ``manual`` it runs alone.
    """

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})
    required_request_types = frozenset(RequestType)


__all__ = ["TraceJobData"]
