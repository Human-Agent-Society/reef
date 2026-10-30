"""Trace job data for harness evolution: trace samples put into job data unmodified."""

from __future__ import annotations

from reef.core import RequestType
from reef.sample2job.distill import DistillJobData


class TraceJobData(DistillJobData):
    """Put trace samples into job data unmodified; the backends consume them without more processing.

    In ``manual`` a queued instruction takes the traces that automatic job
    data would take next, up to ``buffer_size`` and possibly none, so the
    proposer reads the request beside them.
    """

    reads_instructions = True
    required_request_types = frozenset(RequestType)


__all__ = ["TraceJobData"]
