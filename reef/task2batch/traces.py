"""Trace batches for harness evolution: trace tasks batched unmodified."""

from __future__ import annotations

from reef.core import RequestType
from reef.task2batch.distill import DistillBatches


class TraceBatches(DistillBatches):
    """Batch trace tasks unmodified; the backends consume them without more processing.

    In ``hybrid`` a queued instruction batches with the traces that an
    automatic batch would take next, up to ``evolution.batch_size``, so the proposer
    reads the request beside them; in ``manual`` it runs alone.
    """

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})
    required_request_types = frozenset(RequestType)


__all__ = ["TraceBatches"]
