"""Job data of harness evolution: trace samples as they come, and offline rollouts of the imported train tasks.

``TraceJobData`` puts the trace samples of the buffer into one job's data.
``TrainTaskRollouts`` also accepts the scenario's imported train tasks, so an
uploaded train task enters the buffer like a trace. A job of task samples is
an offline step: the harness backend rolls the tasks out on the head tree as
the step's material. That is offline harness evolution: a step needs no
traffic, so uploaded train tasks start a run.
"""

from __future__ import annotations

from reef.core import RequestType
from reef.sample2job.distill import DistillJobData


class TraceJobData(DistillJobData):
    """Put trace samples into job data unmodified; the backends consume them without more processing.

    In ``manual`` a queued instruction takes the traces that automatic job
    data would take next, up to ``buffer_size`` and possibly none, so the
    method reads the request beside them.
    """

    reads_instructions = True
    required_request_types = frozenset(RequestType)


class TrainTaskRollouts(TraceJobData):
    """``TraceJobData`` that also trains the scenario's imported train tasks: offline harness evolution.

    Values: as ``TraceJobData``, and the stage accepts imported tasks
    (``accepts_imported_tasks``). A job's samples have one kind, so a job is
    either traces or train tasks. The sample store counts how many jobs held
    each task, and ``auto`` trains a task once; ``job_samples`` and
    ``buffer_size`` set how many tasks one step rolls out.
    Purpose: a deployment names the class in ``pipeline.sample2job`` for an
    offline run. The harness backend rolls a job's tasks out on the head tree
    before the method runs, and the method reads the rollouts as traces.
    """

    accepts_imported_tasks = True


__all__ = ["TraceJobData", "TrainTaskRollouts"]
