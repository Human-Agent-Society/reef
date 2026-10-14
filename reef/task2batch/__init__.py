"""Task to batch: the stage that shapes the tasks of the selected record sets into one batch.

Reef reuses records in a pipeline of three stages::

    record ─► record2record ─► record set ─► record2task ─► task ─► task2batch ─► batch

The training pipeline (``reef.train.pipeline.TrainingPipeline``) reserves the
next batch's record sets from ``reef.record2record`` and gives their tasks,
which ``reef.record2task`` made, to this stage.

Entry point: ``Task2Batch`` in ``base.py``. A recipe writes ``make_batch`` and
names the class in its ``pipeline_spec()``.

* ``NoBatches`` (``base.py``): the no-update default; it never makes a batch.
* ``DistillBatches`` (``distill.py``): tasks as they come, named by
  ``batch_label``; the distilling recipes and harness evolution use it.

Everything numeric about a method, such as advantages and the loss family,
lives in its backend training objective, not here. The package depends only
on ``reef.core``, ``reef.observability``, ``reef.pipeline_context``, and the
two earlier stage packages, whose classes ``Task2Batch.from_config`` receives.
"""

from reef.task2batch.base import BatchSettings, NoBatches, Task2Batch
from reef.task2batch.distill import DistillBatches

__all__ = [
    "BatchSettings",
    "DistillBatches",
    "NoBatches",
    "Task2Batch",
]
