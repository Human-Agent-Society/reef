"""Task to batch: the stage that buffers samples and shapes the samples of the selected record sets into one batch.

Reef reuses records in a pipeline of three stages::

    record ─► record2record ─► record set ─► record2task ─► task ─► task2batch ─► batch

The tasks of a record set enter this stage as samples. The stage's buffer
holds at most ``buffer_size`` samples. When it holds the candidate backend's
batch size, the training pipeline (``reef.train.pipeline.TrainingPipeline``)
reserves whole record sets from ``reef.record2record`` and gives their
samples to this stage.

Entry point: ``Task2Batch`` in ``base.py``. A recipe writes ``make_batch`` and
names the class in its ``pipeline_spec()``.

* ``NoBatches`` (``base.py``): the no-update default; it never makes a batch.
* ``DistillBatches`` (``distill.py``): tasks as they come; the distilling
  recipes, the SFT recipe, and harness evolution use it.

Everything numeric about a method, such as advantages and the loss family,
lives in its backend training objective, not here. The package depends only
on ``reef.core``, ``reef.observability``, ``reef.pipeline_context``, and the
two earlier stage packages, whose classes ``Task2Batch.from_config`` receives.
"""

from reef.task2batch.base import BufferSettings, NoBatches, Task2Batch
from reef.task2batch.distill import DistillBatches

__all__ = [
    "BufferSettings",
    "DistillBatches",
    "NoBatches",
    "Task2Batch",
]
