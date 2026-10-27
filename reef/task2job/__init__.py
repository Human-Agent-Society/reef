"""Task to job: the stage that buffers samples and shapes the samples of the selected record sets into one job's data.

Reef reuses records in a pipeline of three stages::

    record ─► record2record ─► record set ─► record2task ─► task ─► task2job ─► job data

The tasks of a record set enter this stage as samples. The stage's buffer
holds at most ``buffer_size`` samples. When it holds at least ``job_samples``
samples, the training pipeline (``reef.train.pipeline.TrainingPipeline``)
reserves whole record sets from ``reef.record2record`` and gives their
samples to this stage. The stage makes one ``JobData`` value, and Reef submits
one training job for it. The training framework cuts the job's data into its
own optimizer batches.

Entry point: ``Task2Job`` in ``base.py``. A recipe writes ``make_job_data``
and names the class in its ``pipeline_spec()``.

* ``NoJobData`` (``base.py``): the no-update default; it never makes job data.
* ``DistillJobData`` (``distill.py``): tasks as they come; the distilling
  recipes, the SFT recipe, and harness evolution use it.

Everything numeric about a method, such as advantages and the loss family,
lives in its backend training objective, not here. The package depends only
on ``reef.core``, ``reef.observability``, ``reef.pipeline_context``, and the
two earlier stage packages, whose classes ``Task2Job.from_config`` receives.
"""

from reef.task2job.base import DEFAULT_BUFFER_JOBS, NoJobData, Task2Job, Task2JobSettings
from reef.task2job.distill import DistillJobData

__all__ = [
    "DEFAULT_BUFFER_JOBS",
    "DistillJobData",
    "NoJobData",
    "Task2Job",
    "Task2JobSettings",
]
