"""Sample to job: the stage that buffers samples and shapes the samples of the selected record sets into one job's data.

Reef reuses records in a pipeline of three stages::

    record ─► record2record ─► record set ─► record2sample ─► sample ─► sample2job ─► job data

The samples of the record sets that wait for a job form this stage's buffer, which
holds at most ``buffer_size`` samples. When it holds at least ``job_samples``
samples, the training pipeline (``reef.train.pipeline.TrainingPipeline``)
reserves whole record sets from ``reef.record2record`` and gives their
samples to this stage. The stage makes one ``JobData`` value, and Reef submits
one training job for it. The training framework cuts the job's data into its
own optimizer batches.

Entry point: ``Sample2Job`` in ``base.py``. A recipe writes ``make_job_data``
and names the class in its ``pipeline_spec()``.

* ``NoJobData`` (``base.py``): the no-update default; it never makes job data.
* ``DistillJobData`` (``distill.py``): samples as they come; the distilling
  recipes, the SFT recipe, and harness evolution use it.

Everything numeric about a method, such as advantages and the loss family,
lives in its backend training objective, not here. The package depends only
on ``reef.core``, ``reef.observability``, ``reef.pipeline_context``, and the
two earlier stage packages, whose classes ``Sample2Job.from_config`` receives.
"""

from reef.sample2job.base import DEFAULT_BUFFER_JOBS, NoJobData, Sample2Job, Sample2JobSettings
from reef.sample2job.distill import DistillJobData

__all__ = [
    "DEFAULT_BUFFER_JOBS",
    "DistillJobData",
    "NoJobData",
    "Sample2Job",
    "Sample2JobSettings",
]
