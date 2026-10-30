"""Sample to job: the stage that sizes the buffer and shapes the samples of the buffer into one job's data.

Reef reuses records in a pipeline of three stages::

    record ─► record2record ─► record set ─► record2sample ─► sample ─► sample2job ─► job data

The training pipeline (``reef.train.pipeline.TrainingPipeline``) stores the
samples of each ready record set unit in the scenario's sample store. The
buffer is the first ``buffer_size`` untrained samples in store order, and the
rest of each group that those touch. When it holds at least ``job_samples``
samples, the pipeline gives its samples to this stage. The stage makes one ``JobData`` value, and Reef submits
one training job for it. The training framework cuts the job's data into its
own optimizer batches.

Entry point: ``Sample2Job`` in ``base.py``. A recipe writes ``make_job_data``
and names the class in its ``pipeline_spec()``.

* ``NoJobData`` (``base.py``): the no-update default; it never makes job data.
* ``DistillJobData`` (``distill.py``): samples as they come; the distilling
  recipes and the SFT recipe use it.
* ``TraceJobData`` (``traces.py``): harness evolution's trace samples as they
  come, in every training mode, and ``TrainTaskRollouts``: offline harness
  evolution, which also trains the scenario's imported train tasks.

Everything numeric about a method, such as advantages and the loss family,
lives in its backend training objective, not here. The package depends only
on ``reef.core``, ``reef.observability``, ``reef.pipeline_context``,
``reef.storage.tasks``, and the two earlier stage packages, whose classes
``Sample2Job.from_config`` receives.
"""

from reef.sample2job.base import NoJobData, Sample2Job, Sample2JobSettings
from reef.sample2job.distill import DistillJobData

__all__ = [
    "DistillJobData",
    "NoJobData",
    "Sample2Job",
    "Sample2JobSettings",
]
