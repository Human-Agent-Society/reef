"""Record to sample: the stage that turns one record set into samples.

Reef reuses records in a pipeline of three stages::

    record ─► record2record ─► record set ─► record2sample ─► sample ─► sample2job ─► job data

``reef.record2record`` decides which records Reef uses and forms record sets.
This package turns each set into samples. A sample is a ``Sample``: a
``TrajectorySample`` that trains directly, or a ``TaskSample`` that the algorithm
rolls out. A recipe's ``Sample2Job`` (``reef.sample2job``) shapes the
samples into job data, and ``reef.train.pipeline.TrainingPipeline`` drives the
three stages.

Entry point: ``Record2Sample`` in ``base.py``. The recipe's ``pipeline_spec()``
names the class, and ``from_config`` builds it for each scenario.

Implementations:

* ``ReportSamples`` (``reports.py``): one valid report and its inference
  records into one sample. A recipe writes ``make_sample``.
* ``JudgedSamples`` (``judged.py``): one judged record into one sample. A recipe
  writes ``make_sample``.
* ``DistillSamples`` (``distill.py``): one report into one distillation sample,
  the student's sample plus the teacher sequence. ``TeacherRenderer`` renders
  the teacher's request with the served model's chat template.
* ``GeneratedTaskSamples`` (in the built-in recipe ``reef.recipe.generated_tasks``):
  new Harbor tasks from a description and, optionally, from recorded sessions.
  Each round generates and checks tasks. Their rollout episodes are not
  records: ``GeneratedTaskJobData`` turns them into samples.

Shared parts of ``make_sample``:

* ``SampleAssembly`` (``assembly.py``): a report and its inference records into
  one policy sample, or one multi-turn sample.
* ``samples.py``: readers of recorded inferences and the policy sample builders.

The subpackage ``generator``, which this module does not import, is the
generator service. It runs the slow steps that need containers (the designer
prompt, the authoring rules, the oracle check, and the task episodes) in a
process beside the HTTP service.

The modules in this directory depend only on ``reef.core``,
``reef.record2record``, and ``reef.pipeline_context``. Thus the trainer can use
them without the generator service.
"""

from reef.record2sample.base import ImmediateRecord2Sample, NoSamples, Record2Sample, SampleResult
from reef.record2sample.judged import JudgedSamples
from reef.record2sample.reports import ReportSamples, SkippedReport

__all__ = [
    "ImmediateRecord2Sample",
    "JudgedSamples",
    "NoSamples",
    "Record2Sample",
    "ReportSamples",
    "SampleResult",
    "SkippedReport",
]
