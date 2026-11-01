"""Record to sample: the stage that decides which records Reef trains on, and turns them into samples.

Reef reuses records in a pipeline of two stages::

    record ─► record2sample ─► sample ─► sample2job ─► job data

This package reads records and makes samples. A sample is a ``Sample``: a ``TrajectorySample`` that trains
directly, or a ``TaskSample`` that the algorithm rolls out. A recipe's ``Sample2Job`` (``reef.sample2job``) shapes
the samples into job data, and ``reef.train.pipeline.TrainingPipeline`` drives both stages.

Entry point: ``Record2Sample`` in ``base.py``.

* Each training pipeline has one stage: the class the recipe's ``pipeline_spec()`` names, or the one the
  deployment's ``pipeline.record2sample`` names, built with ``from_config``. The pipeline pushes every record
  (``push``).
* The stage collects the records that train together into ``RecordSet`` values and forms each one (``form``). No
  record is in two record sets. ``form`` makes the set's samples (``make_samples``) and holds them in a
  ``RecordSetUnit``, the record sets that train together, until the unit is ready. The pipeline takes the ready
  units (``take_ready_units``), stores each unit's samples (``unit_samples``), and the stage ends the unit
  (``consume``). The stage never calls the pipeline.
* Before sample creation, ``form`` rejects any set containing an engine-aborted inference, including report
  references. Replay applies the same check; stored response data stays unchanged.
* Every stage runs the staleness check. It is the only place where Reef compares the artifact that produced
  training data with the artifact that the scenario serves, and drops data that is more than ``max_staleness``
  artifacts behind. ``max_staleness`` is the stage's setting (``Record2SampleSettings``,
  ``pipeline.record2sample.config.max_staleness``).

Stages:

* ``NoSamples`` (``base.py``): the default of a pipeline that trains nothing.
* ``TrajectorySamples`` (``trajectories.py``): each trajectory that ended without a report, after
  ``trajectory_idle_seconds`` without a continuing call, into one ATIF trajectory sample. A method without reports
  extends it.
* ``ReportSamples`` (``reports.py``): one valid report and the calls of the trajectories that it scores into one
  sample, alone or in groups. A report references the last call of each trajectory; the stage expands it to the
  trajectory's calls (``reef.core.trajectory_index``). The default sample is the trajectory with the report's score;
  a recipe overrides ``make_sample``, and its ``ReportGrouping`` decides which reports train and how they form
  groups.
* ``JudgedSamples`` (``judging.py``): records that a model judges on a background worker after later traffic
  completes them, one sample each. A recipe's ``RecordJudging`` holds the correlation and the judge.
* ``InferenceRecordSamples`` (``base.py``): one set per inference record.
* ``TraceSamples`` and ``InferenceSamples`` (``traces.py``): the requests a session sent, as served, into one trace
  sample for harness evolution, with the reported score or unscored.
* ``DistillSamples`` (``distill.py``): one report into one distillation sample, the student's sample plus the
  teacher sequence. ``TeacherRenderer`` renders the teacher's request with the served model's chat template.
* ``GeneratedTaskSamples`` (in the built-in recipe ``reef.recipe.generated_tasks``): new Harbor tasks from a
  description and, optionally, from recorded sessions. The stage gives each published train task to the pipeline
  (``made_samples``), which stores it as a sample; the training component rolls the tasks out.

Values and shared parts:

* ``RecordSet`` and ``RecordSetUnit`` (``sets.py``): the records whose samples are made together, and the record
  sets whose samples the pipeline stores as one group, which trains together.
* ``is_stale`` (``staleness.py``): whether training data is more than ``max_staleness`` artifacts behind the
  artifact that the scenario serves.
* ``SampleAssembly`` (``assembly.py``): a report and its inference records into one policy sample, or one
  multi-turn sample. ``samples.py``: readers of recorded inferences and the policy sample builders.

The subpackage ``generator``, which this module does not import, is the generator service. It runs the slow steps
that need containers (the designer prompt, the authoring rules, the oracle check, and the task rollouts) in a
process beside the HTTP service.

The modules in this directory depend only on ``reef.core``, ``reef.runtime.interfaces``, ``reef.execution.jobs``,
and ``reef.pipeline_context``. Thus the trainer can use them without the generator service.
"""

from reef.record2sample.base import (
    InferenceRecordSamples,
    NoSamples,
    Record2Sample,
    Record2SampleSettings,
    SampleResult,
    SampleStatus,
)
from reef.record2sample.judging import JudgedSamples, JudgedSet, RecordJudging
from reef.record2sample.reports import (
    GroupDecision,
    ReportCandidate,
    ReportContext,
    ReportGrouping,
    ReportSamples,
    ReportSet,
    SkippedReport,
)
from reef.record2sample.sets import RecordSet, RecordSetUnit, SingleRecordSet, SingleSetUnit
from reef.record2sample.trajectories import TrajectorySamples, TrajectorySet, TrajectorySettings

__all__ = [
    "GroupDecision",
    "InferenceRecordSamples",
    "JudgedSamples",
    "JudgedSet",
    "NoSamples",
    "Record2Sample",
    "Record2SampleSettings",
    "RecordJudging",
    "RecordSet",
    "RecordSetUnit",
    "ReportCandidate",
    "ReportContext",
    "ReportGrouping",
    "ReportSamples",
    "ReportSet",
    "SampleResult",
    "SampleStatus",
    "SingleRecordSet",
    "SingleSetUnit",
    "SkippedReport",
    "TrajectorySamples",
    "TrajectorySet",
    "TrajectorySettings",
]
