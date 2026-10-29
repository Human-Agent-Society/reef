"""Record to record: the shared stage that decides which records Reef uses, and which train together.

Reef uses records in two ways. It trains on records directly, and it turns
records into runnable tasks (``reef.recipe.generated_tasks``). Both need the same
decisions about each record. This package owns those decisions.

Entry point: ``Record2Record`` in ``base.py``.

* Each training pipeline (``reef.train.pipeline.TrainingPipeline``) has one
  stage: the class the recipe's ``pipeline_spec()`` names, or the one the
  deployment's ``pipeline.record2record`` names, built with ``from_config``.
  The pipeline pushes every record except training instructions (``push``).
* The stage forms ``RecordSet`` values, which the pipeline pulls (``pull``):
  the records that the record-to-sample stage makes samples from, together. No
  record is in two record sets. The pipeline gives each sample result back
  (``settle``). The stage makes each ``RecordSetUnit``, the record sets that
  train together, ready when its samples are. The pipeline takes the ready units
  (``take_ready_units``), stores each unit's samples, and the stage ends the
  unit (``consume``). The stage never calls the other stages.
* Before task creation, ``form`` rejects any set containing an engine-aborted
  inference, including report references. Replay applies the same check; stored
  response data stays unchanged.
* Every stage runs the staleness check. It is the only place where Reef
  compares the version that produced training data with the version that the
  scenario serves, and drops data that is more than ``max_staleness`` versions
  behind. ``max_staleness`` is the stage's setting (``Record2RecordSettings``,
  ``pipeline.record2record.config.max_staleness``).

Stages:

* ``NoRecordSets`` (``base.py``): the default of a pipeline that trains
  nothing. It forms no record set.
* ``ReportRecord2Record`` (``reports.py``): reports and the inference records that
  they reference, alone or in groups. A recipe's ``ReportGrouping`` decides
  which reports train and how they form groups. ``ExclusiveReportRecord2Record``
  trains every valid report by itself, and each report owns its inference records.
* ``JudgingRecord2Record`` (``judging.py``): records that a model judges on a
  background worker after later traffic completes them. A recipe's
  ``RecordJudging`` holds the correlation and the judge.
* ``InferenceRecord2Record`` (``base.py``): one set per inference record.

Values and checks:

* ``RecordSet`` and ``RecordSetUnit`` (``sets.py``): the input of the
  record-to-sample stage, and the record sets whose samples the pipeline stores
  as one group, which trains together.
* ``is_stale`` (``staleness.py``): whether training data is more than
  ``max_staleness`` versions behind the version that the scenario serves.
  The lag counts the versions that the scenario served after the producing one.

The package depends only on ``reef.core``, ``reef.runtime.interfaces``,
``reef.execution.jobs``, and ``reef.pipeline_context``. Thus record-to-sample and
the trainer can both use it.
"""

from reef.record2record.base import (
    InferenceRecord2Record,
    NoRecordSets,
    Record2Record,
    Record2RecordSettings,
    SampleStatus,
)
from reef.record2record.judging import JudgedSet, JudgingRecord2Record, RecordJudging
from reef.record2record.reports import (
    ExclusiveReportRecord2Record,
    GroupDecision,
    ReportCandidate,
    ReportContext,
    ReportGrouping,
    ReportRecord2Record,
    ReportSet,
)
from reef.record2record.sets import RecordSet, RecordSetUnit, SingleRecordSet, SingleSetUnit

__all__ = [
    "ExclusiveReportRecord2Record",
    "GroupDecision",
    "InferenceRecord2Record",
    "JudgedSet",
    "JudgingRecord2Record",
    "NoRecordSets",
    "Record2Record",
    "Record2RecordSettings",
    "RecordJudging",
    "RecordSet",
    "RecordSetUnit",
    "ReportCandidate",
    "ReportContext",
    "ReportGrouping",
    "ReportRecord2Record",
    "ReportSet",
    "SampleStatus",
    "SingleRecordSet",
    "SingleSetUnit",
]
