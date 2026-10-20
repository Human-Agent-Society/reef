"""Record to record: the shared stage that decides which records Reef uses, and in what order.

Reef uses records in two ways. It trains on records directly, and it turns
records into runnable tasks (``reef.recipe.generated_tasks``). Both need the same
decisions about each record. This package owns those decisions.

Entry point: ``Record2Record`` in ``base.py``.

* Each training pipeline (``reef.train.pipeline.TrainingPipeline``) has one
  stage: the class the recipe's ``pipeline_spec()`` names, or the one the
  deployment's ``pipeline.record2record`` names, built with ``from_config``.
  The pipeline pushes every record except training instructions (``push``).
* The stage forms ``RecordSet`` values, which the pipeline pulls (``pull``):
  the records that the record-to-task stage makes tasks from, together. No
  record is in two record sets. The pipeline gives each task result back
  (``settle``). The stage holds the sets that train in ``RecordSetUnit`` values
  in its pool, drops the units that can no longer train, and gives the
  pipeline the next batch's units in order. The stage never calls the other stages.
* Every stage runs the staleness check. It is the only place where Reef
  compares the version that produced training data with the version that the
  scenario serves, and drops data that is more than ``max_staleness`` versions
  behind.

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

Components of the pool:

* ``RecordSet`` and ``RecordSetUnit`` (``sets.py``): the input of the
  record-to-task stage, and the record sets that train in one batch, the unit
  that a batch takes and that ``batch_size`` counts.
* ``RecordSetUnitPool`` (``pool.py``): the held units, and the reserved batch.
* ``SelectionPolicy`` (``selection.py``): the order in which units go into a
  batch. ``ArrivalOrder`` takes the oldest first. ``GroupKeyOrder`` takes
  ungrouped units, then groups by key.
* ``stale_units`` (``staleness.py``): finds held units whose training data is more
  than ``max_staleness`` versions behind the version that the scenario serves.
  The lag counts the versions that the scenario served after the producing one.

The package depends only on ``reef.core``, ``reef.runtime.interfaces``,
``reef.observability``, ``reef.background_jobs``, and ``reef.pipeline_context``. Thus record-to-task and
the trainer can both use it.
"""

from reef.record2record.base import InferenceRecord2Record, NoRecordSets, Record2Record, TaskStatus
from reef.record2record.judging import JudgedSet, JudgingRecord2Record, RecordJudging
from reef.record2record.pool import RecordSetUnitPool
from reef.record2record.reports import (
    ExclusiveReportRecord2Record,
    GroupDecision,
    ReportCandidate,
    ReportContext,
    ReportGrouping,
    ReportRecord2Record,
    ReportSet,
)
from reef.record2record.selection import ArrivalOrder, GroupKeyOrder, SelectionPolicy
from reef.record2record.sets import RecordSet, RecordSetUnit, SingleRecordSet, SingleSetUnit

__all__ = [
    "ArrivalOrder",
    "ExclusiveReportRecord2Record",
    "GroupDecision",
    "GroupKeyOrder",
    "InferenceRecord2Record",
    "JudgedSet",
    "JudgingRecord2Record",
    "NoRecordSets",
    "Record2Record",
    "RecordJudging",
    "RecordSet",
    "RecordSetUnit",
    "RecordSetUnitPool",
    "ReportCandidate",
    "ReportContext",
    "ReportGrouping",
    "ReportRecord2Record",
    "ReportSet",
    "SelectionPolicy",
    "SingleRecordSet",
    "SingleSetUnit",
    "TaskStatus",
]
