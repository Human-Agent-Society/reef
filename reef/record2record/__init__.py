"""Record to record: the shared stage that decides which records Reef uses, and in what order.

Reef uses records in two ways. It trains on records directly, and it turns
records into runnable tasks (``reef.recipe.generated_tasks``). Both need the same
decisions about each record. This package owns those decisions.

Entry point: ``Record2Record`` in ``base.py``.

* Each training pipeline (``reef.train.pipeline.TrainingPipeline``) has one
  stage: the class the recipe's ``pipeline_spec()`` names, or the one the
  deployment's ``pipeline.record2record`` names, built with ``from_config``.
  The pipeline gives the stage every record except training instructions.
* The stage forms ``RecordSet`` values: records that are used together. The
  record-to-task stage makes tasks from each set, and the pipeline gives the
  result back (``settle``). The stage holds the sets that train in its pool,
  drops the sets that can no longer train, and gives the pipeline the next
  batch's sets in order. The stage never calls the other stages.

Stages:

* ``PassThroughRecord2Record`` (``base.py``): the default. It forms one set per
  record and decides nothing.
* ``ReportRecord2Record`` (``reports.py``): reports and the inference records that
  they reference, alone or in groups. A recipe's ``ReportGrouping`` decides
  which reports train and how they form groups.
* ``JudgingRecord2Record`` (``judging.py``): records that a model judges on a
  background worker after later traffic completes them. A recipe's
  ``RecordJudging`` holds the correlation and the judge.
* ``InferenceRecord2Record`` (``base.py``): one set per inference record.
* ``RoundRecord2Record`` (``rounds.py``): a ``ReportRecord2Record`` that also
  paces task-generation rounds and holds their source sessions. A recipe's
  ``RoundGrouping`` decides which reports are source sessions for the rounds.

Components of the pool:

* ``RecordSet`` (``sets.py``): the base of every set, with the fields that the
  pool, the selection policies, and the eligibility checks read.
* ``RecordSetPool`` (``pool.py``): the held sets, and the reserved batch.
* ``SelectionPolicy`` (``selection.py``): the order in which sets go into a
  batch. ``ArrivalOrder`` takes the oldest first. ``GroupKeyOrder`` takes
  ungrouped sets, then groups by key.
* ``EligibilityCheck`` (``eligibility.py``): finds held sets that can no
  longer train. ``StalenessCheck`` drops sets that are more than
  ``max_staleness`` versions behind the newest held set.

The package depends only on ``reef.core``, ``reef.runtime.interfaces``,
``reef.observability``, ``reef.background_jobs``, and ``reef.pipeline_context``. Thus record-to-task and
the trainer can both use it.
"""

from reef.record2record.base import InferenceRecord2Record, PassThroughRecord2Record, Record2Record, TaskStatus
from reef.record2record.eligibility import EligibilityCheck, IneligibleSet, StalenessCheck
from reef.record2record.judging import JudgedSet, JudgingRecord2Record, RecordJudging, judged_sets
from reef.record2record.pool import RecordSetPool
from reef.record2record.reports import (
    GroupDecision,
    ReportCandidate,
    ReportContext,
    ReportEntry,
    ReportGrouping,
    ReportRecord2Record,
    ReportSet,
    accepted_by,
    report_score,
    report_sets,
)
from reef.record2record.rounds import RoundGrouping, RoundRecord2Record
from reef.record2record.selection import ArrivalOrder, GroupKeyOrder, SelectionPolicy
from reef.record2record.sets import RecordSet, SingleRecordSet

__all__ = [
    "ArrivalOrder",
    "EligibilityCheck",
    "GroupDecision",
    "GroupKeyOrder",
    "IneligibleSet",
    "InferenceRecord2Record",
    "JudgedSet",
    "JudgingRecord2Record",
    "PassThroughRecord2Record",
    "Record2Record",
    "RecordJudging",
    "RecordSet",
    "RecordSetPool",
    "ReportCandidate",
    "ReportContext",
    "ReportEntry",
    "ReportGrouping",
    "ReportRecord2Record",
    "ReportSet",
    "RoundGrouping",
    "RoundRecord2Record",
    "SelectionPolicy",
    "SingleRecordSet",
    "StalenessCheck",
    "TaskStatus",
    "accepted_by",
    "judged_sets",
    "report_score",
    "report_sets",
]
