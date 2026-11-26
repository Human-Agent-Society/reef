Samples service: the record-to-sample stage
===========================================

Implement ``Record2Sample`` to select records, group them and create training
samples. The samples service runs this stage; control later builds jobs
from its stored output.

A ``TrajectorySample`` carries an ATIF trajectory, reward and metadata.
A ``TaskSample`` names a Harbor task that training rolls out.
The training component renders tokens and log probabilities.

When the service calls the stage
--------------------------------

Each trainer processes one record selection at a time. New records, commits,
data writes and completed background work can request a selection.
Each selection delivers job outcomes, checks staleness, pushes new records
in append order, advances timed work and stores ready samples.

Stateful stage work runs only in the samples service.
Control also calls ``uploaded_sample``; keep that method pure and dependent
only on its argument and configuration.

The contract
------------

Subclass ``reef.samples.Record2Sample`` or a `built-in stage <#built-in-stages>`__.
Implement these methods:

- ``from_config(config, *, context, environ)`` builds the scenario's stage.
  ``context`` is a ``PipelineContext``.
- ``push(record)`` collects related records and calls ``form(record_set)``.
- ``make_samples(record_set)`` returns ``SampleResult.ready(samples)`` or
  ``SampleResult.skipped()``.

Keep ``push``, ``advance`` and ``make_samples`` free of blocking network or
model calls. Raise on invalid training data; do not silently discard broken
contracts.

Optional hooks provide timed work (``advance``, ``next_due``), staleness,
background work, generated samples and job outcomes.
Use ``status`` and ``operational_metrics`` for monitoring.
Release resources in ``close``.

Select and configure the stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Name the stage in ``Recipe.pipeline_spec()``. Configure it in
``pipeline.record2sample``; ``module`` can select another class.
The default ``NoSamples`` produces no training samples.

Parse settings with a frozen ``PluginSettings`` dataclass:

.. code:: python

   from dataclasses import dataclass

   from reef.core.config import config_option
   from reef.core.plugins import PluginSettings
   from reef.samples import ReportSamples, SkippedReport


   @dataclass(frozen=True)
   class MySettings(PluginSettings):
       min_score: float = config_option(0.0)


   class MySamples(ReportSamples):
       def __init__(self, context, settings: MySettings) -> None:
           super().__init__(context)
           self.min_score = settings.min_score

       @classmethod
       def from_config(cls, config, *, context, environ):
           return cls(context, MySettings.parse(config, plugin_class=cls))

       def make_sample(self, context):
           if context.require_score() < self.min_score:
               return SkippedReport("Score below the training threshold.")
           return super().make_sample(context)

This stage trains reports at or above the configured threshold.
``Settings.parse`` rejects unknown keys; ``NoSettings`` rejects all keys.

Failure behavior
~~~~~~~~~~~~~~~~

Invalid stage classes raise ``RecipeConfigError`` at deployment load.
Invalid settings raise ``PluginConfigError`` when the scenario builds its pipeline.
A stage exception fails the selection: ``ValueError`` is permanent;
other errors can be retried. Failed sets remain unconsumed and produce no samples.
An aborted inference or incomplete stream rejects its entire record set.

Built-in stages
---------------

- ``ReportSamples``: samples from reports, optionally grouped.
- ``TrajectorySamples``: completed trajectories after
  ``trajectory_idle_seconds`` without continuation; default 300 seconds.
- ``JudgedSamples``: background model judgments.
- ``InferenceRecordSamples``: one set per inference record; ends other records.
- ``DistillSamples``: student trajectories with ``extra.reef.teacher``.
  Override ``teacher_request``.
- ``TraceSamples`` and ``InferenceSamples``: inputs for harness evolution.
- ``NoSamples``: no output.

Record sets and units
---------------------

A ``RecordSet`` contains records whose samples are made together.
A report normally brings its referenced trajectories into that set.
A stage cannot put one record in two sets; ``form`` raises ``ValueError``.

``form`` either creates ready samples or ends the records as skipped.
A ``RecordSetUnit`` joins sets that must train in one job, such as grouped
reports. The service stores the unit's samples and marks its records consumed.
Its samples share a group, which job selection keeps whole.

Recovery skips consumed records. Stored samples therefore survive restarts
without rerunning a nondeterministic judge.

Reported feedback
-----------------

Extend ``ReportSamples(context, grouping=None, *, exclusive_sources=False)``.
Override ``make_sample(context)`` to return a sample or ``SkippedReport(reason)``.
The default combines the report's trajectories with its score and feedback.

``context.inferences`` contains calls in chronological order;
``context.ends`` contains each trajectory's final call.
Use ``context.require_score()`` when training requires a finite reward.
Set ``exclusive_sources=True`` when each report owns its inference records.

For grouped reports, implement ``ReportGrouping`` without blocking calls:

- ``is_training_report(report)`` filters reports for this method.
- ``placement(context)`` returns ``(group_key, slot)``.
  ``(None, None)`` means an independent sample. A missing slot uses the
  report ID; duplicate slots keep the first accepted report.
- ``decide(key, members)`` returns ``INCOMPLETE``, ``READY`` or ``DISCARD``.
  Members arrive in report order.

Ready groups are fixed. Later reports with the same key form another group.
Conflicting ownership of inference records ends a report without training.

``SampleAssemblySettings`` controls policy samples:
``accept_multi_turn_policy_samples=false`` trains only the last reply;
``true`` trains every reply. ``realign_threshold`` defaults to 1024 and
``scaffold_tolerance`` to 0 when prompts rewrite earlier turns.

Computed feedback
-----------------

Implement ``RecordJudging`` and extend ``JudgedSamples``:

- ``observe(record, now, stage)`` correlates records with ``track``,
  ``tracked_record``, ``dispatch`` and ``retire``. It must not block.
- ``async judge(job)`` runs on a ``BackgroundJobWorker`` and can call models.
- ``expire(now)`` returns tracked receipts whose wait expired; default none.
- ``make_sample(record, judgment)`` returns a ``TrajectorySample`` or ``None``.

Construct ``JudgedSamples(judging, worker)``. Without a worker, records can
track and expire but receive no judgments.
See ``recipes/openclawrl/`` for an implementation.

Slow work
---------

Submit slow work from ``push`` to a worker owned by the stage.
Return ``True`` from ``background_jobs_running`` while work remains.
Collect results in ``advance(now)`` and form record sets there.
Stop the worker in ``close``; failed cleanup can raise ``TimeoutError``.

For a timer alone, return the monotonic deadline from ``next_due()``.
See `Background job ownership <sandboxes.rst#background-job-ownership>`__
for cancellation and cleanup rules.

Staleness
---------

``is_stale(samples, served)`` decides whether an untrained sample group can
still train automatically. ``served`` describes this scenario's served
artifact history and artifacts served beside the active selection.

The service rechecks after serving changes or stage configuration changes.
Control waits for the current judgment and excludes stale groups.

- The default ``lags_behind(samples, served, 0)`` is on-policy.
  Samples from another artifact, or with no source artifact, are stale.
- Pass a larger bound to ``lags_behind`` to permit artifact lag.
  ``SAOSamples`` uses ``max_artifact_lag``, default 0.
- SFT and harness trace stages set ``goes_stale=False``.
- Tasks and uploaded trajectories are never stale.
  Neither are samples from scenarios without served weight artifacts.

Staleness is permanent once marked. Serving an earlier artifact again does
not restore its samples. Manual requests can explicitly train stale samples.
Lag counts only artifacts that this scenario actually served.

Samples from no record set
--------------------------

``made_samples()`` returns new ``(name, sample)`` pairs without record sets.
Names must be stable: one name always identifies the same sample.
Existing samples are unchanged, so replay after restart is safe.
These samples have no group or consumed records.

``job_ended(outcome)`` receives a durable job end: job ID, backend,
``finished`` or ``failed`` event, metrics and error.
Handle repeated outcomes; startup can deliver the last outcome again.

Both hooks default to no work. `Task generators <task-generators.rst>`__
use them to train generated tasks. Access task storage through
``context.task_storage.open(name)`` when a scenario session exists;
otherwise ``task_storage`` is ``None``.
