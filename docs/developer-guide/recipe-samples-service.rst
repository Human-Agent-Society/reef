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

Selections run serially per trainer and push records in append order.
They also deliver job outcomes, check staleness and advance background work.
Control calls ``uploaded_sample`` separately; keep it pure and dependent
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

Release resources in ``close``. The `base contract <../../reef/samples/base.py>`__
defines optional hooks for timing, monitoring and background work.

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

Use ``ReportSamples`` for reported feedback, ``JudgedSamples`` for model
judgments, or ``TrajectorySamples`` for trajectories completed after an idle timeout.
``DistillSamples`` adds teacher context; ``TraceSamples`` supplies harness traces.
The `package exports <../../reef/samples/__init__.py>`__ list other stages.

Record sets and units
---------------------

A ``RecordSet`` contains records whose samples are made together.
A report normally brings its referenced trajectories into that set.
A stage cannot put one record in two sets; ``form`` raises ``ValueError``.

``form`` produces samples or skips records. A ``RecordSetUnit`` joins sets
that must train together. Storage consumes their records atomically;
recovery reuses stored samples without rerunning the judge.

Reported feedback
-----------------

Extend ``ReportSamples(context, grouping=None, *, exclusive_sources=False)``.
Override ``make_sample(context)`` to return a sample or ``SkippedReport(reason)``.
The default combines the report's trajectories with its score and feedback.

``context.inferences`` contains calls in chronological order;
``context.ends`` contains each trajectory's final call.
Use ``context.require_score()`` when training requires a finite reward.
Set ``exclusive_sources=True`` when each report owns its inference records.

For groups, implement nonblocking ``ReportGrouping`` methods:
``is_training_report`` filters reports, ``placement`` assigns group and slot,
and ``decide`` returns ``INCOMPLETE``, ``READY`` or ``DISCARD``.
Ready groups are fixed; duplicate slots keep the first report.
Conflicting record ownership skips the report.
See the contracts for `grouping <../../reef/samples/reports.py>`__ and
`multi-turn assembly <../../reef/samples/assembly.py>`__.

Computed feedback
-----------------

Implement ``RecordJudging`` and extend ``JudgedSamples``:

``observe(record, now, stage)`` tracks and dispatches work without blocking.
``async judge(job)`` can call models; ``make_sample(record, judgment)``
returns a ``TrajectorySample`` or ``None``.
Construct ``JudgedSamples(judging, worker)`` with a ``BackgroundJobWorker``;
without one, no judgments run. See `OpenClaw-RL <../../recipes/openclawrl>`__.

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

The default ``lags_behind(samples, served, 0)`` requires on-policy samples.
A larger bound permits lag, counting only artifacts this scenario served.
Reef rechecks after serving or configuration changes.

Staleness is permanent; manual requests can bypass it.
Tasks, uploads and scenarios without served weights never become stale.
SFT and harness stages disable checks with ``goes_stale=False``.

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
use them for generated training tasks.
