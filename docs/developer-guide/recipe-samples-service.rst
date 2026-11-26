Samples service: the record-to-sample stage
===========================================

``Record2Sample`` selects records and creates ``TrajectorySample`` or ``TaskSample`` values.
Training later renders trajectories or runs tasks.

The contract
------------

Implement ``from_config(config, *, context, environ)``, ``push(record)`` and
``make_samples(record_set)``. ``push`` groups records with ``form``;
``make_samples`` returns ``SampleResult.ready(samples)`` or ``SampleResult.skipped()``.

Selections run serially per trainer, with records in append order.
Keep these hooks nonblocking. Release resources in ``close``.
Invalid settings raise ``PluginConfigError``.
Selection errors leave sets unconsumed; ``ValueError`` is permanent, other failures can retry.

Select and configure the stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Select the stage in ``Recipe.pipeline_spec()`` or ``pipeline.record2sample.module``.
This example extends reported feedback with a score threshold:

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

``PluginSettings.parse`` rejects unknown keys.
See the `base contract <../../reef/samples/base.py>`__ for optional timing and monitoring hooks.

Reported feedback
-----------------

``ReportSamples.make_sample(context)`` returns a sample or ``SkippedReport``.
Use ``context.require_score()`` for finite rewards and ``context.ends`` for final trajectory calls.
``exclusive_sources=True`` gives each report ownership of its inference records.

A record belongs to one ``RecordSet``. ``RecordSetUnit`` groups sets that
must train together. Storing their samples consumes records atomically;
recovery reuses samples without rerunning the judge.

``ReportGrouping`` assigns slots and returns ``INCOMPLETE``, ``READY`` or ``DISCARD``.
Ready groups are fixed; duplicate slots keep the first report.
See `grouping <../../reef/samples/reports.py>`__ and
`trajectory assembly <../../reef/samples/assembly.py>`__ for detailed contracts.

Computed feedback
-----------------

Use ``JudgedSamples`` with a ``RecordJudging`` implementation and ``BackgroundJobWorker``.
``observe`` dispatches without blocking; ``async judge`` can call models.
``make_sample`` returns a trajectory or ``None``.
See `OpenClaw-RL <../../recipes/openclawrl>`__ for an example.
Stage-owned workers must finish cleanup before publishing results and close with the stage.

Staleness
---------

``is_stale(samples, served)`` judges automatic-training eligibility.
The default ``lags_behind(samples, served, 0)`` requires on-policy samples.
Larger bounds permit lag, counting only artifacts this scenario served.
Checks rerun after serving or configuration changes.

Staleness is permanent; manual requests can bypass it.
Tasks, uploads and scenarios without served weights never become stale.
Set ``goes_stale=False`` to disable checks, as SFT and harness stages do.

Generated samples
-----------------

``made_samples()`` returns stable ``(name, sample)`` pairs without record sets.
Reusing a name must identify the same sample.
``job_ended(outcome)`` receives durable job outcomes and can replay after restart.
See `Task generators <task-generators.rst>`__.
