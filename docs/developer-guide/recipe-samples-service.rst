Samples service: the record-to-sample stage
===========================================

The record-to-sample stage decides which records train, which records train
together, and what sample each group becomes. The samples service runs it.
This page defines the stage contract, the built-in stages to extend, and the
staleness rule.

:For: authors of a weight method's record-to-sample stage
:Before you start: the plug-in map in `Write a recipe <write-a-recipe.rst>`__
:You will have: a stage class, its config, and the behavior you can rely on

.. code:: text

   record ─► record2sample ─► sample ─► sample2job ─► job data ─► training job
             (samples service)          (control service)

A sample is a ``TrajectorySample``, which trains directly, or a
``TaskSample``, a Harbor task that the training component rolls out. A sample
holds an ATIF trajectory with its reward, feedback and metadata. It holds no
tokens and no log probabilities: the training component renders it (see
`Rendering <recipe-training-service.rst#rendering>`__).

When the service calls the stage
--------------------------------

The samples service takes one record selection at a time for each trainer.
Control requests a selection when records arrive, after each commit and
after a data write. The stage's own background work requests one when it
ends. In each selection the service:

1. gives the stage the end of each training job that it has not read
   (``job_ended``);
2. marks the stale stored samples (``is_stale``);
3. pushes each new record, in append order (``push``);
4. advances the stage to the current time (``advance(now)``);
5. stores the samples of each unit that became ready, and ends the unit
   (``consume``).

Only the samples service feeds a stage. A stage that holds records or runs
background work keeps that state in the samples service's process. Control
calls one method, ``uploaded_sample``, so that method must depend only on
its argument and the stage's config (see `Uploaded trajectories
<recipe-control-service.rst#uploaded-trajectories>`__).

The contract
------------

Subclass ``reef.record2sample.Record2Sample``, or one of the `built-in
stages`_.

Required members:

- ``from_config(config, *, context, environ)``: class method. Build the
  stage of one scenario from its config. ``context`` is the scenario's
  ``PipelineContext``.
- ``push(record)``: read one record. Collect the records that train
  together into a ``RecordSet`` and call ``form(record_set)``.
- ``make_samples(record_set)``: return ``SampleResult.ready(samples)`` or
  ``SampleResult.skipped()``.

Optional members:

- ``advance(now)``: take in background work that ended and form the sets
  that are due. ``now`` is a monotonic time.
- ``is_stale(samples, served)`` and ``goes_stale``: the `staleness`_ rule.
  The default is on-policy.
- ``background_jobs_running()`` and ``next_due()``: tell the service to come
  back without a new record (`Slow work`_).
- ``made_samples()`` and ``job_ended(outcome)``: `samples from no record
  set`_.
- ``uploaded_sample(trajectory)``: the training sample of an uploaded
  trajectory. The default returns it unchanged.
- ``status()``, ``operational_metrics()`` and ``close()``: state for ``GET
  /reef/status``, numeric metrics, and release of the stage's resources.

``push``, ``advance`` and ``make_samples`` must not block on network or model
calls (see `Slow work`_). Raise on a broken training contract; do not drop
the record silently.

Select and configure the stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recipe names its stage class in ``pipeline_spec()``. The deployment's
``pipeline.record2sample`` section gives the stage its ``config`` and can
name another class with ``module`` (see `the pipeline section
<../reference/configuration.rst#the-pipeline-section>`__). The default
class, ``NoSamples``, makes no sample, so the pipeline makes no job.

A stage declares its keys as a frozen dataclass that inherits
``reef.core.plugins.PluginSettings``:

.. code:: python

   from dataclasses import dataclass

   from reef.core.config import config_option
   from reef.core.plugins import PluginSettings
   from reef.record2sample import ReportSamples


   @dataclass(frozen=True)
   class MySettings(PluginSettings):
       min_score: float = config_option(0.0, help="Reports below this score do not train.")


   class MySamples(ReportSamples):
       def __init__(self, context, settings: MySettings) -> None:
           super().__init__(context)
           self.min_score = settings.min_score

       @classmethod
       def from_config(cls, config, *, context, environ):
           return cls(context, MySettings.parse(config, plugin_class=cls))

``Settings.parse`` refuses an unknown key. A stage without config parses
with ``NoSettings``, which refuses every key.

Failure behavior
~~~~~~~~~~~~~~~~

- Reef imports and checks the stage classes when it loads the deployment. A
  wrong class raises ``RecipeConfigError`` then.
- A stage parses its config when the first scenario builds its pipeline. A
  bad config raises ``PluginConfigError`` then.
- An exception from the stage fails the record selection. A ``ValueError``
  fails it at once; the service retries another error. The records of the
  failed set are not marked consumed, and Reef stores no sample for them.
- An inference response that the engine aborted, or a stream that did not
  complete, rejects its whole record set, including a report that references
  it.

Built-in stages
---------------

Extend one of these classes of ``reef.record2sample`` when it fits:

- ``NoSamples``: nothing. The default of a recipe that trains no weights.
- ``ReportSamples``: one sample per valid report, alone or in groups
  (`Reported feedback`_).
- ``TrajectorySamples``: one sample per trajectory that ended without a
  report, after ``trajectory_idle_seconds`` (default 300) without a
  continuing call.
- ``JudgedSamples``: one sample per record that a model judged on a
  background worker (`Computed feedback`_).
- ``InferenceRecordSamples``: one set per inference record. The stage ends
  every other record.
- ``DistillSamples``: one distillation sample per report: the student's call
  plus the request that its teacher reads (``extra.reef.teacher``). A
  subclass overrides ``teacher_request``.
- ``TraceSamples`` and ``InferenceSamples``: harness evolution's trace
  samples.

Record sets and units
---------------------

A record set (``RecordSet``) is one or more records whose samples the stage
makes together. Reef links the inference records of a scenario into
trajectories. A report references the last call of each trajectory that it
scores; the report and the calls of those trajectories form one record set.
No record is in two record sets of one stage: ``form`` refuses such a set
with ``ValueError``.

``form`` makes the samples of a set at once and settles all its records
together:

- ``READY``: one or more samples. The records wait in a record set unit.
- ``SKIPPED``: no sample. The stage ends the records, and they never train.

A record set unit (``RecordSetUnit``) holds the record sets that must train
in the same job: one report, or all reports of a ready group. When a unit is
ready, the service stores its samples as data items of the scenario and
records its records as consumed. The samples of one unit share a group, so
they train in the same job.

A restart never makes stored samples again: recovery skips the consumed
records. A nondeterministic stage, such as a judge, therefore does not run
twice on the same records.

Reported feedback
-----------------

Reported feedback arrives in reports. Extend ``ReportSamples``:

.. code:: python

   ReportSamples(context, grouping=None, *, exclusive_sources=False)

- ``make_sample(context) -> Sample | SkippedReport``: override it when the
  default does not fit. The default makes the report's trajectories one ATIF
  trajectory sample with the report's score and feedback.
  ``context.inferences`` holds the calls of the trajectories, oldest first.
  ``context.ends`` holds the last call of each trajectory. Call
  ``context.require_score()`` when the method needs a finite reward. Return
  ``SkippedReport(reason)`` for a valid report that cannot train.
- ``grouping``: a ``ReportGrouping``. Without one, every valid report trains
  by itself.
- ``exclusive_sources=True``: a report that ends owns its inference records.
  Use it when each report has its own inferences.

A ``ReportGrouping`` reads only records and must not block:

- ``is_training_report(report)``: return ``False`` for a valid report that
  is not this method's training data, such as another role's report. The
  stage releases it. The default takes every report.
- ``placement(context)``: return ``(group_key, slot)``. The default
  ``(None, None)`` is an independent sample. A ``None`` slot uses the report
  ID. A repeated slot keeps the first accepted report.
- ``decide(key, members)``: return ``GroupDecision.INCOMPLETE``, ``READY``
  or ``DISCARD`` for the group's ``ReportContext`` values, in arrival order.

A group is fixed when it is first ready. Later reports with the same group
key form a new group. A report whose trajectory holds an inference of
another report ends without training.

A stage that builds policy samples with ``SampleAssembly`` takes these
config keys (``SampleAssemblySettings``):

- ``accept_multi_turn_policy_samples`` (default ``false``): with ``false``,
  only the last reply of the report's trajectory trains. With ``true``,
  every reply of the trajectory trains.
- ``realign_threshold`` (1024) and ``scaffold_tolerance`` (0): how a
  multi-turn sample handles a prompt that rewrote earlier turns.

Computed feedback
-----------------

Computed feedback is derived from the traffic, usually by a model call.
Write a ``RecordJudging`` and a ``JudgedSamples`` subclass:

- ``RecordJudging.observe(record, now, stage)``: correlate one record with
  the stage's ``track``, ``tracked_record``, ``dispatch`` and ``retire``. It
  must not block.
- ``async RecordJudging.judge(job)``: the judgment of one job. It runs on
  the worker and may call models.
- ``RecordJudging.expire(now)``: return the tracked receipts whose wait is
  over. The default expires nothing.
- ``JudgedSamples(judging, worker)``: the stage constructor, with the
  judging and the ``BackgroundJobWorker`` that runs ``judge``.
- ``JudgedSamples.make_sample(record, judgment)``: return one
  ``TrajectorySample``, or ``None`` for a record that cannot train.

Without a worker, records track and expire, and the stage judges nothing.
``recipes/openclawrl/`` is an example.

Slow work
---------

A stage that needs slow work must not block the samples service:

1. ``push`` keeps the records and submits the work to a worker that the
   stage owns.
2. ``background_jobs_running()`` returns ``True`` while the work runs.
3. When the work ends, the next ``advance(now)`` takes the result and forms
   the set.
4. ``close`` stops the worker. A worker that cannot finish cleanup within
   its close timeout raises ``TimeoutError``.

A stage that waits for time alone, such as an idle trajectory, returns the
monotonic due time from ``next_due()``. The service runs the next selection
at that time. See `background job ownership
<sandboxes.rst#background-job-ownership>`__ for cancellation and resource
ownership.

Staleness
---------

The samples service alone decides whether a stored sample may still train.
The stage gives the rule:

.. code:: python

   def is_stale(self, samples: Sequence[Sample], served: ServedArtifacts) -> bool: ...

``samples`` is one group of stored samples that no job trained. ``served``
holds the artifacts that the scenario served, oldest first, and the artifacts
that it serves beside its active selection. The service asks again after
each commit that changes what the scenario serves, and when the stage's
config changes. It marks every sample of a stale group in the store.
Control's automatic jobs leave a marked sample out, and wait until the
service judged against what the scenario serves now.

The rule of each built-in stage:

- Default (``lags_behind(samples, served, 0)``): on-policy. A sample is
  stale when its data is of an artifact other than the served one. A sample
  whose record names no artifact is stale. A task is never stale.
- ``SFTSamples``: never stale (``goes_stale = False``). The target is fixed
  text.
- ``TraceSamples`` and ``InferenceSamples``: never stale. A harness proposer
  reads what the served system did.
- ``SAOSamples``: accepts a lag of ``max_artifact_lag`` artifacts
  (default 0).

A stage whose method accepts a bounded lag passes its bound to
``lags_behind``. Guarantees:

- A stale sample stays in the store and is never judged again. An artifact
  that serves again does not make its samples fresh.
- A training request can still name a stale sample: the client chose it.
- The lag counts only the artifacts that this scenario served.
- A scenario without a runtime, or without a weight component, has no
  served artifacts: no sample is stale.
- An uploaded trajectory is never stale.

Samples from no record set
--------------------------

A stage can make samples that come from no record, such as generated tasks.
Both hooks default to doing nothing:

- ``made_samples()`` returns the ``(name, sample)`` pairs that the stage made
  since the last call. The service stores each as sample ``<name>``, with no
  group and no consumed record. A name must always name the same sample. A
  sample that the store holds already changes nothing, so a stage may return
  it again after a restart.
- ``job_ended(outcome)`` tells the stage that a training job ended and that
  the commit of its end is durable. ``outcome`` is a ``JobOutcome``
  (``reef.core.job_data``): ``job_id``, ``backend``, ``event`` (``finished``
  or ``failed``), ``metrics`` and ``error``. A trainer that starts with no
  running job gives the last job's outcome again, so handle a repeated
  outcome.

The built-in generated-task recipe uses both hooks (see `Task generators
<task-generators.rst>`__). A stage that saves tasks opens a store with
``context.task_storage.open(name)``; ``task_storage`` is ``None`` when the
trainer has no scenario session (see `Task stores
<../reference/python-api.rst#task-stores>`__).
