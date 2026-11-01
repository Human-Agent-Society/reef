Training pipeline
=================

A scenario's training pipeline turns records into typed job data: the data
of one training job. It has two stages, with samples between them. A recipe
writes the stages; Reef owns the pipeline that drives them. This page
explains the stages, what a recipe writes for each, the job size, the training
modes, the task generation contract, and the path a record takes to a job.
Storage owns deletion.

.. page::
   :for: recipe authors writing the stages of a method, and anyone reading ``reef/train/pipeline.py``
   :needs: the stage hooks from `Python API <../reference/python-api.rst#pipeline-stages>`__
   :outcome: which stages a method needs, what it writes, and what it never writes

The two stages
--------------

.. code:: text

   record ─► record2sample ─► sample ─► sample2job ─► job data ─► trainer

- The **record-to-sample stage** (``Record2Sample``, ``reef.record2sample``)
  decides which records train, and in which record sets and units (see
  `Record sets and their units <#record-sets-and-their-units>`__). It turns
  each record set into samples. A sample is a ``Sample``: a
  ``TrajectorySample`` that trains directly, or a ``TaskSample`` (a Harbor
  task) that an algorithm rolls out. The buffer, and so one job's data, holds
  samples of one kind.
- The **sample-to-job stage** (``Sample2Job``, ``reef.sample2job``)
  shapes the samples of the buffer into one ``JobData``. Grouped
  recipes set ``group_id`` on each member. The stage also sets
  ``job_samples``, the fewest samples of one job, and ``buffer_size``, the
  most samples that one job takes from the buffer (see `Job size <#job-size>`__).

.. list-table::
   :header-rows: 1

   * - Stage
     - In
     - Out
   * - ``Record2Sample``
     - ``push(record)``
     - ``take_ready_units() -> tuple[RecordSetUnit, ...]``, and
       ``unit_samples(unit) -> ((set_id, samples), ...)``
   * - ``Sample2Job``
     - ``make_job_data(samples, data_id)``
     - ``JobData``

The record-to-sample stage never calls the pipeline. When the samples service
asks the pipeline for new samples (``produce``), the pipeline first calls the
stage's ``advance(now)``. The stage then takes in background work that ended,
such as judgments, and forms the sets that are due, such as the trajectories
that went idle.

The recipe names a default class for each stage in its ``pipeline_spec()``.
The deployment's ``pipeline`` section gives each stage its config and can name
another class (see `the pipeline section
<../reference/configuration.rst#the-pipeline-section>`__). For each scenario,
Reef builds the stages in pipeline order with the class method ``from_config``
and composes them into a ``TrainingPipeline``:

- ``Record2Sample.from_config(config, *, context, environ)``
- ``Sample2Job.from_config(config, *, context, record2sample, environ)``

The sample-to-job stage receives the record-to-sample stage. It reads a value
that the record-to-sample stage owns from that stage, so the value is
configured once: TTTD's sample-to-job stage reads the step grid from
``TTTDSamples``, and OpenClaw-RL's reads the judging of ``OpenClawRLSamples``.
``config`` is the stage's own config. A stage
declares its keys as a frozen settings dataclass that inherits
``reef.core.plugins.PluginSettings`` (fields made with
``reef.core.config.config_option``). Its ``from_config`` parses the config
with ``Settings.parse(config, plugin_class=cls)`` and gives the settings
object to the constructor; the stage never reads a raw mapping. A stage
without config parses with ``NoSettings``, which refuses any key. A
record-to-sample stage that holds units parses ``Record2SampleSettings`` or a
settings class that extends it, and gives its ``max_staleness`` to
``Record2Sample.__init__`` (see `Staleness <#staleness>`__). Reading the
deployment checks the stage classes; a bad config fails when the first
scenario builds its pipeline. The default record-to-sample stage,
``NoSamples``, makes no sample, so the pipeline makes no job data.

Nothing numeric lives in the stages. Advantages and the loss family are the
step objective's. See `job data values <../reference/python-api.rst#job-data>`__ for
formats and algorithm support.

Record sets and their units
---------------------------

The record-to-sample stage forms two kinds of value. Each kind has an exact
definition.

A record set (``RecordSet``) is the unit that the stage makes samples from. A
value is a record set if and only if it holds one or more
records, and it holds each record that its records reference. Reef links the
inference records of a scenario into trajectories: a call's parent is the
earlier call with the same API format and tools whose request is the longest
prefix of its own (``reef.core.trajectory_index``). A report references the last
call of each trajectory that it scores; the report and the calls of those
trajectories form one record set. The calls of a trajectory that ended without
a report (``TrajectorySamples``), one judged record, or one inference
record also form one. No record is in two record sets of one stage: ``form``
refuses a set that holds a record of another set. Thus a report whose
trajectory holds an inference of another record set ends without training.

``form`` makes the samples of a record set (``make_samples``) at once. The
result is one ``SampleResult``. Its status settles all records of the set
together (``settle``):

- ``READY``: one or more samples. The stage holds them, by set ID, in
  ``set_samples``, and the records wait in one record set unit.
- ``SKIPPED``: no sample. The stage ends the records.

Before it makes samples, ``form`` checks the inference responses of the set
for an engine abort. An abort rejects the entire set, including a report and
all its referenced inferences. The stage uses its normal rejection and release
lifecycle; it does not alter the stored responses. Retained records pass the
same check during replay. A record holds the provider exchange and no
training tensors; the training component renders each sample (see `Rendering
in the training component`_).

A record set unit (``RecordSetUnit``) holds one or more record sets that must
train in the same job. A report by itself is a unit of one record set. A
ready group is a unit of the record sets of all its reports. The stage makes
a unit ready when the samples of all its record sets are made.

When a unit is ready, the pipeline saves the unit's samples
(``unit_samples``) to the scenario's sample store in one transaction
(``TrainerSamples.add_record_samples``). The transaction also saves the IDs of
the unit's records, so it consumes the records at once. The stage then ends
the unit (``consume``) and forgets its samples. Sample ``i`` of record set
``s`` gets the ID ``<trainer>:<s>:<i>``, or ``<s>:<i>`` for the only trainer
of a scenario. The samples of a unit with more than one sample share a group
in the store, so they train in the same job.

A restart never makes stored samples again: recovery skips the records of
every sample that the trainer stored. Thus a nondeterministic stage, such as
a judge, does not run a second time on the same records.

The pipeline
------------

``TrainingPipeline`` (``reef/train/pipeline.py``) is the only object that the
trainer gives records to and takes job data from. Recipes do not subclass it.
The trainer calls it on its own thread, under its lock; no call may block:

- ``push(record)``: one record arrives. The pipeline pushes it to the
  record-to-sample stage, which can form record sets and make their samples.
  Then the pipeline saves the samples of each unit that became ready to the
  sample store (``store_samples``).
- ``produce()``: advance the record-to-sample stage to the current time
  (``advance(now)``), then save the samples of each unit that became ready.
  The samples service calls it, so background work, such as a judgment, can
  make samples without a new record (see `Slow work <#slow-work>`__).
- ``advance()``: read the samples that were stored since the last call. It
  needs no sample count and reserves no job data. Call it on the same
  serialized path as other pipeline operations; it does not make the pipeline
  thread-safe.
- ``ready(sample_count)``: call ``advance()`` and check whether job data of
  ``sample_count`` samples is available. The trainer passes ``job_samples``.
  With ``sample_count`` None, the buffer never makes the pipeline ready.
- ``build_job_data(sample_count)``: give the samples of the buffer (see `The
  buffer <#the-buffer>`__) to ``make_job_data``, in store order, and reserve
  them. The trainer validates the job data against ``output_schema``. The
  same job data stays reserved until it is acknowledged or released
  (``release_job_data``). Released samples stay in the buffer.
- ``acknowledge(data_id)``: the commit that submits the job consumed that job
  data. The pipeline returns ``sample:<sample id>`` for each sample of the
  data, and the instruction's ID in ``manual``. The commit lists these IDs.
  In the same transaction, the sample store adds one to the train count of
  each listed sample for the commit's trainer. A job that later fails does
  not give its samples back: they never train again for that trainer.

A record-to-sample stage can also make samples from no record set, such as
generated tasks. Two optional hooks serve such a stage; their defaults do
nothing:

- ``made_samples()``: return the samples that the stage made since the last
  call, as ``(name, sample)`` pairs. The pipeline stores each one as sample
  ``<name>`` (``<component>:<name>`` for a trainer of a scenario of several),
  with no group and no consumed record. A sample that the store holds already
  changes nothing, so a stage may return a sample again after a restart. A
  stored sample trains from the buffer like every other sample.
- ``job_ended(outcome)``: a training job ended, and the commit of its end is
  durable. ``outcome`` is a ``JobOutcome`` (``reef.core.job_data``): the job
  ID, the component name, ``event`` (``finished`` or ``failed``), the job's
  ``metrics``, and ``error``, which is set exactly when the job failed. A
  trainer that starts with no running job calls it again for the last job
  that ended, before any other call to a new stage.

The generated-task recipe trains its saved tasks this way (see `Task
generation contract <#task-generation-contract>`__).

The pipeline also merges the stages' ``status()`` and metrics.
``background_jobs_running()`` is true while the record-to-sample stage runs a
background job, such as a judgment or a task-generation round. The scenario's
worker then polls at a fixed interval. ``close()`` closes both stages.
``releasable_record_ids()`` returns
the completed records that no held set references, and
``release_records(ids)`` frees their memory after a successful commit. New
commits retain the stored bodies; consumption progress prevents retraining
on restart. Storage can independently evict any body under capacity pressure,
with warnings and durable loss totals. In a scenario with one trainer per
component, every trainer reads the same rows and keeps its own consumption,
so a row one pipeline releases is still stored for the others. When the
scenario admits a wider report contract than a recipe's own,
``admit_reports_of(report_type)`` names it, and the record-to-sample stage
releases a report shaped for another component instead of failing it. The
pipeline passes such changes to the stage as its updated ``PipelineContext``
(``context_changed``).

Job size
--------

Reef submits one training job for each job's data. ``job_samples`` is the
fewest samples that one job trains, a positive int. The deployment sets it in
``pipeline.sample2job.config.job_samples``. The training component then cuts the
job's data into optimizer batches by its own batch size: Slime's
``--global-batch-size`` or Tinker's ``batch_size``.

- A weight recipe has no default ``job_samples``. A deployment without it
  fails the recipe build with ``RecipeConfigError``. Usually, set it to the
  component's batch size, so that one job is one optimizer step.
- The harness evolution stages (``TraceJobData``, ``TrainTaskJobData``)
  default ``job_samples`` to 1 when the stage config sets none.
- A stage whose job data is one fixed unit, such as the TTTD and SDPO grid,
  passes ``max_job_samples`` to ``Sample2Job.__init__``. The stage refuses a
  larger ``job_samples`` with ``PluginConfigError``.

The pipeline is ready when the buffer holds at least ``job_samples`` samples.
The buffer takes whole groups, so one job's data can hold more. Thus
``job_samples`` and the component's batch size can differ:

- A unit packs several samples, for example a group of rollouts. The job's
  data can then hold more samples than ``job_samples``, and the component trains
  more than one optimizer step.
- A job's data that is smaller than the component's batch size trains as one
  partial step.

The ``remainder`` field of the job's optional ``scheduling`` parameter
decides what happens to a shorter last step. With ``packing_length`` in the
same parameter, a step trains batch-size packs of tokens, not batch-size
samples. ``job_samples`` still counts samples, so the job's packs seldom fill
whole steps. Set ``job_samples`` to the samples of many steps.

The buffer
~~~~~~~~~~

The buffer is the samples that the next job takes. The pipeline computes it
from the scenario's sample store, in the same way in every training mode:

1. It takes the stored samples that the trainer reads and did not train yet:
   the samples that the trainer made from records, and the uploaded
   trajectories (see `Uploaded trajectories <#uploaded-trajectories>`__).
   They are in store order: units in the order they became ready, and
   uploads in upload order, interleaved by arrival. The samples of job data
   that the pipeline handed out are not in the buffer. Imported ``train`` tasks
   are in it when the sample-to-job stage sets ``accepts_imported_tasks``;
   imported ``eval`` tasks never are, since an evaluation runs them.
2. It keeps the samples of one kind, trajectory or task: the kind of the first
   sample. In ``auto`` a kind that the recipe gives no job parameters does not
   count; such samples train only for an instruction that gives them.
3. It leaves out each stale sample that the trainer made from records (see
   `Staleness <#staleness>`__), and every other sample of its group. An
   uploaded trajectory is never stale.
4. It takes the first ``buffer_size`` samples that remain, and the rest of
   each group that those samples touch.

Thus ``buffer_size`` is the most samples that one job takes from the
buffer, beside the rest of a group. The ``Sample2Job`` sets it, and refuses one
smaller than ``job_samples``. Without a ``buffer_size`` setting, it is
``job_samples``. The pipeline never releases or drops a sample: samples
beyond the buffer wait in the store for later jobs, and a stale sample stays
in the store untrained.

``status()`` reports ``ready_samples`` (the samples of the buffer) and
``buffer_size``; the trainer adds
``job_samples``. A stage adds an entry only when a terminal outcome cannot become job data and an
external runner must stop waiting (TTTD reports a complete mixed-artifact
step as an invariant failure).

Slow work
---------

``push``, ``advance`` and ``make_samples`` run on the trainer thread and must
not block on network or model calls. Most stages make the samples of a set in
``form``, when ``push`` forms it. They implement ``push`` and
``make_samples(record_set)``.

A stage that needs slow work, such as a judgment, must not block the trainer.
Its ``push`` keeps the records and submits the work to a worker that the stage
owns. When the work ends, ``advance(now)`` takes the result and forms the set,
which makes its samples. Until then the stage holds the records, and
``background_jobs_running()`` returns True. The stage's ``close`` stops its
worker. A worker that cannot finish cleanup within its close timeout raises
``TimeoutError``; callers must not treat that as completed shutdown. See
`background job ownership <sandboxes.rst#background-job-ownership>`__ for
cancellation and resource ownership.

Uploaded trajectories
---------------------

A client can give a scenario complete ATIF trajectories as data items (see the
`HTTP API <../reference/http-api.rst#data-items>`__). A trajectory item is not a
record. Reef copies each item that a scenario uses into the scenario's sample
store, as an upload: a sample of no component. The pipeline reads it from there.
When the item leaves the scenario, Reef removes the upload, and each trainer
forgets it.

A recipe takes uploads when its ``Sample2Job`` sets ``accepts_trajectories``
to True. The default is False. Reef refuses with 400 an item that names a
scenario with no trainer that accepts uploaded trajectories.

The record-to-sample stage turns each upload into its training sample with
``Record2Sample.uploaded_sample(trajectory)``. The default returns the trajectory
as uploaded. A stage can override it to check the structure or add
metadata: ``SFTSamples`` refuses a trajectory with no agent turn to train and
adds its ``max_tokens``. The method must be pure. Admission calls it from the request
thread for each trajectory and each trainer that accepts uploads; a
``ValueError`` refuses the whole upload with 400, and the message names the
trajectory. The pipeline calls it again when it reads the upload. A
trajectory that ``uploaded_sample`` refuses when the pipeline reads it is
skipped with a warning and stays unconsumed. The training component renders
the trajectory when a job trains it; a trajectory that cannot render, or that
is longer than its ``max_tokens``, is left out of the job.

An uploaded trajectory is not a record, so the stage never forms a set of it.
In the store, it is a sample like the samples that the trainer makes from
records: it waits in the buffer in store order, and it counts as one sample toward ``job_samples`` and
``buffer_size`` (see `The buffer <#the-buffer>`__). In a scenario
of several components, each trainer that accepts uploaded trajectories trains
on them. The commit lists
each uploaded trajectory as ``sample:<trajectory_id>``, and the scenario's
sample store adds one to the trajectory's train count for the commit's trainer
in the same transaction. Thus a rebuilt trainer does not read again a
trajectory that it trained, and each trainer of several components still
trains it.

Like a sample made from records, an uploaded trajectory has no tokens and
no log probabilities: the training component renders it and computes them.
An objective that needs a reward, such as a policy-gradient objective, needs
a finite ``extra.reef.reward`` in the trajectory; an SFT sample needs none.

The record-to-sample stage
--------------------------

Besides the record sets and their samples (see `Record sets and their units
<#record-sets-and-their-units>`__), the stage owns the record lifecycle and
the staleness check. It places reports in their groups, and it ends units and
records that can no longer train.

A record that the stage ends, for example a report in a taken slot, appears
in ``releasable_record_ids()``. The next commit
names it, so the record never trains after a restart. The default bookkeeping
keeps ended records until their release; a stage with its own record
lifecycle overrides the lifecycle methods.

Staleness
~~~~~~~~~

Every record-to-sample stage runs the staleness check. No other part of Reef
checks which artifact produced training data. The limit is the stage's setting
``max_staleness`` (``Record2SampleSettings``), set in
``pipeline.record2sample.config``:

.. code:: yaml

   pipeline:
     record2sample:
       config:
         max_staleness: 2

``max_staleness`` is a non-negative int, and its default is 0. With 0, only
data of the artifact that the scenario serves trains. A negative value fails
the pipeline build. A stage settings class with more keys extends
``Record2SampleSettings``, so each stage takes the key next to its own keys.
The check has these steps:

1. The inference service writes the ``artifact_id`` of the artifact that
   answered into each inference record. The sample keeps each record's
   payload in ``extra.reef.records``, and ``TrajectorySample.artifact_id``
   reads the ``artifact_id`` of its last call. A task rollout of a training
   component names its artifact in ``training.artifact_id`` instead.
2. When the scenario reserves job data, it gives the pipeline the
   ``artifact_id`` of the artifact that serves it now
   (``observe_serving_artifact_id``). The stage keeps these artifacts, oldest
   first.
3. Each time the pipeline computes the buffer, it asks the stage whether each
   sample that the trainer made from records has data more than
   ``max_staleness`` artifacts behind the current artifact
   (``Record2Sample.is_stale``). The buffer leaves out a stale sample and
   every other sample of its group. The sample stays in the store untrained.

The lag counts only the artifacts that this scenario served. A publication of
another scenario on a shared engine does not age the data. An ``artifact_id``
names the same weights in every process and after an engine restart. These
rules also apply:

* A sample whose record names no artifact is stale.
* A sample from an artifact that the stage did not see is stale. A rebuilt
  trainer knows only the artifacts that it sees after the rebuild.
* A scenario that has no runtime, or whose surface has no component that a
  runtime loads (no ``loader``), has no staleness check: no data is stale.
* Until the scenario first gives the stage a served artifact, no data is
  stale.

Reef's check is the only staleness filter that Reef runs. A training
framework can have its own off-policy limit. Such a limit goes in that
backend's ``training.options``, not in the record-to-sample stage. No
built-in backend has one now.

Two stages do not take ``max_staleness``. ``NoSamples`` holds no unit.
``SFTSamples`` never finds a sample stale (its ``is_stale`` returns False),
because an SFT target is fixed text, not sampled tokens. Both refuse the key.

Built-in stages
~~~~~~~~~~~~~~~

A recipe's stage extends one of these classes of the ``reef.record2sample`` package:

.. list-table::
   :header-rows: 1

   * - Stage
     - Decides and makes
   * - ``NoSamples``
     - nothing; it makes no sample. The default of a pipeline that trains no weights
   * - ``ReportSamples``
     - report validation, groups and slots, source release; one sample per report (``make_sample``). With
       ``exclusive_sources=True``, a report that ends owns its inference records
   * - ``TrajectorySamples``
     - one sample per trajectory that ended without a report, after ``trajectory_idle_seconds`` without a
       continuing call
   * - ``JudgedSamples``
     - correlation, the judging worker, and one sample per judged record (``make_sample``)
   * - ``InferenceRecordSamples``
     - one set per inference record (``make_samples``); other records are released
   * - ``DistillSamples``
     - one distillation sample per report: the student's call plus the request that its teacher reads

Rendering in the training component
-----------------------------------

A sample that the record-to-sample stage makes is an ATIF trajectory with its
reward, feedback and metadata. It holds no tokens and no log probabilities:
serving returns none. ``SampleAssembly.build`` makes a report's sample:

- Without ``accept_multi_turn_policy_samples``, the sample is the
  trajectory's last call. Its earlier agent steps are ``is_copied_context:
  true``, and only the last reply trains.
- With ``accept_multi_turn_policy_samples: true``, the sample holds every
  call of the report's trajectories, and every reply trains.

The training component renders each sample with the model's chat template
(``reef.core.chat_rendering``). It computes the log probabilities that its
objective needs with the weights of the job's start, the current weights.
They are exact when ``max_staleness`` is 0 and an approximation above it.
A stage can add what the component renders with:

- ``extra.reef.teacher``: the teacher request of a distillation sample. The
  component renders the teacher sequence: the teacher prompt followed by the
  student's reply tokens, or the student's own tokens for a request without
  messages.
- ``extra.reef.teacher_candidates``: one teacher request for each hint, as
  OpenClaw-RL writes them.
- ``extra.reef.max_tokens``: the longest sample that may train, as SFT
  writes it.

The component leaves out a sample that cannot render, that is longer than
its ``max_tokens``, or whose teacher sequence is longer than the request's
``max_tokens``. The job metric ``unrenderable_samples`` counts them. The
component must render as the serving engine did: give it the engine's chat
template arguments (see `Chat template arguments
<../reference/configuration.rst#chat-template-arguments>`__). A task rollout
of a training component already holds the exact tokens and log
probabilities of its own engine, and the component trains them as they are.

Training modes
--------------

``training_mode`` is an attribute of each pipeline. The recipe passes its
initial value through ``Trainer.build`` and ``PipelineContext``; it defaults
to ``auto``. A ``Sample2Job`` declares the modes it supports in
``supported_training_modes``; the default is ``auto`` and ``manual``. Record push,
sample storage and acknowledgement use the same methods and the same buffer
in both modes. ``GET /reef/status`` reports ``pending_instructions`` as the
pipeline's ``buffered_requests`` plus the instructions still unread in
storage.

- ``auto`` makes job data when the buffer holds ``job_samples`` samples.
- ``manual`` makes job data only for a queued TRAIN instruction. The
  instruction's data is the trainer's manual buffer
  (``TrainerSamples.manual_buffer``): the samples that a client chose with
  ``TrainingPipeline.insert_into_buffer`` and ``remove_from_buffer`` (`Choose
  the samples that train
  <../reference/http-api.rst#choose-the-samples-that-train>`__), in store
  order, whatever ``job_samples``. New samples never join it by themselves.
  The pipeline waits until the manual buffer holds a sample. A ``Sample2Job``
  that sets ``reads_instructions`` runs the instruction at once, also with an
  empty buffer; harness evolution sets it, because its proposer reads the
  instruction. The commit of the job removes the trained samples from the
  manual buffer. A switch from ``auto`` to ``manual`` puts the automatic
  buffer into the manual buffer.

The pipeline runs the oldest instruction first, one per step. It gives the
instruction's data the id ``<scenario>:instruction:<request id>`` and attaches
the instruction to ``JobData.request``. Automatic job data gets the id
``<scenario>:data:<number>``. Acknowledgement also consumes the
instruction. ``make_job_data`` must not call models or perform training.

When the pipeline supports ``manual``, the scenario gives it the queued
training instructions before each step (``TrainingPipeline.queue_instructions``);
instructions are not records. The pipeline keeps them FIFO regardless of the
selected mode; ``auto`` leaves them for ``manual``. The pipeline stores the samples of records in both modes, so
changing to auto can train on data already collected.

Unsupported modes raise ``NotImplementedError``. An invalid mode name raises
``ValueError``.

Changing training mode
----------------------

``POST /reef/scenarios/{scenario}/update`` selects ``auto`` or ``manual`` on
the existing pipeline. The trainer serializes selection with record push and
reservation. Reserved job data stays unchanged and acknowledgement consumes
its actual contents, independently of subsequent mode changes.

Buffered data and queued instructions stay on the same instance. Reef stores
the selected mode in the scenario's settings, so a reload or a service restart
applies it again.

The two feedback paths
----------------------

Reported feedback arrives in reports. Computed feedback is derived from the
traffic, potentially with slow model calls on a worker. Reported feedback has
no ``judge`` hook; computed feedback keeps its asynchronous ``judge``.

Reported feedback
~~~~~~~~~~~~~~~~~

Before accepting a new report, Reef checks its schema and verifies that every
reference identifies an existing inference in the same scenario, and that no
reference lies inside the trajectory of another: a report references only the
last call of each trajectory. Missing, wrong-kind, duplicate, foreign, or inner
references raise an error. Reports are not
queued waiting for future inference records. Storage is authoritative; a record
need not have reached the pipeline's in-memory cache at admission time.

Reports cannot opt out of training: ``metadata.training.eligible`` is rejected,
including when its value is ``true``. Valid low or negative scores are feedback,
not a reason to discard the report.

A reported-feedback recipe writes:

- a ``ReportSamples`` subclass as its record-to-sample stage. Its constructor
  takes the recipe's ``ReportGrouping`` (below), ``exclusive_sources``, and
  ``max_staleness``: ``ReportSamples(context, grouping=None, *,
  exclusive_sources=False, max_staleness=0)``. With no grouping, every valid
  report trains by itself.
- ``make_sample(context) -> Sample`` on that subclass, when the default does
  not fit. The default, as in ``TrajectorySamples``, makes the report's
  trajectories one ATIF trajectory sample with the report's score. ``context.inferences`` holds the calls of the
  trajectories, oldest first, and ``context.ends`` the referenced last calls; a
  method that trains one call reads an end, whose request holds the earlier
  turns. Use ``context.require_score()`` when the method needs a reward. The
  stage attaches the source and report ids to the returned item.
- ``make_job_data(samples, data_id)`` on a ``Sample2Job`` subclass: assemble
  the flat tuple of the buffer's samples into job data with the ID ``data_id``.
  The job's commit consumes every sample of the tuple, including samples that
  the recipe removes from training.

A report that ends after it was a record set owns its referenced inference
records outright. ``exclusive_sources`` gives the same ownership to a report
that ends before.
The grouping runs on the trainer thread and reads only records:

- ``is_training_report(report)`` when another role's reports share the
  scenario: return ``False`` for a valid report that is not this method's
  training data (SPADE's Designer reports its own score there). The stage
  releases such a report and never assembles it; its sources go with it
  only under ``exclusive_sources`` or when it references more than one
  inference, and a single referenced inference stays retained for a report
  that trains on it. The default takes every report.
- ``placement(context)`` for grouped methods: return ``(group_key, slot)``.
  The default ``(None, None)`` makes an independent sample. A None slot uses
  the report id; repeated slots preserve the first accepted report.
- ``decide(key, members)`` for grouped methods: return ``INCOMPLETE``,
  ``READY``, or ``DISCARD``. The members are the group's ``ReportContext``
  values in arrival order. The collection group can span multiple training
  comparison groups, as in TTTD.

``make_sample`` can return ``SkippedReport(reason)`` for a valid report that
cannot train. The stage releases the report.

The stage handles report-id deduplication, group-slot retries, consumption,
and retention. Job data is ready in automatic mode after the buffer holds
``job_samples`` samples; a unit is one report or one complete group. A group
is fixed when it is first ready: the pipeline stores its samples, and later
reports for the same group key form a new group. A report whose trajectory
holds an inference of another report ends without training, whether the
other report waits or already trained.

Training data errors propagate: an unsupported trajectory raises an error
rather than dropping the report. Failed sample assembly keeps its input records
protected and does not mark the report successfully processed.

Computed feedback
~~~~~~~~~~~~~~~~~

A computed-feedback recipe writes a ``RecordJudging``: correlation in
``observe(record, now, stage)`` with the stage's ``dispatch``, ``track``,
``tracked_record``, and ``retire`` operations, ``async judge``, and ``expire``
for tracked records that time out. The recipe's record-to-sample stage is a
``JudgedSamples`` subclass, ``JudgedSamples(judging, worker, *,
max_staleness=0)``, with the judging and the worker that runs ``judge``. It
implements ``make_sample(record, judgment)``, which returns one sample or
``None`` for a record that cannot train. Its ``Sample2Job`` supplies
``make_job_data``. When a judgment arrives, the stage forms a judged set and
makes its sample. The stage holds the sample until its unit is ready; the
pipeline then stores it.

Task generation contract
------------------------

A deployment can train on tasks that a task generator makes. The deployment
selects the built-in generated-task recipe,
``reef.recipe.generated_tasks:GeneratedTaskRecipe``, configures its stages in
the ``pipeline`` section, and names a ``TaskGenerator`` class in
``pipeline.record2sample.config.generator`` (see `generated tasks
<../reference/configuration.rst#generated-tasks>`__). The recipe does not
depend on the generator's method, so a team can replace the generator without
a new recipe. SPADE (``recipes/beta/spade``) is one generator.

``TaskGenerator`` is in ``reef.recipe.generated_tasks``, and
``TaskGenerationRequest`` in ``reef.record2sample.generator``. Reef calls its class method
``from_config(config, *, scenario, environ)`` once for each scenario. The
generator gets what it needs from its ``config`` and from these calls only:

- ``async generate(request: TaskGenerationRequest) -> tuple[HarborTask, ...]``
  returns zero or more tasks. Preserve the request's source record IDs in each
  task's ``source_agent_record_ids``. ``request.generation`` numbers the call.
  An exception is a failure to run, and the recipe retries the generation.
- ``async validate(task: HarborTask) -> TaskValidationResult`` checks one
  task. Empty ``errors`` means the task passed. Return task defects as errors.
  Raise when the checks cannot run; the recipe retries the generation. The task is a value, so the check does not
  need a file system that it shares with Reef.
- ``async record_results(results: tuple[GeneratedTaskResult, ...], *, generation: int)``
  receives the results of the ``generate`` call numbered ``generation``, once
  every train task of it has a result. Later generations may have started
  before, and the process may have restarted since. A generator keeps what it
  needs for each open generation across a restart, and ignores a second call
  for a generation that it already closed. The default keeps nothing.

Each ``GeneratedTaskResult`` is one of four cases, by the candidate's outcome:

- ``RejectedTask(task, reason)``: the candidate failed validation or
  conflicted with the task store, so it was never published.
- ``HeldOutTask(task)``: the task is in the eval split. It runs no rollouts. A
  generator must not use it as training feedback or as experience for later
  generations.
- ``RolledOutTask(task, rollout_rewards)``: the first rollout of the train task
  ran. ``rollout_rewards`` holds one verifier reward for each training rollout
  that ran, as the rollout's policy scored it before it trained on it. An
  rollout that ran without a reward counts as 0.
- ``FailedTask(task, error, infrastructure_failure)``: the first rollout ran no
  rollout. ``infrastructure_failure`` is true when the task could not run at
  all, for example because the sandbox failed; it is not a failure of the
  solver. The built-in recipe sends it for a task that a task job trained
  without reporting its rewards, as an infrastructure failure.

The training component scores a training rollout whose verifier ran past its
timeout with ``rollout.verifier_timeout_reward``, 0 by default, so a rollout
of that kind trains. The task's oracle passed the same verifier, so the
timeout is the agent's.

The recipe's record-to-sample stage, ``GeneratedTaskSamples``, runs the
rounds. A round is not a record set, and it does not train. The stage decides
when a round is due, holds the source sessions, and runs the round's
generation on its own ``BackgroundJobWorker``, outside the trainer lock. At
each ``advance``, it ends the rounds that finished and starts the next round
when it is due. One generation runs at a time.

A generation job runs no rollouts. It:

1. Asks the generator for tasks with the configured ``description`` and the
   round's source sessions.
2. Saves the candidates in the scenario's task store, then validates each
   task with the generator.
3. Publishes each valid task with its train/eval split. It runs no eval
   tasks.

A generator or check that fails to run is retried from the saved candidates,
up to ``generation_attempts`` times within ``generation_deadline_seconds``.
After that the round ends without a task, and the next round takes the same
sessions again, so it finishes the same generation. A task identical to one
that was already published is rejected as a duplicate. The record-to-sample
stage hands each train task of a complete generation to the pipeline
(``made_samples``), which stores it in the scenario's sample store as
``generated-tasks/<task name>``.

Stored tasks train from the buffer as task jobs, like every other sample.
Reef runs no rollout of them: the training component rolls each job's tasks
out on its own sampler, at the job's start weights, and trains on their
rollouts.

1. ``GeneratedTaskJobData`` makes one job's data of the buffer's tasks: the
   ``TaskSample`` values. A job takes at least ``job_samples`` tasks, 1 by
   default, and at most ``buffer_size``, 4 by default.
2. The trainer submits the job. Its ``rollout`` parameter (``recipe.config.job_parameters.rollout``)
   tells the component how many rollouts of each task to run, and with which
   agent and sandbox. The Tinker component runs them, assembles one sample
   from each rollout that ran, scored with the task's verifier reward, and
   trains on the groups of at least two samples. It reports each task's
   rollout rewards in the job's metrics (``rollout_rewards``).
3. The commit of the submission counts the tasks trained in the sample
   store, as it does for every sample.
4. When the commit of the job's end is durable, the trainer gives the job's
   outcome to the record-to-sample stage (``Record2Sample.job_ended``). The
   stage gives each task that a committed job trained and that has no result
   its rewards from ``rollout_rewards``. A task without rewards, for example
   in a job that failed or that a ``train-from`` cancelled, gets an
   infrastructure-failure result, so its generation's feedback still goes out.

A result that the task store fails to save is saved again by the next job. On
a restart, the sample store still holds the train tasks that did not train,
and the buffer takes them again. A job that runs across a restart saves its
tasks' results when it ends. A trainer that starts with no running job gives
the stage the last job's outcome again, so a restart just after a job's end
loses no result. In ``manual`` mode, a client chooses the stored tasks that
train, as for every sample.

When every train task of a generation has a result, the next job sends the
generation's results to ``record_results``. A generation can therefore close
after later generations started. A generation whose delivery fails waits for
the next job, and later generations still go out.

A generator can run rollouts of a task through the generator service, as
SPADE does for its hint plays; those rollouts are not records, and do not
train.
The ``task_group`` objective centers and scales each reward within its task
group, on the ``importance_sampling`` loss family (Tinker's built-in loss, or
truncated importance sampling on Slime). The stage trains on no
report: it holds the reports that are source sessions, and releases the others,
such as a generator's own reports, without assembling them.

A round can also generate tasks from recorded sessions. Set
``source_records_per_round`` above 0 and ``source_tag`` to a
``name=value`` tag. A report without a training arm, whose inference records all carry the
header ``x-reef-tag-<name>: <value>``, is a source session. The stage
holds source sessions in arrival order. Each round takes at most
``source_records_per_round`` of them, and a round without a waiting session
has none. The generator receives each session's inference records and then
its report as ``source_records``. The stage keeps a session's records
while its round runs, then releases them. A session that arrives after the
last round is released at once.

The first round starts when the pipeline first looks for job data. The next
round starts after ``jobs_per_round`` jobs trained since the previous
round started, and only while fewer than ``refill_tasks`` stored train tasks
wait to train, so saved tasks train before more are generated. A task waits
until a committed job trains it; a task in a running job does not wait. A
round that made no train task and a failed task job are followed at once,
because no job end comes from them. So is a state in which every train task
has a result. The ``rounds``
limit counts the rounds that finished a generation; a round of replayed
sessions does not count. A round whose generation ran out of attempts counts
too, and the next round retries the same sessions and finishes the same
generation from its saved candidates.

The record-to-sample stage keeps its tasks in the task store ``generated-tasks``
of the scenario session (see `Where the stages live`_). It records each
publication or rejection, so a retry of a generation uses the saved candidates
and does not generate the tasks again. Tasks that share a source record get the
same split, and a published task keeps its split. A task whose sources join
tasks of both splits is rejected. The store also keeps each train task's first
rollout result and the generations whose results the generator received. The
sample store counts the tasks that trained.

After a restart, the record-to-sample stage continues with the next generation number.
The sample store still holds the train tasks that did not train, so the buffer takes them again. The stage sends the results that the generator did not receive, even when no other job is due. A task that trained before its result was saved gets an infrastructure-failure result, so its generation's feedback completes.
A replayed session that a complete generation already used makes no new generation.
``GET /reef/status`` shows the generator, the generation in flight, the completed generations and the last error under ``task_generation``, task counts under ``task_store``, and the stored tasks that wait under ``ready_samples``.

Container operations run in the generator service, ``reef.record2sample.generator``.
The record-to-sample stage calls this service over HTTP. The task store is the
source of each task; the service's tasks root holds staged copies. Staging the
same task again under its name returns the staged copy, so rollouts that a stop
interrupted, or a service that restarted with an empty tasks root, can still
run a saved task. The ``generator`` deployment
section configures the service that ``reef serve`` starts.
See `generator configuration
<../reference/configuration.rst#the-generator-section>`__ for its settings.

A record's path to a job
------------------------

.. code:: text

   report -> record2sample: expand each reference to its trajectory, filter -> record set
       -> make_sample -> Sample -> group -> record set unit
       -> sample store -> buffer -> sample2job: make_job_data -> job data -> acknowledge

   inference -> record2sample: trajectory idle for trajectory_idle_seconds -> trajectory set
       -> trajectory_sample -> TrajectorySample -> record set unit
       -> sample store -> buffer -> sample2job: make_job_data -> job data -> acknowledge

   computed record -> record2sample: observe/track -> dispatch -> async judge -> judged set
       -> make_sample -> Sample -> record set unit
       -> sample store -> buffer -> sample2job: make_job_data -> job data -> acknowledge

   uploaded trajectory -> sample store -> record2sample: uploaded_sample -> TrajectorySample
       -> buffer -> sample2job: make_job_data -> job data -> acknowledge

Where the stages live
---------------------

Each stage has its own package: ``reef/record2sample/`` and
``reef/sample2job/``. The pipeline lives in ``reef/train/pipeline.py``, and
the context that the stages read in ``reef/pipeline_context.py``. A stage that
saves generated tasks opens a store with
``context.task_storage.open(name)`` (see `Task stores
<../reference/python-api.rst#task-stores>`__). In a composite recipe,
each component opens its store names under its component name.
``task_storage`` is ``None`` if the trainer has no scenario session. The stage
packages also hold shared implementations: ``SampleAssembly``
(``reef/record2sample/assembly.py``), and ``DistillSamples``
(``reef/record2sample/distill.py``) with ``DistillJobData``
(``reef/sample2job/distill.py``) for the distilling recipes. ``DistillSamples``
emits the student's call with a teacher request in ``extra.reef.teacher``: the
request that the recipe composes from the recorded request and the report's
``teacher_context`` (its ``teacher_request`` override). The training component
renders it into the teacher sequence. A method's own stages live in
``recipes/<name>/processor.py`` (the record-to-sample and sample-to-job
classes), and ``recipes/<name>/recipe.py`` names them in ``pipeline_spec()``;
harness evolution's live in ``reef/record2sample/traces.py`` and
``reef/sample2job/traces.py``. Recipe-specific correlation and
model clients belong beside the method's stages.

Compatibility
-------------

A recipe's processor class becomes a ``Sample2Job`` with its ``make_job_data``,
named in the recipe's ``pipeline_spec()``; the trainer drives a
``TrainingPipeline`` instead of a ``DataProcessor``. Reported-feedback recipes
must replace ``judge`` and ``ReportDecision`` with ``make_sample`` returning
``Sample`` on a ``ReportSamples`` subclass. Remove ``WAIT``/``NEVER`` branches: invalid
references fail admission, and training contract failures must raise. Existing
stored records replay through the new contract; legacy reports with invalid
references or eligibility flags fail explicitly and need correction before replay.

The harness recipe no longer supports ``max_score`` or filters successful
reports. Remove that setting from configuration and Python construction.
Identical report retries still return their original receipt, including after
source eviction; changed content with the same id still conflicts.
