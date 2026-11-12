Training pipeline
=================

A scenario's training pipeline turns records into typed job data: the data
of one training job. It has two stages, with samples between them. A recipe
writes the stages; Reef owns the pipeline that drives them. This page
explains the stages, what a recipe writes for each, the job size, the training
modes, the task generation contract, and the path a record takes to a job.
Storage owns deletion.

:For: recipe authors writing the stages of a method, and anyone reading ``reef/train/pipeline.py``
:Before you start: the stage hooks from `Python API <../reference/python-api.rst#pipeline-stages>`__
:You will have: which stages a method needs, what it writes, and what it never writes

The two stages
--------------

.. code:: text

   record ─► record2sample ─► sample ─► sample2job ─► job data ─► trainer

- The **record-to-sample stage** (``Record2Sample``, ``reef.record2sample``)
  decides which records train, and in which record sets and units (see
  `Record sets and their units <#record-sets-and-their-units>`__). It turns
  each record set into samples. A sample is a ``Sample``: a
  ``TrajectorySample`` that trains directly, or a ``TaskSample`` (a Harbor
  task) that an algorithm rolls out. One job's data holds samples of one
  kind.
- The **sample-to-job stage** (``Sample2Job``, ``reef.sample2job``)
  shapes the samples of the next job into one ``JobData``. Grouped
  recipes set ``group_id`` on each member. The stage also sets
  ``job_samples``, the fewest samples of one automatic job, and ``buffer_size``,
  the most samples that one automatic job takes (see `Job size <#job-size>`__).

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
that went idle. The samples service runs when the scenario accepts records and
after each commit: a commit can end a job, and the job's end can make background
work due, such as the next round of generated tasks.

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

When a unit is ready, the pipeline saves each of the unit's samples
(``unit_samples``) as a data item of the scenario's owner that the scenario
uses (``TrainerSamples.write_record_samples``, RFC #1140). Then the scenario's
training state records the unit's records as consumed by the trainer. The stage
then ends the unit (``consume``) and forgets its samples. Sample ``i`` of record
set ``s`` is the item ``<scenario>.<trainer>:<s>:<i>``, or
``<scenario>.<s>:<i>`` for the only trainer of a scenario
(``stage_item_id``). Its metadata ``origin`` names the scenario, the trainer
and the sample's records. The samples of a unit with more than one sample share
the item metadata ``group``, so they train in the same job.

A sample that a sample processing task stores also has the metadata
``stored_by_task``, the ID of that task, and ``read_by_task``, the ID of the
task that read the last of its records (RFC #1267). The two differ when
background work, such as a judgment, ends in a later task. A key is left out
when the task is unknown, as for records read before a restart. A data query
with ``group_key=stored_by_task`` or ``group_key=read_by_task`` finds the
samples of one task.

A restart never makes saved samples again: recovery skips the records that the
trainer's stages consumed. Thus a nondeterministic stage, such as a judge, does
not run a second time on the same records. A process that stops between the two
writes makes the samples again; an item that exists stays as it is.

The pipeline
------------

``TrainingPipeline`` (``reef/train/pipeline.py``) is the only object that the
trainer gives records to and takes job data from. Recipes do not subclass it.
The trainer calls it on its own thread, under its lock; no call may block:

- ``push(record)``: one record arrives. The pipeline pushes it to the
  record-to-sample stage, which can form record sets and make their samples.
  Then the pipeline saves the samples of each unit that became ready as items
  (``store_samples``).
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
  With ``sample_count`` None, held samples never make the pipeline ready in
  ``auto``.
- ``build_job_data(sample_count)``: give the samples of the next job (see `The
  next job <#the-next-job>`__) to ``make_job_data``, in use order, and
  reserve them.
  The same job data stays reserved until it is acknowledged or released
  (``release_job_data``). A released sample can train in a later job.
- ``acknowledge(data_id)``: the commit that submits the job consumed that job
  data. The pipeline returns ``sample:<sample id>`` for each sample of the
  data, and the request's ID in ``manual``. The commit lists these IDs.
  In the same transaction, the scenario's training state adds one to the train
  count of each listed sample for the commit's trainer. A job that later fails does
  not give its samples back: they never train again for that trainer.

A record-to-sample stage can also make samples from no record set, such as
generated tasks. Two optional hooks serve such a stage; their defaults do
nothing:

- ``made_samples()``: return the samples that the stage made since the last
  call, as ``(name, sample)`` pairs. The pipeline stores each one as sample
  ``<name>`` (``<component>:<name>`` for a trainer of a scenario of several),
  with no group and no consumed record. A sample that the store holds already
  changes nothing, so a stage may return a sample again after a restart. A
  stored sample trains like every other sample.
- ``job_ended(outcome)``: a training job ended, and the commit of its end is
  durable. The samples service calls it, also when control runs in another
  process. ``outcome`` is a ``JobOutcome`` (``reef.core.job_data``): the job
  ID, the component name, ``event`` (``finished`` or ``failed``), the job's
  ``metrics``, and ``error``, which is set exactly when the job failed. A
  trainer that starts with no running job calls it again for the last job
  that ended, before it reads a record.

The generated-task recipe trains its saved tasks this way (see `Task
generation contract <#task-generation-contract>`__).

The pipeline also merges the stages' ``status()`` and metrics.
``background_jobs_running()`` is true while the record-to-sample stage runs a
background job, such as a judgment or a task-generation round. The scenario's
worker then polls at a fixed interval. ``next_due()`` is the monotonic time at
which a set that waits for time alone is due, such as an idle trajectory, or
None. The samples service then runs the next selection at that time, without a
new record. ``close()`` closes both stages.
``releasable_record_ids()`` returns
the completed records that no held set references, and
``release_records(ids)`` frees their memory once a commit names them. The
samples service calls both, and it reads the commits from the commit history. New
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

In ``auto``, the pipeline is ready when the next job holds at least
``job_samples`` samples. The next job takes whole groups, so one job's data can
hold more. Thus
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

The next job
~~~~~~~~~~~~

In ``auto``, the pipeline selects the samples of the next job from the items
that the scenario uses (``TrainingPipeline.auto_entries``):

1. It takes the samples that the trainer reads and did not train yet: the
   samples that the trainer made from records, and the uploaded trajectories
   (see `Uploaded trajectories <#uploaded-trajectories>`__). They are in use
   order, the order that their uses of the scenario began: a unit's samples when
   it became ready, and an upload when it joined the scenario. The samples of job data
   that the pipeline handed out are left out. Imported ``train`` tasks
   are candidates when the sample-to-job stage sets ``accepts_imported_tasks``;
   imported ``eval`` tasks never are, since an evaluation runs them.
2. It keeps the samples of one kind, trajectory or task: the kind of the first
   sample. A kind that the recipe gives no job parameters does not count; such
   samples train only for a request that gives them.
3. It leaves out each stale sample that the trainer made from records (see
   `Staleness <#staleness>`__), and every other sample of its group. An
   uploaded trajectory is never stale.
4. It takes the first ``buffer_size`` samples that remain, and the rest of
   each group that those samples touch.

Thus ``buffer_size`` is the most samples that one automatic job takes, beside
the rest of a group. The ``Sample2Job`` sets it, and refuses one smaller than
``job_samples``. Without a ``buffer_size`` setting, it is ``job_samples``. The
pipeline never releases or drops a sample: the other samples wait in the store
for later jobs, and a stale sample stays in the store untrained.

In ``manual``, the next job takes the samples that the oldest queued
request names (see `Training modes <#training-modes>`__).

``status()`` reports ``ready_samples`` (the samples of the next automatic job) and
``buffer_size``; the trainer adds
``job_samples``. A stage adds an entry only when a terminal outcome cannot become job data and an
external runner must stop waiting (TTTD reports a complete mixed-artifact
step as an invariant failure).

The job data directory
~~~~~~~~~~~~~~~~~~~~~~

A training component does not get the job data as an object. Before Reef
submits a job, it writes the job's data as a directory, and the component
reads the directory (RFC #1140). The jobs queue keeps only a reference to it.
The directory holds the samples, not the training request: Reef keeps the
request for its own accounting.

Format 2 of a directory:

.. code:: text

   job.json        {"format": 2, "key", "sample_count", "kind"}
   samples.jsonl   one sample per line, in job order

- ``job.json`` gives the directory's key, its number of samples, and the kind
  of its samples: ``trajectory``, ``task``, or ``null`` without samples.
- Each line of ``samples.jsonl`` is the entry of ``JobData.to_dict``, with one
  difference for a task. The line of a task has no ``task_path``. Its
  ``files`` names the task's directory in the task file store
  (``storage.task_file_backend``): ``partition`` and ``digest``, and ``name``,
  the name of the directory.

The directory holds no file of a task. The task file store keeps the one copy
of each task's files, so a job of many tasks writes two small files. Each
task of a job must therefore be in the task file store: ``TaskSample.files``
names it there, and Reef refuses job data with a task that has no ``files``.
The samples that Reef makes from imported and generated tasks have it.

A directory does not change after Reef writes it, and several jobs can read
it, so a component must not write in it. A component reads the data with
``job.read_data()``. Each task sample of the result has a local
``task_path``, which the data store took from the task file store when it
opened the job. The component must not change that directory.

A component that runs the job elsewhere, such as in a container, gives that
place ``job.data_uri`` and mounts or copies the directory there. That place
reads ``files`` from each task line and gets the directory from the task file
store of the deployment by ``partition`` and ``digest``.

The data store keeps the directories (``DataStore.write_job_directory``,
``open_job_directory`` and ``delete_unused_job_directories``) in its segment
store (``storage.segment_backend``), or in
``<agent_record_dir>/job-directories`` when none is set. The key of a directory is the SHA-256 of its lines (``job_data_key``): the
samples, and the digest of each task's files. So a job of the same samples, such as
a second job on one dataset, uses the directory that exists and writes
nothing. The data store records when a job last used each directory, and its
periodic work (``DataStore.maintain``, which the data service calls) deletes a
directory that no job used for ``storage.job_directory_ttl_s`` (7 days by
default). Keep this time above the longest job, since a component can read
the directory again when it resumes the job.

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
record. The pipeline reads each item that the scenario uses from the data
store, in the order that the uses began (``DataStore.scenario_items``). An item
that no stage of the scenario made is an upload: a sample of no component. When
the item leaves the scenario before a job takes it, the job leaves it out.

A recipe takes uploads when its ``Sample2Job`` sets ``accepts_trajectories``
to True. The default is False. Reef refuses with 400 an item that names a
scenario with no trainer that accepts uploaded trajectories.

The record-to-sample stage turns each upload into its training sample with
``Record2Sample.uploaded_sample(trajectory)``. The default returns the trajectory
as uploaded. A stage can override it to check the structure or add
metadata: ``SFTSamples`` refuses a trajectory with no agent turn to train and
adds its ``max_tokens``. The method must be pure. The pipeline calls it when it
reads the upload. A trajectory that it refuses with ``ValueError`` is skipped
with a warning and stays unconsumed. A data write does not call it: to add
items to a scenario, Reef checks only that a trainer accepts uploads, and it
does not read the trajectories. The training component renders
the trajectory when a job trains it; a trajectory that cannot render, or that
is longer than its ``max_tokens``, is left out of the job.

An uploaded trajectory is not a record, so the stage never forms a set of it.
It is a sample like the samples that the trainer makes from records: it waits
in use order, and it counts as one sample toward ``job_samples`` and
``buffer_size`` (see `The next job <#the-next-job>`__). In a scenario
of several components, each trainer that accepts uploaded trajectories trains
on them. The commit lists
each uploaded trajectory as ``sample:<trajectory_id>``, and the scenario's
training state adds one to the trajectory's train count for the commit's trainer
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
in ``releasable_record_ids()``. The samples service stores its ID, and the
next commit names it, so the record never trains after a restart. A restart
before that commit reads the record again, so the stage makes the same decision
again. The default bookkeeping keeps ended records until their release; a stage
with its own record lifecycle overrides the lifecycle methods.

Only the samples service feeds a stage. Control asks only ``is_stale`` and
``uploaded_sample``, so these methods must depend only on their arguments and
the stage's config.

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
2. When the scenario reserves job data, it gives the pipeline the artifacts
   that it served, oldest first. The commit history holds them, so a reload or
   a restart keeps them.
3. Each time the pipeline selects the next automatic job, it asks the stage
   whether each sample that the trainer made from records has data more than
   ``max_staleness`` artifacts behind the current artifact
   (``Record2Sample.is_stale(samples, served_artifact_ids)``). The job leaves out a stale sample and
   every other sample of its group. The sample stays in the store untrained.

The lag counts only the artifacts that this scenario served. A publication of
another scenario on a shared engine does not age the data. An ``artifact_id``
names the same weights in every process and after an engine restart. These
rules also apply:

* A sample whose record names no artifact is stale.
* A sample from an artifact that the stage did not see is stale. A rebuilt
  trainer knows only the artifacts that it sees after the rebuild.
* A scenario that has no runtime, or no component of a weight kind for the
  runtime to load, has no staleness check: no data is stale.
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
       continuing call, counted from the last call's ``created_at``
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

The component renders each trained turn as its call rendered it. When a
call's prompt does not extend the tokens of the call before it, as with a
template that drops earlier turns' reasoning, the trajectory trains as
several segments. Each segment is one sample with
``extra.reef.training.segment``: ``{"index": i, "count": n}``. The segments
keep the trajectory's group and reward. A recipe whose objective counts
trajectories, such as one that compares the samples of a group, must read
``segment``.

The component leaves out a sample that cannot render, that is longer than
its ``max_tokens``, or whose teacher sequence is longer than the request's
``max_tokens``. It also leaves out a sample with a teacher and several
segments. The job metric ``unrenderable_samples`` counts them. The
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
sample storage and acknowledgement use the same methods in both modes.
``GET /reef/status`` reports ``pending_requests``, the queued requests
that no commit consumed yet.

- ``auto`` makes job data when the next job holds ``job_samples`` samples.
- ``manual`` makes job data only for a queued TRAIN request. The
  request's data is the samples that it names (``TrainingRequest.samples``,
  ``TrainingPipeline.request_entries``), in use order, whatever
  ``job_samples``. A sample that trained before trains again, and staleness
  does not apply. When Reef accepts the request, a trainer must take its
  samples (``TrainingPipeline.check_request``): the store holds each one,
  the trainer can train each one, they are of one kind, and they are at most
  the stage's ``max_job_samples``. A sample that leaves the scenario before the
  job is left out of it. A ``Sample2Job`` that sets ``reads_requests`` also
  runs a request that names no sample; harness evolution sets it, because
  its proposer reads the request.

The pipeline runs the oldest request first, one per step. It gives the
request's data the id ``<scenario>:request:<request id>`` and attaches
the request to ``JobData.request``. Automatic job data gets the id
``<scenario>:data:<number>``. Acknowledgement also consumes the
request. ``make_job_data`` must not call models or perform training.

When the pipeline supports ``manual``, the scenario gives it the queued
training requests before each step (``TrainingPipeline.queue_requests``);
requests are not records. The pipeline keeps them FIFO regardless of the
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

The training state and queued requests stay on the same instance. Reef stores
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
  the flat tuple of the next job's samples into job data with the ID ``data_id``.
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
and retention. Job data is ready in automatic mode after the next job holds
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
``from_config(config, *, scenario, environ, state)`` once for each scenario. The
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

``state`` is the ``GeneratorState`` of the scenario's generator
(``reef.recipe.generated_tasks``). A generator keeps there what it must read
again after a restart, such as its open generations, and keeps no file of its
own. The state holds text documents by key, and saved directories:

- ``read(key)`` returns the document of ``key``, or ``None``.
- ``write(key, document)`` saves the document. It replaces an earlier document
  of the key.
- ``remove(key)`` removes the document. An unknown key changes nothing.
- ``keys(prefix="")`` returns the keys that start with ``prefix``, in
  increasing string order.
- ``save_files(directory)`` saves a copy of a directory and returns the digest
  of its files.
- ``files(digest)`` returns a local directory with the saved files. Do not
  change it.

Each write is durable and atomic when it returns. The record-to-sample stage
builds the state from its task store, so the documents are in the deployment's
database and the saved directories are in the task file store. A restart on
another machine reads the same state, and an archive of the scenario retires
it. The generator gets no other part of the task store: it cannot publish a
task or read the stage's own documents. For work that needs a directory for a
short time, such as a task under validation, use a temporary directory. A
config that names a generator is refused when the scenario has no task
storage.

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
   task with the generator, ``validation_concurrency`` tasks at once.
3. Publishes each valid task with its train/eval split. It runs no eval
   tasks.

A generator or check that fails to run is retried from the saved candidates,
up to ``generation_attempts`` times within ``generation_deadline_seconds``.
After that the round ends without a task, and the next round takes the same
sessions again, so it finishes the same generation. A task identical to one
that was already published is rejected as a duplicate. The record-to-sample
stage hands each train task of a complete generation to the pipeline
(``made_samples``), which saves it as the task item
``<scenario>.generated-tasks.<task name>`` that the scenario uses.

Stored tasks train as task jobs, like every other sample.
Reef runs no rollout of them: the training component rolls each job's tasks
out on its own sampler, at the job's start weights, and trains on their
rollouts.

1. ``GeneratedTaskJobData`` makes one job's data of the next job's tasks: the
   ``TaskSample`` values. A job takes at least ``job_samples`` tasks, 1 by
   default, and at most ``buffer_size``, 4 by default.
2. The trainer submits the job. Its rollouts parameter (``rollout`` on Tinker, ``harbor`` on Slime, in ``recipe.config.job_parameters``)
   tells the component with which agent and sandbox to run the rollouts of
   each task. Their count is ``rollout.rollouts_per_task`` on Tinker and Slime's
   ``n-samples-per-prompt`` on Slime. The component assembles one sample from each
   rollout that ran, scored with the task's verifier reward. Tinker trains on the
   groups of at least two samples, and Slime on the samples that its own loop
   generates. It reports each task's rollout rewards in the job's metrics
   (``rollout_rewards``).
3. The commit of the submission counts the tasks trained in the sample
   store, as it does for every sample.
4. When the commit of the job's end is durable, the samples service reads it
   and gives the job's outcome to the record-to-sample stage
   (``Record2Sample.job_ended``). The
   stage gives each task that a committed job trained and that has no result
   its rewards from ``rollout_rewards``. A task without rewards, for example
   in a job that failed or that a ``train-from`` cancelled, gets an
   infrastructure-failure result, so its generation's feedback still goes out.

A result that the task store fails to save is saved again by the next job. On
a restart, the scenario still uses the task items that did not train, and a
later job takes them again. A job that runs across a restart saves its
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
The deployment names the objective that trains the samples, as a dotted reference
in ``recipe.config.job_parameters.objective``. The stage trains on no
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
wait to train, so saved tasks train before more are generated. With
``jobs_per_round`` 0, the next round starts when the previous one ends, so
generation runs ahead of training until ``refill_tasks`` tasks wait. A task waits
until a committed job trains it; a task in a running job does not wait. A
round that made no train task and a failed task job are followed at once,
because no job end comes from them. So is a state in which every train task
has a result. The ``rounds``
limit counts the rounds that finished a generation; a round of replayed
sessions does not count. A round whose generation ran out of attempts counts
too, and the next round retries the same sessions and finishes the same
generation from its saved candidates.

With ``pool_capacity`` above 0, the stage keeps a training pool, and each train
task trains again until a review retires it. A round is then due while fewer
train tasks of complete generations than ``pool_capacity`` are active, in place
of the ``refill_tasks`` rule. When a job's end commits, the stage saves each use
that the end ends, with its rewards, and runs the review that is due after every
``review_interval`` jobs. Then it makes the sample of each active task's next
use, ``generated-tasks/<task name>@<k>`` for use ``k``, which waits after the
samples that wait already. A review that retires a task ends the scenario's use
of the task's waiting sample (``TrainerSamples.delete_samples``), so no later
job takes it; the item stays in the data store with its train counts. See
`generated tasks <../reference/configuration.rst#generated-tasks>`__ for the
keys.

The record-to-sample stage keeps its tasks in the task store ``generated-tasks``
of the scenario session (see `Where the stages live`_). It records each
publication or rejection, so a retry of a generation uses the saved candidates
and does not generate the tasks again. Tasks that share a source record get the
same split, and a published task keeps its split. A task whose sources join
tasks of both splits is rejected. The store also keeps each train task's first
rollout result and the generations whose results the generator received, and,
as state documents, the generator's own state. The
scenario's training state counts the tasks that trained.

After a restart, the record-to-sample stage continues with the next generation number.
The scenario still uses the task items that did not train, so a later job takes them again. The stage sends the results that the generator did not receive, even when no other job is due. A task that trained before its result was saved gets an infrastructure-failure result, so its generation's feedback completes.
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
       -> data item -> next job -> sample2job: make_job_data -> job data -> acknowledge

   inference -> record2sample: trajectory idle for trajectory_idle_seconds -> trajectory set
       -> trajectory_sample -> TrajectorySample -> record set unit
       -> data item -> next job -> sample2job: make_job_data -> job data -> acknowledge

   computed record -> record2sample: observe/track -> dispatch -> async judge -> judged set
       -> make_sample -> Sample -> record set unit
       -> data item -> next job -> sample2job: make_job_data -> job data -> acknowledge

   uploaded trajectory -> data item -> record2sample: uploaded_sample -> TrajectorySample
       -> next job -> sample2job: make_job_data -> job data -> acknowledge

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
