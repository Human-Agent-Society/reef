Training pipeline
=================

A scenario's training pipeline turns records into typed job data: the data
of one training job. It has three stages, with one data type between each
pair. A recipe writes the stages; Reef owns the pipeline that drives them.
This page explains the stages, what a recipe writes for each, the job size,
the training modes, the task generation contract, and the path a record takes
to a job. Storage owns deletion.

.. page::
   :for: recipe authors writing the stages of a method, and anyone reading ``reef/train/pipeline.py``
   :needs: the stage hooks from `Python API <../reference/python-api.rst#pipeline-stages>`__
   :outcome: which stages a method needs, what it writes, and what it never writes

The three stages
----------------

.. code:: text

   record ─► record2record ─► record set ─► record2task ─► task ─► task2job ─► job data ─► trainer

- The **record-to-record stage** (``Record2Record``, ``reef.record2record``)
  decides which records train, in which record sets and units, and in
  what order (see `Record sets and their units <#record-sets-and-their-units>`__).
- The **record-to-task stage** (``Record2Task``, ``reef.record2task``) turns
  one record set into tasks. A task is a ``TrainDataItem``: a ``TrajectoryItem``
  that trains directly, or a ``TaskItem`` (a Harbor task) that an algorithm
  rolls out. Job data may mix both.
- The **task-to-job stage** (``Task2Job``, ``reef.task2job``)
  shapes the tasks of the selected record set units into one ``JobData``. Grouped
  recipes set ``group_id`` on each member. The stage also sets
  ``job_samples``, the fewest samples of one job, and ``buffer_size``, the
  most samples that wait for a job (see `Job size <#job-size>`__).

The pipeline moves values through the stages with the same two operations,
``push`` and ``pull``. The last stage is a function:

.. list-table::
   :header-rows: 1

   * - Stage
     - In
     - Out
   * - ``Record2Record``
     - ``push(record)``
     - ``pull() -> tuple[RecordSet, ...]``
   * - ``Record2Task``
     - ``push(record_set)``
     - ``pull() -> tuple[(set_id, TaskResult), ...]``
   * - ``Task2Job``
     - ``make_job_data(tasks, data_id)``
     - ``JobData``

When the pipeline looks for job data, it first calls the record stage's
``advance(now)``. The stage then takes in background work that ended, such as
judgments, and starts work that is due, such as a task-generation round.

The recipe names a default class for each stage in its ``pipeline_spec()``.
The deployment's ``pipeline`` section gives each stage its config and can name
another class (see `the pipeline section
<../reference/configuration.rst#the-pipeline-section>`__). For each scenario,
Reef builds the stages in pipeline order with the class method ``from_config``
and composes them into a ``TrainingPipeline``:

- ``Record2Record.from_config(config, *, context, environ)``
- ``Record2Task.from_config(config, *, context, record2record, environ)``
- ``Task2Job.from_config(config, *, context, record2record, record2task, environ)``

A later stage receives the stages built before it. It reads a value that an
earlier stage owns from that stage, so the value is configured once: TTTD's
task-to-job stage reads the step grid from its record stage, and
OpenClaw-RL's reads the record stage's judging. ``config`` is the stage's own config. A stage
declares its keys as a frozen settings dataclass that inherits
``reef.core.plugins.PluginSettings`` (fields made with
``reef.core.config.config_option``). Its ``from_config`` parses the config
with ``Settings.parse(config, plugin_class=cls)`` and gives the settings
object to the constructor; the stage never reads a raw mapping. A stage
without config parses with ``NoSettings``, which refuses any key. A record
stage that holds units parses ``Record2RecordSettings`` or a settings class
that extends it, and gives its ``max_staleness`` to ``Record2Record.__init__``
(see `Staleness <#staleness>`__). Reading the
deployment checks the stage classes; a bad config fails when the first
scenario builds its pipeline. Without a record stage, the pipeline forms no
record set and makes no tasks.

Nothing numeric lives in the stages. Advantages and the loss family are the
step objective's. See `job data values <../reference/python-api.rst#job-data>`__ for
formats and algorithm support.

Record sets and their units
---------------------------

The record stage forms two kinds of value. Each kind has an exact definition.

A record set (``RecordSet``) is the unit that the record-to-task stage makes
tasks from. A value is a record set if and only if it holds one or more
records, and it holds each record that its records reference. A report and the inference records that it
references form one record set. One judged record, or one inference record,
also forms one. No record is in two record sets of one stage. Thus a report
that references an inference of another record set ends without training.

The record-to-task stage returns one ``TaskResult`` for each record set that
the pipeline pushes. Its status settles all records of the set together:

- ``READY``: one or more tasks. The records wait in one record set unit.
- ``SKIPPED``: no task. The stage ends the records.

A unit is what one job's data takes all of or none of. A unit is a record set
unit or one uploaded trajectory (see `Uploaded trajectories
<#uploaded-trajectories>`__). The pipeline counts units in samples: a record
set unit counts the tasks of its record sets, and an uploaded trajectory
counts one.

A record set unit (``RecordSetUnit``) holds one or more record sets that must
train in the same job. A report by itself is a unit of one record set. A
ready group is a unit of the record sets of all its reports. The record stage
holds these units in its pool. Consuming a unit writes the IDs of all its
records to the commit.

The pipeline
------------

``TrainingPipeline`` (``reef/train/pipeline.py``) is the only object that the
trainer gives records to and takes job data from. Recipes do not subclass it.
The trainer calls it on its own thread, under its lock; no call may block:

- ``push(record)``: one record arrives. A TRAIN instruction stays in the
  pipeline. The pipeline pushes every other record to the record stage. It
  then moves the record sets: it pulls the sets that the record stage
  formed, pushes each one to the record-to-task stage, pulls the results
  that are ready, and gives each result back to the record stage
  (``settle``). A set without a result waits, and its records stay held (see
  `Slow record-to-task work <#slow-record-to-task-work>`__).
- ``advance()``: advance the record stage's background work, move and settle
  record sets, and apply buffer and staleness limits. It needs no sample
  count and reserves no job data. Pending job data keeps its reservation while
  new results enter the buffer. Call it on the same serialized path as
  other pipeline operations; it does not make the pipeline thread-safe.
- ``ready(sample_count)``: call ``advance()`` and check whether job data of
  ``sample_count`` samples is available. The trainer passes ``job_samples``.
  With ``sample_count`` None, held samples never make the pipeline ready.
- ``build_job_data(sample_count)``: reserve whole record set units from the record
  stage, in its selection order, until the data holds at least
  ``sample_count`` samples,
  and give their tasks to ``make_job_data``. Uploaded trajectories come first
  (see `Uploaded trajectories <#uploaded-trajectories>`__). The trainer
  validates the job data against ``output_schema``. The same job data stays
  reserved until it is acknowledged or released.
- ``acknowledge(data_id)``: the commit that submits the job consumed that job
  data. The pipeline consumes the reservation from the record stage's pool,
  and the stage ends the units and returns the IDs of the records that they
  consumed. The commit record persists these IDs, so recovery never pushes
  them again. A job that later fails does not give its records back: they
  never train again.

A task-to-job stage can also make samples from no record and train them as
job data of its own. The pipeline drives this through the stage's optional
hooks, whose defaults make no such samples:

- ``advance()``: start or take in the stage's own work. The pipeline calls it
  in each ``advance()``.
- ``own_job_data()``: return the samples that train now, or ``None``. While it
  returns samples, the pipeline is ready and its next job data holds exactly
  these samples, whatever their count. The pipeline asks before it selects
  record sets.
- ``own_job_data_submitted()``: the commit that submits a job of the stage's
  own job data is being prepared. Return the IDs that the commit lists for it.
  The pipeline gives them to ``release_own_ids(ids)`` once the commit is
  durable.
- ``job_ended(outcome)``: a training job ended, and the commit of its end is
  durable. ``outcome`` is a ``JobOutcome`` (``reef.core.job_data``): the job
  ID, the component name, ``event`` (``finished`` or ``failed``), the job's
  ``metrics``, and ``error``, which is set exactly when the job failed. A
  trainer that starts with no running job calls it again for the last job
  that ended, before any other call to a new stage.
- ``background_jobs_running``, ``status``, and ``close``: as on the record
  stage.

A stage that trains only its own job data sets the class attribute
``trains_own_job_data_only``. It takes no ``buffer_size`` and no
``job_samples``, and held samples never make its pipeline ready.

Released own job data stays the stage's, and the next ``build_job_data`` builds
it again. The generated-task recipe rolls its saved tasks out this way (see
`Task generation contract <#task-generation-contract>`__).

Job size
--------

Reef submits one training job for each job's data. ``job_samples`` is the
fewest samples that one job trains, a positive int. The deployment sets it in
``pipeline.task2job.config.job_samples``. The training component then cuts the
job's data into optimizer batches by its own batch size: Slime's
``--global-batch-size`` or Tinker's ``batch_size``.

- A weight recipe has no default ``job_samples``. A deployment without it
  fails the recipe build with ``RecipeConfigError``. Usually, set it to the
  component's batch size, so that one job is one optimizer step.
- Harness evolution uses ``evolution.batch_size`` when the stage config sets
  no ``job_samples``.
- A stage whose job data is one fixed unit, such as the TTTD and SDPO grid,
  passes ``max_job_samples`` to ``Task2Job.__init__``. The stage refuses a
  larger ``job_samples`` with ``PluginConfigError``.

The pipeline is ready when it holds at least ``job_samples`` samples. It takes
whole units, so one job's data can hold more. Thus ``job_samples`` and the
component's batch size can differ:

- A unit packs several samples, for example a group of rollouts. The job's
  data then holds more samples than ``job_samples``, and the component trains
  more than one optimizer step.
- A job's data that is smaller than the component's batch size trains as one
  partial step.

The ``remainder`` field of the job's optional ``scheduling`` parameter
decides what happens to a shorter last step. With ``packing_length`` in the
same parameter, a step trains batch-size packs of tokens, not batch-size
samples. ``job_samples`` still counts samples, so the job's packs seldom fill
whole steps. Set ``job_samples`` to the samples of many steps.

The held units are a buffer of at most ``buffer_size`` samples. The
``Task2Job`` sets it, and refuses one smaller than ``job_samples``. Without a ``buffer_size`` setting, it holds the data of four
jobs: four times ``job_samples``. After each move of
record sets, in every training mode, the pipeline releases the oldest
unreserved units while their samples are more than ``buffer_size``. It
keeps the newest unit, so a unit larger than the buffer still trains. The
record stage ends a released unit (``drop_held_unit``), and the pipeline logs
one warning. Uploaded trajectories do not count toward the buffer.

The pipeline also merges the record and task stages' ``status()`` and metrics.
``background_jobs_running()`` is true while a set waits for its result or the
record stage runs a background job, such as a judgment or a task-generation round. The trainer then polls ``ready()`` at a fixed
interval. ``close()`` closes the record and task stages.
``releasable_record_ids()`` returns
the completed records that no held or waiting set references, and
``release_records(ids)`` frees their memory after a successful commit. New
commits retain the stored bodies; consumption progress prevents retraining
on restart. Storage can independently evict any body under capacity pressure,
with warnings and durable loss totals. In a scenario with one trainer per
component, every trainer reads the same rows and keeps its own consumption,
so a row one pipeline releases is still stored for the others. When the
scenario admits a wider report contract than a recipe's own,
``admit_reports_of(report_type)`` names it, and the record stage releases a
report shaped for another component instead of failing it. The pipeline
passes such changes to the record stage as its updated ``PipelineContext``
(``context_changed``).

``status()`` reports ``ready_samples`` (the samples of the held record set
units and uploaded trajectories) and ``buffer_size``; the trainer adds
``job_samples``. A stage adds an entry only when a terminal outcome cannot become job data and an
external runner must stop waiting (TTTD reports a complete mixed-artifact
step as an invariant failure).

Slow record-to-task work
------------------------

Most record-to-task stages make the tasks of a set when the pipeline pushes
it. They inherit ``ImmediateRecord2Task`` and implement ``make_tasks(record_set)``.

A stage that needs slow work must not block the trainer. Its ``push`` submits
the work to a worker that the stage owns, and its ``pull`` returns the result
when the work ends. Until then the set waits, and the pipeline holds its
records. The stage returns one result for each pushed set, one time. A result
for a set that does not wait makes ``ready()`` raise ``ValueError``. The
stage's ``close`` stops its worker.

Uploaded trajectories
---------------------

A client can upload complete ATIF trajectories to ``POST /reef/trajectories``
(see the `HTTP API <../reference/http-api.rst#upload-trajectories>`__). An
uploaded trajectory is not a record. Reef keeps it in the scenario's trajectory
store, and the trainer gives it to the pipeline with ``push_trajectory``.

A recipe takes uploads when its ``Task2Job`` sets ``accepts_trajectories``
to True. The default is False. Admission refuses an upload with 400 when no
trainer of the scenario accepts uploaded trajectories.

The record-to-task stage turns each upload into its training sample with
``Record2Task.uploaded_sample(trajectory)``. The default returns the trajectory
as uploaded. A stage whose backend trains token-level data overrides it:
``SFTTasks`` renders the conversation with the chat template and trains every
agent turn that is not marked ``is_copied_context``. The method must be pure. Admission calls it from the request
thread for each trajectory and each trainer that accepts uploads; a
``ValueError`` refuses the whole upload with 400, and the message names the
trajectory. The pipeline calls it again when it reads the upload. A
trajectory that admission accepted but that no longer renders, for example
after ``max_tokens`` was lowered, is skipped with a warning and stays
unconsumed.

An uploaded trajectory does not go to the record stage. The pipeline holds
its sample until job data takes it. Job data takes uploaded trajectories first, in upload
order, and then record set units. Each uploaded trajectory is one unit
and counts as one sample. It does not count toward ``buffer_size``. In ``manual`` mode, an instruction's data takes no units. In a scenario
of several components, each trainer that accepts uploaded trajectories trains
on them. Acknowledgement consumes all units of the job data. The commit lists
each uploaded trajectory as ``trajectory:<trajectory_id>``. Thus a rebuilt
trainer does not read it again.

An uploaded trajectory has no captured rollout tensors or rollout log
probabilities. Thus an objective that needs them, such as a policy-gradient
objective, cannot train on it; an SFT sample needs neither.

The record-to-record stage
--------------------------

1. The stage reads each record (``push``) and forms record sets (``form``).
   The pipeline pulls them (``pull``).
2. The record-to-task stage makes tasks from each record set.
3. The stage holds the record sets that train in record set units in its
   ``pool``, places reports in their groups, and drops units that can no
   longer train. The selection policy orders the pool.
4. ``build_job_data`` reserves units from the pool. ``acknowledge`` consumes the
   reservation, and the stage ends the consumed units (``consume``).

A record that the stage ends, for example a report in a taken slot or a unit
that is too stale, appears in ``releasable_record_ids()``. The next commit
names it, so the record never trains after a restart.

Before exposing a record set to the task stage, ``Record2Record.form`` checks
its inference responses for an engine abort. An abort rejects the entire set,
including a report and all its referenced inferences. The stage uses its normal
rejection and release lifecycle; it does not alter the stored responses or
delete captured tensors. Retained records pass the same check during replay.
A missing ``training`` block alone does not reject a record: some methods use
records to generate tasks without captured policy tensors.

Staleness
~~~~~~~~~

Every record stage runs the staleness check. No other part of Reef checks the
version of training data. The limit is the record stage's setting
``max_staleness`` (``Record2RecordSettings``), set in
``pipeline.record2record.config``:

.. code:: yaml

   pipeline:
     record2record:
       config:
         max_staleness: 2

``max_staleness`` is a non-negative int, and its default is 0. With 0, only
data of the version that the scenario serves trains. A negative value fails
the pipeline build. A stage settings class with more keys extends
``Record2RecordSettings``, so each stage takes the key next to its own keys.
The check has these steps:

1. When the scenario reserves job data, it gives the pipeline the runtime load
   ID that serves its committed release (``observe_serving_runtime_load_id``).
   The stage keeps these versions, oldest first.
2. Before each job's data, the stage ends every held unit that has data more than
   ``max_staleness`` versions behind the current version (``drop_stale_units``).

The lag counts only the versions of this scenario. A publication of another
scenario on a shared engine does not age the data. These rules also apply:

* A sample without a recorded version is stale.
* A sample from a version that the stage did not see is stale. A rebuilt
  trainer knows only the versions that it sees after the rebuild.
* While the serving version is unknown, for example before the first
  publication, no data is stale.

Reef's check is the only staleness filter that Reef runs. A training
framework can have its own off-policy limit. Such a limit goes in that
backend's ``training.options``, not in the record stage. No built-in backend
has one now.

Two stages do not take ``max_staleness``. ``NoRecordSets`` holds no unit.
``SFTRecord2Record`` never drops a unit as stale, because an SFT target is
fixed text, not sampled tokens. Both refuse the key.

A stage that holds groups overrides ``drop_held_unit(unit, *, reason)`` to
end a whole group. ``ReportRecord2Record`` does this for a ready group.
The caller supplies ``UnitDropReason.BUFFER_LIMIT`` for buffer eviction or
``UnitDropReason.STALE`` for the serving-version check. Overrides must accept
the reason; it lets a stage distinguish the two outcomes without inspecting
already-released records. Neither outcome confirms training.

.. list-table::
   :header-rows: 1

   * - Stage
     - Decides
   * - ``NoRecordSets``
     - nothing; it forms no record set. The default of a pipeline that trains no weights
   * - ``ReportRecord2Record``
     - report validation, groups and slots, source release
   * - ``ExclusiveReportRecord2Record``
     - a ``ReportRecord2Record`` that trains every report by itself; a report that ends owns its inference records
   * - ``JudgingRecord2Record``
     - correlation and the judging worker
   * - ``InferenceRecord2Record``
     - one set per inference record; other records are released

Training modes
--------------

``training_mode`` is an attribute of each pipeline. The recipe passes its
initial value through ``Trainer.build`` and ``PipelineContext``; it defaults
to ``auto``. A ``Task2Job`` declares the modes it supports in
``supported_training_modes``; the default is ``auto`` only. Record push,
acknowledgement and buffer release use the same methods and buffers in every
mode. ``GET /reef/status`` reports ``pending_instructions`` as the
pipeline's ``buffered_requests`` plus the instructions still unread in
storage.

- ``auto`` makes job data when the held units and uploaded trajectories hold
  ``job_samples`` samples.
- ``manual`` makes job data for a queued TRAIN instruction. The instruction's
  data has no units, so ``make_job_data`` receives no tasks.
- ``hybrid`` makes job data for either, and a queued instruction goes first.
  The instruction's data takes the units that automatic job data would.

The pipeline runs the oldest instruction first, one per step. It gives the
instruction's data the id ``<scenario>:instruction:<request id>`` and attaches
the instruction to ``JobData.request``. Automatic job data gets the id
``<scenario>:data:<number>``. Acknowledgement also consumes the
instruction. ``make_job_data`` must not call models or perform training.

The trainer gives TRAIN records to the pipeline when ``RequestType.TRAIN`` is
in the ``Task2Job``'s ``required_request_types``. The pipeline queues them
FIFO regardless of the selected mode; ``auto`` leaves the queue for a mode that
takes it. The pipeline takes records normally in manual mode, so changing to auto
or hybrid can train on data already collected. In manual mode, as in the other
modes, the pipeline releases the oldest held units beyond ``buffer_size``.

Unsupported modes raise ``NotImplementedError``. An invalid mode name raises
``ValueError``.

Changing training mode
----------------------

``POST /reef/scenarios/{scenario}/update`` selects ``auto``, ``manual`` or
``hybrid`` on the existing pipeline. The trainer serializes selection with
record push and reservation. Reserved job data stays unchanged and acknowledgement
consumes its actual contents, independently of subsequent mode changes.

Buffered data and queued instructions stay on the same instance. The selector
is runtime state: a scenario reload after a failed step keeps the selected
mode, and a service restart starts from the recipe's configured mode.

The two feedback paths
----------------------

Reported feedback arrives in reports. Computed feedback is derived from the
traffic, potentially with slow model calls on a worker. Reported feedback has
no ``judge`` hook; computed feedback keeps its asynchronous ``judge``.

Reported feedback
~~~~~~~~~~~~~~~~~

Before accepting a new report, Reef checks its schema and verifies that every
reference identifies an existing inference in the same scenario. Missing,
wrong-kind, duplicate, or foreign references raise an error. Reports are not
queued waiting for future inference records. Storage is authoritative; a record
need not have reached the pipeline's in-memory cache at admission time.

Reports cannot opt out of training: ``metadata.training.eligible`` is rejected,
including when its value is ``true``. Valid low or negative scores are feedback,
not a reason to discard the report.

A reported-feedback recipe writes:

- a ``ReportRecord2Record`` subclass as its record stage, with its
  ``ReportGrouping`` (below);
- ``make_sample(context) -> TrainDataItem`` on a ``ReportTasks`` subclass,
  its record-to-task stage: assemble an ATIF trajectory or Harbor task
  directly. Use ``context.require_score()`` when the method needs a reward.
  The stage attaches the source and report ids to the returned item.
- ``make_job_data(tasks, data_id)`` on a ``Task2Job`` subclass: assemble
  the flat tuple of selected tasks into job data with the ID ``data_id``.
  The pipeline tracks consumption by record set unit, including tasks that the
  recipe removes from training.

A report that ends after it was a record set owns its referenced inference
records outright. ``exclusive_sources`` gives the same ownership to a report
that ends before, and ``ordered_groups`` selects ready groups in key order.
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
cannot train, as Distill does for a teacher sequence that is too long. The
stage releases the report.

The stage handles report-id deduplication, group-slot retries, consumption,
and retention. Job data is ready in automatic mode after the held units hold
``job_samples`` samples; a unit is one report or one complete group. A report that
references an inference of another report ends without training, whether the
other report waits or already trained.

Training data errors propagate: missing required tokens/logprobs are checked by
the training backend, and unsupported trajectory assembly raises an error rather
than dropping the report. Failed sample assembly keeps its input records
protected and does not mark the report successfully processed.

Computed feedback
~~~~~~~~~~~~~~~~~

A computed-feedback recipe writes a ``RecordJudging``: correlation in
``observe(record, now, stage)`` with the stage's ``dispatch``, ``track``,
``tracked_record``, and ``retire`` operations, ``async judge``, and ``expire``
for tracked records that time out. The recipe's record stage is a
``JudgingRecord2Record`` with the judging and the worker that runs
``judge``. Its record-to-task stage is a ``JudgedTasks`` with
``make_sample(record, judgment)``, and its ``Task2Job`` supplies
``make_job_data``. The stage holds judged records until job data uses them.

Task generation contract
------------------------

A deployment can train on tasks that a task generator makes. The deployment
selects the built-in generated-task recipe,
``reef.recipe.generated_tasks:GeneratedTaskRecipe``, configures its stages in
the ``pipeline`` section, and names a ``TaskGenerator`` class in
``pipeline.record2task.config.generator`` (see `generated tasks
<../reference/configuration.rst#generated-tasks>`__). The recipe does not
depend on the generator's method, so a team can replace the generator without
a new recipe. SPADE (``recipes/beta/spade``) is one generator.

``TaskGenerator`` is in ``reef.recipe.generated_tasks``, and
``TaskGenerationRequest`` in ``reef.record2task.generator``. Reef calls its class method
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
- ``HeldOutTask(task)``: the task is in the eval split. It runs no episodes. A
  generator must not use it as training feedback or as experience for later
  generations.
- ``RolledOutTask(task, rollout_rewards)``: the first rollout of the train task
  ran. ``rollout_rewards`` holds one verifier reward for each training episode
  that ran, as the rollout's policy scored it before it trained on it. An
  episode that ran without a reward counts as 0.
- ``FailedTask(task, error, infrastructure_failure)``: the first rollout ran no
  episode. ``infrastructure_failure`` is true when the task could not run at
  all, for example because the sandbox failed; it is not a failure of the
  solver. It is false when the generator service refused the task.

The generator service scores an episode whose verifier ran past its timeout
as 0, and reports it, so a training episode of that kind trains. The task's
oracle passed the same verifier, so the timeout is the agent's.

The recipe splits the work across its stages. A round is not a record set,
and it does not train. The record stage (``GeneratedTaskRecord2Record``)
decides when a round is due, holds the source sessions, and
gives the round number and its sessions to the record-to-task stage
(``GeneratedTaskTasks``). That stage runs the round's generation on its own
``BackgroundJobWorker``, outside the trainer lock, and tells the record stage
when the round ends. One generation runs at a time.

A generation job runs no episodes. It:

1. Asks the generator for tasks with the configured ``description`` and the
   round's source sessions.
2. Saves the candidates in the scenario's task store, then validates each
   task with the generator.
3. Publishes each valid task with its train/eval split. It runs no eval
   tasks.

A generator or check that fails to run is retried from the saved candidates,
up to ``generation_attempts`` times within ``generation_deadline_seconds``.
After that the round ends without a task, and the next round takes the same
sessions again, so it finishes the same generation. A round that made train
tasks puts them in the task pool. A task identical to one that
was already published is rejected as a duplicate.

Saved tasks train as task jobs. Reef runs no episode of them: the training
component rolls each job's tasks out on its own sampler, at the job's start
weights, and trains on their episodes. ``GeneratedTaskJobData`` drives the
loop through the stage's own-job-data hooks:

1. When no tasks are handed out, the stage takes up to ``tasks_per_step``
   tasks from its task pool as the next job's data: the ``TaskItem`` values.
2. The trainer submits the job. Its ``rollout`` parameter (``recipe.config.job_parameters.rollout``)
   tells the component how many episodes of each task to run, and with which
   agent and sandbox. The Tinker component runs them, assembles one sample
   from each episode that ran, scored with the task's verifier reward, and
   trains on the groups of at least two samples. It reports each task's
   episode rewards in the job's metrics (``rollout_rewards``).
3. The commit of the submission consumes the tasks: each is marked trained
   in the task store once that commit is durable. The tasks stay in the pool
   until the job ends.
4. When the commit of the job's end is durable, the trainer gives the job's
   outcome to the stage (``Task2Job.job_ended``). The tasks leave the pool.
   The stage gives each trained task without a result its rewards from
   ``rollout_rewards``. A task without rewards, for example in a job that
   failed or that a rollback cancelled, gets an infrastructure-failure
   result, so its generation's feedback still goes out.

A result that the task store fails to save is saved again by the next job. On
a restart, the train tasks that did not train go back to the task pool. A job
that runs across a restart saves its tasks' results when it ends. A trainer
that starts with no running job gives the stage the last job's outcome again,
so a restart just after a job's end loses no result. The loop is the
recipe's (``reef.recipe.generated_tasks.rollout``); the pipeline only reads
the stage's own job data. Its ``job_samples`` is ``tasks_per_step``, so the
deployment does not set it.

When every train task of a generation has a result, the next job sends the
generation's results to ``record_results``. A generation can therefore close
after later generations started. A generation whose delivery fails waits for
the next job, and later generations still go out.

A generator can run episodes of a task through the generator service, as
SPADE does for its hint plays; those episodes are not records, and do not
train.
The ``task_group`` objective centers and scales each reward within its task
group, on Tinker's ``importance_sampling`` loss. The record stage trains on no
report: it holds the reports that are source sessions, and releases the others,
such as a generator's own reports, without assembling them.

A round can also generate tasks from recorded sessions. Set the record
stage's ``source_records_per_round`` above 0 and its ``source_tag`` to a
``name=value`` tag. A report without a training arm, whose inference records all carry the
header ``x-reef-tag-<name>: <value>``, is a source session. The record stage
holds source sessions in arrival order. Each round takes at most
``source_records_per_round`` of them, and a round without a waiting session
has none. The generator receives each session's inference records and then
its report as ``source_records``. The record stage keeps a session's records
while its round runs, then releases them. A session that arrives after the
last round is released at once.

The first round starts when the pipeline first looks for job data. The next
round starts after ``jobs_per_round`` jobs trained since the previous
round started, and only while the task pool holds fewer than ``refill_tasks``
tasks, so saved tasks roll out before more are generated. The pool counts
tasks that wait for a rollout, not tasks in a rollout that runs. Job data that
the backend dropped as stale does not count. A round that made no train task
and a rollout that trained nothing are followed at once, because no job data comes
from them, and so is an empty task pool with no rollout running. The ``rounds``
limit counts the rounds that finished a generation; a round of replayed
sessions does not count. A round whose generation ran out of attempts counts
too, and the next round retries the same sessions and finishes the same
generation from its saved candidates.

The record-to-task stage keeps its tasks in the task store ``generated-tasks``
of the scenario session (see `Where the stages live`_). It records each
publication or rejection, so a retry of a generation uses the saved candidates
and does not generate the tasks again. Tasks that share a source record get the
same split, and a published task keeps its split. A task whose sources join
tasks of both splits is rejected. The store also keeps each train task's first
rollout result, the generations whose results the generator received, and the
tasks that trained.

After a restart, the record-to-task stage continues with the next generation number.
It puts the train tasks that did not train back in the task pool and sends the results that the generator did not receive, even when no other job is due. A task that trained before its result was saved gets an infrastructure-failure result, so its generation's feedback completes.
A replayed session that a complete generation already used makes no new generation.
A task whose episodes were running at the stop has no result, so it runs again.
``GET /reef/status`` shows the generator, the generation in flight, the completed generations and the last error under ``task_generation``, task counts under ``task_store``, and the task pool size.

Container operations run in the generator service, ``reef.record2task.generator``.
The record-to-task stage calls this service over HTTP. The task store is the
source of each task; the service's tasks root holds staged copies. Staging the
same task again under its name returns the staged copy, so episodes that a stop
interrupted, or a service that restarted with an empty tasks root, can still
run a saved task. The ``generator`` deployment
section configures the service that ``reef serve`` starts.
See `generator configuration
<../reference/configuration.rst#the-generator-section>`__ for its settings.

A record's path to a job
------------------------

.. code:: text

   report -> record2record: validate references, filter -> record set
       -> record2task: make_sample -> TrainDataItem -> record2record: group -> record set unit
       -> task2job: make_job_data -> job data -> acknowledge

   computed record -> record2record: observe/track -> dispatch -> async judge -> judged set
       -> record2task: make_sample -> TrainDataItem -> record2record: record set unit
       -> task2job: make_job_data -> job data -> acknowledge

   uploaded trajectory -> TrajectoryItem -> task2job: make_job_data -> job data -> acknowledge

Where the stages live
---------------------

Each stage has its own package: ``reef/record2record/``, ``reef/record2task/``,
and ``reef/task2job/``. The pipeline lives in ``reef/train/pipeline.py``, and
the context that the stages read in ``reef/pipeline_context.py``. A stage that
saves generated tasks opens a store with
``context.task_storage.open(name, split=...)`` (see `Task stores
<../reference/python-api.rst#task-stores>`__). In a composite recipe,
each component opens its store names under its component name.
``task_storage`` is ``None`` if the trainer has no scenario session. The stage
packages also hold shared implementations: ``SampleAssembly``
(``reef/record2task/assembly.py``), and ``DistillTasks``
(``reef/record2task/distill.py``) with ``DistillJobData``
(``reef/task2job/distill.py``) for the distilling recipes. ``DistillTasks``
emits the student's rollout plus the teacher sequence: the request that the
recipe composes from the recorded request and the report's
``teacher_context`` (its ``teacher_request`` override), rendered with the
served model's chat template. A method's own stages live in
``recipes/<name>/processor.py`` (the record-to-task and task-to-job classes)
and ``recipes/<name>/recipe.py`` (the record stage); harness evolution's live
in ``reef/train/cordis_backend/processor.py``. Recipe-specific correlation and
model clients belong beside the method's stages.

Compatibility
-------------

A recipe's processor class becomes a ``Task2Job`` with its ``make_job_data``,
named in the recipe's ``pipeline_spec()``; the trainer drives a
``TrainingPipeline`` instead of a ``DataProcessor``. Reported-feedback recipes
must replace ``judge`` and ``ReportDecision`` with ``make_sample`` returning
``TrainDataItem`` on a ``ReportTasks`` subclass. Remove ``WAIT``/``NEVER`` branches: invalid
references fail admission, and training contract failures must raise. Existing
stored records replay through the new contract; legacy reports with invalid
references or eligibility flags fail explicitly and need correction before replay.

The harness recipe no longer supports ``max_score`` or filters successful
reports. Remove that setting from configuration and Python construction.
Identical report retries still return their original receipt, including after
source eviction; changed content with the same id still conflicts.
