Training pipeline
=================

A scenario's training pipeline turns records into typed training batches. It
has three stages, with one data type between each pair. A recipe writes the
stages; Reef owns the pipeline that drives them. This page explains the
stages, what a recipe writes for each, the training modes, the task
generation contract, and the path a record takes to a batch. Storage owns
deletion.

.. page::
   :for: recipe authors writing the stages of a method, and anyone reading ``reef/train/pipeline.py``
   :needs: the stage hooks from `Python API <../reference/python-api.rst#pipeline-stages>`__
   :outcome: which stages a method needs, what it writes, and what it never writes

The three stages
----------------

.. code:: text

   record ─► record2record ─► record set ─► record2task ─► task ─► task2batch ─► batch ─► trainer

- The **record-to-record stage** (``Record2Record``, ``reef.record2record``)
  decides which records train, in which record sets and batch units, and in
  what order (see `Record sets and batch units <#record-sets-and-batch-units>`__).
- The **record-to-task stage** (``Record2Task``, ``reef.record2task``) turns
  one record set into tasks. A task is a ``TrainDataItem``: a ``TrajectoryItem``
  that trains directly, or a ``TaskItem`` (a Harbor task) that an algorithm
  rolls out. A batch may mix both.
- The **task-to-batch stage** (``Task2Batch``, ``reef.task2batch``)
  shapes the tasks of the selected batch units into one ``TrainingBatch``. Grouped
  recipes set ``group_id`` on each member.

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
   * - ``Task2Batch``
     - ``make_batch(tasks, batch_id)``
     - ``TrainingBatch``

When the pipeline looks for a batch, it first calls the record stage's
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
- ``Task2Batch.from_config(config, *, context, record2record, record2task, environ)``

A later stage receives the stages built before it. It reads a value that an
earlier stage owns from that stage, so the value is configured once: TTTD's
batches read the step grid from its record stage, and OpenClaw-RL's batches
read the record stage's judging. ``config`` is the stage's own config. A stage
declares its keys as a frozen settings dataclass that inherits
``reef.core.plugins.PluginSettings`` (fields made with
``reef.core.config.config_option``). Its ``from_config`` parses the config
with ``Settings.parse(config, plugin_class=cls)`` and gives the settings
object to the constructor; the stage never reads a raw mapping. A stage
without config parses with ``NoSettings``, which refuses any key. Reading the
deployment checks the stage classes; a bad config fails when the first
scenario builds its pipeline. Without a record stage, the pipeline forms no
record set and makes no tasks.

Nothing numeric lives in the stages. Advantages and the loss family are the
step objective's. See `batch values <../reference/python-api.rst#batch>`__ for
formats and algorithm support.

Record sets and batch units
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

- ``READY``: one or more tasks. The records wait in one batch unit.
- ``SKIPPED``: no task. The stage ends the records.

A batch unit is what a batch takes all of or none of, and what
``batch_size`` counts. A unit is a record set unit or one uploaded trajectory
(see `Uploaded trajectories <#uploaded-trajectories>`__).

A record set unit (``RecordSetUnit``) holds one or more record sets that must
train in the same batch. A report by itself is a unit of one record set. A
ready group is a unit of the record sets of all its reports. The record stage
holds these units in its pool. Consuming a unit writes the IDs of all its
records to the commit.

The pipeline
------------

``TrainingPipeline`` (``reef/train/pipeline.py``) is the only object that the
trainer gives records to and takes batches from. Recipes do not subclass it.
The trainer calls it on its own thread, under its lock; no call may block:

- ``push(record)``: one record arrives. A TRAIN instruction stays in the
  pipeline. The pipeline pushes every other record to the record stage. It
  then moves the record sets: it pulls the sets that the record stage
  formed, pushes each one to the record-to-task stage, pulls the results
  that are ready, and gives each result back to the record stage
  (``settle``). A set without a result waits, and its records stay held (see
  `Slow record-to-task work <#slow-record-to-task-work>`__).
- ``ready()``: is a batch available. It first advances the record stage
  (``advance``), for example to take in finished judgments or to start a
  task-generation round that is due. Then it moves the record sets as
  ``push`` does.
- ``build_batch()``: reserve ``batch_size`` batch units from the record stage, in
  its selection order, and give their tasks to ``make_batch``. Uploaded
  trajectories come first (see `Uploaded trajectories <#uploaded-trajectories>`__). The trainer
  validates the batch against ``output_schema``. The same batch stays
  reserved until it is acknowledged or released.
- ``acknowledge(batch_id, trained=True)``: the training step consumed that
  batch. The pipeline consumes the reservation from the record stage's pool,
  and the stage ends the units and returns the IDs of the records that they
  consumed. The commit record persists these IDs, so recovery never pushes
  them again. ``trained=False`` says the backend dropped the batch as stale,
  for a record stage that paces work on what trained.

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

``status()`` reports ``ready_units`` (the held record set units and uploaded trajectories) and
``batch_size``. A stage adds an entry only when a terminal outcome cannot become a batch and an
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

A recipe takes uploads when its ``Task2Batch`` sets ``accepts_trajectories``
to True. The default is False. Admission refuses an upload with 400 when no
trainer of the scenario accepts uploaded trajectories.

An uploaded trajectory does not go to the record stage or the record-to-task
stage. The pipeline holds it as a ``TrajectoryItem`` of the uploaded document
until a batch takes it. A batch takes uploaded trajectories first, in upload
order, and then record set units. Each uploaded trajectory counts as one batch
unit. In ``manual`` mode, an instruction's batch takes no units. In a scenario
of several components, each trainer that accepts uploaded trajectories trains
on them. Acknowledgement consumes all units of the batch. The commit lists
each uploaded trajectory as ``trajectory:<trajectory_id>``. Thus a rebuilt
trainer does not read it again.

An uploaded trajectory has no captured rollout tensors. Thus a backend that
needs these tensors cannot train on it.

The record-to-record stage
--------------------------

1. The stage reads each record (``push``) and forms record sets (``form``).
   The pipeline pulls them (``pull``).
2. The record-to-task stage makes tasks from each record set.
3. The stage holds the record sets that train in batch units in its
   ``pool``, places reports in their groups, and drops units that can no
   longer train. The selection policy orders the pool.
4. ``build_batch`` reserves units from the pool. ``acknowledge`` consumes the
   reservation, and the stage ends the consumed units (``consume``).

A record that the stage ends, for example a report in a taken slot or a unit
that is too stale, appears in ``releasable_record_ids()``. The next commit
names it, so the record never trains after a restart.

Every record stage runs the staleness check. No other part of Reef checks the
version of training data. The check has these steps:

1. When the scenario reserves a batch, it gives the pipeline the runtime load
   ID that serves its committed release (``observe_serving_runtime_load_id``).
   The stage keeps these versions, oldest first.
2. Before each batch, the stage ends every held unit that has data more than
   ``max_staleness`` versions behind the current version (``drop_stale_units``).

The lag counts only the versions of this scenario. A publication of another
scenario on a shared engine does not age the data. These rules also apply:

* A sample without a recorded version is stale.
* A sample from a version that the stage did not see is stale. A rebuilt
  trainer knows only the versions that it sees after the rebuild.
* While the serving version is unknown, for example before the first
  publication, no data is stale.

A stage that holds groups overrides ``drop_held_unit`` to end a whole group.
``ReportRecord2Record`` does this for a ready group.

.. list-table::
   :header-rows: 1

   * - Stage
     - Decides
   * - ``NoRecordSets``
     - nothing; it forms no record set. The default of a pipeline that trains no weights
   * - ``ReportRecord2Record``
     - report validation, groups and slots, the manual-mode limit, source release
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
to ``auto``. A ``Task2Batch`` declares the modes it supports in
``supported_training_modes``; the default is ``auto`` only. Record push,
acknowledgement and buffer release use the same methods and buffers in every
mode. ``GET /reef/status`` reports ``pending_instructions`` as the
pipeline's ``buffered_requests`` plus the instructions still unread in
storage.

- ``auto`` batches when ``batch_size`` batch units are held.
- ``manual`` batches on a queued TRAIN instruction. The instruction's batch
  has no units, so ``make_batch`` receives no tasks.
- ``hybrid`` batches on either, and a queued instruction goes first. The
  instruction's batch takes the units that an automatic batch would.

The pipeline runs the oldest instruction first, one per step. It gives the
instruction's batch the id ``<scenario>:instruction:<request id>`` and attaches
the instruction to ``batch.request``. Acknowledgement also consumes the
instruction. Batch construction must not call models or perform training.

The trainer gives TRAIN records to the pipeline when ``RequestType.TRAIN`` is
in the ``Task2Batch``'s ``required_request_types``. The pipeline queues them
FIFO regardless of the selected mode; ``auto`` leaves the queue for a mode that
takes it. The pipeline takes records normally in manual mode, so changing to auto
or hybrid can batch data already collected. In manual mode the record stage
holds at most four batches of units and releases the oldest beyond that with a
warning, at the switch and as reports arrive.

Unsupported modes raise ``NotImplementedError``. An invalid mode name raises
``ValueError``.

Changing training mode
----------------------

``POST /reef/scenarios/{scenario}/update`` selects ``auto``, ``manual`` or
``hybrid`` on the existing pipeline. The trainer serializes selection with
record push and reservation. A reserved batch stays unchanged and acknowledgement
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
- ``make_batch(tasks, batch_id)`` on a ``Task2Batch`` subclass: assemble
  the flat tuple of selected tasks into a batch with the ID ``batch_id``.
  The pipeline tracks consumption by batch unit, including tasks that the
  recipe removes from training.

A report that ends after it was a record set owns its referenced inference
records outright. ``exclusive_sources`` gives the same ownership to a report
that ends before, and ``ordered_groups`` batches ready groups in key order.
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
and retention. A batch is ready in automatic mode after ``batch_size`` batch
units accumulate; a unit is one report or one complete group. A report that
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
``make_sample(record, judgment)``, and its ``Task2Batch`` supplies
``make_batch``. The stage holds judged records until a batch uses them.

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
  task's ``source_agent_record_ids``.
- ``async validate(task: HarborTask) -> TaskValidationResult`` checks one
  task. Empty ``errors`` means the task passed. Return task defects as errors.
  Raise when the checks cannot run. The task is a value, so the check does not
  need a file system that it shares with Reef.
- ``async record_results(results: tuple[TaskPlayResult, ...])`` receives the
  results of the last ``generate`` call. The default keeps nothing.

A ``TaskPlayResult`` holds the task, ``training_rewards``, a ``split``
(``train`` by default), and an ``error``. The rewards include only episodes
that ran. An episode that ran
without a reward counts as 0. A non-empty ``error`` means that the task was not
played: it failed validation, the generator service refused it, or the agent
could not start on it.

An eval result has ``split="eval"``, no rewards, and no error. The task was
not played. A generator must not use an eval result as training feedback or as
experience for later generations.

The generator service scores an episode whose verifier ran past its timeout
as 0, and reports it, so a training play of that kind trains. The task's
oracle passed the same verifier, so the timeout is the agent's.

The recipe splits the rounds across its stages. A round is not a record set,
and it does not train. The record stage (``GeneratedTaskRecord2Record``)
decides when a round is due, holds the source sessions, and gives the round
number and its sessions to the record-to-task stage (``GeneratedTaskTasks``).
That stage runs the round on its own ``BackgroundJobWorker``, outside the
trainer lock, and tells the record stage when the round ends. One round runs
at a time. In each round it:

1. Asks the generator for tasks with the configured ``description`` and the
   request's source sessions.
2. Saves the candidates in the scenario's task store, then validates each
   task with the generator.
3. Publishes each valid task with its train/eval split.
4. Writes each training task through the generator service and plays it
   ``plays_per_task`` times on the training arm. It does not play eval tasks.
5. Sends the results to ``record_results``.

The training plays carry the arm ``plain``, and only reports with that label
train. A generator can play a task more times through the generator service
on another arm, as SPADE does for its hint plays; those reports do not train.
The record stage groups
the training reports by ``metadata.task.name``. A group is complete at
``plays_per_task`` episodes, and a batch holds ``tasks_per_step`` complete
groups. A training play that reports nothing, such as a failed trial, never
joins its group. When the task's plays end, its group is complete at the
number of plays that reported; a group of one is discarded. The
``task_group`` objective centers and scales each reward within its
task group, on Tinker's ``importance_sampling`` loss. The record stage
releases the reports of other arms and reports without an arm, such as a
generator's own reports, without assembling them.

A round can also generate tasks from recorded sessions. Set the record
stage's ``source_records_per_round`` above 0 and its ``source_tag`` to a
``name=value`` tag. A report without a training arm, whose inference records all carry the
header ``x-reef-tag-<name>: <value>``, is a source session. The record stage
holds source sessions in arrival order. Each round takes at most
``source_records_per_round`` of them, and a round without a waiting session
has none. The generator receives each session's inference records and then
its report as ``source_records``. The record stage keeps a session's records
while its round runs, then releases them. A session that arrives
after the last round is released at once. A restart before the release can use a
session again.

The first round starts when the pipeline first looks for a batch. The next
round starts after ``batches_per_round`` batches trained since the previous
round started; a batch that the backend dropped as stale does not count. A
round that played no task is followed at once. A round whose job failed waits
for batches like a played round, so a failing worker does not start round
after round. If the worker is closed or broken, it refuses the next round and
no round runs again. An exception from the generator stops the round. The
record-to-task stage still sends the results it has to ``record_results``. A
summary of each round goes to ``state_dir/<scenario>/round-<round>.json``,
with the IDs of its source records. After a restart, the record-to-task stage
continues with the next round. ``GET /reef/status`` shows the generator, the
round, the task counts, and the last error under ``task_generation``.

The record-to-task stage keeps its tasks in the task store ``generated-tasks``
of the scenario session (see `Where the stages live`_). It records each
publication or rejection before play. Thus a retry of a round uses the saved
candidates and does not generate the tasks again. Tasks that share a source
record get the same split, and a published task keeps its split. A task whose
sources join tasks of both splits is rejected. ``GET /reef/status`` shows the
number of train, eval, rejected, and pending tasks under ``task_store``.

Container operations run in the generator service, ``reef.record2task.generator``.
The record-to-task stage calls this service over HTTP. The ``generator`` deployment
section configures the service that ``reef serve`` starts.
See `generator configuration
<../reference/configuration.rst#the-generator-section>`__ for its settings.

A record's path to a batch
--------------------------

.. code:: text

   report -> record2record: validate references, filter -> record set
       -> record2task: make_sample -> TrainDataItem -> record2record: group -> batch unit
       -> task2batch: make_batch -> batch -> acknowledge

   computed record -> record2record: observe/track -> dispatch -> async judge -> judged set
       -> record2task: make_sample -> TrainDataItem -> record2record: batch unit
       -> task2batch: make_batch -> batch -> acknowledge

   uploaded trajectory -> TrajectoryItem -> task2batch: make_batch -> batch -> acknowledge

Where the stages live
---------------------

Each stage has its own package: ``reef/record2record/``, ``reef/record2task/``,
and ``reef/task2batch/``. The pipeline lives in ``reef/train/pipeline.py``, and
the context that the stages read in ``reef/pipeline_context.py``. A stage that
saves generated tasks opens a store with
``context.task_storage.open(name, split=...)`` (see `Task stores
<../reference/python-api.rst#task-stores>`__). In a composite recipe,
each component opens its store names under its component name.
``task_storage`` is ``None`` if the trainer has no scenario session. The stage
packages also hold shared implementations: ``SampleAssembly``
(``reef/record2task/assembly.py``), and ``DistillTasks``
(``reef/record2task/distill.py``) with ``DistillBatches``
(``reef/task2batch/distill.py``) for the distilling recipes. ``DistillTasks``
emits the student's rollout plus the teacher sequence: the request that the
recipe composes from the recorded request and the report's
``teacher_context`` (its ``teacher_request`` override), rendered with the
served model's chat template. A method's own stages live in
``recipes/<name>/processor.py`` (the record-to-task and task-to-batch classes)
and ``recipes/<name>/recipe.py`` (the record stage); harness evolution's live
in ``reef/train/cordis_backend/processor.py``. Recipe-specific correlation and
model clients belong beside the method's stages.

Compatibility
-------------

A recipe's processor class becomes a ``Task2Batch`` with its ``make_batch``,
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
