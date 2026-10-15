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
  decides which records train, in which sets, and in what order. A set is a
  report with the inference records that it references, a ready group, one
  judged record, or a round request.
- The **record-to-task stage** (``Record2Task``, ``reef.record2task``) turns
  one record set into tasks. A task is a ``TrainDataItem``: a ``TrajectoryItem``
  that trains directly, or a ``TaskItem`` (a Harbor task) that an algorithm
  rolls out. A batch may mix both.
- The **task-to-batch stage** (``Task2Batch``, ``reef.task2batch``)
  shapes the tasks of the selected sets into one ``TrainingBatch``. Grouped
  recipes set ``group_id`` on each member.

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
scenario builds its pipeline. Without a record stage, the pipeline passes
each record through and makes no tasks.

Nothing numeric lives in the stages. Advantages and the loss family are the
step objective's. See `batch values <../reference/python-api.rst#batch>`__ for
formats and algorithm support.

The pipeline
------------

``TrainingPipeline`` (``reef/train/pipeline.py``) is the only object that the
trainer gives records to and takes batches from. Recipes do not subclass it.
The trainer calls it on its own thread, under its lock; no call may block:

- ``ingest(record)``: one record arrives. A TRAIN instruction stays in the
  pipeline; every other record goes to the record stage. The pipeline gives
  each set that the stage forms to the record-to-task stage and gives the
  result back to the record stage (``settle``): ``READY`` with tasks, or a
  skip, which ends the set's records.
- ``ready()``: is a batch available. It first lets background work land: the
  record-to-task stage's ``poll`` (for example a finished round), then the
  record stage's ``poll`` (for example finished judgments).
- ``build_batch()``: reserve ``batch_size`` sets from the record stage, in
  its selection order, and give their tasks to ``make_batch``. The trainer
  validates the batch against ``output_schema``. The same batch stays
  reserved until it is acknowledged or released.
- ``acknowledge(batch_id, trained=True)``: the training step consumed that
  batch. The pipeline consumes the reservation from the record stage's pool,
  and the stage ends the sets and returns the IDs of the records that they
  consumed. The commit record persists these IDs, so recovery never ingests
  them again. ``trained=False`` says the backend dropped the batch as stale,
  for a record stage that paces work on what trained.

The pipeline also merges the record and task stages' ``status()``, metrics, background
work (``background_jobs_running()``), and ``close()``. ``releasable_record_ids()``
returns the completed records that no held set references, and
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

``status()`` reports ``ready_units`` (the held sets) and ``batch_size``. A
stage adds an entry only when a terminal outcome cannot become a batch and an
external runner must stop waiting (TTTD reports a complete mixed-artifact
step as an invariant failure).

The record-to-record stage
--------------------------

1. The stage reads each record and forms record sets.
2. The record-to-task stage makes tasks from each formed set.
3. The stage holds the sets that train in its ``pool``, places reports in
   their groups, and drops sets that can no longer train. The selection
   policy orders the pool.
4. ``build_batch`` reserves sets from the pool. ``acknowledge`` consumes the
   reservation, and the stage ends the consumed sets (``consume``).

A record that the stage ends, for example a report in a taken slot or a set
that is too stale, appears in ``releasable_record_ids()``. The next commit
names it, so the record never trains after a restart.

Every record stage runs the staleness check. No other part of Reef checks the
version of training data. The check has these steps:

1. When the scenario reserves a batch, it gives the pipeline the runtime load
   ID that serves its committed release (``observe_serving_runtime_load_id``).
   The stage keeps these versions, oldest first.
2. Before each batch, the stage ends every held set that has data more than
   ``max_staleness`` versions behind the current version (``drop_stale_sets``).

The lag counts only the versions of this scenario. A publication of another
scenario on a shared engine does not age the data. These rules also apply:

* A sample without a recorded version is stale.
* A sample from a version that the stage did not see is stale. A rebuilt
  trainer knows only the versions that it sees after the rebuild.
* While the serving version is unknown, for example before the first
  publication, no data is stale.

A stage that holds combined sets overrides ``drop_held_set`` to end a whole
set. ``ReportRecord2Record`` does this for a ready group.

.. list-table::
   :header-rows: 1

   * - Stage
     - Decides
   * - ``PassThroughRecord2Record``
     - nothing; the default of a pipeline that trains no weights
   * - ``ReportRecord2Record``
     - report validation, groups and slots, the manual-mode limit, source release
   * - ``ExclusiveReportRecord2Record``
     - a ``ReportRecord2Record`` that trains every report by itself; a report that ends owns its inference records
   * - ``JudgingRecord2Record``
     - correlation and the judging worker
   * - ``InferenceRecord2Record``
     - one set per inference record; other records are released
   * - ``RoundRecord2Record``
     - a ``ReportRecord2Record`` that also forms a request when a task-generation round is due

Training modes
--------------

``training_mode`` is an attribute of each pipeline. The recipe passes its
initial value through ``Trainer.build`` and ``PipelineContext``; it defaults
to ``auto``. A ``Task2Batch`` declares the modes it supports in
``supported_training_modes``; the default is ``auto`` only. Ingestion,
acknowledgement and buffer release use the same methods and buffers in every
mode. ``GET /reef/status`` reports ``pending_instructions`` as the
pipeline's ``buffered_requests`` plus the instructions still unread in
storage.

- ``auto`` batches when ``batch_size`` sets are held.
- ``manual`` batches on a queued TRAIN instruction. The instruction's batch
  has no sets, so ``make_batch`` receives no tasks.
- ``hybrid`` batches on either, and a queued instruction goes first. The
  instruction's batch takes the sets that an automatic batch would.

The pipeline attaches the instruction to ``batch.request`` and replaces the
batch id with ``<scenario>:instruction:<request id>``, the oldest instruction
first, one per step. Acknowledgement also consumes the instruction. Batch
construction must not call models or perform training.

The trainer gives TRAIN records to the pipeline when ``RequestType.TRAIN`` is
in the ``Task2Batch``'s ``required_request_types``. The pipeline queues them
FIFO regardless of the selected mode; ``auto`` leaves the queue for a mode that
takes it. Data ingestion continues normally in manual mode, so changing to auto
or hybrid can batch data already collected. In manual mode the record stage
holds at most four batches of sets and releases the oldest beyond that with a
warning, at the switch and as reports arrive.

Unsupported modes raise ``NotImplementedError``. An invalid mode name raises
``ValueError``.

Changing training mode
----------------------

``POST /reef/scenarios/{scenario}/update`` selects ``auto``, ``manual`` or
``hybrid`` on the existing pipeline. The trainer serializes selection with
ingestion and reservation. A reserved batch stays unchanged and acknowledgement
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
- ``make_batch(tasks, batch_number)`` on a ``Task2Batch`` subclass: assemble
  the flat tuple of selected tasks into a batch. The pipeline tracks
  consumption by set, including tasks that the recipe removes from training.

``exclusive_sources`` makes a terminal report own its referenced inference
records outright, and ``ordered_groups`` batches ready groups in key order.
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
and retention. A batch is ready in automatic mode after ``batch_size`` sets
accumulate; a set is one report or one complete group. Reports arriving later
for already consumed sources cannot train those sources again.

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

``TaskGenerator`` is in ``reef.core.tasks``. Reef calls its class method
``from_config(config, *, scenario, environ)`` once for each scenario. The
generator gets what it needs from its ``config`` and from these calls only:

- ``async generate(request: TaskGenerationRequest) -> tuple[GeneratedTask, ...]``
  returns zero or more tasks. A ``GeneratedTask`` holds a ``HarborTask`` and
  its ``measurement_arms``. Preserve the request's source record IDs in each
  task's ``source_agent_record_ids``.
- ``async validate(task: HarborTask) -> TaskValidationResult`` checks one
  task. Empty ``errors`` means the task passed. Return task defects as errors.
  Raise when the checks cannot run. The task is a value, so the check does not
  need a file system that it shares with Reef.
- ``async record_results(results: tuple[TaskPlayResult, ...])`` receives the
  results of the last ``generate`` call. The default keeps nothing.

A ``PlayArm(name, count, extra_instruction_files)`` asks for ``count`` more
plays of a task. Each play shows the agent the files that
``extra_instruction_files`` names after the instruction, for example
``solution/hint.txt``. The recipe never trains on a measurement arm.

A ``TaskPlayResult`` holds the task, ``training_rewards``,
``measurement_rewards`` (by arm name), and an ``error``. The rewards include
only episodes that ran. An episode that ran
without a reward counts as 0. A non-empty ``error`` means that the task was not
played: it failed validation, the generator service refused it, or the agent
could not start on it.

The generator service scores an episode whose verifier ran past its timeout
as 0, and reports it, so a training play of that kind trains. The task's
oracle passed the same verifier, so the timeout is the agent's.

The recipe splits the rounds across its stages. The record stage
(``GeneratedTaskRecord2Record``, a ``RoundRecord2Record``) decides when a round is due
and holds the source sessions. The record-to-task stage (``GeneratedTaskTasks``) runs one round
at a time, on its own ``BackgroundJobWorker``, outside the trainer lock: each
time the pipeline looks for a batch, it starts a round when one is due. In each round it:

1. Asks the generator for tasks with the configured ``description`` and the
   request's source sessions.
2. Validates each task with the generator.
3. Writes each valid task through the generator service and plays it
   ``plays_per_task`` times on the training arm, then plays each measurement arm.
4. Sends the results to ``record_results``.

The training plays carry the arm ``plain``, and only reports with that label
train. A generated task whose measurement arm is
named ``plain`` comes back unplayed with an error. The record stage groups
the training reports by ``metadata.task.name``. A group is complete at
``plays_per_task`` episodes, and a batch holds ``tasks_per_step`` complete
groups. A training play that reports nothing, such as a failed trial, never
joins its group. When the task's plays end, its group is complete at the
number of plays that reported; a group of one is discarded. The
``task_group`` objective centers and scales each reward within its
task group, on Tinker's ``importance_sampling`` loss. The record stage
releases the reports of measurement arms and reports without an arm, such as a
generator's own reports, without assembling them.

A round can also generate tasks from recorded sessions. Set the record
stage's ``source_records_per_round`` above 0 and its ``source_tag`` to a
``name=value`` tag. A report without a training arm, whose inference records all carry the
header ``x-reef-tag-<name>: <value>``, is a source session. The record stage
holds source sessions in arrival order. Each round takes at most
``source_records_per_round`` of them, and a round without a waiting session
has none. The generator receives each session's inference records and then
its report as ``source_records``. The record stage keeps a session's records
until its round finishes, then releases them. A session that arrives after the
last round is released at once. A restart before the release can use a
session again.

The first round starts when the pipeline first looks for a batch. The next
round starts after ``batches_per_round`` batches trained since the previous
round started; a batch that the backend dropped as stale does not count. A
round that played no task is followed at once. An exception from the
generator stops the round. The record-to-task stage still sends the results
it has to ``record_results``. A summary of each round goes to
``state_dir/<scenario>/round-<round>.json``, with the IDs of its source records. After a restart, the
record-to-task stage continues with the next round. ``GET /reef/status`` shows the generator, the
round, the task counts, and the last error under ``task_generation``.

Container operations run in the generator service, ``reef.record2task.generator``.
The record-to-task stage calls this service over HTTP. The ``generator`` deployment
section configures the service that ``reef serve`` starts.
See `generator configuration
<../reference/configuration.rst#the-generator-section>`__ for its settings.

A record's path to a batch
--------------------------

.. code:: text

   report -> record2record: validate references, filter, group -> record set
       -> record2task: make_sample -> TrainDataItem -> task2batch: make_batch -> batch -> acknowledge

   computed record -> record2record: observe/track -> dispatch -> async judge -> judged set
       -> record2task: make_sample -> TrainDataItem -> task2batch: make_batch -> batch -> acknowledge

Where the stages live
---------------------

Each stage has its own package: ``reef/record2record/``, ``reef/record2task/``,
and ``reef/task2batch/``. The pipeline lives in ``reef/train/pipeline.py``, and
the context that the stages read in ``reef/pipeline_context.py``. The stage
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
