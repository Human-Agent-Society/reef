Pipeline stage API
==================

The signatures of the two stages that turn records into job data, of the report and judging engines that a record-to-sample stage builds on, and of the task generator plugin.

The guide explains the behavior: `Samples service <../developer-guide/recipe-samples-service.rst>`__ for the record-to-sample stage, `Control service <../developer-guide/recipe-control-service.rst>`__ for the sample-to-job stage, and `Task generators <../developer-guide/task-generators.rst>`__.

Pipeline stages
---------------

.. code:: python

   from reef.samples import (
       GroupDecision,
       JudgedSamples,
       RecordJudging,
       ReportContext,
       ReportGrouping,
       ReportSamples,
       SkippedReport,
   )
   from reef.train.sample2job import DistillJobData, Sample2Job

A training pipeline (``reef.train.pipeline.TrainingPipeline``) turns durable records into typed job data through two stages. A recipe names a default class for each in ``pipeline_spec()``: ``PipelineSpec(record2sample=..., sample2job=...)``.

Reef owns replay, retention, deduplication, pending job data, and exactly-once consumption. A method implements only the hooks below. The hooks run synchronously on the trainer thread, so they must not block on network or model latency. Reports must reference existing inference records in the same scenario.

The record-to-sample stage
~~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.samples.Record2Sample`` decides which records train and which train together. The default stage, ``NoSamples``, makes no sample, so nothing trains.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Contract
   * - ``push(record)``
     - Required. The pipeline pushes every record to the stage.
   * - ``make_samples(record_set)``
     - Required. Makes the samples of one record set. A sample is a ``Sample`` value.
   * - ``form(record_set)``
     - Calls ``make_samples`` and holds the samples in a record set unit until the unit is ready.
   * - ``advance(now)``
     - A stage with slow work keeps its records and forms the set here when the work ends (`Slow work <../developer-guide/recipe-samples-service.rst#slow-work>`__).
   * - ``is_stale(samples, served)``
     - Whether a group of the stage's stored samples can no longer train, now that the scenario serves ``served`` (``ServedArtifacts``). The samples service calls it only when the class attribute ``goes_stale`` is true (the default). It marks the stale samples, and the automatic jobs leave them out. The default is on-policy (`Staleness <../developer-guide/recipe-samples-service.rst#staleness>`__).
   * - ``uploaded_sample(trajectory) -> TrajectorySample``
     - Turns an uploaded ATIF trajectory into its training sample. It must be pure. It raises ``ValueError`` to skip the trajectory (`Uploaded trajectories <../developer-guide/recipe-control-service.rst#uploaded-trajectories>`__).
   * - ``job_ended(outcome)``
     - Called once the commit of a job's end is durable (`The job loop <python-api-training.rst#the-job-loop>`__). The default does nothing.
   * - ``releasable_record_ids()``, ``release_records(ids)``
     - The samples service calls them to free the stage's memory. Storage controls capacity independently of these buffers.

A ``Record2Sample`` keeps its state in memory and in the store, not in a directory of its own, so a scenario delete archives nothing for it.

The sample-to-job stage
~~~~~~~~~~~~~~~~~~~~~~~

``reef.train.sample2job.Sample2Job.make_job_data(samples, data_id) -> JobData`` shapes the samples of the next job, in use order, into one job's data with the ID ``data_id``. The pipeline assigns the ID: ``<scenario>:data:<number>`` for automatic job data and ``<scenario>:request:<request id>`` for a request's data. The number counts the job data of one pipeline. It is not a training step.

``Sample2Job.__init__(context, buffer_size, job_samples, *, max_job_samples=None)`` sets the most samples that one automatic job takes and the fewest samples of one job (`Job size <../developer-guide/recipe-control-service.rst#job-size>`__):

- A ``job_samples`` of ``None`` is a stage that sets no job size. ``Trainer.build`` raises ``PluginConfigError`` for a trainer that trains with such a stage.
- A ``buffer_size`` of ``None`` takes ``job_samples``.
- The constructor raises ``PluginConfigError`` for a ``buffer_size`` smaller than ``job_samples``, and for a ``job_samples`` above ``max_job_samples`` (an SDPO or TTTD grid).
- ``Sample2JobSettings`` parses the ``buffer_size`` and ``job_samples`` keys, and a stage with more keys extends it.

The stage also declares these class attributes:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Attribute
     - Contract
   * - ``supported_training_modes``
     - The modes that the stage supports. ``TrainingPipeline.training_mode`` selects automatic (``auto``) or request-triggered (``manual``) training. In ``manual``, a request's data holds the samples that the request names.
   * - ``reads_requests``
     - Whether a request that names no sample runs. Default ``False``.
   * - ``required_request_types``
     - Restricts which record types reach the pipeline. The default takes inference records and reports.
   * - ``accepts_trajectories``
     - True to train on uploaded ATIF trajectories. Default ``False``.
   * - ``accepts_imported_tasks``
     - True for a stage that trains imported tasks. Default ``False``. The service takes a task item in the ``train`` split of a scenario only when a trainer of the scenario sets it (`Task stores <python-api-data-stores.rst#task-stores>`__).

Building the stages
~~~~~~~~~~~~~~~~~~~

Reef builds the stages for each scenario, in this order, with these class methods. The sample-to-job stage receives the record-to-sample stage:

- ``Record2Sample.from_config(config, *, context, environ)``
- ``Sample2Job.from_config(config, *, context, record2sample, environ)``

``config`` is the stage's config from the deployment's ``pipeline`` section. ``environ`` holds the environment that the deployment was read with.

- Declare the stage's keys in a frozen ``reef.core.plugins.PluginSettings`` subclass with ``reef.core.config.config_option`` fields, and validate them in ``__post_init__``. In ``from_config``, parse the config with ``Settings.parse(config, plugin_class=cls)`` (``NoSettings`` for a stage without config). A bad config fails when the first scenario builds its pipeline (`Select and configure the stage <../developer-guide/recipe-samples-service.rst#select-and-configure-the-stage>`__).
- A value that both stages need belongs to the record-to-sample stage. A sample-to-job stage that needs the record-to-sample stage as a specific class checks it with ``isinstance`` and raises ``PluginConfigError`` otherwise.

Every stage gets the scenario's experiment logger as ``context.experiment_logger`` (``self.experiment_logger`` on a ``Sample2Job``). Log finite numeric metrics under the method's namespace. Stage code never imports W&B, and the logger does nothing when tracking is off.

Reported feedback
~~~~~~~~~~~~~~~~~

The record-to-sample stage of a reported-feedback recipe is a subclass of ``ReportSamples(context, grouping=None, *, exclusive_sources=False)``. `Reported feedback <../developer-guide/recipe-samples-service.rst#reported-feedback>`__ explains the engine.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Contract
   * - ``ReportSamples.make_sample(context) -> Sample | SkippedReport``
     - Assemble one valid report and its resolved inferences. Raise on data errors. Return ``SkippedReport(reason)`` for a valid report that cannot train; the stage releases it. The default makes the report's trajectories one ATIF trajectory sample with the report's score.
   * - ``grouping``
     - A ``ReportGrouping``. With ``None``, the stage trains every report by itself.
   * - ``exclusive_sources``
     - With ``True``, a report that ends owns its inference records. Use it when each report has its own inferences.

The stage keeps these rules:

- A report references the last call of each trajectory that it scores. The stage expands each reference to the calls of its trajectory (``reef.core.trajectory_index``) and adds the calls and the report ID to the sample's ``source_agent_record_ids``.
- No inference is in two record sets. A report whose trajectory holds an inference of another report ends without training, and so does a stored report that lists calls inside one trajectory.
- An invalid reference raises at admission. An incomplete group waits for more valid reports. A training input violation raises; the stage does not filter reports silently.

A ``ReportGrouping`` runs in the record-to-sample stage and reads only records:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Contract
   * - ``placement(context) -> (group_key, slot)``
     - Optional collection group and retry slot. The default, ``(None, None)``, is an independent report. A group key collects reports, and a slot deduplicates retries within that group (``None`` uses the report ID).
   * - ``decide(key, members) -> GroupDecision``
     - Required when ``placement`` supplies a group key. ``members`` are the group's ``ReportContext`` values in arrival order. Return ``READY``, ``INCOMPLETE`` or ``DISCARD``.
   * - ``is_training_report(report) -> bool``
     - Optional. ``False`` for a valid report that is not this method's training data. The stage releases it and never assembles it. Its sources go with it only under ``exclusive_sources`` or when it references more than one inference.
   * - ``status() -> Mapping``
     - Optional JSON-safe group outcomes, such as steps that can never train. The pipeline adds them to its status.

``ReportContext`` carries ``report``, ``inferences`` (the calls of the report's trajectories, oldest first), optional ``score``, and the recipe's ``parsed_report``. ``ends`` holds the referenced last calls, in reference order. ``require_score()`` returns a finite reward, or raises if the training method cannot use the supplied feedback.

A group is fixed when it is first ready; later reports for its key form a new group. This collection group differs from ``TrajectorySample.group_id``, the comparison group inside one job's data. The job's commit consumes every sample that ``make_job_data`` receives, even when the recipe omits some from its output (`The next job <../developer-guide/recipe-control-service.rst#the-next-job>`__).

``TrajectorySamples(context, *, trajectory_idle_seconds=300.0)`` is the stage of a method that needs no report:

- It forms a ``TrajectorySet`` of a trajectory's calls once the trajectory's last call has had no continuing call and no retry for ``trajectory_idle_seconds``, counted from the call's ``created_at``.
- It makes the set one ATIF ``TrajectorySample`` (``trajectory_sample``).
- Its ``next_due()`` is the time at which the earliest waiting trajectory goes idle.
- Every other record ends at once.

Computed feedback
~~~~~~~~~~~~~~~~~

Use this engine when later traffic completes an earlier record and judging it calls a model or another slow service. The recipe writes a ``RecordJudging``. Its record-to-sample stage is a ``JudgedSamples(judging, worker)`` subclass, and its ``Sample2Job`` shapes the judged samples into job data. `Computed feedback <../developer-guide/recipe-samples-service.rst#computed-feedback>`__ explains the engine.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Hook
     - Contract
   * - ``RecordJudging.observe(record, now, stage)``
     - Correlate one record with the stage's verbs. It must not block.
   * - ``async RecordJudging.judge(job)``
     - The slow judgment, on the worker thread.
   * - ``RecordJudging.expire(now)``
     - Optional. Return the receipts whose completion window ended.
   * - ``JudgedSamples.make_sample(record, judgment)``
     - A ``TrajectorySample``, or ``None`` to retire the record.

Before each ``observe``, the stage retires expired records and absorbs finished judgments. ``observe`` then uses the stage's ``track(record)``, ``tracked_record(receipt)``, ``dispatch(job)``, ``retire(receipt)`` and ``abandon(receipt)``. A failed judgment or a ``None`` sample retires the record and fails no training step. All correlation state must be reconstructible from replay.

Task generation
~~~~~~~~~~~~~~~

.. code:: python

   from reef.core.tasks import HarborTask
   from reef.recipe.generated_tasks import TaskGenerator, GeneratedTaskResult, TaskValidationResult
   from reef.samples.generator import TaskGenerationRequest

``TaskGenerator`` is the plugin base class that a deployment names in ``pipeline.record2sample.config.generator``. The built-in generated-task recipe, ``reef.recipe.generated_tasks:GeneratedTaskRecipe``, calls its methods. `Task generators <../developer-guide/task-generators.rst>`__ gives the round, its failure behavior and the meaning of each result.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Member
     - Contract
   * - ``from_config(config, *, scenario, environ, state)``
     - Class method. Returns the generator of one scenario. ``state`` is a ``GeneratorState``.
   * - ``async generate(request) -> tuple[HarborTask, ...]``
     - Generate tasks for one ``TaskGenerationRequest``.
   * - ``async validate(task) -> TaskValidationResult``
     - Check one ``HarborTask``. ``TaskValidationResult(errors=())`` exposes ``is_valid``. Each error is a non-empty explanation of a task defect. A failure to execute the checks raises an exception.
   * - ``async record_results(results, *, generation)``
     - Optional. Receives the ``GeneratedTaskResult`` values of one earlier ``generate`` call, named by ``TaskGenerationRequest.generation``, possibly after later calls and after a restart. A second call for the same generation must have no effect. The default keeps nothing.

``GeneratorState`` (``reef.recipe.generated_tasks``) is the durable state of the scenario's generator, kept in the scenario's task store: text documents by key (``read``, ``write``, ``remove``, ``keys``) and saved directories by the digest of their files (``save_files``, ``files``). A generator keeps no file of its own.

``TaskGenerationRequest(source_records, description, assets=(), asset_files={}, generation=0)`` carries source records, non-empty requirements text, and the number of the generation:

- Each ``AgentRecord`` must have a distinct ID and belong to the same scenario. Use an empty record tuple for description-only generation.
- For local materials, pass a tuple of ``Path`` values in ``assets``. For materials in memory, use ``asset_files`` to map relative file names to UTF-8 text. Supplying both fields is an error.
- The constructor validates fields without reading files.

Record2dataset accepts the same input through ``DesignerRequest(inputs=request, ...)``. Its HTTP client reads local materials and sends their contents. The service rebuilds ``TaskGenerationRequest`` from the JSON fields. The generated task preserves source record IDs in order.

``GeneratedTaskResult`` is ``RejectedTask(task, reason)``, ``HeldOutTask(task)``, ``RolledOutTask(task, rollout_rewards)`` or ``FailedTask(task, error, infrastructure_failure)``. ``rollout_rewards`` is a non-empty tuple of finite numbers; ``reason`` and ``error`` are non-empty.
