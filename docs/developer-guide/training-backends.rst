Training backends
=================

A training backend is the integration that trains weights for a deployment: Slime, Tinker, or one of your own. A recipe does not implement it. A backend has three parts:

- a deployment definition (``TrainingDeployment``), which ``training.backend`` selects
- a training component (``TrainingComponent``), which runs whole training jobs with its framework's own training loop and saves checkpoints
- the backend's facts (``TrainingBackendFacts``), which answer questions about the component in a service that does not build it

:For: authors of a training integration beside Slime and Tinker
:Before you start: `Write a recipe <write-a-recipe.rst>`__ and `Adding components <../contributing/adding-components.rst#add-a-training-integration>`__
:You will have: the deployment definition, the component contract, the backend facts, and the format of a job's data

A recipe author does not need this page: a weight method plugs into an existing backend with an objective (see `Training service <recipe-training-service.rst>`__).

.. code:: mermaid

   flowchart TD
       T[Weight trainer / JobLoop] -- submit, status, cancel --> Q[Jobs queue and training checks]
       Q -- job, check --> C[TrainingComponent in the training service]
       C --> W[Framework workers: Slime, Tinker]
       C -- saved checkpoint --> Q
       Q -- status --> T
       T -- evaluate, select --> E[Candidate evaluation]
       T -- publish snapshot --> R[Artifact repository]
       R -- activate, restore --> I[InferenceRuntime]
       I --> S[Serving engines or provider]

Reef submits jobs and reads their status. It does not drive optimizer steps. Only the training service builds a training component. Control and every other service hold the backend's facts (`Backend facts`_) and reach the component through the work queue (`Control and the training service`_). `Serving runtimes <serving-runtimes.rst>`__ covers the inference side, and `Executors <executors.rst>`__ the worker launch.

The deployment definition
-------------------------

``training.backend`` selects one ``TrainingDeployment``
(``reef.train.deployment``). The built-in names are ``slime`` (the default),
``tinker`` and ``spindle_full``. A definition describes the deployment. It
must not download models, allocate devices or build a component.

- ``prepare(config, settings)``: required. Validate the backend's
  combinations and bind derived values in ``config``. Return the process
  definitions that ``reef serve`` waits for, such as Slime's optional
  teacher engine; an empty tuple for an in-process backend.
- ``component_config(settings, *, default_parameters)``: required. Return
  the config of the training component. Its ``type`` is the dotted
  reference of a ``TrainingComponentFactory``. ``default_parameters`` is the
  recipe's job parameters by sample kind, or empty before a recipe is
  selected.
- ``http_environment(settings)``: variables for the service process. A value
  can name ``${endpoints.<service>}`` of a service that ``prepare``
  returned. Default: none.
- ``trainer_serving_config(settings)``: the runtime config of the backend's
  own serving (``inference.serving: trainer``), such as Tinker's sampler.
  Default: ``None``.
- ``engine_weights(settings)``: how Reef's engines get weights for snapshots
  that hold none. Default: ``None``.

Class attributes:

- ``requires_local_model`` (default ``True``): a hosted integration sets
  ``False``, so that deployment resolution keeps a remote model identifier.
- ``supports_teacher_engine`` (default ``False``): whether the backend
  accepts a ``teacher`` section.
- ``default_serving`` (default ``"engines"``): the serving of a deployment
  that sets no ``inference.serving`` and no engine setting; ``engines`` or
  ``trainer``.

Only the training service builds the component that the factory returns
(see `Backend facts`_). The backend does not build
serving: the ``inference`` section selects it, and the component's
``artifact_format`` tells the serving runtime what each snapshot holds (see
`Serving saved weights <serving-runtimes.rst#serving-saved-weights>`__).

An in-process backend
~~~~~~~~~~~~~~~~~~~~~

An integration whose component trains in the service's process reuses the
supplied definition:

.. code:: python

   from reef.train.deployment import InProcessTrainingDeployment


   class Deployment(InProcessTrainingDeployment):
       component_type = "my_backend.runtime:factory"

``factory`` is a ``TrainingComponentFactory`` instance. Its ``config_type()``
declarations validate ``training.options``. Import execution libraries only
when the factory builds the component. The base ``prepare`` refuses Ray and
training executor settings, and ``training.options`` cannot override the
managed fields (``type``, ``model_path``, the timeouts, ``connect``).

Select the backend
~~~~~~~~~~~~~~~~~~

Select the class directly with ``--training.backend
my_backend.launch:Deployment``, or register an installed name:

.. code:: toml

   [project.entry-points."reef.training_backends"]
   my-backend = "my_backend.launch:Deployment"

This enables ``--training.backend my-backend``. Reef imports only the
selected definition. A missing or ambiguous name fails; Reef does not fall
back to Slime.

The component contract
----------------------

A ``TrainingJob`` is one bounded job. It has:

- a job ID and the backend name
- the start, which the component made, such as a saved checkpoint's ``path``
- the job data directory that it trains (``data_dir``, and its URI ``data_uri``; see `The job data directory`_)
- the backend's parameters, which Reef does not read
- the scenario whose trainer submits it

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Contract
   * - ``name``
     - The backend name that jobs carry.
   * - ``submit(job)``
     - Record the job durably and return; the job runs in the background. Submitting a job ID again does nothing. Raise ``ValueError`` for a job that the component refuses.
   * - ``check_job(job)``
     - Raise ``ValueError`` for a job that the component cannot train. The training service calls it before control adds the job to the queue, so a refused job consumes no sample.
   * - ``status(job_id)``
     - Report ``running``, ``finished`` or ``failed``, the checkpoints saved so far and the backend's metrics. Raise ``KeyError`` for an unknown job. A restarted component resumes each unfinished job or reports it failed.
   * - ``cancel(job_id)``
     - Stop the job; it then reports ``failed``.
   * - ``import_start(artifact_dir)``
     - Copy a published snapshot into the component's own storage and return it as a start. The training service calls it when it takes a job whose start names a snapshot. Raise ``ValueError`` when the component cannot start from those weights. The same snapshot must give the same start again.
   * - ``root_start(root)``
     - The start of a job at a scenario's root. It must accept a root that holds the files of the facts' ``default_start()``. The default is ``import_start`` of the root's files.
   * - ``changed_parts(start, checkpoint)``
     - The parts of the scenario component that the checkpoint changed from ``start``, which the activation rule reads. The default is none; the harness component diffs the trees.
   * - ``check_remote_artifact(names)``
     - Raise ``ValueError`` unless what ``names`` names resolves on the component's service. ``names`` is what the facts' ``check_artifact`` returned for an import. A backend whose facts return ``None`` does not implement it.
   * - ``start_jobs()``
     - Start running jobs in this process and resume the jobs that have not ended. Only the training service's job worker calls it. A component that keeps its job state on disk, such as Tinker's ``state_dir``, locks it here, so one process at a time runs its jobs.
   * - ``shares_scenarios``
     - Whether jobs of several scenarios can run on the component. The default is false.
   * - ``concurrent_jobs``
     - Whether a job depends only on its own start, data and parameters, so that several jobs of one scenario can run together. The default is false.

Each ``SavedCheckpoint`` has a ``path`` (what a later job starts from) and an ``artifact_path`` (the weights that serving loads, without training state). Both are local directories of the training service. The training worker stores the files of ``artifact_path`` in the artifact backend and records the content's name in the job's item. Control commits the checkpoint by that name.

A component never learns the scenario. Per-scenario state, such as one adapter per scenario, belongs to serving. Two scenarios share a component only with ``shares_scenarios``:

- The Tinker component allows it, because each job carries its own start.
- The Slime component does not: its workers hold one set of weights, so a Slime deployment trains one scenario.

The inference runtime must also serve each scenario its own weights.

**A base class for local jobs.** ``reef.train.local_jobs.LocalJobComponent`` is the base of the built-in components. It keeps each job in a directory under the component's job root: ``job.json``, ``checkpoints.json`` and, once the job ended, ``result.json``. It runs jobs one at a time in a thread; with ``concurrent_jobs``, every submitted job runs in its own thread. A subclass implements ``check_job`` and ``run``, and calls ``start_jobs()`` once it can train. Until then no job runs. After a restart, ``start_jobs()`` resumes the unfinished jobs in submission order.

`Tinker integration <../reference/python-api-training.rst#tinker-integration>`__ in the Python API describes the Tinker component.

Backend facts
-------------

A service without a training component still needs some answers about it, and a new scenario needs its root before any job runs. A backend states both from its configuration as a ``reef.train.component.TrainingBackendFacts`` value. ``TrainingComponentFactory.facts(config, model_path, environ)`` returns it, and every factory must define it. ``training_backend_facts(config, model_path=...)`` calls it.

Stating the facts builds no component, starts nothing, calls no service and imports no training framework or SDK.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Member
     - Contract
   * - ``name``, ``artifact_format``, ``shares_scenarios``, ``concurrent_jobs``
     - The component's members of the same names.
   * - ``default_start_inputs``
     - The ``inputs`` of the default start: the model and the settings that decide its content.
   * - ``default_start()``
     - The content of a new scenario's root, as a ``DefaultStart``: a few small text files and ``default_start_inputs``. It holds no weight file. A backend that trains a base model writes a base root (``base.json``, ``reef.artifact.base_root``) or a reference to the base model's commit (``hf-reference.json``).
   * - ``parameters_schema(kind)``
     - The JSON Schema of a job's parameters, or ``None``.
   * - ``start_files``
     - The paths of the only files of a start that the next two checks read, such as ``("base.json", "tinker-checkpoint.json")``. ``None`` when they read the whole component, as the harness checks do. Control makes only these files local before a check, so a check fetches no weight file.
   * - ``check_root_start(root)``
     - Raise ``ValueError`` for a scenario root that the component's ``root_start`` refuses. It reads the root's ``start_files`` and writes nothing.
   * - ``check_import_start(artifact_dir)``
     - Raise ``ValueError`` for a published snapshot that the component's ``import_start`` refuses. It reads the ``start_files`` and writes nothing. Control calls it for a ``train-from`` and for a request's ``train_from``.
   * - ``check_artifact(files)``
     - Check the files of an import against the layout of the backend's snapshots, as far as the files and the settings decide. Return ``None`` when that is the whole check. Return a JSON object when something must also resolve on the backend's service; the component's ``check_remote_artifact`` gets it. The default checks ``artifact_format`` and returns ``None``.

The component holds a value of the same type and answers these members from it, so the facts and the component give the same answers. The facts are the one place that states the default start: Reef asks no component for a root.

Everything else stays a call to the component in the training service, because it needs the backend's service, its files or its framework: ``root_start``, ``import_start``, ``check_remote_artifact``, ``check_task_jobs``, ``check_job``, ``changed_parts`` and the jobs.

The harness component has no factory. ``HarnessFacts.from_settings`` makes its facts from the ``HarnessSettings``.

How a trainer runs jobs
-----------------------

A weight trainer has one training component and drives it through ``reef.train.jobs.JobLoop``. The trainer stores its state, a ``JobLoopState`` (``start``, ``jobs``, ``finished_jobs``, ``last_job``), in the scenario's commit record.

#. **Reserve.** When no job runs, the trainer reserves ready job data. The pipeline's sample-to-job stage gives the job's parameters, by the kind of its samples.
#. **Submit.** The job ID is a hash of the scenario, the backend, the start, the records of the job's data and the parameters, so a resubmission after a crash names the same job. Reef consumes those records when it submits the job.
#. **Poll.** The scenario's training worker reads the job's status every 2 seconds. Each new checkpoint is a candidate. The recipe's candidate evaluation decides whether Reef publishes it; the default selects every checkpoint.
#. **End.** A failed job ends like a finished one. Its records do not train again.

The start of a job follows these rules:

- A job continues the trainer's training start unless its request names another start with ``train_from``. It moves the training start to each checkpoint that it saves, whether or not evaluation selected the checkpoint. The trainer runs one such job at a time. Evaluation decides only what is served.
- A job with its own start begins from that artifact and never moves the training start. With ``concurrent_jobs``, the trainer submits it while other jobs of its scenario run. Without it, the trainer runs one job at a time.
- For a ``train-from``, control first asks the facts (``check_import_start``) and cancels the running job that continues the old start. It then writes the ``train_from`` commit, and the training start is the name of the snapshot. The training service imports the snapshot with ``import_start`` when it takes the next job that names no start. A backend that cannot load the snapshot refuses it in the facts, and Reef answers HTTP 400.
- Activating an artifact changes only what serves. The component never sees it.

Control and the training service
--------------------------------

Control's trainers see ``reef.train.job_queue.JobQueueComponent``, which holds the facts and the work queue, and no component. A one-process deployment takes the same path: its dispatcher runs the training service's workers beside control. An outside backend gets all of this from its component and its facts, and adds nothing to control.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Operation
     - Path through the queue
   * - Job check
     - ``submit`` adds a check to the ``training_checks`` queue and raises ``JobCheckPendingError``. The trainer releases the job's data and builds the job again at its next poll; the same data names the same job, so the poll finds the check. Nothing is locked or reserved while the answer is missing, and no sample is consumed. ``TrainingCheckWorker`` answers with ``check_job``. A refusal comes back as the component's ``ValueError``.
   * - Job
     - An accepted job enters the ``jobs`` queue, and ``TrainingJobWorker`` runs it. A job item that does not parse is a failed job with that error; the trainer goes on.
   * - Start
     - ``JobLoopState.start`` and a job's start can name a snapshot of the scenario (``SnapshotStart``): the root of a new trainer, or the artifact of a ``train-from``. The training service reads the snapshot from the artifact store and makes the component's start of it (``resolved_job``).
   * - Import
     - The layout of the weight component (``TrainingArtifactLayout``) runs the facts' ``check_artifact``. When the facts return names, control adds an ``artifact`` check and waits up to 60 seconds. A refusal is HTTP 400. Without an answer, the import ends with HTTP 503.
   * - Checkpoint
     - The training worker records the parts that each checkpoint changed (``changed_parts``) in the job's progress, so control reads them from the queue.

A check that could not run, for example because the training service cannot reach the artifact store, ends after its attempts with an error that says so. A request's job then fails the request, and its samples stay untrained. The check worker answers no check that control did not ask for within 10 minutes. Control's sweep deletes ended checks with the other ended work.

The job data directory
----------------------

A training component does not get a job's data as an object. Before Reef
submits a job, it writes the job's data as a directory. The job holds a
reference to it (``JobDataRef``), and the component reads the directory.

.. code:: text

   job.json        {"format": 2, "key", "sample_count", "kind"}
   samples.jsonl   one sample per line, in job order

- ``job.json`` gives the directory's key, its number of samples, and the
  kind of its samples: ``trajectory``, ``task``, or ``null`` without
  samples.
- Each line of ``samples.jsonl`` is the entry of ``JobData.to_dict``, with
  one difference: the line of a task has no ``task_path``. Its ``files``
  names the task's directory in the task file store
  (``storage.task_file_backend``): ``partition``, ``digest`` and ``name``.
- The directory holds no file of a task and no training request.

How a component reads it:

- A component that trains on Reef's host reads ``job.read_data()``, or the
  directory at ``job.data_dir``. The read makes the local copy. For a job
  of tasks it also fetches each task's directory, which is the sample's
  ``task_path``.
- A component that runs the job elsewhere, such as in a container, gives
  that place ``job.data_uri`` and mounts or copies the directory there.
  That place reads ``files`` from each task line and gets the directory
  from the task file store. Such a component must not read
  ``job.data_dir`` or ``job.read_data()``.

Rules:

- A directory does not change after Reef writes it, and several jobs can
  read it. A component must not write in it, and must not change a task's
  directory.
- A missing directory or task fails the read.
- Every task of a job must be in the task file store. Reef refuses job data
  with a task that has no ``files``.
- The key of a directory is the SHA-256 of its lines. A job of the same
  samples reuses the directory that exists.
- Reef deletes a directory that no job used for
  ``storage.job_directory_ttl_s`` (7 days by default). Keep this time above
  the longest job: a component can read the directory again when it resumes
  the job.

Decide what a scenario trains
-----------------------------

In ``auto`` mode, a trainer trains when ``job_samples`` untrained samples wait. A deployment can replace this rule with a training manager: a ``TrainingManager`` plugin that ``training.manager`` names. An agent that plans experiments is one use.

Control asks the manager every 2 seconds, once for each trainer of each loaded scenario in ``auto`` mode. ``actions(view)`` reads a ``ManagerView`` and returns the actions that Reef runs, in order:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Action
     - Effect
   * - ``Train(samples, reason, parameters, train_from)``
     - Files a training request, as ``POST /reef/train`` does. With ``train_from``, the job starts from that artifact and runs beside the other jobs. Without it, the job continues the training start.
   * - ``TrainFrom(artifact_id)``
     - Sets the training start.
   * - ``Cancel(job_id)``
     - Cancels a job.

The view holds the training start, whether the selected artifact serves, the untrained samples that an automatic job would train next, the job size, the number of waiting requests, the jobs that have not ended, and the artifact listing with each artifact's metrics.

Reef limits a manager in three ways:

- It checks each action as it checks a person's request, because the view can be old when the action runs.
- It drops a ``Train`` action while ``job_limit`` jobs of the manager's requests wait or run. The default limit is 1.
- A manager has no action that changes what a scenario serves. The activation rule decides that.

A manager's request has the session ``training-manager``, and the commit of its job keeps the action's ``reason`` as the request's text. The manager runs on one thread of control, outside every scenario lock, so a slow decision delays only the other managed trainers. An error in ``actions``, or a refused action, is logged, and the next round reads the trainer again.

Slime integration
-----------------

``reef.train.slime_backend.component.SlimeTrainingComponent`` runs in the training service's process and holds no GPU. It parses Slime's arguments once at startup. For each job, it starts a driver process (``reef.train.slime_backend.driver``) with the job's directory.

**Drivers and the serving session.** The driver owns one Ray session and runs ``SlimeJobRunner``: Slime's actor (and critic) workers and, for a dataset job or a task job, Slime's own rollout engines (``reef.train.slime_backend.rollout_group``).

- The engines belong to Slime's serving session, a detached ``ServingCluster`` named after the save root. The first such job's driver creates it, and later drivers attach through Slime's ``create_rollout_manager``.
- The session holds the engines, their router and the placement group of the training GPUs. While it runs, every driver trains on its placements. A driver detaches when it exits, also after a failure.
- Without a session, a driver places its job with Slime's own placement and starts no engines.
- The session's actors run under a process lease of their own, recorded in the save root's ``reef-serving.json``, so a driver's process cleanup leaves them running. The component disposes the session when it closes (``driver --dispose``).
- Serving never shares GPUs with training. With ``--colocate``, the rollout engines share the training GPUs, and Slime moves the memory between them.

A driver records each saved checkpoint in the job's directory, where the component reports it, and releases its workers and GPUs when it exits. A cancel or a close stops the driver. A restarted component starts the job's driver again.

**Rollouts and checkpoints.** A job trains its data as Slime rollouts (a ``rollout_id`` is Slime's unit of training):

- Without the job parameter ``save_interval``, the job is one rollout. With it, each run of ``save_interval`` optimizer steps is one rollout.
- After each rollout, the component saves the paired Hugging Face and Megatron checkpoints. A checkpoint's step is the job's optimizer steps up to it. The snapshot is the Hugging Face export, or with LoRA the PEFT adapter.
- The job's ``objective`` parameter selects the loss family, and ``learning_rate_schedule`` the learning-rate schedule. The actor workers switch to them when they receive the job's data.
- The job's last checkpoint records the objective's state after the job's data. An earlier checkpoint records the state of the job's start, because the objective computes its state for the whole data. When the objective skips the job's data, the job saves no checkpoint, so the next job continues the start's weights and objective state. The Tinker component does the same.
- The optimizer state carries from one job to the next in the Megatron checkpoint.

**Starts.** Every driver loads Slime's latest checkpoint, so a job starts from the component's latest checkpoint. Because the jobs continue one save root, a Slime deployment trains one scenario.

- A scenario trains on Slime only when its root came from the default start of the configured inputs (``root_start``). The default start is a base root. The workers make the startup weights from the model, so ``root_start`` reads the root's source and none of its files.
- ``import_start`` refuses every snapshot, so a Slime scenario refuses ``train-from``.
- The component checks the start when Reef submits the job, and the driver again when the job runs. A queued job whose start an earlier job moved past fails.
- The component refuses to start when its options, as written, state other facts than Slime's parser read from them.

**Restarts.** A restarted job continues after its last recorded checkpoint. When a restart stopped a job after Megatron saved a rollout but before the component recorded it, the job records that checkpoint and does not train the rollout's data twice.

`Slime worker executors <executors.rst#slime-worker-executors>`__ describes how the component launches its workers.
