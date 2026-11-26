Training backends
=================

Implement a training backend when adding a framework or hosted training service.
Weight-method authors normally need only an
`objective <recipe-training-service.rst>`__.

A backend has three parts: ``TrainingDeployment`` describes startup,
``TrainingComponent`` runs jobs, and ``TrainingBackendFacts`` describes its
capabilities without starting training. Only the training service builds
the component. Control submits jobs and reads results through queues.

The deployment definition
-------------------------

``training.backend`` selects a ``TrainingDeployment`` from
``reef.train.deployment``. Built-in names are ``slime`` (default),
``tinker`` and ``spindle_full``.
The definition must not download models, allocate devices or build components.

Required methods:

- ``prepare(config, settings)`` validates combinations, binds derived settings
  and returns service process definitions. In-process backends return an empty tuple.
- ``component_config(settings, *, default_parameters)`` returns component
  configuration whose ``type`` references a ``TrainingComponentFactory``.
  Parameters are grouped by sample kind and may be empty before recipe selection.

Optional methods provide service environment variables (``http_environment``),
trainer-owned serving configuration (``trainer_serving_config``) and engine
weights for snapshots without files (``engine_weights``).

Class attributes declare ``requires_local_model`` (default true),
``supports_teacher_engine`` (default false) and ``default_serving``
(``engines`` by default, or ``trainer``).
The inference configuration selects serving independently.

An in-process backend
~~~~~~~~~~~~~~~~~~~~~

Reuse ``InProcessTrainingDeployment`` for a component that trains in the
service process:

.. code:: python

   from reef.train.deployment import InProcessTrainingDeployment


   class Deployment(InProcessTrainingDeployment):
       component_type = "my_backend.runtime:factory"

``factory`` is a ``TrainingComponentFactory`` instance.
Its ``config_type()`` validates ``training.options``.
Import execution libraries only when building the component.
This deployment rejects Ray/training executor settings and overrides of
managed fields such as type, model path, timeouts and connect settings.

Select the backend
~~~~~~~~~~~~~~~~~~

Select ``my_backend.launch:Deployment`` directly, or register a name:

.. code:: toml

   [project.entry-points."reef.training_backends"]
   my-backend = "my_backend.launch:Deployment"

Then set ``training.backend: my-backend``.
Missing or ambiguous names fail without falling back to Slime.

The component contract
----------------------

A ``TrainingJob`` carries its ID, backend, starting checkpoint, data reference,
parameters and scenario. The component owns the framework's training loop.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Contract
   * - ``name``
     - Backend name carried by jobs.
   * - ``check_job(job)``
     - Raise ``ValueError`` for unsupported jobs before samples are consumed.
   * - ``submit(job)``
     - Durably record and run the job asynchronously. Duplicate IDs are no-ops.
       Raise ``ValueError`` for refused jobs.
   * - ``status(job_id)``
     - Return ``running``, ``finished`` or ``failed``, checkpoints and metrics.
       Unknown jobs raise ``KeyError``. Restart must resume unfinished jobs or fail them.
   * - ``cancel(job_id)``
     - Stop the job and report it failed.
   * - ``import_start(artifact_dir)``
     - Copy a snapshot into backend storage and return a stable start identifier.
       The same snapshot must return the same start; incompatible weights raise ``ValueError``.
   * - ``root_start(root)``
     - Resolve a scenario root, including the facts' default start.
       By default, import its files.
   * - ``changed_parts(start, checkpoint)``
     - Report changed component parts for activation. Default: none.
   * - ``check_remote_artifact(names)``
     - Validate names returned by the facts' import check on the training service.
   * - ``start_jobs()``
     - Start or resume jobs. Acquire exclusive ownership of persistent job state here.

``shares_scenarios`` and ``concurrent_jobs`` default to false.
Enable them only when jobs remain independent across scenarios or starting
checkpoints respectively. Serving must also preserve per-scenario weights.

``SavedCheckpoint.path`` identifies the next training start.
``artifact_path`` contains publishable weights without training state.
Both are local to training. Do not change artifact files after reporting
a checkpoint. Its ``step`` is nonnegative and counts within the job;
metrics must encode as JSON. The worker stores artifact files and sends their
content reference to control; control does not read training-local paths.

``LocalJobComponent`` provides durable per-job directories and background
threads. Implement ``check_job`` and ``run``. Jobs wait for ``start_jobs``;
restart resumes unfinished jobs in submission order.
Concurrent components use one thread per job.

Backend facts
-------------

Every factory must implement
``facts(config, model_path, environ) -> TrainingBackendFacts``.
This call must build no component, start no work, contact no service and
import no training framework or SDK.

Facts and the live component must agree on backend name, artifact format,
scenario sharing and concurrency. Facts also define:

- ``default_start_inputs``: settings that determine root content.
- ``default_start()``: small text files plus those inputs, without weights.
  Common forms are ``base.json`` or ``hf-reference.json``.
- ``parameters_schema(kind)``: parameter JSON Schema, or ``None``.
- ``start_files``: files needed for start validation; ``None`` requests
  the whole component. Declare narrow paths to avoid fetching weights.
- ``check_root_start(root)`` and ``check_import_start(directory)``:
  read-only checks matching the component's acceptance rules.
  Raise ``ValueError`` for unsupported starts.
- ``check_artifact(files)``: validate an import locally.
  Return ``None`` when complete, or a JSON object for
  ``check_remote_artifact`` to resolve on training.

Facts are the sole source of default roots.
Service-dependent operations remain on the component.
Harness training uses ``HarnessFacts.from_settings`` instead of a factory.

How a trainer runs jobs
-----------------------

Control reserves samples, checks and submits a deterministic job ID, then
commits each saved checkpoint and the job's final status.
Submission consumes samples; failed jobs do not return them.
The activation rule decides whether a committed checkpoint serves.

A normal job advances the training start with each checkpoint.
Only one job can continue that start at a time.
A request with ``train_from`` uses its own start and never advances the
shared training start. ``concurrent_jobs`` permits these independent jobs to overlap.

A scenario-level train-from validates the snapshot, cancels the job continuing
the old start and commits the new start.
Training imports its files when taking the next job.
Unsupported starts return HTTP 400.
Activation changes serving only and never resets training.

Control and the training service
--------------------------------

Control uses ``JobQueueComponent`` in both split and single-process deployments.
Backend authors implement the component and facts; they add no control-side worker.

Job checks run before submission. While a check is pending, control releases
the reservation and consumes no samples. Refusal returns the component's error.
Accepted jobs enter the jobs queue; training resolves snapshot starts and
reports checkpoints through queue progress.

Imports can require a remote artifact check. Control waits up to 60 seconds:
refusal returns HTTP 400, and no answer returns HTTP 503.
Checks that cannot run eventually fail. Request samples remain untrained.
Checks no longer requested for ten minutes are not executed.

The job data directory
----------------------

``TrainingJob`` holds a ``JobDataRef`` for immutable data:

.. code:: text

   job.json        format 2, key, sample_count, kind
   samples.jsonl   one sample per line, in job order

The kind is ``trajectory``, ``task`` or null for no samples.
Task lines name stored files by partition, digest and name; they contain
no local ``task_path``. The directory includes neither task files nor the
training request.

Host components use ``job.read_data()`` or ``job.data_dir``.
These materialize data and fetch task directories.
Remote components pass ``job.data_uri``, transfer or mount the directory,
and fetch task files from their store. They must not access host-local data paths.

Do not modify job or task directories. Missing files fail reads;
tasks without stored file references are refused.
Directory keys are SHA-256 hashes of their lines, so identical data is reused.
``storage.job_directory_ttl_s`` defaults to seven days.
Keep it above the longest job, including restart reads.

Decide what a scenario trains
-----------------------------

A ``TrainingManager`` selected by ``training.manager`` replaces automatic
sample-count scheduling. Control calls ``actions(view)`` every two seconds
for each loaded automatic trainer.

``ManagerView`` includes the training start, serving readiness, available
samples, job size, waiting requests, active jobs and artifact metrics.
Return actions in execution order:

- ``Train(samples, reason, parameters, train_from)`` files a training request.
- ``TrainFrom(artifact_id)`` changes the training start.
- ``Cancel(job_id)`` cancels a job.

Reef validates actions again because views can become stale.
``job_limit`` defaults to 1 and bounds manager requests waiting or running.
Managers cannot change serving selection. Requests use session
``training-manager`` and retain ``reason`` as their text.

Decisions run outside scenario locks on one control thread.
Slow decisions delay other managed trainers. Errors are logged and retried
on the next scheduling round.

Slime integration
-----------------

Slime's component launches a driver per job and holds no GPU itself.
Drivers own training workers. Dataset/task rollout engines can persist across
jobs in a shared serving session; component closure disposes it.
Production inference never shares training GPUs. Slime ``colocate`` applies
to its own rollout engines.

For a sample job, each ``save_interval`` optimizer steps form one Slime rollout.
Without this parameter, the sample job is one rollout.
Task and dataset jobs run Slime's own loop; their ``save_interval`` counts rollout iterations.
Checkpoint steps count optimizer steps for sample jobs and rollout iterations for task and dataset jobs.
Saved checkpoints pair publishable Hugging Face weights or PEFT adapters
with Megatron training state.

The final checkpoint stores the objective's next state; intermediate ones
retain its starting state. Skipped jobs save no checkpoint and retain the
starting weights and objective state. Optimizer state continues between jobs.

Slime supports one scenario and one continuing save root.
Its root must match the configured default-start inputs.
It rejects ``import_start`` and train-from.
Queued jobs whose starts became obsolete fail before training.

Restart resumes from the last checkpoint. If Megatron saved before the
component recorded it, recovery records that checkpoint without retraining
its rollout. See `Slime worker executors
<executors.rst#slime-worker-executors>`__ for launch contracts.
