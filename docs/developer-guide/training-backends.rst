Training backends
=================

Implement a training backend when adding a framework or hosted training service.
Weight-method authors normally need only an
`objective <recipe-training-service.rst>`__.

``TrainingDeployment`` defines startup; ``TrainingComponent`` runs jobs;
``TrainingBackendFacts`` describes capabilities without starting training.
Only the training service builds components.

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

Optional methods and capability flags declare serving, environment and
model requirements. See `TrainingDeployment <../../reef/train/deployment.py>`__.
Inference settings select serving independently.

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
   * - ``start_jobs()``
     - Start or resume jobs. Acquire exclusive ownership of persistent job state here.

``shares_scenarios`` and ``concurrent_jobs`` default to false.
Enable them only when jobs remain independent across scenarios or starting
checkpoints respectively. Serving must also preserve per-scenario weights.

``SavedCheckpoint.path`` identifies the next training start;
``artifact_path`` contains publishable weights without training state.
Do not modify reported artifacts. ``step`` is nonnegative within the job;
metrics must encode as JSON. Control receives stored references, not local paths.

``LocalJobComponent`` supplies durable directories and background execution.
Implement ``check_job`` and ``run``; jobs start or resume through ``start_jobs``.
See `component contracts <../../reef/train/component.py>`__ for optional hooks.

Backend facts
-------------

Every factory must implement
``facts(config, model_path, environ) -> TrainingBackendFacts``.
This call must build no component, start no work, contact no service and
import no training framework or SDK.

Facts must agree with the live component on name, artifact format,
scenario sharing and concurrency. They define default roots, parameter schemas
and read-only start/import checks.

``default_start`` returns small text files and their determining inputs,
without downloading weights. ``start_files`` narrows validation reads.
Reject unsupported starts with ``ValueError``.
``check_artifact`` returns ``None`` when validation finishes locally, or a
JSON object for the component's ``check_remote_artifact``.

Facts alone define default roots. Service-dependent work stays on the component.
See `TrainingBackendFacts <../../reef/train/component.py>`__ for complete signatures.

How a trainer runs jobs
-----------------------

Control reserves samples, checks and submits a deterministic job ID, then
commits each saved checkpoint and the job's final status.
Submission consumes samples; failed jobs do not return them.
The activation rule decides whether a committed checkpoint serves.

Only one job advances the shared training start at a time.
Per-request ``train_from`` jobs never advance it; ``concurrent_jobs`` allows
them to overlap. Scenario-level train-from cancels the continuing job and
sets a validated new start. Activation never resets training.

Control and the training service
--------------------------------

Control uses ``JobQueueComponent`` in both split and single-process deployments.
Backend authors implement the component and facts; they add no control-side worker.

Pending job checks consume no samples. Accepted jobs enter the queue;
training resolves their starts and reports checkpoints through queue progress.
Remote artifact checks return HTTP 400 on refusal and 503 on timeout.

The job data directory
----------------------

``TrainingJob`` holds a ``JobDataRef`` for immutable data:

.. code:: text

   job.json        format 2, key, sample_count, kind
   samples.jsonl   one sample per line, in job order

The kind is ``trajectory``, ``task`` or null.
Task entries reference stored files; the directory contains neither task files
nor the training request.

Host components use ``job.read_data()`` or ``job.data_dir``.
Remote components use ``job.data_uri`` and fetch task files separately;
never access host-local paths remotely.
Directories are immutable. Missing files fail reads.
Keep ``storage.job_directory_ttl_s`` above the longest job, including restart reads.

Decide what a scenario trains
-----------------------------

A ``TrainingManager`` selected by ``training.manager`` replaces automatic
sample-count scheduling. Control calls ``actions(view)`` every two seconds
for each loaded automatic trainer.

Return ordered ``Train``, ``TrainFrom`` or ``Cancel`` actions from the current
``ManagerView``. Reef revalidates them because views can become stale.
``job_limit`` bounds waiting and running manager requests; default 1.
Managers cannot change serving selection.
Keep decisions short: they share one thread. Errors retry on the next round.
See `manager contracts <../../reef/train/manager.py>`__.

Slime integration
-----------------

Slime owns training workers and task/dataset rollout engines.
``colocate`` shares their GPUs, never production inference GPUs.

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

Restart resumes the last saved checkpoint without repeating its rollout.
See `Slime worker executors <executors.rst#slime-worker-executors>`__ for launch contracts.
