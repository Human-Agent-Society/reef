Training backends
=================

Add a backend for a framework or hosted training service.
Method authors normally need only an `objective <recipe-training-service.rst>`__.

Deployment and factory
----------------------

``TrainingDeployment`` describes startup without downloading models or allocating devices.
``prepare(config, settings)`` validates settings and returns service definitions.
``component_config(settings, *, default_parameters)`` names a ``TrainingComponentFactory``.
Only the training service builds its component.

For in-process backends:

.. code:: python

   from reef.train.deployment import InProcessTrainingDeployment


   class Deployment(InProcessTrainingDeployment):
       component_type = "my_backend.runtime:factory"

``factory`` is a ``TrainingComponentFactory`` instance.
Its ``config_type()`` validates ``training.options``.
Import execution dependencies only when building the component.
This deployment rejects training executor settings and overrides of managed fields.

Select ``my_backend.launch:Deployment`` in ``training.backend``, or register
it under the ``reef.training_backends`` entry-point group.
Unknown or ambiguous names fail without fallback.
See `deployment contracts <../../reef/train/deployment.py>`__ for optional capabilities.

The component contract
----------------------

``TrainingComponent`` owns the training loop:

- ``check_job(job)`` rejects unsupported jobs with ``ValueError`` before sample consumption.
- ``submit(job)`` durably starts asynchronous work; duplicate IDs are no-ops.
- ``status(job_id)`` returns running/finished/failed, checkpoints and metrics.
  Unknown IDs raise ``KeyError``. Restart must resume or fail unfinished jobs.
- ``cancel(job_id)`` stops work and reports failure.
- ``import_start(directory)`` returns a stable start for compatible weights.
- ``root_start(root)`` resolves initial training state.
- ``start_jobs()`` acquires exclusive persistent-state ownership and starts/resumes work.

``shares_scenarios`` and ``concurrent_jobs`` default false.
Enable them only with independent scenario state and independent starting checkpoints.

``SavedCheckpoint.path`` names the next training start;
``artifact_path`` contains publishable weights without training state.
Reported artifacts are immutable. ``step`` is nonnegative within the job;
metrics must encode as JSON.
``LocalJobComponent`` supplies persistence and background execution; implement ``check_job`` and ``run``.

Backend facts
-------------

Factories implement ``facts(config, model_path, environ)`` without components,
network calls or training-framework imports.
Facts must match live capabilities and define default roots, schemas and read-only import checks.

``default_start`` returns small files and determining inputs without weights.
``start_files`` narrows validation reads. Unsupported starts raise ``ValueError``.
``check_artifact`` returns ``None`` for local acceptance or JSON for the component's remote check.
See `TrainingBackendFacts <../../reef/train/component.py>`__.

Jobs and data
-------------

Control submits through queues and commits every saved checkpoint.
Submission consumes samples; failure does not return them.
Activation selects serving independently of training state.
Only one job advances the shared start; per-request train-from jobs never advance it.

Use ``job.read_data()`` locally or transfer ``job.data_uri`` remotely.
Fetch referenced task files separately; never use host-local paths remotely.
Job/task directories are immutable. Retention must cover the longest job and restart reads.

Decide what a scenario trains
-----------------------------

A ``TrainingManager.actions(view)`` replaces sample-count scheduling.
Return ordered ``Train``, ``TrainFrom`` or ``Cancel`` actions.
Reef revalidates stale views; managers cannot change serving.
``job_limit`` defaults to one waiting/running request.
Keep decisions short because managers share a thread.
See `manager contracts <../../reef/train/manager.py>`__.

Slime constraints
-----------------

Slime supports one scenario and one continuing save root; it rejects train-from.
Sample jobs count optimizer steps in ``save_interval`` and checkpoint steps.
Task and dataset jobs count rollout iterations.
Only the final checkpoint retains next objective state; skipped jobs retain no new state.
``colocate`` shares training/rollout GPUs, never production inference GPUs.
