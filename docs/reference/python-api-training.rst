Training API
============

The contracts of the training service: the objective of a job, its step schedule and parameters, the training job and the training component that runs it, the job loop, and the Tinker integration.

The guide explains how to use them: `Training service <../developer-guide/recipe-training-service.rst>`__ for objectives and job parameters, `Training backends <../developer-guide/training-backends.rst>`__ for components, and `Loss families <../developer-guide/loss-families.rst>`__.

Training objective
------------------

Each training backend owns its objectives: ``SlimeObjective`` in ``reef.train.slime_backend.objectives`` and ``TinkerObjective`` in ``reef.train.tinker_backend.objectives``. An objective prepares the training signal over a job's full data and declares the backend's loss family. Each backend also has its own ``register_objective``, ``resolve_objective``, ``TrainingSignal``, ``next_steps`` and ``StepScheduling``; the examples here show Slime's.

`The objective <../developer-guide/recipe-training-service.rst#the-objective>`__ has an example.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Member
     - Contract
   * - ``name``, ``loss_family``
     - The name that a job's ``objective`` gives, and the name of a registered loss family of the same backend.
   * - ``prepare(data, state) -> TrainingSignal``
     - Required. Runs on the complete reserved job data, before optimizer or worker partitioning. Compute group-relative statistics here. It must not mutate the job data, execute training, or commit state. The class attributes ``rollout_unit`` and ``supports_multiple_epochs`` set the job's schedule (`Step scheduling`_). Keep it free of torch, Ray, Slime and Tinker imports. Model-dependent terms (critic-based GAE, reference-model KL) stay in the backend's loss implementation, after the required forward passes.

A job names its objective in its ``objective`` parameter, usually from the recipe's ``default_parameters()``:

- **A registered name.** A method package's public half calls ``register_objective(name, "my_method.slime:MyObjective")`` from the backend's ``objectives`` module. The call records the reference and imports nothing. The backend imports the module when a job first names the objective. The function also accepts an instance in place of the reference. Core never imports method packages.
- **A dotted reference**, ``my_method.slime:MyObjective``. It names a class with a zero-argument constructor. Its module must be importable in both the service and training processes.

Each backend has a built-in ``sft`` objective. Slime also has ``slime_dataset``. The selected backend resolves its own loss implementation lazily.

``TrainingSignal`` carries:

- ``action``: ``train`` runs backend training; ``skip`` commits a state-only transition.
- ``advantages``: optional per-trajectory values, in the job data's order.
- ``next_algorithm_state``: proposed state, committed through the trainer's existing step lifecycle.
- ``metrics``: method telemetry carried to the commit record.

Step scheduling
~~~~~~~~~~~~~~~

The training component cuts each job's data into optimizer steps. The backend's batch size (Slime's ``--global-batch-size``, Tinker's ``batch_size``) is the rollouts in one step, or the packs in one step when the job sets ``packing_length``. Two class attributes of the objective set what the component can do with the job's data:

- ``rollout_unit`` is the rows that one step keeps together. ``"sample"``, the default, makes each sample its own rollout. ``"comparison_set"`` keeps the samples that share a ``group_id`` in one step.
- ``supports_multiple_epochs`` (default ``False``) says whether passes after the first are valid. Those passes train off-policy against log-probs computed once, which a clipped ratio tolerates.

A job's optional ``scheduling`` parameter is a ``StepScheduling`` (``reef.train.slime_backend.schedule`` or ``reef.train.tinker_backend.schedule``) as an object:

.. container:: config

   ``epochs`` : 1
       the passes over the job's data, a positive int

   ``shuffle`` : ``false``
       reorder the rollouts in every pass, deterministically from the data's rows

   ``remainder`` : ``"partial"``
       a last step with fewer rollouts (or packs) than the batch size: ``"partial"`` trains it as a smaller step, ``"drop"`` leaves it out, ``"error"`` refuses the job's data

   ``packing_length`` : ``null``
       pack the rollouts into packs of at most this many tokens (``training["tokens"]`` of each trajectory) before the steps are cut; one step then trains batch-size packs. A pack never splits a rollout, and a longer rollout is a pack of its own. ``shuffle`` and ``remainder`` apply to the packs.

``pack_rollouts(token_counts, packing_length)`` of each backend's ``schedule`` module is the packing: best-fit decreasing, which puts each rollout, longest first, into the pack whose free tokens it fills most closely. The packs decide only which rollouts share a step. The loss and the micro-batches stay the component's.

Job parameters
~~~~~~~~~~~~~~

``WeightTrainingRecipe.parameters_by_kind()`` returns the parameters of each training job by the kind of the job's samples: ``trajectory`` or ``task``. By default, the parameters of both kinds are the recipe's ``default_parameters()`` with its ``job_parameters`` config over them, key by key, such as ``scheduling`` or ``save_interval``. The parameters are part of the job's ID, so ``parameters_by_kind()`` must return the same value at every call. `Job parameters <../developer-guide/recipe-training-service.rst#job-parameters>`__ shows a recipe that trains each kind with its own method.

The trainer gives the parameters to its pipeline (``PipelineContext.job_parameters``). When a job's data is ready, the sample-to-job stage gives the job's parameters, ``Sample2Job.job_parameters(data)``:

- They are those of the kind of the data's samples, with the request's ``parameters`` (``TrainingRequest.parameters``) merged in by ``reef.core.values.deep_merge``: nested objects merge key by key, and every other value of the request wins.
- The stage refuses data that holds both kinds, and a kind that the recipe does not name when no request gives parameters. A kind that the recipe does not name never trains in ``auto``.
- Reef reads none of the parameters. It submits them to the deployment's training component, whose name is the job's ``backend``.

Each component parses a job's parameters into its own typed value: ``reef.train.slime_backend.parameters.SlimeJobParameters`` and ``reef.train.tinker_backend.parameters.TinkerJobParameters``. Slime and Tinker parse when they accept a job and again when they run it. The parser raises ``ValueError``:

- for an unknown key;
- when ``objective`` is missing or does not resolve;
- when ``scheduling`` has another key or a value out of range;
- when the objective's ``validate_scheduling(scheduling)`` refuses the schedule. The default refuses ``epochs > 1`` without ``supports_multiple_epochs``.

``TrainingComponent.parameters_schema(kind)`` returns the JSON Schema of those parameters for a sample kind, or ``None``. The scenario contract serves it.

Training jobs and components
----------------------------

``reef.train.component`` defines bounded training jobs.

.. code:: python

   from reef.train.component import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob

Training jobs
~~~~~~~~~~~~~

A ``TrainingJob`` has these fields:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Value
   * - ``job_id``, ``backend``
     - The job's ID and the name of the training component.
   * - ``start``
     - The non-empty start: one that the component made, such as ``SavedCheckpoint.path``, or the component's start of a published snapshot. A new trainer starts at ``TrainingComponent.root_start``.
   * - ``data``
     - A ``JobDataRef``: a reference to the job data directory that the job trains. ``data_uri`` is its URI, and ``data_id``, ``sample_count`` and ``kind`` describe the data; these copy nothing. ``read_data()`` returns its samples and ``data_dir`` is its local path. With a remote store, the first read copies the directory, and the directories of its tasks, to the host.
   * - ``parameters``
     - The backend's parameters. Reef reads none of them. They override the backend's own configuration.
   * - ``scenario``
     - The scenario whose trainer submits the job. The jobs of one scenario run one at a time, in the order of their submission, unless the component has ``concurrent_jobs``.
   * - ``request_id``
     - The ID of the training request that the job trains, or ``None``.
   * - ``component``
     - The scenario component that the job trains, or ``None``.
   * - ``previous_job_id``
     - The last job whose end the trainer committed, or ``None`` before the first. A harness continues its state after that job, since a job that keeps no tree still advances it. A weight component ignores it.
   * - ``train_from_artifact_id``
     - The artifact that ``start`` came from, or ``None``. Reef's job listing reads it; a component does not need it.

``reef.train.jobs.job_id_for`` names a job by a hash of its backend, its start, its data and its parameters. The weight trainer puts the scenario and the records of the job's data into the data identity. A retry of the same request is therefore the same job.

A ``JobStatus`` has:

- ``state``: ``running``, ``finished`` or ``failed``;
- ``checkpoints``: the ``SavedCheckpoint`` values so far, in step order. A finished job without checkpoints trained nothing, for example because its objective skipped the job's data;
- ``metrics``: the backend's metrics;
- ``error``: set exactly when the job failed.

A ``SavedCheckpoint`` has:

- ``step``: its step within its job;
- ``path``: the start of a later job;
- ``artifact_path``: the directory that Reef publishes and serving loads. The component does not change it after it reports the checkpoint;
- ``metrics``: the backend's metrics.

Both paths are local directories of the process that runs the component. The training worker stores the files of ``artifact_path`` in the artifact backend (``RepositoryBackend.store_content``). It writes the checkpoint into the jobs queue with the name of that ``content`` in place of ``artifact_path``.

Training components
~~~~~~~~~~~~~~~~~~~

A ``TrainingComponent`` runs whole jobs with its framework's own training loop. It implements the members marked *required*:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Contract
   * - ``name`` (required)
     - The backend name that the component's jobs carry.
   * - ``artifact_format`` (required)
     - An ``ArtifactFormat``: the weight files that each saved checkpoint's snapshot holds. Its ``layout`` is ``full`` (a Hugging Face checkpoint) or ``adapter`` (a PEFT adapter in ``adapter/``, with ``adapter_rank`` and ``adapter_target_modules``). ``None`` when the snapshots hold no weight files that an engine loads, such as Tinker's manifests. The serving assembly configures the engines from it and refuses a serving runtime that cannot load it.
   * - ``submit(job)`` (required)
     - Record the job durably and return; the job runs in the background. Submitting a job ID again does nothing. Raise ``ValueError`` for a job that the component refuses and ``RuntimeError`` after ``close``; nothing is recorded then.
   * - ``status(job_id)`` (required)
     - The job's ``JobStatus``. Raise ``KeyError`` for a job that the component never saw. A restarted component resumes each unfinished job or reports it failed.
   * - ``cancel(job_id)`` (required)
     - Stop a running job, which then reports ``failed``. Nothing for an unknown or ended job.
   * - ``import_start(artifact_dir)`` (required)
     - Copy a published snapshot into the component's own storage and return it as a start. The training service calls it when it takes a job whose start names a snapshot, such as the first job after a ``train-from``. Raise ``ValueError`` when no job can start from those weights.
   * - ``root_start(root)``
     - The start of the first job of a trainer without committed state, a non-empty string. ``root`` is a ``ScenarioRoot``: the trainer's component of the scenario's root snapshot and the root's ``RootSource``. The default is ``import_start`` of the root's files. A root of a new scenario holds the facts' default start, and the job then starts at the base model. Tinker starts a new LoRA from its ``seed`` for a base root. With ``seed: null``, it saves one untrained adapter in ``state_dir`` when the first such job runs, and each such job restores it. Slime imports no snapshot. It returns its startup weights when the root's source is its default start with equal ``inputs``, and raises ``ValueError`` that names both otherwise.
   * - ``snapshot_start(artifact)``, ``start_files``
     - ``snapshot_start`` is the start of a job that trains from the trainer's component of a published snapshot. The default is ``import_start`` of its directory. ``start_files`` are the only paths that ``root_start`` and ``snapshot_start`` read, or ``None``, the default, for the whole component.
   * - ``shares_scenarios``, ``concurrent_jobs``
     - Whether jobs of several scenarios can run on the component, and whether several jobs of one scenario can run at the same time. Both default to ``False``. Return true only when a job depends on nothing but its own ``start``, data and parameters.
   * - ``check_job(job)``
     - Raise ``ValueError`` for a job that the component cannot train. The training service calls it before control adds the job to the queue. The default accepts every job.
   * - ``check_task_jobs(parameters)``
     - Raise ``ValueError`` unless the component can train jobs of ``TaskSample`` values with ``parameters``. The training service calls it at its start for a recipe whose jobs train tasks. The default refuses.
   * - ``parameters_schema(kind)``
     - The JSON Schema of a job's parameters for a sample kind, or ``None``, the default (`Job parameters`_).
   * - ``check_remote_artifact(names)``
     - Raise ``ValueError`` unless what ``names`` names resolves on the component's service. ``names`` is what the facts' ``check_artifact`` returned for an import. Tinker looks up the manifest's ``state_path`` on its service. The Tinker API describes no sampler or exported-adapter path, so those are checked only for their ``tinker://`` form.
   * - ``changed_parts(start, checkpoint)``
     - The parts of the scenario component that the checkpoint changed from the job's start, empty by default. The scenario's activation rule reads them as ``changed_kinds``.
   * - ``job_progress(job_id)``, ``job_records(job_id)``
     - Where a running job stands, as a JSON object, and a local directory of the files that record how an ended job ran. Both default to ``None``. The training worker copies them to where control reads them.
   * - ``archive_scenario(scenario)``
     - Move aside what the component keeps for a deleted scenario alone. The default keeps nothing for one scenario.
   * - ``start_jobs()``
     - Start running jobs in this process: the training service's job worker calls it once. No other process builds the component. The default does nothing.
   * - ``close()``
     - Release the component's resources; safe to call repeatedly.

**Backend facts.** A backend's ``TrainingBackendFacts`` state what a service needs without a component (`Backend facts <../developer-guide/training-backends.rst#backend-facts>`__ lists the members). They check an import's files with ``check_artifact(files)``:

- The default reads ``artifact_format``. ``full`` needs ``config.json`` and weight files. ``adapter`` needs a PEFT adapter in ``adapter/`` with a rank of at most ``adapter_rank`` and target modules within ``adapter_target_modules``. ``None`` refuses.
- Tinker checks its manifest and returns it for ``check_remote_artifact``.
- The default and Tinker first accept a `base root <python-api-artifacts.rst#base-roots>`__ whose ``inputs`` equal ``default_start_inputs``, and raise ``ValueError`` that names each input that differs.

**Local jobs.** ``reef.train.local_jobs.LocalJobComponent`` is the base of the built-in components. It keeps one directory per job (``job.json``, ``checkpoints.json``, and ``result.json`` once the job ended). It runs one job at a time in a worker thread, and it resumes unfinished jobs after a restart. A subclass implements ``check_job(job)`` and ``run(job, directory)``. It must call ``start_jobs()`` once it can train. Until then no job runs, including the jobs that a restart left.

Factories and modules
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Name
     - Role
   * - ``TrainingComponentFactory``, ``build_training_component`` (``reef.train.component``)
     - Define and build the ``TrainingComponent``. A training deployment's component factory returns the component alone.
   * - ``TrainingComponentFactory.facts``, ``training_backend_facts``
     - State a backend's ``TrainingBackendFacts`` without a component. Every factory defines ``facts``.

Control and the training service hold the facts, and control writes a new scenario's root from ``facts.default_start()``. The inference service of a split deployment reads the facts' ``artifact_format`` and builds no component.

Two training scenarios can share a deployment only if its training component allows it (``shares_scenarios``) and its runtime serves each scenario its own weights. The Tinker component allows it, because each job carries its own start. The Slime component does not: its workers hold one set of weights, so a Slime deployment trains one scenario.

The concrete components live with their owners: ``SlimeTrainingComponent`` in ``reef.train.slime_backend.component`` and ``TinkerTrainingComponent`` in ``reef.train.tinker_backend.component``. `Inference <python-api-inference.rst#factories-and-modules>`__ lists the runtime factories.

The job loop
~~~~~~~~~~~~

``reef.train.jobs.JobLoop`` is the control side of training. It holds the trainer's one training component, and weight training and harness evolution share it. ``Trainer.build`` takes it as ``job_loop``; a trainer without one trains nothing. `How a trainer runs jobs <../developer-guide/training-backends.rst#how-a-trainer-runs-jobs>`__ gives the steps and the rules for the start of a job and for a ``train-from``. The contract facts are:

- The trainer commits the submission of a job, which consumes the records of the job's data. A failed job ends like a finished one, so those records do not train again.
- The trainer commits each new saved checkpoint while the job runs: its stored ``content`` becomes a new artifact, with the checkpoint's training metrics. The backend makes the snapshot of the stored files (``RepositoryBackend.publish_stored``), so control reads none of them. The scenario's activation rule decides whether it serves.
- When the job ends, the trainer commits its outcome.
- A backend that cannot load a snapshot raises ``ValueError`` in the facts' ``check_import_start``, and Reef refuses the ``train-from`` with HTTP 400.

What a job needs beside its data comes from its stage's ``job_parameters``: the harness stage (``TraceJobData``) adds the training request and the task paths of the scenario's eval split.

Training is linear: the next job starts from the newest saved checkpoint, whether or not it serves. A restarted job replaces the trainer's start with its fresh start. A job continues the objective state of its start only when it trains the same objective. The trainer commits the loop's state as its algorithm state. This ``JobLoopState`` has four fields:

- ``start``: the training start as the component reads it, a non-empty string.
- ``jobs``: the running jobs as ``RunningJob`` values, in the order of their submission.
- ``finished_jobs``: the number of jobs that ended, finished or failed.
- ``last_job``: the outcome of the last job that ended, or ``None`` before the first one.

Once the commit of a job's end is durable, the samples service calls ``Record2Sample.job_ended(outcome)`` with ``last_job``. A trainer that starts with no running job calls it again for the last job that ended, before it reads a record. A stage whose job data holds ``TaskSample`` values reads each task's rollout results from the outcome's ``metrics``: the Tinker and Slime components report ``rollout_rewards``, the rewards of each task's rollouts, by task name. A component that cannot roll out tasks refuses such a job in ``check_job``.

Tinker integration
------------------

`Train with Tinker <../user-guide/tinker.rst>`__ gives the supported contracts.

``reef.train.tinker_backend.launch.TinkerDeployment`` implements the optional ``tinker`` backend as an ``InProcessTrainingDeployment``. Its ``component_factory`` builds a ``TinkerTrainingComponent`` from ``TinkerSettings``, only after deployment selection. Each job trains remotely from the job's start manifest. A saved checkpoint is both the start of a later job and the snapshot that Reef publishes. Train-from works: the component imports the snapshot's manifest as a start. A job takes these parameters:

- ``objective``;
- optional ``learning_rate_schedule`` and ``scheduling``;
- optional ``batch_size``, ``kl_coef`` and ``learning_rate`` overrides;
- ``save_interval``, the optimizer steps between saved checkpoints (by default, one checkpoint when the job ends);
- ``rollout``, the rollouts of each task, for a job of Harbor tasks.

``TrainingDeployment`` (``reef.train.deployment``) is a training backend's deployment definition:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Contract
   * - ``default_serving``
     - The serving of a deployment that names none and sets no engine setting: ``engines`` unless the backend says otherwise. Tinker's is ``trainer``.
   * - ``trainer_serving_config(settings)``
     - The runtime config of the backend's own serving, or ``None`` without one. Tinker's selects ``reef.inference.tinker.TinkerInferenceRuntime`` (runtime kind ``tinker``), which reads each snapshot's ``tinker-checkpoint.json`` manifest and samples from the immutable sampler that it names. A snapshot holds that manifest alone, so the component's ``artifact_format`` is ``None``.
   * - ``engine_weights(settings)``
     - An ``EngineWeights`` for a backend whose snapshots hold no weight files: the ``ArtifactFormat`` that Reef's engines load and the ``ArtifactWeights`` plugin (``reef.core.weight_layout``) that gives each snapshot's files. The default is ``None``: engines cannot serve such snapshots. ``ArtifactWeightFiles``, the default plugin, reads a snapshot's own files. Tinker names ``reef.tinker.weights.TinkerSamplerWeights``, which downloads the adapter at each snapshot's ``sampler_path`` and converts it into a PEFT adapter, so Reef's engines can serve it (``inference.serving: engines``).
   * - ``requires_local_model``
     - True by default. Hosted integrations set it to false to preserve remote model identifiers during deployment resolution.

Methods implement ``TinkerLoss`` from ``reef.train.tinker_backend.losses`` with a ``loss_family`` name, and decorate the class with ``@register_loss_family`` from ``reef.train.tinker_backend.loss_families``, in the module that defines their ``TinkerObjective``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Contract
   * - ``loss_fn``
     - The Tinker built-in loss that the family trains with. Default ``importance_sampling``.
   * - ``inputs(rows, base_logprobs, *, kl_coef)``
     - That function's token-aligned ``loss_fn_inputs``, one dictionary per row. For ``importance_sampling``, ``TokenRow.inputs`` builds the dictionary from response-token advantages, shifting prediction targets by one token and padding prompt positions with zeros.
   * - ``needs_base_logprobs``
     - Requests frozen-base scores for a nonzero KL coefficient.
   * - ``check_kl_coef(kl_coef)``
     - Raises ``ValueError`` for a coefficient that the family cannot train. Reef calls it when it prepares a job, before any remote work. The default accepts every coefficient.
   * - ``TinkerCustomLoss.loss(data, logprobs)``
     - The SDK boundary passes it to ``forward_backward_custom``, so the family computes its loss tensor on the Reef host. ``Datum`` objects are constructed only after this shaping.
