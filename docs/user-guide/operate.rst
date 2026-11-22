Operate a deployment
====================

Check a deployment, select its artifacts, follow training, and recover after a restart.

:For: whoever runs a Reef deployment
:Before you start: a running deployment from `Evolve your model <evolve-your-model.rst>`__ or `Evolve your harness <evolve-your-harness.rst>`__, its URL and token
:You will have: the routine checks, the artifact operations, and the recovery rules

The examples use ``REEF_URL`` and ``REEF_TOKEN`` as in `HTTP API <../reference/http-api.rst>`__. Two related pages cover the deployment itself:

- `Run the services in separate processes <split-services.rst>`__: the seven services and the split layout.
- `Choose where a deployment stores its data <storage.rst>`__: the storage backends, backup and upgrade.

Check health and status
-----------------------

.. code:: bash

   curl -f "$REEF_URL/healthz"
   curl -sS -H "Authorization: Bearer $REEF_TOKEN" "$REEF_URL/reef/status"

``/healthz`` answers as soon as the HTTP service is up. It says nothing about training. ``/reef/status`` reports the training side:

- For the deployment: the last asynchronous error, model preload failures, and the number of loaded scenarios (``loaded_scenarios``).
- For each scenario:

  - the commit number (``commit_number``) and the outcome of the latest training commit (``last_training_commit``), with the recipe's metrics
  - the artifact that answers its requests (``served_artifact_id``)
  - whether a job's data is waiting, and the processor's state
  - whether inference is admitted or paused while a snapshot loads
  - for a weight-training scenario, the running training jobs (``training_jobs``) and the number of ended jobs (``finished_jobs``)

The committed outcome tells a skipped or rejected update from one that still runs. Look here first when an old version keeps answering.

Each service that ``reef serve`` starts writes ``<service>.log`` and ``<service>.pid`` under ``reef.run-dir`` (``.reef/run/`` by default).

Read the artifacts
------------------

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

An artifact is one immutable output, such as a checkpoint's weights or a harness tree. Training, imports and harness syncs make artifacts. Its ``artifact_id`` is the only identifier that callers need. Keep the ID as an opaque string: do not parse a Git revision or a component name from it.

The list is newest first, with one row per artifact:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``active``
     - The selected artifact of its component.
   * - ``activated``
     - The artifact has served as active.
   * - ``served``
     - The scenario serves the artifact beside the active one (`Set what a scenario serves`_).
   * - ``operation: creation``
     - A starting artifact (`Starting artifacts`_). Its ``source`` says where it came from.

`Artifacts <../reference/http-api-artifacts.rst#http-api-artifacts>`__ lists every field. A step that made no artifact has no row: a rejected or skipped training step, an activation, a train-from, a change of the served artifacts. ``GET /reef/scenarios/{scenario}/operations`` lists every step (`Record and commit history <../reference/http-api-records.rst#record-and-commit-history>`__).

For a harness scenario, ``GET /reef/harness/artifacts`` lists the harness artifacts, with their step pages. ``GET /reef/harness?artifact_id=<id>`` pulls one tree. An artifact that waits for activation has ``active: false`` and ``activated: false``.

Starting artifacts
------------------

Reef creates one starting artifact for each component when it creates a scenario. The scenario serves these artifacts first, and each trainer starts its first job from its own. A starting artifact comes from one of two sources:

- **An artifact item** of the scenario's owner, which the create request names (`Choose a scenario's start`_). The row's ``source`` is ``{"kind": "item", "item": ..., "uri": ..., "version": ...}``.
- **The default start**, when the request names no item. Control writes it from the configuration. The row's ``source`` is ``{"kind": "default", "inputs": {...}}``, with the values that decide the start, such as the model and the seed's checksum.

The default start depends on the training backend:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Backend
     - Default start
   * - ``slime``, ``tinker``
     - A base root: one file, ``base.json``, that names the base model and its adapter settings. It holds no weight file. Reef serves the base model from it, and training starts at the base model.
   * - ``spindle_full``
     - A reference to the base revision on Hugging Face.
   * - a harness recipe
     - The files of ``evolution.seed``, else of the recipe's own seed. Without a seed, the artifact is empty.
   * - several components
     - Each component's default start under its name.

With ``tinker``, the first job from a base root makes a new LoRA from ``training.options.seed``, so every such job begins with equal weights. With ``seed: null``, the training service saves one untrained adapter in its ``state-dir`` at the first such job, and every such job starts from it.

A configuration change does not change an existing scenario: it keeps its stored starting artifacts, and only a new scenario gets the new default start. A start that no longer fits the configuration fails only its own scenario. The runtime refuses to activate it, or the training component refuses to start from it. The ``slime`` backend trains only the weights that its workers load from ``inference.model-path``: a scenario whose starting artifact came from other inputs does not load, and the error names both inputs.

A scenario that an earlier Reef created keeps its starting artifact (a copy of the model, a zero adapter, or an untrained adapter's checkpoint). The exception is a scenario stored with a shared base snapshot (``reef-scenario/4`` metadata): Reef refuses it, so create it again.

Choose a scenario's start
-------------------------

To start a new scenario from another artifact, name an artifact item of the scenario's owner when you create the scenario:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "code-repair-2", "owner": "acme", "artifact_id": "code-repair/<artifact_id>"}' \
     "$REEF_URL/reef/scenarios"

The owner's artifact library holds two kinds of item:

- ``<scenario>/<artifact_id>`` for each artifact that a scenario of the owner made. Such an artifact keeps its artifact ID in each scenario, and it gets no second item.
- The artifact items that you stored, such as an upload to S3 or a Hugging Face snapshot (`Data items <../reference/http-api-data-items.rst#http-api-data-items>`__).

Reef reads and checks the item's files as for an import (`Import an artifact`_). It then stores them, by reference or as a copy, as the component's starting artifact. That artifact is the active artifact and the training start of its component. Add ``component`` when the scenario has several components; each other component gets its default start. The answer's ``components`` maps each component to its starting ``artifact_id``. A scenario with one component also returns that ID as ``artifact_id``.

Reef registers no scenario when the start is not usable, and the name stays available:

- Reef cannot read the item, or the layout check refuses it. The answer has the status of the import, such as HTTP 400 for a directory without the component layout, or HTTP 404 for an item that the owner does not store.
- A training component refuses the artifact as a training start. The ``slime`` backend refuses every item, also in a composite where the item fills another component. Use the default start there.

A scenario that already exists ignores ``artifact_id`` and ``component``. A report or a record import creates a missing scenario from the default start. An inference request never creates a scenario, and no request header selects a start or an older snapshot. Each inference response names the answering artifact in ``x-reef-artifact-id``.

Import an artifact
------------------

To serve or train an artifact that was made outside Reef, import it by a URI that Reef reads:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"source": "s3://my-bucket/checkpoints/step-400/"}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - URI
     - Notes
   * - ``hf://org/model@revision``
     - A branch or a tag can move, so Reef copies the files. With a full 40-digit commit SHA, Reef keeps a reference to the revision and downloads its weights only when a serving engine loads them. Keep the repo on the Hub while you serve or train the artifact.
   * - ``s3://bucket/prefix/``
     - Needs the ``s3`` extra (``uv pip install 'reef-infra[s3]'``). Reads with boto3's standard credentials: the environment, a profile or an instance role. Reef stores no credential.
   * - ``file:///absolute/path``
     - A directory on the service's machine. To import a directory from your own machine, upload it to one of these places first.

``storage.artifact-sources`` adds readers for other schemes (`Python API <../reference/python-api-artifacts.rst#artifact-imports>`__). A reader can read whatever the service can read, so give import access only to callers that may train the scenario.

The directory must have the layout of the scenario's component: the training component's weight files, a Tinker manifest whose saved state is on the configured service, or a harness tree of UTF-8 text files that pass the adapter's validation. Otherwise Reef answers HTTP 400 and changes nothing. `Import an artifact <../reference/http-api-artifacts.rst#import-an-artifact>`__ in the HTTP API lists each layout and status.

An import adds an ``import`` row with the URI and the version in ``source``. It serves nothing, changes no training start, and does not run the activation rule. Activate the artifact or train from it as any other artifact. Reef copies the files into the artifact store, so a later change at the source does not change the artifact. While it runs, an import needs twice the model's size on disk: once under ``storage.artifact-cache-dir`` and once in the artifact store.

A harness sync is a different operation. ``POST /reef/harness/import`` merges the files that a person's session changed onto the active tree and commits a ``sync`` row. The new artifact becomes the active artifact and the training start of its component (`Evolve your harness <evolve-your-harness.rst>`__).

Set what a scenario serves
--------------------------

A scenario serves one or more artifacts of each component. One of them is the active artifact: harness routes read the active tree, and the default inference routes answer from the active artifact that completed startup. The ``served`` setting maps artifact IDs of one component to weights:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {"<id>": 100}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

The weights are integers from 0 to 100 that sum to 100. Afterwards the scenario serves exactly the artifacts of the mapping, and the artifact with the largest weight is active. In these examples, ``a`` and ``b`` are artifacts of one component, and ``a`` is active before the request:

.. list-table::
   :header-rows: 1

   * - Goal
     - ``served``
     - Result
   * - Activate an artifact
     - ``{"b": 100}``
     - ``b`` becomes the active artifact, and Reef stops every other served artifact.
   * - Split the traffic
     - ``{"a": 95, "b": 5}``
     - ``b`` answers 5% of the model requests that name no artifact. One conversation stays on one artifact (`Traffic split <../reference/http-api-serving.rst#traffic-split>`__).
   * - Serve an artifact beside the active one
     - ``{"a": 100, "b": 0}``
     - Reef loads ``b``, which answers only the requests that name it.
   * - Stop an artifact
     - ``{"a": 100}``
     - Reef stops serving ``b``.
   * - Serve nothing
     - ``{}``
     - Reef stops every engine route of the scenario. Its inference requests answer HTTP 404, and training still runs (`Serve nothing <../reference/http-api-serving.rst#serve-nothing>`__).

Rules of the request:

- Each ID is any row of the artifact list.
- A scenario of several components requires ``"component": "<name>"`` beside ``served``. An unknown component is HTTP 400. ``{"served": {}}`` without ``component`` makes the whole scenario serve nothing.
- ``POST /reef/scenarios`` with ``"serves": false`` creates a scenario that serves nothing.
- ``GET /reef/scenarios/{scenario}/settings`` returns ``served`` for each component.
- The request never changes a trainer's training start.

The response accepts the mapping, and inference loads the artifacts in the background. Until a new artifact has started, the active artifact answers its requests. Poll the `startup status <../reference/http-api-inference.rst#startup-status>`__ before you use a new active artifact; add ``?model=<artifact-id>`` for an artifact that serves beside the active one. Sending the request again retries a failed startup at once.

Reef appends an ``activate``, ``serve`` or ``stop_serving`` operation for each artifact that changes. It creates no artifact. Each operation stores the complete selection of all components, so a change of the active ``harness`` leaves the active weights as they are, and recovery restores the same combination. The served artifacts survive a restart.

**One artifact's own URL.** Each served artifact also answers at its own URL. With weight 0, an older artifact keeps answering while a newer one is active, or you can try a new artifact before it is active:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model": "reef", "messages": [{"role": "user", "content": "hi"}]}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts/<id>/v1/chat/completions"

The record of such a call names the ``artifact_id`` that answered. Choose which clients call each URL, for example with a gateway.

**Runtime limits.** The runtime sets how many weight artifacts a scenario serves together. Reef refuses a mapping that names more with HTTP 400, and changes nothing.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Runtime
     - Weight artifacts served together
   * - Tinker
     - Any of the scenario's artifacts.
   * - Reef's adapter serving
     - One.
   * - Reef's full-weight serving
     - One checkpoint on each inference component. A deployment has one component by default; ``inference.remote.slot-urls`` names further components (`The inference section <../reference/configuration-inference.rst#the-inference-section>`__).

**Several checkpoints on several components.** Each full-weight checkpoint of the mapping has its own component:

- A change of traffic weights alone loads nothing. A split, a promotion and a rollback keep each served checkpoint on its component.
- A stopped checkpoint keeps its component for a while. It takes no new requests, and the admitted requests can finish. Reef schedules its retirement ``inference.timeout-s`` plus 5 seconds after it publishes the new routes (305 seconds by default). A busy serving worker or a failed release makes the retirement later, and no API reports its end. The component then keeps its engines and their GPU memory for the next checkpoint.
- A new checkpoint needs a free component. One mapping that removes ``a`` and adds ``c``, such as ``{"b": 50, "c": 50}``, works only when another component is free. Otherwise Reef accepts it, the startup fails before Reef publishes new routes, ``a`` keeps its route and its component, and each retry fails in the same way.

To replace ``a`` with ``c`` when two components serve ``a`` and ``b``:

#. Send a mapping that names neither ``a`` nor ``c``: ``{"b": 100}``.
#. Wait about ``inference.timeout-s`` plus 5 seconds. This is an estimate: the time starts when Reef publishes the new routes, not at the HTTP 200. A ``ready`` status of ``b`` does not show a free component.
#. Add ``c`` with a mapping such as ``{"b": 100, "c": 0}``, and poll the status of ``c`` itself: ``GET /reef/scenarios/{scenario}/inference/status?model=<id-of-c>``. Without ``model``, the status describes the active ``b``. If ``c`` fails because no component is free yet, wait and send the mapping again.

`Set what a scenario serves <../reference/http-api-serving.rst#set-what-a-scenario-serves>`__ in the HTTP API lists the answers and status codes.

Activation rule
---------------

The deployment's `activation section <../reference/configuration-recipe.rst#the-activation-section>`__ names the rule that decides whether Reef activates each new artifact:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Rule
     - Effect
   * - ``always`` (default)
     - Activates each new artifact when the training step makes it. Its ``review_kinds`` holds back a harness tree that changes one of the listed tree kinds.
   * - ``never``
     - Makes each new artifact but does not activate it. A person activates it with ``served``.
   * - ``evaluated``
     - Activates an artifact once an evaluation of it reaches ``min_mean_score``.

A scenario can select a rule for itself: send ``{"activation": {"module": "never"}}`` to ``POST /reef/scenarios/{scenario}/update``. ``"activation": null`` follows the deployment's rule again. ``GET /reef/scenarios/{scenario}/settings`` returns the scenario's rule and the rule that applies (`Scenario updates <../reference/http-api-scenarios.rst#scenario-updates>`__).

An artifact that the rule does not activate still becomes the start of the next training job.

Train from an artifact
----------------------

Each trainer continues from its training start, which is its own newest output: a weight trainer's newest saved checkpoint, or the tree that a harness trainer kept. To make a trainer continue from another artifact, set its training start:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

- Reef cancels the running job that continues the training start. The next job that names no start, or the next evolve step, starts from the artifact.
- A scenario of several components requires ``"component": "<name>"`` in the body. An unknown component is HTTP 400.
- Train-from never changes the active selection.
- A weight artifact without optimizer state, such as a default start, trains with a new optimizer state.
- Each train-from starts a new run in experiment tracking.

One training request can instead name its own start with ``train_from`` (`Manual training <../reference/http-api-training.rst#manual-training>`__). Such a job does not move the training start, and a train-from does not cancel it.

Whether a trainer can start from an artifact depends on the backend:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Backend
     - Train-from
   * - ``tinker``
     - Any artifact of the scenario. Reef imports the artifact's checkpoint manifest, with the saved training state and the optimizer state.
   * - ``slime``
     - Refused with HTTP 400 for every artifact, because the workers continue from their own latest checkpoint.
   * - harness
     - The next evolve step starts from the artifact's tree.

Reef can activate an earlier checkpoint of a Slime scenario, but it cannot train from it. To train again from an earlier Slime checkpoint, start a new deployment for a new scenario:

- Give the deployment its own ``save`` and ``save-hf`` directories. The Slime component does not overwrite a rollout that it saved before.
- In ``training.options``, set ``load`` to the old ``save`` directory and ``ckpt-step`` to the saved rollout.

The new scenario's first job continues from the weights that the workers loaded. An existing scenario cannot move to the new deployment: its committed start names a checkpoint of the old one, so the Slime component refuses its jobs.

Go back to an older artifact
----------------------------

For a scenario with several components, restore an earlier complete selection:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" -d '{"commit_number": 12}' \
     "$REEF_URL/reef/scenarios/code-repair/restore"

Choose a commit number from ``GET /reef/scenarios/code-repair/operations``, or use 0 for the starting artifacts. Reef restores all active component IDs together and appends a ``restore`` operation. It creates no artifact and leaves the training starts unchanged. The answer's ``components`` gives the restored mapping. Poll the startup status before you send inference requests.

Activation alone does not last under the default rule: the trainer continues from its training start, and Reef activates the next artifact that it makes. To go back for good, activate the older artifact and do one of these:

- Send ``train-from`` with the same artifact, so that the next artifact descends from it.
- Select the ``never`` rule, so that no new artifact serves until a person activates it.

Set the checkpoint cadence
--------------------------

Every checkpoint that a job saves becomes an artifact. Without an ``evaluation`` section, Reef activates each of them. The next job always starts from the last saved checkpoint, whether or not evaluation selected it.

A Slime or Tinker job saves a checkpoint every ``save_interval`` optimizer steps, and at its end. ``save_interval`` is a job parameter that ``recipe.config.job_parameters`` sets. Without it, a job saves one checkpoint at its end.

For Slime:

- A checkpoint is the Megatron checkpoint and its HF export (with LoRA, the PEFT adapter). Each run of ``save_interval`` steps trains as one Slime rollout (one ``rollout_id``, Slime's unit of training).
- ``save-interval`` in ``training.options`` does not change this cadence. Megatron requires it together with ``save``, so keep both set.
- ``--critic-save-interval`` (default 1) in ``training.options`` sets how many saved rollouts pass between two critic checkpoints.
- These ``training.options`` bound the checkpoints on disk:

  - ``--reef-checkpoint-policy``: ``latest`` or ``best_reward``
  - ``--reef-checkpoint-max-storage-fraction`` (default 0.8) and ``--reef-checkpoint-min-free-space-fraction`` (default 0.1)
  - ``--reef-checkpoint-max-storage`` and ``--reef-checkpoint-min-free-space``

Retention never deletes the checkpoint that the next job starts from. When storage cannot hold the next checkpoint, the job fails. The service log names the error, and the commit that ends the job records it in the ``training_job`` metric.

A failed job ends like a finished one. Reef consumed the records of its data when it submitted the job, so they do not train again. The next job starts from the last saved checkpoint.

Track experiments with W&B
--------------------------

Tracking is optional and off by default. ``observability.wandb`` in `Configuration <../reference/configuration-observability.rst#experiment-tracking>`__ lists every key. Export ``WANDB_API_KEY`` before you start the stack. There is no key field, and Reef refuses Slime's ``--wandb-key`` flag, so a credential never enters a command line or a run config.

W&B shows one group and one run per scenario, and a new run after each train-from. An activation does not end a run.

- **Training results.** Each result lands on the run's ``train/step`` axis with the monotonic ``reef/commit_number``, which joins it to the commit history. The commit metrics record ``experiment/run_id``, and the run's ``reef/training_job_id`` leads back to the job.
- **Several trainers.** Each step's metrics carry the component name as a prefix, on the same ``train/step`` axis, with the name in ``reef/component``. The run summary names the last trainer that committed and its job.
- **Operations.** Every 10 seconds the run receives ``operations/*`` against ``operations/time_seconds``: unread records, processor queues, waiting ages, training execution durations, request counts and latency, admission waits, record acceptance and rejection, and version-mismatch rejections. A pure inference scenario uploads them too. They need neither status polling nor a training commit. Read the `metric definitions <../reference/configuration-observability.rst#operational-metrics>`__ first: reserved job data may be executing, and manual-mode data may wait on purpose.

Reef sends each event from a background thread after its commit, in commit order. A slow tracker therefore delays neither serving nor training. A tracking failure is logged and never fails a training step or its commit. Events still queued at shutdown are sent before the tracker closes.

Restart and recovery
--------------------

With persistent storage paths, a restart keeps this state:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - State
     - Guarantee
   * - Records
     - Persisted before the processor sees them. Never trained twice.
   * - The commit history
     - Append-only per scenario. The database transaction that adds a step is the commit point.
   * - Published artifacts
     - Kept in the artifact repository. Reef serves the active selection that the commit history names.
   * - Algorithm state and record progress
     - Restored from the head record of the commit history.
   * - A training job in flight
     - Resumed: the training component runs the job again, and Reef keeps polling it. A Tinker job continues after its last saved checkpoint. A Slime job trains its data again from the latest checkpoint, unless it saved its checkpoint before the stop.
   * - Checkpoint decisions of a running job
     - Each decision is committed. A restart evaluates only the checkpoints that have no committed decision.
   * - A harness step in flight
     - Resumed, as any training job. A step's outcome is durable before its checkpoint, so a step that settled commits without running again. A step that did not settle runs again from the same start and harness state.

`State model <../advanced_topics/state-model.rst#commit-ordering>`__ describes the commit ordering behind the table. On ephemeral storage, none of these guarantees hold past its loss. The state lives under ``.reef/`` by default (`Choose where a deployment stores its data <storage.rst>`__).

**One control service.** Each scenario needs one control service. For a second deployment, use other ports and storage paths, not the same store. A process that stops normally releases the control lease. After a crash, the next process waits until the old lease expires (at most 30 seconds) before it loads scenarios.

**Repository synchronization.** After a step commits, ``/reef/status`` reports the scenario's ``artifact_head_sync``, with a ``state`` and any ``error``. Repository revisions stay internal.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``state``
     - Meaning
   * - ``synchronized``
     - The backend head names the chain tip.
   * - ``pending``
     - Updating the head failed.
   * - ``conflict``
     - Another writer moved the head to an unrelated snapshot. Reef never overwrites that head.

The committed step remains successful. Reef retries the synchronization before the next commit and at a restart, before it loads the scenario. If the retry fails, the new commit stops before it publishes or writes its commit record.

**Delete a scenario.** ``DELETE /reef/scenarios/{scenario}`` renames the scenario's repository ref into ``refs/reef/archived/``. From then on the scenario does not exist, and its ID is never used again. Control's sweep of ended work, which the delete wakes, then frees the scenario's space:

- It cancels the scenario's training jobs and discards its stored content.
- The storage backend archives its records, commits, settings and task files.
- The training service moves its history and step records under an ``archived/`` sibling.
- Each other service drops its clone and cached snapshots of the scenario.

A step that fails stays for the next sweep, an hour later. The first sweep after a restart finishes a delete that stopped.

Run the services in separate processes
--------------------------------------

A deployment has seven services: ``control``, ``inference``, ``samples``, ``training``, ``evaluation``, ``data`` and ``harness``. By default, ``reef serve`` runs all of them in one process. To keep inference up while control restarts, or to run more replicas of one service, start each service in its own process with ``reef serve --component NAME``. `Run the services in separate processes <split-services.rst>`__ describes the layout, the routes of each service, and what happens while control restarts.

Import Harbor tasks
-------------------

A scenario can hold your own Harbor tasks in its train and eval splits, for example a benchmark that a recipe evaluates on. A task can hold binary files, such as images. Name a split manifest and the directory of the tasks in the ``task-import`` section:

.. code-block:: yaml

   task-import:
     manifest: /data/benchmark/split.json
     tasks-root: /data/benchmark/tasks

The manifest is a JSON file that names each task once, in ``train`` or ``eval``:

.. code-block:: json

   {"version": 1, "seed": 0, "eval_fraction": 0.5, "train": ["sum-391"], "eval": ["sort-17"]}

Each name is a Harbor task directory under ``tasks-root``. A name starts with a lowercase letter or a digit, and holds only lowercase letters, digits, ``.``, ``_`` and ``-``.

When a scenario opens, Reef checks each directory with Harbor's task loader. It stores the task as a task item of the scenario's owner, which the scenario uses in its split (`Data items <../reference/http-api-data-items.rst#http-api-data-items>`__). The task file store keeps a copy of the files.

The import runs again each time a scenario opens:

- To add tasks, add them to the manifest and the directory, then restart the service or reload the scenario.
- A task that moves to the other split in the manifest moves in the scenario.
- A task that is already imported with the same files does not change. To change an imported task, give it a new name.
- To add tasks to one running scenario without a restart, create task items with ``POST /reef/namespaces/{owner}/data/tasks``.

A scenario does not open, and the error says why, when:

- a task directory is missing, or Harbor cannot load it;
- an imported task now has other files;
- a task has the name of another sample of the scenario.

A harness evolution recipe evaluates on the imported eval tasks with ``evolution.evaluation.imported_tasks: true``. Each task runs in its own container (`Task directories <evolve-your-harness.rst#task-directories>`__). Each step reads the eval split again, so a task that you import later joins the next step. See `Harness evolution keys <../reference/configuration-harness.rst#harness-evolution-keys>`__.

Limit the scenarios a service holds
-----------------------------------

By default, a service loads every registered scenario when it starts, keeps each one loaded until you delete it, and accepts any number of scenarios. A loaded scenario holds a storage session, its trainers and its local worker threads. Four ``reef`` settings change this. All of them are off by default.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Effect
   * - ``scenario-idle-unload-s``
     - Reef unloads a scenario that is idle for this number of seconds.
   * - ``max-loaded-scenarios``
     - The maximum number of loaded scenarios. To load one more, Reef first unloads the idle scenario with the oldest use, without waiting for the idle time. If no scenario is idle, the request gets HTTP 503 with a ``Retry-After`` header. The start-up preload obeys the limit too: a scenario that it does not preload loads on its first request.
   * - ``max-scenarios``
     - The maximum number of registered scenarios. A request that would create one more gets HTTP 409. To free a place, delete a scenario.
   * - ``scenario-admission``
     - A ``ScenarioAdmission`` plugin. It replaces the two limits above, and Reef refuses a configuration that sets both.

.. code-block:: yaml

   reef:
     scenario-idle-unload-s: 3600
     max-loaded-scenarios: 50
     max-scenarios: 1000

An unload keeps all durable state: the records, the commit history, the settings and the snapshots. The next request to the scenario loads it again, as after a restart. Counters in memory, such as the operation metrics, start again from zero.

A scenario is idle when all of these conditions are true:

- No request or local training cycle runs on it.
- No trainer has records that it would still train on.
- The storage backend is durable.
- For ``scenario-idle-unload-s`` only: no request, record or local training cycle used it during the idle time.

Reef never unloads a weight-training scenario: it holds the deployment's training component and its running job, which only a delete releases.

**Write your own policy.** Subclass ``reef.scenario.admission.ScenarioAdmission``. Name it as ``package.module:ClassName``, or as a mapping with ``module`` and ``config``. Reef makes the plugin with ``from_config(config)`` and calls it before it creates a scenario and before it loads one. Reef makes these calls one at a time, so both methods must return quickly:

- ``check_creation(scenario, registered_count=...)``: to refuse a new scenario, raise ``reef.core.errors.ScenarioLimitReachedError`` (HTTP 409).
- ``admit_load(scenario, loaded=...)``: return a ``LoadAdmission``. Its ``unload_count`` is the number of loaded scenarios that Reef must unload first, and its ``candidates`` are the scenarios that Reef tries, in that order. Reef unloads only idle candidates. If too few candidates are idle, Reef refuses the load with ``reef.core.errors.ScenarioCapacityFullError`` (HTTP 503). The plugin can also raise that error itself.

``max-scenarios`` and ``max-loaded-scenarios`` configure the built-in plugin, ``ScenarioLimits``.
