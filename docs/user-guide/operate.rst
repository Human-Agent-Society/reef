Operate a deployment
====================

What to check while a deployment runs, how to read and steer its release chain, how to keep an eye on training, and what survives a restart.

.. page::
   :for: whoever runs a Reef deployment
   :needs: a running deployment from `Evolve your model <evolve-your-model.rst>`__ or `Evolve your harness <evolve-your-harness.rst>`__, its URL and token
   :outcome: the routine checks, the version operations, and the recovery rules

The examples use ``REEF_URL`` and ``REEF_TOKEN`` as in `HTTP API <../reference/http-api.rst>`__.

Check health and status
-----------------------

.. code:: bash

   curl -f "$REEF_URL/healthz"
   curl -sS -H "Authorization: Bearer $REEF_TOKEN" "$REEF_URL/reef/status"

``/healthz`` answers as soon as the HTTP service is up; it says nothing about training. ``/reef/status`` is the training side. It reports these values for the deployment:

- the last asynchronous error
- model preload failures
- the number of loaded scenarios (``loaded_scenarios``)

It reports these values for every scenario:

- the step counter and the latest committed step outcome
- the runtime load ID that the engines serve
- whether a job's data is waiting, and the processor's state
- whether inference is admitted or paused while a release loads
- for a weight-training scenario, its running training job (``training_job``) and the number of jobs that ended, finished or failed (``finished_jobs``)

The committed outcome includes the recipe-owned metrics. You can therefore tell a skipped or rejected update from one that is still running. Look here first when requests keep being served by an old version.

The service and every process ``reef serve`` started write logs under ``reef.run-dir`` (``.reef/run/`` by default), one ``<service>.log`` and one ``<service>.pid`` each.

Read the artifacts
------------------

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

An artifact is the content of one component, such as a checkpoint's weights or a harness tree. Each training step that saves a checkpoint or keeps a tree makes one. The list is newest first, one row for each artifact. ``active`` marks the component's active artifact, and ``activated`` marks each artifact that has been active at some time. `Artifacts <../reference/http-api.rst#artifacts>`__ in the HTTP API lists every field. In a scenario of one component, which every shipped recipe is, ``artifact_id`` equals the ``release_id`` of the release that made the artifact.

A training step that made no artifact, such as a rejected or skipped step, has no row. Neither do activations and train-from operations. ``GET /reef/scenarios/{scenario}/commits`` lists every step (`Record and commit history <../reference/http-api.rst#record-and-commit-history>`__).

For harness scenarios, ``GET /reef/harness/releases`` lists the release chain oldest first with each step's evaluation metrics, and ``GET /reef/harness?release_id=<id>`` returns any listed tree. It lists every training step, also a rejected or skipped step, which has ``checkpoint: false`` and carries the active release's ID. A new release that the activation rule did not activate sits in the catalog with its evaluation metrics, ``active: false`` and ``activated: false``. No session gets it until a person activates it. Clients label such a release ``Waiting for activation``.

Read the proposal inbox
-----------------------

.. code:: bash

   ls -R .reef/proposals/code-repair
   cat .reef/proposals/code-repair/settled/*.json

A harness scenario keeps the proposals its agents sent through ``POST /reef/harness/proposals`` as plain JSON files under ``evolution.proposals_dir`` (default ``.reef/proposals``), one directory per scenario. Each file holds the body the agent sent (``mutations``, ``reason``, ``session``, ``release_id``), the ``proposal_id`` the route answered, ``received_at``, and ``head_release_id``, the active release that it was admitted against. The file name starts with the receive time, so ``ls`` shows the queue in age order. Where a file sits says what happened to it:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Directory
     - Meaning
   * - the scenario directory itself
     - pending: admitted at the route, not yet taken by a step
   * - ``claimed/``
     - taken by the step now running; a step that failed before it settled leaves the file here, and no later step takes it again
   * - ``refused/``
     - the step's own admission refused it, because the entries of the step's training start no longer admit it; ``refused`` holds the rule
   * - ``settled/``
     - the evaluation settled it; ``verdict`` holds the step, whether it was selected, and the selector's reason

The commit that settled a proposal carries ``proposal: {id, session, release_id}`` in its metrics, so ``GET /reef/harness/releases`` says which session proposed the tree of a release. A step takes the oldest pending proposal before it asks the method's own ``propose``; ``evolution.max_pending_proposals`` (default 8) bounds the queue.

Read what a release requires
----------------------------

A request sent to ``POST /reef/train`` may name what its change needs from the person's machine, and the commit that answered it carries ``training_request.requires`` (``{name, kind, check}`` items; ``kind`` is ``permission``, ``env`` or ``service``; the method may have added items of its own), so ``GET /reef/harness/releases`` and ``GET /reef/harness`` say what a release needs before anyone installs it. A releases row carries its own step's list; the manifest's ``requires`` is the union over the release's chain, the newest definition of a name winning, so a release whose request named nothing still needs what an earlier one added. On the person's machine the ``.reef-harness-release`` release metadata file the install script writes beside the tree carries two keys for it:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Meaning
   * - ``requires``
     - what the installed release needs over its whole chain, as its manifest carried it
   * - ``setup``
     - the check offs ``reef-<adapter> setup`` recorded, ``{name, checked_at, check}`` each, carried over from the previous release metadata file by name, so a check off survives every later install through the script (the stdlib client's pull writes its own release metadata file, without them); an item whose check is not the recorded one counts as unmet, and ``setup`` runs it again

The install script refuses a release with an item ``setup`` does not meet: it prints the setup list and the newest release in the chain that requires nothing, the one that installs on a machine with nothing set up (``?release_id=<id>``; install it, run ``reef-<adapter> setup`` for the active release's list, then install the active release), and exits 1 before it installs the binary or makes a directory. ``reef-<adapter> setup`` is the only thing that runs a check, after the person read it and confirmed; ``reef-<adapter> setup --mark <name>`` records a check off by hand, and ``reef-<adapter> setup --release <id>`` checks off an inactive release's items before you activate and install it. A release metadata file the stdlib client pull or an older install wrote carries neither key, which reads as nothing required and nothing checked off.

Choose a starting release
-------------------------

Inference always answers from the scenario's active release. No request header keeps a client on an older release.

A request can carry ``x-reef-release-id: <id>``. If the request creates the scenario, Reef binds the new scenario to that starting release. On a scenario that already exists, the header must name the bound starting release. Otherwise, Reef refuses the request with HTTP 409. `Headers <../reference/http-api.rst#headers>`__ in the HTTP API lists this header.

To install a specific harness release, add ``?release_id=<id>`` to the harness manifest or install route. To serve an earlier artifact again, activate it as the next section describes.

Activate an artifact
--------------------

The active release is the one that Reef serves. Inference answers from it, and the harness routes serve its tree. It binds one active artifact for each component. To serve another artifact, activate it:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/activate"

The target is any row of the artifact list: an artifact that a training step made, or the creation artifact. Reef loads the artifact into serving and makes it the active artifact of its component. What Reef writes depends on the scenario:

- In a scenario of one component, Reef writes no artifact or release. It appends an ``activate`` record. The release that made the artifact serves again.
- In a scenario of several components, Reef publishes a release: the active release with ``component`` bound to the artifact. This release makes no new content. The other components stay as they serve.

The answer's ``artifact_id`` names the activated artifact, and its ``release_id`` names the active release. Leave out ``component`` when the scenario has one component. A scenario of several components requires it. An unknown component is HTTP 400. History is not rewritten, and the step counter keeps increasing. Activation never changes a trainer's training start.

A full-weight serving runtime cannot activate the creation artifact after it served a checkpoint. The engines do not keep the files of the model that they started with, so Reef answers an error. Activate a saved checkpoint instead. `HTTP API <../reference/http-api.rst#activate-an-artifact>`__ lists the answers and status codes.

The deployment's `activation section <../reference/configuration.rst#the-activation-section>`__ names the rule that decides whether Reef activates each new artifact:

- ``always``, the default, activates each new artifact when the training step makes it. Its ``review_kinds`` holds back a harness tree that changes one of the listed node kinds.
- ``never`` makes each new artifact but does not activate it. A person activates it.
- ``evaluated`` activates an artifact once an evaluation of it reaches ``min_mean_score``.

An artifact that the rule does not activate still becomes the start of the next training job.

Train from an artifact
----------------------

Each trainer continues from its own newest output: a weight trainer from its newest saved checkpoint, a harness trainer from the tree it kept. That artifact is the trainer's training start. To make a trainer continue from another artifact, set its training start:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

Reef cancels the running job. The next training job, or the next evolve step, starts from the artifact. Train-from never changes the active release. Leave out ``"component"`` when the scenario has one component. A scenario of several components requires ``"component": "<name>"`` in the body, and an unknown component is HTTP 400. A weight artifact without optimizer state, such as the creation artifact, starts training with a new optimizer state. Each train-from also starts a new run in experiment tracking.

Whether a trainer can start from an artifact depends on the backend:

- The ``tinker`` backend imports the artifact's checkpoint manifest, so a Tinker scenario can train from any of its artifacts. The import includes the saved training state with the optimizer state.
- The ``slime`` backend refuses every artifact with HTTP 400, because its workers continue from their own latest checkpoint.
- A harness trainer starts its next evolve step from the entries of the artifact's tree.

Reef cannot move the training of a Slime scenario back to an earlier checkpoint, although it can activate one of its artifacts. To train again from an earlier checkpoint, start a new deployment for a new scenario:

- Give the deployment its own ``save`` and ``save-hf`` directories. The Slime component does not overwrite a rollout that it saved before.
- Point Slime at the old checkpoint with its own options in ``training.options``: ``load`` names the old ``save`` directory, and ``ckpt-step`` names the saved rollout.

The new scenario's first job continues from the weights that the workers loaded. An existing scenario cannot move to the new deployment: its committed start names a checkpoint of the old one, so the Slime component refuses its jobs.

Go back to an older artifact
----------------------------

Activation alone does not last under the default rule. The trainer still continues from its training start, and Reef activates the next artifact that it makes. To go back to an older artifact for good, activate it and do one of these:

- Send ``train-from`` with the same artifact, so that the next artifact descends from it.
- Switch to manual activation, so that no new artifact serves until a person activates it.

Set the checkpoint cadence
--------------------------

Every checkpoint that a job saves becomes an artifact. Without an ``evaluation`` section, Reef activates each of them. The next job always starts from the last saved checkpoint, whether or not evaluation selected it.

How often a job saves depends on the backend:

- A Slime or Tinker job saves a checkpoint every ``save_interval`` optimizer steps, and at its end. ``save_interval`` is a job parameter: the recipe's job plan sets it, and ``recipe.config.job_parameters`` sets it for the default plan. Without it, a job saves one checkpoint at its end.
- A Slime checkpoint is the Megatron checkpoint and its HF export (with LoRA, the PEFT adapter). Each run of ``save_interval`` steps trains as one Slime rollout (one ``rollout_id``, Slime's unit of training). ``save-interval`` in ``training.options`` does not change this cadence. Megatron requires ``save-interval`` together with ``save``, so keep both set.

A Slime deployment bounds its checkpoints on disk with options in ``training.options``:

- ``--reef-checkpoint-policy``: ``latest`` or ``best_reward``.
- ``--reef-checkpoint-max-storage-fraction`` (default 0.8) and ``--reef-checkpoint-min-free-space-fraction`` (default 0.1).
- ``--reef-checkpoint-max-storage`` and ``--reef-checkpoint-min-free-space``.

Retention never deletes the checkpoint that the next job starts from. When storage cannot hold the next checkpoint, the job fails. The service log names the error, and the commit that ends the job records it in the ``training_job`` metric. A failed job ends like a finished one. Reef consumed the records of its data when it submitted the job, so they do not train again. The next job starts from the last saved checkpoint.

``--critic-save-interval`` (default 1) in ``training.options`` sets how many saved Slime rollouts pass between two critic checkpoints.

Track experiments with W&B
--------------------------

Tracking is optional and off by default; ``observability.wandb`` in `Configuration <../reference/configuration.rst#experiment-tracking>`__ lists every key. Export ``WANDB_API_KEY`` before starting the stack; there is no key field, and Reef refuses Slime's ``--wandb-key`` flag so a credential never enters a command line or a run config.

What you see in W&B: one group per scenario and one run per scenario, plus a new run after every train-from, so the training from each new start is its own curve. An activation does not end a run. Every training result lands on the run-local ``train/step`` axis carrying the monotonic ``reef/step`` that joins it to the commit history, and the commit metrics record ``experiment/run_id``, so a Reef version leads to its run and the run's ``reef/training_job_id`` leads back. A scenario with several trainers prefixes each step's metrics with its component name, on the same ``train/step`` axis, and records the name as ``reef/component``; the run summary then names the last trainer that committed and its job. Reef sends each event from a background thread after its commit lands, in commit order, so a slow tracker does not delay serving or training. Tracking failures are logged and never fail a training step or its commit. Events still queued at shutdown are sent before the tracker closes.

The same run also receives ``operations/*`` every 10 seconds, plotted against
``operations/time_seconds``. Use these panels to inspect unread records,
processor queues, waiting ages, training execution durations, request
counts and latency, admission waits, record acceptance/rejection and
version-mismatch rejections. Pure inference
scenarios also upload these samples. These samples do not
require status polling or a successful training commit. Read the `metric
definitions <../reference/configuration.rst#operational-metrics>`__ before
interpreting queue counts: reserved job data may be executing, and manual-mode
data may intentionally wait.

Restart and recovery
--------------------

What survives a restart, provided the storage paths are persistent:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - State
     - Guarantee
   * - records
     - persisted before the processor sees them; never trained twice
   * - the commit history
     - append-only per scenario; the database transaction that adds a step is the commit point
   * - checkpointed versions
     - in the Git LFS repository; Reef serves the active release that the commit history names
   * - algorithm state and record progress
     - restored from the head record of the commit history
   * - a training job in flight
     - resumed: the training component runs the job again after the
       restart, and Reef keeps polling it. A Tinker job continues after its
       last saved checkpoint. A Slime job trains its data again from the
       latest checkpoint, unless it saved its checkpoint before the stop
   * - checkpoint decisions of a running job
     - each decision is committed; a restart evaluates only the
       checkpoints that have no committed decision
   * - a local cycle's job data (a harness step) when the service stops
     - kept: once shutdown begins, a local cycle commits nothing and no
       new one starts, since its model calls go through the stopping
       service; after the next start the local cycles of a composite
       recipe open once the service answers on its own address, and the
       rows train then, with no new record needed

After a step commits, ``/reef/status`` reports its scenario's
``artifact_head_sync`` with the ``release_id`` of the `chain tip
<../advanced_topics/state-model.rst#the-release-chain>`__ and any ``error``.
The ``state`` is ``synchronized`` when the backend head names the chain tip,
``pending`` when updating it failed, or ``conflict`` when another writer moved it to an
unrelated release. The committed step remains successful. Before another commit,
Reef retries synchronization; if it still fails, the new commit stops before
publishing or writing its commit record. A conflict never overwrites the other
writer's head. Restart also synchronizes the head before loading the scenario.

The scenario storage and the repository live under ``.reef/`` by default (``agent_record_dir``, ``task_file_dir``, ``artifact_repository``, ``artifact_work_dir``, ``artifact_cache_dir``). On ephemeral storage none of the guarantees above hold past its loss. Each scenario needs one Reef writer; run a second deployment on other ports and storage paths rather than two services on one store. `State model <../advanced_topics/state-model.rst#commit-ordering>`__ describes the commit ordering behind the table. ``DELETE /reef/scenarios/{scenario}`` retires a scenario: the storage backend archives its records, commits, settings and task files, its proposal inbox and step records move under an ``archived/`` sibling and its repository ref is renamed into ``refs/reef/archived/``, so a name can be reused without the old chain. Deleting a weight-training scenario cancels its running training job.

Choose the storage backend
--------------------------

The storage backend keeps the records, commits, and settings of each scenario.
The settings are the model configuration and the selected training mode. You
set both with ``POST /reef/scenarios/{scenario}/update``. After a restart, the
scenario uses its stored training mode. If the recipe does not support that
mode, the scenario uses the default mode and Reef logs a warning.

Select the backend with ``storage.record-backend``:

.. list-table::
   :header-rows: 1

   * - Value
     - Where scenario state is kept
   * - ``sqlite`` (default)
     - One SQLite database per scenario (records, commits and settings) under
       ``agent_record_dir``
   * - ``postgres``
     - Tables in one PostgreSQL schema; ``agent_record_dir`` holds no scenario state
   * - a plugin reference
     - The ``ScenarioStorage`` subclass that the reference names

A model configuration can hold an upstream ``api_key``. Thus only the owner
can read a SQLite database that holds settings. Responses never return the
key; they report only ``has_api_key``.

To use PostgreSQL, install the driver with ``uv pip install 'reef-infra[postgres]'``
(or ``uv pip install -e '.[postgres]'`` from a checkout), then configure:

.. code-block:: yaml

   storage:
     record-backend: postgres
     record-database-url: ${REEF_RECORD_DATABASE_URL}
     record-database-schema: reef_records
     agent-record-dir: .reef/agent-record

Set ``REEF_RECORD_DATABASE_URL`` to a PostgreSQL connection URL, such as
``postgresql://reef:password@db.example.com/reef?sslmode=require``. The database
must already exist; Reef creates its tables in the selected dedicated schema.
The database role needs permission to create that schema and its tables on
first startup. Give independent deployments distinct schemas (default:
``reef_records``). PostgreSQL 14 and newer are supported. Connection strings
using ``postgresql+psycopg://`` are also accepted.

A shared connection pool returns connections after each operation. Archived
scenarios stay in PostgreSQL under their old storage identifier; a recreated
scenario gets a new identifier. The same retention settings apply across all
active and archived stores in the configured schema, while preserving active
records and retry hashes.

When you upgrade, Reef imports the model configuration files that earlier
versions kept in ``agent_record_dir`` into the storage backend. Reef then moves
the imported files to ``agent_record_dir/archived/imported-*``. Reef does not
read the ``*.commits.jsonl`` files of earlier versions. A deployment upgraded
from such a version recovers each scenario from its last checkpointed release,
as after a lost commit history.

A change from one backend to another does not move records or commit history.
Use a new deployment directory or schema for a new backend.

A plugin backend is a subclass of ``reef.storage.scenario.ScenarioStorage``.
It implements all abstract methods, including the ``from_config`` class method.
Give its name as ``package.module:ClassName``, or give the name and a configuration:

.. code-block:: yaml

   storage:
     record-backend:
       module: my_package.storage:ObjectStoreScenarioStorage
       config:
         bucket: reef-state
     agent-record-dir: .reef/agent-record

Reef calls ``from_config(config, directory=...)`` with the ``config`` mapping and
``agent_record_dir``. The `Python storage contract
<../reference/python-api.rst#scenario-stores>`__ gives the necessary behavior.
An embedded deployment gives a ``ScenarioStorage`` instance to
``Dispatcher(..., scenario_storage=...)`` instead.

With all backends, each scenario needs one Reef writer.

Choose the artifact backend
---------------------------

The artifact backend stores the releases of each scenario. The default,
``git_lfs``, keeps them in the Git LFS repository that
``storage.artifact-repository`` names. Git and Git LFS must be installed.

To use a different store, name a subclass of
``reef.artifact.ConfigurableRepositoryBackendFactory`` in
``storage.artifact-backend``. Give ``package.module:ClassName``, or give the
name and a ``config`` mapping, as for a storage plugin:

.. code-block:: yaml

   storage:
     artifact-backend:
       module: my_package.artifacts:ObjectStoreArtifactBackend
       config:
         bucket: reef-releases

Reef calls ``from_config(config, context=...)`` to make the backend. The
`Python artifact backend contract <../reference/python-api.rst#artifact-backends>`__
gives the necessary behavior. With a plugin backend, Reef does not use Git LFS
and does not need its tools. Do not set ``storage.artifact-repository``: Reef
refuses that configuration, because the plugin reads its own config.

A change from one backend to another does not move releases. Keep the old
store readable while scenarios refer to its releases, or use a new deployment.
With all backends, each scenario needs one Reef writer.

Choose the task file store
--------------------------

A recipe that generates tasks saves each published task as a Harbor task
directory. The scenario storage keeps the state of the tasks. The task file
store keeps the files of the task directories, which can be text or binary.

The default, ``local``, keeps the files in ``storage.task-file-dir``
(default: ``.reef/task-files``). Put this directory on a persistent volume.
Archiving a scenario moves its files under ``archived/`` in that directory.

To use a different store, name a subclass of
``reef.storage.tasks.TaskFileStore`` in ``storage.task-file-backend``.
Give ``package.module:ClassName``, or give the name and a ``config`` mapping:

.. code-block:: yaml

   storage:
     task-file-backend:
       module: my_package.tasks:ObjectStoreTaskFileStore
       config:
         bucket: reef-tasks
     task-file-dir: .reef/task-files

Reef calls ``from_config(config, directory=...)`` with the ``config`` mapping
and ``storage.task-file-dir``. A remote store can use that directory as a
local cache. The `Python task file store contract
<../reference/python-api.rst#task-file-stores>`__ gives the necessary
behavior.

Import Harbor tasks
-------------------

Each scenario can hold your own Harbor tasks in its train and eval splits, for
example a benchmark that a recipe evaluates on. A task can hold binary files,
such as images. Name a split manifest and the directory of the tasks in the
``task-import`` section:

.. code-block:: yaml

   task-import:
     manifest: /data/benchmark/split.json
     tasks-root: /data/benchmark/tasks

The manifest is a JSON file. It names each task once, in ``train`` or ``eval``:

.. code-block:: json

   {"version": 1, "seed": 0, "eval_fraction": 0.5, "train": ["sum-391"], "eval": ["sort-17"]}

Each name is a Harbor task directory under ``tasks-root``. A name starts
with a lowercase letter or a digit, and holds only lowercase letters, digits,
``.``, ``_`` and ``-``. When a scenario
opens, Reef checks each directory with Harbor's task loader, and adds it to the
scenario's task store ``imported-tasks`` in its split. The task file store
keeps a copy of the files.

The import runs again each time a scenario opens. To add tasks, add them to the
manifest and the directory, and then restart the service or reload the
scenario. To add tasks to one running scenario without a restart, upload them
with ``POST /reef/scenarios/{scenario}/tasks`` (`Upload Harbor tasks
<../reference/http-api.rst#upload-harbor-tasks>`__). A task that is already imported with the same files and split does
not change. A scenario does not open, and the error says why, when:

- a task directory is missing, or Harbor cannot load it;
- an imported task now has other files, or it moved to the other split.

To change an imported task, give it a new name.

A harness evolution recipe evaluates on the imported eval tasks with
``evolution.eval_task_store: imported-tasks`` and an adapter that runs a task
directory, such as ``terminus``. Each step reads the eval split again, so a
task that you import later joins the next step. See `Harness evolution keys
<../reference/configuration.rst#harness-evolution-keys>`__.

Limit the scenarios a service holds
-----------------------------------

A service loads every registered scenario when it starts. By default, each
scenario stays loaded until you delete it. A service also accepts any number
of scenarios. A loaded scenario holds a storage session, its trainers and its
local worker threads. Four ``reef`` settings change this. All of them are off
by default.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Effect
   * - ``scenario-idle-unload-s``
     - Reef unloads a scenario that is idle for this number of seconds.
   * - ``max-loaded-scenarios``
     - The maximum number of loaded scenarios. To load one more, Reef first
       unloads the idle scenario with the oldest use. If no scenario is idle,
       the request gets HTTP 503 with a ``Retry-After`` header.
   * - ``max-scenarios``
     - The maximum number of registered scenarios. A request that would
       create one more gets HTTP 409. To free a place, delete a scenario.
   * - ``scenario-admission``
     - A ``ScenarioAdmission`` plugin. It replaces the two limits above.

.. code-block:: yaml

   reef:
     scenario-idle-unload-s: 3600
     max-loaded-scenarios: 50
     max-scenarios: 1000

An unload keeps all durable state: the records, the commit history, the settings
and the releases. The next request to the scenario loads it again, as after a
restart. Counters in memory, such as the operation metrics, start again from
zero.

A scenario is idle when all of these conditions are true:

- No request, record or local training cycle used it during the idle time.
  Only ``scenario-idle-unload-s`` uses this condition. To free a place for
  ``max-loaded-scenarios``, Reef does not wait for the idle time.
- No request or local training cycle runs on it.
- No trainer has records that it would still train on.
- The storage backend is durable.

A weight-training scenario holds the deployment's training component and its
running job. Only a delete releases them, so Reef never unloads the scenario.

The start-up preload also obeys ``max-loaded-scenarios``. If no loaded
scenario is idle, Reef does not preload the next scenario. That scenario loads
on its first request.

To set a different policy, write a subclass of
``reef.scenario.admission.ScenarioAdmission``. Give its name as
``package.module:ClassName``, or give the name and a ``config`` mapping, as for
a storage plugin. Reef calls ``from_config(config)`` to make the plugin. The
plugin implements two methods:

- ``check_creation(scenario, registered_count=...)``: to refuse a new
  scenario, raise ``reef.core.errors.ScenarioLimitReached`` (HTTP 409).
- ``admit_load(scenario, loaded=...)``: return a ``LoadAdmission``. Its
  ``unload_count`` is the number of loaded scenarios that Reef must unload
  first. Its ``candidates`` are the scenarios that Reef tries, in that order.
  Reef unloads only idle candidates. If too few candidates are idle, Reef
  refuses the load with ``reef.core.errors.ScenarioCapacityFull`` (HTTP 503).
  The plugin can also raise that error itself.

Reef calls the plugin before it creates a scenario and before it loads one.
Reef makes these calls one at a time, so the methods must return quickly.
``max-scenarios`` and ``max-loaded-scenarios`` configure the built-in plugin,
``ScenarioLimits``. If you set ``scenario-admission``, do not set them: Reef
refuses that configuration.

Back up a deployment
--------------------

A backup must hold the scenario state and the artifact repository
(``storage.artifact_repository``). Reef rebuilds ``artifact_work_dir`` and
``artifact_cache_dir`` from the repository, so they do not need a backup.
With a plugin artifact backend, back up the plugin's store in place of the
artifact repository.

.. list-table::
   :header-rows: 1

   * - Backend
     - What to back up
   * - ``sqlite``
     - ``agent_record_dir`` and the artifact repository, copied while the
       service is stopped
   * - ``postgres``
     - A dump of the configured schema and the artifact repository; with a
       remote ``artifact_repository``, the database dump and the Git remote
   * - a plugin
     - The plugin's own store and the artifact repository

With all backends, also back up the task file store: ``task_file_dir`` with
the ``local`` store, or the plugin's own store.

With PostgreSQL, a remote artifact repository and a remote task file store, a
new host with the same configuration continues every scenario from the
database and the two remote stores.

Some recipes have their own directories, for example Reefine's
``step_record_dir`` and ``proposals_dir``. These directories are not scenario
storage. Keep them on a persistent volume.
