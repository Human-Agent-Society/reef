Operate a deployment
====================

Check deployment health, select served artifacts and recover training jobs.
Start with a running `weight <evolve-your-model.rst>`__ or
`harness <evolve-your-harness.rst>`__ deployment.

Set ``REEF_URL`` and ``REEF_TOKEN`` for the commands below. Use
`split services <split-services.rst>`__ for several processes and
`storage <storage.rst>`__ for backups and upgrades.

Check health and status
-----------------------

.. code:: bash

   curl -f "$REEF_URL/healthz"
   curl -sS -H "Authorization: Bearer $REEF_TOKEN" "$REEF_URL/reef/status"

``/healthz`` checks HTTP readiness, not training progress. ``/reef/status``
shows asynchronous errors, loaded scenarios, served artifacts and training jobs.
Check ``last_training_commit`` and ``processor`` when training appears stuck.
A skipped or rejected update differs from a job that is still running.

Managed services write ``<service>.log`` and ``<service>.pid`` under
``reef.run-dir`` (default ``.reef/run/``).

Read the artifacts
------------------

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

The list shows immutable outputs, newest first. Treat ``artifact_id`` as an
opaque string.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``active``
     - The component's selected artifact.
   * - ``activated``
     - The artifact has served as active.
   * - ``served``
     - The artifact serves beside the active artifact.
   * - ``operation: creation``
     - A starting artifact; ``source`` identifies its origin.

Skipped and rejected jobs create no artifact row. Read
``GET /reef/scenarios/{scenario}/operations`` for all operations.
See `Artifacts <../reference/http-api-artifacts.rst#http-api-artifacts>`__
for the complete response.

For harness trees, use ``GET /reef/harness/artifacts`` and
``GET /reef/harness?artifact_id=<id>``. An artifact awaiting activation has
``active: false`` and ``activated: false``.

Starting artifacts
------------------

A new scenario gets one starting artifact per component. It serves that artifact
and uses it for the first training job. An explicit artifact item overrides
the component's default start.

.. list-table::
   :header-rows: 1

   * - Backend
     - Default start
   * - Slime or Tinker
     - The configured base model and adapter settings, without copied weight files.
   * - ``spindle_full``
     - The configured Hugging Face base revision.
   * - Harness
     - ``evolution.seed``, the recipe's seed, or an empty tree.

Existing scenarios keep their stored starts after a configuration change.
A start incompatible with the new configuration fails that scenario.
Slime accepts only the inputs its workers loaded from ``inference.model-path``.
See `Train with Tinker <tinker.rst#jobs-checkpoints-and-recovery>`__ for
adapter initialization and seed behavior.

Choose a scenario's start
-------------------------

Name an artifact item owned by the new scenario's owner:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "code-repair-2", "owner": "acme", "artifact_id": "code-repair/<artifact_id>"}' \
     "$REEF_URL/reef/scenarios"

The item can be an existing scenario artifact or an uploaded artifact item.
Add ``component`` for a scenario with several components. Other components use
their defaults. The response's ``components`` gives their starting artifact IDs.

Reef checks access, file layout and backend support before registering the scenario.
An unusable start leaves the name available. Missing items return 404; invalid
layouts or unsupported starts return 400. Slime refuses artifact-item starts,
including in a composite scenario: use its default start.

An existing scenario ignores ``artifact_id`` and ``component`` in a create request.
Inference never creates a scenario. Reports and record imports can create one
from defaults when implicit creation is enabled.

Import an artifact
------------------

Import weights or a harness tree from a URI the service can read:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"source": "s3://my-bucket/checkpoints/step-400/"}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

.. list-table::
   :header-rows: 1

   * - Source
     - Requirements
   * - ``hf://org/model@revision``
     - A full 40-digit commit SHA remains a remote reference. Branches and tags are copied.
   * - ``s3://bucket/prefix/``
     - Install ``reef-infra[s3]`` and supply boto3 credentials through its standard environment, profile or role.
   * - ``file:///absolute/path``
     - The directory must be on the service's host.

The files must match the component's layout. A failed layout check returns
400 and changes nothing. The `import API
<../reference/http-api-artifacts.rst#import-an-artifact>`__ lists supported layouts.
Restrict import access to callers allowed to read those sources and train the scenario.

An import creates an artifact but does not activate it or change the training
start. Copied imports need cache and artifact-store space, approximately twice
the model size. A pinned Hugging Face reference downloads weights when needed;
keep the referenced repository available.

Harness sync differs: ``POST /reef/harness/import`` merges changed files and
makes the result both active and the next training start.

Set what a scenario serves
--------------------------

Set artifact IDs and integer traffic weights from 0 to 100:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {"<id>": 100}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

Nonempty weights must sum to 100. The highest-weight artifact is active.
The mapping replaces the component's previous selection.

.. list-table::
   :header-rows: 1

   * - Goal
     - ``served``
   * - Activate ``b`` and stop other artifacts
     - ``{"b": 100}``
   * - Send 5% of traffic to ``b``
     - ``{"a": 95, "b": 5}``
   * - Serve ``b`` only when a request names it
     - ``{"a": 100, "b": 0}``
   * - Stop ``b``
     - ``{"a": 100}``
   * - Stop all serving; keep training
     - ``{}``

Add ``component`` when several components exist. An empty mapping without
``component`` stops serving for the whole scenario. Inference then returns 404.
The update never changes the training start. Read the selection from
``GET /reef/scenarios/{scenario}/settings``.

Startup is asynchronous. Poll
``GET /reef/scenarios/{scenario}/inference/status`` before using a new active artifact.
For an artifact beside it, add ``?model=<artifact-id>``.
Resend the mapping to retry a failed startup immediately.
Selections survive restarts.

A served artifact also has its own inference URL:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model": "reef", "messages": [{"role": "user", "content": "hi"}]}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts/<id>/v1/chat/completions"

See `Traffic split <../reference/http-api-serving.rst#traffic-split>`__ for
conversation routing and explicit artifact selection.

**Capacity.** Tinker can serve any of a scenario's artifacts. Reef's adapter
serving supports one artifact per scenario. Full-weight serving needs one
inference component per checkpoint; configure additional components with
``inference.remote.slot-urls``.

Traffic-weight changes reuse loaded checkpoints. Removing a checkpoint schedules
its component for retirement after ``inference.timeout-s`` plus 5 seconds
(default 305 seconds). Running work or release failures can delay retirement;
no API reports when the component becomes free.

To replace ``a`` while two components serve ``a`` and ``b``:

#. Send ``{"served": {"b": 100}}``.
#. Allow retirement time after the new routes are published, not after HTTP 200.
#. Add ``c`` with ``{"served": {"b": 100, "c": 0}}`` and poll ``c``'s startup status.

If no component is free, startup fails and the previous routes remain.
Wait, then resend the mapping. The `serving API
<../reference/http-api-serving.rst#set-what-a-scenario-serves>`__ describes limits and errors.

Activation rule
---------------

The deployment's ``activation`` section controls new artifacts:

.. list-table::
   :header-rows: 1

   * - Rule
     - Effect
   * - ``always`` (default)
     - Activates new artifacts, except harness changes held for review by ``review_kinds``.
   * - ``never``
     - Requires manual activation through ``served``.
   * - ``evaluated``
     - Requires an evaluation to reach ``min_mean_score``.

Override the rule for a scenario with
``{"activation": {"module": "never"}}`` in its update request.
Set ``activation`` to null to use the deployment rule again.
An inactive training artifact can still become the next training start.
See `Activation configuration
<../reference/configuration-recipe.rst#the-activation-section>`__.

Train from an artifact
----------------------

Change where subsequent training starts:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

Add ``component`` for a composite scenario. This cancels the job continuing
the previous training start, changes the next default start and begins a new
tracking run. It does not change serving.

Tinker restores the artifact's weights and optimizer state. Harness training
starts from its tree. An artifact without optimizer state starts a new optimizer.
Slime refuses ``train-from`` with HTTP 400 because its workers continue their
latest checkpoint.

A `manual training request <../reference/http-api-training.rst#manual-training>`__
can instead name ``train_from`` for that job alone. It does not change the
default training start, and a later train-from does not cancel that job.

To resume an earlier Slime checkpoint, create a new deployment and scenario:

- Give it new ``save`` and ``save-hf`` directories.
- Set ``training.options.load`` to the old checkpoint directory.
- Set ``training.options.ckpt-step`` to the saved rollout.

Do not move an existing scenario to this deployment. Its recorded training
start names the old deployment's checkpoint and will be refused.

Go back to an older artifact
----------------------------

Restore all active components from a previous operation:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" -d '{"commit_number": 12}' \
     "$REEF_URL/reef/scenarios/code-repair/restore"

Read commit numbers from ``GET /reef/scenarios/code-repair/operations``.
Commit 0 selects the starting artifacts. Restore creates no artifact and
leaves training starts unchanged. Poll inference startup before sending requests.

With automatic activation, the next training artifact can replace this selection.
To keep the older artifact serving, select ``activation: {module: never}``.
Where supported, train-from can also make later training descend from it.

Set the checkpoint cadence
--------------------------

Set ``recipe.config.job_parameters.save_interval`` for intermediate saves.
Tinker and Slime sample jobs count optimizer steps. Slime task and dataset jobs
count rollout iterations. Both backends also save at job end; without an
interval, they save only there. Activation decides what serves; training
continues from the last save.

Slime saves a Megatron checkpoint and an HF export, or a PEFT adapter for LoRA.
Its native ``save-interval`` flag does not control Reef's cadence, but Megatron
requires it when ``save`` is set. ``critic-save-interval`` defaults to 1 saved
rollout between critic checkpoints.

Slime retention uses these ``training.options``:

- ``reef-checkpoint-policy``: ``latest`` or ``best_reward``.
- ``reef-checkpoint-max-storage-fraction``: default 0.8.
- ``reef-checkpoint-min-free-space-fraction``: default 0.1.
- ``reef-checkpoint-max-storage`` and ``reef-checkpoint-min-free-space``:
  absolute size limits.

Retention preserves the next job's start. A job fails when the next checkpoint
cannot fit. Its error appears in the log and ``training_job`` commit metric.
A failed job's submitted samples remain consumed; the next job uses the last
saved checkpoint.

Track experiments with W&B
--------------------------

Tracking is off by default. Configure ``observability.wandb`` and export
``WANDB_API_KEY`` before startup. See the `tracking example
<../reference/configuration-observability.rst#experiment-tracking>`__.
Reef refuses Slime's ``--wandb-key``; credentials do not belong in the config.

Each scenario has one group and one run. Train-from starts a new run;
activation does not. Training results use ``train/step`` and include the
commit number and job ID. Composite trainers prefix metrics by component.

``operations/*`` metrics arrive every 10 seconds, including for inference-only
scenarios. See `Operational metrics
<../reference/configuration-observability.rst#operational-metrics>`__ before
interpreting queue counts or waiting times.

Tracking runs in the background. A tracking failure is logged and does not
fail training or its commit. Shutdown flushes queued events.

Restart and recovery
--------------------

Keep the deployment's `stores and backend state <storage.rst#back-up-a-deployment>`__
on persistent storage. On restart:

- Reef restores committed settings, artifact selections and record progress.
- Training resumes the submitted job from its last saved checkpoint.
  Unsaved work can run again.
- Checkpoints with committed evaluation decisions are not evaluated again.
- A settled harness step commits its durable result without rerunning.
  An unsettled step reruns from the same start and harness state.

One control instance holds the deployment's lease. A normal stop releases it;
a crash can delay takeover until its 30-second lease expires. Use separate
stores for independent deployments. See `Control restarts
<split-services.rst#control-restarts>`__ for standby instances.

**Repository synchronization.** ``artifact_head_sync`` in status is
``synchronized``, ``pending`` or ``conflict``. A conflict means another writer
moved the repository head to an unrelated snapshot. Reef does not overwrite it.
The committed operation remains successful, but later commits wait for
synchronization. Reef retries before the next commit and after a restart.

**Deletion.** ``DELETE /reef/scenarios/{scenario}`` removes the scenario from
use immediately; its ID cannot be reused. Background cleanup cancels jobs,
archives state and removes cached or discarded content. Failed cleanup retries
at the next hourly sweep, including after restart. Harness history moves under
an ``archived/`` sibling when the training service handles the request.

Run the services in separate processes
--------------------------------------

Use ``reef serve --service NAME`` to scale services separately or keep inference
running during control restarts. See `Split services <split-services.rst>`__
for shared-store requirements and request routing.

Import Harbor tasks
-------------------

Configure task directories and a train/eval split:

.. code:: yaml

   task-import:
     manifest: /data/benchmark/split.json
     tasks-root: /data/benchmark/tasks

.. code:: json

   {"version": 1, "seed": 0, "eval_fraction": 0.5, "train": ["sum-391"], "eval": ["sort-17"]}

Each name identifies a Harbor directory under ``tasks-root``. Names start with
a lowercase letter or digit and contain only lowercase letters, digits,
``.``, ``_`` and ``-``. Binary files are allowed.

Reef imports tasks when the scenario loads. To add tasks or change splits,
update the manifest and reload the scenario. Unchanged tasks stay unchanged.
Give changed task files a new name. Missing directories, invalid Harbor tasks,
changed imported files or conflicting sample names prevent the scenario from loading.

For a running scenario, use
``POST /reef/namespaces/{owner}/data/tasks`` instead of restarting.
See `Create task items <../reference/http-api-data-upload.rst#create-task-items>`__.

Harness evaluation uses the eval split when
``evolution.evaluation.imported_tasks: true``. Each step reads that split again,
so later imports join the next evaluation. See
`Task directories <evolve-your-harness.rst#task-directories>`__.

Limit the scenarios a service holds
-----------------------------------

By default, Reef loads all registered scenarios and imposes no count limit.
For persistent scenarios, configure:

.. code:: yaml

   reef:
     scenario-idle-unload-s: 3600
     max-loaded-scenarios: 50
     max-scenarios: 1000

``scenario-idle-unload-s`` unloads idle scenarios after the stated time.
``max-loaded-scenarios`` unloads the oldest idle scenario before loading another.
If none is idle, the request returns 503 with ``Retry-After``.
``max-scenarios`` refuses additional registrations with 409; delete a scenario
to free a place. These settings are off by default.

Unloading preserves durable state. The next request reloads it; in-memory
metrics restart at zero. A scenario is idle only without active requests,
local training or pending training records. Weight-training scenarios are never
unloaded; deletion releases them.

``reef.scenario-admission`` selects a custom policy and cannot accompany the
two count limits. Its contract is in
``reef.control.scenario.admission.ScenarioAdmission``.
