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

Artifacts appear newest first. IDs are opaque; ``active``, ``activated`` and
``served`` describe their serving history. Skipped and rejected jobs appear
only in ``GET /reef/scenarios/{scenario}/operations``.

Use ``GET /reef/harness/artifacts`` for harness step results.
See `artifact fields <../reference/http-api-artifacts.rst#http-api-artifacts>`__.

Starting artifacts
------------------

A new scenario starts each component from its configured base model or harness
seed. An explicit artifact item overrides that default. Existing scenarios
keep stored starts; incompatible configuration changes fail the scenario.
See `backend starts <../reference/http-api-artifacts.rst>`__ and
`Tinker initialization <tinker.rst#jobs-checkpoints-and-recovery>`__.

Choose a scenario's start
-------------------------

Name an artifact item owned by the scenario's owner:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "code-repair-2", "owner": "acme", "artifact_id": "code-repair/<artifact_id>"}' \
     "$REEF_URL/reef/scenarios"

Add ``component`` for a composite scenario. Missing items return 404;
invalid or unsupported starts return 400 without reserving the name.
Slime requires its default start and rejects artifact-item starts.

Existing scenarios ignore start fields. Inference never creates scenarios;
reports and record imports can, when implicit creation is enabled.

Import an artifact
------------------

Import from a URI the service can read:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"source": "s3://my-bucket/checkpoints/step-400/"}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

Imports create artifacts without changing serving or training starts.
Invalid layouts return 400. Supported sources:

- ``hf://org/model@revision``: a full commit SHA stays a remote reference;
  branches and tags are copied.
- ``s3://bucket/prefix/``: requires ``reef-infra[s3]`` and boto3 credentials.
- ``file:///absolute/path``: files on the service's host.

Copied imports need approximately twice the model size across cache and storage.
Pinned references require the source repository to remain available.
See `layouts and access <../reference/http-api-artifacts.rst#import-an-artifact>`__.
Harness sync instead merges files and changes both serving and the training start.

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

Use an artifact's dedicated URL to select it explicitly; see
`Traffic split <../reference/http-api-serving.rst#traffic-split>`__.

**Capacity.** Tinker can serve several artifacts per scenario; adapter engines
support one. Full weights need one component per checkpoint, configured through
``inference.remote.slot-urls``.

Removing full weights retires their component after ``inference.timeout-s``
plus 5 seconds (default 305), measured from route publication. Work or release
failures can delay this. No API reports when the component becomes free.
If replacement startup fails, wait and resend the mapping. See
`replacement procedure <../reference/http-api-serving.rst#set-what-a-scenario-serves>`__.

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

Tracking is off by default. Set ``observability.wandb`` and export
``WANDB_API_KEY``; see the `tracking example
<../reference/configuration-observability.rst#experiment-tracking>`__.
Reef rejects Slime's ``--wandb-key``.

Each scenario has one group and run. Train-from starts a new run; activation
does not. Tracking failures are logged without failing training.
See `operational metrics
<../reference/configuration-observability.rst#operational-metrics>`__ for queue and timing data.

Restart and recovery
--------------------

Keep `stores and backend state <storage.rst#back-up-a-deployment>`__ persistent.
Reef restores committed selections and training progress. Jobs resume from
saved checkpoints; unsaved work can run again. Settled harness steps commit
their durable result, while unsettled steps rerun.

``artifact_head_sync`` reports ``synchronized``, ``pending`` or ``conflict``.
A repository conflict does not undo a committed operation, but blocks later
commits until synchronization succeeds. See the `recovery contract
<../advanced_topics/state-model.rst#commit-ordering>`__.

**Deletion.** ``DELETE /reef/scenarios/{scenario}`` removes the scenario immediately;
its ID cannot be reused. Background cleanup cancels jobs and archives state.
Failed cleanup retries hourly, including after restart.
See `Control restarts <split-services.rst#control-restarts>`__ for failover.

Run the services in separate processes
--------------------------------------

Use ``reef serve --service NAME`` to scale services separately or keep inference
running during control restarts. See `Split services <split-services.rst>`__
for shared-store requirements and request routing.

Import Harbor tasks
-------------------

Configure a task directory and split manifest:

.. code:: yaml

   task-import:
     manifest: /data/benchmark/split.json
     tasks-root: /data/benchmark/tasks

.. code:: json

   {"version": 1, "seed": 0, "eval_fraction": 0.5, "train": ["sum-391"], "eval": ["sort-17"]}

Names identify Harbor directories under ``tasks-root``. Reef imports them when
the scenario loads. Update the manifest and reload to add tasks or change
splits; changed files need new names. Invalid tasks, changed imported files
or conflicting names prevent loading.

For running scenarios, use the `task upload API
<../reference/http-api-data-upload.rst#create-task-items>`__.
Harness evaluation rereads the eval split each step when
``evolution.evaluation.imported_tasks: true``.

Limit the scenarios a service holds
-----------------------------------

By default, Reef loads all registered scenarios and imposes no count limit.
For persistent scenarios, configure:

.. code:: yaml

   reef:
     scenario-idle-unload-s: 3600
     max-loaded-scenarios: 50
     max-scenarios: 1000

Idle limits unload persistent scenarios; the next request reloads them.
If no loaded scenario is idle, admission returns 503 with ``Retry-After``.
The registered-scenario limit returns 409; deletion frees a place.
Weight-training scenarios never unload automatically.

These limits default to off. ``reef.scenario-admission`` selects a custom
policy and cannot accompany the count limits. See `service settings
<../reference/configuration-settings.rst#service>`__ for idle conditions and options.
