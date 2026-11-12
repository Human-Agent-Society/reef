Operate a deployment
====================

Check a deployment, select its artifacts, follow training, and recover after a restart.

:For: whoever runs a Reef deployment
:Before you start: a running deployment from `Evolve your model <evolve-your-model.rst>`__ or `Evolve your harness <evolve-your-harness.rst>`__, its URL and token
:You will have: the routine checks, the version operations, and the recovery rules

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
- the artifact that answers the scenario's requests (``served_artifact_id``)
- whether a job's data is waiting, and the processor's state
- whether inference is admitted or paused while a snapshot loads
- for a weight-training scenario, its running training jobs (``training_jobs``) and the number of jobs that ended, finished or failed (``finished_jobs``)

The committed outcome includes the recipe-owned metrics. You can therefore tell a skipped or rejected update from one that is still running. Look here first when requests keep being served by an old version.

The service and every process ``reef serve`` started write logs under ``reef.run-dir`` (``.reef/run/`` by default), one ``<service>.log`` and one ``<service>.pid`` each.

Read the artifacts
------------------

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

An artifact is one immutable output, such as a checkpoint's weights or a harness tree. Its ``artifact_id`` is the only identifier callers need. Keep the ID as an opaque string: do not parse a Git revision or component name from it.

Training, imports and harness syncs make artifacts. The list is newest first, with one row per artifact. ``active`` marks the selected artifact of each component. ``activated`` marks artifacts that have served as active. ``served`` marks artifacts that the scenario serves beside the active artifact (`Set what a scenario serves`_). The ``creation`` rows hold the starting artifacts (`Starting artifacts`_). Their ``source`` says where they came from. See `Artifacts <../reference/http-api.rst#artifacts>`__ for the fields.

A training step that made no artifact, such as a rejected or skipped step, has no row. Neither do activations, train-from operations and changes to the served artifacts. ``GET /reef/scenarios/{scenario}/operations`` lists every step (`Record and commit history <../reference/http-api.rst#record-and-commit-history>`__).

For harness scenarios, ``GET /reef/harness/artifacts`` lists the harness artifacts. Pull a listed tree with ``GET /reef/harness?artifact_id=<id>``. An artifact waiting for activation has ``active: false`` and ``activated: false``. The older ``/reef/harness/artifacts`` route remains for clients that need its step pages.

Starting artifacts
------------------

Reef creates a starting artifact for each component when it creates a scenario. Each has its own ``artifact_id`` and an ``operation: creation`` row. The scenario selects these artifacts for serving and as the trainers' first starts.

The starting artifacts come from one of two sources:

- The deployment's default start, when the create request names no artifact item. Reef writes it from the configuration:

  - The ``slime`` backend with full weights writes the model that ``inference.model-path`` names. Each scenario's starting artifact holds a full copy of the model's files, about 2 bytes per parameter in bf16. A Git LFS remote stores each file once.
  - The ``slime`` backend with LoRA writes an adapter whose weights are zero. It serves the model unchanged.
  - The ``tinker`` backend writes the checkpoint of its untrained adapter, which stays on the Tinker service.
  - A harness recipe writes the files of its seed tree: ``evolution.seed``, else the recipe's own seed. Without a seed, the starting artifact is empty.
  - A recipe of several components writes each component's default start under its name.
- An artifact item of the scenario's owner (`Choose a scenario's start`_).

The creation row's ``source`` is ``{"kind": "default", "inputs": {...}}``, with the configuration values that decide the default start, such as the model and the seed's checksum. For an item, it is ``{"kind": "item", "item": ..., "uri": ..., "version": ...}``, as an import records it.

A new trainer starts its first job from its starting artifact. A train-from or the trainer's own training step changes the start later (`Train from an artifact`_). Activate that artifact to serve it again, or train from it to start training again from it; the ``slime`` backend refuses train-from, as it refuses every artifact.

A change of the configuration does not change an existing scenario. It keeps its stored starting artifacts, and a new scenario gets the new default start. A start that no longer fits the configuration fails only its own scenario: the runtime refuses to activate it, or the training component refuses to start from it, with that component's message. The ``slime`` backend trains only the weights that its workers load from ``inference.model-path``. It refuses to create a scenario from an artifact item. A scenario whose starting artifact came from other inputs does not load, and the error names both inputs. Reef refuses a scenario that an earlier Reef stored with a shared base snapshot (``reef-scenario/4`` metadata). Create such a scenario again.

Choose a scenario's start
-------------------------

The default inference routes answer from the scenario's active selection. No request header keeps a client on an older snapshot. To keep an older artifact answering, serve it beside the active artifact (`Set what a scenario serves`_).

A new scenario uses the default starting artifacts. To start it from another artifact, name an artifact item of the scenario's owner in ``artifact_id`` when you create the scenario:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "code-repair-2", "owner": "acme", "artifact_id": "code-repair/<artifact_id>"}' \
     "$REEF_URL/reef/scenarios"

The owner's artifact library holds the item ``<scenario>/<artifact_id>`` for each artifact that a scenario of the owner made. An artifact from an item keeps its artifact ID in each scenario, and it gets no second item. The library also holds the artifact items that you stored, such as an upload to S3 or a Hugging Face snapshot (`Data items <../reference/http-api.rst#data-items>`__). Reef reads the item's files and checks them as `Import an artifact`_ describes. Then it stores them, by reference or as a copy, as the component's starting artifact. The item is thus the active artifact and the training start of its component. Add ``component`` when the scenario has several components; each other component holds its default start. The answer's ``components`` maps each component to its starting ``artifact_id``. A scenario with one component also returns that ID as ``artifact_id``.

Before registration, Reef checks that every training component accepts its starting artifact as a training start. A composite with Slime refuses an item start even when the item fills another component. Use the default start for that composite.

When Reef cannot read the item, or a layout or training-start check refuses it, Reef registers no scenario. It answers with the status of the import, such as HTTP 400 for a directory without the component layout, or HTTP 404 for an item that the owner does not store. The scenario name remains available for a valid creation request. A scenario that already exists ignores ``artifact_id`` and ``component``.

A report or a record import creates a missing scenario from the default start. A request header does not select the starting artifact. An inference response identifies the answering artifact with ``x-reef-artifact-id``. An inference request never creates a scenario.

To pull a specific harness artifact, add ``?artifact_id=<id>`` to the harness manifest route. To serve an earlier artifact again, make it the active artifact as `Set what a scenario serves`_ describes.

Import an artifact
------------------

To serve or train an artifact that was made outside Reef, such as an adapter that you fine-tuned elsewhere, a checkpoint on S3, or a harness that you wrote, import it by a URI that Reef reads:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"source": "s3://my-bucket/checkpoints/step-400/"}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

Reef reads ``hf://org/model@revision``, ``s3://bucket/prefix/`` and ``file:///absolute/path``, a directory on the service's machine. To import a directory from your own machine, upload it to one of these places first. S3 needs the ``s3`` extra (``uv pip install 'reef-infra[s3]'``) and reads with boto3's standard credentials: the environment, a profile or an instance role. Reef stores no credential. A reader can read whatever the service can read, so only give import access to callers that may train the scenario. ``storage.artifact-sources`` adds readers for other schemes (`Python API <../reference/python-api.rst#artifact-imports>`__).

The directory must have the layout of the scenario's component artifacts: the training component's weight files, a Tinker manifest whose saved state is on the configured service, or a harness source tree whose files are UTF-8 text and pass validation for the adapter. Otherwise Reef answers HTTP 400 and changes nothing. `Import an artifact <../reference/http-api.rst#import-an-artifact>`__ in the HTTP API lists each layout and status.

The import adds an ``import`` row to the artifact list, with the URI and the version that Reef read in ``source``. It does not serve the artifact and does not change a training start, and the activation rule does not run. Activate it, or train from it, as any other artifact. Reef copies the files into the artifact store, so a later change at the source does not change the artifact. An import needs the model's size in disk space twice while it runs: once under ``storage.artifact-cache-dir`` and once in the artifact store.

To import model weights from Hugging Face without a copy, name a full commit SHA: ``hf://org/model@<40-digit commit>``. Reef then keeps a reference to that revision, and downloads its weights only when a serving engine loads them. A commit names fixed files, so the artifact cannot change, but it can be lost: keep the repo on the Hub while you serve or train the artifact. A branch or a tag can move, so Reef copies those.

A harness sync is different. ``POST /reef/harness/import`` merges the files that a person's session changed onto the active selection. It commits the result as a ``sync`` row. In that operation the new artifact becomes the active selection and training start of its component (`Evolve your harness <evolve-your-harness.rst>`__).

Set what a scenario serves
--------------------------

A scenario serves one or more artifacts of each component. One of them is the active artifact: harness routes read the active tree, and the default inference routes answer from the active artifact that completed startup. To change what the scenario serves, send the ``served`` setting, which maps artifact IDs of one component to weights:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {"<id>": 100}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

The weights are integers from 0 to 100 that sum to 100. Afterwards the scenario serves exactly the artifacts of the mapping. The artifact with the largest weight is the active artifact. These examples use the artifacts ``a`` and ``b`` of one component, where ``a`` is active before the request:

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
     - ``b`` answers 5% of the model requests. One conversation stays on one artifact.
   * - Serve an artifact beside the active one
     - ``{"a": 100, "b": 0}``
     - Reef loads ``b``, which answers only the requests that name it.
   * - Stop an artifact
     - ``{"a": 100}``
     - Reef stops serving ``b``.
   * - Serve nothing
     - ``{}``
     - Reef stops every engine route of the scenario. Its inference requests answer HTTP 404.

Each ID is any row of the artifact list. Leave out ``component`` when the scenario has one component. A scenario of several components requires ``"component": "<name>"`` beside ``served``, and an unknown component is HTTP 400. The answer contains ``scenario`` and the ``served`` mapping. ``GET /reef/scenarios/{scenario}/settings`` returns ``served`` for each component.

An empty mapping, ``{"served": {}}`` without ``component``, makes the whole scenario serve nothing. Reef then starts no engine for it, and training still runs. ``POST /reef/scenarios`` with ``"serves": false`` creates a scenario in this state. The scenario serves again when ``served`` names an artifact (`Serve nothing <../reference/http-api.rst#serve-nothing>`__).

The response accepts the mapping. Inference then loads the artifacts in the background. Poll the `startup status <../reference/http-api.rst#startup-status>`__ before using a new active artifact. Repeating the request retries a failed startup at once.

Reef appends an ``activate``, ``serve`` or ``stop_serving`` operation for each artifact that changes. It creates no artifact or repository revision. For example, changing the active ``harness`` from artifact B to C changes ``{weights: A, harness: B}`` to ``{weights: A, harness: C}``. The weights keep artifact A and its identity. Each operation stores the complete selection, so recovery restores the same combination. History is not rewritten, and the step counter keeps increasing. The request never changes a trainer's training start.

**Traffic split.** With more than one weight above 0, Reef divides the model requests that name no artifact by the weights. Reef starts each artifact of the mapping that does not serve yet, and the active artifact answers its requests until it has started. `Traffic split <../reference/http-api.rst#traffic-split>`__ describes how Reef keeps a conversation on one artifact and which requests do not use the weights.

**One artifact's own URL.** Each served artifact also answers at its own URL. Use weight 0 to keep an older artifact answering while a newer one is active, or to try a new artifact before it is active:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model": "reef", "messages": [{"role": "user", "content": "hi"}]}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts/<id>/v1/chat/completions"

A call to the artifact's URL is recorded with the ``artifact_id`` that answered. Choose which clients call each URL, for example with a gateway. The served artifacts survive a restart.

**Runtime limits.** To serve several artifacts, the runtime must answer each call from the artifact that the call names. A Tinker scenario can serve any of its artifacts together. A runtime that loads weights into engines serves one artifact, so Reef refuses a mapping of several artifacts there. A full-weight serving runtime that makes its starting artifact active after a checkpoint loads the starting artifact's files from the artifact repository. `Set what a scenario serves <../reference/http-api.rst#set-what-a-scenario-serves>`__ in the HTTP API lists the answers and status codes.

Activation rule
---------------

The deployment's `activation section <../reference/configuration.rst#the-activation-section>`__ names the rule that decides whether Reef activates each new artifact:

- ``always``, the default, activates each new artifact when the training step makes it. Its ``review_kinds`` holds back a harness tree that changes one of the listed tree kinds.
- ``never`` makes each new artifact but does not activate it. A person activates it with ``served``.
- ``evaluated`` activates an artifact once an evaluation of it reaches ``min_mean_score``.

A scenario can select one of these rules for itself with ``POST /reef/scenarios/{scenario}/update`` and ``{"activation": {"module": "never"}}``. Send ``"activation": null`` to follow the deployment's rule again. ``GET /reef/scenarios/{scenario}/settings`` reads back the scenario's rule and the rule that applies (`Scenario updates <../reference/http-api.rst#scenario-updates>`__).

An artifact that the rule does not activate still becomes the start of the next training job.

Train from an artifact
----------------------

Each trainer continues from its own newest output: a weight trainer from its newest saved checkpoint, a harness trainer from the tree it kept. That artifact is the trainer's training start. A job whose request names its own start with ``train_from`` is the exception: it does not move the training start. To make a trainer continue from another artifact, set its training start:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

Reef cancels the running job that continues the training start; a job with its own start keeps running. The next training job that names no start, or the next evolve step, starts from the artifact. One training request can also name its own start with ``train_from`` (see `Manual training <../reference/http-api.rst#manual-training>`__). Train-from never changes the active selection. Leave out ``"component"`` when the scenario has one component. A scenario of several components requires ``"component": "<name>"`` in the body, and an unknown component is HTTP 400. A weight artifact without optimizer state, such as a starting artifact made from the default configuration, starts training with a new optimizer state. Each train-from also starts a new run in experiment tracking.

Whether a trainer can start from an artifact depends on the backend:

- The ``tinker`` backend imports the artifact's checkpoint manifest, so a Tinker scenario can train from any of its artifacts. The import includes the saved training state with the optimizer state.
- The ``slime`` backend refuses every artifact with HTTP 400, because its workers continue from their own latest checkpoint.
- A harness trainer starts its next evolve step from the artifact's tree.

Reef cannot move the training of a Slime scenario back to an earlier checkpoint, although it can activate one of its artifacts. To train again from an earlier checkpoint, start a new deployment for a new scenario:

- Give the deployment its own ``save`` and ``save-hf`` directories. The Slime component does not overwrite a rollout that it saved before.
- Point Slime at the old checkpoint with its own options in ``training.options``: ``load`` names the old ``save`` directory, and ``ckpt-step`` names the saved rollout.

The new scenario's first job continues from the weights that the workers loaded. An existing scenario cannot move to the new deployment: its committed start names a checkpoint of the old one, so the Slime component refuses its jobs.

Go back to an older artifact
----------------------------

For a scenario with several components, restore an earlier complete selection:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" -d '{"step": 12}' \
     "$REEF_URL/reef/scenarios/code-repair/restore"

Choose a step from ``GET /reef/scenarios/code-repair/operations``, or use step 0 for the starting artifacts. Reef restores all active component IDs together and appends a ``restore`` operation. It creates no artifacts and leaves the training starts unchanged. The answer's ``components`` gives the restored mapping. Poll startup status before sending inference requests.

Activation alone does not last under the default rule. The trainer still continues from its training start, and Reef activates the next artifact that it makes. To go back to an older artifact for good, activate it and do one of these:

- Send ``train-from`` with the same artifact, so that the next artifact descends from it.
- Switch to manual activation, so that no new artifact serves until a person activates it.

Set the checkpoint cadence
--------------------------

Every checkpoint that a job saves becomes an artifact. Without an ``evaluation`` section, Reef activates each of them. The next job always starts from the last saved checkpoint, whether or not evaluation selected it.

How often a job saves depends on the backend:

- A Slime or Tinker job saves a checkpoint every ``save_interval`` optimizer steps, and at its end. ``save_interval`` is a job parameter: ``recipe.config.job_parameters`` sets it. Without it, a job saves one checkpoint at its end.
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
     - in the Git LFS repository; Reef serves the active selection that the commit history names
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
   * - a harness step in flight
     - resumed, as any training job: a step's outcome is durable before its
       checkpoint, so a step that settled commits after the restart without
       running again, and a step that did not settle runs again from the
       same start and harness state

A process that stops normally releases the control lease. After a crash, the
next process waits until the old lease expires (at most 30 seconds) before it
loads scenarios; see `Run the services in separate processes`_.

After a step commits, ``/reef/status`` reports its scenario's
``artifact_head_sync`` with the repository synchronization state and any
``error``. Repository revisions stay internal.
The ``state`` is ``synchronized`` when the backend head names the chain tip,
``pending`` when updating it failed, or ``conflict`` when another writer moved it to an
unrelated snapshot. The committed step remains successful. Before another commit,
Reef retries synchronization; if it still fails, the new commit stops before
publishing or writing its commit record. A conflict never overwrites the other
writer's head. Restart also synchronizes the head before loading the scenario.

The scenario storage and the repository live under ``.reef/`` by default (``agent_record_dir``, ``task_file_dir``, ``artifact_repository``, ``artifact_work_dir``, ``artifact_cache_dir``). On ephemeral storage none of the guarantees above hold past its loss. Each scenario needs one control service; run a second deployment on other ports and storage paths rather than two deployments on one store. The services of one deployment share its store; see `Run the services in separate processes`_. `State model <../advanced_topics/state-model.rst#commit-ordering>`__ describes the commit ordering behind the table. ``DELETE /reef/scenarios/{scenario}`` retires a scenario: the storage backend archives its records, commits, settings and task files, its history and step records move under an ``archived/`` sibling and its repository ref is renamed into ``refs/reef/archived/``, so a name can be reused without the old chain. Deleting a weight-training scenario cancels its running training job.

Run the services in separate processes
--------------------------------------

A deployment has seven services. By default, ``reef serve`` runs all of them in
one process. Run them in separate processes when inference must stay up while
control restarts or deploys a new version, or when one service needs more
replicas than the others. The services do not send work to each other: each one
reads and writes the deployment's shared store, and takes its work from a queue
there. No service calls another, and control calls no engine: it queues serving
work, and inference starts the artifact.

.. figure:: ../assets/services-light.svg
   :alt: The seven services of a deployment around the shared store. Inference reads served_models and appends records, and one inference worker owns the engines. Control commits and queues work for the other services. Data, harness, samples, training and evaluation take work from their queues and write their results back to the store.

   The services and the shared store. Each edge reads or writes the store;
   ``python3 docs/assets/render_service_architecture.py`` rebuilds the drawing.

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Service
     - What it does
     - What it writes
   * - ``control``
     - The HTTP API except the data routes; scenarios, commits and activation.
       It syncs each loaded scenario after a data write
     - serving requests, training jobs, sample requests, record copy
       requests, evaluations, commits
   * - ``inference``
     - Starts selected artifacts and answers requests with their published routes
     - ``served_models``, serving results, records
   * - ``samples``
     - Makes training samples from new records
     - samples
   * - ``training``
     - Runs training jobs on the training component, and harness evolution
       steps on the ``harness`` training component
     - job progress, trained artifacts, harness step records
   * - ``evaluation``
     - Runs the rollouts of artifact evaluations. It serves each candidate
       artifact itself, from the artifact repository
     - rollout results
   * - ``data``
     - Serves the `data routes <../reference/http-api.rst#data-items>`__ and
       the reads of what each scenario stores, copies new records to record
       items and new artifacts to artifact items, and runs data imports and
       exports
     - data items, record items, artifact items, data sync requests
   * - ``harness``
     - Serves the routes that a harness client calls: the active tree, the
       harness artifacts, the import of a person's files, and the adapters
     - harness import requests

Start each service with the same config and ``--component``:

.. code:: bash

   export REEF_TOKEN="$(openssl rand -hex 16)"
   reef serve -c deploy.yaml --component control --reef.port 8901
   reef serve -c deploy.yaml --component inference --reef.port 8902
   reef serve -c deploy.yaml --component samples --reef.port 8903
   reef serve -c deploy.yaml --component training --reef.port 8904
   reef serve -c deploy.yaml --component evaluation --reef.port 8905
   reef serve -c deploy.yaml --component data --reef.port 8906
   reef serve -c deploy.yaml --component harness --reef.port 8907

A ``--component`` process starts no other Reef service. Start any external
engines and training backends that the config requires. Inference owns the
serving runtime's startup work. All processes must reach the same store:

* On one host, the default ``sqlite`` record backend works. The processes share
  the files under ``storage.agent-record-dir``.
* On several hosts, use ``storage.record-backend: postgres``. The services keep
  their shared tables in the schema ``<record schema>_coordination``
  (``reef_records_coordination`` by default), so deployments with their own
  record schemas share no tables.

Give each process its own ``storage.artifact-work-dir`` and
``storage.artifact-cache-dir``, for example with ``--storage.artifact-work-dir``
on its command line. A process keeps its working copy of the artifact
repository there, and two processes in one working copy break each other's
fetches.

**Harness steps.** A harness evolution step is a training job of the
``harness`` training component, so the training service runs the method, the
coding agent and the evaluation runs. Install the adapter's binary and the
sandbox where the training service runs. The training service writes the step
records and the running step's progress under ``evolution.state_dir``. Control
reads them for the snapshot and request pages, so both must reach that
directory, as they reach the artifact backend.

.. _Harness clients:

**Harness clients.** Send ``GET /reef/harness``, ``GET /reef/harness/artifacts``,
``POST /reef/harness/import`` and ``GET /reef/harness/adapters`` to the harness
service. It reads the commit history and the artifact backend, and loads a
scenario without activating it. It writes no snapshot: an import goes to control
as a ``harness_sync`` request, control merges the files onto the newest active
tree and commits the sync, and the harness service answers with control's
answer. It waits up to 30 seconds for control, and answers ``503`` when control
did not commit the import in that time; the import stays queued, and the same
files sent again wait for the same request. The harness service needs no control
lease, so you can run several replicas of it behind one address. The snapshot and
request pages, the step records and ``/reef/status`` stay with control. Set
``harness.inference_url`` to the address at which clients reach the inference
service: a client's layer then sends its model calls there, not to the harness
service.

Every service except inference must also reach the same data store
(``storage.data-store``, which follows the record backend by default; see
`Choose the data store`_). Control, inference, training, evaluation and data must reach
the same artifact backend.

**Data requests.** Send the data routes (``/reef/namespaces/{owner}/data...``),
``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files`` and
``GET /reef/scenarios/{scenario}/jobs`` to the data service. In the split
layout, control does not serve these routes. The data service also answers the
reads of what each scenario stores: the scenario list, settings, contract,
records, commits, artifacts, samples, tasks, and evaluations with their
rollouts. Control answers these reads too. Writes to a scenario, the next-job
read, the cancel of a training job, ``/reef/status`` and the harness pages
stay with control; the harness client routes go to the harness service (see
`Harness clients`_ below). The data service needs no control lease,
so you can run several replicas of it behind one address.

After each data write, the data service asks control to sync the scenarios
that the write touched, and waits up to 30 seconds for the syncs. The response
says whether they ended (``synced``). A sync that did not end runs when control
takes it, or when the scenario next loads.

Control queues each selected artifact for inference to start. The inference
service opens the serving runtime, loads the artifact, and publishes its route.
Scenario creation and activation do not wait for this work.
The samples, evaluation and data services load a scenario without changing
its production engine.

**Inference requests.** Send model requests to the inference service, with the
scenario in ``x-reef-scenario``. The inference service answers
``/v1/chat/completions``, ``/v1/responses``, ``/v1/messages``,
``/v1/messages/count_tokens``, the evaluation routes and the routes of an
artifact that the scenario serves. A one-process ``reef serve`` answers
the same routes in the same way, in control's process.

Send ``GET /reef/scenarios/{scenario}/inference/status`` to inference to read
the selected artifact's startup result. It reports ``pending``, ``starting``,
``ready`` or ``failed`` with the artifact ID. See `Startup status
<../reference/http-api.rst#startup-status>`__ for the response and retry rules.

For each request, the inference service reads the scenario's rows in
``served_models``. The request's ``model`` selects the row of a served
artifact when it names one; any other ``model`` selects the active row, or the
row that a traffic split gives the request. The service sends the request to the engine at the row's URL,
under the row's model name. The only credential that it sends is the
deployment's provider key (``inference.upstream-api-key``), when one is set.
No artifact changes the request on its way to the engine. The service
appends the record to the scenario's record store and returns the record ID in
``x-reef-agent-record-id``, and the ID of the artifact that answered in
``x-reef-artifact-id``. ``x-reef-tag-*`` headers go into the record. The
inference service never creates a scenario: create it first with
``POST /reef/scenarios`` or a harness install. The inference service answers:

* ``404`` when the scenario is unknown or serves no model.
* ``503`` with ``inference_starting`` and ``Retry-After: 30`` when its model
  is pending or starting and no route can serve it. A failed startup returns
  ``503`` with ``inference_start_failed`` until a retry of the startup
  succeeds.
* the engine's own ``4xx`` and message when the engine refuses the request, for
  example a prompt longer than the model's context. An agent reads it to end or
  compact its run.
* ``502`` when the engine fails otherwise or cannot be reached, and ``504`` when
  it does not answer within ``inference.timeout-s``.
* ``409`` when the engine answers with another model name or aborts the
  request twice. The service reads the row again and retries once, for
  example during a change of the served model.
* ``503`` when it cannot read ``served_models`` and has no earlier read from
  the last 300 seconds, or when the record store does not take the record. The
  service never returns a buffered answer without its record. A stream goes
  out as the engine sends it, and its record is appended when it ends.

**Control restarts.** While control is down, the inference service continues to
answer from ``served_models`` and to record. Training jobs, sample requests and
evaluation rollouts that are already queued continue; new ones wait for control,
because control adds them. The data service continues to take data writes, but
their syncs wait for control: each write waits 30 seconds and then answers
``"synced": false``.

Control restarts do not reopen engines. One inference worker holds runtime
ownership through an existing work lease, including while it is idle. Other
inference replicas only serve requests. An interrupted startup returns to the
queue after its work lease expires.

A failed startup does not stay failed, because the selection still states what
the scenario serves. Control queues the startup again 60 seconds after it
failed. After each automatic retry that fails, control doubles the wait, up to
one hour. To retry at once, send the ``served`` mapping of the scenario again
(``POST /reef/scenarios/{scenario}/update``). This also sets the wait back to
60 seconds. The service that fails a work item writes one warning to its log,
with the queue, the item ID, the scenario and the error.

After ownership changes, the new owner opens its runtime and offers it the
routes from before. A route stays when the runtime reports that its engine
still serves it, as a remote endpoint can; requests to it go on without a gap.
The other routes go, and their scenarios' requests get ``inference_starting``.
Then completed selections return to starting, and the new owner activates them
again. An activation replaces a scenario's route when it serves from another
engine. This status is not a continuous engine health check. Runtime backends still control cancellation of engine calls
that were already in progress when a worker lost its lease.

Control runs as one active instance and any number of standby instances on the
same store. The active instance holds the control lease and extends it. A
standby instance waits at start-up until it can take the lease: it loads no
scenario and answers no request, ``/healthz`` included, so a load balancer that
checks ``/healthz`` sends requests to the active instance alone. When the active
instance stops, it releases the lease, and a standby takes over within 10
seconds. When the active instance crashes, a standby takes over after the lease
expires (30 seconds). An instance that loses the lease stops its training work
and ends its process, so that its supervisor starts it again as a standby. Each
serving request carries the control lease number. Route publication checks the
work lease, its attempt, and the current request before it commits. To deploy a new control version, start the new
instance as a standby, then stop the old one.

**Limits.** The inference service serves what ``served_models`` names: the
active artifact of each scenario and the artifacts served beside it. A
candidate under evaluation is not one of them: the evaluation service loads the
candidate from the artifact repository and answers its rollouts with the
scenario's runtime, so it needs no other service. Inference records name the
answering artifact with ``artifact_id``, including the starting artifact. A runtime
that loads full weights in place serves the next artifact under the same model
name after an activation, so a request in flight at that moment can be answered
by the next artifact while its record names the old one. Adapter runtimes serve
each artifact under its own name and do not have this limit.

Choose the storage backend
--------------------------

The storage backend keeps the records, commits, and settings of each scenario.
The settings are the model configuration, the selected training mode, the
automatic evaluation settings and the activation rule. You set them with
``POST /reef/scenarios/{scenario}/update`` and read them back with
``GET /reef/scenarios/{scenario}/settings``. After a restart, the scenario uses
its stored settings. If the recipe does not support the stored training mode,
the scenario uses the default mode and Reef logs a warning.

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
     coordination-backend: sqlite
     agent-record-dir: .reef/agent-record

Reef calls ``from_config(config, directory=...)`` with the ``config`` mapping and
``agent_record_dir``. A plugin holds no tables of the services, so name where
they share them: ``coordination-backend: sqlite`` for processes on one host, or
``postgres`` with ``record-database-url`` for several hosts. The `Python storage contract
<../reference/python-api.rst#scenario-stores>`__ gives the necessary behavior.
An embedded deployment gives a ``ScenarioStorage`` instance to
``Dispatcher(..., scenario_storage=...)`` instead.

With all backends, each scenario needs one Reef writer.

Choose the artifact backend
---------------------------

The artifact backend stores the snapshots of each scenario. The default,
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

A change from one backend to another does not move snapshots. Keep the old
store readable while scenarios refer to its snapshots, or use a new deployment.
With all backends, each scenario needs one Reef writer.

To keep the Git LFS repository on Hugging Face, give the built-in factory with
its options. Reef still reads every snapshot from its local clone, and pushes
each snapshot to the Hub as it publishes it:

.. code-block:: yaml

   storage:
     artifact-backend:
       module: reef.artifact.git_lfs:GitLFSRepositoryBackendFactory
       config:
         repository: https://huggingface.co/org/reef-artifacts
         refs: branches-and-tags
         token_env: HF_TOKEN

- ``refs: branches-and-tags`` keeps each snapshot on a tag
  ``reef/releases/<revision>``, and the refs that move on branches under
  ``reef/``, because the Hub accepts no other refs. The default, ``reef``,
  keeps every ref under ``refs/reef/``. A repository keeps the layout it
  started with, and Reef refuses the other.
- ``hf://org/reef-artifacts@<revision>`` names a stored snapshot by its Git
  commit revision, for a serving runtime or an import. Artifact IDs used by
  Reef's API are independent of these storage revisions.
- ``token_env`` names the variable of a token that can write the repository.
  Reef sends it to the repository's host only, as an HTTP header, so it
  appears in no URL, file or log. Keep the repository private unless you mean
  to publish what Reef trains.
- Archiving a deleted scenario's ref works with a local repository only.

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

Choose the data store
---------------------

The data store keeps the deployment's data items: the trajectories and the
tasks that clients create, and a record item for each inference call and
report, each with its metadata (`Data items
<../reference/http-api.rst#data-items>`__). Each item belongs to an owner, and
its metadata names the scenarios that use it. Archiving a scenario does not
delete its items. Record items stay when record retention removes the
records from a scenario's record store, so the data store grows until clients
delete items.

``storage.data-store`` selects the store. Without it, the store follows
``storage.record-backend``:

- ``sqlite`` keeps the items in ``<agent-record-dir>/data/data-store.sqlite3``.
  Put the directory on a persistent volume.
- ``postgres`` keeps them in ``storage.record-database-url``, in the schema
  ``reef_data``. Deployments that use one database share their items.
- A plugin record backend uses ``sqlite``.

To use another database or schema, or your own store, name a subclass of
``reef.storage.data.DataStore``:

.. code-block:: yaml

   storage:
     data-store:
       module: reef.storage.sql_data:PostgresDataStore
       config:
         database_url: ${REEF_DATA_DATABASE_URL}
         schema: reef_data

The task file store keeps the files of task items, apart from the files of each
scenario.

Keep large content in segment files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The data store's database grows with each trajectory and each record item.
With ``storage.segment-backend``, a SQL data store keeps that content in
segment files instead. A segment is one JSON Lines file per owner. The
database keeps the place of each line, and a read fetches only the lines that
it needs.

- A write of new trajectory items with at least 1 MiB of content, such as an
  upload, goes to one segment when the items are created.
- Every 5 minutes, the data service moves the content that the database still
  holds into segments: the messages of record items, the trajectory items of
  smaller writes, and trajectory items from before you set the backend. It
  moves an owner's content when at least 64 MiB of it waits, oldest first, in
  segments of at least 64 MiB.
- In the same run, the data service merges segments of less than 1 MiB into
  segments of at least 64 MiB, when an owner has at least 64 MiB in them. A
  read of many items then sends fewer requests to the segment store.
- A segment that no item uses for 24 hours is deleted.

``local`` keeps the segments in ``<agent-record-dir>/segments``. Only the
services of one host can share this directory. To share segments between
hosts, name a subclass of ``reef.storage.segments.SegmentStore``, for example
a store in object storage:

.. code-block:: yaml

   storage:
     segment-backend:
       module: my_package.segments:ObjectStoreSegmentStore
       config:
         bucket: reef-data

Reef calls ``from_config(config, directory=...)`` with
``<agent-record-dir>/segments``, which a remote store can use as a local cache.
The `Python segment store contract
<../reference/python-api.rst#segment-stores>`__ gives the necessary behavior.
Keep the backend set after you set it: a data store without it cannot read
content that it moved, and it raises an error for such an item.

When Reef opens a scenario for the first time after an upgrade, it moves the
scenario's uploaded trajectories and tasks into the data store. They become
items of the owner with the scenario's name, which the scenario uses in their
split. The data service makes record items from the scenario's records, in
pages of 256, while the scenario serves. It keeps how far it copied each
scenario in the deployment's shared store, so it continues from its last page
after a restart. A deployment that copied its records before this mark existed
copies them once more, which changes no item.

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
opens, Reef checks each directory with Harbor's task loader, and stores it as a
task item of the scenario's owner, which the scenario uses in its split (`Data
items <../reference/http-api.rst#data-items>`__). The task file store keeps a
copy of the files.

The import runs again each time a scenario opens. To add tasks, add them to the
manifest and the directory, and then restart the service or reload the
scenario. A task that moves to the other split in the manifest moves in the
scenario. To add tasks to one running scenario without a restart, create task
items with ``POST /reef/namespaces/{owner}/data/tasks``. A task that is already
imported with the same files does not change. A scenario does not open, and the
error says why, when:

- a task directory is missing, or Harbor cannot load it;
- an imported task now has other files;
- a task has the name of another sample of the scenario.

To change an imported task, give it a new name.

A harness evolution recipe evaluates on the imported eval tasks with
``evolution.evaluation.imported_tasks: true``. Each task runs in its own
container (see `Task directories
<evolve-your-harness.rst#task-directories>`__). Each step reads the eval split
again, so a task that you import later joins the next step. See `Harness
evolution keys <../reference/configuration.rst#harness-evolution-keys>`__.

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
and the snapshots. The next request to the scenario loads it again, as after a
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
  scenario, raise ``reef.core.errors.ScenarioLimitReachedError`` (HTTP 409).
- ``admit_load(scenario, loaded=...)``: return a ``LoadAdmission``. Its
  ``unload_count`` is the number of loaded scenarios that Reef must unload
  first. Its ``candidates`` are the scenarios that Reef tries, in that order.
  Reef unloads only idle candidates. If too few candidates are idle, Reef
  refuses the load with ``reef.core.errors.ScenarioCapacityFullError`` (HTTP 503).
  The plugin can also raise that error itself.

Reef calls the plugin before it creates a scenario and before it loads one.
Reef makes these calls one at a time, so the methods must return quickly.
``max-scenarios`` and ``max-loaded-scenarios`` configure the built-in plugin,
``ScenarioLimits``. If you set ``scenario-admission``, do not set them: Reef
refuses that configuration.

Upgrade from a Reef that wrote older formats
--------------------------------------------

This Reef reads only the formats that it writes: ``reef-commit/8`` commit
records, ``reef-scenario/7`` scenario metadata, and the current tables of the
record database, the data store and the coordination store. It does not
upgrade a store that an older Reef wrote.

A scenario with an older commit record or older scenario metadata does not
load, and its error names the format. A PostgreSQL record schema of another
version stops the start with ``unsupported PostgreSQL record schema version``.
Reef does not change or delete older data.

To move to this Reef from one that wrote older formats, use a new deployment
directory and, for PostgreSQL, a new schema. Then create the scenarios again.

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

Some recipes have their own directories, for example the harness evolution
recipe's ``evolution.state_dir``, which holds each scenario's history and
step records. These directories are not scenario storage. Keep
them on a persistent volume.
