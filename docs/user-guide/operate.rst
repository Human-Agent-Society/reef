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
- the artifact that answers the scenario's requests (``served_artifact_id``)
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

An artifact is the content of one component, such as a checkpoint's weights or a harness tree. Each training step that saves a checkpoint or keeps a tree makes one, and so does each import (`Import an artifact`_) and each harness sync. The list is newest first, one row for each artifact. ``active`` marks the component's active artifact, and ``activated`` marks each artifact that has been active at some time. ``served`` marks each artifact that the scenario serves beside its active release (`Serve an artifact beside the active release`_). `Artifacts <../reference/http-api.rst#artifacts>`__ in the HTTP API lists every field. In a scenario of one component, which every shipped recipe is, ``artifact_id`` equals the ``release_id`` of the release that made the artifact. The creation artifact is the artifact that the scenario started from, usually the base artifact (`The base artifact`_). Its ``artifact_id`` is the ``release_id`` of the release that made it.

A training step that made no artifact, such as a rejected or skipped step, has no row. Neither do activations, train-from operations and serves. ``GET /reef/scenarios/{scenario}/commits`` lists every step (`Record and commit history <../reference/http-api.rst#record-and-commit-history>`__).

For harness scenarios, ``GET /reef/harness/releases`` lists the release chain oldest first with each step's evaluation metrics. Each row names its scenario ``step``, 0 for the creation row. ``GET /reef/harness?release_id=<id>`` returns any listed tree. The catalog lists every training step, also a rejected or skipped step, which has ``checkpoint: false`` and carries the active release's ID. A new release that the activation rule did not activate sits in the catalog with its evaluation metrics, ``active: false`` and ``activated: false``. No session gets it until a person activates it. Clients label such a release ``Waiting for activation``.

The base artifact
-----------------

Every scenario starts from the deployment's base artifact. Reef imports it once, when the artifact repository is new, from the configuration:

- A weight recipe imports its training component's untrained checkpoint, with the layout of the component's other checkpoints:

  - The ``slime`` backend with full weights imports the model that ``inference.model-path`` names. The artifact repository keeps a full copy of the model's files, about 2 bytes per parameter in bf16.
  - The ``slime`` backend with LoRA imports an adapter whose weights are zero. It serves the model unchanged.
  - The ``tinker`` backend imports the checkpoint of its untrained adapter, which stays on the Tinker service.
- A harness recipe imports the files of its seed tree: ``evolution.seed``, else the recipe's own seed. Without a seed, the base harness is empty.
- A recipe of several components imports each component's base under its name.

The base artifact's ``artifact_id`` is the ID of the repository's base release. Activate it to serve the base model or the base harness again. Train from it to start training again from the base weights or the seed. The ``slime`` backend refuses this, as it refuses every artifact (`Train from an artifact`_).

Reef reads the model and the seed only to make the base artifact. When a restart configures another model path or another seed, Reef refuses to start and names both values. To change the base, restore the configuration, or start a new deployment with a new artifact repository. Reef also refuses an artifact repository that an older Reef made, because its base was not imported. Start such a deployment with a new artifact repository.

Choose a starting release
-------------------------

The default inference routes answer from the scenario's active release. No request header keeps a client on an older release. To keep an older artifact answering, serve it beside the active release (`Serve an artifact beside the active release`_).

A new scenario starts from the base artifact. To start it from another release, create it with ``POST /reef/scenarios`` and ``{"name": "<scenario>", "release_id": "<id>"}``. A report or a record import that creates the scenario can carry ``x-reef-release-id: <id>`` instead. On a scenario that already exists, the header must name the bound starting release. Otherwise, Reef refuses the request with HTTP 409. `Headers <../reference/http-api.rst#headers>`__ in the HTTP API lists this header. An inference request never creates a scenario.

To pull a specific harness release, add ``?release_id=<id>`` to the harness manifest route. To serve an earlier artifact again, activate it as `Activate an artifact`_ describes.

Import an artifact
------------------

To serve or train an artifact that was made outside Reef, such as an adapter that you fine-tuned elsewhere, a checkpoint on S3, or a harness that you wrote, import it by a URI that Reef reads:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"source": "s3://my-bucket/checkpoints/step-400/"}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

Reef reads ``hf://org/model@revision``, ``s3://bucket/prefix/`` and ``file:///absolute/path``, a directory on the service's machine. To import a directory from your own machine, upload it to one of these places first. S3 needs the ``s3`` extra (``uv pip install 'reef-infra[s3]'``) and reads with boto3's standard credentials: the environment, a profile or an instance role. Reef stores no credential. A reader can read whatever the service can read, so only give import access to callers that may train the scenario. ``storage.artifact-sources`` adds readers for other schemes (`Python API <../reference/python-api.rst#artifact-imports>`__).

The directory must have the layout of the scenario's releases: the training component's weight files, a Tinker manifest whose saved state is on the configured service, or a harness source tree whose files are UTF-8 text and pass validation for the adapter. Otherwise Reef answers HTTP 400 and changes nothing. `Import an artifact <../reference/http-api.rst#import-an-artifact>`__ in the HTTP API lists each layout and status.

The import adds an ``import`` row to the artifact list, with the URI and the version that Reef read in ``source``. It does not serve the artifact and does not change a training start, and the activation rule does not run. Activate it, or train from it, as any other artifact. Reef copies the files into the artifact store, so a later change at the source does not change the artifact. An import needs the model's size in disk space twice while it runs: once under ``storage.artifact-cache-dir`` and once in the artifact store.

To import model weights from Hugging Face without a copy, name a full commit SHA: ``hf://org/model@<40-digit commit>``. Reef then keeps a reference to that revision, and downloads its weights only when a serving engine loads them. A commit names fixed files, so the artifact cannot change, but it can be lost: keep the repo on the Hub while you serve or train the artifact. A branch or a tag can move, so Reef copies those.

A harness sync is different. ``POST /reef/harness/import`` merges the files that a person's session changed onto the active release. It commits the result as a ``sync`` row. In that one commit the new release becomes the active release and the training start of its component (`Evolve your harness <evolve-your-harness.rst>`__).

Activate an artifact
--------------------

The active release is the one that Reef serves. Inference answers from it, and the harness routes serve its tree. It binds one active artifact for each component. To serve another artifact, activate it:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/activate"

The target is any row of the artifact list: an artifact that a training step or an import made, or the creation artifact. Reef loads the artifact into serving and makes it the active artifact of its component. What Reef writes depends on the scenario:

- In a scenario of one component, Reef writes no artifact or release. It appends an ``activate`` record. The release that made the artifact serves again.
- In a scenario of several components, Reef publishes a release: the active release with ``component`` bound to the artifact. This release makes no new content. The other components stay as they serve.

The answer's ``artifact_id`` names the activated artifact, and its ``release_id`` names the active release. Leave out ``component`` when the scenario has one component. A scenario of several components requires it. An unknown component is HTTP 400. History is not rewritten, and the step counter keeps increasing. Activation never changes a trainer's training start.

A full-weight serving runtime that activates the base artifact after it served a checkpoint loads the base artifact's files from the artifact repository. `HTTP API <../reference/http-api.rst#activate-an-artifact>`__ lists the answers and status codes.

The deployment's `activation section <../reference/configuration.rst#the-activation-section>`__ names the rule that decides whether Reef activates each new artifact:

- ``always``, the default, activates each new artifact when the training step makes it. Its ``review_kinds`` holds back a harness tree that changes one of the listed tree kinds.
- ``never`` makes each new artifact but does not activate it. A person activates it.
- ``evaluated`` activates an artifact once an evaluation of it reaches ``min_mean_score``.

A scenario can select one of these rules for itself with ``POST /reef/scenarios/{scenario}/update`` and ``{"activation": {"module": "never"}}``. Send ``"activation": null`` to follow the deployment's rule again. ``GET /reef/scenarios/{scenario}/settings`` reads back the scenario's rule and the rule that applies (`Scenario updates <../reference/http-api.rst#scenario-updates>`__).

An artifact that the rule does not activate still becomes the start of the next training job.

Train from an artifact
----------------------

Each trainer continues from its own newest output: a weight trainer from its newest saved checkpoint, a harness trainer from the tree it kept. That artifact is the trainer's training start. To make a trainer continue from another artifact, set its training start:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

Reef cancels the running job. The next training job, or the next evolve step, starts from the artifact. Train-from never changes the active release. Leave out ``"component"`` when the scenario has one component. A scenario of several components requires ``"component": "<name>"`` in the body, and an unknown component is HTTP 400. A weight artifact without optimizer state, such as the base artifact, starts training with a new optimizer state. Each train-from also starts a new run in experiment tracking.

Whether a trainer can start from an artifact depends on the backend:

- The ``tinker`` backend imports the artifact's checkpoint manifest, so a Tinker scenario can train from any of its artifacts. The import includes the saved training state with the optimizer state.
- The ``slime`` backend refuses every artifact with HTTP 400, because its workers continue from their own latest checkpoint.
- A harness trainer starts its next evolve step from the artifact's tree.

Reef cannot move the training of a Slime scenario back to an earlier checkpoint, although it can activate one of its artifacts. To train again from an earlier checkpoint, start a new deployment for a new scenario:

- Give the deployment its own ``save`` and ``save-hf`` directories. The Slime component does not overwrite a rollout that it saved before.
- Point Slime at the old checkpoint with its own options in ``training.options``: ``load`` names the old ``save`` directory, and ``ckpt-step`` names the saved rollout.

The new scenario's first job continues from the weights that the workers loaded. An existing scenario cannot move to the new deployment: its committed start names a checkpoint of the old one, so the Slime component refuses its jobs.

Serve an artifact beside the active release
-------------------------------------------

A scenario can serve other artifacts beside its active release, each at its own URL. Use it to:

- keep the current artifact answering while you move traffic to a new one;
- try a new artifact before it is active.


.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/serve"

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model": "reef", "messages": [{"role": "user", "content": "hi"}]}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts/<id>/v1/chat/completions"

The default routes keep answering from the active release. A call to the artifact's URL is recorded like any other call, and its record names the release that answered. An activation does not change what is served, so serve the active artifact before you activate another one to keep it answering. ``POST /reef/scenarios/{scenario}/stop-serving`` with the same body ends the serve. Both survive a restart.

Only the active release trains. Which client calls which URL is your choice, for example a gateway that moves traffic from one URL to the other. The runtime must answer each call from the release that the call names. A Tinker scenario can serve any of its artifacts. A runtime that loads weights into engines refuses a serve. `Serve an artifact beside the active release <../reference/http-api.rst#serve-an-artifact-beside-the-active-release>`__ in the HTTP API lists the answers and status codes.

Go back to an older artifact
----------------------------

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
   * - a harness step in flight
     - resumed, as any training job: a step's outcome is durable before its
       checkpoint, so a step that settled commits after the restart without
       running again, and a step that did not settle runs again from the
       same start and harness state

A process that stops normally releases the control lease. After a crash, the
next process waits until the old lease expires (at most 30 seconds) before it
loads scenarios; see `Run the services in separate processes`_.

After a step commits, ``/reef/status`` reports its scenario's
``artifact_head_sync`` with the ``release_id`` of the `chain tip
<../advanced_topics/state-model.rst#the-release-chain>`__ and any ``error``.
The ``state`` is ``synchronized`` when the backend head names the chain tip,
``pending`` when updating it failed, or ``conflict`` when another writer moved it to an
unrelated release. The committed step remains successful. Before another commit,
Reef retries synchronization; if it still fails, the new commit stops before
publishing or writing its commit record. A conflict never overwrites the other
writer's head. Restart also synchronizes the head before loading the scenario.

The scenario storage and the repository live under ``.reef/`` by default (``agent_record_dir``, ``task_file_dir``, ``artifact_repository``, ``artifact_work_dir``, ``artifact_cache_dir``). On ephemeral storage none of the guarantees above hold past its loss. Each scenario needs one control service; run a second deployment on other ports and storage paths rather than two deployments on one store. The services of one deployment share its store; see `Run the services in separate processes`_. `State model <../advanced_topics/state-model.rst#commit-ordering>`__ describes the commit ordering behind the table. ``DELETE /reef/scenarios/{scenario}`` retires a scenario: the storage backend archives its records, commits, settings and task files, its history and step records move under an ``archived/`` sibling and its repository ref is renamed into ``refs/reef/archived/``, so a name can be reused without the old chain. Deleting a weight-training scenario cancels its running training job.

Run the services in separate processes
--------------------------------------

A deployment has six services. By default, ``reef serve`` runs all of them in
one process. Run them in separate processes when inference must stay up while
control restarts or deploys a new version, or when one service needs more
replicas than the others. The services do not send work to each other: each one
reads and writes the deployment's shared store. The one call between services
goes from evaluation to control, for the candidate that a rollout evaluates.

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Service
     - What it does
     - What it writes
   * - ``control``
     - The HTTP API except the data routes; scenarios, commits and activation.
       It syncs each loaded scenario after a data write
     - ``served_models``, training jobs, sample requests, record copy
       requests, evaluations, commits
   * - ``inference``
     - Answers model requests with the model that ``served_models`` names
     - records
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
       harness releases, the import of a person's files, and the adapters
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

A ``--component`` process starts no other child. Start the inference engines
and the training backend that the config names on their own. All processes
must reach the same store:

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
reads them for the release and request pages, so both must reach that
directory, as they reach the artifact backend.

.. _Harness clients:

**Harness clients.** Send ``GET /reef/harness``, ``GET /reef/harness/releases``,
``POST /reef/harness/import`` and ``GET /reef/harness/adapters`` to the harness
service. It reads the commit history and the artifact backend, and loads a
scenario without activating it. It writes no release: an import goes to control
as a ``harness_sync`` request, control merges the files onto the newest active
tree and commits the sync, and the harness service answers with control's
answer. It waits up to 30 seconds for control, and answers ``503`` when control
did not commit the import in that time; the import stays queued, and the same
files sent again wait for the same request. The harness service needs no control
lease, so you can run several replicas of it behind one address. The release and
request pages, the step records and ``/reef/status`` stay with control. Set
``harness.inference_url`` to the address at which clients reach the inference
service: a client's layer then sends its model calls there, not to the harness
service.

Every service except inference must also reach the same data store
(``storage.data-store``, which follows the record backend by default; see
`Choose the data store`_). Control, training, evaluation and data must reach
the same artifact backend.

**Data requests.** Send the data routes (``/reef/namespaces/{owner}/data...``),
``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files`` and
``GET /reef/scenarios/{scenario}/jobs`` to the data service. In the split
layout, control does not serve these routes. The data service also answers the
reads of what each scenario stores: the scenario list, settings, contract,
records, commits, artifacts, samples, tasks, and evaluations with their
rollouts. Control answers these reads too. Writes to a scenario, the buffer
routes, the cancel of a training job, ``/reef/status`` and the harness pages
stay with control; the harness client routes go to the harness service (see
`Harness clients`_ below). The data service needs no control lease,
so you can run several replicas of it behind one address.

After each data write, the data service asks control to sync the scenarios
that the write touched, and waits up to 30 seconds for the syncs. The response
says whether they ended (``synced``). A sync that did not end runs when control
takes it, or when the scenario next loads.

Only control activates a scenario's artifact on the engines. The samples,
evaluation and data services load a scenario without activating it.

**Inference requests.** Send model requests to the inference service, with the
scenario in ``x-reef-scenario``. The inference service answers
``/v1/chat/completions``, ``/v1/responses``, ``/v1/messages``,
``/v1/messages/count_tokens``, the evaluation routes and the routes of an
artifact served beside the active release. A one-process ``reef serve`` answers
the same routes in the same way, in control's process.

For each request, the inference service reads the scenario's rows in
``served_models``. The request's ``model`` selects the row of an artifact
served beside the active release when it names one; any other ``model`` selects
the active row. The service sends the request to the engine at the row's URL,
under the row's model name. The only credential that it sends is the
deployment's provider key (``inference.upstream-api-key``), when one is set.
No artifact changes the request on its way to the engine. The service
appends the record to the scenario's record store and returns the record ID in
``x-reef-agent-record-id``, and the ID of the artifact that answered in
``x-reef-release-id``. ``x-reef-tag-*`` headers go into the record. The
inference service never creates a scenario: create it first with
``POST /reef/scenarios`` or a harness install. The inference service answers:

* ``404`` when the scenario is unknown or serves no model.
* ``502`` when the engine fails or cannot be reached, and ``504`` when it does
  not answer within ``inference.timeout-s``.
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

When control starts, it does not load again what the engines still serve. A
runtime that serves one adapter per scenario takes over each adapter that
``served_models`` names and that the engines still report, and loads only what
they lost. A runtime that loads full weights in place loads the active artifact
again. When a replaced value's stop is due, the engines unload its adapter.

Control runs as one active instance and any number of standby instances on the
same store. The active instance holds the control lease and extends it. A
standby instance waits at start-up until it can take the lease: it loads no
scenario and answers no request, ``/healthz`` included, so a load balancer that
checks ``/healthz`` sends requests to the active instance alone. When the active
instance stops, it releases the lease, and a standby takes over within 10
seconds. When the active instance crashes, a standby takes over after the lease
expires (30 seconds). An instance that loses the lease stops its training work
and ends its process, so that its supervisor starts it again as a standby. Each
``served_models`` write carries the lease number, so an old instance cannot
overwrite the table. To deploy a new control version, start the new
instance as a standby, then stop the old one.

**Limits.** The inference service serves what ``served_models`` names: the
active artifact of each scenario and the artifacts served beside it. A
candidate under evaluation is not one of them: the evaluation service loads the
candidate from the artifact repository and answers its rollouts with the
scenario's runtime, so it needs no other service. A record of the
artifact that a scenario started from carries no ``artifact_ref``. A runtime
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

To keep the Git LFS repository on Hugging Face, give the built-in factory with
its options. Reef still reads every release from its local clone, and pushes
each release to the Hub as it publishes it:

.. code-block:: yaml

   storage:
     artifact-backend:
       module: reef.artifact.git_lfs:GitLFSRepositoryBackendFactory
       config:
         repository: https://huggingface.co/org/reef-artifacts
         refs: branches-and-tags
         token_env: HF_TOKEN

- ``refs: branches-and-tags`` keeps each release on a tag
  ``reef/releases/<release_id>``, and the refs that move on branches under
  ``reef/``, because the Hub accepts no other refs. The default, ``reef``,
  keeps every ref under ``refs/reef/``. A repository keeps the layout it
  started with, and Reef refuses the other.
- A release's ID is its commit, so ``hf://org/reef-artifacts@<release_id>``
  names the release on the Hub, for a serving runtime or an import.
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

Some recipes have their own directories, for example the harness evolution
recipe's ``evolution.state_dir``, which holds each scenario's history and
step records. These directories are not scenario storage. Keep
them on a persistent volume.
