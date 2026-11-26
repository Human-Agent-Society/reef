Run the services in separate processes
======================================

Run services separately to scale them independently or keep inference available
during control restarts. Start from a working single-process deployment.

The seven services coordinate through shared stores and work queues.
By default, ``reef serve`` runs them in one process.

Start the services
------------------

Use the same configuration for every service. Run each command in its own
process or supervisor:

.. code:: bash

   export REEF_TOKEN="$(openssl rand -hex 16)"
   reef serve -c deploy.yaml --service control --reef.port 8901
   reef serve -c deploy.yaml --service inference --reef.port 8902
   reef serve -c deploy.yaml --service samples --reef.port 8903
   reef serve -c deploy.yaml --service training --reef.port 8904
   reef serve -c deploy.yaml --service evaluation --reef.port 8905
   reef serve -c deploy.yaml --service data --reef.port 8906
   reef serve -c deploy.yaml --service harness --reef.port 8907

A ``--service`` process starts no other Reef service.
Start the external engines and training backends that the configuration requires.

Every process must reach the same stores:

- On one host, share SQLite files under ``storage.agent-record-dir``.
  On several hosts, use ``storage.record-backend: postgres``.
  Coordination uses schema ``<record schema>_coordination``.
- Share ``storage.data-store`` and the task and segment stores it uses.
  The data store follows the record backend by default; see
  `Data storage <storage.rst#choose-the-data-store>`__.
- Share ``storage.artifact-backend``. Training needs write access to save checkpoints.

Give each process separate ``storage.artifact-work-dir`` and
``storage.artifact-cache-dir`` paths. Shared working copies can corrupt concurrent fetches.

Send each request to its service
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Service
     - Routes
   * - ``inference``
     - ``/v1/chat/completions``, ``/v1/responses``, ``/v1/messages``, ``/v1/messages/count_tokens``, the evaluation routes, the routes of each served artifact, and ``GET /reef/scenarios/{scenario}/inference/status``.
   * - ``data``
     - The ``/reef/namespaces/{owner}/data...`` routes, artifact file reads and
       ``GET /reef/scenarios/{scenario}/jobs``. Only data serves these routes.
       Data and control both answer scenario reads: lists, settings, contracts,
       records, commits, artifacts, samples, tasks, evaluations and rollouts.
   * - ``harness``
     - ``GET /reef/harness``, ``GET /reef/harness/artifacts``, ``POST /reef/harness/import`` and ``GET /reef/harness/adapters``.
   * - ``control``
     - Other routes: scenario writes, next-job reads, training cancellation,
       ``/reef/status``, harness snapshot and request pages, and step records.
   * - ``samples``, ``training``, ``evaluation``
     - ``/healthz`` only. They take their work from queues.

Data and harness services need no control lease. You can run several replicas
of each behind one address. A one-process ``reef serve`` answers all routes.

Inference
---------

Inference loads selected artifacts asynchronously. Check
``GET /reef/scenarios/{scenario}/inference/status`` for ``pending``, ``starting``,
``ready`` or ``failed`` and the artifact ID.

Send inference to this service with ``x-reef-scenario``. The scenario must
already exist. Requests use the active selection or traffic split unless they
name a served artifact. Reef returns ``x-reef-agent-record-id`` and
``x-reef-artifact-id`` with successful recorded responses.

A scenario without a route returns 503 ``inference_starting`` with
``Retry-After: 30`` during startup, or ``inference_start_failed`` after failure.
An unknown scenario or one serving nothing returns 404.
See `Inference errors <../reference/http-api-inference.rst#request>`__ for
provider errors, timeouts and retry behavior.

Buffered responses are recorded before delivery. Streams are recorded when
they end; a stream that fails after headers has no receipt.

**Startup retries.** Control retries a failed startup after 60 seconds.
The delay doubles after each automatic failure, up to one hour.
Resend the scenario's ``served`` mapping to retry immediately and reset the delay.

**Runtime ownership.** One inference worker owns the serving runtime.
Other replicas handle requests. After ownership changes, remote runtimes can
retain routes only when they confirm that the engines still serve them.
Reef's native engines restart and reload artifacts, interrupting requests in flight.
Startup status is not a continuous health check.

**Request pinning.** Reef's engines keep each request on its selected weights.
Weight changes wait up to ``inference.timeout-s`` for requests to finish,
then interrupt remaining requests. They never continue on different weights.
See `Request pinning <../developer-guide/serving-runtimes.rst#request-pinning>`__.

- Use the same Reef version for control, inference replicas and components.
- Keep native engine addresses private; clients call Reef's inference API.
- Restart the inference component after resetting coordination storage.
- A failed adapter load leaves its engine stack unavailable. Correct the artifact
  or restart the serving worker before selecting a replacement.
- Control calls wait at most 30 seconds behind another call, then return retryable 503.
  A native call that never returns requires operator recovery.

Training
--------

Install backend dependencies where the training service runs.
Other services do not build training components:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Backend
     - What only the training service needs
   * - ``slime``
     - Slime, Megatron, Ray, PyTorch and Transformers
   * - ``tinker``, ``spindle_full``
     - The backend's API key and its ``state-dir``
   * - a harness recipe
     - The adapter's binary, the sandbox, and ``evolution.state_dir``
   * - a training backend plugin
     - The plugin's own dependencies

When training is unavailable, new jobs wait for its admission checks without
consuming samples. Status shows no running job until that check completes.

Training reads starts from the artifact backend and stores checkpoints there.
Control commits them by reference; it needs no shared training directory.
Uncommitted checkpoint content is removed when the ended job is deleted
(``storage.work-retention-s``, default 7 days).

Tinker and Spindle artifact imports also need the training service.
They return 503 if it does not answer within 60 seconds.

**Harness training.** The training service runs evolution methods, agents and
their evaluations. Only it reads ``evolution.state_dir``. Control reads job
progress from the shared store and ended step records from the artifact backend.

After scenario deletion, cleanup asks training to archive its history directory.
That request waits while training is unavailable.

Harness clients
---------------

Point harness clients at the harness service. Set ``harness.inference_url``
to the address those clients can reach for model calls.

The service reads trees and forwards file imports to control through the
``harness_sync`` queue. Control merges and commits the files.
After 30 seconds without an answer, import returns 503 but remains queued.
Resending the same files waits for the same request.

Data
----

Data writes request a scenario sync and wait up to 30 seconds.
The response's ``synced`` field reports whether it finished.
Pending syncs run when control handles them or the scenario next loads.
Data, samples and evaluation workers do not change the production engine merely
by loading a scenario.

Evaluation
----------

Evaluation needs a running inference service. It asks inference to load each
candidate, waits for readiness and sends rollout calls to that candidate.
These loads do not publish a production route. Their calls do not become
scenario training records.

Set ``evaluation.on-stop`` and ``evaluation.finish-timeout-s`` for graceful
shutdown. See `Evaluation settings
<../reference/configuration-settings.rst#evaluation>`__.
Keep the finish timeout below your supervisor's kill deadline.

A rollout returned during graceful shutdown uses no work attempt.
After a forced kill, its lease must expire before another worker takes it,
and that attempt counts toward ``storage.work-max-attempts``.

Control restarts
----------------

While control is down:

- Inference keeps serving loaded artifacts and recording requests.
  Control restarts do not reopen engines.
- Queued training jobs, sample requests and evaluation rollouts continue.
  Control must return before it can submit new work.
- Data accepts writes. Each write waits 30 seconds for control to sync it,
  then returns ``"synced": false``.

**Standby instances.** Run one active control instance and any number of
standby instances on the same store:

- The active instance holds the control lease and extends it.
- A standby waits for the lease. It loads no scenarios and answers no
  requests, including ``/healthz``. Use that probe to route traffic to the
  active instance.
- A normal stop releases the lease; a standby takes over within 10 seconds.
  After a crash, takeover waits for lease expiry (30 seconds).
- An instance that loses its lease stops its training work and exits.
  Its supervisor can restart it as a standby.

For a control upgrade, start the new instance as a standby. Then stop the old instance.
