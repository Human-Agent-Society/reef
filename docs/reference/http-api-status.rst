HTTP API: status, status codes and browser access
=================================================

Use ``/healthz`` to check a process and ``/reef/status`` to inspect its scenarios.

Health
------

``GET /healthz`` returns ``{"ok": true}`` without authentication. In external routing mode, inference also exposes
`GET /readyz <configuration-routing.rst#external-router-lifecycle>`__ to check request readiness.

Status
------

``GET /reef/status`` returns deployment state. This excerpt shows where to find a scenario's active and training selections.

.. code:: json

   {
     "loaded_scenarios": 1,
     "scenarios": {
       "hello-reef": {
         "commit_number": 3,
         "served_artifact_id": "A",
         "artifacts": {"weights": {"active": "A", "training_start": "B"}}
       }
     }
   }

``error`` and ``preload_errors`` report asynchronous training and preload failures. ``loaded_scenarios`` counts scenarios in this process.
``serving`` groups runtime status by recipe. Shared adapter residency appears under ``adapters``, nested under the weight component for composites.

Look up a scenario in ``scenarios``:

- ``commit_number`` counts all committed operations. ``last_training_commit`` gives the latest durable training commit's number, time and recipe metrics, or null before training.
- ``artifact_head_sync`` reports repository pointer synchronization as ``{state, error}``. See `commit ordering <../advanced_topics/state-model.rst#commit-ordering>`__.
- ``served_artifact_id`` identifies the published route for the weight trainer or sole component. It is null before inference publishes the route.
- ``artifacts`` gives each component's active artifact and training start. Both begin at creation. Unknown IDs are null.
- ``training_mode`` and ``job_data_ready`` give the training mode and whether data awaits training.
  Under ``processor``, ``ready_samples`` counts untrained samples, ``job_samples`` gives the required count and ``buffer_size`` limits the batch, extended to finish a group.
  Stages that accept requests also report ``pending_requests``.
- For weight training, ``training_jobs`` lists running jobs in submission order. Each gives ``job_id``, ``backend``, committed checkpoint count (``checkpoints_handled``), and ``start`` (checkpoint path or ``snapshot:`` followed by JSON with ``revision`` and ``root``).
  ``train_from`` names the requested starting artifact, or null for the trainer's default. ``finished_jobs`` includes successes and failures.
- Adapter runtimes also report ``adapter_runtime_load_id``, or null when serving the base model.

With several trainers, ``components`` reports readiness, mode, processor and last training commit per component.
Scenario fields describe the first trainer that supports jobs, or the first trainer if none does.
The scenario's last training commit spans all components.

Status codes
------------

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Status
     - Cause
   * - 400
     - Invalid body, missing scenario header, report schema error, or invalid artifact for serving or training.
   * - 401
     - Invalid or missing bearer token. Reef also accepts ``x-api-key`` when Authorization is absent. The two `harness pages <http-api-harness.rst#request-page>`__ also accept ``?key=``.
   * - 403
     - Routed request for an unconfigured scenario, target pin outside that scenario, rejected browser request, or upstream 403. Set per-scenario permissions at the gateway.
   * - 404
     - Unknown scenario, snapshot, artifact or adapter, or no configured model, harness recipe or served files.
   * - 409
     - Conflicting record content, selection change during a pending training commit, or an engine that aborted both attempts.
   * - Upstream 4xx
     - Reef preserves the message and statuses 400, 401, 403, 404, 408, 409, 422 and 429. Other upstream 4xx responses become 400.
   * - 502
     - Engine connection failure, upstream 5xx or invalid JSON object. The body gives the engine status and message, or its URL and connection error.
   * - 503
     - Unavailable store, failed record write, scenario capacity, or an unresolved artifact binding after retry. A full scenario cache or a route lost during retry includes ``Retry-After``.
   * - 504
     - Engine exceeded ``inference.timeout-s``.

Browser consoles
----------------

Add exact trusted origins to the deployment YAML, then restart Reef. Omit paths, trailing slashes, credentials and wildcards.

.. code:: yaml

   reef:
     console-origins: ["https://api.reefinfra.ai", "http://localhost:3000"]

Without this setting, Reef sends no CORS headers. Unlisted origins are rejected when the setting is active.
Listed origins can send GET, POST or DELETE with ``Authorization``, ``Content-Type`` and ``x-reef-scenario``.
Preflight requests need no token. Actual requests use the service's configured authentication. CORS does not enable cookies.

CORS applies to errors and streams and exposes ``x-reef-agent-record-id`` and ``x-reef-artifact-id``.
Browser HTTPS and local network restrictions still apply. Clients without an ``Origin`` header are unaffected.
