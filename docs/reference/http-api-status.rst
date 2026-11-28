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

``error`` and ``preload_errors`` report asynchronous failures.
Each scenario shows active/training artifacts, committed progress, training mode
and the state of the training pipeline (``pipeline``). ``served_artifact_id`` stays null until a route publishes.
``processor`` is the older name of ``pipeline`` and holds the same object. Read ``pipeline`` in new code.
For multiple trainers, inspect ``components``. Use the dedicated
`job <http-api-training.rst#training-jobs>`__ and
`startup <http-api-inference.rst#startup-status>`__ routes for details.

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
