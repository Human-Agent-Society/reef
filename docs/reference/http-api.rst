HTTP API: inference, feedback, and artifacts
============================================

Reef serves the provider's own inference routes: OpenAI Chat Completions at
``/v1/chat/completions``, OpenAI Responses at ``/v1/responses``, and Anthropic
at ``/v1/messages``. It forwards each request to the runtime unchanged. It
adds ``/reef/*`` routes for feedback, data, scenarios, artifacts, and status.

The service accepts request bodies up to 64 MiB, so coding agents can send
large contexts. The record import routes accept bodies up to 10 MiB.

For a complete request, receipt, and feedback example, start with the
`inference and feedback quickstart <../getting-started/quickstart.rst>`__.
To put the snapshot routes into practice, follow the `agent harness tutorial
<../user-guide/evolve-your-harness.rst>`__ or the `model weight training guide
<../user-guide/evolve-your-model.rst>`__.

.. code:: bash

   export REEF_TOKEN=reef-local
   curl -f http://127.0.0.1:8900/healthz     # {"ok": true}

Routes
------

+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| Route                                                                                                                 | Response                                          |
+=======================================================================================================================+===================================================+
| ``GET /healthz``                                                                                                      | readiness; the only unauthenticated route         |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /v1/chat/completions``                                                                                         | OpenAI-format inference                           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /v1/responses``                                                                                                | OpenAI Responses-format inference                 |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /v1/messages``                                                                                                 | Anthropic-format inference                        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /v1/messages/count_tokens``                                                                                    | count request tokens; recorded like any inference |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/v1/{route}``                                                                        | inference with the scenario in the path           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluation/v1/{route}``                                                             | inference by the active model; keeps no record    |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/components/{component}/evaluation/v1/{route}``                                      | the same, for a candidate of one component        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route}``                                                | inference by one served artifact, recorded        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/records``                                                                                                | import one existing inference or report           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/records/batch``                                                                                          | atomically import a batch of records              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data``                                                                                | create record and artifact items                  |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/trajectories``                                                                   | create trajectory items from JSON Lines           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/tasks``                                                                          | create task items from a tar.gz, or by reference  |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``PUT /reef/namespaces/{owner}/data/tasks/{id}``                                                                      | replace a task item's files, or by reference      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data``                                                                                 | the owner's items, in arrival order               |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/groups/{key}``                                                                    | the owner's items grouped by a metadata value     |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/sets/{kind}``                                                                     | the owner's benchmarks or trajectory sets         |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/{id}``                                                                            | one item, with its content                        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/metadata``                                                                       | patch the metadata of items                       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/data/{id}``                                                                         | delete one item; restore it for 30 days           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/deleted-items``                                                                        | the owner's deleted items, newest first           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/deleted-items/restore``                                                               | restore deleted items                             |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/deleted-items/{id}``                                                                | purge one deleted item for good                   |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/{id}/archive``                                                                    | download one item: a task as tar.gz, else JSON    |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/imports``                                                                        | import the files at a URI as items                |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/imports/{id}``                                                                    | the status of one import                          |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/exports``                                                                        | export items to a tar.gz at a URI                 |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/exports/{id}``                                                                    | the status of one export                          |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/collections``                                                                          | the owner's collections of benchmarks             |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``PUT /reef/namespaces/{owner}/collections/{name}``                                                                   | create or replace one collection                  |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/collections/{name}``                                                                   | one collection, with its task counts              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/collections/{name}``                                                                | delete one collection                             |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/collections/{name}/scenarios``                                                        | make a scenario hold a collection's benchmarks    |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks``                                                      | the benchmarks that a scenario holds              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{name}``                                               | make a scenario hold a benchmark in a split       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{name}``                                            | end a scenario's hold of a benchmark              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets``                                                 | the trajectory sets that a scenario holds         |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets/{name}``                                          | make a scenario hold a trajectory set             |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/scenarios/{scenario}/trajectory-sets/{name}``                                       | end a scenario's hold of a trajectory set         |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/report``                                                                                                 | submit feedback about one or more trajectories    |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/train``                                                                                                  | enqueue one training request                      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios``                                                                                               | scenario selections and harness adapter           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios``                                                                                              | create a scenario explicitly                      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/inference/status``                                                                   | startup state of a served artifact                |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/update``                                                                            | update a scenario's settings and what it serves   |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/settings``                                                                           | read a scenario's settings                        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract``                                                                           | what this scenario accepts                        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/records``                                                                            | retained record metadata                          |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/commits``                                                                            | compatibility route for operation history         |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/operations``                                                                         | paginated operation history and selections        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/records/{record_id}``                                                                | one retained record and its trace payload         |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/artifacts``                                                                          | ``{scenario, artifacts}``, newest first           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files``                                                      | an artifact's file list, or one file              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/artifacts``                                                                         | import a directory as a new artifact              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/restore``                                                                           | restore the selection of an earlier commit        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/train-from``                                                                        | set the artifact that a trainer starts from       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/redeploy``                                                                          | deploy the served engines again                   |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/tasks``                                                                              | the tasks the scenario uses, which trained        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/jobs``                                                                               | the scenario's training jobs in the jobs queue    |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/jobs/{job_id}/cancel``                                                              | cancel one training job of the scenario           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/requests/{request_id}/cancel``                                                      | cancel one training request of the scenario       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/samples``                                                                            | the scenario's training samples, in use order     |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/sample-processing``                                                                  | the scenario's sample processing tasks and counts |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/sample-processing/{task_id}``                                                        | one sample processing task of the scenario        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/sample-processing/{task_id}/logs``                                                   | the log of one attempt of a task                  |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/next-job``                                                                           | the samples that the next automatic job trains    |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluations``                                                                       | evaluate one artifact of a snapshot               |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/cancel``                                                | cancel one evaluation                             |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/retry``                                                 | run the failed rollouts again                     |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/resume``                                                | run the cancelled and the unstarted rollouts      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations``                                                                        | artifact evaluations, newest first                |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}``                                                        | one evaluation: its state and result              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts``                                               | the rollouts of one evaluation, as they run       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls``                   | the model calls of one rollout, without bodies    |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls/{attempt}/{index}`` | one model call of a rollout, with its bodies      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/trajectories``            | the ATIF trajectories of each attempt of a rollout|
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories``                                           | the trajectories of an evaluation, as NDJSON      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/scenarios/{scenario}``                                                                                 | delete a scenario; the sweep archives its state   |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness``                                                                                                 | the harness tree of the active selection          |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/harness/import``                                                                                         | merge and commit a person's harness files         |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/artifacts``                                                                                       | harness artifact catalog, oldest first            |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/artifacts/{step}/page``                                                                           | the HTML page of one catalog step                 |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/requests/{record_id}/page``                                                                       | the HTML page of one harness request              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/requests/{record_id}/progress``                                                                   | the same state as JSON, for a polling client      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/artifacts/{step}/records``                                                                        | retained raw step file inventory or file body     |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/adapters``                                                                                        | every harness adapter this process resolves       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/status``                                                                                                  | training, serving, and storage state              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+

In the split layout, the harness service answers ``GET /reef/harness``,
``POST /reef/harness/import``, ``GET /reef/harness/artifacts`` and
``GET /reef/harness/adapters``; control answers the other harness routes (`Run
the services in separate processes
<../user-guide/split-services.rst>`__).

Headers
-------

+-----------------------------------+---------------------------------------------------------+
| Header                            | Required for                                            |
+===================================+=========================================================+
| ``x-reef-scenario``               | inference, report, harness manifest, artifacts and      |
|                                   | import. Names the workload a record belongs to.         |
+-----------------------------------+---------------------------------------------------------+
| ``Authorization: Bearer <token>`` | every route except ``GET /healthz``, when auth is       |
|                                   | configured; with no Authorization header, ``x-api-key`` |
|                                   | carries the token. The evaluation routes also accept    |
|                                   | the token the service issues the recipe at startup.     |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-artifact-id``            | response only: the artifact that answered inference or  |
|                                   | supplied the harness tree. To pull a specific harness   |
|                                   | artifact, use ``?artifact_id=``.                        |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-tag-<name>``             | optional on inference: context stored under             |
|                                   | ``metadata.tags``; see `Client artifact tags`_ below.   |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-rollout: 1``             | optional on inference: a call of a rollout that Reef    |
|                                   | runs itself. It is served like any other call, but no   |
|                                   | record is kept. The response has no ``training`` block; |
|                                   | ``x-reef-artifact-id`` names the answering artifact.    |
|                                   | The call must not stream (HTTP 400). Only the service   |
|                                   | token opens it.                                         |
+-----------------------------------+---------------------------------------------------------+

Client artifact tags
~~~~~~~~~~~~~~~~~~~~

``x-reef-tag-artifact_id`` names the harness artifact that the native client
installed. Reef stores it under ``metadata.tags`` like any other tag, so a
processor can read which harness artifact made the call. The tag does not
change which artifact serves the call, or the record's ``artifact_id``, which
names the artifact that answered. It is separate from ``x-reef-artifact-id``,
the response header that names the model artifact.

Routed external-model requests
------------------------------

When `Scenario routing <configuration.rst#scenario-routing-between-students-and-teachers>`__
is enabled, clients send ``POST /v1/chat/completions`` directly to Reef with a
service token. Only the base record-only Recipe is supported. A routed Reef
serves only this route, ``POST /reef/report`` and ``GET /healthz``; other
routes return 404. With the `external router lifecycle
<configuration.rst#external-router-lifecycle>`__, the ``inference`` service
applies this section to the manifest's Scenarios only, and also serves
``GET /readyz``. Send ``POST /reef/report`` to the ``control`` service there.
Calls with ``x-reef-rollout: 1`` or ``true`` return 400
(``unsupported_rollout``). Rollout calls remain available when routing is disabled.

.. list-table:: Client fields
   :header-rows: 1

   * - Field
     - Required value
   * - ``x-reef-scenario``
     - A Scenario listed in the manifest.
   * - JSON ``model``
     - The manifest's automatic model name, or one of the Scenario's target IDs.

For the automatic model, Reef sends one ``POST /api/v1/routing/preview`` to
the Scenario's private Semantic Router entrypoint, with the request's
messages and the router runtime hash Reef verified at startup. Reef accepts
the reply only if it names that hash, the Scenario's entrypoint, and either
``default`` or one of the Scenario's rules with its configured target. A pin is
resolved locally and never calls the router. Reef then sets the provider
``model``; the rest of the JSON body is the client's original body, not the
router's re-encoded copy. The target is frozen for every attempt of the
request. These checks precede Scenario creation and provider execution:

* An unconfigured Scenario returns 403 (``unknown_scenario``), and so does a
  pin to a target outside the Scenario (``target_not_allowed``).
* A missing Scenario or model, an unknown target, a private router entrypoint
  as ``model``, a repeated ``x-reef-*`` header, a client-supplied
  ``x-reef-routing-*``, ``x-reef-sr-decision`` or ``x-selected-model`` header,
  and a nonempty ``x-reef-artifact-id`` return 400.
* A disabled target returns 503.
* A managed deployment returns 408 when the request body does not finish
  uploading within 30 seconds.
* For the automatic model only: messages the router cannot classify return
  400, a preview body over the router's 10 MiB limit returns 413 without
  calling the router, and a full router returns 429. A router connection that
  is refused, reset or closed before any reply arrives, as when SR stops or
  restarts, returns 503 (``routing_unavailable``), and so do an unverified
  runtime (``routing_unpublished``) and a runtime that changed
  (``routing_runtime_changed``). A router timeout returns 504. A reply that
  starts but is malformed or cut off, or a selection that does not match the
  manifest and runtime, returns 502 (``routing_protocol`` or
  ``routing_selection``).

None of these failures is retried or sent to another target. See the
`Semantic Router tutorial <../../tutorials/semantic-router/README.md>`__.

Each dispatched attempt produces an INFERENCE record in the request's
Scenario. Server-generated ``payload.metadata.routing`` replaces
caller-supplied routing metadata and contains ``target_id``, ``role``,
``provider_model``, ``model_revision``, ``snapshot_version``,
``router_config_hash`` (``null`` for a pin),
``original_selector``, the selection ``reason`` (``default``, ``rule`` or
``explicit_pin``) and ``rule_name`` (``null`` for pins). ``request_id``,
``attempt_index`` and ``prior_attempt_ids`` link retries; the successful
attempt's normal receipt remains the reference for Scenario-local REPORT
feedback.

``model_revision`` is declared by the deployment manifest, with
``identity_source: deployment_manifest``. The external model serves the request.
Its record has no Reef ``artifact_id``. The declared revision identifies the
configured target; it does not prove which weights the provider loaded. Reef
sends the target no artifact identity. It preserves the provider's returned
log probabilities in the response and the record.

``execution_outcome`` is ``completed``, ``aborted``, ``error``, ``timeout`` or
``cancelled``; interrupted streams whose execution cannot be established use
``unknown``. Failed attempts are diagnostic records without training data.
``delivery_outcome`` is ``not_started`` for failures
before delivery, or ``incomplete`` for interrupted streams. Successful
generation uses ``unconfirmed``: Reef persists the record before returning a
buffered response or writing the SSE receipt and terminal event, so it cannot
confirm client receipt. ``client_visible`` likewise describes the serving
attempt, not a delivery acknowledgement.

Persistence failure produces no success receipt. An SSE response that has
already started ends without a successful terminal event. This recording
mode does not train models or supply the sampled tensors required for OPD.

Managed routing lifecycle
-------------------------

With ``reef.routing_manifest`` set, the deployment controller commands the
routed Reef child through ``GET /reef/lifecycle``, ``POST /reef/lifecycle``
and ``POST /reef/lifecycle/check``. This interface is private to Reef:

- The child serves it on a private Unix socket (directory 0700, socket 0600),
  never on ``reef.port``.
- It requires a separate per-generation control credential. Service and
  evaluation tokens cannot open it, and the control credential opens no
  public route.
- Operators use ``reef serve`` and the deployment status and cleanup commands
  instead (`Managed routing lifecycle
  <configuration.rst#managed-routing-lifecycle>`__).

A client of the public routes sees these effects:

- Managed inference starts closed. A closed deployment rejects new inference
  with 503 without invoking a provider. ``GET /healthz`` remains a liveness
  check; it does not mean that the deployment accepts inference.
- Reef fixes admission and the request's routing policy together before it
  reads the body. An admitted request keeps the manifest and router runtime
  hash that it captured, also while the router previews it and when a policy
  is published meanwhile.
- An accepted request remains active through local response EOF or connection
  termination, including SSE terminal frames. Local EOF does not establish
  client receipt.
- Reports can finish during a ``close``. A ``stop`` rejects new mutations and
  is terminal for the generation.

The controller's side, in brief:

- ``GET /reef/lifecycle`` returns ``deployment_id``, ``generation``,
  ``snapshot_version``, ``router_config_hash`` (empty until the owner
  publishes a verified runtime), ``control_revision``, ``accepting``,
  ``stopping``, ``policy_closed``, ``active_requests``,
  ``pending_selections``, ``pending_writes``, ``drain_pending_writes`` and the
  diagnostic counts ``forced_requests`` and ``failed_writes``.
- ``POST /reef/lifecycle`` takes the deployment ID, the generation, a
  nonnegative integer ``control_revision`` and an ``action``: ``open``,
  ``close``, ``close_policy``, ``abort_policy``, ``activate`` or ``stop``.
  Revisions increase monotonically. The same revision and command again is
  idempotent. Stale revisions, conflicting commands and identity mismatches
  return 409.
- ``open`` and ``abort_policy`` require ``open_deadline_monotonic``. ``open``
  returns 409 until a router runtime hash is published and until the writes of
  a close's drain finish. On a routed deployment it first checks every
  provider and sends one preview per Scenario; a failed check, or a missed
  deadline, returns 503 and changes nothing.
- ``activate`` publishes ``manifest_path``, ``snapshot_version``,
  ``expected_snapshot_version`` and a 64-hex ``router_config_hash`` together,
  and leaves admission closed.
- ``close`` and ``stop`` accept ``cancel_active: true`` to cancel accepted
  inference requests after their drain deadline.
- ``POST /reef/lifecycle/check`` checks a candidate manifest and changes
  nothing. It returns 200 with ``{"snapshot_version": ...}``, 409 with the
  fixed text ``routing candidate rejected`` for a candidate that the child
  would not publish, 503 when a provider fails or the deadline passes, and
  400 for a malformed body.

``reef/service/lifecycle.py`` defines the fields and the drain accounting of
each command.

Import records for cold-start learning
--------------------------------------

``POST /reef/records`` accepts an existing inference or report without calling
an inference provider. It uses the usual Bearer token and ``x-reef-scenario``
header. Submit one record per request and use a stable ``agent_record_id``:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/records \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'x-reef-scenario: my-agent' \
     -H 'Content-Type: application/json' \
     -d '{
       "agent_record_id": "offline-0001",
       "request_type": "inference",
       "payload": {
         "model": "example-model",
         "messages": [{"role": "user", "content": "What is 2 + 2?"}],
         "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]}
       }
     }'

The response is ``{"agent_record_id":"offline-0001","scenario":"my-agent",
"request_type":"inference"}``. HTTP 200 acknowledges record admission; it does
not mean that a training step has finished.

- The envelope has exactly the three fields shown; Reef rejects other fields.
  The scenario comes only from the header.
- ``request_type`` is ``inference`` or ``report``. Send a training request to
  ``POST /reef/train`` instead.
- An inference payload follows the selected recipe's recorded-input contract:
  native provider request fields with an existing ``response``, or another
  complete example that its processor understands.
- A report uses the same schema and reference validation as ``POST
  /reef/report``. Import the referenced inferences first.

Reef answers these statuses:

- 200 for an identical retry, including after capacity eviction: it returns
  the existing receipt.
- 400 for an invalid envelope, report schema or reference.
- 400 for an inference record with an image, when the deployment's recipes
  include a weight-training recipe, because training renders text only. A
  batch then stores nothing.
- 401 for an authentication failure.
- 409 for a known ID with different content.

Import does not run inference, and it does not assign the active artifact to
externally generated examples. Training reads no token IDs or log
probabilities from an imported record: the training component renders the
recorded exchange with the model's chat template. To train on complete ATIF
trajectories, see `Data items <#data-items>`__.

Imported records enter the same scenario storage and wake the same training
worker as live traffic. Keep the same scenario header for later inference
calls: their records continue through the same processor, with no processor
replacement or end-of-dataset signal. A processor that needs report feedback
needs it for imported and live records alike. Submit each example's feedback
promptly so that completed jobs can drain:

.. code:: json

   {"agent_record_id": "offline-score-0001", "request_type": "report",
    "payload": {"references": ["offline-0001"], "score": 1.0}}

The selected recipe and processor own job size, buffering, retention and
training policy. Uploading can overlap training. This API adds no dataset
epochs, no cold-start completion barrier and no automatic algorithm switch.
Use the recipe's progress and commit status to learn when training has
consumed the imported examples.

Batch imports and resume
~~~~~~~~~~~~~~~~~~~~~~~~

``POST /reef/records/batch`` accepts ``{"records": [record, ...]}``, where
each record uses the envelope of ``POST /reef/records``. The batch holds 1 to
1000 records within the 10 MiB body limit. All records share the request's
scenario and tag headers. Put an inference before the reports that reference
it, including within one batch.

Reef validates the whole batch and stores it in one database transaction. The
response is ``{"records": [receipt, ...]}``, in input order. A successful
import wakes the consumer once after commit. Atomicity applies to one request,
not to the whole dataset or to later training.

- Malformed records or missing references return 400, and conflicting IDs
  return 409. Neither stores any new record of the batch.
- Identical retries are accepted, including after training or capacity
  eviction.
- Oversized requests return 413.

For a large local dataset, use the importer of the standalone ``reef-client``
package, which needs no Reef service or training dependencies. Save one
envelope per line in an immutable UTF-8 JSONL file. Keep the original record
IDs and report references: the importer generates no IDs and converts no
other dataset schema.

.. code:: bash

   # Authentication is read from REEF_TOKEN when set.
   reef-client import records.jsonl --url http://127.0.0.1:8900 --scenario my-agent

The importer behaves as follows:

- **Batches.** It reads incrementally and sends batches limited by
  ``--batch-size`` (default 128 records) and ``--max-batch-bytes`` (default
  524288 bytes). It buffers one batch and one look-ahead record. One JSONL
  line must fit the byte limit; raise the limit, below 1048576, for larger
  records.
- **Progress.** It saves progress to ``records.jsonl.reef-import.json``, or
  to ``--progress PATH``: the destination, scenario, source checksum and last
  acknowledged byte offset and count, with no token or record payload. Each
  run first hashes the whole file to verify its identity, then seeks to the
  saved offset. The source must not change during the import. A local file
  lock prevents two runs on one progress file.
- **Resume.** Run the same command again after an interruption. A modified
  source or another destination needs a separate progress file; known IDs
  still deduplicate at the server.
- **Retries.** It retries transient connection failures and HTTP 408, 429,
  500, 502, 503 and 504 up to ``--retries`` times (default 3), with backoff
  and unchanged IDs. If a reply is lost after commit, server deduplication
  prevents a second insert. A permanent error stops the import without
  advancing that batch's progress; earlier acknowledged batches stay stored.
- **Count.** The reported count includes every acknowledged source row,
  including retries of records already present. It does not count trained
  samples.

The importer supports local JSONL on platforms with POSIX file locking,
including Linux and macOS. It does not convert Parquet or Hugging Face
schemas, and it runs no server-side S3 import. Import completion acknowledges
storage only.

The batch limits do not bound processor memory. A processor that waits for
reports retains inference records, so place each report close to the
inferences that it references.

Data items
----------

Reef keeps the data that scenarios train on as data items, in one data store
for the deployment. An item is content and metadata:

- The content is an ATIF trajectory, a Harbor task directory, a record, or a
  reference to the files of an artifact. It never changes. To change data,
  create a new item.
- The metadata is a JSON object. You can add your own keys. Reef reads these:

  - ``kind`` (``trajectory``, ``task``, ``record`` or ``artifact``) and
    ``owner``.
  - ``scenarios``: maps each scenario that uses the item by this mark to its
    split, ``train`` or ``eval``. A trajectory is always ``train``. An
    artifact item has no ``scenarios``: a scenario uses it by importing it
    (`Artifact items`_).
  - ``origin``: the scenario that made a record or an artifact. A record's
    ``origin.source`` is ``inference`` or ``evaluation``; a record without it
    is ``inference``.
  - ``benchmarks``, on a task: the names of the benchmarks that it is in
    (`Benchmarks and collections`_).
  - ``trajectory_sets``, on a trajectory: the names of the trajectory sets
    that it is in (`The sets that a scenario holds`_).

  ``kind``, ``owner``, ``origin`` and ``evaluation_id`` never change.

Each item belongs to one owner, and the routes have the owner in the path:
``/reef/namespaces/{owner}/data``. A scenario uses only the items of its
owner. A scenario's owner is the scenario's name, or the ``owner`` that ``POST
/reef/scenarios`` gave when it created the scenario. Reef refuses an item that
names a scenario of another owner. A scenario can also use a task through a
benchmark that it holds, and a trajectory through a trajectory set, with no
mark on the item (`The sets that a scenario holds`_).

The data service answers the data routes. A one-process ``reef serve`` runs
it with the other services. When the services run in separate processes,
send the data routes to the data service: control does not answer them (`Run
the services in separate processes
<../user-guide/split-services.rst>`__).

Training reads the items that a scenario uses from the data store:

- Each trajectory counts as one sample toward ``job_samples`` and
  ``buffer_size``. Job data takes the samples in the order they joined the
  scenario.
- A train task trains like a trajectory. In ``sft`` it trains only in
  ``manual`` mode, for a request that gives the job's parameters (`Manual
  training <#manual-training>`__).
- An item that leaves the scenario before a job takes it does not train. It
  keeps its train counts there, so ``auto`` does not train it again if it
  comes back.
- Archiving a scenario does not delete its items. It removes the scenario
  from their ``scenarios``, except from the record items that the scenario
  made.

For the recipe side, see `Uploaded trajectories
<../developer-guide/recipe-control-service.rst#uploaded-trajectories>`__.

After each write that adds items to a scenario or removes them, the data
service asks control to sync the scenarios that the write touched, and waits
up to 30 seconds. Each create, update and delete response has ``synced``:
true when every sync ended, false when one still waits. A sync that waits runs
when control takes it, or when the scenario loads.

An unknown item ID returns 404. A deployment without a data store returns 503.

Create trajectory items
~~~~~~~~~~~~~~~~~~~~~~~

Send trajectories as JSON Lines to ``POST
/reef/namespaces/{owner}/data/trajectories`` with ``Content-Type:
application/x-ndjson`` (or ``application/jsonl``):

- Each line is one ATIF document. Reef skips blank lines.
- A document must have an ``ATIF-v`` ``schema_version``, an ``agent`` object,
  one or more steps, and a ``session_id``, which is the item's ID.
- One request holds 1 to 1000 trajectories with distinct ``session_id``
  values, in at most 10 MiB.
- The query parameter ``metadata`` is the JSON metadata of every trajectory of
  the request. Its ``kind`` is ``trajectory``.

.. code:: bash

   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/trajectories" \
     --get --data-urlencode 'metadata={"scenarios": {"my-agent": "train"}}' \
     -X POST \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/x-ndjson' \
     --data-binary @trajectories.jsonl

The response has status 201: ``{"owner", "items": [{"id", "sequence",
"digest", "metadata"}], "synced"}``. Reef stores all of the items in one
transaction, or none. An identical retry changes nothing.

Before Reef stores the items, it checks each scenario that they name. The
scenario must have the owner, and its recipe must train on the items:

- A trajectory needs a recipe that accepts uploaded trajectories. Reef does
  not read the trajectories for this check. The recipe turns each one into a
  training sample when it reads it, and skips one that cannot train with a
  warning. The SFT recipe trains every agent turn, except the steps marked
  ``is_copied_context: true`` (`SFT
  <../user-guide/recipes/sft.rst#train-on-whole-conversations>`__), and skips
  a trajectory without such a turn. The training component renders the
  trajectory with the model's chat template when a job trains it, and leaves
  out one that cannot render.
- A task in the ``train`` split needs a recipe that trains imported tasks
  (``Sample2Job.accepts_imported_tasks``), such as ``sft``, the
  generated-task recipe, or harness evolution that runs offline (`Offline runs
  on uploaded tasks
  <../user-guide/evolve-your-harness.rst#offline-runs-on-uploaded-tasks>`__).
  The ``eval`` split takes tasks for any recipe, because an `artifact
  evaluation <#evaluate-an-artifact>`__ runs them.

Reef checks a scenario that does not exist yet when the scenario opens.

Reef returns these errors and stores nothing:

- 400 for a line that is not an ATIF document; the error names the line.
- 400 for an item that a named scenario cannot train on; the error names the
  item.
- 400 for a trajectory or a record item with an image, also one that names no
  scenario, when the deployment's recipes include a weight-training recipe,
  because training renders text only. The error names the item.
- 409 for a known ID with other content.

``POST /reef/namespaces/{owner}/data`` refuses a trajectory; it creates record
and artifact items only.

Create task items
~~~~~~~~~~~~~~~~~

``POST /reef/namespaces/{owner}/data/tasks`` creates one task item for each
Harbor task directory. The task name is the item's ID: a lowercase letter or a
digit, then lowercase letters, digits, ``.``, ``_`` and ``-``. A request
holds 1 to 1000 tasks. The request has one of two forms:

- An archive (``Content-Type: application/gzip``). The data service unpacks
  the archive and copies each task directory into the task file store.
- A reference (``Content-Type: application/json``). The client wrote each task
  directory into the task file store itself. The data service copies no file.
  Use this form for a task that is too large for an archive.

Every task must be within the limits on its files. The module
``reef.storage.tasks`` defines them, so a client can check a task before it
sends it (``task_limit_errors``):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Limit
     - Value
   * - Size of one file
     - 500 GB (``MAX_TASK_FILE_BYTES``)
   * - Entries, files and directories, directly in one directory
     - 10,000 (``MAX_TASK_DIRECTORY_ENTRIES``)
   * - Files in one task
     - 100,000 (``MAX_TASK_FILES``)
   * - Size of one task
     - no limit

**Archive form.** Each top-level entry of the gzip-compressed tar archive is
one task directory, named by its task name. The query parameter ``metadata``
is the JSON metadata of every task of the archive:

.. code:: bash

   tar -czf tasks.tar.gz -C /data/benchmark/tasks sum-391 sort-17
   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks" \
     --get --data-urlencode 'metadata={"scenarios": {"my-agent": "train"}}' \
     -X POST \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/gzip' \
     --data-binary @tasks.tar.gz

The setting ``data.max-task-archive-bytes`` limits the body (default 512 MiB;
see `The data section <configuration.rst#the-data-section>`__). The files of
one archive unpack to at most 8 GiB; create larger tasks in the reference
form. Harbor's task loader must accept each task directory.

**Reference form.** First write each task directory into the owner's
partition of the task file store, in the layout of the deployment's
``TaskFileStore`` (``storage.task-file-backend``). The partition of the owner
``<owner>`` is ``data/<owner>``, and the directory has the task name as its
name. Compute its digest with ``reef.storage.tasks.FilesDigest``, or with
``files_digest`` for a local directory. Then send the task names and digests
as JSON. ``metadata`` is optional, as in the archive form:

.. code:: bash

   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks" \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"tasks": [{"name": "sum-391", "digest": "<files digest>"}],
          "metadata": {"scenarios": {"my-agent": "train"}}}'

For each task, the task file store must hold a complete directory with that
name and digest in the owner's partition (``TaskFileStore.contains``). The
item then gets that digest. The data service does not run Harbor's task
loader on a reference: the client that wrote the files checks them.

Both forms return the same response as for trajectory items, with the items in
name order. Reef returns these errors and stores nothing:

- 400 for an archive that is not tar.gz, a link or a device in it, a path
  outside its task directories, a file at the top level, a task over a limit
  of the table above, or a directory that Harbor's task loader refuses;
- 400 for a reference body that is not ``{"tasks": [{"name", "digest"}],
  "metadata"}``, or a digest that is not 64 lowercase hexadecimal digits;
- 400 for a name that is not a task name, more than 1000 tasks, or metadata
  that is not a JSON object of kind ``task``;
- 404 for a reference to a task directory that the owner's partition does not
  hold;
- 409 for a task name that the owner holds with other files. To change the
  files of a task, replace them (below);
- 413 for an archive body larger than ``data.max-task-archive-bytes``;
- 415 for another content type.

Replace a task's files
~~~~~~~~~~~~~~~~~~~~~~

``PUT /reef/namespaces/{owner}/data/tasks/{id}`` gives the task item ``id``
new files. The new task directory has the item's ID as its name. The request
has the same two forms as a create:

- An archive of that one directory (``Content-Type: application/gzip``):

  .. code:: bash

     tar -czf sum-391.tar.gz -C /data/benchmark/tasks sum-391
     curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks/sum-391" \
       -X PUT \
       -H "Authorization: Bearer $REEF_TOKEN" \
       -H 'Content-Type: application/gzip' \
       --data-binary @sum-391.tar.gz

- A reference (``Content-Type: application/json``) to a directory that the
  client wrote into the owner's partition of the task file store:
  ``{"digest": "<files digest>"}``.

The item keeps its ID, metadata and splits. The same files again change
nothing. After a change:

- Each evaluation attempt reads the task's current files. A new evaluation,
  and a retry of a started one, run the new files. An attempt that already
  runs finishes with the old files, so one evaluation can hold scores of both.
- A trainer of a scenario that uses the task reads it again.

The response is ``{"owner", "item", "synced"}``, without content. Reef
returns these errors and changes nothing:

- 400 for an archive of no task or of several, a directory with another name,
  a directory that Harbor's task loader refuses, a reference body that is not
  ``{"digest"}``, or an item that is not a task;
- 404 for an unknown item, or a reference to a directory that the owner's
  partition does not hold;
- 413 and 415 as for a create.

Record items
~~~~~~~~~~~~

Reef makes a record item from each inference call and report of a scenario.
A training request gets no item. The item belongs to the scenario's owner:

- Its ID is ``<scenario>/<agent_record_id>``.
- Its content is ``{"agent_record_id", "request_type", "payload",
  "created_at", "references", "artifact_id"}``: the record as the scenario
  stores it.
- Its metadata is ``{"kind": "record", "scenarios": {"<scenario>": "train"},
  "origin": {"scenario", "artifact_id"}}``. ``artifact_id`` names the artifact
  that answered the call, or is null when the record names none.

The data service makes the items after the records are stored, so an item
can arrive a short time after its record. When a scenario with older records
loads for the first time after an upgrade, the data service makes their items
in pages while the scenario serves.

The scenario that made a record trains on its own record store, and always
uses its item: a patch that removes it returns 400. To train another scenario
of the same owner on a record, add that scenario to the item's ``scenarios``:

- Reef reads the record from the data store for the other scenario, with the
  item ID as its record ID. It stores no copy.
- Each reference ``x`` of a report reads as ``<origin scenario>/x``, so add a
  report together with the inferences that it references.
- The other scenario's recipe decides whether it trains on the record. For
  example, ``sft`` trains on it, and an on-policy recipe skips an inference
  that another runtime load served.

When an item leaves a scenario, new reads of that scenario stop giving it. A
training set that a process already holds in memory can still train until the
process restarts. A record store that an older Reef wrote can hold a copy of
an item, and that copy stays.

You can also create record items with ``POST /reef/namespaces/{owner}/data``,
for example records that you keep from an older system. The content must have
the keys above, ``request_type`` must be ``inference`` or ``report``, and the
ID must be ``<origin scenario>/<agent_record_id>``. Otherwise Reef returns 400.

Evaluation calls
~~~~~~~~~~~~~~~~

Reef also keeps each model call of an evaluation rollout as a record item of
the scenario's owner (`Model calls of a rollout`_). Such an item differs from
a scenario's own record item:

- Its ID is ``<scenario>/<agent_record_id>``. The record ID is
  ``<evaluation_id>:<task_index>:<repeat>:<attempt>:<index>`` for a call of
  the agent, and ``<evaluation_id>:<task_index>:<repeat>:<attempt>:verifier:<index>``
  for a call that the verifier wrote (`Calls that a verifier writes`_).
- Its content is a record of an inference. The payload is the request body,
  plus ``response``, the response body or null. ``created_at`` is the start
  of the call.
- Its metadata is ``{"kind": "record", "scenarios": {}, "evaluation_id",
  "origin": {"scenario", "artifact_id", "source": "evaluation", "evaluation"}}``.
  ``artifact_id`` names the evaluated artifact. ``evaluation`` holds the
  ``evaluation_id`` and the call's summary: ``task_index``, ``repeat``,
  ``attempt``, ``index``, ``path``, ``status``, ``started_at``,
  ``duration_s``, ``prompt_tokens``, ``completion_tokens``,
  ``finish_reason``, ``tool_calls``, ``caller`` and ``label``, as `Model
  calls of a rollout`_ describes them. So a list of the calls reads no
  content. A call without ``caller`` is a call of the agent, and has no
  ``label``.

An evaluation call is the data of no scenario:

- A patch that adds a scenario to its ``scenarios`` returns 400, so no
  scenario trains on it.
- The scenario's records route does not list it. To read the calls of one
  evaluation, query the data with ``kind=record``, ``group_key=evaluation_id``
  and the evaluation's ID as ``group_value``.
- An evaluation call links only to the earlier calls of the same attempt, so
  its trajectory node keeps only the messages after its parent's. A verifier
  call links only to the earlier verifier calls of the same attempt. A
  scenario's own call never links to an evaluation call.

A trajectory that a verifier wrote is a trajectory item of the scenario's
owner. Its ID is
``<scenario>/<evaluation_id>:<task_index>:<repeat>:<attempt>:verifier:trajectory:<index>``,
and its content is the ATIF document. Its metadata is ``{"kind":
"trajectory", "scenarios": {}, "evaluation_id", "evaluation":
{"evaluation_id", "task_index", "repeat", "attempt", "index", "name"}}``,
where ``index`` counts the trajectory files of the attempt in name order and
``name`` is the file's name. It has no ``origin``. A patch that adds a
scenario to it returns 400, so no scenario trains on it.

Artifact items
~~~~~~~~~~~~~~

The artifact items of an owner are its artifact library: the artifacts of all
of its scenarios, also of scenarios in other deployments that share the data
store, and artifacts whose files you stored yourself. An artifact item is a
reference. Its content says where the files are; the files stay there. The
content is ``{source_uri, kind, component}``:

- ``source_uri``: the directory of the files, as an import reads it (`Import
  an artifact`_), or null when there is none.
- ``kind``: the artifact's kind, such as ``model`` or ``model_adapter``, or
  null.
- ``component``: the component of the scenario that made the artifact, or null
  for your own item.

Use the item's ``origin.artifact_id`` to identify a Reef artifact. Older items
may include ``content_id``; readers accept that field but do not use it as an
artifact identity. New writes discard it. A retry of an older item preserves
its original content and digest when every other content field matches.

The data service makes an item for each artifact that a scenario lists
(``GET /reef/scenarios/{scenario}/artifacts``), after each commit. Its ID is
``<scenario>/<artifact_id>``, and its ``origin`` is ``{scenario, artifact_id}``.
Its ``source_uri`` comes from the artifact backend. A backend that keeps
snapshots where other deployments can read them, such as an object store,
gives the URI of the snapshot's directory. The built-in ``git_lfs`` backend
gives none, so its items have a null ``source_uri``.

To add files that you stored, such as an upload to an S3 prefix, create an
item with ``POST /reef/namespaces/{owner}/data``, ``metadata`` ``{"kind":
"artifact"}`` and no ``origin``. Its content needs a ``source_uri`` and a null
``component``.

To use an artifact item in a scenario, import it:
``POST /reef/scenarios/{scenario}/artifacts`` with ``{"item": "<id>"}``. The
scenario must belong to the item's owner.

- Reef reads the item's ``source_uri`` and imports the files into the
  scenario. The import's ``source`` names the item. An item without a
  ``source_uri`` returns 400.
- The artifact keeps its ID, so the ID names one artifact in each scenario
  that uses it, and the library keeps one item of it. A second import of the
  item returns that artifact and reads nothing.
- An upload gets its ID at its first use, in ``metadata.artifact_id``, which
  never changes after that.

A new scenario can also start from an artifact item: see `Create a
scenario`_.

Read, query, update and delete items
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**List items.**
``GET /reef/namespaces/{owner}/data`` lists the owner's items in the order
they arrived: ``{"owner", "items", "next"}``. The query parameters are:

- ``kind`` and ``scenario`` filter the items. ``benchmark`` keeps only the
  tasks of one benchmark, ``trajectory_set`` only the trajectories of one
  trajectory set, and ``id_prefix`` only the items whose ID starts with it.
- ``group_key`` keeps only the items of one group of the groups route: those
  with the string ``group_value`` under ``group_key`` in their metadata, or
  with no string there when ``group_value`` is not given. ``group_value``
  without ``group_key`` returns 400.
- ``after`` is the ``sequence`` of the last item of the previous page, 0 by
  default. ``limit`` is 1 to 1000, 1000 by default. ``next`` is the ``after``
  of the next page, or null after the last item.
- ``before`` pages backward: the items come newest first, from the one just
  before the sequence ``before``, and ``next`` is the ``before`` of the next
  page. For example, ``trajectory_set=week-12``, ``before`` set to a
  trajectory's sequence and ``limit=1`` give the trajectory before it in the
  set.
- ``content=1`` adds each item's ``content``, as a read of the item gives it;
  the default ``content=0`` leaves it out. Use it with ``kind=artifact`` to
  read the artifact library's kinds and sources in one page. A page of
  trajectories with content can be large.

**Group items.**
``GET /reef/namespaces/{owner}/data/groups/{key}?kind=&scenario=&before=&limit=``
groups the owner's items of ``kind`` by the string under ``key`` in their
metadata, for example the evaluation calls by ``evaluation_id``: ``{"owner",
"groups", "next"}``.

- Each group is ``{"value", "item_count", "first_sequence", "last_sequence",
  "metadata"}``. ``value`` is null for the items without a string under
  ``key``. ``metadata`` is that of the group's first item.
- The groups come newest first, by ``first_sequence``.
- ``kind`` is required. ``key`` is 1 to 64 letters, digits and underscores.
- ``scenario`` groups only the items that the scenario uses, and the counts
  and sequences cover only those items.
- ``before`` returns only the groups whose ``first_sequence`` is lower.
  ``limit`` is 1 to 1000, 1000 by default. ``next`` is the ``before`` of the
  next page, or null after the last group.
- To read the items of one group, list the items with ``group_key`` and its
  ``value``, and page with ``after`` from its ``first_sequence``. Give the
  same ``scenario`` to that query.

**List sets.**
``GET /reef/namespaces/{owner}/data/sets/{kind}?scenario=&before=&limit=``
lists the owner's sets of ``kind``, ``benchmark`` or ``trajectory_set``, that
have at least one item: ``{"owner", "sets", "next"}``.

- Each set is ``{"name", "item_count", "first_sequence", "last_sequence",
  "metadata"}``. ``metadata`` is that of the set's first item.
- The sets come newest first, by ``first_sequence``. ``before`` and ``limit``
  page them as in the groups route.
- ``scenario`` lists only the sets with an item that the scenario uses, and
  counts only those items.
- The listing takes the same time for sets of ten items and of ten thousand.
- To read the items of one set, list the items with ``benchmark`` or
  ``trajectory_set``.

**Read one item.**
``GET /reef/namespaces/{owner}/data/{id}`` returns one item with its
``content``. A task's ``content`` is null.

**Patch metadata.**
``POST /reef/namespaces/{owner}/data/metadata`` takes ``{"ids": [...],
"patch": {...}}``. Reef applies the JSON merge patch (RFC 7386) to the
metadata of each item, all or none. A null value removes a key. For example,
``{"scenarios": {"my-agent": null}}`` removes the items from ``my-agent``, and
``{"scenarios": {"my-agent": "eval"}}`` moves tasks to the ``eval`` split.
Reef checks the new uses as for a create call. A patch that changes ``kind``,
``owner`` or ``origin`` returns 400. The response is ``{"owner", "items",
"synced"}``, without content.

**Delete one item.**
``DELETE /reef/namespaces/{owner}/data/{id}`` deletes one item and returns
``{"owner", "id", "deleted": true, "synced"}``.

- The scenarios that used a trajectory or a task lose its copy. A record
  stays in the record store of the scenario that made it.
- The item of an artifact that a scenario made returns 400 while the scenario
  exists, because the next copy would make it again.
- Reef keeps the deleted item for 30 days, so you can restore it (`Restore
  deleted items`_).

**List a scenario's tasks.**
``GET /reef/scenarios/{scenario}/tasks`` returns ``{"scenario", "tasks":
[{"name", "split", "benchmark", "files_digest", "trained"}]}``: the uses of
tasks by the scenario, in the order that they began. ``benchmark`` is the held
benchmark that gives the use (`The sets that a scenario holds`_), or ``null``
for the task's own mark. A task that the scenario uses more than once is
listed once for each use. ``trained`` is true once a committed task job of any
trainer of the scenario trained the task.

**Reserved IDs.** An item ID can hold ``/``, and two GET routes come before
the item routes: ``.../data/<id>/archive`` downloads the item ``<id>``, and
``.../data/imports/<x>`` and ``.../data/exports/<x>`` read a transfer. So GET
cannot read an item whose ID ends in ``/archive``, or whose ID is
``imports/<x>`` or ``exports/<x>`` with ``<x>`` one path segment. Give such
items other IDs.

Restore deleted items
~~~~~~~~~~~~~~~~~~~~~

A delete does not remove an item at once. Reef keeps it as a deleted item of
its owner for 30 days, and then purges it. A deleted item is in no listing,
set, scenario or read of the routes above. A new item can take its ID.

- ``GET /reef/namespaces/{owner}/deleted-items?kind=&before=&limit=`` lists
  the owner's deleted items, newest item first by ``sequence``: ``{"owner",
  "items", "next"}``. Each item is ``{"id", "sequence", "digest", "metadata",
  "deleted_at", "purge_at"}``, without content. ``deleted_at`` and
  ``purge_at`` are seconds since the epoch. ``kind`` keeps one kind of item.
  ``before`` and ``limit`` page the items as in the item list; ``next`` is the
  ``before`` of the next page, or null after the last item.
- ``POST /reef/namespaces/{owner}/deleted-items/restore`` takes ``{"ids":
  [...]}``, 1 to 1000 IDs, and restores the items, all or none. An item comes
  back with its ``sequence``, metadata and content, and in its sets. Reef
  checks its ``scenarios`` as for a create call. Each scenario that uses it
  again reads it again, also through a held set. The response is ``{"owner",
  "items", "synced"}``, without content. An ID without a deleted item returns
  404. An ID that the owner uses for an item again returns 409.
- ``DELETE /reef/namespaces/{owner}/deleted-items/{id}`` purges one deleted
  item for good and returns ``{"owner", "id", "purged": true}``. An ID without
  a deleted item returns 404.

The archive of a scenario removes it from the ``scenarios`` of the deleted
items too. A reader that pages items with ``after`` and is past the
``sequence`` of a restored item does not see it again; a scenario reads it
again through its uses.

Benchmarks and collections
~~~~~~~~~~~~~~~~~~~~~~~~~~

A benchmark is a name for a set of the owner's tasks. Reef does not store it:
a task is in a benchmark when the list ``benchmarks`` of its metadata has the
name. Set the list when you create the tasks, or with a metadata patch:
``metadata={"benchmarks": ["terminal-bench"]}``. It holds distinct non-empty
strings, so a task can be in several benchmarks. Only a task has it.

A collection is a stored list of benchmarks under one name. Use it to make a
scenario hold several benchmarks in one call, and to evaluate an artifact on
them (`Evaluate an artifact`_).

.. code:: bash

   curl -f -X PUT http://127.0.0.1:8900/reef/namespaces/my-agent/collections/coding \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"benchmarks": ["terminal-bench", "swe-bench-lite"]}'

- ``PUT /reef/namespaces/{owner}/collections/{name}`` creates the collection,
  or replaces the one with the name. The body takes ``benchmarks``,
  ``collections`` and ``average``. ``collections`` names other collections of
  the owner: the collection also has their benchmarks. Each list is empty
  by default, so ``{}`` creates an empty collection. ``average`` is ``task`` (the default) or
  ``benchmark``: how an evaluation on the collection computes its
  ``mean_score``. The response is ``{"owner", "name", "benchmarks",
  "collections", "average"}``. Reef returns 400 for a name in ``collections``
  that the owner does not store, and for collections that name each other in
  a cycle.
- ``GET /reef/namespaces/{owner}/collections`` lists the owner's collections
  by name: ``{"owner", "collections"}``.
- ``GET /reef/namespaces/{owner}/collections/{name}`` returns one collection
  with ``benchmark_task_counts``, a list of ``{"benchmark", "task_count"}``
  for each of its benchmarks, and ``task_count``. ``task_count`` counts a task
  that is in two benchmarks once.
- ``DELETE /reef/namespaces/{owner}/collections/{name}`` deletes the
  collection and returns ``{"owner", "name", "deleted": true}``. No task
  changes. Reef returns 400 while another collection names it.
- ``POST /reef/namespaces/{owner}/collections/{name}/scenarios`` takes
  ``{"scenario", "split"}``. The scenario then holds each benchmark that the
  collection has now, in that split (`The sets that a scenario holds`_). A
  benchmark that joins the collection later is not held. The response is
  ``{"owner", "name", "scenario", "split", "holds", "task_count", "synced"}``:
  ``holds`` lists the holds, and ``task_count`` counts a task of two
  benchmarks once. A collection without a benchmark returns 400.

An unknown collection returns 404.

The sets that a scenario holds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A scenario holds sets of items: benchmarks, which are sets of tasks, and
trajectory sets. It then uses the items of the set. A hold writes no item, so
it takes the same time for a set of ten items and of ten thousand.

A trajectory set is a name for a set of the owner's trajectories, as a
benchmark is for tasks: a trajectory is in the set when the list
``trajectory_sets`` of its metadata has the name. Set the list when you
upload the trajectories, or with a metadata patch:
``metadata={"trajectory_sets": ["week-12"]}``. It holds distinct non-empty
strings, so a trajectory can be in several sets. Only a trajectory has it.
The trajectories that a scenario's trainer makes from its records are in the
set ``samples:<scenario>``.

.. code:: bash

   curl -f -X PUT http://127.0.0.1:8900/reef/namespaces/my-agent/scenarios/my-agent/benchmarks/terminal-bench \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"split": "eval"}'

   curl -f -X PUT http://127.0.0.1:8900/reef/namespaces/my-agent/scenarios/my-agent/trajectory-sets/week-12 \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{}'

The routes of benchmarks are under
``/reef/namespaces/{owner}/scenarios/{scenario}/benchmarks``, and those of
trajectory sets under ``…/trajectory-sets``.

- ``PUT …/{name}`` makes the scenario hold the set. For a benchmark the body
  is ``{"split"}``, ``train`` or ``eval``. A scenario can hold one benchmark
  in both splits, with one call for each: the two holds are apart. For a
  trajectory set the body is ``{}``: a trajectory always trains.
  The response is ``{"owner", "scenario", "kind", "name", "split",
  "item_count", "synced"}``; ``kind`` is ``benchmark`` or ``trajectory_set``.
  A ``train`` hold of a benchmark needs a recipe that trains imported tasks,
  and a hold of a trajectory set one that takes uploaded trajectories, as a
  mark does; else Reef returns 400.
- ``DELETE …/{name}?split=<split>`` ends the hold in that split: ``{"owner",
  "scenario", "kind", "name", "split", "released", "synced"}``. A benchmark
  needs ``split``; a trajectory set does not. ``released`` is false when the
  scenario did not hold the set in that split.
- ``GET`` of the two base paths returns ``{"owner", "scenario", "kind",
  "holds"}``: each hold with its ``split`` and its ``item_count``, in the
  order of the names.

The membership is live. An item that joins a held set, by a create or by a
metadata patch, joins the scenario, and an item that leaves the set leaves it.

A scenario uses an item once for each held set that the item is in, and once
more when the item's own ``scenarios`` metadata names the scenario. Reef
compares no item lists and refuses no overlap:

- A task in two held ``train`` benchmarks has two uses. A trainer reads the
  train uses as a set of items, so a job holds the task once, and the task has
  one train count. The same holds for a trajectory with a mark and a held set.
- A task in a ``train`` benchmark and in an ``eval`` benchmark is trained, and
  an evaluation runs it. The same holds for each task of a benchmark that the
  scenario holds in both splits. Reef does not prevent it: hold a benchmark in
  one split when the evaluation must not see trained tasks.

Download an item
~~~~~~~~~~~~~~~~

``GET /reef/namespaces/{owner}/data/{id}/archive`` downloads one item as an
attachment, named after the last segment of its ID:

- A task downloads as ``<name>.tar.gz`` (``application/gzip``), with its files
  in the directory ``<name>``.
- Another item downloads as ``<name>.json`` (``application/json``): its
  content.

An unknown ID returns 404. To let a browser or another service download the
item without the token, ask for a `signed download URL
<#signed-download-urls>`__.

Import and export items
~~~~~~~~~~~~~~~~~~~~~~~

An import stores the files at a URI as items of the owner. An export writes
items of the owner to a tar.gz at a URI. Each runs in the background on a
data service worker:

- ``POST /reef/namespaces/{owner}/data/imports`` and ``POST
  /reef/namespaces/{owner}/data/exports`` start one and return 202 with its
  status. A body that is not valid returns 400.
- ``GET /reef/namespaces/{owner}/data/imports/{id}`` and ``GET
  /reef/namespaces/{owner}/data/exports/{id}`` read the status. An unknown
  ID, or the ID of another owner's transfer, returns 404.

The status is ``{"id", "kind", "state", "request", "progress", "result",
"error"}``:

- ``kind`` is ``data_import`` or ``data_export``.
- ``state`` is ``queued``, ``assigned``, ``running``, ``finished`` or
  ``failed``.
- ``request`` is the body that started the transfer.
- ``progress`` is ``{"items"}`` after a trajectory import stores its first
  transaction: the count of items stored so far. It is ``null`` before
  that, and for a task import and an export. A retry stores the same
  transactions again, which changes no item, and ``progress`` does not go
  back: it changes only when the retry stores more than the attempts before.
- ``result`` is set when the transfer finished: ``{"items"}``, the item count,
  for an import, and ``{"items", "bytes"}`` for an export.
- ``error`` tells why a run failed.

An import body is ``{"uri", "metadata"}``. Every item gets ``metadata``, whose
``kind`` is ``task`` or ``trajectory``. The data service reads the URI with
the readers of `artifact imports <#import-an-artifact>`__: ``hf://``,
``s3://``, ``file://`` and the schemes that ``storage.artifact-sources``
adds. It finds the items as follows:

- ``task``: when the URI's root holds ``task.toml``, the root is one task,
  named after the last segment of the URI's path. Otherwise each directory
  directly under the root that holds ``task.toml`` is one task, named after
  the directory.
- ``trajectory``: each ``*.jsonl`` file and each gzip ``*.jsonl.gz`` file at
  the root holds one ATIF document a line, as the upload takes them. Each
  document's ``session_id`` is its item ID; two documents of the import never
  share one. Reef reads no other file. It reads the files in name order, in
  two passes:

  1. Reef checks every line. A bad line, or a ``session_id`` that an earlier
     line has, fails the import, and Reef stores no item.
  2. Reef stores the items one transaction at a time, each of at most 1000
     items and about 32 MiB of lines. A check that only the data store makes,
     such as a conflict with an item that the owner stores already, stops the
     import. The items of the earlier transactions stay.

The import checks and stores the items as the create routes do. It does not
wait for the syncs of the scenarios that it touched.

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/namespaces/my-agent/data/imports \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"uri": "s3://my-bucket/benchmark/tasks/",
          "metadata": {"kind": "task", "scenarios": {"my-agent": "eval"}}}'

An export body is ``{"uri"}`` and one of these selections:

- ``ids``: 1 to 1000 item IDs. An unknown ID fails the export.
- ``kind`` and ``scenario``, each optional: the items that match, as for the
  list route. With neither, the export writes every item of the owner.

The archive holds ``manifest.jsonl``, one line ``{"id", "kind", "metadata",
"path"}`` for each item, in export order. The content of a trajectory or a
record is at ``items/<n>.json``, and the files of a task are under
``items/<n>/``, where ``<n>`` counts the items from 0. The export writes these
URIs:

- ``file:///absolute/path``: a new file on the data service's machine. The
  parent directory must exist. The export never replaces a file: when the file
  exists, the export fails.
- ``s3://bucket/key``: one object, written with boto3's standard credentials.
  It needs the ``s3`` extra (``uv pip install 'reef-infra[s3]'``).

Another scheme fails the export.

Signed download URLs
~~~~~~~~~~~~~~~~~~~~

Add ``?signed=1`` to ``GET /reef/namespaces/{owner}/data/{id}/archive`` or to
``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files`` (`Artifacts`_)
to get a URL that opens the download without the token. The response is
``{"url", "expires_at"}``:

- ``url`` is the path and query of the download on the same service. It
  carries ``expires`` and a ``signature``, which Reef computes from the token
  that the request presented.
- ``expires_at`` is the time, in Unix seconds, after which the URL returns 401.

``expires_in`` sets the life of the URL in seconds: 1 to 86400, 3600 by
default. Another value returns 400. Send the URL with GET and without an
Authorization header.

- The URL keeps the request's ``path`` and ``component`` queries, and it opens
  only that download: another path, or another ``path`` or ``component``
  query, returns 401.
- When the service no longer accepts the token that signed it, the URL
  returns 401.
- A service without tokens returns the URL without a signature, because the
  download needs none.

Choose the samples that train
-----------------------------

Every training sample of a scenario is a data item that it uses: an uploaded
trajectory, an imported task, or a trajectory or task that a trainer's stages
made. ``GET /reef/scenarios/{scenario}/samples`` lists them in the order that
their uses began, at most ``limit`` (default and maximum 1000) after the
sequence ``after`` (default 0):

.. code:: json

   {"scenario": "my-agent", "next": null, "samples": [
     {"sample_id": "trajectory-0001", "sequence": 1, "kind": "trajectory", "split": "train",
      "component": null, "group": null, "held_set": null, "train_counts": {}},
     {"sample_id": "my-agent.report-17:0", "sequence": 2, "kind": "trajectory", "split": "train",
      "component": "", "group": null, "held_set": null, "train_counts": {"": 1}}]}

- ``component`` is the trainer that made the sample from records (``""`` for
  the only trainer of a scenario), or ``null`` for an upload or an imported
  task.
- Samples that share a ``group`` train together.
- ``held_set`` is the name of the held benchmark or trajectory set that gives
  the use, or ``null``. An item that the scenario uses more than once has one
  entry for each use.
- ``train_counts`` holds, for each trainer, the number of committed jobs that
  trained the sample.
- ``next`` is the ``after`` of the next page, or ``null`` after the last one.

``GET /reef/scenarios/{scenario}/next-job`` returns the samples that the next
automatic job trains, in use order, with the trainer's ``training_mode``.
Each sample has ``sample_id``, ``kind``, ``group``, ``component`` and the
trainer's ``train_count``. A scenario of several components needs
``?component=<name>``.

In ``manual``, a request names the samples that its job trains: the
``samples`` of ``POST /reef/train`` (see `Manual training <#manual-training>`__):

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: my-agent" \
     -H "Content-Type: application/json" \
     -d '{"text": "Train these", "session": "s-1", "artifact_id": "a-1",
          "samples": ["trajectory-0001", "report-17:0"]}'

- ``samples`` is one or more distinct sample IDs, of one kind: trajectories or
  tasks, not both. The request body limit of 64 MiB bounds the list.
- Reef returns 400, and queues nothing, for an ID that the store does not
  hold, or a sample that no trainer can train (an eval or an imported task,
  an upload that its recipe does not take, another trainer's sample).
- A sample that trained before can be named again; it trains again. Staleness
  does not apply: the client chooses.
- A sample that leaves the scenario before the job starts is left out of the
  job.

Evaluate an artifact
--------------------

Use these routes to score one artifact of a scenario on the scenario's eval
tasks. An artifact is one component of one of the scenario's artifacts, such
as the ``weights`` of a model checkpoint. A policy is a harness and a model,
and the artifact is one side of it, so the request configures the other side.
Reef runs the rollouts itself and stores the result with the scenario. The
evaluation never changes what the scenario serves, and its model calls are not
records: Reef keeps them only in the evaluation's call log (`Model calls of a
rollout`_). A scenario can also hold settings that Reef evaluates each new
artifact with (`Automatic evaluation`_).

Reef evaluates a model artifact (a ``model`` or ``model_adapter`` component)
with a Harbor agent, on a runtime that serves any artifact on request
(``InferenceRuntime.serves_any_artifact``). The Tinker runtime is the built-in
runtime that does. Reef refuses a harness artifact, a model artifact on a
runtime without that capability, and a scenario whose artifacts also hold a
harness, with 400.

First create the tasks as task items in the scenario's ``eval`` split
(`Create task items <#create-task-items>`__). Then start the evaluation:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/scenarios/my-agent/evaluations \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"artifact_id": "8f3c...", "harness": {"name": "terminus-2"}, "rollouts_per_task": 2}'

The body takes these fields:

- ``artifact_id`` (required): one of the scenario's artifacts, by the
  ``artifact_id`` that ``GET /reef/scenarios/{scenario}/artifacts`` lists.
- ``component``: the component to evaluate. Leave it out when the artifacts
  hold one component with a kind.
- ``harness``: the agent that runs a model artifact (`The harness field`_).
- ``model``: the model binding that a harness artifact runs on. A request
  sets exactly one of ``harness`` and ``model``.
- ``task_names``: the eval tasks to run. Leave it out to run every eval task.
  The stored request names the tasks that the evaluation runs.
- ``collection``: a collection of the scenario's owner (`Benchmarks and
  collections`_). Reef runs the eval tasks that are in the collection, each
  one once. A request sets ``collection`` or ``task_names``, not both. Reef
  returns 400 for an unknown collection, and for one with no eval task.
- ``num_samples``: the most tasks to run, a positive integer, or null (the
  default) to run every task that ``task_names`` or ``collection`` selects.
  When more tasks are selected, Reef runs the ``num_samples`` tasks whose
  names have the smallest SHA-256 digests. The choice depends on the task
  names only, so evaluations of the same tasks with the same ``num_samples``
  run the same tasks, and their scores compare. The stored request names the
  tasks in ``task_names``, and a stored ``collection`` leaves out a benchmark
  with no chosen task.
- ``sample_seed``: an integer, or null (the default). With a seed, Reef runs
  the tasks in a random order that the seed fixes: it ranks the SHA-256
  digests of ``"<sample_seed>:<task name>"``. With ``num_samples``, the
  evaluation runs the first ``num_samples`` tasks of that order, so a new
  seed runs another random part of the tasks. Without ``num_samples``, it
  runs every task in that order. The stored ``task_names`` are in the order
  that the evaluation runs them.
- ``rollouts_per_task``: the rollouts of each task, 1 by default.
- ``environment``: the Harbor environment type, ``docker`` by default.
- ``max_concurrent_rollouts``: the most rollouts of this evaluation that run
  at once, or null (the default) for no limit. Each rollout holds a task
  sandbox and sends model calls to the candidate, so set it to what the
  sandbox provider and the candidate's serving can take. Reef adds the next
  rollouts to the queue as earlier ones end. On ``e2b``, Reef adds a task's
  rollouts only after it built the task's sandbox templates, and a build holds
  no place of this limit.
- ``max_failed_fraction``: the largest fraction of failed rollouts, a number
  greater than 0 and less than 1, or null (the default) for no limit. A
  rollout fails when its work fails or when it ends with no score. After at
  least 10 rollouts end (or all of them, for a smaller evaluation), Reef stops
  the evaluation when more than this fraction of the ended rollouts failed. It
  fails with an error such as ``too many rollouts failed: 37 of 45 ended
  (82%), over the limit of 50%``. Each rollout that did not end fails with
  ``stopped: too many rollouts failed``, and the rollouts that wait never
  start.
- ``log_probs``: ``true`` to keep the sampled tokens' log probabilities and
  token IDs in each model call of the evaluation; ``false`` by default (`Log
  probabilities of an evaluation`_).

Reef returns:

- 202 with the stored evaluation, whose ``state`` is ``running``;
- 400, and runs nothing, for a request that the scenario cannot run: an
  unknown field or task, the wrong side configured, a component that is no
  side of a policy, an artifact or a runtime that Reef refuses, or no eval
  task;
- 404 for an unknown scenario, artifact or evaluation.

Reef sets no limit on the evaluations that run at once: the evaluations queue
holds the rollouts that wait. An evaluation process runs at most
``evaluation.max_sandboxes`` rollouts at once, across all evaluations (see
`Configuration <configuration.rst>`__).

The harness field
~~~~~~~~~~~~~~~~~

``harness`` has one of three forms:

- ``{"name": "terminus-2"}`` runs Reef's default terminus-2 configuration.
- Any other Harbor agent is its whole Harbor configuration, with ``{model}``,
  ``{base_url}`` and ``{api_key}`` where Reef fills them in, as the Tinker job
  parameter ``rollout.agent`` takes it. A Harbor agent takes variables in its
  own ``env`` key.
- ``{"item": "<item id>"}`` runs a harness tree: a published artifact item of
  the scenario's owner, also from another scenario.

Each agent runs in the task's container. On ``docker``, the container reaches
the model at ``evaluation.agent_host``. In a remote environment, such as
``modal``, the agent bridges the container's loopback to the model through
Reef's sandbox relay and Harbor's exec, also for a task with several compose
services.

These rules apply to an item:

- The item has the kind ``harness_plugin``, and ``adapter`` in its metadata.
  Its adapter must install with npm, uv or git and call the runtime's API
  dialect.
- Reef refuses a task that declares MCP servers or skills.
- The evaluation does not reach the activation rule, and an automatic
  evaluation cannot use an item.
- ``"env": {"NAME": "value"}`` gives each command of the agent these
  variables. A name is a shell variable name, and a value is text without
  ``${``. ``REEF_CONTEXT_WINDOW`` and ``REEF_MAX_OUTPUT_TOKENS`` also set the
  model's token limits in the rendered tree, each a positive number of tokens
  (`Token limits <../developer-guide/harness-adapters.rst#token-limits>`__).
- ``"config": {"<target>": {...}}`` merges each JSON object into the tree's
  ``config/<target>.json``, as if the tree held it. The tree with the config
  must pass the adapter's validation (`Config checks
  <../developer-guide/harness-adapters.rst#config-checks>`__), so the config
  sets no key that Reef's model binding writes. Reef refuses the evaluation
  with 400 otherwise.

Log probabilities of an evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``log_probs: true``, Reef adds the request fields that ask for log
probabilities and token IDs to each call that any agent sends, over the
agent's own values:

- ``/v1/chat/completions``: ``logprobs``, ``return_token_ids`` and
  ``return_prompt_token_ids``, all ``true``.
- ``/v1/messages``: ``logprobs`` and ``return_token_ids``, all ``true``
  (Reef's extension; see `Inference`_).
- ``/v1/responses``: ``message.output_text.logprobs`` in ``include``.

The kept call holds the engine's answer with these fields, and its ATIF
trajectory step holds them in ``metrics``. Each call's body grows with one
entry for each sampled token.

The candidate's engine must accept the fields. Reef's SGLang and vLLM handlers
accept all of them. SGLang's and vLLM's OpenAI servers accept the Chat
Completions and Responses fields. A provider such as Anthropic's API refuses
unknown fields, so do not set ``log_probs`` for an artifact that a provider
serves.

Read an evaluation
~~~~~~~~~~~~~~~~~~

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}`` returns the
evaluation as it runs and after it ends:

.. code:: json

   {"evaluation_id": "1d0e...", "state": "finished",
    "request": {"artifact_id": "8f3c...", "component": "weights", "harness": {"name": "terminus-2"},
                "model": null, "task_names": ["sum-391"], "collection": null, "rollouts_per_task": 2,
                "environment": "docker",
                "max_concurrent_rollouts": null, "max_failed_fraction": 0.5},
    "created_at": 1793169552.2, "finished_at": 1793169840.7,
    "policy_evaluation": {"harness": {"base": "terminus-2", "plugin": {}},
                          "model": {"name": "Qwen/Qwen3-8B", "adapter": "tinker://.../sampler"},
                          "scores": [{"task": "sum-391", "scores": [1.0, 0.0]}]},
    "mean_score": 0.5, "benchmark_scores": [], "error": null,
    "progress": {"total": 2, "ended": 2, "scored": 2, "failed": 0, "cancelled": 0, "running": 0,
                 "mean_score": 0.5, "eta_seconds": null, "benchmarks": []}}

- ``state`` is ``queued``, ``running``, ``finished`` or ``failed``.
- A finished evaluation holds its ``policy_evaluation``: the harness and the
  model that its rollouts ran, and each rollout's score by task (see `Rollout
  <glossary.rst#rollout>`__). A failed one holds its ``error``.
- ``mean_score`` is the mean of the rollouts that have a score: a failed
  rollout does not count as 0.
- An evaluation whose rollouts the evaluations queue lost when the service
  stopped, as an in-memory queue does, reads as ``failed``. Reef does not
  resume it.

``GET /reef/scenarios/{scenario}/evaluations`` lists the evaluations, newest
first, at most ``limit`` (50 by default).

For an evaluation on a collection, the stored ``collection`` is ``{"name",
"average", "benchmarks"}``. ``benchmarks`` gives ``{"benchmark",
"task_names"}`` for each benchmark that has a task in the evaluation.
``benchmark_scores`` then gives ``{"benchmark", "task_count",
"mean_score"}`` for each of them: the number of its tasks in the evaluation,
and the mean of their rollouts that have a score. A task in two benchmarks
counts in both. With the average ``task``, ``mean_score`` is the mean of all
rollouts that have a score. With ``benchmark``, it is the mean of the
benchmark means. ``benchmark_scores`` is empty for another evaluation.

``progress`` counts the rollouts of the evaluation, also while it runs. It is
null when the evaluations queue no longer holds them:

- ``total`` is every rollout of the evaluation: its tasks times
  ``rollouts_per_task``.
- ``ended`` counts the rollouts that finished or failed. ``scored`` counts the
  ended rollouts with a score, and ``failed`` the others.
- ``cancelled`` counts the rollouts that a cancel or a stop for failed
  rollouts ended. They are not in ``ended``.
- ``running`` counts the rollouts that a worker holds now.
- ``mean_score`` is the mean of the scores so far, or null.
- ``eta_seconds`` estimates the seconds until every rollout ends: the
  rollouts left at the rate at which rollouts ended in the last 20 minutes. It
  is null when no rollout ended in that time, or when the evaluation ended.
- ``benchmarks`` gives the progress of each benchmark of an evaluation on a
  collection, in the collection's order: ``{"benchmark", "total", "ended",
  "scored", "failed", "cancelled", "running", "mean_score"}``, with the same
  meaning for the rollouts of the benchmark's tasks. A task in two benchmarks
  counts in both. It is empty for another evaluation.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts``
shows the progress of an evaluation while it runs. It answers ``{scenario,
evaluation_id, rollouts}``: each rollout in task and repeat order, as
``{task, task_index, repeat, state, attempt_count, started_at, ended_at,
score, rewards, error, calls, verifier_calls}``.

- ``state`` is the rollout's state in the evaluations queue: ``queued``,
  ``assigned``, ``running``, ``finished`` or ``failed``. It is ``waiting``
  while ``max_concurrent_rollouts`` keeps the rollout out of the queue, also
  when a retry runs the rollout again. On ``e2b``, it is also ``waiting`` while
  the task's sandbox templates build.
- ``attempt_count`` is how many times a worker took the rollout.
- ``started_at`` is when a worker first took the rollout, and ``ended_at``
  when it finished or failed, in epoch seconds. Each is null until then.
- ``score`` is the rollout's score once it finished, else null.
- ``rewards`` holds every finite reward that the rollout's verifier wrote, by
  name, for example ``{"reward": 0.9, "rule": 1.0, "judge": 0.8}``. The
  score is the primary reward among them. It is null for a rollout that has
  not finished, that failed, whose verifier wrote no reward, or that finished
  before Reef kept rewards. Reef keeps at most 64. Nothing else reads them.
- ``error`` is the reason that a failed rollout failed, else null.
- ``calls`` holds the totals of the rollout's model calls over all its
  attempts: ``{count, failed, prompt_tokens, completion_tokens}``. ``failed``
  counts the calls whose status is not 200. The token totals add the known
  counts only. ``calls`` is null when Reef kept no call of the rollout.
  ``calls`` counts the agent's calls only.
- ``verifier_calls`` holds the same totals of the calls that the rollout's
  verifier wrote (`Calls that a verifier writes`_), or null.

To add up the totals, Reef reads every kept call of the evaluation. The query
``calls=false`` leaves out ``calls`` and ``verifier_calls``, so the read stays
fast for an evaluation with many calls. ``calls`` is ``true`` or ``false``,
and ``true`` when absent. Another value returns 400. An unknown evaluation
returns 404.

Cancel, retry and resume
~~~~~~~~~~~~~~~~~~~~~~~~

Control answers these three routes. An unknown scenario or evaluation returns
404.

``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/cancel``
cancels one evaluation. It answers the evaluation after the cancel, as ``GET
/reef/scenarios/{scenario}/evaluations/{evaluation_id}`` returns it:

- A queued or running evaluation fails with the error ``cancelled by
  request``. Each of its rollouts that did not end fails with the same error.
- A worker that runs one of these rollouts stops its trial and the trial's
  sandbox within seconds.
- The rollouts that wait for ``max_concurrent_rollouts`` do not start. They
  stay ``waiting`` in the rollouts list until a resume.
- An evaluation that ended stays as it is, so a second cancel changes nothing.

Two routes run rollouts of an evaluation again. Each one answers 202 with the
evaluation, which is ``running``:

+------------+----------------------------------------------------------------+
| Route      | Rollouts that it runs again                                    |
+============+================================================================+
| ``retry``  | The failed rollouts: the rollouts whose own run ended with no  |
|            | score. The run failed, or it finished with a failed score.     |
+------------+----------------------------------------------------------------+
| ``resume`` | The rollouts that a cancel or a ``max_failed_fraction`` stop   |
|            | ended, also in the middle of their run, and the rollouts that  |
|            | never started.                                                 |
+------------+----------------------------------------------------------------+

The routes are ``POST
/reef/scenarios/{scenario}/evaluations/{evaluation_id}/retry`` and ``POST
/reef/scenarios/{scenario}/evaluations/{evaluation_id}/resume``. Both work in
the same way:

- Each rollout runs from the start of its task. A rollout that a cancel
  stopped in its run does not continue from where it stopped.
- Its earlier attempts stay, so its ``attempt_count`` goes on and the calls of
  each attempt stay readable.
- Both keep ``max_concurrent_rollouts``, unless the body changes it. When the
  evaluation has fewer free places than rollouts to run, the others are
  ``waiting`` and go back to the queue as earlier rollouts end.
- A rollout that has a score keeps it. A rollout that has not ended stays as
  it is.
- An evaluation that runs goes on, with those rollouts in the queue again.
- An evaluation that ended is ``running`` again, with no result and no error.
- When the rollouts are final, Reef ends the evaluation as it ends a first
  run: it stores the scores, and it gives a finished evaluation to the
  scenario's activation rule, which can activate the artifact.

The body is optional. With no body, or ``{}``, the evaluation keeps its
request. ``{"max_concurrent_rollouts": 8}`` changes the most rollouts of the
evaluation that run at once, and ``{"max_concurrent_rollouts": null}``
removes the limit. The value is a positive integer or ``null``, as in a start
request. The evaluation keeps the new value in its ``request`` for the rest of
its run, also for the rollouts that wait:

.. code-block:: bash

   curl -X POST \
       -H "Authorization: Bearer $REEF_TOKEN" \
       -H 'Content-Type: application/json' \
       -d '{"max_concurrent_rollouts": 8}' \
       "$REEF_URL/reef/scenarios/$SCENARIO/evaluations/$EVALUATION_ID/resume"

A higher limit sends more waiting rollouts to the queue at once. A lower limit
does not stop a rollout that runs. Fewer rollouts start until the rollouts that
are not final are fewer than the limit.

After a cancel or a ``max_failed_fraction`` stop:

- A retry does not run the cancelled rollouts and does not start the other
  rollouts. When the failed rollouts end, the evaluation fails again. Its
  error is ``cancelled by request`` after a cancel. After a stop, it starts
  with ``too many rollouts failed``, as the error of the stop does.
- While a cancelled rollout waits for a resume, ``max_failed_fraction`` does
  not count it.
- To finish an evaluation that too many failures stopped, retry it first, and
  then resume it. A resume first counts the same failures, so it can stop
  again.

The two routes return:

- 409 from ``retry`` for an evaluation with no failed rollout;
- 409 from ``resume`` for an evaluation with no cancelled rollout and no
  rollout that never started;
- 409 for an ended evaluation whose rollouts Reef no longer keeps: Reef
  deletes the rollouts of an ended evaluation after their retention;
- 400, and change nothing, for a body with another key or with a
  ``max_concurrent_rollouts`` that is not a positive integer or ``null``.

Model calls of a rollout
~~~~~~~~~~~~~~~~~~~~~~~~

Reef keeps each model call that a rollout's agent makes to the candidate,
failed calls too. It keeps a call when the call ends, so a client that polls
sees the calls of a running rollout. After each attempt, Reef also keeps the
calls that the task's verifier wrote, for example the calls of a judge
(`Calls that a verifier writes`_).

Each call is a record item of the scenario's owner (`Evaluation calls`_). No
scenario trains on it, and record retention does not delete it. After an
archive of the scenario, the items stay in the data store, but these routes no
longer reach them, because the evaluation is gone.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls``
answers the calls of one rollout, without their bodies:

.. code:: json

   {"scenario": "my-agent", "evaluation_id": "1d0e...", "task_index": 3, "repeat": 1,
    "calls": [{"caller": "agent", "label": null, "attempt": 1, "index": 0,
               "path": "/v1/chat/completions", "status": 200, "started_at": 1762590001.2,
               "duration_s": 1.84, "prompt_tokens": 2100, "completion_tokens": 340,
               "finish_reason": "tool_calls", "tool_calls": 1}]}

- ``caller`` is ``agent`` for a call of the agent, and ``verifier`` for a
  call that the verifier wrote. By default the route answers the agent's
  calls. Add ``?caller=verifier`` for the verifier's calls. Another value
  returns 400.
- ``label`` is the label that the verifier gave the call, else null.
- ``attempt`` is the 1-based attempt of the rollout that made the call, and
  ``index`` the 0-based order in which the calls of that caller and attempt
  ended. The list is in attempt and then index order.
- ``status`` is the HTTP status that the agent got. ``started_at`` is in
  epoch seconds, and ``duration_s`` in seconds.
- ``prompt_tokens``, ``completion_tokens`` and ``finish_reason`` come from the
  reply of a Chat Completions, Responses or Messages call. Each is null when
  the reply does not give it, and for a failed call. A Messages call adds its
  cache tokens to ``prompt_tokens``. The ``finish_reason`` of a Responses call
  is its ``incomplete_details.reason``, else its ``status``.
- ``tool_calls`` is the number of tool calls in the reply, 0 when none.

The list is empty for a rollout without a kept call. An unknown evaluation
returns 404.

``GET .../rollouts/{task_index}/{repeat}/calls/{attempt}/{index}`` answers one
call: the fields above, plus ``request``, the JSON body that the agent sent,
and ``response``, the JSON body that it got or the error body of a failed
call. Each is null when the call had no JSON body. Add ``?caller=verifier``
for a call that the verifier wrote. A call that Reef did not keep returns
404.

Calls that a verifier writes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A judge calls its own model, so its calls do not reach the candidate's
endpoint. A task's verifier can write these calls, and the trajectories of an
agent judge, to ``/logs/verifier/judge/``. Harbor downloads the verifier
directory after the verifier runs, and Reef keeps the files with the attempt.
Both files are optional:

- ``calls.jsonl``: one JSON object for each call, in call order:
  ``{"request", "response", "status", "started_at", "duration_s", "path",
  "label"}``. ``request`` is the JSON body that the judge sent, and
  ``response`` the JSON body that it got, or null. ``status`` is the HTTP
  status (default 200). ``started_at`` is in epoch seconds, and
  ``duration_s`` in seconds. ``path`` defaults to ``/v1/chat/completions``.
  ``label`` is an optional string of 1 to 200 characters, for example
  ``judge:character_fidelity``. Reef ignores other keys.
- ``trajectory*.json``: ATIF documents, one in each file.

Reef reads the files for a failed verifier too. The files never change the
score. A task that sets the verifier's ``include_logs`` or ``exclude_logs``
must keep ``judge/``.

Reef reads at most 32 MiB of these files and keeps at most 2000 calls for
each attempt. It skips these and logs how many it skipped:

- a line that is no such call, and a call over the limit;
- a file that is no ATIF document, and a file over the limit;
- a symbolic link.

Trajectories of an evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reef makes ATIF trajectories from the kept calls of each attempt of a
rollout. It links the calls with status 200 into conversations, as it links
the records of training samples: a call continues the earlier call whose
messages its own messages start with. An agent that plays several roles, each
with its own messages, has one conversation for each role. Reef joins the
calls of each conversation in index order, without the calls that a retry
replaced, and orders the conversations by their first calls. An attempt
without a 200 call has no trajectory.

- Each call is one entry of ``extra.reef.records``. Its ``agent_record_id`` is
  the call's record ID (`Evaluation calls`_), and its
  ``created_at`` is the call's ``started_at``. Its ``payload`` is the request
  body without the conversation, the tools and the response
  (``call_summary``), such as ``model`` and ``chat_template_kwargs``. The
  steps hold the conversation.
- ``extra.reef.reward`` is the rollout's score for each trajectory of the
  attempt that was scored: the last attempt of a finished rollout. It is null
  for every other attempt, and for a failed score.
- ``extra.reef.evaluation`` is ``{evaluation_id, task, task_index, repeat,
  attempt, caller, conversation, scored}``. ``conversation`` counts the
  conversations of the attempt from 0. ``scored`` is true only for the
  attempt that was scored.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/trajectories``
answers ``{scenario, evaluation_id, task_index, repeat, trajectories}``: the
trajectories of each attempt of one rollout, in attempt and then conversation
order. Add ``?caller=verifier`` for the verifier's trajectories; another
``caller`` value returns 400. In each attempt, the verifier's trajectories are
the trajectory files that the verifier wrote, in name order, and then the
trajectories that Reef makes from the verifier's calls. A verifier trajectory
has no reward. Each verifier file has ``extra.reef.evaluation`` as
``{evaluation_id, task, task_index, repeat, attempt, caller, file, scored}``,
where ``file`` is the name of its file and ``scored`` is false.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories``
streams the trajectories of the whole evaluation as ``application/x-ndjson``:
one trajectory on each line, in task and repeat order. It sends the agent's
trajectories only, and by default the last attempt of each rollout. Add
``?attempts=all`` to send every attempt; another ``attempts`` value returns
400. The memory of an export does not grow with the evaluation. A file within
the upload limits uploads as it is as trajectory items (`Create trajectory
items <#create-trajectory-items>`__).

Both routes return 404 for an unknown evaluation.

Automatic evaluation
~~~~~~~~~~~~~~~~~~~~

``POST /reef/scenarios/{scenario}/update`` with ``artifact_evaluation`` stores
the settings that Reef evaluates each new artifact of the scenario with. The
settings are the body of an evaluation request without ``artifact_id``:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/scenarios/my-agent/update \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"artifact_evaluation": {"harness": {"name": "terminus-2"}, "rollouts_per_task": 2}}'

Reef checks the settings against the scenario's active artifact, as it checks
a request, but the eval tasks may still be missing. The answer holds the
stored settings, or 400 for settings that the scenario cannot run. Send
``"artifact_evaluation": null`` to stop the automatic evaluations.

Each training step that publishes a checkpoint makes a new artifact. Reef then
stores an evaluation of it in the ``queued`` state; it is ``running`` once a
worker takes one of its rollouts. The evaluations routes list it like any
other. An automatic evaluation that cannot run, for example because the eval
split holds no task, is not stored; the service logs why.

Manual training
---------------

``POST /reef/train`` queues one training request for a scenario in
``data.training_mode: manual``. In harness evolution, every method takes it.
A request is not a record: it waits in the deployment's training request queue
until a training commit consumes it, and the commit keeps it in its
``training_request`` metrics. The request needs no inference receipts and no
score.

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: coding" \
     -H "Content-Type: application/json" \
     -d '{"agent_record_id":"change-001","text":"Add a skill that runs tests before answering", "session":"session-1", "artifact_id":"artifact-1"}'

The body takes these fields:

- ``text`` (required): the user's request, 1 to 4000 characters.
- ``session`` and ``artifact_id`` (required strings): the session and the
  artifact that the request came from. The backend operates on the current
  committed state; it does not restore that artifact.
- ``agent_record_id``: names the request, so that a retry is safe. An
  identical resend returns the queued request without another step, including
  after capacity eviction. The same ID with another body returns 409. Without
  it, each submission gets a new ID.
- ``samples``: the samples that the request's job trains (`Choose the samples
  that train <#choose-the-samples-that-train>`__).
- ``train_from``: the artifact ID that the job starts from. Without it, the
  job starts from the scenario's training start. A value that names no
  artifact of the scenario returns 400.
- ``parameters``: a weight recipe only, the parameters of the training job
  (`Job parameters`_).
- ``requires``: what the change needs where the harness runs (`Requirements
  of a change`_).
- ``client``: a description of the requesting machine (`The requesting
  machine`_).

The response is ``{agent_record_id, scenario, request_type: "train"}``, plus
``page_path``, the request's page with the query that a browser opens it by
(`Request page`_). HTTP 200 acknowledges durable acceptance, not successful
training. The normal bearer authentication applies. Reef returns:

- 400 for empty text, text longer than 4000 characters, missing or non-string
  ``session`` or ``artifact_id``, or a request to an ``auto`` scenario;
- 400 for text that carries a credential-shaped literal or phrasing that
  overrides instructions. The reason names the rule, never the text, and Reef
  stores nothing;
- 400 from a weight recipe for a request without ``samples``;
- 404 for an unknown scenario. The route never creates a scenario, whatever
  ``reef.allow_implicit_scenario_creation`` says.

How a request trains
~~~~~~~~~~~~~~~~~~~~

The two modes differ in what starts a job. ``auto``, the default, also starts
a job on its own: when ``job_samples`` samples wait, or when the deployment's
training manager says so (``training.manager``, see `Configuration
<configuration.rst>`__). ``manual`` starts a job only for a request. A request
trains before the samples that wait, oldest request first, one per step. The
step trains the samples that the request names, whatever ``job_samples``:

- A weight recipe trains the request's ``samples``. Its ``text`` goes into the
  commit's ``training_request`` metrics.
- Harness evolution runs the request with the traces that it names, possibly
  none. The method reads the request next to them.

``train_from`` decides how the job runs beside other jobs:

- A job without ``train_from`` continues the training start and moves it to
  each checkpoint that it saves, so the next such job begins where it ended.
  Reef runs one such job at a time.
- A job with ``train_from`` never moves the training start, and Reef does not
  wait for another job before it submits it. It runs beside the other jobs
  when the training component allows it and a training worker is free
  (``training.max-jobs``). The Tinker component allows it.
- Requests start in the order they came, so a request that waits for the
  running job also holds the requests after it.

A later request does not change a step that already runs. The evaluation and
publication rules still decide whether the result becomes served.

A step that fails with a request, for example on a method error, is not
retried. The next step consumes the request with a committed row whose
``skipped`` reads ``request failed`` and whose ``error`` carries the failure,
and the queue moves on. Send the request again to run it again. That row
consumes the request alone: the samples that were in the step stay held for
the next job. The ``evolution.max_steps`` and ``evolution.max_failure_streak``
budgets count every step, request steps included, but stop automatic steps
only; a request still runs past them.

``GET /reef/status`` reports ``training_mode``. Processors that take requests
also report ``pending_requests``: the accepted requests that no commit
consumed yet.

The metrics of the commit that answered a request include
``training_request``, with the request ID, text, session, artifact ID and
``requires``:

.. code:: json

   {"training_request": {"id": "change-001", "text": "Text me when the run is blocked",
                         "session": "session-1", "artifact_id": "artifact-1",
                         "requires": [{"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID",
                                       "prompt": "The Twilio account SID"}]}}

The method also says what its step did. The commit metrics keep its summary
as ``method_summary``, plain text, and its notes as ``method_notes``, the
JSON object that it returned. Reef clips their long text and redacts
credential-shaped literals in them, and never reads them to decide. Each is
present only when the method returned it non-empty, and each method chooses
its own note keys.

Job parameters
~~~~~~~~~~~~~~

A weight recipe's request can set the parameters of its training job with
``parameters``, a JSON object of at most 16 KiB. Reef merges it into the
recipe's parameters for the kind of the job's samples: nested objects merge
key by key, and every other value of the request wins, lists included. A
request cannot remove a key that the recipe sets. Reef does not read the
parameters: the training component checks them, as it checks the recipe's. A
refused job fails the request with the component's reason. A job with no
objective, from the recipe or the request, is refused.

- The Tinker component takes ``objective``, ``learning_rate_schedule``,
  ``scheduling``, ``batch_size``, ``learning_rate``, ``kl_coef``,
  ``save_interval`` and ``rollout``; ``rollout`` only on a job of tasks.
- Slime takes ``objective``, ``learning_rate_schedule``, ``scheduling``,
  ``save_interval``, ``dataset`` and ``harbor``; ``harbor`` only on a job of
  tasks.

``GET /reef/scenarios/{scenario}/contract`` serves the JSON Schema. For
example, after an SFT job on uploaded trajectories, run an RL job on uploaded
train tasks:

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: coding" \
     -H "Content-Type: application/json" \
     -d '{"agent_record_id":"rl-001","text":"RL on the uploaded tasks","session":"session-1","artifact_id":"artifact-1",
          "samples":["task-a","task-b"],
          "parameters":{"objective":"my_methods.objectives:MyObjective","rollout":{"rollouts_per_task":4}}}'

Requirements of a change
~~~~~~~~~~~~~~~~~~~~~~~~

``requires`` lists what the change needs where the harness runs: at most 8
``{name, kind, check, prompt}`` items, none by default. Reef never runs a
check. The client runs the checks where the harness runs, after the person
read them.

- ``kind`` is one of these:

  - ``permission``: an OS permission that the person grants.
  - ``env``: a variable that the person sets. The extension reads it from the
    environment, and its value never enters a request or the tree.
  - ``service``: an account or endpoint that the person connects.
  - ``binary``: a program that the person installs. Its ``name`` is the
    program, looked for on the PATH.

- ``name`` matches ``^[A-Za-z0-9][A-Za-z0-9._-]*$``. The pages list the item
  under it.
- ``check`` is optional. For ``env`` it is the variable name. For the other
  kinds it is a shell command whose exit status zero means satisfied. For
  ``env``, the variable named (the check, else the name) is a shell
  identifier, ``^[A-Za-z_][A-Za-z0-9_]*$``.
- ``prompt`` is optional: one sentence that tells the person what to enter or
  grant, at most 200 characters. Reef strips it and drops a blank one. The
  pages show it beside the item.

The credential and instruction-override checks run over every ``name``,
``check`` and ``prompt`` as they run over ``text``, with the same 400 and a
reason that names the rule. A malformed list returns 400 and names the first
bad item.

The method may add items of its own in its step's result, for example for an
extension that reads a variable:

- The backend runs the same checks over each added item. It merges the items
  that pass, by name, after the person's items into the commit's
  ``training_request.requires``. So the snapshots row and the manifest carry
  what the person named and what the change added.
- An added item with the name of a person's item is dropped.
- ``training_request.refused_requires`` lists each item that a check refused
  as ``{"item": <the item as written>, "reason": "..."}``. The key is absent
  when nothing was refused.
- A step whose merged list holds more than 8 items is skipped.

The requesting machine
~~~~~~~~~~~~~~~~~~~~~~

An optional ``client`` reports the requesting machine, so that a method builds
for it and not for the sandbox that it evaluates in: ``platform``, ``arch``
and ``release`` (short words) and ``commands``, a map of command names to
whether each is on the machine's PATH (at most 64). A client fills
``commands`` by looking each name up on the PATH; it runs nothing. The field
only informs the method. Reef drops what does not fit that shape and never
refuses the request for it. ``training_request.client`` carries what was kept.

Scenarios
---------

A scenario isolates the records, trainer, and snapshot chain for one workload.
Create it with ``POST /reef/scenarios`` before you send inference requests to
it:

- An inference request never creates a scenario: for an unknown scenario, it
  answers 404.
- The first report or imported record that carries a new ``x-reef-scenario``
  also creates the scenario, and a harness install without
  ``x-reef-scenario`` creates one with a generated name. With
  ``reef.allow_implicit_scenario_creation: false``, these requests return 404
  for an unknown scenario instead.
- Reef creates every scenario with the deployment's configured recipe.
  Requests never select a recipe. The recipe's components, runtime, inference
  backend and optional report schema are fixed with the scenario; a request
  that names a different recipe with the same scenario returns 409.

.. code:: bash

   curl -sS -X POST http://127.0.0.1:8900/reef/scenarios \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "hello-reef"}'

   curl -sS -i http://127.0.0.1:8900/v1/chat/completions \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: hello-reef" \
     -H "Content-Type: application/json" \
     -d '{"model": "m", "messages": [{"role": "user", "content": "fix it"}]}'

A deployment can limit its scenarios (`Limit the scenarios a service holds
<../user-guide/operate.rst#limit-the-scenarios-a-service-holds>`__):

- A request that would create a scenario past ``reef.max_scenarios`` returns
  409.
- A request that must load a scenario when the service already holds
  ``reef.max_loaded_scenarios`` loaded scenarios, none of them idle, returns
  503 with a ``Retry-After`` header.

Create a scenario
~~~~~~~~~~~~~~~~~

``POST /reef/scenarios`` takes the object ``{"name", "owner"?,
"artifact_id"?, "component"?, "serves"?}``. Any other key returns 400, and the
message names the keys that the route takes:

- ``name``: the scenario's name, a non-empty string.
- ``owner``: the owner whose `data items <#data-items>`__ the scenario uses.
  Without it, the owner is the scenario's name. A scenario gets its owner when
  Reef creates it. A later call with another ``owner`` returns 400 for a
  scenario that already uses data items.
- ``artifact_id``: an `artifact item <#artifact-items>`__ of the owner that
  the new scenario uses as its starting artifact. Without it, Reef uses the
  deployment's default start (`Starting artifacts
  <../user-guide/operate.rst#starting-artifacts>`__).
- ``component``: the component that holds the item. Leave it out when the
  scenario has one component. It needs ``artifact_id``.
- ``serves``: a boolean, true by default. With ``false``, the new scenario
  serves nothing: Reef starts no engine for it, and its inference requests
  answer HTTP 404. It serves after an update sets ``served`` (`Serve nothing`_).
  An existing scenario ignores ``serves``.

Reef stores the starting artifacts and registers the scenario with their
component mapping. They are the initial active selections and training starts.
Their artifact rows have ``operation: creation``, and ``source`` names the
selected item or the default configuration.

With ``artifact_id``, Reef reads the item's files and checks them as ``POST
/reef/scenarios/{scenario}/artifacts`` with ``{"item": "<id>"}`` does (`Import
an artifact`_), by reference or as a copy. Each other component uses its
default start. When Reef cannot read the item, or the check refuses it, Reef
registers no scenario. It answers with the status of the refusal, such as 400
for a directory without the component's layout or 404 for an item that the
owner does not store.

The answer is ``{scenario, owner, components}``. ``components`` maps each
component name to its selected ``artifact_id``. A scenario with one component
also returns that value as ``artifact_id``. The status is 201 for a new
scenario. An existing scenario returns 200 with its current selections and
ignores the requested ``artifact_id`` and ``component``.

.. code:: bash

   curl -sS -X POST http://127.0.0.1:8900/reef/scenarios \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"name": "hello-reef-2", "owner": "acme", "artifact_id": "hello-reef/<artifact_id>"}'

Scenario routes
~~~~~~~~~~~~~~~

- ``GET /reef/scenarios`` lists every known scenario, its selections once
  loaded and, for a harness recipe, the ``adapter`` that its tree is rendered
  for, also while the scenario loads.
- ``GET /reef/scenarios/{scenario}/contract`` returns ``{scenario, processor,
  required_request_types, training_mode, status, parameters_schema}``.
  ``training_mode`` and ``status`` are live values. ``parameters_schema`` maps
  a sample kind to the JSON Schema of a job's parameters, for each kind whose
  schema the training component declares.
- ``DELETE /reef/scenarios/{scenario}`` deletes the scenario (`Deleting a
  scenario`_).
- ``GET /reef/scenarios/{scenario}/jobs`` returns ``{scenario, jobs}``: the
  scenario's training jobs in the jobs queue, oldest first. The data service
  answers it, and an unknown scenario returns 404.

A job object has these keys. A time is an ISO 8601 string in UTC, such as
``2026-10-31T08:15:00.250000+00:00``.

- ``id``: the job ID.
- ``state``: ``queued``, ``assigned``, ``running``, ``finished`` or
  ``failed``. The last two are final.
- ``attempt_count``: how many times a training worker took the job, an
  integer from 0.
- ``checkpoints``: the saved checkpoints that the job reported, in step
  order, each ``{step, path, content, metrics, changed_parts}``. ``path`` is a
  directory of the training service. ``content`` is the name under which the
  training service stored the checkpoint's files in the artifact backend;
  Reef removes that content after it commits the checkpoint.
  ``changed_parts`` lists the parts of the scenario component that the
  checkpoint changed from the job's start, ``[]`` for a weight job.
- ``result``: ``{"state": "finished", "metrics": {...}}`` once the job
  finished, else ``null``.
- ``error``: the error of a failed job, or the last retryable error of a
  queued job, else ``null``.
- ``parameters``: the job's parameters, a JSON object: the recipe's
  parameters for the kind of the job's samples, with the request's
  parameters merged in. The training component owns their names, such as
  ``objective`` and ``scheduling``.
- ``sample_count``: the number of samples that the job trains, an integer
  from 0.
- ``sample_kind``: ``trajectory`` or ``task``, the kind of the job's samples.
  It is ``null`` for a job with no sample.
- ``train_from_artifact_id``: the artifact that the job started from.
- ``request``: ``{id, text, parameters}`` of the training request
  (``POST /reef/train``) that the job trains. ``parameters`` is what the
  request gave, ``{}`` when it gave none. It is ``null`` for a job that
  the ``auto`` training mode made.
- ``created_at``: when control added the job.
- ``started_at``: when a training worker first took the job; ``null`` until
  then, and for a job from a store that an older Reef wrote.
- ``ended_at``: when the job finished or failed; ``null`` until then.
- ``experiment``: ``{tracker, project, entity, run_id, url}`` of the
  experiment run that records the job, such as ``{"tracker": "wandb", ...}``.
  Control writes it when its tracker records the first commit of the job.
  ``entity`` and ``url`` are ``null`` when the tracker does not know them.
  It is ``null`` until then, and always ``null`` without an experiment
  tracker. See `Configuration <configuration.rst>`__, "Commit correlation",
  for how W&B gives the run's URL.

For example:

.. code-block:: json

   {"id": "4f2a9c1e", "state": "finished", "attempt_count": 1,
    "checkpoints": [{"step": 1, "path": "/jobs/4f2a9c1e/step-1", "content": "4f2a9c1e:1",
                     "metrics": {"loss": 0.41}, "changed_parts": []}],
    "result": {"state": "finished", "metrics": {"loss": 0.41}}, "error": null,
    "parameters": {"objective": "sft", "learning_rate": 1e-05},
    "sample_count": 8, "sample_kind": "trajectory",
    "request": {"id": "train-1", "text": "Prefer tests first", "parameters": {"learning_rate": 1e-05}},
    "created_at": "2026-10-31T08:15:00.250000+00:00",
    "started_at": "2026-10-31T08:15:01.020000+00:00",
    "ended_at": "2026-10-31T08:21:44.900000+00:00",
    "experiment": {"tracker": "wandb", "project": "reef", "entity": "acme",
                   "run_id": "0d4c3b2a19f8e7d6c5b4a39281706f5e",
                   "url": "https://wandb.ai/acme/reef/runs/0d4c3b2a19f8e7d6c5b4a39281706f5e"}}

Control answers the two cancel routes below. Both return 404 for an unknown
scenario.

``POST /reef/scenarios/{scenario}/jobs/{job_id}/cancel`` cancels one training
job of the scenario. It answers ``{scenario, job}``, with the job as ``GET
/reef/scenarios/{scenario}/jobs`` lists it after the cancel:

- A queued job fails at once, with the error ``cancelled by request``.
- A running job fails the same way, and its training worker cancels the run
  at its next poll.
- A job that ended stays as it is, so a second cancel changes nothing.

The trainer reads the failed job and records its end. A job that the scenario
does not have returns 404.

``POST /reef/scenarios/{scenario}/requests/{request_id}/cancel`` cancels one
training request of the scenario (``POST /reef/train``), so that it trains no
job, or no further. ``request_id`` is the ``agent_record_id`` of the request.
It answers ``{scenario, request_id, queued, job}``:

- A request that waits in the queue fails with the error ``cancelled by
  request``. The trainer consumes it with a commit that has ``skipped:
  "request failed"`` and that error, reads none of its samples, and submits no
  job. ``queued`` is ``true``.
- Each job of the request that has not ended is cancelled, as ``POST
  /reef/scenarios/{scenario}/jobs/{job_id}/cancel`` does. ``job`` is the newest
  job of the request after the cancel, or ``null`` for a request with no job.
- A job that control adds for the request after the cancel, because it was
  building the job's data, is cancelled when it enters the queue.
- A request whose job ended stays as it is, so a second cancel changes nothing.

A request that the scenario does not have returns 404.

Deleting a scenario
~~~~~~~~~~~~~~~~~~~

``DELETE /reef/scenarios/{scenario}`` deletes the scenario. The request
renames the scenario's ref in the artifact repository into
``refs/reef/archived/`` and returns ``{"scenario", "archived"}``, where
``archived`` names the archived ref. Only a local artifact repository can be
archived; a remote one answers 501 and changes nothing. From then on the
scenario does not exist:

- A read of the scenario and a second delete answer 404.
- A create with the ID, and a record that would create the scenario
  implicitly, answer 409. A scenario ID is never used again, so a new
  scenario needs a new ID.
- The inference service answers the scenario 404.
- A local step in flight for the scenario ends without a commit.
- The training service does not run a job of the scenario. When it takes
  such a job, or stores for one, it finds no registration, cancels the job
  and discards what the job stored.

Control then removes the scenario's state in the background, at once:

- It cancels the scenario's training jobs, discards its stored content, and
  moves its record store and commit log under
  ``<agent_record_dir>/archived/``. The training service moves the recipe's
  own directories (a harness scenario's history and step records) under an
  ``archived/`` sibling; this waits while no training service runs.
- A sample selection that runs past the delete writes no sample.
- Each other service drops its instance, its clone and its cached snapshots
  of the scenario within seconds.
- Every snapshot that the scenario published stays reachable under the
  archived ref.
- On a runtime that serves per-scenario adapters, the scenario's adapter stays
  resident in the serving engines until the adapter residency evicts it.

When a step of this cleanup fails, for example because the artifact store does
not answer, the scenario stays deleted. Control tries again an hour later, and
after a restart.

Scenario updates
~~~~~~~~~~~~~~~~

``POST /reef/scenarios/{scenario}/update`` updates an existing scenario. The
body holds one or more of these settings. Reef rejects other fields:

- ``training_mode``: the mode of the data processor, below.
- ``artifact_evaluation``: the settings that Reef evaluates each new artifact
  with (`Automatic evaluation`_).
- ``activation``: the scenario's activation rule, below.
- ``serving_config``: the serving fields that the scenario sets over the
  deployment's values (`Serving config`_).
- ``served``: the artifacts that the scenario serves of one component, each
  with its weight (`Set what a scenario serves`_). ``component`` names that
  component, and is valid only together with ``served``.
- ``record2sample_config``: the record-to-sample config keys that the scenario
  sets over the deployment's (`Record-to-sample config`_).

The response holds the scenario name and each setting that the request
changed. An unknown scenario returns 404 without implicit creation, and an
invalid payload returns 400. Reef stores the settings with the scenario, so a
reload and a restart keep them.

**Training mode.** The values are ``auto`` and ``manual``. ``auto`` starts a
job whenever ``job_samples`` samples wait. ``manual`` starts one only for a
request, on the samples that it names (`Manual training`_).

.. code:: json

   {"training_mode": "manual"}

HTTP 200 returns ``{"scenario": "agents", "training_mode": "manual"}``.

- The change selects later job data. Job data that is already reserved or
  running completes in its original mode.
- After a switch back to ``auto``, automatic jobs train the samples that no
  job trained, including those collected in ``manual``.
- Accepted requests wait for ``manual``, including requests not yet read when
  the mode changes to ``auto``.
- A processor that does not support the requested mode returns 501 without
  changing its state.
- ``GET /reef/status`` reports the selected ``training_mode``.

A Reef process runs at most one scenario that trains full weights, on a single
thread, so preparation, remote execution, and commit never interleave. It may
run any number of scenarios that produce no updates or that update text
artifacts in process. Each one grows and commits on its own background thread,
so record acceptance never waits for artifact evolution.

**Activation.** The ``activation`` setting selects the rule that decides
whether Reef activates each new artifact of this scenario. It replaces the
deployment's `activation section <configuration.rst#the-activation-section>`__
for this scenario. Send one of the built-in rules:

.. code:: json

   {"activation": {"module": "evaluated", "config": {"min_mean_score": 0.8}}}

The ``module`` is one of these rules:

- ``always``: ``config`` takes ``review_kinds``.
- ``never``: there is no ``config``.
- ``evaluated``: ``config`` takes ``min_mean_score``.

These rules apply to the setting:

- The response holds the stored rule. Send ``"activation": null`` to follow
  the deployment's rule again.
- The setting names only built-in rules. Reef answers 400 for a
  ``package.module:ClassName`` rule, because a client must not choose the code
  that the service imports. Set such a rule in the deployment's ``activation``
  section.
- The ``evaluated`` rule activates an artifact only after an evaluation of it,
  so it needs ``artifact_evaluation``. Reef checks the two settings together,
  and stores both or neither. It answers 400 when, after the request, the rule
  that applies is ``evaluated`` and the scenario has no
  ``artifact_evaluation``. So a request can set both at once, or set
  ``artifact_evaluation`` first, and Reef refuses a request that clears
  ``artifact_evaluation`` under ``evaluated``.
- A new rule applies to the artifacts that training makes after the change. It
  never activates an existing artifact. Under ``evaluated``, Reef activates an
  artifact when an evaluation of it finishes after the change with a
  ``mean_score`` of at least ``min_mean_score``.

Serving config
~~~~~~~~~~~~~~

The ``serving_config`` setting changes how the scenario is served while it
serves, without a restart or an activation. It maps a component name to the
fields that the scenario sets for that component:

.. code:: json

   {"serving_config": {"weights": {"temperature": 0.6, "top_p": 0.95}}}

Reef merges the request into the stored config field by field. A value sets
the field. ``null`` removes the field, so the field takes the deployment's
value again. Fields that the request leaves out keep their stored values. The
response holds the stored ``serving_config`` and the
``effective_serving_config``, as `Read the settings`_ describes.

Only the component that the inference runtime serves has serving fields. The
runtime and its request handler define the fields and check their values:

- The token-native chat handlers have ``temperature``, ``top_p`` and ``top_k``
  (`native backend options <configuration.rst#native-backend-options>`__). A
  runtime can add engine fields, such as a number of replicas.
- Reef answers 400 and stores nothing for a field that they do not have, for
  a value that a field does not take, or for another component.
- A field that takes effect only when an engine starts, such as a GPU type,
  stays in the deployment configuration.

Inference applies a change within a few seconds. Each inference replica sends
the request fields with each later request. The runtime applies its engine
fields to the engine that serves the scenario and to each engine that starts
for it later. A stored field that the deployment no longer accepts is ignored
with a warning, and the scenario serves with the deployment's values until a
person changes the config.

Record-to-sample config
~~~~~~~~~~~~~~~~~~~~~~~

The ``record2sample_config`` setting gives one scenario its own config for the
record-to-sample stage of each trainer. It maps a component name to config
keys. Each key replaces the same key of the deployment's
`pipeline.record2sample.config <configuration.rst#the-pipeline-section>`__ for
the trainer of that component. The stage class stays the deployment's:

.. code:: json

   {"record2sample_config": {"weights": {"accept_multi_turn_policy_samples": true}}}

The request replaces the stored mapping as a whole. Send ``{}`` to use the
deployment's config again. The response holds the stored
``record2sample_config``.

Reef applies a change by loading the scenario again, as a restart does. Each
trainer then builds its stage with the new config and reads the records after
its committed progress again. A samples service in another process loads the
scenario again before its next selection. Reef answers 400, and keeps the
stored setting and the loaded scenario, when the mapping names a component
that has no trainer, or when a stage refuses its config.

Traffic split
~~~~~~~~~~~~~

When the ``served`` setting gives more than one artifact a weight above 0, the
weights divide the scenario's model requests between those artifacts (`Set
what a scenario serves`_). For example, this mapping sends 5% of the requests
to a new artifact:

.. code:: json

   {"served": {"artifact:a": 95, "artifact:b": 5}}

Reef starts each artifact of the mapping that does not serve yet. Until an
artifact has started, the active artifact answers its requests. Without a
ready active route, as at the first startup or when the scenario serves again
after serving nothing, they get HTTP 503 until a route is ready. Inference
replicas use new weights within a few seconds, without a restart. A mapping
with one weight above 0 ends the split, so the active artifact answers again.

The activation rule can make a newer artifact active during a split. The split
keeps its weights. The new active artifact then answers the requests of each
artifact of the split that the scenario no longer serves, and the other
artifacts keep their shares.

The weights apply to a model request that names ``active``, or a model that the
scenario does not serve under that name:

- **One conversation stays on one artifact.** Send the
  ``x-reef-session-key`` header with an identifier of the conversation. Reef then
  computes a bucket from the scenario and that identifier.
- **Without the header, Reef reads the first messages.** It computes the bucket
  from the scenario and the text of the system message, the first message and the
  first user message. Later turns repeat these messages. A client that shortens
  its history can move to another artifact. Conversations that start with the same
  messages get the same artifact. So fixed prompts can change the observed
  percentages.
- **A changed weight moves few conversations.** A change from 95/5 to 80/20 moves
  only the conversations of the 15 buckets that go to another artifact.
- **Some requests do not use the weights.** These requests keep their usual
  target:

  - a request without a message;
  - a request with ``previous_response_id`` and no ``x-reef-session-key``, which
    goes to the active artifact;
  - a request that names an artifact or an artifact's route;
  - a rollout call or an evaluation call.

- **Reef does not move a failed request.** A request to a failed artifact fails.
  To stop new requests to an artifact, set its weight to 0.
- **A retry keeps its artifact.** Reef does not compute the bucket again when
  it retries a request (see `Request`_). When the artifact has no route at
  the retry, Reef answers HTTP 503 with ``Retry-After``.

The weights select the artifact. The engine of the artifact selects its own
replica.

The record names the artifact that answered in ``artifact_id``, and holds the
bucket and the weights in ``metadata.traffic``. While the scenario serves an
artifact, the samples service does not mark its records' samples stale, whatever its weight
(`Call one served artifact`_). A job can then hold samples of several
artifacts. Each sample keeps its own ``artifact_id`` and recorded log
probabilities.

Read the settings
~~~~~~~~~~~~~~~~~

``GET /reef/scenarios/{scenario}/settings`` returns the settings of an existing
scenario:

.. code:: json

   {
     "scenario": "agents",
     "training_mode": "auto",
     "artifact_evaluation": {"harness": {"name": "terminus-2"}, "rollouts_per_task": 2},
     "activation": {"module": "evaluated", "config": {"min_mean_score": 0.8}},
     "effective_activation": {"module": "evaluated", "config": {"min_mean_score": 0.8}},
     "serving_config": {"weights": {"temperature": 0.6}},
     "effective_serving_config": {"weights": {"temperature": 0.6, "top_p": 0.95}},
     "served": {"weights": {"artifact:a": 95, "artifact:b": 5}},
     "record2sample_config": {"weights": {"accept_multi_turn_policy_samples": true}},
     "job_parameters": {"trajectory": {"objective": "sft", "save_interval": 50}}
   }

- ``training_mode`` is what the scenario runs.
- ``artifact_evaluation`` and ``activation`` are the stored settings, or null.
- ``effective_activation`` is the rule that applies: the scenario's
  ``activation``, else the deployment's. When the deployment names a rule
  class of its own, ``module`` is the ``package.module:ClassName`` of that class.
- ``serving_config`` is the stored `serving config <#serving-config>`__, ``{}``
  when the scenario sets no field.
- ``served`` maps each component to the artifacts that the scenario serves of
  it, each with its weight (`Set what a scenario serves`_). Without a traffic
  split, the active artifact has 100 and each other served artifact has 0. With
  a split, each artifact has its weight. The active artifact also has the
  weight of each artifact of the split that the scenario no longer serves,
  because it answers their requests.
- ``effective_serving_config`` maps the served component to each of its serving
  fields that has a value: the scenario's value, else the deployment's.
- ``record2sample_config`` is the stored `record-to-sample config
  <#record-to-sample-config>`__, ``{}`` when the scenario sets no key.
- ``job_parameters`` maps a sample kind (``trajectory`` or ``task``) to the
  recipe's parameters of a job of that kind. A training request's
  ``parameters`` merge into them. A kind that the recipe gives no parameters is
  absent.

An unknown scenario returns ``404``.

Inference
---------

Startup status
~~~~~~~~~~~~~~

Scenario creation and activation select an artifact without waiting for its
engine to start. The inference service loads the artifact and publishes its
route after startup succeeds. Read its state with:

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/scenarios/hello-reef/inference/status \
     -H "Authorization: Bearer $REEF_TOKEN"

.. code:: json

   {"artifact_id": "artifact-1", "state": "starting", "error": null, "engine": null, "engine_detail": null,
    "startup": {"steps": [{"phase": "prepare", "started_at": 1794150000.1, "ended_at": 1794150480.6,
                           "detail": "Quantizing to NVFP4"},
                          {"phase": "start", "started_at": 1794150480.6, "ended_at": null,
                           "detail": "Waiting for a GPU"}],
                "estimate_seconds": null}}

The query parameter ``model`` defaults to ``active``. It can also name an
active artifact alias, or an artifact that the scenario serves beside the
active one (``?model=<artifact-id>``). An unknown name returns 404. One
startup covers the active artifact and each artifact beside it, but an
artifact with a published route stays ``ready`` while another artifact starts
or fails.

``state`` is one of:

* ``pending``: no worker has taken the startup work.
* ``starting``: inference has taken it. Downloads, conversion, loading, and
  retries are part of this state.
* ``ready``: startup succeeded and inference published the route.
* ``failed``: startup ended with an error. ``error`` gives a safe message:
  the message of a runtime's ``StartupFailedError``, such as an engine
  argument that the model refuses, else a fixed text. The service logs hold
  the diagnostic details. ``error`` is null in every other state.

Control retries a failed startup by itself: 60 seconds after the failure, and
then after a wait that doubles at each failed retry, up to one hour. A retry
returns the state to ``pending``. Resend the complete accepted ``served``
mapping to retry at once.

A newer request, such as one for a changed serving config, replaces the
current one at once. The work of the replaced request stops within seconds
when the runtime reads ``reef.inference.startup.stopping()``. Requests of
different scenarios start one after another, or at the same time when the
runtime has more than one ``serving_slots``.

``engine`` is the state of the engine behind the route of a ``ready``
artifact, when the deployment's inference handler reports it:

* ``warm``: an engine answers now.
* ``starting``: an engine runs, but it does not answer yet.
* ``idle``: no engine runs. The next request starts one and waits for it.

``engine`` is null in every other state, and when the handler cannot tell.
While ``engine`` is ``starting``, ``engine_detail`` can say what the engine
does now, as one short line, such as ``Loading weights · 0:42``. It is null
otherwise.

``startup`` gives the artifact's startup in three fixed phases, so that a
client shows one step bar for every backend:

* ``prepare``: the serving work waits, deploys the engine, and converts or
  quantizes the weights.
* ``start``: a machine or container is found, and the weights reach it.
* ``load``: the weights go into GPU memory, and kernels, CUDA graphs and
  warm-up run.

It is ``{steps, estimate_seconds}``, or null:

- Each step is ``{phase, started_at, ended_at, detail}``, in the order of the
  phases, with times in epoch seconds. Only the last step can have
  ``ended_at`` null: it runs now. ``detail`` is a short note of what it does,
  or null. A phase that the backend does not report has no step.
- ``estimate_seconds`` is a guess of the seconds until the engine is warm, or
  null.
- While the state is ``pending`` or ``starting``, the steps are ``prepare``
  from the startup request, then the phases that the runtime reported for
  this artifact.
- While a ready artifact's engine is ``starting``, the steps are the phases
  that the inference handler reports.
- ``startup`` is null otherwise.

The status describes the selected artifact, even while an older route still
serves requests. It is a startup result, not a continuous health check.
Reading it starts no work and no engine. The route requires the same bearer
token as inference. No startup work returns 404. An unreadable state store
returns 503.

Without a usable route, an inference request for a pending or starting
artifact returns HTTP 503 with error type ``inference_starting`` and
``Retry-After: 30``. A failed startup returns HTTP 503 with type
``inference_start_failed``. Both errors use ``{"error": {"type",
"message"}}``. Control and data operations remain available during startup.

``POST /reef/scenarios/{scenario}/redeploy`` deploys the engines of what the
scenario serves again, also when they run its current settings, for example
after a hang. A new request with the same selection replaces the current one,
and the runtime reads ``reef.inference.startup.redeploying()``. It answers
``{scenario}``, also for a scenario that serves nothing.

Request
~~~~~~~

Send the same body that you would send to the provider. Reef never touches
your sampling parameters, and no artifact adds to the request. Set
``"stream": true`` and read the SSE response for streaming.

Reef answers the request from the scenario's row in ``served_models``:

- The request's ``model`` selects the row of a served artifact when it names
  one. Any other ``model`` selects the active row, or the row that a `traffic
  split <#traffic-split>`__ gives the request.
- Reef keeps that row for the whole request. It sends the request to the
  engine at the row's URL under the row's model name, and records the
  snapshot of the row's artifact, even if an activation completes during the
  request.
- To require a specific artifact, use
  ``/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/chat/completions``.
- ``/reef/scenarios/{scenario}/v1/{route}`` is the same call as
  ``/v1/{route}``, with the scenario in the path instead of the header. A
  client install's model binding calls it.
- On a weight-serving deployment, the engine handler selects the served
  adapter itself, and it replaces a ``lora_path`` that the body names.

`Run the services in separate processes
<../user-guide/split-services.rst>`__ gives the
status codes of an inference request.

Reef retries a request once when the engine aborts a non-streaming answer, or
refuses the row's model name before any output. It reads the row again for the
retry. Three kinds of request keep their artifact: a request to an artifact's
route, a request whose ``model`` names its artifact, and a request of a
traffic split. When no row serves that artifact at the retry, Reef answers
HTTP 503 with ``Retry-After``. The retry of any other request can reach a new
active artifact.

Log probabilities depend on what answers the request:

- **A provider or routed target.** Reef forwards ``logprobs`` and
  ``top_logprobs`` in the body. It preserves the provider's returned log
  probabilities in the response and the record.
- **Reef's SGLang or vLLM backend.** On ``/v1/chat/completions`` with
  ``"logprobs": true``, ``choices[0].logprobs.content`` has one entry for each
  sampled token, in output order. A stream sends the entries with their text,
  and the record holds the same entries. Each entry's ``top_logprobs`` holds
  only the sampled token. The SGLang backend returns the engine's alternatives
  in SGLang's own format, in ``choices[0].meta_info.output_top_logprobs``; a
  stream sends them in its last chunk. The vLLM backend returns no
  alternatives.
- **Reef's SGLang or vLLM backend on ``/v1/messages``.** Anthropic Messages
  has no log probabilities. Reef extends it: a buffered request with
  ``"logprobs": true`` gets the same entries in a top-level ``logprobs``
  (``{"content": [...]}``) of the message. A streamed message has none.

Token IDs follow the same rule. On ``/v1/chat/completions``, a request to
Reef's SGLang or vLLM backend with ``"return_token_ids": true`` gets the
rendered prompt's IDs in ``prompt_token_ids`` and the sampled IDs in
``choices[0].token_ids``, as vLLM's extension returns them. A stream sends both
in its last chunk. On ``/v1/messages``, a buffered message gets both at its top
level, as ``prompt_token_ids`` and ``token_ids``. The record holds the same
fields. A request without the flag gets no token IDs. When Reef stores a record as ATIF, the reply's step keeps
the token IDs and the ``logprob`` of each entry in its ``metrics``
(``prompt_token_ids``, ``completion_token_ids`` and ``logprobs``).

The receipt identifies the stored record:

+---------------+-----------------------------------------------------------+
| Response kind | Where the receipt is                                      |
+===============+===========================================================+
| non-streaming | the ``x-reef-agent-record-id`` response header            |
+---------------+-----------------------------------------------------------+
| OpenAI SSE    | ``reef.agent_record_id`` in a final empty-``choices``     |
|               | chunk, immediately before ``data: [DONE]``                |
+---------------+-----------------------------------------------------------+
| Anthropic SSE | the same field on ``message_stop``                        |
+---------------+-----------------------------------------------------------+

Streams carry it only after the record is stored, so an incomplete stream
carries no receipt. A stream that fails after its response headers also
carries no terminal event: Reef closes the connection, so a client SDK
reports a transport error instead of a complete answer. These failures end
a stream in this way:

- an engine or transport error;
- a missing terminal event;
- a handler's report that the engine does not serve the selected weights;
- the inference timeout.

A stream from Reef's SGLang or vLLM backend that the engine aborts keeps its
terminal event. It is still incomplete, because Reef cannot restart a stream
after it sent frames.

Reef stores each incomplete stream as an incomplete delivery record and
preserves the provider response. A harness that reports by receipt does not
reference these records. Reef excludes them from sample creation and
training, also when a report references their IDs directly or retained
records are replayed after a restart. A routed stream follows `Routed
external-model requests`_ instead.

An inference response from a served Reef artifact carries
``x-reef-artifact-id``, the ID of the artifact that answered. Routed external
models have no such header. A client learns of a new harness artifact from the
harness routes (`Harness artifacts`_).

The evaluation routes, ``/reef/scenarios/{scenario}/evaluation/v1/{route}``
and ``.../components/{component}/evaluation/v1/{route}`` for a candidate of
one component of a composed snapshot, answer from the scenario's active model
and keep no record. The answer has no ``training`` block. An unknown scenario,
or one that serves no model, returns 404. A composite recipe's evaluation
rollouts call these routes; a recipe of one component calls its runtime's
endpoint directly. When auth is
configured, these calls present the token that the service issues the recipe
at startup. It opens the evaluation routes alone, so evaluation runs, which
run candidate code, can sample the active selection but reach no other route.
A restart issues a new token.

Artifact of a recorded response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An inference record holds the provider exchange and ``artifact_id``, the
artifact that answered: the same ID as ``x-reef-artifact-id``. The samples
service's staleness check reads it (see `Staleness
<../developer-guide/recipe-samples-service.rst#staleness>`__). Serving returns no training
tensors. Reef removes any ``training`` block that an engine handler returns
from the record, from the client's answer, and from rollout and evaluation
answers. The training component renders each recorded call and computes the
log probabilities that it needs.

Report
------

``POST /reef/report`` stores four fields and drops other top-level keys. Put
harness-specific data inside ``metadata`` or ``feedback``.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Notes
   * - ``score``
     - number
     - no
     - Must be finite; booleans are rejected.
   * - ``feedback``
     - string or object
     - no
     - Opaque feedback interpreted by the recipe.
   * - ``references``
     - list of strings
     - no
     - Unique existing inference receipts in this scenario: the last call of
       each trajectory that the report scores, usually one. Reef expands each
       reference to the calls of its trajectory. Reported-feedback processors
       require at least one reference.
   * - ``metadata``
     - object
     - no
     - Recipe metadata; ``training.eligible`` is not supported. ``metadata.task``
       (``name``, with ``path`` and ``digest`` when known) or
       ``metadata.harbor.task_name`` names the Harbor task the rollout ran;
       the assembled training sample carries it as ``task``, so a recipe can
       group samples by task.

It answers ``{agent_record_id, scenario, request_type}``.

.. code:: python

   client.report("hello-reef", {
       "agent_record_id": "myharness:run42:trial7",
       "score": 1.0,
       "references": ["abc123"],
       "metadata": {"harbor": {"trial_id": "run42:7"}},
   })

The optional top-level ``agent_record_id`` makes posting retry-safe. Reef uses
it as the report's own record ID: an identical resend returns the stored
record and does not process it again, and the same ID with different content
returns 409. It is not a receipt. Receipts go in ``references``.

A recipe may declare a report schema, as `tttd
<../user-guide/recipes/tttd.rst#the-report-contract>`__ does. Reef then
validates the declared ``score`` and ``metadata`` fields at ingress and
answers 400 on a violation. ``feedback`` and undeclared ``metadata`` keys pass
through. The removed ``metadata.training.eligible`` field returns 400 whatever
its value: reports cannot opt out of training.

Every reference must name a stored inference record of the same scenario, and
no reference may lie inside the trajectory of another: a report references
only the last call of each trajectory. Missing, foreign, non-inference,
duplicate, or inner references return 400, and Reef stores no report. Reef
does not wait for future records; the producer may resubmit after the
referenced inference is stored. An identical retry of an accepted report
keeps its original behavior after reference cleanup; it does not restart
training.

Record
------

Reef stores every inference exchange and every report as an ``AgentRecord``
with a record ID. The same ID with different content returns a conflict; it
never overwrites. Reef tracks consumed records, so retried reports, and late
reports whose references already trained, are not counted twice.

Records remain readable after training consumes them (`Record and commit
history`_). Retention deletes the oldest record items only when the shared
capacity budget is exceeded (20 GiB by default), and keeps durable loss
totals. See `Configuration <configuration.rst>`__ and `Python API
<python-api.rst>`__.

Receiving an update
-------------------

A weight-training scenario needs no client action: keep calling the same
inference endpoint, and it serves the active selection.

A scenario whose artifact is the harness tree works the other way: the client
pulls. It fetches the tree of the active selection with ``GET /reef/harness``,
writes the files where its agent reads them, binds the agent's model to this
Reef, and runs the agent on that snapshot. It then reports against the
receipts of that snapshot. ``reef-client harness`` is that client (see `Use a
published tree
<../user-guide/evolve-your-harness.rst#use-a-published-tree>`__).

Harness artifacts
~~~~~~~~~~~~~~~~~

+--------------------------------+---------------------------------------------------------------+
| Route                          | Response                                                      |
+================================+===============================================================+
| ``GET /reef/harness``          | ``{artifact_id, parent_artifact_id, files, evaluation,        |
|                                | requires}``, plus ``components`` (component name to           |
|                                | artifact ID) for several components;                          |
|                                | ``x-reef-artifact-id`` identifies the tree                    |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/artifacts``| ``{scenario, artifacts}``, oldest first: the creation and     |
|                                | every training step of the harness component, also a step     |
|                                | that published nothing (``checkpoint: false``). Each row has  |
|                                | ``artifact_id``, ``component``, ``active`` (true on exactly   |
|                                | one row, the active artifact) and ``activated`` (true when    |
|                                | the row's artifact has been active at some time); each        |
|                                | training row carries the evaluation metrics of its step and   |
|                                | ``page_path``, its step's page with the query a browser opens |
|                                | it by                                                         |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/adapters`` | ``{adapters}`` — every harness adapter this process resolves, |
|                                | each with ``name``, ``binary``, ``trajectory_format``,        |
|                                | ``model_bindings``, the pinned ``install`` spec, ``roots``    |
|                                | (each render root to its ``variable`` and ``default``         |
|                                | directory on a person's machine) and ``files`` (the render    |
|                                | paths of the harness's files, ``{name}`` for one path         |
|                                | segment); both are empty for an adapter that does not sync    |
+--------------------------------+---------------------------------------------------------------+
| ``POST /reef/harness/import``  | ``{artifact_id, parent_artifact_id, changed, merged, files,   |
|                                | ignored}``, with ``x-reef-artifact-id``; see                  |
|                                | `Importing a harness`_                                        |
+--------------------------------+---------------------------------------------------------------+

The manifest and catalog reads take ``x-reef-scenario`` and never create a
scenario. An unknown scenario or one without a harness tree answers HTTP 404.

``?form=`` chooses what ``files`` holds, path to text. Another value returns
400:

- ``rendered`` (the default): the snapshot rendered for the scenario's
  adapter. These are the paths that the agent reads, with the adapter's
  enforced config defaults merged in (see `Rendering
  <../developer-guide/harness-adapters.rst#rendering>`__).
- ``source``: the source tree that the snapshot stores. These are the tree
  paths that the adapter's parts take, the ones that a patch names (see `Tree
  parts <../developer-guide/harness-adapters.rst#tree-parts>`__). A client
  that renders the tree itself reads this form.
- ``client``: the snapshot as a layer over a person's own configuration, for
  an install at ``root``, bound to
  ``<service_url>/reef/scenarios/<scenario>``. Both query parameters are
  required: ``service_url`` is the address at which the client reaches Reef,
  and ``root`` is the directory in which the client installs the files. An
  adapter without a ``layer`` returns 400. The binding asks for the served
  model, and its key is the literal ``REEF_TOKEN_PLACEHOLDER``, which the
  client replaces with its own token. This form adds:

  - ``run``: the command that runs the agent on the files (``binary``,
    ``args`` and ``env``; ``env`` may hold the placeholder too);
  - ``register`` and ``unregister``: the agent's own commands that record the
    install and remove it, each a list of argument lists, empty for most
    agents;
  - ``install``: the agent's pinned install spec;
  - ``synced``: the render templates of the files that the client sends back,
    with ``{name}`` for one path segment.

In the ``rendered`` and ``source`` forms, a config file may hold ``{root}``:
write the directory that holds the render roots there. These two forms carry
no endpoint and no credential: the client writes its own model binding where
its agent reads it.

A new scenario's starting artifact holds ``evolution.seed``, so a client can
pull before any step publishes. A recipe whose seed is empty returns 404 until
a step publishes a tree. Use ``?artifact_id=`` to request a specific harness
artifact, in any form. An unknown or unrestorable artifact returns 404.

``requires`` describes what the harness needs where it runs. It merges the
``training_request.requires`` items from the artifact's training history by
name, with the newest definition winning. A later request that adds no items
does not remove earlier requirements. One request's cap of 8 does not bound
the combined list. Reef runs no check: the client checks each item before it
runs the artifact. `Requirements of a change`_ gives the shape.

Catalog and manifest reads do not wait for an evolve step's method or
evaluation runs: they serve the existing artifacts while a candidate is
prepared. A manifest's artifact and evaluation metrics always come from the
same artifact, because reads do not interleave with a change of the active
selection.

Importing a harness
~~~~~~~~~~~~~~~~~~~

``POST /reef/harness/import`` commits the harness files of a person's agent
as a sync, with ``x-reef-scenario``. A sync is a commit operation of its own:
the merged tree becomes a new artifact, and the operation selects it for
serving and as the harness trainer's start.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: code-repair" -H "Content-Type: application/json" \
     -d '{"files": {"plugin/rules.md": "Run the tests first.\n"},
          "base_artifact_id": "artifact-12"}' \
     "$REEF_URL/reef/harness/import"

- ``files`` maps each file's render path (``roots`` and ``files`` of
  ``GET /reef/harness/adapters``) to its text: every harness file of the
  install, changed or not. Reef maps each file back to the tree path of the
  part that renders there. A path that no part renders to is ignored and
  listed in ``ignored``. A file of the base that ``files`` lacks is one the
  person removed. Config files are not read: the model binding owns them.
- ``base_artifact_id`` is required: the artifact the client installed.
- The head here is the active selection, not the training start: a sync
  serves its result at once, so it never carries a kept tree that the
  activation rule left inactive.
- Reef puts the client's changes since the base on the head, file by file. A
  file that only the client changed takes the client's text; a file that
  only the head changed keeps the head's. A file that both changed merges as
  its part's ``merge`` says: ``lines`` keeps each side's changed lines, and
  ``file`` keeps neither. What cannot merge is HTTP 409, which names the
  file at its render path. An empty file counts as no file. The answer's
  ``merged`` is true when the head had moved on from the base.
- The merged tree must pass the adapter's validation, as a method's
  candidate does; a problem is HTTP 400 naming it. Reef does not evaluate
  the tree.
- A sync that changes nothing answers ``changed: false`` and commits
  nothing. Otherwise the commit's operation is ``sync``, and its metrics
  carry ``sync`` (with ``base_artifact_id``), ``patch`` and
  ``patch_parts``. The snapshot serves at once, without the activation rule,
  and the next step trains from it.
- A sync while a step holds job data is HTTP 409; send it again when the
  step ends. So is a sync that another commit overtook.

Harness requests
~~~~~~~~~~~~~~~~

A client submits a person's request for a harness change through ``POST
/reef/train`` (`Manual training <#manual-training>`__), with ``text``,
``session`` and the ``artifact_id`` that it runs. The scenario must be in
``manual`` mode; a scenario in ``auto`` refuses the request, and the client
shows that error and leaves the mode switch to the person. A request needs no
inference receipts, so the receipts that a client captured stay free for
feedback.

The trainer runs the method's step on the request, then evaluates and
publishes under the same policy as automatic evolution. The snapshot catalog
shows the commit metrics ``training_request``, ``method_summary`` and
``method_notes``.

Artifacts
~~~~~~~~~

An artifact is one immutable output of a component, such as checkpoint
weights or a harness tree. Training, import and harness sync create
artifacts. Scenario creation makes each component's starting artifact from
the default configuration or a selected artifact item.

``artifact_id`` is the only public artifact identity. It is an opaque string:
callers must not derive a repository revision or component name from it.
An unchanged component keeps its artifact ID through another component's
update. Activation and serving existing artifacts append operations and
change scenario selections; they create no artifacts or repository revisions.

``GET /reef/scenarios/{scenario}/artifacts`` answers ``{scenario,
artifacts}``, **newest first**. Each row is one artifact of one component:

+-----------------------+------------------------------------------------------+
| Field                 | Meaning                                              |
+=======================+======================================================+
| ``artifact_id``       | the artifact                                         |
+-----------------------+------------------------------------------------------+
| ``component``         | the component whose content it is                    |
+-----------------------+------------------------------------------------------+
| ``parent_artifact_id``| the artifact this one derives from, or null          |
+-----------------------+------------------------------------------------------+
| ``operation``         | ``creation`` for the creation artifact,              |
|                       | ``training``, ``import``, ``sync``, or ``recovery``  |
|                       | for the artifact that a store without a durable      |
|                       | commit log served at a restart                       |
+-----------------------+------------------------------------------------------+
| ``recorded_at``       | the commit time, ``null`` for the creation artifact  |
+-----------------------+------------------------------------------------------+
| ``active``            | true for the active artifact of each component       |
+-----------------------+------------------------------------------------------+
| ``activated``         | true when the artifact has been active at some time  |
+-----------------------+------------------------------------------------------+
| ``served``            | true when the scenario serves the artifact beside    |
|                       | the active artifact (`Set what a scenario serves`_)  |
+-----------------------+------------------------------------------------------+
| ``metrics``,          | training rows only: the step's metrics and the       |
| ``record_progress``   | record high-water mark that the step consumed        |
+-----------------------+------------------------------------------------------+
| ``source``            | import rows: ``{uri, version}`` that the import      |
|                       | read. The creation row: ``{kind: "default",          |
|                       | inputs}`` or ``{kind: "item", item, uri, version}``  |
+-----------------------+------------------------------------------------------+

``parent_artifact_id`` describes derivation within the scenario. A training
output follows that component's training start; a harness sync follows its
previously active artifact. Creation and ordinary imports have no local
parent, so the field is null. The import source remains in ``source``,
including ``source.item`` for an artifact item; an import from another
scenario creates no parent link across scenarios.

The list holds artifacts only. A training step that made no artifact, such
as a rejected or skipped step, a job submission or a job's end, has no row;
``GET /reef/scenarios/{scenario}/operations`` holds its record (see `Record and
commit history`_). Activations, train-from operations and serves have no rows
either.

``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files`` answers
``{scenario, artifact_id, files}``: each file of the artifact as ``{path,
bytes, url, url_expires_at}``, in path order. The data service answers this
route.

- When the repository backend signs URLs of its files, ``url`` opens the file
  in the backend's storage, without a Reef token, until ``url_expires_at``
  (Unix seconds). Reef then lists the files without copying the artifact to
  its own disk. ``?expires_in=`` sets the life of these URLs in seconds: 1 to
  86400, 3600 by default. Download a large artifact through these URLs, so
  that its bytes do not pass through Reef.
- With a backend that signs no URLs, such as ``git_lfs``, both fields are
  null, and the data service serves the files itself.

With ``?path=<path>``, the route downloads that file. In a scenario of several
components, name the component with ``?component=``. An artifact that the
scenario does not list, or a path that is not one of its files, returns 404.
``?signed=1`` returns a `signed download URL <#signed-download-urls>`__ of the
list or the file.

Serving settings
~~~~~~~~~~~~~~~~

An artifact holds only its content. It does not say which model serves it or
how. A deployment that serves artifacts of several base models gives each
scenario a `serving config <#serving-config>`__ field that chooses the model
of each artifact. The deployment's inference runtime defines that field and
checks it. An artifact without a setting is served with the deployment's
model.

You can change a setting at any time. The runtime applies it when inference
next serves the scenario. An artifact that runs keeps its engine.

An artifact that training makes names its training starts in its own metadata,
``training_starts``: the artifact ID of its training start, then that start's
own training starts. A runtime uses them, so that a training artifact is served
as its start is, unless it has its own setting.

Import an artifact
~~~~~~~~~~~~~~~~~~

``POST /reef/scenarios/{scenario}/artifacts?component=<name>`` makes a new
artifact from the directory that a URI names. Leave out ``component`` when
the scenario has one component. The body is ``{"source": "<uri>"}``, or
``{"item": "<id>"}`` for an `artifact item <#artifact-items>`__ of the
scenario's owner, whose ``source_uri`` Reef reads. Reef checks the layout
below on the source, and then copies the directory from one of these URIs:

- ``hf://org/model@revision``: a Hugging Face snapshot. When ``revision`` is a
  full 40-digit commit SHA and the component holds model weights, Reef keeps a
  reference to the revision instead of a copy (see below).
- ``s3://bucket/prefix/``: every object under the prefix. Empty folder
  markers, keys that end in ``/``, are skipped.
- ``file:///absolute/path``: a directory on the service's file system,
  percent encoded as ``Path.as_uri()`` writes it. Its files are copied.
- A scheme that ``storage.artifact-sources`` adds.

To import a directory from your own machine, put it in one of these places
first, for example an S3 prefix.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"source": "s3://my-bucket/adapters/run-7/"}' \
     "$REEF_URL/reef/scenarios/code-repair/artifacts"

The answer is HTTP 201 with ``{scenario, component, artifact_id}``. It comes
after the artifact is durable, so give a large import a long client timeout.
Each import creates a new artifact, even when another artifact contains the
same files. Storage can reuse equal content without changing either ID.

Reef stores the new artifact and appends an ``import`` operation. Serving
selections and training starts stay unchanged, and the activation rule does
not run. Serve the artifact or train from it as from any other artifact.

**Hugging Face references.** An import of ``hf://org/model@<commit>`` with a
full commit SHA into a component of model weights stores only a reference,
``hf-reference.json``. Any other component, such as a harness, stores a copy,
because its readers read the files themselves. Reef checks the layout from the
revision's file list and its small files, and takes each weight file's SHA-256
from the Hub, so it downloads no weights. The repo must stay on the Hub while
Reef serves or trains the artifact. A private repo needs a token that the
service reads in Hugging Face's standard way, such as ``HF_TOKEN``. A revision
that does not resolve for that token returns 400.

**Layout.** The directory must have the layout of the snapshots that the
scenario's component makes. The training backend and its configuration decide
it. A component without a layout, such as the records of a trainer that serves
nothing, takes no import.

+-----------------------------+------------------------------------------------+------------------+
| Component                   | The directory holds                            | Other files      |
+=============================+================================================+==================+
| Full weights (Slime)        | ``config.json`` and ``*.safetensors`` or       | allowed          |
|                             | ``*.bin`` at the root                          |                  |
+-----------------------------+------------------------------------------------+------------------+
| LoRA adapter (Slime)        | ``adapter/adapter_config.json`` and            | allowed          |
|                             | ``adapter/adapter_model.safetensors`` or       |                  |
|                             | ``.bin``; ``r`` is at most the configured      |                  |
|                             | rank, and ``target_modules`` is a list within  |                  |
|                             | the configured modules                         |                  |
+-----------------------------+------------------------------------------------+------------------+
| Tinker                      | ``tinker-checkpoint.json`` of the configured   | allowed          |
|                             | model, rank and service; its ``state_path``    |                  |
|                             | resolves on that service as a LoRA of that     |                  |
|                             | model and rank                                 |                  |
+-----------------------------+------------------------------------------------+------------------+
| Harness                     | a source tree whose files are UTF-8 text and   | refused: a path  |
|                             | pass validation for the adapter, the files a   | off the harness  |
|                             | harness artifact stores                        | layout fails     |
|                             |                                                | validation       |
+-----------------------------+------------------------------------------------+------------------+

**Base roots.** A weight component of the Slime or the Tinker backend also
takes a base root: a directory that holds ``base.json`` alone. The file names
a deployment's default start by its inputs, for example
``{"format": "reef-base-root-1", "inputs": {"model": "/models/qwen", "lora_rank": "8"}}``,
and the artifact stands for the base model with no adapter (`Artifact imports
<python-api.rst#artifact-imports>`__). Reef accepts it only when the inputs
equal the inputs of this deployment's default start, which the ``source`` of
a scenario's ``creation`` row shows. Another base model, rank or service
returns 400 and names each input that differs. Serving such an artifact
serves the base model, and training from it starts at the base model. A new
scenario of these backends starts from a base root that Reef writes
(`Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__).

+--------+--------------------------------------------------------------------+
| Status | Cause                                                              |
+========+====================================================================+
| 201    | the artifact is imported                                           |
+--------+--------------------------------------------------------------------+
| 400    | a malformed body or URI; a URI of an unknown scheme or of nothing  |
|        | to read; an unknown ``component``, or ``component`` is missing in  |
|        | a scenario with several components; the component takes no         |
|        | import (Reef reads nothing for these three); the directory does    |
|        | not have the component's layout, or its admission check refuses it |
+--------+--------------------------------------------------------------------+
| 404    | unknown scenario; Reef reads nothing then                          |
+--------+--------------------------------------------------------------------+
| 409    | a training result waits for its commit; send the request again     |
+--------+--------------------------------------------------------------------+
| 503    | the source failed, for example S3 refused the read                 |
+--------+--------------------------------------------------------------------+

Set what a scenario serves
~~~~~~~~~~~~~~~~~~~~~~~~~~

A scenario serves one or more artifacts of each component, and one of them is
the active artifact. ``GET /reef/harness`` serves the active harness tree. The
default inference routes answer from the active artifact that completed
startup. Pulling an older artifact changes only your local copy.

The ``served`` setting of ``POST /reef/scenarios/{scenario}/update`` sets what
the scenario serves of one component. It maps artifact IDs to weights:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {"<id>": 100}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

Afterwards the scenario serves exactly the artifacts of the mapping. These
examples use the artifacts ``a`` and ``b`` of one component, where ``a`` is
the active artifact before the request:

+--------------------------+-------------------------------------------------+
| ``served``               | Result                                          |
+==========================+=================================================+
| ``{"b": 100}``           | ``b`` becomes the active artifact, and Reef     |
|                          | stops every other served artifact               |
+--------------------------+-------------------------------------------------+
| ``{"a": 95, "b": 5}``    | a traffic split: ``b`` answers 5% of the        |
|                          | requests                                        |
+--------------------------+-------------------------------------------------+
| ``{"a": 100, "b": 0}``   | Reef loads ``b``, which answers only the        |
|                          | requests that name it                           |
+--------------------------+-------------------------------------------------+
| ``{"a": 100}``           | Reef stops serving ``b``                        |
+--------------------------+-------------------------------------------------+
| ``{}``                   | the scenario serves nothing (`Serve nothing`_)  |
+--------------------------+-------------------------------------------------+

The body is ``{"served": {"<id>": <weight>, ...}, "component": "<name>"}``:

- ``served`` holds one or more artifact IDs of one component. Each weight is
  an integer from 0 to 100, and the weights sum to 100. An empty object makes
  the scenario serve nothing.
- ``component`` names the component. Leave it out when the scenario has one
  component. A scenario with several components requires it. Leave it out with
  an empty ``served``, which applies to the whole scenario.

Reef reads the mapping as follows:

- **The largest weight selects the active artifact.** Among equal weights, the
  first ID in sorted order is active. The active artifact answers each request
  that names no artifact, unless a traffic split gives the request to another
  artifact.
- **More than one weight above 0 is a traffic split.** `Traffic split`_
  describes how Reef divides the requests.
- **Weight 0 loads an artifact.** The artifact answers only the requests that
  name it (`Call one served artifact`_).
- **An artifact that the mapping leaves out stops serving.**

Reef checks the mapping before it changes anything. A mapping that names an
unknown artifact, an artifact of another component, or more weight artifacts
than the runtime keeps loaded at once, changes nothing. `Runtime limits
<../user-guide/operate.rst#set-what-a-scenario-serves>`__ gives that number,
and the wait before a stopped checkpoint frees its place.

The answer is ``{"scenario": "<name>", "served": {...}}`` with the mapping
that Reef accepted for the component. The answer does not show that the
artifacts serve: inference loads them in the background, and publishes the
new routes when the startup completes.

- Poll `Startup status`_ for ``ready`` or ``failed`` before you send requests
  that require a new artifact. Name an artifact that serves beside the active
  one with ``?model=<artifact-id>``.
- A startup can fail after Reef accepted the mapping, for example when a new
  checkpoint finds no free inference component.
- A repeated request retries a failed startup at once without another
  operation, and sets the wait of the automatic retries back to 60 seconds.

Reef appends one operation for each artifact that changes: ``activate`` for a
new active artifact, ``serve`` for an artifact that starts to serve beside it,
and ``stop_serving`` for one that stops. Each operation holds the complete
component mapping, so a restart keeps what the scenario serves. Other
components keep their artifacts. The request creates no artifact or
repository revision, and it never changes a trainer's training start. A
full-weight serving runtime that makes the root snapshot active after it
served a checkpoint loads the root's files.

+--------+--------------------------------------------------------------------+
| Status | Cause                                                              |
+========+====================================================================+
| 200    | Reef accepted the mapping; the artifacts load in the background    |
+--------+--------------------------------------------------------------------+
| 400    | a malformed body; a ``served`` that is no object, a weight that is |
|        | not an integer from 0 to 100, or weights that do not sum to 100;   |
|        | an empty ``served`` with ``component``;                            |
|        | ``component`` without ``served``; an unknown ``component``, or     |
|        | ``component`` is missing in a scenario with several components;    |
|        | an artifact has no durable bytes; the runtime cannot serve the     |
|        | artifact; the mapping names more weight artifacts than the         |
|        | runtime keeps loaded at once, or several artifacts of another      |
|        | component in a scenario whose weights the runtime loads; the       |
|        | mapping names several artifacts, and the store keeps no durable    |
|        | commit history                                                     |
+--------+--------------------------------------------------------------------+
| 404    | unknown scenario; an artifact that is not one of the component     |
+--------+--------------------------------------------------------------------+
| 409    | a training result waits for its commit; send the request again     |
|        | after the commit                                                   |
+--------+--------------------------------------------------------------------+

`Activation rule <../user-guide/operate.rst#activation-rule>`__ explains when
Reef makes a new artifact active automatically. `Go back to an older artifact
<../user-guide/operate.rst#go-back-to-an-older-artifact>`__ explains how to
go back to an older one for good.

Serve nothing
~~~~~~~~~~~~~

A scenario can serve nothing. Use this state to train without an engine for
the scenario, for example on uploaded trajectories. A scenario
enters the state in two ways:

- ``POST /reef/scenarios`` with ``"serves": false`` creates it in this state.
- ``{"served": {}}`` without ``component`` puts an existing scenario in it.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

The state applies to the whole scenario, with every component. While the
scenario serves nothing:

- Reef withdraws the scenario's routes and its traffic split, and stops the
  artifacts that it served beside the active one. The engines stop after the
  delay that a replaced route gets.
- The inference routes and `Startup status`_ answer HTTP 404 for the scenario.
- ``GET /reef/scenarios/{scenario}/settings`` returns an empty ``served``
  mapping for each component.
- The scenario keeps its active selection, and the artifact list still marks
  it ``active``. ``GET /reef/harness`` still serves the active harness tree.
- Training and imports run as before. An activation, also an automatic one,
  changes the active selection and starts no engine.

The scenario serves again when ``served`` names one or more artifacts. The
other components then serve the artifacts of the active selection. The state
is stored, so it holds after a restart.

A nonempty mapping can follow at once: it need not wait for the delayed stop.
The runtime can reuse a verified binding; a retired checkpoint loads again.
Poll the `Startup status`_ of each requested artifact before calling it. A checkpoint that the scenario did not serve
before still needs a free inference component (`Replace a checkpoint
<../user-guide/operate.rst#set-what-a-scenario-serves>`__).

Call one served artifact
~~~~~~~~~~~~~~~~~~~~~~~~

The default inference routes answer from the active artifact, or from the
artifact that a traffic split selects. Each served artifact also answers at
its own URL. Give an artifact weight 0 to:

- keep an older artifact answering while a newer one is active;
- try a new artifact before it is active.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {"<active id>": 100, "<id>": 0}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

Send an inference call to
``/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route}``, where
``{route}`` is ``chat/completions``, ``responses``, ``messages`` or
``messages/count_tokens``. The call needs no ``x-reef-scenario`` header and
never creates a scenario. Reef records it like a call to ``/v1/{route}``,
and the record names the ``artifact_id`` that answered. A call to ``/v1/{route}``
whose ``model`` is the ``artifact_id`` of a served artifact is answered by that
artifact too. An artifact that the scenario does not serve has no URL: its
calls answer HTTP 404.

The same prefix with ``/evaluation/v1/{route}`` makes the same call without a
record.

These rules apply to an artifact that serves beside the active one:

- In a scenario with several components, it stays bound to the artifacts of
  the other components that were active when it started to serve. A later
  activation does not change that binding.
- The activation rule and a restore change the active artifact alone. The
  artifacts served beside it stay served. Training continues from each
  trainer's own training start.
- The samples service does not mark a sample stale while the scenario serves
  the artifact that produced it. After that artifact stops serving, the check
  counts only the artifacts that the scenario served as active: an on-policy
  stage marks a sample whose artifact is not the active one.
- The runtime limits how many weight artifacts a scenario serves together
  (`Runtime limits <../user-guide/operate.rst#set-what-a-scenario-serves>`__).
  A scenario whose components load nothing into a runtime, such as a harness
  scenario, can serve any of its artifacts together.
- A store without a durable commit history refuses a mapping of several
  artifacts.

Restore a scenario selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To restore every component's active artifact together, send
``POST /reef/scenarios/{scenario}/restore`` with ``{"commit_number": 12}``.
The commit number names an existing operation. Reef restores the active
selection as it stood after that commit. Use ``{"commit_number": 0}`` for the
starting artifacts.

The answer is ``{scenario, components}``, where ``components`` maps every
component name to its restored ``artifact_id``. A changed selection appends
a ``restore`` operation. It creates no artifact or repository revision and
leaves each trainer's start, the other served artifacts and the traffic
split unchanged. Restoring the current selection changes nothing. Poll
`Startup status`_ before you use the restored inference artifact.

Reef returns:

- 400 for a body that holds anything but a non-negative integer
  ``commit_number``, for an unknown commit number, and for one whose selection
  cannot be reconstructed from the retained history;
- 404 for an unknown scenario;
- 409 while a training result waits for its commit.

Artifact validation and runtime capabilities still apply.

Train from an artifact
~~~~~~~~~~~~~~~~~~~~~~

Each trainer has a training start: the artifact that its next training job
(a weight trainer) or evolve step (a harness trainer) starts from:

- A new trainer starts from its component's creation artifact.
- After each job or step, the trainer moves its training start to its own
  newest output.
- A harness sync (`Importing a harness`_) moves the harness trainer's
  training start to the artifact that the sync made.

``POST /reef/scenarios/{scenario}/train-from`` sets the start to an existing
artifact and cancels the running job:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

The body is ``{"artifact_id": "<id>", "component": "<name>"}``.
``component`` names the trainer's component. Leave it out when the scenario
has one component; a scenario with several components requires it. The
answer is ``{scenario, component, artifact_id}``. Train-from never changes
the active selection.

Whether a trainer can start from an artifact depends on the trainer:

- The Tinker component imports the artifact's checkpoint manifest, which
  names the saved training state with the optimizer state.
- The Slime component refuses every artifact, because its workers continue
  from their own latest checkpoint.
- A harness trainer starts its next evolve step from the artifact's tree.

A weight artifact without optimizer state, such as a root snapshot made from
the default start, starts training with a new optimizer state.

The route answers these statuses:

+--------+--------------------------------------------------------------------+
| Status | Cause                                                              |
+========+====================================================================+
| 200    | the artifact is the trainer's start                                |
+--------+--------------------------------------------------------------------+
| 400    | a malformed body; the artifact has no durable bytes; an unknown    |
|        | ``component``, or ``component`` is missing in a scenario with      |
|        | several components; the trainer cannot start from the artifact     |
+--------+--------------------------------------------------------------------+
| 404    | unknown scenario or artifact                                       |
+--------+--------------------------------------------------------------------+
| 409    | a training result waits for its commit; send the request again     |
|        | after the commit                                                   |
+--------+--------------------------------------------------------------------+

Inactive artifacts
~~~~~~~~~~~~~~~~~~

The deployment's activation rule can leave a new tree inactive. An example is
a tree whose patch changes a file of a kind listed in the ``always`` rule's
``review_kinds``, or of a kind that is always reviewed (native's
``native_loop``). The step creates an artifact but does not activate it:

- Its row carries ``active: false`` and ``activated: false``.
- The manifest keeps serving the active selection, and ``?artifact_id=`` pulls
  the inactive tree for a trial run.
- To make it serve, send ``{"served": {"<artifact_id>": 100}}`` with the row's
  ``artifact_id`` to ``POST /reef/scenarios/{scenario}/update`` (`Set what a
  scenario serves`_).

``GET /reef/harness/artifacts`` lists only the harness component's artifacts.
A weights update does not create another harness artifact. Each row's
``parent_artifact_id`` identifies the artifact from which it derives, when
known. Every listed artifact can be requested through ``?artifact_id=``.

Version page
~~~~~~~~~~~~

``GET /reef/harness/artifacts/{step}/page`` answers one self-contained HTML
page (``text/html``) for one catalog row. ``step`` is the row's position in
``GET /reef/harness/artifacts``, oldest first, with the creation row as 0. A
rejected step creates no artifact, so the step number identifies the page
independently of its artifact.

The page shows:

- why the step ran (the request, or a failure in the job data);
- ``method_summary``, the patch against the artifact that the step ran on, the
  evaluation's numbers, ``method_notes`` and ``requires``;
- links to the parent step and to later training or sync operations;
- ``Active selection`` for the selected artifact, and ``Waiting for
  activation`` for a step whose artifact was never activated (``activated:
  false``).

The row itself is in a ``<script type="application/json">`` block at the end
of the page. A step outside the catalog, or one that is not a number of at
most nine digits, returns 404.

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: code-repair" \
     "$REEF_URL/reef/harness/artifacts/3/page" > harness-step-3.html

Request page
~~~~~~~~~~~~

``GET /reef/harness/requests/{record_id}/page`` answers one self-contained
HTML page (``text/html``, ``Cache-Control: no-store``) for a harness request
filed through ``POST /reef/train``. ``record_id`` is the ``agent_record_id``
of the request's response, and its ``page_path`` links this page.

- Until a step settles the request, the page reloads every five seconds and
  shows the request's state and the step's activity (its latest 80 lines).
- The catalog row whose ``metrics.training_request.id`` is the record ID
  settles the page. It then shows the result, the next action (``GET
  /reef/harness`` for an active selection, the update request that serves an
  inactive one), ``method_summary``, the patch with each file's kind of the
  harness tree (from the row's ``patch_parts``), and ``method_notes``.
- An unknown ID, or one that is not a training request, returns 404.

``GET /reef/harness/requests/{record_id}/progress`` answers the same state as
JSON (``Cache-Control: no-store``) for a client that polls:

- ``request_id`` and ``settled``;
- ``step``: the step the row landed as once it settles, else ``null``;
- ``state``: ``queued`` (no step has taken the request), ``proposing`` (the
  method makes the candidate), ``evaluating`` (the candidate's rollouts run),
  ``running`` (the trainer holds the request and reports no phase), or the
  settled row's result, ``failed`` included;
- ``meaning``: the words the page prints beside the state, ``null`` once
  settled;
- while a step holds the request: ``started_at``, ``runs_total``,
  ``step_record`` and ``activity``, the activity lines oldest first, each
  ``{at, kind, text}`` with ``failed: true`` on a failed line. ``kind`` is
  ``model`` for a model call, ``evaluate`` for the method's rollouts, or a
  kind that the method notes itself.

This route reads the headers alone; a ``?key=`` is HTTP 401. An unknown id is
HTTP 404.

Page links
~~~~~~~~~~

A browser sends no header, so the two pages also take the scenario and a
credential as query parameters, ``?scenario=<name>&key=<page key>``; a header
wins when present. Clients never build these links: ``POST /reef/train``
answers ``page_path`` for the request page, and ``GET /reef/harness/artifacts``
one per row for its step's page. A client prints its service URL followed by
that path.

.. code:: text

   $REEF_URL/reef/harness/requests/<record_id>/page?scenario=<scenario>&key=<page key>

- The path holds a page key only when the request presented a service token.
  An evaluation token gets none, and with authentication off the query names
  the scenario alone.
- The key is an HMAC SHA-256 of the scenario, keyed by the token's digest. It
  opens these two pages of that one scenario and no other route, and the
  token cannot be read back from it. So a client may print the links where a
  session's model reads them.
- A request whose ``x-reef-scenario`` header names another scenario returns
  401.
- Reef never reads the token itself from a query: ``?token=`` returns 401 on
  every route, and ``?key=`` returns 401 on every route but the two pages.

Retained step files
~~~~~~~~~~~~~~~~~~~

``GET /reef/harness/artifacts/{step}/records`` returns the raw file inventory
of the same catalog step's record directory, authenticated and scoped to the
scenario like the version page:

- Without a query, the response is ``{"status": "retained", "files":
  [{"path": "calls.json", "bytes": 123}]}``.
- With ``?path=calls.json`` (or another inventory path, such as
  ``output/notes.json`` or ``material/traces/0/trace.json``), it is
  ``{"status": "retained", "path": "...", "text": "..."}``.
- A step whose row names no record directory returns ``status:
  not_recorded``, and an absent directory returns ``status: missing``. Both
  carry an empty files list.
- All successful reads use ``Cache-Control: no-store``.

Reef reads only the step directory that the selected catalog row names, under
the scenario's ``steps/`` directory:

- It rejects absolute paths, traversal and symlinks.
- It exposes only JSON and JSONL files. An inventory holds at most 1000
  filesystem entries, and a file at most 4 MiB. Past these limits it returns
  400; it never truncates.
- A missing file or catalog step returns 404.

The service does not interpret model replies or agent events. ``calls.json``
holds the recorded messages and replies of the method's model calls. The
evaluation runs are in the scenario's history, not in the step directory, so
this route does not serve them.

Reads copy no record into another store and change no retention. Step files
remain separate from the capacity eviction of online record bodies.

Status
------

Read ``GET /reef/status`` to see what trains and what serves, for example
while inference still serves an older snapshot during a training job.

.. code:: json

   {
     "error": null,
     "loaded_scenarios": 1,
     "preload_errors": {},
     "scenarios": {
       "hello-reef": {
         "commit_number": 3,
         "artifact_head_sync": {"state": "synchronized", "error": null},
         "last_training_commit": {
           "commit_number": 3,
           "recorded_at": 1756400000.0,
           "metrics": {"selected": false, "selection": {"reason": "candidate lost"}}
         },
         "served_artifact_id": "5e1c...",
         "job_data_ready": false,
         "training_mode": "auto",
         "processor": {"...": "..."},
         "inference_admission": {"...": "..."},
         "artifacts": {"weights": {"active": "5e1c...", "training_start": "5e1c..."}},
         "training_jobs": [{"job_id": "9c41...", "start": "/checkpoints/iter_0000004", "checkpoints_handled": 0,
                            "train_from": null, "backend": "slime"}],
         "finished_jobs": 2
       }
     },
     "serving": {"...": "..."}
   }

Top-level fields:

- ``error`` and ``preload_errors`` report asynchronous training and preload
  failures.
- ``loaded_scenarios`` counts the scenarios loaded in this process.
- ``serving`` is runtime-wide and keyed by recipe name. The runtime of a
  weight-training recipe that serves per-scenario adapters reports the
  engines' shared adapter residency there, under ``adapters``. For a composite
  recipe, ``adapters`` is under the name of the weight component.

Fields of each scenario:

- ``commit_number`` is the number of the scenario's newest commit.
- ``last_training_commit`` holds the commit number of the latest durable
  training commit, its commit time, and its recipe-owned metrics. It is
  ``null`` before the first training commit. It distinguishes a step that
  still runs from a completed step that skipped or rejected its candidate. An
  activation, a train-from, an import or a sync advances ``commit_number``
  without replacing it.
- ``served_artifact_id`` is the ``artifact_id`` of the artifact that answers
  the scenario's requests now, the one that ``served_models`` names: the
  active artifact of the weight trainer's component, or of the only component.
  It is null before inference publishes a route.
- ``job_data_ready`` says whether the processor has a job's data waiting.
- ``training_mode`` is the selected mode.
- ``artifacts`` gives two ``artifact_id`` values for each component: its
  active artifact and its trainer's training start. A value is ``null`` when
  Reef does not know it. Both values start at the creation artifact.
- ``processor`` includes ``ready_samples``, the untrained samples that the
  next automatic job takes, from recorded traffic and uploaded trajectories;
  ``job_samples``, the samples that automatic job data needs
  (``pipeline.sample2job.config.job_samples``); and ``buffer_size``, the most
  samples that one automatic job takes, beside the rest of a group. A
  processor that takes requests also reports ``pending_requests``.

A scenario that trains weights also reports its training jobs:

- ``training_jobs`` lists the jobs that the scenario's training component
  runs, in the order of their submission. The list is empty while no job
  runs. Each job holds:

  - ``job_id`` and ``backend``, the backend that runs the job;
  - ``start``, which the job trains from: a checkpoint's path as the component
    reads it, or ``snapshot:`` and a JSON object ``{revision, root}`` that
    names the snapshot of the scenario;
  - ``checkpoints_handled``, the number of the job's saved checkpoints that
    Reef has committed;
  - ``train_from``, the artifact that the job's request named as its start,
    or ``null`` for a job that continues the training start.

- ``finished_jobs`` counts the jobs of the scenario that ended, finished or
  failed.
- A runtime that serves each scenario its own adapter adds
  ``adapter_runtime_load_id``: the runtime load ID of the scenario's served
  adapter, or ``null`` while the scenario samples the base model.

A scenario whose recipe runs one trainer per recipe component adds a
``components`` object. For each recipe component, it holds the component's
own ``job_data_ready``, ``training_mode``, ``processor`` status and
``last_training_commit``. The scenario-wide ``job_data_ready``,
``training_mode`` and ``processor`` then describe the first trainer that
trains, else the first trainer. The scenario-wide ``last_training_commit`` is
the latest training commit of any recipe component.

Status codes
------------

+--------+-------------------------------------------------------------+
| Status | Cause                                                       |
+========+=============================================================+
| 400    | malformed body, a missing or empty ``x-reef-scenario`` on a |
|        | scenario-scoped route, a report violating the recipe's      |
|        | declared schema, or a ``served`` mapping or a train-from    |
|        | naming an artifact that Reef cannot serve or train from     |
+--------+-------------------------------------------------------------+
| 401    | missing or wrong bearer token, or ``x-api-key`` when no     |
|        | Authorization header is sent; the two harness pages also    |
|        | read ``?key=`` (see Request page)                           |
+--------+-------------------------------------------------------------+
| 403    | a routed request for an unconfigured Scenario or a pin to a |
|        | target outside it (see Routed external-model requests), a   |
|        | console origin, method or headers that are not allowed, or  |
|        | an engine's own 403; an unaccepted token is 401, and        |
|        | per-scenario authorization belongs to the gateway in front  |
|        | of Reef                                                     |
+--------+-------------------------------------------------------------+
| 404    | unknown scenario (on inference, on ``POST /reef/train``,    |
|        | or with implicit creation off), a scenario that serves no   |
|        | model, unknown snapshot or artifact, unknown adapter, no    |
|        | configured harness recipe, or a scenario that serves no     |
|        | files                                                       |
+--------+-------------------------------------------------------------+
| 409    | a record id resent with different content, a ``served``     |
|        | mapping or a train-from while a training result waits for   |
|        | its commit, or an engine that aborted the request on both   |
|        | attempts                                                    |
+--------+-------------------------------------------------------------+
| 4xx    | an engine's own 4xx, with its message, such as for a prompt |
|        | that fills the model's context window: an engine's 400,     |
|        | 401, 403, 404, 408, 409, 422 or 429 passes through as sent, |
|        | and any other engine 4xx becomes 400                        |
+--------+-------------------------------------------------------------+
| 502    | the engine failed with a 5xx, answered with no JSON object, |
|        | or could not be reached; the body carries the engine's      |
|        | status and message for a 5xx, and the engine's URL and the  |
|        | connection error for an engine that could not be reached    |
+--------+-------------------------------------------------------------+
| 503    | the artifact store is unreachable, no loaded scenario is    |
|        | idle at ``reef.max_loaded_scenarios`` (with                 |
|        | ``Retry-After``), ``served_models`` cannot be read, the     |
|        | record store does not take an inference record, the engine  |
|        | did not serve the selected weights on either attempt, or a  |
|        | retry found no row for its artifact (with ``Retry-After``)  |
+--------+-------------------------------------------------------------+
| 504    | the engine did not answer within ``inference.timeout-s``    |
+--------+-------------------------------------------------------------+

Browser consoles
----------------

A browser console sends requests to this HTTP API directly, as every other
client does. The browser must be able to open the service URL. To let a console
connect, add ``reef.console-origins`` to the serve YAML. List each trusted
console origin explicitly, with no path, trailing slash, credentials, or
wildcard:

.. code-block:: yaml

   reef:
     console-origins:
       - "https://api.reefinfra.ai"
       - "http://localhost:3000"

Restart the service after you change the configuration.

- Without this option, Reef adds no CORS headers. With it, Reef rejects
  unlisted browser origins before a route runs.
- CORS preflight requests from listed origins need no service token. Actual
  requests keep the configured Bearer authentication.
- Browser requests may use GET, POST or DELETE with the ``Authorization``,
  ``Content-Type`` and ``x-reef-scenario`` headers. CORS enables no cookies.
- CORS headers also cover errors and streamed responses, and expose
  ``x-reef-agent-record-id`` and ``x-reef-artifact-id`` to the browser.
- Clients without an ``Origin`` header keep their behavior.
- A browser can also require local network permission or HTTPS. An origin that
  Reef allows does not override browser policy.

A connected console acts with the permissions of the service token, on the
scenarios of this service. The CORS connection does not create a cloud
deployment or upload history.

Record and commit history
-------------------------

All these reads require service authentication and return ``Cache-Control:
no-store``. Payloads and recorded metrics can contain request content, so a
platform must bind the upstream scenario to the authenticated workspace. Each
response is a live read: concurrent training and retention can change later
pages.

**Records.** ``GET /reef/scenarios/{scenario}/records`` reads retained record
metadata, including consumed records, oldest first:

- ``after_sequence`` defaults to 0, and ``limit`` to 50 (1 to 100).
  ``next_after_sequence`` is null at the end.
- ``request_type`` (``inference``, ``report`` or ``train``) lists one type
  only; another value returns 400.
- Each row contains ``sequence``, ``agent_record_id``, ``request_type``,
  ``created_at``, ``references``, ``artifact_id`` and the payload's ``score``
  field. ``artifact_id`` names the artifact that answered an inference call,
  or is null when the record names none. The row has no payload.

``GET /reef/scenarios/{scenario}/records/{record_id}`` returns that metadata
and the stored ``payload``. A missing body returns 404: it may have expired,
or Reef never retained it. A read never reactivates a record or changes
retention.

**Operations.** ``GET /reef/scenarios/{scenario}/operations`` returns
``{scenario, operations, next_after_commit_number}``, oldest first:

- ``after_commit_number`` defaults to 0, and ``limit`` to 50 (1 to 100).
  ``next_after_commit_number`` is null at the end.
- ``/commits`` remains an alias, with the array named ``commits``.
- Repeat ``record_id`` to keep the operations whose ``consumed_ids`` contain
  any requested ID (at most 100 IDs of 1 to 256 characters). This tests exact
  set membership, not whether a record was eligible for training.
- Each row has ``commit_number``, ``operation``, ``operation_verified``,
  ``recorded_at``, ``components``, ``component``, ``activated``,
  ``consumed_ids`` and ``metrics``. ``components`` maps component names to the
  artifact IDs of the operation. ``component`` identifies the changed
  component when present. The row has no private storage locator, algorithm
  state or checkpoint recovery data.
- Operations remain readable after trace bodies expire.

The operations are:

- ``training``: a training outcome, including steps that create no artifact.
  ``activated`` reports whether the step selected its new artifact to serve.
- ``activate``: select an existing artifact for one component.
- ``restore``: restore the complete active selection from an earlier step.
- ``train_from``: change one trainer's starting artifact.
- ``import``: create an artifact from an external source.
- ``sync``: create a harness artifact and select it for serving and training
  in one operation. Its ``activated`` value is true.
- ``serve`` and ``stop_serving``: add or remove an artifact that the scenario
  serves beside the active one.

A ``served`` request appends ``activate``, ``serve`` and ``stop_serving``
operations, one for each artifact that changes (`Set what a scenario
serves`_). Activation, restore and serving create no artifacts or repository
revisions. An import or a training step can create an artifact without
changing what serves. Restore appends a new operation; it does not remove or
rewind history. Recovery replays serving selections and training starts
independently.

Reef does not join these routes into learning links, and it assigns no
learning state, explanation, policy capability flag or evaluation result. The
console owns that interpretation. Consumption does not prove that a snapshot
made from the records serves.

Sample processing tasks
-----------------------

``GET /reef/scenarios/{scenario}/sample-processing`` reads the samples queue
through the data service. The scenario must exist. The read starts no
processing. Tasks appear newest first. The query parameters are optional:

- ``state``: ``queued``, ``assigned``, ``running``, ``finished`` or
  ``failed``.
- ``before``: a positive integer. It excludes creation sequences at or above
  it.
- ``limit``: 1 to 100, 50 by default.
- ``changed``: ``true`` or ``false`` (the default). With ``true``, the page
  omits finished tasks that gave the trainer nothing new. Reef reads at most
  5,000 tasks to fill the page.

The response has ``scenario``, ``read_at``, ``retention_seconds`` and these
fields:

- ``enabled``: the scenario has a trainer that generates samples. It does not
  report worker health.
- ``records_waiting``: the number of the scenario's records after the last
  record that a trainer read, for the trainer furthest behind. Each trainer's
  newest finished task with ``record_sequences`` gives that record. It is null
  when a trainer has no such task in the retained history.
- ``counts``: covers all retained tasks, independent of pagination and state
  filters. Reef reads the page and the counts separately, so tasks can change
  between them.
- ``tasks`` and ``next``: the page, and the next page's ``before`` value or
  null.
- ``hidden``: each run of omitted tasks; empty without ``changed=true``. A
  run has ``after``, ``count``, ``first_created_at`` and ``last_created_at``.
  ``after`` is the ID of the shown task just newer than the run, or null at
  the top of the page.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}`` returns one
task. An unknown task, or a task of another scenario, returns 404.

Each task has ``id``, ``component``, ``state``, ``attempt_count``,
``created_at``, ``ended_at``, ``available_at`` and the fields below. Times are
Unix seconds, and unavailable fields are null. A task from an older Reef has
null counts, no ``record_sequences`` and no triggers.

- ``started_at`` is the first attempt's start. ``error`` can also describe the
  previous failed attempt of a queued task.
- ``records_read`` and ``samples_created`` count the completed attempt only,
  without earlier failed attempts. ``records_read`` includes records that
  replay read but the pipeline ignored. ``samples_created`` counts new
  inserts, without duplicate writes.
- ``record_sequences`` is ``[first, last]``: the data sequences of the first
  and last record of the scenario that the completed attempt read, or null
  when it read none.
- ``changed``: the task gave the trainer something new. This can also be a
  new training request, without new samples.
- ``background``: ``true`` means that work remained active when the task
  finished; later tasks check it.
- ``triggers`` lists why the task ran, oldest first. Each entry has ``name``
  and ``at``, the time of the first request.
- ``progress`` holds the counts so far of the attempt that runs now:
  ``attempt``, ``records_read``, ``samples_created`` and ``at``. It is null
  for a task that does not run.

A trigger name is ``records``, ``data``, ``commit``, ``reload``,
``background`` or ``due``. Only ``records`` and ``data`` bring new records.
The others let a stage make samples without a new record. ``background``
comes from a background job of a stage that ended, or from a fallback look
while such jobs run. The fallback waits 1 second, then twice as long after
each look that changed nothing, up to 300 seconds.

A sample that a task stored has the item metadata ``stored_by_task`` and
``read_by_task``. A data query with ``group_key`` set to one of them finds the
samples of one task.

Control removes ended tasks after ``storage.work_retention_s``, which defaults
to seven days. This removes task history, not stored samples. Active tasks do
not expire.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}/logs`` returns
the log of one attempt of a task:

- The samples worker keeps each line that the attempt's work logs. This
  includes the stage's background jobs that the attempt submitted, also when
  they end after the attempt. It does not keep lines of threads that the work
  starts by hand.
- Reef writes the lines every 2 seconds and when the attempt ends.
- One attempt keeps at most 2,000 lines of at most 4,096 characters. The line
  after the last one says that the log was cut.
- The log goes with its task when the retention deletes the task.

Optional query parameters are ``attempt`` (1 to the task's ``attempt_count``;
the last attempt by default), ``after`` (-1 or more, -1 by default; the page
holds lines with a larger number) and ``limit`` (1 to 1000, 500 by default).
The response has ``task_id``, ``attempt``, ``attempt_count``, ``running``,
``lines`` and ``next``:

- ``attempt`` is null for a task that no worker took.
- ``running`` is true while the chosen attempt runs, so more lines can come.
- Each line has ``number``, ``at``, ``level``, ``logger`` and ``message``.
- ``next`` is the next page's ``after`` value, or null at the end of the
  lines kept so far.
