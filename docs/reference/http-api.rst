HTTP API: inference, feedback, and artifacts
============================================

Reef serves the provider's own inference routes: OpenAI Chat Completions at
``/v1/chat/completions``, OpenAI Responses at ``/v1/responses``, and Anthropic
at ``/v1/messages``. It forwards each
request to the runtime unchanged. It adds a small set of ``/reef/*`` routes for
feedback, scenarios, artifacts, and status.

The service accepts request bodies up to 64 MiB so coding agents can send
large contexts. Record import routes accept request bodies up to 10 MiB.

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
| ``POST /reef/scenarios/{scenario}/v1/{route}``                                                                        | one of the inference routes above, with the       |
|                                                                                                                       | scenario in the path instead of a header: served  |
|                                                                                                                       | and recorded as the same call to ``/v1/{route}``; |
|                                                                                                                       | what a client install's model binding calls       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluation/v1/{route}``                                                             | one of the inference routes above, answered by    |
|                                                                                                                       | the scenario's active model and kept by nobody:   |
|                                                                                                                       | what a recipe's evaluation rollouts call. The     |
|                                                                                                                       | answer has no ``training`` block. 404 for an      |
|                                                                                                                       | unknown scenario or one that serves no model      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/components/{component}/evaluation/v1/{route}``                                      | the same, for a candidate of one component of a   |
|                                                                                                                       | composed snapshot                                 |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route}``                                                | one of the inference routes above, answered and   |
|                                                                                                                       | recorded by an artifact that the scenario serves  |
|                                                                                                                       | beside its active selection; 404 for any other    |
|                                                                                                                       | artifact. The same prefix with                    |
|                                                                                                                       | ``/evaluation/v1/{route}`` makes the same call    |
|                                                                                                                       | without a record                                  |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/records``                                                                                                | import one existing inference or report           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/records/batch``                                                                                          | atomically import a batch of records              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data``                                                                                | create record and artifact items                  |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/trajectories``                                                                   | create trajectory items from JSON Lines           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/tasks``                                                                          | create task items from a tar.gz of Harbor tasks   |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``PUT /reef/namespaces/{owner}/data/tasks/{id}``                                                                      | replace a task item's files                       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data``                                                                                 | the owner's items, in arrival order               |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/groups/{key}``                                                                    | the owner's items grouped by a metadata value     |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/{id}``                                                                            | one item, with its content                        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/namespaces/{owner}/data/metadata``                                                                       | patch the metadata of items                       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/data/{id}``                                                                         | delete one item                                   |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/namespaces/{owner}/data/{id}/archive``                                                                    | one item to download: a task's files as tar.gz,   |
|                                                                                                                       | another item's content as JSON                    |
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
| ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{benchmark}``                                          | make a scenario hold a benchmark in a split       |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{benchmark}``                                       | end a scenario's hold of a benchmark              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/report``                                                                                                 | submit feedback about one or more trajectories    |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/train``                                                                                                  | enqueue one training request                      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios``                                                                                               | scenario selections and harness adapter           |
|                                                                                                                       | adapter                                           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios``                                                                                              | create a scenario explicitly                      |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/inference/status``                                                                   | artifact startup state; ``?model=active`` by      |
|                                                                                                                       | default                                           |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/update``                                                                            | update a scenario's settings, and set the         |
|                                                                                                                       | artifacts that it serves                          |
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
| ``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files``                                                      | an artifact's file list, or one file with         |
|                                                                                                                       | ``?path=``                                        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/artifacts``                                                                         | import a directory as a new artifact              |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/restore``                                                                           | restore the selection of an earlier commit        |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/train-from``                                                                        | set the artifact a trainer starts its next job or |
|                                                                                                                       | step from                                         |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/redeploy``                                                                          | deploy the served engines again (RFC #1299)       |
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
| ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/retry``                                                 | run the rollouts without a score again            |
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
| ``DELETE /reef/scenarios/{scenario}``                                                                                 | remove a scenario; its state is archived          |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness``                                                                                                 | the harness tree of the active selection, rendered|
|                                                                                                                       | for the adapter, as its source tree, or bound for |
|                                                                                                                       | a client install                                  |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/harness/import``                                                                                         | merge a person's harness files onto the newest    |
|                                                                                                                       | head and commit them as a sync                    |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/artifacts``                                                                                       | harness artifact catalog, oldest first            |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/artifacts/{step}/page``                                                                           | one HTML page per catalog step: why, the method's |
|                                                                                                                       | summary, what changed, result, the method's       |
|                                                                                                                       | notes, setup, chain                               |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/requests/{record_id}/page``                                                                       | one HTML page per filed harness request: its      |
|                                                                                                                       | step's state, then the result; reloads itself     |
+-----------------------------------------------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/requests/{record_id}/progress``                                                                   | the same reading as JSON: the step's phase, for a |
|                                                                                                                       | client with no browser to open the page           |
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
<../user-guide/operate.rst#run-the-services-in-separate-processes>`__).

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

``x-reef-tag-artifact_id`` identifies the harness artifact installed by the native client.
Reef stores it under ``metadata.tags`` like any other tag, so a processor can
read which harness artifact made the call.
The tag does not change which artifact
serves inference, and it does not change the record's ``artifact_id``: the
record names the artifact that answered. This tag is separate
from ``x-reef-artifact-id``, the response header that names the model artifact.

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

With ``reef.routing_manifest`` set, the deployment controller uses
``GET /reef/lifecycle`` and ``POST /reef/lifecycle`` on a private Unix socket
that the routed Reef child serves (directory 0700, socket 0600). These routes
are never served on ``reef.port``. They require a separate per-generation
control credential: service and evaluation tokens cannot open them, and the
control credential is not a service token on public routes. Operators normally use
``reef serve`` and the deployment status/cleanup commands rather than calling
this private interface.

``GET`` returns ``deployment_id``, ``generation``, ``snapshot_version``,
``router_config_hash`` (empty until the owner publishes a verified runtime),
``control_revision``, ``accepting``, ``stopping``, ``active_requests`` and
``pending_writes``, plus ``drain_pending_writes``, ``policy_closed``,
``pending_selections`` and diagnostic
``forced_requests`` and ``failed_writes`` counts. An admitted request keeps the
manifest and router runtime hash it captured at admission, including while the
router previews it. ``GET /healthz`` remains a
liveness check; it does not mean the managed
routing deployment is ready to accept inference.

``POST`` takes the deployment ID, generation, a nonnegative integer
``control_revision`` and an ``action``: ``open``, ``close``, ``close_policy``,
``abort_policy``, ``stop`` or ``activate``.
``close_policy`` closes admission for a normal policy update. The controller
waits for ``pending_selections`` to reach zero before sending the SR update.
Subsequent publication and reopening commands identify that close with
``policy_close_revision``. If selection times out before an SR update was
attempted, the controller can use ``abort_policy`` to reopen the verified
original policy after readiness. A full close or stop supersedes the policy close.
``activate`` also takes ``manifest_path``, ``snapshot_version``,
``expected_snapshot_version`` and a 64-hex ``router_config_hash``; it publishes
them together and leaves admission closed. During a policy close it may
publish while previously selected requests and their writes are still active.
``open`` and ``abort_policy`` require ``open_deadline_monotonic``: a positive, finite
value of the owner's ``time.monotonic()`` clock, which the child shares because
both run on the same host. ``open`` returns 409 until a router runtime hash has
been published, and on a routed deployment first checks every provider and
sends one preview per Scenario under that hash; a failed check returns 503 and
stays closed. An ``open`` that has not opened admission by its deadline returns
503 and changes nothing, including the revision. Revisions increase
monotonically. Repeating the same revision and command, including the deadline,
is idempotent: an applied ``open`` returns the current status, also after its
deadline. Stale revisions, conflicting commands and identity mismatches
return 409. ``stop`` is terminal for the generation. ``close`` and ``stop``
also accept ``cancel_active: true`` to cancel accepted inference requests
after their drain deadline; ``open`` and ``activate`` cannot use this option.

``POST /reef/lifecycle/check`` on the same socket, with the same control
credential, checks a candidate manifest before the owner changes anything. It
takes exactly ``deployment_id``, ``generation``, ``manifest_path``,
``snapshot_version`` (the candidate's digest), ``expected_snapshot_version``
(the child's current snapshot) and ``check_deadline_monotonic``, a positive,
finite value of the owner's ``time.monotonic()`` clock. The child refuses the
candidate with 409 when it would not publish it: an unreadable manifest,
another digest, a changed automatic model, Scenario names or domain
classifier, or a target credential variable missing from the child's own
environment. That reply is the fixed text ``routing candidate rejected``; the
reason goes only to the child's log. It then checks each enabled
target's provider model list with the child's credentials and returns 200 with
``{"snapshot_version": ...}``, or 503 with only an error code when a provider
fails or the deadline passes. It is not a command: it has no revision, does not
wait for or block other commands, publishes nothing, does not change admission
and keeps nothing. A check that a ``stop`` or another publication overtakes
returns 409. A malformed body returns 400, and identity mismatches return 409.

Managed inference starts closed. A closed deployment rejects new inference
with 503 without invoking a provider. Admission and the request's routing
policy are fixed together before the body is read, so the upload counts toward
a later close's drain and a policy published meanwhile does not change it.
An accepted request remains active through local response EOF or connection
termination, including SSE terminal frames. Local EOF does not establish
client receipt. Reports can finish during ``close``;
``stop`` rejects new mutations. Record writes already submitted remain counted
after request cancellation and are not retried by the lifecycle controller.
During ``close``, ``drain_pending_writes`` counts all inference-owned writes,
including finalization submitted after close, and other writes submitted before
the first close. Reports accepted after close do not extend that drain. Repeated
close commands preserve its boundary, and ``open`` rejects with 409 until these
writes finish. During terminal ``stop``, the drain count includes every pending
write. ``pending_writes`` always reports the total.

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
not mean a training step has finished. Identical retries return the existing
receipt, including after capacity eviction; reusing the ID with different content
returns HTTP 409. Authentication failures return 401 and invalid envelopes,
report schemas or references return 400. Scenario selection comes only from
headers; envelope fields other than the three shown above are rejected.

``request_type`` may be ``inference`` or ``report``. Inference payloads follow
the selected recipe's recorded-input contract: native provider request fields
with an existing ``response``, or another complete example that its processor
understands. Import does not run inference or assign the active artifact to
externally generated examples. Training reads no token IDs or log
probabilities from an imported record: the training component renders the
recorded exchange with the model's chat template.
Reports use the same schema and reference validation as ``POST /reef/report``;
import referenced inferences first. Training requests use ``POST /reef/train``.
To train on ATIF trajectories that are already complete, see `Data items
<#data-items>`__.

For existing datasets, use the batch endpoint and resumable JSONL importer
below. Both endpoints accept request bodies up to 10 MiB.

Imported records enter the same scenario storage and wake the same training
worker as live traffic. Keep the same scenario header when subsequently calling
``POST /v1/chat/completions``: its records continue through the existing
processor, with no processor replacement or end-of-dataset signal. A processor
that needs reward/report feedback still needs it for both imported and live
records; submit each example's feedback promptly so completed jobs can drain.
For example, feedback for the imported record can be sent as:

.. code:: json

   {"agent_record_id": "offline-score-0001", "request_type": "report",
    "payload": {"references": ["offline-0001"], "score": 1.0}}

Job size, buffering, retention and training policy remain owned by the selected
recipe and processor. Uploading can overlap training; this API adds no dataset
epochs, cold-start completion barrier, or automatic algorithm switch. To start
chat traffic after cold-start training has finished, use the recipe's progress
and commit status to determine when its imported examples have been consumed.

Batch imports and resume
~~~~~~~~~~~~~~~~~~~~~~~~

``POST /reef/records/batch`` accepts ``{"records": [record, ...]}``, where each
record uses the same three-field envelope as ``POST /reef/records``. The batch
must contain 1 to 1000 records and fit within the 10 MiB request-body limit.
All records share the request's scenario and tag headers. Put an
inference before reports that reference it, including within the same batch.

Reef validates the entire batch and persists it in one database transaction.
Malformed records or missing references return 400; conflicting IDs return
409. Neither failure persists any new records from the batch. Identical retries
are accepted, including after training or capacity eviction. Oversized requests return
413. The response is ``{"records": [receipt, ...]}``, in input order, with the
same receipt fields as the single-record endpoint. A successful import wakes
the existing consumer once after commit. Atomicity applies to one request,
not the entire dataset or subsequent training.

On the user machine, install the standalone ``reef-client`` package; no Reef
service or training dependencies are needed. For a large local dataset, save
one envelope per line in an immutable UTF-8 JSONL file. Keep original record IDs
and report references; the importer does not generate new IDs or convert
arbitrary dataset schemas.

.. code:: bash

   # Authentication is read from REEF_TOKEN when set.
   reef-client import records.jsonl --url http://127.0.0.1:8900 --scenario my-agent

The CLI reads incrementally and sends batches limited by both ``--batch-size``
(default 128 records) and ``--max-batch-bytes`` (default 524288 bytes). Only a
bounded batch and one look-ahead record are buffered. The defaults are starting
settings, not throughput guarantees. A single JSONL line must fit the byte
limit; increase it, below 1048576, for larger records.

Progress is saved to ``records.jsonl.reef-import.json``, or ``--progress PATH``.
It contains the destination, scenario, source checksum and last acknowledged
byte offset/count; it contains no token or record payloads. Each invocation
first hashes the complete file with bounded memory to verify its identity,
then seeks to the saved offset. The source must remain unchanged during import.
A local file lock prevents simultaneous use of the same checkpoint.

Run the same command after interruption to continue. Transient connection
failures and HTTP 408, 429, 500, 502, 503 or 504 are retried up to ``--retries``
(default 3) with backoff and unchanged IDs. If a reply is lost after commit,
server deduplication prevents the replayed batch from being inserted again.
Permanent errors stop without advancing that batch's checkpoint. Earlier
acknowledged batches remain stored. A modified source or different destination
requires a separate progress file; existing IDs still deduplicate at the server.
The reported count includes every acknowledged source row, including retries
of records already present, and does not count completed training samples.

This importer supports local JSONL on platforms with POSIX file locking, including
Linux and macOS. Parquet/Hugging Face schema conversion and server-side S3 import
jobs are not part of this interface. Import completion acknowledges storage;
use the scenario's training progress before treating cold-start learning as done.

The upload buffer limit does not bound processor memory. Report-based processors
may retain inference records while waiting for feedback; placing all inferences
before all reports can therefore build a large in-memory backlog. Where possible,
place reports close to their referenced inferences. Large-dataset throughput and
end-to-end memory use require measurement with the selected recipe.

Data items
----------

Reef keeps the data that scenarios train on as data items, in one data store
for the deployment (RFC #1028). An item is content and metadata:

- The content is an ATIF trajectory, a Harbor task directory, a record, or a
  reference to the files of an artifact. It never changes. To change data,
  create a new item.
- The metadata is a JSON object. Reef reads ``kind`` (``trajectory``,
  ``task``, ``record`` or ``artifact``), ``owner``, ``scenarios`` and
  ``origin``: the scenario that made a record or an artifact. A record's
  ``origin.source`` is ``inference`` or ``evaluation``; a record without it is
  ``inference``. ``kind``, ``owner``, ``origin`` and ``evaluation_id`` never
  change. A task can have ``benchmarks``: the names of the benchmarks that
  it is in (`Benchmarks and collections`_). You can add other keys.

Each item belongs to one owner. The routes have the owner in the path:
``/reef/namespaces/{owner}/data``. ``scenarios`` maps each scenario that uses
the item by this mark to its split, ``train`` or ``eval``. A trajectory is always
``train``. A scenario can also use a task through a benchmark that it holds,
with no mark on the task (`A scenario's benchmarks`_). An artifact item has no ``scenarios``: a scenario uses it by
importing it (`Artifact items`_).

The data service answers the data routes. A one-process ``reef serve`` runs
it with the other services. When the services run in separate processes,
send the data routes to the data service: control does not answer them (`Run
the services in separate processes
<../user-guide/operate.rst#run-the-services-in-separate-processes>`__).

A scenario uses only the items of its owner. A scenario's owner is the
scenario's name, or the ``owner`` that ``POST /reef/scenarios`` gave when it
created the scenario. Reef refuses an item that names a scenario of another
owner. Archiving a scenario does not delete its items. It removes the
scenario from their ``scenarios``, except from the record items that the
scenario made.

Training reads the items that a scenario uses from the data store; an item that
leaves the scenario before a job takes it does not train. Each trajectory
counts as one sample toward ``job_samples`` and ``buffer_size``. Job data takes
the samples in the order they joined the scenario. A train task trains like a trajectory. In ``sft`` it trains only in
``manual`` mode, for a request that gives the job's parameters (see
`Manual training <#manual-training>`__). For the recipe side, see `Uploaded
trajectories <../developer-guide/processors.rst#uploaded-trajectories>`__.

After each write that adds items to a scenario or removes them, the data
service asks control to sync the scenarios that the write touched. It waits up
to 30 seconds for the syncs. Each create, update and delete response has
``synced``: true when every sync ended, false when one still waits. A sync that
waits runs when control takes it, or when the scenario loads.

Create trajectory items
~~~~~~~~~~~~~~~~~~~~~~~

Send trajectories as JSON Lines to ``POST
/reef/namespaces/{owner}/data/trajectories`` with ``Content-Type:
application/x-ndjson`` (or ``application/jsonl``). Each line is one ATIF
document, and Reef skips blank lines. A document must have an ``ATIF-v``
``schema_version``, an ``agent`` object, one or more steps, and a
``session_id``, which is the item's ID. One request holds 1 to 1000
trajectories, two of which never share a ``session_id``, and at most 10 MiB.
The query parameter ``metadata`` is the JSON metadata of every trajectory of
the request; its ``kind`` is ``trajectory``:

.. code:: bash

   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/trajectories" \
     --get --data-urlencode 'metadata={"scenarios": {"my-agent": "train"}}' \
     -X POST \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/x-ndjson' \
     --data-binary @trajectories.jsonl

The response has status 201: ``{"owner", "items": [{"id", "sequence",
"digest", "metadata"}], "synced"}``. Reef stores all of the items in one transaction,
or none. An identical retry changes nothing. The same ID with other content
returns 409. A line that is not an ATIF document returns 400 and names the
line. ``POST /reef/namespaces/{owner}/data`` refuses a trajectory; it creates
record items only.

Before Reef stores the items, it checks each scenario that they name. The
scenario must have the owner, and its recipe must train on the items:

- A trajectory needs a recipe that accepts uploaded trajectories. Reef does
  not read the trajectories for this check. The recipe turns each one into a
  training sample when it reads it, and skips one that cannot train with a
  warning. The SFT recipe trains every agent turn, except the steps marked
  ``is_copied_context: true`` (`SFT
  <../user-guide/recipes/sft.rst#train-on-whole-conversations>`__). It skips
  a trajectory without such a turn. The training component renders the
  trajectory with the model's chat template when a job trains it, and leaves
  out one that cannot render.
- A task in the ``train`` split needs a recipe that trains imported tasks
  (``Sample2Job.accepts_imported_tasks``), such as ``sft``, the
  generated-task recipe, or harness evolution that runs offline (`Offline runs
  on uploaded tasks
  <../user-guide/evolve-your-harness.rst#offline-runs-on-uploaded-tasks>`__).
  The ``eval`` split takes tasks for any recipe,
  because an `artifact evaluation <#evaluate-an-artifact>`__ runs them.

A refusal returns 400, names the item, and stores nothing. Reef checks a
scenario that does not exist yet when the scenario opens.

Create task items
~~~~~~~~~~~~~~~~~

Send a gzip-compressed tar archive to ``POST
/reef/namespaces/{owner}/data/tasks`` with ``Content-Type:
application/gzip``. Each top-level entry is one Harbor task directory, named
by its task name: a lowercase letter or a digit, then lowercase letters,
digits, ``.``, ``_`` and ``-``. The task name is the item's ID. The query
parameter ``metadata`` is the JSON metadata of every task of the archive:

.. code:: bash

   tar -czf tasks.tar.gz -C /data/benchmark/tasks sum-391 sort-17
   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks" \
     --get --data-urlencode 'metadata={"scenarios": {"my-agent": "train"}}' \
     -X POST \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/gzip' \
     --data-binary @tasks.tar.gz

The response is the same as for trajectory items, in name order. Reef
returns:

- 400, and stores nothing, for an archive that is not tar.gz, a link or a
  device in it, a path outside its task directories, a file at the top level,
  a name that is not a task name, more than 1000 tasks or 10,000 entries, more
  than 256 MiB unpacked, a directory that Harbor's task loader refuses, or
  metadata that is not a JSON object of kind ``task``;
- 413 for a body larger than 256 MiB;
- 409, and stores nothing, for a task name that the owner holds with other
  files. To change the files of a task, replace them (below);
- 415 for another content type.

Replace a task's files
~~~~~~~~~~~~~~~~~~~~~~

Send the task's new directory, as a gzip-compressed tar archive of that one
directory, to ``PUT /reef/namespaces/{owner}/data/tasks/{id}`` with
``Content-Type: application/gzip``. The directory has the item's ID as its
name:

.. code:: bash

   tar -czf sum-391.tar.gz -C /data/benchmark/tasks sum-391
   curl -f "http://127.0.0.1:8900/reef/namespaces/my-agent/data/tasks/sum-391" \
     -X PUT \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/gzip' \
     --data-binary @sum-391.tar.gz

The item keeps its ID, metadata and splits. Each evaluation attempt reads
the task's current files, so a new evaluation and a retry of a started one
(``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/retry``) run
the new files. An attempt that already runs finishes with the old files, so one
evaluation can hold scores of both. A trainer of a scenario that uses the
task reads it again. The same files again change nothing.

The response is ``{"owner", "item", "synced"}``, without content. Reef
returns 400, and changes nothing, for an archive of no task or of several, a
directory with another name, a directory that Harbor's task loader refuses,
or an item that is not a task; 404 for an unknown item; 413 and 415 as for a
create.

Record items
~~~~~~~~~~~~

Reef makes a record item from each inference call and report of a scenario.
A training request gets no item. The item belongs to the scenario's
owner:

- Its ID is ``<scenario>/<agent_record_id>``.
- Its content is ``{"agent_record_id", "request_type", "payload",
  "created_at", "references", "artifact_id"}``: the record as the scenario
  stores it.
- Its metadata is ``{"kind": "record", "scenarios": {"<scenario>": "train"},
  "origin": {"scenario", "artifact_id"}}``. ``artifact_id`` names the artifact
  that answered the call, or is null when the record names none.


The data service makes the items after the records are stored, so an item
can arrive a short time after its record. When a scenario loads for the first
time after an upgrade, the data service also makes items from its older
records, in pages. The scenario serves while it copies them.

The scenario that made a record trains on its own record store, as before.
To train another scenario of the same owner on a record, add that scenario to
the item's ``scenarios``. Reef then reads the record from the data store for
the other scenario, with the item ID as its record ID, and it stores no copy.
Each reference ``x`` of a report reads as ``<origin scenario>/x``, so add a
report together with the inferences that it references. The record is not the
other scenario's own traffic: its recipe decides whether it trains on it. For
example, ``sft`` trains on it, and an on-policy recipe skips an inference that
another runtime load served.

When an item leaves a scenario, new reads of that scenario stop giving it. A
training set that a process already holds in memory can still train until it
restarts. A record store from before this release can hold a copy of an item,
and that copy stays. The scenario that made a record always uses its item: a
patch that removes it returns 400.

You can also create record items with ``POST /reef/namespaces/{owner}/data``,
for example records that you keep from an older system. The content must have
the keys above, ``request_type`` must be ``inference`` or ``report``, and the
ID must be ``<origin scenario>/<agent_record_id>``. Otherwise Reef returns 400.

Evaluation calls
~~~~~~~~~~~~~~~~

Reef also keeps each model call of an evaluation rollout as a record item of
the scenario's owner (`Model calls of a rollout`_). Such an item differs from
a scenario's own record item:

- Its ID is ``<scenario>/<agent_record_id>``, with the record ID
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
  content. A call kept before ``caller`` existed has no ``caller`` and no
  ``label``, and it is a call of the agent.

An evaluation call is the data of no scenario: a patch that adds a scenario to
its ``scenarios`` returns 400, so no scenario trains on it. The scenario's
records route does not list it. To read the calls of one evaluation, query
the data with ``kind=record``, ``group_key=evaluation_id`` and the
evaluation's ID as ``group_value``. An index on the owner and
``evaluation_id`` finds them. An evaluation call links only to the
earlier calls of the same attempt, so its trajectory node keeps only
the messages after its parent's. A verifier call links only to the earlier
verifier calls of the same attempt. A scenario's own call never links to an
evaluation call.

A trajectory that a verifier wrote is a trajectory item of the scenario's
owner. Its ID is
``<scenario>/<evaluation_id>:<task_index>:<repeat>:<attempt>:verifier:trajectory:<index>``,
and its content is the ATIF document. Its metadata is ``{"kind":
"trajectory", "scenarios": {}, "evaluation_id", "evaluation":
{"evaluation_id", "task_index", "repeat", "attempt", "index", "name"}}``,
where ``index`` counts the trajectory files of the attempt in name order and
``name`` is the file's name. It has no ``origin``, and a patch that adds a
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
Its ``source_uri`` comes from the artifact backend: a backend
that keeps snapshots where other deployments can read them, such as an object
store, gives the URI of the snapshot's directory. The built-in ``git_lfs``
backend gives none, so its items have a null ``source_uri``.

To add files that you stored, such as an upload to an S3 prefix, create an
item with ``POST /reef/namespaces/{owner}/data``, ``metadata`` ``{"kind":
"artifact"}`` and no ``origin``. Its content needs a ``source_uri`` and a null
``component``.

To use an artifact item in a scenario, import it:
``POST /reef/scenarios/{scenario}/artifacts`` with ``{"item": "<id>"}``. The
scenario must belong to the item's owner. Reef reads the item's
``source_uri`` and imports the files into the scenario; the import's ``source``
names the item. The artifact keeps its ID, so the ID names one artifact in each
scenario that uses it, and the library keeps one item of it. A second import of
the item returns that artifact, and it reads nothing. An upload gets its ID at
its first use, in ``metadata.artifact_id``, which never changes after that. An
item without a ``source_uri`` returns 400. A new scenario can also start from
an artifact item: see `Create a scenario`_.

Read, query, update and delete items
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``GET /reef/namespaces/{owner}/data?kind=&scenario=&benchmark=&group_key=&group_value=&after=&limit=&content=``
  lists the owner's items in the order they arrived: ``{"owner", "items",
  "next"}``. ``kind`` and ``scenario`` filter the items. ``benchmark`` keeps
  only the tasks of one benchmark. ``group_key`` keeps
  only the items of one group of the groups route below: those with the
  string ``group_value`` under ``group_key`` in their metadata, or with no
  string there when ``group_value`` is not given. ``group_value`` without
  ``group_key`` returns 400. ``after`` is the
  ``sequence`` of the last item of the previous page, 0 by default. ``limit``
  is 1 to 1000, 1000 by default. ``next`` is the ``after`` of the next page,
  or null after the last item. With ``content=1``, each item has its
  ``content``, as a read of the item gives it; the default ``content=0``
  leaves it out. Use it with ``kind=artifact`` to read the artifact library's
  kinds and sources in one page; a page of trajectories with content can be
  large.
- ``GET /reef/namespaces/{owner}/data/groups/{key}?kind=&scenario=&before=&limit=``
  groups the owner's items of ``kind`` by the string under ``key`` in their
  metadata, for example the trajectories of each uploaded file by
  ``set_id``: ``{"owner", "groups", "next"}``. Each group is ``{"value",
  "item_count", "first_sequence", "last_sequence", "metadata"}``. ``value`` is
  null for the items without a string under ``key``. ``metadata`` is that of
  the group's first item. The groups come newest first, by
  ``first_sequence``. ``kind`` is required. ``key`` is 1 to 64 letters, digits
  and underscores. ``scenario`` groups only the items that the scenario uses,
  and ``item_count``, ``first_sequence`` and ``last_sequence`` count only
  those items. ``before`` returns only the groups whose
  ``first_sequence`` is lower. ``limit`` is 1 to 1000, 1000 by default.
  ``next`` is the ``before`` of the next page, or null after the last group.
  To read the items of one group, query with ``group_key`` and its
  ``value``, and page with ``after`` from its ``first_sequence``. Give the
  same ``scenario`` to that query.
- ``GET /reef/namespaces/{owner}/data/{id}`` returns one item with its
  ``content``. A task's ``content`` is null.
- ``POST /reef/namespaces/{owner}/data/metadata`` takes ``{"ids": [...],
  "patch": {...}}``. Reef applies the JSON merge patch (RFC 7386) to the
  metadata of each item, all or none. A null value removes a key. For
  example, ``{"scenarios": {"my-agent": null}}`` removes the items from
  ``my-agent``, and ``{"scenarios": {"my-agent": "eval"}}`` moves tasks to the
  ``eval`` split. Reef checks the new uses as for a create call. A patch that
  changes ``kind``, ``owner`` or ``origin`` returns 400. The response is
  ``{"owner", "items", "synced"}``, without content.
- ``DELETE /reef/namespaces/{owner}/data/{id}`` deletes one item and returns
  ``{"owner", "id", "deleted": true, "synced"}``. The scenarios that used a
  trajectory or a task lose its copy. A record stays in the record store of the
  scenario that made it. The item of an artifact that a scenario made returns 400 while
  the scenario exists, because the next copy would make it again.

An unknown ID returns 404. A deployment without a data store returns 503.
An item that leaves a scenario keeps its train counts there, so ``auto`` does
not train it again if it comes back.

``GET /reef/scenarios/{scenario}/tasks`` returns ``{"scenario", "tasks":
[{"name", "split", "benchmark", "files_digest", "trained"}]}``: the uses of
tasks by the scenario, in the order that they began. ``benchmark`` is the held
benchmark that gives the use, or ``null`` for the task's own mark. A task
that the scenario uses more than once is listed once for each use. ``trained``
is true once a committed task job of any trainer of the scenario trained the
task.

Two GET routes come before the item routes, because an item ID can hold
``/``. A GET of ``.../data/<id>/archive`` downloads the item ``<id>``, and a
GET of ``.../data/imports/<x>`` or ``.../data/exports/<x>`` reads a transfer.
So GET cannot read an item whose ID ends in ``/archive``, or whose ID is
``imports/<x>`` or ``exports/<x>`` with ``<x>`` one path segment. Give such
items other IDs.

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
  collection has now, in that split (`A scenario's benchmarks`_). A benchmark
  that joins the collection later is not held. The response is ``{"owner",
  "name", "scenario", "split", "benchmarks", "task_count", "synced"}``:
  ``benchmarks`` lists the holds, and ``task_count`` counts a task of two
  benchmarks once. A collection without a benchmark returns 400.

An unknown collection returns 404.

A scenario's benchmarks
~~~~~~~~~~~~~~~~~~~~~~~

A scenario holds benchmarks, each in one split (RFC #1420). It then uses the
tasks of the benchmark in that split. A hold writes no task, so it takes the
same time for a benchmark of ten tasks and of ten thousand.

.. code:: bash

   curl -f -X PUT http://127.0.0.1:8900/reef/namespaces/my-agent/scenarios/my-agent/benchmarks/terminal-bench \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"split": "eval"}'

- ``PUT /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks/{benchmark}``
  takes ``{"split"}``, ``train`` or ``eval``. A hold that exists gets the
  split. The response is ``{"owner", "scenario", "benchmark", "split",
  "task_count", "synced"}``. A ``train`` hold needs a recipe that trains
  imported tasks, as a ``train`` mark does; else Reef returns 400.
- ``DELETE`` of the same path ends the hold: ``{"owner", "scenario",
  "benchmark", "released", "synced"}``. ``released`` is false when the scenario
  did not hold the benchmark.
- ``GET /reef/namespaces/{owner}/scenarios/{scenario}/benchmarks`` returns
  ``{"owner", "scenario", "benchmarks"}``: each hold with its ``split`` and its
  ``task_count``, in the order of the names.

The membership is live. A task that joins a held benchmark, by a create or by
a metadata patch, joins the scenario, and a task that leaves the benchmark
leaves it.

A scenario uses a task once for each held benchmark that the task is in, and
once more when the task's own ``scenarios`` metadata names the scenario. Reef
compares no task lists and refuses no overlap:

- A task in two held ``train`` benchmarks has two uses. A trainer reads the
  train uses as a set of tasks, so a job holds the task once, and the task has
  one train count.
- A task in a ``train`` benchmark and in an ``eval`` benchmark is trained, and
  an evaluation runs it. Keep such a task out of one of the two benchmarks
  when the evaluation must not see trained tasks.

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

The status is ``{"id", "kind", "state", "request", "result", "error"}``:

- ``kind`` is ``data_import`` or ``data_export``.
- ``state`` is ``queued``, ``assigned``, ``running``, ``finished`` or
  ``failed``.
- ``request`` is the body that started the transfer.
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
- ``trajectory``: each ``*.jsonl`` file at the root holds one ATIF document a
  line, as the upload takes them. Each document's ``session_id`` is its item
  ID; two documents of the import never share one. Reef stores the items in
  transactions of at most 1000, so an import that fails can leave the items of
  its earlier transactions. Reef reads no other file.

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
Authorization header. It keeps the request's ``path`` and ``component``
queries, and it opens only that download: another path, or another ``path`` or
``component`` query, returns 401. When the service no longer accepts the token that
signed it, the URL returns 401. A service without tokens returns the URL
without a signature, because the download needs none.

Choose the samples that train
-----------------------------

Every training sample of a scenario is a data item that it uses: an uploaded
trajectory, an imported task, or a trajectory or task that a trainer's stages
made (RFC #1140). ``GET /reef/scenarios/{scenario}/samples`` lists them in the
order that their uses began, at most ``limit`` (default and maximum 1000) after
the sequence ``after`` (default 0):

.. code:: json

   {"scenario": "my-agent", "next": null, "samples": [
     {"sample_id": "trajectory-0001", "sequence": 1, "kind": "trajectory", "split": "train",
      "component": null, "group": null, "benchmark": null, "train_counts": {}},
     {"sample_id": "my-agent.report-17:0", "sequence": 2, "kind": "trajectory", "split": "train",
      "component": "", "group": null, "benchmark": null, "train_counts": {"": 1}}]}

``component`` is the trainer that made the sample from records (``""`` for the
only trainer of a scenario), or ``null`` for an upload or an imported task.
Samples that share a ``group`` train together. ``benchmark`` is the held
benchmark that gives a task's use, or ``null``; a task that the scenario uses
more than once has one entry for each use. ``train_counts`` holds, for each
trainer, the number of committed jobs that trained the sample. ``next`` is the
``after`` of the next page, or ``null`` after the last one.

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

This version evaluates a model artifact (a ``model`` or ``model_adapter``
component) with a Harbor agent, on a runtime that serves any artifact on
request (``InferenceRuntime.serves_any_artifact``). The Tinker runtime is the
built-in runtime that does. Reef refuses a harness artifact, a model artifact
on a runtime without that capability, and a scenario whose artifacts also hold
a harness, with 400.

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
- ``harness``: the Harbor agent that runs a model artifact. ``{"name":
  "terminus-2"}`` runs Reef's default terminus-2 configuration. Any other
  agent is its whole Harbor configuration, with ``{model}``, ``{base_url}``
  and ``{api_key}`` where Reef fills them in, as the Tinker job parameter
  ``rollout.agent`` takes it. ``{"item": "<item id>"}`` runs a harness tree: a
  published artifact item of the scenario's owner, also from another scenario.
  The item has the kind ``harness_plugin``, and ``adapter`` in its metadata. Its
  adapter must install with npm, uv or git and call the runtime's API dialect. Each
  agent runs in the task's container. On ``docker``, the container reaches the
  model at ``evaluation.agent_host``. In a remote environment, such as
  ``modal``, the agent bridges the container's loopback to the model through
  Reef's sandbox relay and Harbor's exec, also for a task with several compose
  services (RFC #986). Reef refuses a task that declares MCP servers or skills. This
  evaluation does not reach the activation rule, and an automatic evaluation
  cannot use an item.
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
  Use it to score an artifact on part of a large benchmark. When more tasks
  are selected, Reef runs the ``num_samples`` tasks whose names have the
  smallest SHA-256 digests. The choice depends on the task names only, so
  evaluations of the same tasks with the same ``num_samples`` run the same
  tasks, and their scores compare. The stored request names the tasks in
  ``task_names``, and a stored ``collection`` leaves out a benchmark with no
  chosen task.
- ``sample_seed``: an integer, or null (the default). With a seed, Reef runs
  the tasks in a random order that the seed fixes: it ranks the SHA-256
  digests of ``"<sample_seed>:<task name>"``. With ``num_samples``, the
  evaluation runs the first ``num_samples`` tasks of that order, so a new
  random seed runs another random part of the tasks. Without ``num_samples``,
  it runs every task in that order. The same seed gives the same tasks in the
  same order again. The stored ``task_names`` are in the order that the
  evaluation runs them.
- ``rollouts_per_task``: the rollouts of each task, 1 by default.
- ``environment``: the Harbor environment type, ``docker`` by default.
- ``max_concurrent_rollouts``: the most rollouts of this evaluation that run
  at once, or null (the default) for no limit. Each rollout holds a task
  sandbox and sends model calls to the candidate, so set it to what the
  sandbox provider and the candidate's serving can take. Reef adds the next
  rollouts to the queue as earlier ones end.
- ``max_failed_fraction``: the largest fraction of failed rollouts, a number
  greater than 0 and less than 1, or null (the default) for no limit. A
  rollout fails when its work fails or when it ends with no score. After at
  least 10 rollouts end (or all of them, for a smaller evaluation), Reef stops
  the evaluation when more than this fraction of the ended rollouts failed. It
  fails with an error such as ``too many rollouts failed: 37 of 45 ended
  (82%), over the limit of 50%``. Each rollout that did not end fails with
  ``stopped: too many rollouts failed``, and the rollouts that wait never
  start (RFC #1287).

The answer is 202 with the stored evaluation, whose ``state`` is ``running``.
``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}`` returns it as
it runs and after it ends:

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

``state`` is ``queued``, ``running``, ``finished`` or ``failed``. A finished
evaluation holds its ``policy_evaluation``: the harness and the model that its
rollouts ran, and each rollout's score by task (see `Rollout
<glossary.rst#rollout>`__). A failed one holds its ``error``. An evaluation
that was queued or running when the service stopped reads as ``failed``; Reef
does not resume it. ``mean_score`` is the mean of the rollouts that have a
score: a failed rollout does not count as 0. ``GET
/reef/scenarios/{scenario}/evaluations`` lists the evaluations, newest first,
at most ``limit`` (50 by default).

For an evaluation on a collection, the stored ``collection`` is ``{"name",
"average", "benchmarks"}``. ``benchmarks`` gives ``{"benchmark",
"task_names"}`` for each benchmark that has a task in the evaluation.
``benchmark_scores`` then gives ``{"benchmark", "task_count",
"mean_score"}`` for each of them: the number of its tasks in the evaluation,
and the mean of their rollouts that have a score. A task in two benchmarks
counts in both. With the average ``task``, ``mean_score`` is the mean of all rollouts
that have a score. With ``benchmark``, it is the mean of the benchmark means.
``benchmark_scores`` is empty for another evaluation.

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
  while ``max_concurrent_rollouts`` keeps the rollout out of the queue.
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
``calls=false`` leaves out ``calls`` and ``verifier_calls``. Reef then reads
the rollouts only, so the read stays fast for an evaluation with many calls.
``calls`` is ``true`` or ``false``, and ``true`` when absent. Another value
returns 400. An unknown evaluation returns 404.

Reef returns:

- 400, and runs nothing, for a request that the scenario cannot run: an
  unknown field or task, the wrong side configured, a component that is no
  side of a policy, an artifact or a runtime that this version refuses, or no
  eval task;
- 404 for an unknown scenario, artifact or evaluation.

Reef sets no limit on the evaluations that run at once: the evaluations queue
holds the rollouts that wait. An evaluation process runs at most
``evaluation.max_sandboxes`` rollouts at once, across all evaluations (see
`Configuration <configuration.rst>`__).

``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/cancel``
cancels one evaluation. It answers the evaluation after the cancel, as ``GET
/reef/scenarios/{scenario}/evaluations/{evaluation_id}`` returns it:

- A queued or running evaluation fails with the error ``cancelled by
  request``. Each of its rollouts that did not end fails with the same error.
- A worker that runs one of these rollouts stops its trial and the trial's
  sandbox within seconds.
- The rollouts that wait for ``max_concurrent_rollouts`` never start. They
  stay ``waiting`` in the rollouts list.
- An evaluation that ended stays as it is, so a second cancel changes nothing.

An unknown scenario or evaluation returns 404. Control answers this route.

``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/retry`` runs
the rollouts of an evaluation that ended with no score once more. It answers
202 with the evaluation, which runs:

- A rollout has no score when it failed, when a cancel or a stop of the
  evaluation ended it, or when it finished with a failed score. Each of them
  goes back to the queue. Its earlier attempts stay, so its ``attempt_count``
  goes on and the calls of each attempt stay readable.
- A rollout that has a score keeps it. A rollout that has not ended stays as
  it is.
- An evaluation that runs goes on, with those rollouts in the queue again.
- An evaluation that ended is ``running`` again, with no result and no error.
  The rollouts that waited for ``max_concurrent_rollouts`` start as the
  others end.
- When the rollouts are final, Reef ends the evaluation as it ends a first
  run: it stores the scores, and it gives a finished evaluation to the
  scenario's activation rule, which can activate the artifact.

It returns 409 for an evaluation with no rollout to run again, and for an
ended one whose rollouts Reef no longer keeps: Reef deletes the rollouts of an
ended evaluation after their retention. An unknown scenario or evaluation
returns 404. Control answers this route.

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

Reef reads at most 32 MiB of these files and keeps at most 2000 calls for
each attempt. It skips a line that is no such call, a call over the limit, a
file that is no ATIF document, a file over the limit and a symbolic link, and
it logs how many it skipped. It reads the files for a failed verifier too.
The files never change the score. A task that sets the verifier's
``include_logs`` or ``exclude_logs`` must keep ``judge/``.

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
  body plus ``response``, the response body.
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
order. Add ``?caller=verifier`` for the verifier's trajectories. In each
attempt, these are the trajectory files that the verifier wrote, in name
order, and then the trajectories that Reef stitches from the verifier's
calls. A verifier trajectory has no reward. Each verifier file has
``extra.reef.evaluation`` as ``{evaluation_id, task, task_index, repeat,
attempt, caller, file, scored}``, where ``file`` is the name of its file and
``scored`` is false. Another ``caller`` value returns 400.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories``
streams the trajectories of the whole evaluation as ``application/x-ndjson``:
one trajectory on each line, in task and repeat order. It sends the agent's
trajectories only. By default it sends the last attempt of each rollout. Add ``?attempts=all`` to send every attempt.
Reef reads the calls of one rollout at a time, so the memory of an export does
not grow with the evaluation. A file within the upload limits uploads as it
is as trajectory items (`Create trajectory items <#create-trajectory-items>`__).

Both routes return 404 for an unknown evaluation. Another ``attempts`` value
returns 400.

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

Each training step that publishes a checkpoint is a new artifact. Reef then
stores an evaluation of it in the ``queued`` state and runs it when fewer than
two evaluations run. The evaluations routes list it like any other. An
automatic evaluation that cannot run, for example because the eval split holds
no task, is not stored; the service logs why.

Manual training
---------------

``POST /reef/train`` queues one training request for a scenario in
``data.training_mode: manual``; in harness evolution, every method takes it.
A request is not a record: it waits in the deployment's training request queue
until a training commit consumes it, and the commit keeps it in its
``training_request`` metrics. It takes the user's ``text``,
originating ``session`` and ``artifact_id``. These fields identify the session
and artifact the request came from. The backend operates on the current
committed state; it does not restore the originating artifact. The API
requires no inference receipts or score. ``samples`` names the samples that
the request's job trains (see `Choose the samples that train
<#choose-the-samples-that-train>`__). ``train_from`` names the artifact that
the job starts from, by artifact ID; without it the job starts from the
scenario's training start. A ``train_from`` that names no artifact of the
scenario is HTTP 400.

A job without ``train_from`` continues the training start and moves it to
each checkpoint that it saves, so the next such job begins where it ended;
Reef runs one such job at a time. A job with ``train_from`` never moves the
training start, and Reef does not wait for another job before it submits it.
It runs beside the other jobs when the training component allows it and a
training worker is free (``training.max-jobs``); the Tinker component allows
it. Requests start in the order they came, so a request that waits for the
running job also holds the requests after it.

A request may also carry ``requires``: what the change needs where the
harness runs, at most 8 ``{name, kind, check, prompt}`` items, default none.
``kind`` is one of these:

- ``permission``: an OS permission the person grants.
- ``env``: a variable the person sets. The extension reads it from the
  environment, and its value never enters a request or the tree.
- ``service``: an account or endpoint the person connects.
- ``binary``: a program the person installs. Its ``name`` is the program,
  looked for on the PATH.

``name`` matches ``^[A-Za-z0-9][A-Za-z0-9._-]*$``, and the pages list the
item under it. ``check`` is optional. For ``env`` it is the variable name;
for the other kinds it is a shell command whose exit status zero means
satisfied. For ``env`` the variable named (the check, else the name) is a
shell identifier, ``^[A-Za-z_][A-Za-z0-9_]*$``. ``prompt`` is optional: one
sentence telling the person what to enter or grant, at most 200 characters,
stripped and dropped when blank; the pages show it beside the item. The
credential and directive screens run over every ``name``, ``check`` and
``prompt`` as they run over ``text``, with the same HTTP 400 and a reason
that names the rule. A malformed list is HTTP 400 naming the first bad item.

The method may add items of its own (an extension that reads a variable,
say) in its step's result. The backend runs the same credential and directive screens
over each added item, and merges the ones that pass by name after the
person's items into the commit's ``training_request.requires``, so the
snapshots row and the manifest carry what the person named and what the change
added. An added item with a person's item's name is dropped: the person's copy
wins. ``training_request.refused_requires`` lists each item a screen refused
as ``{"item": <the item as written>, "reason": "..."}``; the key is absent
when nothing was refused. A step whose merged list holds more than 8 items is
skipped. Reef never runs a check: the
client runs the checks where the harness runs, after the person read them.

The two modes differ in what starts a job. ``auto``, the default, also
starts a job on its own: when ``job_samples`` samples wait, or when the
deployment's training manager says so (``training.manager``, see
`Configuration <configuration.rst>`__). ``manual`` starts a job only for a
request. In both modes a request trains, oldest first, one per step, and
before the samples that wait. The step trains the samples that the request
names, whatever ``job_samples``:

- A weight recipe needs ``samples``, and refuses a request without it
  with HTTP 400. Its ``text`` goes
  into the commit's ``training_request`` metrics.
- Harness evolution runs the request with the traces that it names,
  possibly none. The method reads the request next to them.

A weight recipe's request can set the parameters of its training job with
``parameters``, a JSON object of at most 16 KiB. Reef merges it into the
recipe's parameters for the kind of the job's samples: nested objects merge
key by key, and every other value of the request wins, lists included. A
request cannot remove a key that the recipe sets. Reef does not read them: the
training component checks them, as it checks the recipe's. A refused job fails
the request with the component's reason. For example, after an SFT job on
uploaded trajectories, run an RL job on uploaded train tasks:

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: coding" \
     -H "Content-Type: application/json" \
     -d '{"agent_record_id":"rl-001","text":"RL on the uploaded tasks","session":"session-1","artifact_id":"artifact-1",
          "samples":["task-a","task-b"],
          "parameters":{"objective":"my_methods.objectives:MyObjective","rollout":{"rollouts_per_task":4}}}'

The Tinker component takes ``objective``, ``learning_rate_schedule``,
``scheduling``, ``batch_size``, ``learning_rate``, ``kl_coef``, ``save_interval``
and ``rollout``; ``rollout`` only on a job of tasks. Slime takes ``objective``,
``learning_rate_schedule``, ``scheduling``, ``save_interval``, ``dataset`` and
``harbor``; ``harbor`` only on a job of tasks. ``GET
/reef/scenarios/{scenario}/contract`` serves the JSON Schema. A job with no
objective, from the recipe or the request, is refused.

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: coding" \
     -H "Content-Type: application/json" \
     -d '{"agent_record_id":"change-001","text":"Add a skill that runs tests before answering", "session":"session-1", "artifact_id":"artifact-1"}'

The response is ``{agent_record_id, scenario, request_type: "train"}``,
plus ``page_path``, the request's page with the query a browser opens it by
(see Request page).
An optional top-level ``agent_record_id`` names the request: an identical
resend returns the queued request, and the same ID with another body is
HTTP 409. HTTP 200 acknowledges durable acceptance, not successful training. Requests
are executed one at a time by the normal training worker; later requests
do not change a step already in flight. A step that fails with a
request (a method error, for one) is not retried: the next step
consumes the request with a committed row whose ``skipped`` reads
``request failed`` and whose ``error`` carries the failure, and the
queue moves on. Send the request again to run it again. That row
consumes the request alone: the units that rode beside it
stay held for the next job, so no failing trace is consumed unread. The
``evolution.max_steps`` and ``evolution.max_failure_streak`` budgets count
every step, request steps included, and stop automatic steps only; a
request still runs past them. The existing evaluation and publication
rules still determine whether the result becomes served.
``GET /reef/status`` reports ``training_mode``. Processors that take
requests also report ``pending_requests`` (accepted requests that
no commit consumed yet, read from the queue).
Committed step metrics include ``training_request`` with the request
id, text, session, artifact id and ``requires``; the snapshots row of a step
that answered a request carries, for example:

.. code:: json

   {"training_request": {"id": "change-001", "text": "Text me when the run is blocked",
                         "session": "session-1", "artifact_id": "artifact-1",
                         "requires": [{"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID",
                                       "prompt": "The Twilio account SID"}]}}

The method also says what its step did. The commit metrics keep its summary
as ``method_summary``, plain text, and its notes as ``method_notes``, the
JSON object it returned. Reef clips their long text and redacts
credential-shaped literals in them, and never reads them to decide. Each is
present only when the method returned it non-empty, and each method chooses
its own note keys.

An optional ``client`` reports the requesting machine, so a method builds
for it rather than for the sandbox it evaluates in: ``platform``,
``arch`` and ``release`` (short words) and ``commands``, a map of command
names to whether each is on the machine's PATH (at most 64). A client fills
``commands`` by looking each name up on the PATH; it runs nothing.
It only informs the method: what does not fit that shape is dropped, never
a reason to refuse the request, and ``training_request.client`` carries what
was kept.

Supply ``agent_record_id`` to retry safely: an identical request is accepted
without another step, including after capacity eviction; reusing the id with
different content returns HTTP 409. Without it, each submission gets a fresh
id. Empty text, text longer than 4000 characters, missing or non-string
session/artifact fields, or a request to an ``auto`` scenario returns HTTP 400.
Text that carries a credential shaped literal or an instruction override
phrasing is refused with HTTP 400 and a reason that names the rule, never
the text; nothing is stored. The scenario must exist: an unknown scenario
answers HTTP 404 and creates nothing, whatever the implicit-scenario-creation
setting says. The normal bearer authentication applies.

Scenarios
---------

A scenario isolates the records, trainer, and snapshot chain for one workload.
Create it with ``POST /reef/scenarios`` before you send inference requests to
it. An inference request never creates a scenario: for an unknown scenario, it
answers HTTP 404. The first report or imported record carrying a new
``x-reef-scenario`` also creates the scenario, and a harness install without
``x-reef-scenario`` creates one with a generated name. Reef creates every scenario with
the deployment's configured recipe. Requests never select a recipe.

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

The bindings never change; a request naming a different recipe with the same
scenario returns HTTP 409. This means the components, runtime, inference backend,
and optional report schema chosen when the recipe is constructed are fixed with
it.

A deployment can limit its scenarios. See `Limit the scenarios a service holds
<../user-guide/operate.rst#limit-the-scenarios-a-service-holds>`__. A request
that would create a scenario past ``reef.max_scenarios`` returns HTTP 409.
A request can also need to load a scenario when the service already holds
``reef.max_loaded_scenarios`` loaded scenarios. If none of them is idle, the
request returns HTTP 503 with a ``Retry-After`` header.

Create a scenario
~~~~~~~~~~~~~~~~~

``POST /reef/scenarios`` takes the object ``{"name", "owner"?,
"artifact_id"?, "component"?, "serves"?}``. Any other key is HTTP 400, and the
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
component mapping. With ``artifact_id``, Reef reads the item's files and checks
them as ``POST /reef/scenarios/{scenario}/artifacts`` with ``{"item": "<id>"}``
does (`Import an artifact`_), by reference or as a copy. Each other component
uses its default start. These are the initial active selections and training
starts. Their artifact rows have ``operation: creation``; ``source`` names
the selected item or default configuration. When Reef cannot read the item, or the
check refuses it, Reef registers no scenario. It answers with the status of
the refusal, such as 400 for a directory without the component's layout or 404
for an item that the owner does not store.

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

If the deployment sets ``reef.allow_implicit_scenario_creation: false``, a
report, a record import or a harness install for an unknown scenario also
returns HTTP 404, and you create the scenario first:

+---------------------------------------------+---------------------------------------------+
| Route                                       | Body and response                           |
+=============================================+=============================================+
| ``POST /reef/scenarios``                    | ``{"name", "owner"?, "artifact_id"?,        |
|                                             | "component"?, "serves"?}`` → ``{scenario,   |
|                                             | owner,                                      |
|                                             | components, artifact_id?}``;                |
|                                             | 201 created, 200 already existed            |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios``                     | every known scenario, its selections        |
|                                             | once loaded and, for a harness recipe, the  |
|                                             | ``adapter`` its tree is rendered for, also  |
|                                             | while the scenario loads                    |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract`` | ``{scenario, processor,                     |
|                                             | required_request_types, training_mode,      |
|                                             | status,                                     |
|                                             | parameters_schema}``;                       |
|                                             | ``parameters_schema`` maps a sample kind    |
|                                             | to the JSON Schema of a job's parameters,   |
|                                             | for each kind whose schema the training     |
|                                             | component declares                          |
+---------------------------------------------+---------------------------------------------+
| ``DELETE /reef/scenarios/{scenario}``       | → ``{scenario, archived}``; 404 unknown     |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios/{scenario}/jobs``     | → ``{scenario, jobs}``: the scenario's      |
|                                             | training jobs in the jobs queue, oldest     |
|                                             | first, each a job object (see below). Read  |
|                                             | only, from the data service; 404 unknown    |
+---------------------------------------------+---------------------------------------------+

A job object has these keys. A time is an ISO 8601 string in UTC, such as
``2026-10-31T08:15:00.250000+00:00``.

- ``id``: the job ID.
- ``state``: ``queued``, ``assigned``, ``running``, ``finished`` or
  ``failed``. The last two are final.
- ``attempt_count``: how many times a training worker took the job, an
  integer from 0.
- ``checkpoints``: the saved checkpoints that the job reported, in step
  order, each ``{step, path, artifact_path, metrics, changed_parts}``.
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
  then. A job that a store kept from before Reef recorded this time also has
  ``null``.
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

   {
     "id": "4f2a9c1e",
     "state": "finished",
     "attempt_count": 1,
     "checkpoints": [
       {"step": 1, "path": "/jobs/4f2a9c1e/step-1",
        "artifact_path": "/jobs/4f2a9c1e/step-1/artifact", "metrics": {"loss": 0.41},
        "changed_parts": []}
     ],
     "result": {"state": "finished", "metrics": {"loss": 0.41}},
     "error": null,
     "parameters": {"objective": "sft", "learning_rate": 1e-05},
     "sample_count": 8,
     "sample_kind": "trajectory",
     "request": {"id": "train-1", "text": "Prefer tests first",
                     "parameters": {"learning_rate": 1e-05}},
     "created_at": "2026-10-31T08:15:00.250000+00:00",
     "started_at": "2026-10-31T08:15:01.020000+00:00",
     "ended_at": "2026-10-31T08:21:44.900000+00:00",
     "experiment": {"tracker": "wandb", "project": "reef", "entity": "acme",
                    "run_id": "0d4c3b2a19f8e7d6c5b4a39281706f5e",
                    "url": "https://wandb.ai/acme/reef/runs/0d4c3b2a19f8e7d6c5b4a39281706f5e"}
   }

``POST /reef/scenarios/{scenario}/jobs/{job_id}/cancel`` cancels one training
job of the scenario. It answers ``{scenario, job}``, with the job as ``GET
/reef/scenarios/{scenario}/jobs`` lists it after the cancel:

- A queued job fails at once, with the error ``cancelled by request``.
- A running job fails the same way, and its training worker cancels the run
  at its next poll.
- A job that ended stays as it is, so a second cancel changes nothing.

The trainer reads the failed job and records its end. An unknown scenario, or
a job that the scenario does not have, returns 404. Control answers this
route.

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

An unknown scenario, or a request that the scenario does not have, returns
404. Control answers this route.

Deleting a scenario
-------------------

``DELETE /reef/scenarios/{scenario}`` removes the scenario from the running
service and frees its name. Nothing is destroyed: the record store and commit
log move under ``<agent_record_dir>/archived/``, the recipe's own directories
(a harness scenario's history and step records) move under an
``archived/`` sibling, and the
scenario's ref in the artifact repository is renamed into
``refs/reef/archived/``, so every snapshot it published stays reachable. A
later create of the same name writes a new root snapshot. A local step in flight for the scenario ends without a
commit. Only a local artifact repository can be archived; a remote one
answers 501.

For a scenario that trains weights, the delete also cancels the scenario's
running training job at its training component, since nothing would read the
job's checkpoints afterwards. On a runtime that serves per-scenario adapters,
the scenario's adapter stays resident in the serving engines until the
adapter residency evicts it.

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
changed. To change the data processor's mode:

.. code:: json

   {"training_mode": "manual"}

HTTP 200 returns ``{"scenario": "agents", "training_mode": "manual"}``.
The values are ``auto`` and ``manual``: ``"auto"`` starts a job whenever
``job_samples`` samples wait, and ``"manual"`` starts one only for a
request, on the samples that it names. The change selects subsequent job
data; job data already reserved or running completes in its original mode.
After a switch back, automatic jobs train the samples that no job trained,
including those collected while manual was selected. Accepted requests wait for manual, including
requests not yet read when the selector changes to auto.

A Reef process runs at most one scenario that trains full weights, on a single
thread, so preparation, remote execution, and commit never interleave. It may
run any number of scenarios that produce no updates or that update text
artifacts in process. Each one grows and commits on its own background thread,
so record acceptance never waits for artifact evolution.

Unknown scenarios return ``404`` without implicit creation; invalid payloads
return ``400``. A processor that does not support the requested mode returns
``501`` without changing its state.

``GET /reef/status`` reports the selected ``training_mode``. Reef stores the
mode with the scenario's settings, so a reload and a restart keep it.

The ``activation`` setting selects the rule that decides whether Reef
activates each new artifact of this scenario. It replaces the deployment's
`activation section <configuration.rst#the-activation-section>`__ for this
scenario. Send one of the built-in rules:

.. code:: json

   {"activation": {"module": "evaluated", "config": {"min_mean_score": 0.8}}}

The ``module`` is one of these rules:

- ``always``: ``config`` takes ``review_kinds``.
- ``never``: there is no ``config``.
- ``evaluated``: ``config`` takes ``min_mean_score``.

The response holds the stored rule. Send ``"activation": null`` to follow the
deployment's rule again. The setting names only built-in rules. Reef answers
HTTP 400 for a ``package.module:ClassName`` rule, because a client must not
choose the code that the service imports. Set such a rule in the deployment's
``activation`` section.

The ``evaluated`` rule activates an artifact only after an evaluation of it.
Thus it needs ``artifact_evaluation``. Reef checks the two settings together,
and stores both or neither. Reef answers HTTP 400 when, after the request, the
rule that applies is ``evaluated`` and the scenario has no
``artifact_evaluation``. Thus a request can set both at once, or set
``artifact_evaluation`` first. Reef also refuses a request that clears
``artifact_evaluation`` under ``evaluated``.

A new rule applies to the artifacts that training makes after the change. It
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
runtime and its request handler define the fields and check their values. Reef
answers HTTP 400 and stores nothing for a field that they do not have, for a
value that a field does not take, or for another component. The token-native
chat handlers have ``temperature``, ``top_p`` and ``top_k``
(`native backend options <configuration.rst#native-backend-options>`__). A runtime can
add engine fields, such as a number of replicas.

Inference applies a change within a few seconds. Each inference replica sends
the request fields with each later request. The runtime applies its engine
fields to the engine that serves the scenario and to each engine that starts
for it later. A field that takes effect only when an engine starts, such as a
GPU type, stays in the deployment configuration. Reef stores the config with
the scenario's settings, so a reload and a restart keep it. A stored field that
the deployment no longer accepts is ignored with a warning, and the scenario
serves with the deployment's values until a person changes the config.

Record-to-sample config
~~~~~~~~~~~~~~~~~~~~~~~

The ``record2sample_config`` setting gives one scenario its own config for the
record-to-sample stage of each trainer. It maps a component name to config
keys. Each key replaces the same key of the deployment's
`pipeline.record2sample.config <configuration.rst#the-pipeline-section>`__ for
the trainer of that component. The stage class stays the deployment's:

.. code:: json

   {"record2sample_config": {"weights": {"max_staleness": 2}}}

The request replaces the stored mapping as a whole. Send ``{}`` to use the
deployment's config again. The response holds the stored
``record2sample_config``.

Reef applies a change by loading the scenario again, as a restart does. Each
trainer then builds its stage with the new config and reads the records after
its committed progress again. A samples service in another process loads the
scenario again before its next selection. Reef answers HTTP 400 and keeps the
stored setting and the loaded scenario when the mapping names a component that
has no trainer, or when a stage refuses its config. Reef stores the setting
with the scenario's settings, so a restart keeps it.

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
artifact, the staleness check does not refuse its records, whatever its weight
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
     "record2sample_config": {"weights": {"max_staleness": 2}},
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
active artifact alias or an artifact selected for serving beside the active
one. An unknown name returns 404. One startup covers the active artifact and
each artifact beside it, but an artifact with a published route stays
``ready`` while another artifact starts or fails. To read the state of an
artifact beside the active one, name it:

.. code:: text

   GET /reef/scenarios/{scenario}/inference/status?model=<artifact-id>

``state`` is one of:

* ``pending``: no worker has taken the startup work.
* ``starting``: inference has taken it. Downloads, conversion, loading, and
  retries are part of this state.
* ``ready``: startup succeeded and inference published the route.
* ``failed``: startup ended with an error. ``error`` gives a safe message:
  the message of a runtime's ``StartupFailedError`` (RFC #1299), such as an engine
  argument that the model refuses, else a fixed text. The service logs hold the
  diagnostic details. Control retries a failed startup by itself: 60 seconds
  after the failure, and then after a wait that doubles at each failed retry,
  up to one hour. A retry returns the state to ``pending``. Resend the complete
  accepted ``served`` mapping to retry at once.

A newer request, such as one for a changed serving config, replaces the
current one at once. The work of the replaced request stops within seconds
when the runtime reads ``reef.inference.startup.stopping()`` (RFC #1299).
Requests of different scenarios start one after another, or at the same time
when the runtime has more than one ``serving_slots``.

``POST /reef/scenarios/{scenario}/redeploy`` deploys the engines of what the
scenario serves again, also when they run its current settings, for example
after a hang. A new request with the same selection replaces the current one,
and the runtime reads ``reef.inference.startup.redeploying()``. It answers
``{scenario}``, also for a scenario that serves nothing.

``error`` is null in every other state.

``engine`` is the state of the engine behind the route of a ``ready`` artifact
(RFC #1269), when the deployment's inference handler reports it:

* ``warm``: an engine answers now.
* ``starting``: an engine runs, but it does not answer yet.
* ``idle``: no engine runs. The next request starts one and waits for it.

``engine`` is null in every other state, and when the handler cannot tell.
While ``engine`` is ``starting``, ``engine_detail`` can say what the engine
does now, as one short line, such as ``Loading weights · 0:42``. It is null
otherwise.

``startup`` gives the artifact's startup in three fixed phases (RFC #1283), so
that a client shows one step bar for every backend:

* ``prepare``: the serving work waits, deploys the engine, and converts or
  quantizes the weights.
* ``start``: a machine or container is found, and the weights reach it.
* ``load``: the weights go into GPU memory, and kernels, CUDA graphs and
  warm-up run.

It is ``{steps, estimate_seconds}``. Each step is ``{phase, started_at,
ended_at, detail}``, in the order of the phases, with times in epoch seconds.
Only the last step can have ``ended_at`` null: it runs now. ``detail`` is a
short note of what it does, or null. ``estimate_seconds`` is a guess of the
seconds until the engine is warm, or null. While the state is ``pending`` or
``starting``, the steps are ``prepare`` from the startup request, then the
phases that the runtime reported for this artifact. While a ready artifact's
engine is ``starting``, they are the phases that the inference handler reports.
``startup`` is null otherwise. A phase that the backend does not report has no
step.
Reading it starts no engine.

The status describes the selected
artifact, even while an older route still serves requests. It is a startup
result, not a continuous health check. Reading it starts no work.
No startup work returns 404. An unreadable state store returns 503.
The route requires the same bearer token as inference.

Without a usable route, a pending or starting artifact returns HTTP 503
with error type ``inference_starting`` and ``Retry-After: 30``.
A failed startup returns HTTP 503 with type ``inference_start_failed``.
Both errors use ``{"error": {"type", "message"}}``.
Control and data operations remain available during startup.

Request
~~~~~~~

Send the same body you would send to the provider. Reef never touches your
sampling parameters, and no artifact adds to the request.

Reef answers the request from the scenario's row in ``served_models``. The
request's ``model`` selects the row of a served artifact when it names one;
any other ``model`` selects the active row, or the row that a `traffic split
<#traffic-split>`__ gives the request. Reef
keeps that row for the whole request. It sends the request to the engine at
the row's URL under the row's model name, and records the snapshot of the row's
artifact, even if an activation completes mid-request. `Run the services in
separate processes
<../user-guide/operate.rst#run-the-services-in-separate-processes>`__ gives the
status codes of an inference request.

To require a specific artifact, use
``/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/chat/completions``.

Reef retries a request once when the engine aborts a non-streaming answer, or
refuses the row's model name before any output. It reads the row again for the
retry. Three kinds of request keep their artifact: a request to an artifact's
route, a request whose ``model`` names its artifact, and a request of a
traffic split. When no row serves that artifact at the retry, Reef answers
HTTP 503 with ``Retry-After``. The retry of any other request can reach a new
active artifact.

On a weight-serving deployment, the engine handler selects the served adapter
itself, and it replaces a ``lora_path`` that the body names. Set
``"stream": true`` and read the SSE response for streaming.

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
  alternatives. ``/v1/messages`` returns no log probabilities.

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

An inference response from a served Reef artifact carries ``x-reef-artifact-id``,
the ID of the artifact that answered. Routed external models have no such header.
A client learns
of a new harness artifact from the harness routes (`Harness artifacts`_).

A composite recipe's evaluation calls come through these routes; a recipe of
one component calls its runtime's endpoint directly, as before composites.
The evaluation calls do not present a service token. When auth is configured,
the service issues a random token at startup that opens the evaluation routes
alone, and hands it to the recipe. The evaluation runs, which run candidate code, and
the method's calls to the served model use it: they can sample the active
selection but reach no other route. A restart issues a new one.

Artifact of a recorded response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An inference record holds the provider exchange and ``artifact_id``, the
artifact that answered: the same ID as ``x-reef-artifact-id``. The staleness
check reads it (see `Staleness
<../developer-guide/processors.rst#staleness>`__). Serving returns no training
tensors. Reef removes any ``training`` block that an engine handler returns
from the record, from the client's answer, and from rollout and evaluation
answers. The training component renders each recorded call and computes the
log probabilities that it needs.

Report
------

Reef stores four fields and drops other top-level keys. Put harness-specific
data inside ``metadata`` or ``feedback``.

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

The optional top-level ``agent_record_id`` makes posting retry-safe: Reef uses
it as the report's own record id, so an identical resend returns the stored
record instead of reprocessing it, while the same id with different content is
HTTP 409. It is not a receipt. Receipts go in ``references``.

A recipe may declare a report schema. `tttd
<../user-guide/recipes/tttd.rst#the-report-contract>`__ declares one. In that case, Reef
validates the declared ``score`` and ``metadata`` fields at ingress and answers
HTTP 400 on a violation; ``feedback`` and undeclared ``metadata`` keys pass
through, except the removed ``metadata.training.eligible`` field, which is
rejected with HTTP 400 regardless of its value. Reports cannot opt out of training.

Every supplied reference must identify an already stored inference record in the
same scenario, and no reference may lie inside the trajectory of another: a
report references only the last call of each trajectory. Missing, foreign,
non-inference, duplicate, or inner references produce HTTP 400 without
persisting the report. Reef does not wait for future records;
the producer may resubmit after the referenced inference has been stored.
Identical retries of previously accepted reports retain their original behavior
after reference cleanup; they do not restart training.

Record
------

Reef stores every exchange of inference and every report as ``AgentRecord``.
These records are referenced via a record id. In ``/reef/train`` and
``/reef/report`` API calls, users can provide an optional ``agent_record_id``
field for retry-safety. Appending the same id with different content returns a
conflict rather than overwriting. Reef keeps track of consumed records so
retried reports and late reports whose references already trained are not
counted twice.

Training records remain readable after consumption. Retention deletes the oldest
record items only when the shared capacity budget is exceeded (20 GiB by
default), and keeps durable loss totals. Audit endpoints expose retained requests,
responses and references. Per-step ``consumed_ids`` records processed
inputs, including intentional skips, rather than proving a model update.
See `Configuration <configuration.rst>`__ and `Python API <python-api.rst>`__.


Receiving an update
-------------------

For weight-training scenarios there is nothing to do: keep calling the same
inference endpoint and it serves the active selection. The artifact
lives in the inference runtime, so requests reach the new snapshot directly.

A scenario whose artifact is the harness tree works the other way: the client
pulls. It fetches the tree of the active selection with ``GET /reef/harness``,
writes the files where its agent reads them, binds the agent's model to this
Reef, and runs the agent on that snapshot. The snapshot it runs is then the one
whose receipts it reports against. ``reef-client harness`` is that client (see
`Use a published tree
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

``?form=`` chooses what ``files`` holds, path to text:

- ``rendered`` (the default): the snapshot rendered for the scenario's
  adapter. These are the paths the agent reads, with the adapter's enforced
  config defaults merged in (see `Rendering
  <../developer-guide/harness-adapters.rst#rendering>`__).
- ``source``: the source tree the snapshot stores. These are the tree paths
  that the adapter's parts take, the ones a patch names (see `Tree parts
  <../developer-guide/harness-adapters.rst#tree-parts>`__). A client that
  renders the tree itself reads this form.
- ``client``: the snapshot as a layer over a person's own configuration,
  for an install at ``root``, bound to
  ``<service_url>/reef/scenarios/<scenario>``. Both query parameters are
  required: ``service_url`` is the address the client reaches Reef at, and
  ``root`` is the directory the client installs the files in. The binding
  asks for the served model, and its key is the literal
  ``REEF_TOKEN_PLACEHOLDER``, which the client replaces with its own token.
  This form adds ``run``, the command that runs the agent on the files
  (``binary``, ``args`` and ``env``; ``env`` may hold the placeholder too),
  the agent's own commands that record the install and remove it
  (``register`` and ``unregister``, each a list of argument lists, empty for
  most agents), the agent's pinned ``install``,
  and ``synced``, the render templates of the files the client sends back,
  ``{name}`` standing for one path segment. An adapter without a ``layer``
  is HTTP 400.

Another value is HTTP 400. In the other forms, a config file may hold
``{root}``: write the directory that holds the render roots there. The first two forms carry no endpoint and no
credential: the client writes its own model binding where its agent reads
it. A new scenario's starting artifact holds ``evolution.seed``, so a
client can pull before any step publishes. A recipe whose seed is empty
serves HTTP 404 until a step publishes a tree.

``requires`` describes what the harness needs where it runs. It merges the
``training_request.requires`` items from the artifact's training history by
name, with the newest definition winning. A later request that adds no items
does not remove earlier requirements. One request's cap of 8 does not bound
the combined list. Reef runs no check: the client checks each item before it
runs the artifact. `Manual training <#manual-training>`__ gives the shape.

Use ``?artifact_id=`` to request a specific harness artifact, in any form.
An unknown or unrestorable artifact returns HTTP 404.

Catalog and manifest reads do not wait for an evolve step's method or
evaluation runs: they continue serving the existing artifacts while a
candidate is being prepared. Reads still serialize with publication and
activation so a manifest's artifact and evaluation metrics come from the same
artifact: reads do not interleave with a change of the active selection and its
commit-history update.

Importing a harness
~~~~~~~~~~~~~~~~~~~

``POST /reef/harness/import`` commits the harness files of a person's agent
as a sync, with ``x-reef-scenario``. A sync is a commit operation of its own:
the merged tree becomes a new artifact. That operation selects the artifact
for serving and as the harness trainer's start.

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

A client submits a person's request for a harness change through
``POST /reef/train``, described under `Manual training <#manual-training>`__.
Set ``data.training_mode: manual``, or switch an existing scenario with
``POST /reef/scenarios/{scenario}/update``. The client sends ``text``,
``session`` and the ``artifact_id`` it runs. HTTP 200 acknowledges durable
acceptance and returns ``agent_record_id`` and ``page_path``. A request needs
no inference receipts, so the receipts a client captured stay free for
feedback. A scenario in ``auto`` refuses the request; the client shows that
error and leaves the mode switch to the person.

The trainer runs the method's step on the request, then evaluates and
publishes under the same policy as automatic evolution. Its commit metrics
carry ``training_request: {id, text, session, artifact_id, requires}`` (with
``client`` when the request sent one) and the method's ``method_summary`` and
``method_notes``, all visible through the snapshot catalog.

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
including ``source.item`` when importing an artifact-library item; importing
from another scenario does not create a cross-scenario parent link.

The list holds artifacts only. A training step that made no artifact, such
as a rejected or skipped step, a job submission or a job's end, has no row;
``GET /reef/scenarios/{scenario}/operations`` holds its record (see `Record and
commit history`_). Activations, train-from operations and serves have no rows
either.

``GET /reef/scenarios/{scenario}/artifacts/{artifact_id}/files`` answers
``{scenario, artifact_id, files}``: each file of the artifact as ``{path,
bytes, url, url_expires_at}``, in path order.

- When the repository backend signs URLs of its files, ``url`` opens the file
  in the backend's storage, without a Reef token, until ``url_expires_at``
  (Unix seconds). Reef then lists the files without copying the artifact to
  its own disk. ``?expires_in=`` sets the life of these URLs in seconds: 1 to
  86400, 3600 by default. Download a large artifact through these URLs, so
  that its bytes do not pass through Reef.
- With a backend that signs no URLs, such as ``git_lfs``, both fields are
  null, and the data service serves the files itself.

With ``?path=<path>``, the route downloads that file. In a
scenario of several components, name the component with ``?component=``. An
artifact that the scenario does not list, or a path that is not one of its
files, returns 404. ``?signed=1`` returns a `signed download URL
<#signed-download-urls>`__ of the list or the file. The data service answers
this route.

Serving settings
~~~~~~~~~~~~~~~~

An artifact holds only its content. It does not say which model serves it or
how (RFC #1263). A deployment that serves artifacts of several base models
gives each scenario a `serving config <#serving-config>`__ field that chooses
the model of each artifact. The deployment's inference runtime defines that
field and checks it. An artifact without a setting is served with the
deployment's model.

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

The answer is HTTP 201 with ``{scenario, component, artifact_id}``.
Each import creates a new artifact, even when another artifact contains the
same files. Storage can reuse equal content without changing either ID.

An import of ``hf://org/model@<commit>`` with a full commit SHA into a
component of model weights stores only a reference, ``hf-reference.json``.
Any other component, such as a harness, stores a copy, because its readers
read the files themselves. Reef checks the layout from the revision's
file list and its small files, and takes each weight file's SHA-256 from the
Hub, so it downloads no weights. The repo must stay on the Hub while Reef serves or trains
the artifact. A private repo needs a token that the service reads in Hugging
Face's standard way, such as ``HF_TOKEN``. A revision that does not resolve for
that token answers HTTP 400.

The directory must have the layout of the snapshots that the scenario's
component makes. The training backend and its configuration decide it:

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

A weight component of the Slime or the Tinker backend also takes a base root:
a directory that holds ``base.json`` alone. The file names a deployment's
default start by its inputs, for example
``{"format": "reef-base-root-1", "inputs": {"model": "/models/qwen", "lora_rank": "8"}}``,
and the artifact stands for the base model with no adapter (`Artifact imports
<python-api.rst#artifact-imports>`__). Reef accepts it only when the inputs
equal the inputs of this deployment's default start, which the ``source`` of
a scenario's ``creation`` row shows. Another base model, rank or service
answers HTTP 400 and names each input that differs. Serving such an artifact
serves the base model, and training from it starts at the base model. A new
scenario of these backends starts from a base root that Reef writes
(`Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__).

A component without a layout, such as the records of a trainer that serves
nothing, takes no import.

Reef stores the new artifact and appends an ``import`` operation. Serving
selections and training starts stay unchanged, and the activation rule does
not run. Serve the artifact or train from it as from any other artifact.
The answer comes after the artifact is durable, so give a large import a
long client timeout.

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

Pulling an older artifact changes only your local copy. A scenario serves one
or more artifacts of each component, and one of them is the active artifact.
``GET /reef/harness`` serves the active harness tree. The default inference
routes answer from the active artifact that completed startup.

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
new routes when the startup completes. Poll `Startup status`_ for ``ready``
or ``failed`` before sending requests that require a new artifact; name an
artifact that serves beside the active one with ``?model=<artifact-id>``. A
startup can fail after Reef accepted the mapping, for example when a new
checkpoint finds no free inference component. A repeated request retries a
failed startup at once without writing another operation, and sets the wait
of the automatic retries back to 60 seconds.

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

In a scenario with several components, an artifact served beside the active
one stays bound to the artifacts of the other components that were active
when it started to serve. A later activation does not change that binding.

The activation rule and a restore change the active artifact alone. The
artifacts served beside it stay served. Training continues from each
trainer's independently selected training start.

The staleness check of a recipe does not refuse a sample while the scenario
serves the artifact that produced it beside the active one. After that
artifact stops serving, the check counts only the artifacts that the scenario
served as active: it refuses a sample whose artifact was not active within
``max_staleness`` artifacts.

The runtime limits how many weight artifacts a scenario serves together
(`Runtime limits <../user-guide/operate.rst#set-what-a-scenario-serves>`__).
A scenario whose components load nothing into a runtime, such as a harness
scenario, can serve any of its artifacts together. A store without a durable
commit history refuses a mapping of several artifacts.

Restore a scenario selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To restore every component's active artifact together, send
``POST /reef/scenarios/{scenario}/restore`` with ``{"commit_number": 12}``.
The commit number names an existing operation. Reef restores the active selection
as it stood after that commit. Use ``{"commit_number": 0}`` for the starting artifacts.

The answer is ``{scenario, components}``, where ``components`` maps every
component name to its restored ``artifact_id``. A changed selection appends
a ``restore`` operation. It creates no artifact or repository revision and
leaves each trainer's start, the other served artifacts and the traffic
split unchanged.
Restoring the current selection is a no-op. Poll `Startup status`_ before
using the restored inference artifact.

The body must contain only a non-negative integer ``commit_number``. An invalid or
unknown commit number, or one whose selection cannot be reconstructed from the
retained history, returns HTTP 400; an unknown scenario returns 404. A pending
training commit returns 409. Artifact validation and runtime capabilities
still apply.

Train from an artifact
~~~~~~~~~~~~~~~~~~~~~~

Each trainer has a training start: the artifact that its next training job
(a weight trainer) or evolve step (a harness trainer) starts from. A new
trainer starts from its component's creation artifact. After each job or step, the
trainer moves its training start to its own newest output.
A harness sync (`Importing a harness`_) also moves the harness trainer's
training start, to the artifact that the sync made.
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

The deployment's activation rule can leave a new tree inactive, for example
a tree whose patch changes a file of a kind listed in the ``always`` rule's
``review_kinds``, or of a kind that is always reviewed (native's
``native_loop``). The step creates an artifact but does not activate it. Its row carries ``active: false`` and
``activated: false``. The manifest keeps serving the active selection, and
``?artifact_id=`` pulls the inactive tree for a trial run. To make it serve,
send ``{"served": {"<artifact_id>": 100}}`` with the row's ``artifact_id`` to
``POST /reef/scenarios/{scenario}/update`` (`Set what a scenario serves`_).

``GET /reef/harness/artifacts`` lists only the harness component's artifacts.
A weights update does not create another harness artifact. Each row's
``parent_artifact_id`` identifies the artifact from which it derives, when known.
Every listed artifact can be requested through ``?artifact_id=``.

Version page
~~~~~~~~~~~~

``GET /reef/harness/artifacts/{step}/page`` answers one self-contained HTML
page (``text/html``) for one catalog row. ``step`` is the row's position in
``GET /reef/harness/artifacts``, oldest first, with the creation row as 0. A
rejected step creates no artifact. The step number therefore identifies the
page independently of its artifact.

The page shows why the step ran (the request, or a failure in the job data),
``method_summary``, the patch against the artifact the step ran on,
the evaluation's numbers, ``method_notes`` and ``requires``. It also links
the parent step and later training or sync operations. The selected artifact's
page reads ``Active selection``. A step that made an artifact that was never activated
(``activated: false``) reads ``Waiting for activation``. The row itself is in
a ``<script type="application/json">`` block at the end of the page. A step
outside the catalog, or one that is not a number of at most nine digits, is
HTTP 404.

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: code-repair" \
     "$REEF_URL/reef/harness/artifacts/3/page" > harness-step-3.html

Request page
~~~~~~~~~~~~

``GET /reef/harness/requests/{record_id}/page`` answers one self-contained
HTML page (``text/html``, ``Cache-Control: no-store``) for a harness request
filed through ``POST /reef/train``; ``record_id`` is the ``agent_record_id``
that the request answered, and its ``page_path`` links this page. Until a
step settles the request, the page reloads every five seconds and shows the
request's state and the step's activity (its latest 80 lines). The catalog row
whose ``metrics.training_request.id`` is the record id settles the page: it
then shows the result, the next action (``GET /reef/harness`` for an active
selection, the update request that serves an inactive one), ``method_summary``, the
patch with each file's kind of the harness tree (from the row's
``patch_parts``), and ``method_notes``. An unknown id, or one that is not a
training request, is HTTP 404.

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

The path holds a page key only when the request presented a service token;
an evaluation token gets none, and with authentication off the query names
the scenario alone. The key is an HMAC SHA-256 of the scenario, keyed by the
token's digest. It opens these two pages of that one scenario and no other
route, and the token cannot be read back from it, so a client may print the
links where a session's model reads them. A request whose
``x-reef-scenario`` header names another scenario is HTTP 401. The token
itself is never read from a query: ``?token=`` is HTTP 401 on every route, and
``?key=`` is HTTP 401 on every route but the two pages.

Retained step files
~~~~~~~~~~~~~~~~~~~

``GET /reef/harness/artifacts/{step}/records`` returns the raw file inventory
of the same catalog step's record directory, authenticated and
scenario-scoped like the version page. The response is
``{"status": "retained", "files": [{"path": "calls.json", "bytes": 123}]}``.
Add ``?path=calls.json`` (or another inventory path, such as ``output/notes.json``
or ``material/traces/0/trace.json``) to read
``{"status": "retained", "path": "...", "text": "..."}``. The service does not
interpret model replies or agent events. A console can render those persisted
formats without changing their learning semantics. The evaluation runs
are in the scenario's history, not in the step directory, so this route does
not serve them.

The backend reads only the step directory referenced by the selected catalog
row under the scenario's ``steps/`` directory. Absolute paths, traversal and
symlinks are rejected. Only JSON and JSONL files are exposed; inventories are
limited to 1000 filesystem entries and individual files to 4 MiB. Exceeding
these limits returns HTTP 400 instead of silently truncating records. A missing
file or catalog step is HTTP 404. A step whose row names no record directory
returns ``status: not_recorded``, and an absent directory returns
``status: missing``. These states carry an empty files list. All successful
reads use ``Cache-Control: no-store``.

``calls.json`` holds the recorded messages and replies of the method's model
calls. A run's native session log, in the history, retains request
headers, assistant messages and tool events; reconstructed inputs are not
exact provider request bodies. It does not retain provider response IDs, and a
compaction event may prevent complete input reconstruction. Reads neither copy
records into another store nor change retention. Step files remain separate
from online record-body capacity eviction.

Status
------

Read ``GET /reef/status`` when inference is still serving an older snapshot while
a training job runs or an update is being published.

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

``error`` and ``preload_errors`` report asynchronous training and preload
failures. ``loaded_scenarios`` counts the scenarios loaded in this process.
``served_artifact_id`` is the ``artifact_id`` of the artifact that answers
the scenario's requests now, the one that ``served_models`` names: the active
artifact of the weight trainer's component, or of the only component.
It is null before inference publishes a route.
``job_data_ready`` says whether the processor has a job's data waiting.
``artifacts`` gives two ``artifact_id`` values for each component: its
active artifact and its trainer's training start. A value is ``null`` when
Reef does not know it. Both values start at the creation artifact.

``processor`` includes ``ready_samples``, the untrained samples that the next
automatic job takes, from recorded traffic and uploaded trajectories;
``job_samples``, the samples that automatic job data needs
(``pipeline.sample2job.config.job_samples``); and ``buffer_size``, the most
samples that one automatic job takes, beside the rest of a group.

A scenario that trains weights also reports its training jobs.
``training_jobs`` lists the jobs that the scenario's training component runs,
in the order of their submission. Each holds the ``job_id``, the ``backend``
that runs the job, the ``start`` that the job trains from,
``checkpoints_handled`` and ``train_from``. ``start`` is a checkpoint's path
as the component reads it, or ``snapshot:`` and a JSON object
``{revision, root}`` that names the snapshot of the scenario that the job
starts from. ``checkpoints_handled``
is the number of the job's saved checkpoints that Reef has committed.
``train_from`` is the artifact that the job's request named as its start, or
``null`` for a job that continues the training start. The list is empty while
no job runs. ``finished_jobs`` counts the jobs of the scenario
that ended, finished or failed. A runtime that serves each scenario its own
adapter adds ``adapter_runtime_load_id``. It is the runtime load ID of the
scenario's served adapter, or ``null`` while the scenario samples the base
model.

A scenario whose recipe runs one trainer per recipe component adds a
``components`` object. For each recipe component, it holds the component's
own ``job_data_ready``, ``training_mode``, ``processor`` status and
``last_training_commit``. The scenario-wide ``job_data_ready``, ``training_mode``
and ``processor`` describe the first trainer that trains, else the first
trainer. The scenario-wide ``last_training_commit`` is the latest training commit of
any recipe component.

Every row of ``GET /reef/scenarios/{scenario}/artifacts`` names the
``component`` whose artifact it is.

``last_training_commit`` reports the commit number of the latest durable training commit,
its commit time, and its recipe-owned metrics; it is ``null`` before the first
training commit. This distinguishes a step still in flight from a completed
step that skipped or rejected its candidate. An activation, a train-from, an
import or a sync advances ``commit_number`` without replacing the latest
training outcome. ``commit_number`` is the number of the scenario's newest commit.

``serving`` is runtime-wide and keyed by recipe name. The runtime of a
weight-training recipe that serves per-scenario adapters reports the engines'
shared adapter residency there, under ``adapters``. For a composite recipe,
``adapters`` is under the name of the weight component.

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

Restart the service after changing the configuration. Without this option, Reef
does not add CORS headers. With it, unlisted browser origins are rejected before
a route runs. CORS preflight requests from listed origins do not need a service
token; actual requests retain the configured Bearer authentication. Browser
requests may use GET, POST or DELETE with Authorization, Content-Type and
x-reef-scenario headers. Cookies are not enabled through CORS.

CORS headers also cover errors and streamed responses, exposing
``x-reef-agent-record-id`` and ``x-reef-artifact-id`` to the browser. Clients without an Origin header keep their existing behavior.
A browser can also require local network permission or HTTPS. An origin that
Reef allows does not override browser policy.

A connected console acts with the existing permissions of the service token.
Its requests operate on the scenarios of this service. The CORS connection
does not create a cloud deployment or upload history.

Record and commit history
-------------------------

``GET /reef/scenarios/{scenario}/records`` reads retained record metadata,
including consumed records. ``after_sequence`` defaults to 0 and ``limit``
defaults to 50 (1 to 100). ``request_type`` (``inference``, ``report`` or
``train``) lists one type only; another value is HTTP 400. Records are oldest
first; ``next_after_sequence`` is null at the end. Each row contains
``sequence``, ``agent_record_id``, ``request_type``, ``created_at``,
``references``, ``artifact_id`` and the payload's ``score`` field.
``artifact_id`` names the artifact that answered an inference call, or is
null when the record names none. No record payload or learning classification
is included.

``GET /reef/scenarios/{scenario}/records/{record_id}`` returns that metadata
and the stored ``payload``. A missing body returns 404: it may have expired or
never been retained. Reading never reactivates records or changes retention.

``GET /reef/scenarios/{scenario}/operations`` returns ``{scenario,
operations, next_after_commit_number}``, oldest first. ``after_commit_number`` defaults to 0;
``limit`` defaults to 50 and accepts 1–100. ``next_after_commit_number`` is null at
the end. ``/commits`` remains an alias with the array named ``commits``.

Repeat ``record_id`` to filter operations whose ``consumed_ids`` contain any
requested ID (at most 100 IDs of 1–256 characters). This tests exact set
membership, not whether a record was eligible for training.

Each row exposes ``commit_number``, ``operation``, ``operation_verified``,
``recorded_at``, ``components``, ``component``, ``activated``,
``consumed_ids`` and ``metrics``. ``components`` maps component names to
artifact IDs associated with the operation. ``component`` identifies the
changed component when present. Private storage locators, algorithm state
and checkpoint recovery data are excluded. Operations remain readable after
trace bodies expire.

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
serves`_).

Activation, restore and serving do not create artifacts or repository
revisions. An import or training step can create an artifact without changing
what serves. Recovery replays serving selections and training starts
independently. Restore appends a new operation; it does not remove or rewind
history.

The existing ``GET /reef/scenarios/{scenario}/contract`` also reports
``training_mode`` and processor ``status`` alongside ``processor`` and
``required_request_types``. These are live runtime facts, not policy versions
or historical rule decisions.

All these reads require service authentication and return
``Cache-Control: no-store``. Payloads and recorded metrics can contain request
content. A platform must bind the upstream scenario to the authenticated
workspace. Each response is a live read; concurrent training and retention
can change subsequent pages. Reading commit pages may scan the scenario's
cached commit history; the response is bounded, not a new persisted index.

Reef does not join these endpoints into learning links or assign learning
states, human-readable explanations, policy capability flags, or evaluation
results. The console owns that interpretation. Consumption is not proof that a
snapshot made from the records serves.


Sample processing tasks
-----------------------

``GET /reef/scenarios/{scenario}/sample-processing`` reads the samples queue through the data service.
The scenario must exist. It does not start processing.

Optional query parameters are ``state``, ``before``, ``limit`` and ``changed``.
``state`` is ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``.
``before`` excludes creation sequences at or above its positive integer value.
``limit`` is 1 to 100, with default 50. Tasks appear newest first.
``changed`` is ``true`` or ``false``, with default ``false``.
With ``true``, the page omits finished tasks that gave the trainer nothing new.
It reads at most 5,000 tasks to fill the page.

The response includes ``scenario``, ``enabled``, ``read_at``, ``retention_seconds``, ``records_waiting``,
``counts``, ``tasks``, ``hidden`` and ``next``.
``records_waiting`` is the number of the scenario's records after the last record that a trainer read, for the
trainer furthest behind. Each trainer's newest finished task with ``record_sequences`` gives that record.
It is null when a trainer has no such task in the retained history.
``enabled`` means the scenario has a trainer that generates samples.
It does not report worker health. ``counts`` covers all retained tasks, independent of pagination and state filters.
``hidden`` lists each run of omitted tasks, and is empty without ``changed=true``.
A run has ``after``, ``count``, ``first_created_at`` and ``last_created_at``.
``after`` is the ID of the shown task just newer than the run, or null at the top of the page.
``next`` is the next page's ``before`` value, or null.
Reads of the page and counts are separate snapshots; tasks can change between them.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}`` returns one task.
An unknown task, or a task of another scenario, returns 404.

Each task has ``id``, ``component``, ``state``, ``attempt_count``, ``created_at``, ``started_at``,
``ended_at``, ``available_at``, ``error``, ``changed``, ``background``, ``records_read``, ``samples_created``,
``record_sequences``, ``triggers`` and ``progress``.
Times are Unix seconds. Unavailable fields are null.
``triggers`` lists why the task ran, oldest first. Each entry has ``name`` and ``at``, the time of the first request.
A name is ``records``, ``data``, ``commit``, ``reload``, ``background`` or ``due``.
``background`` comes from a background job of a stage that ended, or from a fallback look while such jobs run.
The fallback waits 1 second, then twice as long after each look that changed nothing, up to 300 seconds.
Only ``records`` and ``data`` bring new records. The others let a stage make samples without a new record.
A task from an older Reef has no triggers.
``progress`` holds the counts so far of the attempt that runs now: ``attempt``, ``records_read``,
``samples_created`` and ``at``. It is null for a task that does not run.
``record_sequences`` is ``[first, last]``: the data sequences of the first and last record of the scenario that
the completed attempt read. It is null when the attempt read none, and for a task from an older Reef.
A sample that a task stored has the item metadata ``stored_by_task`` and ``read_by_task``.
A data query with ``group_key`` set to one of them finds the samples of one task.
``started_at`` is the first attempt's start. ``error`` can describe a queued task's previous failed attempt.

Processing counts describe the completed attempt only; they exclude earlier failed attempts.
``records_read`` includes records that replay read but the pipeline ignored.
``samples_created`` counts new inserts, excluding duplicate writes.
Legacy results report null counts.
``changed`` can also mean a new training request, without new samples.
``background=true`` means work remained active when the task finished; later tasks check it.

Control removes ended tasks after ``storage.work_retention_s``, which defaults to seven days.
This removes task history, not stored samples. Active tasks do not expire through this retention policy.

``GET /reef/scenarios/{scenario}/sample-processing/{task_id}/logs`` returns the log of one attempt of a task.
The samples worker keeps each line that the attempt's work logs.
This includes the stage's background jobs that the attempt submitted, also when they end after the attempt.
Reef writes the lines every 2 seconds and when the attempt ends.
Lines of threads that the work starts by hand are not kept.
One attempt keeps at most 2,000 lines of at most 4,096 characters.
The line after the last one says that the log was cut.

Optional query parameters are ``attempt``, ``after`` and ``limit``.
``attempt`` is 1 to the task's ``attempt_count``, with default the last attempt.
``after`` is -1 or more, with default -1. The page holds lines with a larger number.
``limit`` is 1 to 1000, with default 500.

The response includes ``task_id``, ``attempt``, ``attempt_count``, ``running``, ``lines`` and ``next``.
``attempt`` is null for a task that no worker took.
``running`` is true while the chosen attempt runs, so more lines can come.
Each line has ``number``, ``at``, ``level``, ``logger`` and ``message``.
``next`` is the next page's ``after`` value, or null at the end of the lines kept so far.
The log goes with its task when the retention deletes the task.
