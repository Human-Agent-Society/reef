HTTP API
========

Reef serves the provider's own inference routes — OpenAI at
``/v1/chat/completions``, Anthropic at ``/v1/messages`` — and forwards each
request to the runtime unchanged. It adds a small set of ``/reef/*`` routes for
feedback, scenarios, artifacts, and status.

.. code:: bash

   export REEF_TOKEN=reef-local
   curl -f http://127.0.0.1:8900/healthz     # {"ok": true}

Routes
------

+-------------------------------------------------+---------------------------------------------------+
| Route                                           | Response                                          |
+=================================================+===================================================+
| ``GET /healthz``                                | readiness; the only unauthenticated route         |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /v1/chat/completions``                   | OpenAI-format inference                           |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /v1/messages``                           | Anthropic-format inference                        |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /v1/messages/count_tokens``              | count request tokens; recorded like any inference |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/report``                           | submit feedback about one or more receipts        |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios``                         | every known scenario, recipe, and current version |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios``                        | create a scenario explicitly                      |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract``     | what this scenario accepts                        |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/versions``     | ``{scenario, versions}``, newest first            |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/rollback``    | republish an earlier version as the head          |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness``                           | the served harness tree                           |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/versions``                  | the harness version catalog, oldest first         |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/install``                   | a shell script that installs the tree             |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/status``                            | training, serving, and storage state              |
+-------------------------------------------------+---------------------------------------------------+

Headers
-------

+-----------------------------------+---------------------------------------------------------+
| Header                            | Required for                                            |
+===================================+=========================================================+
| ``x-reef-scenario``               | inference, report, harness manifest and versions;       |
|                                   | optional on harness install. Names the workload a       |
|                                   | record belongs to.                                      |
+-----------------------------------+---------------------------------------------------------+
| ``Authorization: Bearer <token>`` | every route except ``GET /healthz``, when auth is       |
|                                   | configured.                                             |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-artifact-version``       | optional: on the request that creates a scenario, the   |
|                                   | starting artifact version to bind. A later request      |
|                                   | naming a different one is HTTP 409.                     |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-tag-<name>``             | optional on inference: opaque key/value context stored  |
|                                   | on the record under ``metadata.tags``, for a processor  |
|                                   | to correlate on. Reef never reads a value.              |
+-----------------------------------+---------------------------------------------------------+

Scenarios
---------

The first inference request or report carrying a new ``x-reef-scenario`` creates
the scenario and binds it to the deployment's recipe.

.. code:: bash

   curl -sS -i http://127.0.0.1:8900/v1/chat/completions \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: hello-reef" \
     -H "Content-Type: application/json" \
     -d '{"model": "m", "messages": [{"role": "user", "content": "fix it"}]}'

If the deployment sets ``reef.allow_implicit_scenario_creation: false``, an
unknown scenario returns HTTP 404 and you create it first:

+---------------------------------------------+---------------------------------------------+
| Route                                       | Body and response                           |
+=============================================+=============================================+
| ``POST /reef/scenarios``                    | ``{"name", "recipe", "artifact_version"?}`` |
|                                             | → ``{scenario, recipe, artifact_version}``; |
|                                             | 201 created, 200 already existed            |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios``                     | every known scenario with its recipe and    |
|                                             | current version once loaded                 |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract`` | ``{scenario, recipe, processor,             |
|                                             | required_request_types}``                   |
+---------------------------------------------+---------------------------------------------+

Inference
---------

Send the same body you would send to the provider. Reef adds and changes
nothing, sampling parameters included. Set ``"stream": true`` and read the SSE
response for streaming.

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

Streams carry it only after the record is stored.

Report
------

Reef stores four fields and drops other top-level keys. Put harness-specific
data inside ``metadata`` or ``feedback``.

+----------------+------------------+----------+-------------------------------------------+
| Field          | Type             | Required | Notes                                     |
+================+==================+==========+===========================================+
| ``score``      | number           | no       | a bool is not a number and is rejected    |
+----------------+------------------+----------+-------------------------------------------+
| ``feedback``   | string or object | no       | opaque to Reef's core: a rubric, judge    |
|                |                  |          | output, plain text                        |
+----------------+------------------+----------+-------------------------------------------+
| ``references`` | list of strings  | yes      | the receipts this report grades           |
+----------------+------------------+----------+-------------------------------------------+
| ``metadata``   | object           | no       | opaque, except                            |
|                |                  |          | ``training.eligible`` (default ``true``)  |
+----------------+------------------+----------+-------------------------------------------+

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
HTTP 409. It is not a receipt — receipts go in ``references``.

A recipe may declare a report schema — `tttd
<../user-guide/recipes/tttd.rst#the-report-contract>`__ declares one — in which case Reef
validates the declared ``score`` and ``metadata`` fields at ingress and answers
HTTP 400 on a violation; ``feedback`` and undeclared ``metadata`` keys pass
through unvalidated. To record a report but keep it out of training, send
``"metadata": {"training": {"eligible": false}}``.

Receiving an update
-------------------

For weight-training scenarios there is nothing to do: keep calling the same
inference endpoint and it serves the latest published weights.

Harness artifacts
~~~~~~~~~~~~~~~~~

+--------------------------------+---------------------------------------------------------------+
| Route                          | Response                                                      |
+================================+===============================================================+
| ``GET /reef/harness``          | ``{artifact_version, parent_artifact_version, files, gate}``, |
|                                | plus an ``x-reef-artifact-version`` response header           |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/versions`` | ``{scenario, versions}``, oldest first, each training row     |
|                                | carrying the gate metrics of the step that published it       |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/install``  | a self-contained POSIX shell script that installs the vendor  |
|                                | binary and writes the tree                                    |
+--------------------------------+---------------------------------------------------------------+

All three are read-only and take ``x-reef-scenario``. Install also requires
``?adapter=``, whose value may be ``pi``, ``opencode``, or an external
descriptor. If install omits ``x-reef-scenario``, Reef creates a scenario with a
generated ``harness-`` name and embeds that assignment in the wrapper script;
when exactly one configured recipe serves harness files, it selects that recipe
automatically.

Use ``?version=`` on the manifest or install route to request a specific catalog
version. An unknown or unrestorable version returns HTTP 404.

Rollback
~~~~~~~~

Pulling an older version changes only your local copy. To move the version Reef
*serves*, send ``POST /reef/scenarios/{scenario}/rollback`` with
``{"artifact_version": "…"}``; it answers the new head. Reef republishes that
checkpoint as a new commit rather than rewinding history, so step numbers stay
monotonic.

Choose a target from ``GET /reef/scenarios/{scenario}/versions``, which lists
**newest first**; ``GET /reef/harness/versions`` lists oldest first. Only
versions marked ``restorable`` can be rolled back.

Status
------

Read ``GET /reef/status`` when inference is still serving an older version while
an update is being trained or published.

.. code:: json

   {
     "error": null,
     "last_drain_at": 1756400000.0,
     "preload_errors": {},
     "scenarios": {
       "hello-reef": {
         "scenario_step": 3,
         "current_weight_version": "7f2a:12",
         "checkpoint_storage": {"...": "..."},
         "batch_ready": false,
         "processor": {"...": "..."},
         "inference_admission": {"...": "..."}
       }
     },
     "serving": {"...": "..."}
   }

``error`` and ``preload_errors`` report asynchronous training and preload
failures. ``batch_ready`` says whether the processor has a batch waiting.
``serving`` is runtime-wide and recipe-shaped — a LoRA deployment reports each
scenario's ``adapter_weight_version`` under it.

Status codes
------------

+--------+-------------------------------------------------------------+
| Status | Cause                                                       |
+========+=============================================================+
| 400    | malformed body, a missing or empty ``x-reef-scenario`` on a |
|        | scenario-scoped route, or a report violating the recipe's   |
|        | declared schema                                             |
+--------+-------------------------------------------------------------+
| 401    | missing or wrong bearer token                               |
+--------+-------------------------------------------------------------+
| 403    | relayed from the upstream provider. Reef issues none of its |
|        | own: an unaccepted token is 401, and per-scenario           |
|        | authorization belongs to the gateway in front of Reef.      |
+--------+-------------------------------------------------------------+
| 404    | unknown scenario (with implicit creation off), unknown      |
|        | artifact version, unknown adapter, no configured harness    |
|        | recipe, or a scenario that serves no files                  |
+--------+-------------------------------------------------------------+
| 409    | a recipe or base artifact conflicting with the binding, a   |
|        | record id resent with different content, or an engine that  |
|        | reports no serving weight version                           |
+--------+-------------------------------------------------------------+
| 502    | the upstream provider failed on its own account             |
+--------+-------------------------------------------------------------+
| 503    | the artifact store is unreachable, or inference kept losing |
|        | the weight-update race until its deadline                   |
+--------+-------------------------------------------------------------+

Reef relays upstream 4xx responses with the provider's original message.
