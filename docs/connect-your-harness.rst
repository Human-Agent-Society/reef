Connect your harness
====================

Before connecting your harness, make sure a Reef deployment is running. If you
operate Reef yourself, follow `Get started <get-started.rst>`__ to install and serve it.
For example, the complete SAO deployment lives in
``examples/sao/serve.yaml``; its README explains how to start it.

Check the deployment before sending model traffic:

.. code:: bash

   export REEF_URL=http://127.0.0.1:8900  # or other port depending on your config
   export REEF_TOKEN=reef-local
   curl -f "$REEF_URL/healthz"
   # {"ok": true}

``GET /healthz`` is unauthenticated and reports whether the service process is
ready to answer requests.

The following sections explain the additional headers and requests a harness
uses after Reef is running.

Request headers
---------------

These are headers that a harness sends to Reef:

+-----------------------------------+----------------------+------------------------+
| Header                            | Required for         | Purpose                |
+===================================+======================+========================+
| ``x-reef-scenario``               | inference, report,   | identifies what the    |
|                                   | harness manifest and | record belongs to.     |
|                                   | versions; optional   | Each scenario manages  |
|                                   | on harness install   | one set of artifacts.  |
+-----------------------------------+----------------------+------------------------+
| ``Authorization: Bearer <token>`` | every route except   | authenticates the      |
|                                   | ``/healthz``, when   | caller                 |
|                                   | auth is configured   |                        |
+-----------------------------------+----------------------+------------------------+


Create the scenario
-------------------

A new scenario is created by sending an inference
request or report with a new ``x-reef-scenario`` value. The scenario binds to
the recipe the deployment serves (``reef.recipe`` in its config), permanently.

.. code:: bash

   curl -sD - "$REEF_URL/v1/chat/completions" \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: code-repair" \
     -H "content-type: application/json" \
     -d '{"model": "m", "messages": [{"role": "user", "content": "fix it"}]}'

If the deployment sets ``reef.allow_implicit_scenario_creation: false``, a
request for an unknown scenario returns HTTP 404. Create the scenario
explicitly before sending traffic:

+---------------------------------------------+---------------------------------------------+-----------------------------------------------------------+
| Endpoint                                    | Request body                                | Response                                                  |
+=============================================+=============================================+===========================================================+
| ``POST /reef/scenarios``                    | ``{"name", "recipe", "artifact_version"?}`` | ``{scenario, recipe, artifact_version}``, 201 when        |
|                                             |                                             | created, 200 when it already existed                      |
+---------------------------------------------+---------------------------------------------+-----------------------------------------------------------+
| ``GET /reef/scenarios``                     | none                                        | every known scenario, with its recipe and current         |
|                                             |                                             | artifact version once loaded                              |
+---------------------------------------------+---------------------------------------------+-----------------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract`` | none                                        | ``{scenario, recipe, processor, required_request_types}`` |
+---------------------------------------------+---------------------------------------------+-----------------------------------------------------------+

Inference
---------

Send OpenAI-format inference requests to ``POST /v1/chat/completions`` and
Anthropic-format requests to ``POST /v1/messages``. Use the same request body
you would send directly to the provider. Reef does not add or change sampling
parameters. For streaming inference, set ``"stream": true`` and read the SSE response.
Anthropic clients may use ``POST /v1/messages/count_tokens`` to count request
tokens without creating an inference record.

Set the Reef request header ``x-reef-capture: false`` for a serve-only call.
Reef still resolves the scenario's current artifact and applies its normal
inference safety checks, but it does not create an inference record or expose
the call to the scenario's processor. The header accepts only ``true`` or
``false`` (case-insensitive) and defaults to ``true`` when absent. Because it
is a Reef transport control, it is not forwarded to the inference provider.

Reef returns a receipt that identifies the stored inference record. A
non-streaming response carries it in the ``x-reef-agent-record-id`` header. A
streaming response carries it only after the record has been stored: OpenAI
streams receive a final empty-``choices`` chunk containing
``reef.agent_record_id`` immediately before ``data: [DONE]``; Anthropic streams
carry the same field in ``message_stop``. Serve-only responses carry no
receipt; their provider terminal event is forwarded unchanged. Capture only
controls whether a record exists. For captured traffic, the recipe remains
responsible for deciding whether and how the record trains.

Report
------

Submit feedback with ``POST /reef/report``. Reef stores the following four
fields and drops other top-level fields. Put harness-specific data inside
``metadata`` or ``feedback``.

+----------------+------------------+----------------------------------+
| Field          | Type             | Notes                            |
+================+==================+==================================+
| ``score``      | number           | optional; a bool is not a number |
|                |                  | and is rejected                  |
+----------------+------------------+----------------------------------+
| ``feedback``   | string or object | opaque to Reef's core: a rubric, |
|                |                  | judge output, plain text         |
+----------------+------------------+----------------------------------+
| ``references`` | list of strings  | the receipts this report grades  |
+----------------+------------------+----------------------------------+
| ``metadata``   | object           | opaque, except the training      |
|                |                  | eligibility flag below           |
+----------------+------------------+----------------------------------+

Suppose Reef returned the receipt ``abc123`` for a model response in scenario
``code-repair``. The harness grades that call like this:

.. code:: python

   client.report("code-repair", {
       "agent_record_id": "myharness:run42:trial7",
       "score": 1.0,
       "references": ["abc123"],
       "metadata": {"harbor": {"trial_id": "run42:7"}},
   })

The optional top-level ``agent_record_id`` makes posting retry-safe. Reef
strips it from the payload and uses it as the report's own record id, so an
identical resend returns the stored record instead of reprocessing it, while
the same id with different content is HTTP 409. It is distinct from the
receipts in ``references``, which identify the inference records being graded.

A recipe may declare a report schema, in which case Reef validates keys in
``feedback`` and ``metadata`` at ingress and answers HTTP 400 on a violation.
To keep a report out of the training data, send
``"metadata": {"training": {"eligible": false}}``. The flag defaults to true.

Receive an updated artifact
---------------------------

How a harness receives an update depends on the artifact type:

- For weight-training scenarios, continue calling the same inference
  endpoint. It automatically serves the latest published weights.
- For harness-evolution scenarios, new harness installations are published on
  the endpoint ``GET /reef/harness/install``. Reef's harness has a
  self-contained wrapper that automatically checks for updates and prompts the
  user to install them.

The following endpoints expose a harness tree:

+--------------------------------+---------------------------------------------------------------+
| Endpoint                       | Response                                                      |
+================================+===============================================================+
| ``GET /reef/harness``          | the served tree:                                              |
|                                | ``{artifact_version, parent_artifact_version, files, gate}``, |
|                                | plus an ``x-reef-artifact-version`` response header           |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/versions`` | ``{scenario, versions}``, the full catalog, oldest first,     |
|                                | each training row carrying the gate metrics of the step that  |
|                                | published it                                                  |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/install``  | a self-contained POSIX shell script that installs the vendor  |
|                                | binary and writes the tree                                    |
+--------------------------------+---------------------------------------------------------------+

The manifest and version endpoints are read-only and require
``x-reef-scenario``. The install endpoint is also read-only when the request
names an existing scenario.

If ``GET /reef/harness/install`` omits ``x-reef-scenario``, Reef creates a
scenario with a generated ``harness-`` name and embeds that assignment in the
wrapper script. When exactly one configured recipe serves harness files, Reef
selects it automatically.

Use ``?version=`` on the manifest or install endpoint to request a particular
catalog version instead of the latest one. An unknown or unrestorable version
returns HTTP 404. Installation also requires ``?adapter=``. The value may be
``pi``, ``opencode``, or an external descriptor.

Pulling an older version changes only the local copy. To roll back the version
served by Reef, send ``POST /reef/scenarios/{scenario}/rollback`` with
``{"artifact_version": "…"}``. Reef republishes the checkpoint as a new commit
instead of rewinding history, which keeps step numbers monotonic.

Use ``GET /reef/scenarios/{scenario}/versions`` to choose a rollback target.
This endpoint lists versions **newest first**, while
``GET /reef/harness/versions`` lists them oldest first. Only versions marked
``restorable`` can be rolled back.

Inspect deployment status
-------------------------

``GET /reef/status`` reports asynchronous training and preload failures,
current weight versions, inference admission, processor state, and checkpoint
storage state. It is useful when inference continues serving an older version
while an update is being trained or published.

Status codes
------------

+--------+-------------------------------------------------------------+
| Status | Cause                                                       |
+========+=============================================================+
| 400    | a malformed body, a missing or empty ``x-reef-scenario`` on |
|        | a scenario-scoped route, a report violating the recipe's    |
|        | declared schema                                             |
+--------+-------------------------------------------------------------+
| 401    | missing or wrong bearer token                               |
+--------+-------------------------------------------------------------+
| 403    | a valid bearer token that does not own the scenario         |
+--------+-------------------------------------------------------------+
| 404    | unknown scenario (with implicit creation off), unknown      |
|        | artifact version, unknown adapter, no configured harness    |
|        | recipe, or a scenario that serves no files                  |
+--------+-------------------------------------------------------------+
| 409    | a recipe or base artifact conflicting with the binding, a   |
|        | record id resent with different content, an engine that     |
|        | reports no serving weight version                           |
+--------+-------------------------------------------------------------+
| 502    | the upstream provider failed on its own account             |
+--------+-------------------------------------------------------------+
| 503    | the artifact store is unreachable, or inference kept losing |
|        | the weight-update race until its deadline                   |
+--------+-------------------------------------------------------------+

Reef relays upstream 4xx responses with the provider's original message so an
agent can use the details to repair its next request.

Next, read `Write a recipe <define-a-recipe.rst>`__ to decide what Reef does
with these records, or `Deploy and train <deploy-and-train.rst>`__ to configure
the service and training driver. The `glossary <glossary.rst>`__ defines the
terms used throughout the docs.
