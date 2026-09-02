HTTP API
========

Reef serves the provider's own inference routes: OpenAI at
``/v1/chat/completions`` and Anthropic at ``/v1/messages``. It forwards each
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
| ``GET /reef/scenarios``                         | every known scenario and current release          |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios``                        | create a scenario explicitly                      |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract``     | what this scenario accepts                        |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/releases``     | ``{scenario, releases}``, newest first            |
+-------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/rollback``    | republish an earlier release as the head          |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness``                           | the served harness tree                           |
+-------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/releases``                  | the harness release catalog, oldest first         |
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
| ``x-reef-scenario``               | inference, report, harness manifest and releases;       |
|                                   | optional on harness install. Names the workload a       |
|                                   | record belongs to.                                      |
+-----------------------------------+---------------------------------------------------------+
| ``Authorization: Bearer <token>`` | every route except ``GET /healthz``, when auth is       |
|                                   | configured.                                             |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-release-id``             | optional: bind a new scenario to this starting release; |
|                                   | on an existing scenario it must name the bound starting |
|                                   | release, or the request is HTTP 409. Inference always   |
|                                   | answers from the scenario's current release; to pull a  |
|                                   | specific release, use ``?release_id=`` on the harness   |
|                                   | manifest or install route.                              |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-tag-<name>``             | optional on inference: opaque key/value context stored  |
|                                   | on the record under ``metadata.tags``, for a processor  |
|                                   | to correlate on. Reef never reads a value.              |
+-----------------------------------+---------------------------------------------------------+

Scenarios
---------

The first inference request or report carrying a new ``x-reef-scenario`` creates
the scenario using the deployment's configured recipe. Requests never select a
recipe.

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
| ``POST /reef/scenarios``                    | ``{"name", "release_id"?}``                 |
|                                             | → ``{scenario, release_id,                  |
|                                             | content_id}``; 201 created, 200 already     |
|                                             | existed                                     |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios``                     | every known scenario and its current        |
|                                             | release once loaded                         |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract`` | ``{scenario, processor,                     |
|                                             | required_request_types}``                   |
+---------------------------------------------+---------------------------------------------+

Inference
---------

Send the same body you would send to the provider. Reef never touches your
sampling parameters. On a weight-serving deployment it adds engine
bookkeeping keys: ``lora_path`` to address the served adapter and
``return_meta_info`` so the record proves which weights answered; a body
naming a different ``lora_path`` is refused. Set ``"stream": true`` and read
the SSE response for streaming.

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
| ``references`` | list of strings  | no       | the receipts this report grades; several  |
|                |                  |          | batch as one trajectory sample, none is   |
|                |                  |          | accepted but never trains                 |
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
HTTP 409. It is not a receipt. Receipts go in ``references``.

A recipe may declare a report schema. `tttd
<../user-guide/recipes/tttd.rst#the-report-contract>`__ declares one. In that case, Reef
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
| ``GET /reef/harness``          | ``{release_id, content_id, parent_release_id, files, gate}``, |
|                                | plus an ``x-reef-release-id`` response header                 |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/releases`` | ``{scenario, releases}``, oldest first, each training row     |
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

Use ``?release_id=`` on the manifest or install route to request a specific
catalog release. An unknown or unrestorable release returns HTTP 404.

Rollback
~~~~~~~~

Pulling an older release changes only your local copy. To move the release Reef
*serves*, send ``POST /reef/scenarios/{scenario}/rollback`` with
``{"release_id": "…"}``; it answers the new head. Reef republishes that
checkpoint as a new commit rather than rewinding history, so step numbers stay
monotonic.

Choose a target from ``GET /reef/scenarios/{scenario}/releases``, which lists
**newest first**; ``GET /reef/harness/releases`` lists oldest first. Only
releases marked ``restorable`` can be rolled back.

Status
------

Read ``GET /reef/status`` when inference is still serving an older release while
an update is being trained or published.

.. code:: json

   {
     "error": null,
     "last_drain_at": 1756400000.0,
     "preload_errors": {},
     "scenarios": {
       "hello-reef": {
         "scenario_step": 3,
         "last_committed_step": {
           "step": 3,
           "recorded_at": 1756400000.0,
           "metrics": {"published": false, "selection": {"reason": "candidate lost"}}
         },
         "current_runtime_load_id": "7f2a:12",
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
``last_committed_step`` reports the latest durable training step number,
commit time, and its recipe-owned metrics; it is ``null`` before the first
training commit. This distinguishes a step still in flight from a completed
step that skipped or rejected its candidate. A rollback advances
``scenario_step`` without replacing the latest training outcome. A deployment
without an agent-record directory has no historical commit log, so after a
restart from a rollback checkpoint this field is ``null`` until the next
training commit.
``serving`` is runtime-wide but recipe-shaped: a LoRA deployment reports the
engine's shared adapter residency there, keyed by recipe. Each scenario's
``adapter_runtime_load_id`` appears in its own ``scenarios`` block.

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
|        | release, unknown adapter, no configured harness             |
|        | recipe, or a scenario that serves no files                  |
+--------+-------------------------------------------------------------+
| 409    | a base artifact conflicting with the scenario registration, |
|        | record id resent with different content, a rollback naming  |
|        | a release that is not restorable, or an engine that reports |
|        | no serving runtime load ID                                  |
+--------+-------------------------------------------------------------+
| 502    | the upstream provider failed on its own account             |
+--------+-------------------------------------------------------------+
| 503    | the artifact store is unreachable, or inference kept losing |
|        | the weight-update race until its deadline                   |
+--------+-------------------------------------------------------------+

Reef relays upstream 4xx failures with the provider's original message; the
common client statuses (400, 401, 403, 404, 408, 409, 422, 429) keep their
status code, and any other upstream 4xx comes back as 400.
