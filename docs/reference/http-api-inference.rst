HTTP API: inference and feedback
================================

Send model requests to Reef, read the startup state of a served artifact, find the receipt of each response, and report feedback about the calls. `HTTP API <http-api.rst>`__ lists every route and the headers.

Startup status
--------------

Scenario creation and activation select an artifact and do not wait for its engine. The inference service publishes the artifact's route after startup succeeds. Read the startup state with:

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/scenarios/hello-reef/inference/status \
     -H "Authorization: Bearer $REEF_TOKEN"

.. code:: json

   {"artifact_id": "artifact-1", "state": "starting", "error": null, "engine": null, "engine_detail": null,
    "engine_replicas": null,
    "startup": {"steps": [{"phase": "prepare", "started_at": 1794150000.1, "ended_at": 1794150480.6,
                           "detail": "Quantizing to NVFP4"},
                          {"phase": "start", "started_at": 1794150480.6, "ended_at": null,
                           "detail": "Waiting for a GPU"}],
                "estimate_seconds": null}}

The query parameter ``model`` defaults to ``active``. It can also name an active artifact alias, or an artifact that the scenario serves beside the active one (``?model=<artifact-id>``). An unknown name returns 404. One startup covers the active artifact and each artifact beside it, but an artifact with a published route stays ``ready`` while another artifact starts or fails.

``state`` is one of:

* ``pending``: no worker has taken the startup work.
* ``starting``: inference has taken it. Downloads, conversion, loading, and retries are part of this state.
* ``ready``: startup succeeded and inference published the route.
* ``failed``: startup ended with an error. ``error`` gives a safe message: the message of a runtime's ``StartupFailedError``, such as an engine argument that the model refuses, else a fixed text. The service logs hold the details. ``error`` is null in every other state.

Control retries a failed startup 60 seconds after the failure, then after a wait that doubles at each failed retry, up to one hour. A retry returns the state to ``pending``. To retry at once, resend the complete accepted ``served`` mapping.

A newer request, such as one for a changed serving config, replaces the current one at once. The replaced work stops within seconds when the runtime reads ``reef.inference.startup.stopping()``. Requests of different scenarios start one after another, or together when the runtime has more than one ``serving_slots``.

``engine`` is the engine state behind the route of a ``ready`` artifact, when the deployment's inference handler reports it:

* ``warm``: an engine answers now.
* ``starting``: an engine runs, but it does not answer yet.
* ``idle``: no engine runs. The next request starts one and waits for it.

``engine`` is null in every other state, and when the handler cannot tell. While ``engine`` is ``starting``, ``engine_detail`` can give one short line about what the engine does now, such as ``Loading weights · 0:42``. It is null otherwise.

``engine_replicas`` counts the engine replicas behind the route of a ``ready`` artifact, such as the containers of a server that scales, when the handler reports them: ``{"serving": 1, "starting": 2, "replicas": [...]}``. ``serving`` replicas answer now, and ``starting`` ones run but do not answer yet. ``replicas`` names each replica when the handler can, and is empty otherwise: ``{"id": "ta-01M8...", "state": "starting", "started_at": 1795590000.0, "startup": {...}}``. ``id`` names the replica, such as its container ID. ``started_at`` is when it started, in epoch seconds, or null. ``startup`` gives its own ``start`` and ``load`` phases while it starts, in the form of the status's ``startup``, or null. It is null in every other state, and when the handler cannot tell. An engine can be ``warm`` while more replicas start. Then ``engine_detail`` and the ``start`` and ``load`` phases of ``startup`` describe a replica that starts.

``startup`` gives the artifact's startup in three fixed phases, the same for every backend:

* ``prepare``: the serving work waits, deploys the engine, and converts or quantizes the weights.
* ``start``: a machine or container is found, and the weights reach it.
* ``load``: the weights go into GPU memory, and kernels, CUDA graphs and warm-up run.

It is ``{steps, estimate_seconds}``, or null:

- Each step is ``{phase, started_at, ended_at, detail}``, in phase order, with times in epoch seconds. Only the last step can have ``ended_at`` null: it runs now. ``detail`` is a short note of what the step does, or null. A phase that the backend does not report has no step.
- ``estimate_seconds`` is a guess of the seconds until the engine is warm, or null.
- While the state is ``pending`` or ``starting``, the steps are ``prepare`` from the startup request, then the phases that the runtime reported for this artifact.
- While a ready artifact's engine is ``starting``, the steps are the phases that the inference handler reports.
- ``startup`` is null otherwise.

The status describes the selected artifact, even while an older route still serves requests. It is a startup result, not a continuous health check. A read starts no work and no engine. The route requires the same bearer token as inference. No startup work returns 404. An unreadable state store returns 503.

Without a usable route, an inference request for a pending or starting artifact returns HTTP 503 with error type ``inference_starting`` and ``Retry-After: 30``. A failed startup returns HTTP 503 with type ``inference_start_failed``. Both errors use ``{"error": {"type", "message"}}``. Control and data operations remain available during startup.

``POST /reef/scenarios/{scenario}/redeploy`` deploys the engines of what the scenario serves again, also when they run its current settings, for example after a hang. A new request with the same selection replaces the current one, and the runtime reads ``reef.inference.startup.redeploying()``. It answers ``{scenario}``, also for a scenario that serves nothing.

Request
-------

Send the body that you would send to the provider. Reef never changes your sampling parameters, and no artifact adds to the request. For streaming, set ``"stream": true`` and read the SSE response.

Reef answers the request from the scenario's row in ``served_models``:

- The request's ``model`` selects the row of a served artifact when it names one. Any other ``model`` selects the active row, or the row that a `traffic split <http-api-serving.rst#traffic-split>`__ gives the request.
- Reef keeps that row for the whole request, even if an activation completes during it. It sends the request to the engine at the row's URL under the row's model name, and records the snapshot of the row's artifact.
- To require a specific artifact, use ``/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/chat/completions``.
- ``/reef/scenarios/{scenario}/v1/{route}`` is the same call as ``/v1/{route}``, with the scenario in the path instead of the header. A client install's model binding calls it.
- On a weight-serving deployment, the engine handler selects the served adapter itself, and it replaces a ``lora_path`` that the body names.

`Run the services in separate processes <../user-guide/split-services.rst>`__ gives the status codes of an inference request.

Reef retries a request once when the engine aborts a non-streaming answer, or refuses the row's model name before any output. It reads the row again for the retry. Three kinds of request keep their artifact: a request to an artifact's route, a request whose ``model`` names its artifact, and a request of a traffic split. When no row serves that artifact at the retry, Reef answers HTTP 503 with ``Retry-After``. The retry of any other request can reach a new active artifact.

Log probabilities depend on what answers the request:

- **A provider or routed target.** Reef forwards ``logprobs`` and ``top_logprobs`` in the body. It preserves the provider's returned log probabilities in the response and the record.
- **Reef's SGLang or vLLM backend.** On ``/v1/chat/completions`` with ``"logprobs": true``, ``choices[0].logprobs.content`` has one entry for each sampled token, in output order. A stream sends the entries with their text, and the record holds the same entries. Each entry's ``top_logprobs`` holds only the sampled token. The SGLang backend returns the engine's alternatives in SGLang's own format, in ``choices[0].meta_info.output_top_logprobs``; a stream sends them in its last chunk. The vLLM backend returns no alternatives.
- **Reef's SGLang or vLLM backend on ``/v1/messages``.** Anthropic Messages has no log probabilities. Reef extends it: a buffered request with ``"logprobs": true`` gets the same entries in a top-level ``logprobs`` (``{"content": [...]}``) of the message. A streamed message has none.

Token IDs follow the same rule. On ``/v1/chat/completions``, a request to Reef's SGLang or vLLM backend with ``"return_token_ids": true`` gets the rendered prompt's IDs in ``prompt_token_ids`` and the sampled IDs in ``choices[0].token_ids``, as vLLM's extension returns them. A stream sends both in its last chunk. On ``/v1/messages``, a buffered message gets both at its top level, as ``prompt_token_ids`` and ``token_ids``. The record holds the same fields. A request without the flag gets no token IDs. When Reef stores a record as ATIF, the reply's step keeps the token IDs and the ``logprob`` of each entry in its ``metrics`` (``prompt_token_ids``, ``completion_token_ids`` and ``logprobs``).

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

A stream carries the receipt only after the record is stored, so an incomplete stream carries none. A stream that fails after its response headers also carries no terminal event: Reef closes the connection, and a client SDK reports a transport error. These failures end a stream in this way:

- an engine or transport error;
- a missing terminal event;
- a handler's report that the engine does not serve the selected weights;
- the inference timeout.

A stream from Reef's SGLang or vLLM backend that the engine aborts keeps its terminal event. It is still incomplete, because Reef cannot restart a stream after it sent frames.

Reef stores each incomplete stream as an incomplete delivery record and preserves the provider response. A harness that reports by receipt does not reference these records. Reef excludes them from sample creation and training, also when a report references their IDs directly or retained records are replayed after a restart. A routed stream follows `Routed external-model requests <http-api-routing.rst#routed-external-model-requests>`__ instead.

An inference response from a served Reef artifact carries ``x-reef-artifact-id``, the ID of the artifact that answered. Routed external models have no such header. A client learns of a new harness artifact from the harness routes (`Harness artifacts <http-api-harness.rst#harness-artifacts>`__).

Evaluation routes
-----------------

The evaluation routes are ``/reef/scenarios/{scenario}/evaluation/v1/{route}`` and, for a candidate of one component of a composed snapshot, ``.../components/{component}/evaluation/v1/{route}``. They answer from the scenario's active model and keep no record. The answer has no ``training`` block. An unknown scenario, or one that serves no model, returns 404. A composite recipe's evaluation rollouts call these routes; a recipe of one component calls its runtime's endpoint directly. When auth is configured, these calls present the token that the service issues the recipe at startup. That token opens the evaluation routes and no other route. A restart issues a new token.

Artifact of a recorded response
-------------------------------

An inference record holds the provider exchange and ``artifact_id``, the artifact that answered: the same ID as ``x-reef-artifact-id``. The samples service's staleness check reads it (see `Staleness <../developer-guide/recipe-samples-service.rst#staleness>`__). Serving returns no training tensors. Reef removes any ``training`` block that an engine handler returns from the record, from the client's answer, and from rollout and evaluation answers.

Report
------

``POST /reef/report`` stores four fields and drops other top-level keys. Put harness-specific data inside ``metadata`` or ``feedback``.

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
     - Unique existing inference receipts in this scenario: the last call of each trajectory that the report scores, usually one. Reef expands each reference to the calls of its trajectory. Reported-feedback processors require at least one reference.
   * - ``metadata``
     - object
     - no
     - Recipe metadata; ``training.eligible`` is not supported. ``metadata.task`` (``name``, with ``path`` and ``digest`` when known) or ``metadata.harbor.task_name`` names the Harbor task the rollout ran; the assembled training sample carries it as ``task``.

It answers ``{agent_record_id, scenario, request_type}``.

.. code:: python

   client.report("hello-reef", {
       "agent_record_id": "myharness:run42:trial7",
       "score": 1.0,
       "references": ["abc123"],
       "metadata": {"harbor": {"trial_id": "run42:7"}},
   })

The optional top-level ``agent_record_id`` makes a retry safe. Reef uses it as the report's own record ID: an identical resend returns the stored record without processing it again, and the same ID with different content returns 409. It is not a receipt. Receipts go in ``references``.

A recipe may declare a report schema, as `tttd <../user-guide/recipes/tttd.rst#the-report-contract>`__ does. Reef then validates the declared ``score`` and ``metadata`` fields at ingress and answers 400 on a violation. ``feedback`` and undeclared ``metadata`` keys pass through. The removed ``metadata.training.eligible`` field returns 400 whatever its value: reports cannot opt out of training.

Missing, foreign, non-inference or duplicate references return 400, and Reef stores no report. So does a reference inside the trajectory of another reference: a report references only the last call of each trajectory. Reef does not wait for future records; the producer may resubmit after the referenced inference is stored. An identical retry of an accepted report keeps its original behavior after reference cleanup; it does not restart training.
