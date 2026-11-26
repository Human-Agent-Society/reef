HTTP API: inference and feedback
================================

Send model requests, track startup, and report feedback using response receipts. See `HTTP API <http-api.rst>`__ for routes and headers.

Startup status
--------------

Creation and activation return before the engine is ready. Inference publishes the artifact's route after startup succeeds. Poll status with the inference bearer token:

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

**Select an artifact.** ``model`` defaults to ``active``. It also accepts an active artifact alias or another served artifact's ID (``?model=<artifact-id>``). Unknown names or absent startup work return 404; an unreadable state store returns 503.

Status describes the selected artifact, even while an older route handles requests. Reading starts no work and provides no continuous health check.

**State.** One startup covers all served artifacts, but each published route stays ``ready`` while another artifact starts or fails.

- ``pending``: waiting for a worker.
- ``starting``: inference is downloading, converting or loading the artifact, including retries.
- ``ready``: startup succeeded and the route is published.
- ``failed``: startup ended with an error.

``error`` contains the runtime's ``StartupFailedError`` message, or fixed text for other failures. It is null outside ``failed``. Read service logs for details.

**Startup progress.** ``startup`` contains ``steps`` and ``estimate_seconds``. Every backend uses these phases in order:

1. ``prepare``: wait for serving work, deploy the engine, and convert or quantize weights.
2. ``start``: allocate a machine or container and transfer weights to it.
3. ``load``: load weights into GPU memory, run kernels and CUDA graphs, and warm up.

Progress fields:

- ``steps``: reported phases in order, each ``{phase, started_at, ended_at, detail}``. Unreported phases are omitted.
- ``started_at`` and ``ended_at``: epoch seconds. Only the current, last step can have a null ``ended_at``.
- ``detail``: a short progress note, or null.
- ``estimate_seconds``: estimated seconds until the engine is warm, or null when unavailable.

The source of progress depends on state:

- ``pending`` or ``starting``: the request's ``prepare`` phase, followed by phases the runtime reports for this artifact.
- ``ready`` with a starting engine: phases reported by the inference handler.
- Otherwise: ``startup`` is null, except when additional replicas are starting, as described below.

**Engine status.** For ready artifacts, ``engine`` and ``engine_replicas`` describe the engine behind the route. Each field is null outside ``ready`` or when the handler cannot report it. ``engine`` is:

- ``warm``: answers requests now.
- ``starting``: runs but cannot answer yet.
- ``idle``: no engine runs; the next request starts one and waits.

``engine_detail`` can describe a starting engine, for example ``Loading weights · 0:42``. It is null when unavailable or no engine or replica is starting.

``engine_replicas`` has the form ``{"serving": 1, "starting": 2, "replicas": [...]}``:

- ``serving`` counts replicas that answer requests; ``starting`` counts those running but not yet ready.
- ``replicas`` lists ``{id, state, started_at, startup}``, or is empty when the handler cannot identify replicas.
- ``id`` identifies the replica, such as its container ID. ``started_at`` is its start time in epoch seconds, or null.
- A replica's ``startup`` uses the format above for its ``start`` and ``load`` phases. It is null unless starting and reported.

A warm engine can have additional replicas starting. Then ``engine_detail`` describes one starting replica, and the top-level ``startup`` reports its ``start`` and ``load`` phases.

**Retries.** Control retries failures after 60 seconds, doubling the wait after each failed retry up to one hour. Retries return to ``pending``. Resend the complete accepted ``served`` mapping to retry immediately.

New startup requests replace current work immediately, for example after serving configuration changes. Replaced work stops within seconds as the runtime checks for cancellation. Scenarios start sequentially, or concurrently with multiple ``serving_slots``.

**Inference during startup.** Without a usable route, requests return 503:

- ``pending`` or ``starting``: ``inference_starting`` with ``Retry-After: 30``.
- ``failed``: ``inference_start_failed``.

Both use ``{"error": {"type", "message"}}``. Control and data operations remain available.

``POST /reef/scenarios/{scenario}/redeploy`` restarts the scenario's served engines, even when their settings have not changed. Use it to recover from a hang. The request replaces pending startup work with the same artifact selection. It returns ``{scenario}``, including for a scenario that serves nothing.

Request
-------

Send the provider's request body. Reef preserves sampling parameters; artifacts add no request fields. For streaming, set ``"stream": true`` and read the SSE response.

Reef answers the request from the scenario's row in ``served_models``:

- The request's ``model`` selects the row of a served artifact when it names one. Any other ``model`` selects the active row, or the row that a `traffic split <http-api-serving.rst#traffic-split>`__ gives the request.
- Reef keeps that row for the whole request, even if an activation completes during it. It sends the request to the engine at the row's URL under the row's model name, and records the snapshot of the row's artifact.
- To require a specific artifact, use ``/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/chat/completions``.
- ``/reef/scenarios/{scenario}/v1/{route}`` is the same call as ``/v1/{route}``, with the scenario in the path instead of the header. A client install's model binding calls it.
- On a weight-serving deployment, the engine handler selects the served adapter itself, and it replaces a ``lora_path`` that the body names.

`Run the services in separate processes <../user-guide/split-services.rst>`__ gives the status codes of an inference request.

Reef retries once if the engine aborts a non-streaming answer or rejects the selected model name before output. It reads the serving row again. Artifact routes, artifact IDs in ``model``, and traffic splits keep their original artifact. If that artifact is no longer served, the retry returns 503 with ``Retry-After``. Other retries can reach a new active artifact.

Log probabilities depend on what answers the request:

- **A provider or routed target.** Reef forwards ``logprobs`` and ``top_logprobs`` in the body. It preserves the provider's returned log probabilities in the response and the record.
- **Reef's SGLang or vLLM backend on Chat Completions.** With ``"logprobs": true``, ``choices[0].logprobs.content`` lists sampled tokens in output order. Streams send entries with their text; records retain the same entries. Each ``top_logprobs`` contains only the sampled token. SGLang returns alternatives in ``choices[0].meta_info.output_top_logprobs``, in its own format and in the last streaming chunk. vLLM returns no alternatives.
- **Reef's SGLang or vLLM backend on Messages.** Buffered requests with ``"logprobs": true`` get top-level ``logprobs: {"content": [...]}``, a Reef extension. Streamed messages have no log probabilities.

To receive token IDs from Reef's SGLang or vLLM backend, set ``"return_token_ids": true``:

- Chat Completions returns the rendered prompt in ``prompt_token_ids`` and sampled IDs in ``choices[0].token_ids``. Streams send both in the last chunk.
- Buffered Messages returns top-level ``prompt_token_ids`` and ``token_ids``.
- Records retain the same fields. Without the flag, responses contain no token IDs.

In ATIF records, the reply step's ``metrics`` stores ``prompt_token_ids``, ``completion_token_ids`` and each token's ``logprob`` in ``logprobs``.

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

Reef sends a stream's receipt only after storing its record. Incomplete streams have no receipt. After headers are sent, these failures close the connection without a terminal event, causing an SDK transport error:

- an engine or transport error;
- a missing terminal event;
- a handler's report that the engine does not serve the selected weights;
- the inference timeout.

A stream from Reef's SGLang or vLLM backend that the engine aborts keeps its terminal event. It is still incomplete, because Reef cannot restart a stream after it sent frames.

Reef stores incomplete streams with their provider responses, marked as incomplete delivery. They cannot produce training samples, even through direct report references or replay after restarts. Harnesses reporting by receipt cannot reference them. Routed streams follow `Routed external-model requests <http-api-routing.rst#routed-external-model-requests>`__.

An inference response from a served Reef artifact carries ``x-reef-artifact-id``, the ID of the artifact that answered. Routed external models have no such header. A client learns of a new harness artifact from the harness routes (`Harness artifacts <http-api-harness.rst#harness-artifacts>`__).

Evaluation routes
-----------------

The evaluation routes are ``/reef/scenarios/{scenario}/evaluation/v1/{route}`` and ``.../components/{component}/evaluation/v1/{route}`` for a component of a composed snapshot.

- They use the scenario's active model and store no record. Responses have no ``training`` block.
- An unknown scenario or a scenario without a served model returns 404.
- Composite recipes use these routes for evaluation rollouts. Single-component recipes call their runtime's endpoint directly.
- With authentication enabled, recipes use a startup token restricted to evaluation routes. A restart issues a new token.

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

Use the optional top-level ``agent_record_id`` as an idempotency key for the report. An identical retry returns the stored report without processing it again. Reusing the ID with different content returns 409. Put inference receipts in ``references``.

A recipe may declare a report schema, as `tttd <../user-guide/recipes/tttd.rst#the-report-contract>`__ does. Reef then validates the declared ``score`` and ``metadata`` fields at ingress and answers 400 on a violation. ``feedback`` and undeclared ``metadata`` keys pass through. The removed ``metadata.training.eligible`` field returns 400 whatever its value: reports cannot opt out of training.

Invalid references return 400 and store no report. This includes missing, foreign, non-inference, duplicate, or overlapping references within one trajectory. Reference only the last call of each trajectory. If an inference record is not stored yet, resubmit after it arrives. An identical retry remains idempotent after reference cleanup and does not restart training.
