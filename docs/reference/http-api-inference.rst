HTTP API: inference and feedback
================================

Send model requests, track startup, and report feedback using response receipts. See `HTTP API <http-api.rst>`__ for routes and headers.

Startup status
--------------

Creation and activation return before startup publishes the artifact's route. Poll status with the inference bearer token:

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

**Select an artifact.** ``model`` defaults to ``active``; active aliases and served artifact IDs also work. Unknown names or absent startup work return 404; unreadable state returns 503.

Status describes the selected artifact, even while an older route handles requests. Reads start no work; status is not a health check.

**State.** Startup covers all served artifacts. Published routes stay ``ready`` while other artifacts start or fail.

- ``pending``: waiting for a worker.
- ``starting``: downloading, converting or loading, including retries.
- ``ready``: startup succeeded and the route is published.
- ``failed``: startup ended with an error.

``error`` is null unless ``failed``: then it contains the runtime's ``StartupFailedError`` message or fixed text. Logs provide details.

**Startup progress.** ``startup`` contains ``steps`` and ``estimate_seconds``. Phases run in this order:

1. ``prepare``: wait for serving work, deploy the engine, and convert or quantize weights.
2. ``start``: allocate a machine or container and transfer weights to it.
3. ``load``: load weights into GPU memory, run kernels and CUDA graphs, and warm up.

- ``steps``: reported phases, each ``{phase, started_at, ended_at, detail}``, in order. Times use epoch seconds; only the last step can have a null ``ended_at``. ``detail`` is a progress note or null.
- ``estimate_seconds``: estimated seconds until warm, or null when unavailable.

Pending/starting artifacts report the request's ``prepare`` phase, then the runtime's phases. Ready artifacts report the handler's phases while an engine or replica starts. Otherwise, ``startup`` is null.

**Engine status.** ``engine`` and ``engine_replicas`` describe ready routes; each is null otherwise or when unreported. ``engine`` is:

- ``warm``: answers requests now.
- ``starting``: runs but cannot answer yet.
- ``idle``: no engine runs; the next request starts one and waits.

``engine_detail`` describes a starting engine/replica, for example ``Loading weights · 0:42``; null if unavailable or none starts.

``engine_replicas`` has the form ``{"serving": 1, "starting": 2, "replicas": [...]}``:

- ``serving``/``starting`` count ready/not-yet-ready replicas.
- ``replicas`` lists ``{id, state, started_at, startup, load}``; empty if the handler cannot identify replicas.
- ``id`` identifies the replica; ``started_at`` is its start time in epoch seconds, or null.
- ``startup`` reports ``start``/``load`` phases as above; null unless starting and reported.
- ``load`` is the latest serving-replica measurement: ``{running_requests, waiting_requests, output_tokens_per_second, measured_at}``; null if starting or unavailable. Counts cover running/queued requests; token rate covers the last measurement interval. ``measured_at`` uses epoch seconds.

A warm engine with starting replicas reports one replica's ``engine_detail`` and top-level ``startup``.

**Retries.** Failures return to ``pending`` after 60 seconds, doubling after each failed retry up to one hour. Resend the complete accepted ``served`` mapping to retry immediately.

New startup requests replace current work immediately; replaced work stops within seconds. Scenarios start sequentially, or concurrently with multiple ``serving_slots``.

**Inference during startup.** Without a usable route, requests return 503:

- ``pending`` or ``starting``: ``inference_starting`` with ``Retry-After: 30``.
- ``failed``: ``inference_start_failed``.

Both use ``{"error": {"type", "message"}}``. Control and data operations remain available.

``POST /reef/scenarios/{scenario}/redeploy`` restarts served engines, replacing pending startup work without changing artifact selection or requiring changed settings. It returns ``{scenario}``, even when serving nothing.

Request
-------

Send the provider's body; Reef preserves sampling parameters. Artifacts add no fields. Set ``"stream": true`` for SSE.

Reef selects a serving route:

- ``model`` selects a named served artifact, otherwise the active artifact or a `traffic split <http-api-serving.rst#traffic-split>`__ selection.
- The request keeps this route through activations. Reef uses its engine URL/model name and records its artifact snapshot.
- To require a specific artifact, use ``/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/chat/completions``.
- ``/reef/scenarios/{scenario}/v1/{route}`` equals ``/v1/{route}`` with a scenario header. Client model bindings use it.
- Weight-serving handlers replace the body's ``lora_path`` with the served adapter.

`Run the services in separate processes <../user-guide/split-services.rst>`__ gives the status codes of an inference request.

Reef retries once after an aborted non-streaming answer or model rejection before output. It rereads the route. Artifact routes, artifact IDs in ``model``, and traffic splits keep their artifact; if no longer served, they return 503 with ``Retry-After``. Other retries may reach a new active artifact.

Log probabilities depend on what answers the request:

- **Provider/routed target:** forwards ``logprobs``/``top_logprobs`` and preserves returned values in responses and records.
- **SGLang/vLLM Chat Completions:** ``"logprobs": true`` returns sampled tokens in ``choices[0].logprobs.content``, in output order. Streams send entries with text; records retain them. ``top_logprobs`` includes only the sampled token. SGLang alternatives use its format in ``choices[0].meta_info.output_top_logprobs``, in the last streaming chunk. vLLM returns no alternatives.
- **SGLang/vLLM Messages:** ``"logprobs": true`` returns top-level ``logprobs: {"content": [...]}``, a Reef extension; streams carry it in ``message_delta``.

To receive token IDs from Reef's SGLang or vLLM backend, set ``"return_token_ids": true``:

- Chat Completions returns the rendered prompt in ``prompt_token_ids`` and sampled IDs in ``choices[0].token_ids``. Streams send both in the last chunk.
- Messages returns top-level ``prompt_token_ids`` and ``token_ids``, or carries them in ``message_delta`` when streaming.
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

Stream receipts follow record storage; incomplete streams have none. After headers, these failures close the connection without a terminal event, causing an SDK transport error:

- an engine or transport error;
- a missing terminal event;
- a handler's report that the engine does not serve the selected weights;
- the inference timeout.

Aborted SGLang/vLLM streams keep their terminal event but remain incomplete: Reef cannot restart after sending frames.

Stored incomplete streams retain provider responses but cannot produce training samples, including through references or replay after restart. Receipt-based reports cannot reference them. See also `routed streams <http-api-routing.rst#routed-external-model-requests>`__.

Served Reef artifacts return ``x-reef-artifact-id``; routed external models do not. Clients find harness updates through `harness routes <http-api-harness.rst#harness-artifacts>`__.

Evaluation routes
-----------------

Use ``/reef/scenarios/{scenario}/evaluation/v1/{route}``, or ``.../components/{component}/evaluation/v1/{route}`` for composed snapshot components.

- Routes use the active model, store no record and return no ``training`` block.
- An unknown scenario or a scenario without a served model returns 404.
- Composite recipes use these routes; single-component recipes call the runtime directly.
- Authenticated recipes use an evaluation-only startup token, renewed on restart.

Artifact of a recorded response
-------------------------------

Records hold the provider exchange and ``artifact_id``, matching ``x-reef-artifact-id``, for the `staleness check <../developer-guide/recipe-samples-service.rst#staleness>`__. Serving returns no training tensors: Reef removes handler ``training`` blocks from records and client, rollout and evaluation responses.

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
     - Last inference receipt of each scored trajectory in this scenario. Reef expands each to its trajectory's calls. Reported-feedback processors require at least one.
   * - ``metadata``
     - object
     - no
     - Recipe metadata. ``task`` (``name``, optional ``path``/``digest``) or ``harbor.task_name`` identifies the Harbor task, copied to the sample's ``task``. ``training.eligible`` is unsupported.

It answers ``{agent_record_id, scenario, request_type}``.

.. code:: python

   client.report("hello-reef", {
       "agent_record_id": "myharness:run42:trial7",
       "score": 1.0,
       "references": ["abc123"],
       "metadata": {"harbor": {"trial_id": "run42:7"}},
   })

Optional ``agent_record_id`` is the report's idempotency key. Identical retries return the stored report; different content returns 409. Put inference receipts in ``references``.

Recipe report schemas, such as `tttd's <../user-guide/recipes/tttd.rst#the-report-contract>`__, validate declared ``score``/``metadata`` fields at ingress; violations return 400. ``feedback`` and undeclared metadata pass through. ``metadata.training.eligible`` always returns 400: reports cannot opt out of training.

Missing, foreign, non-inference, duplicate or overlapping references return 400 without storing a report. Reference each trajectory's last call; wait for pending records before resubmitting. Identical retries remain idempotent after reference cleanup and never restart training.
