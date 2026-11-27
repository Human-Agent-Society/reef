HTTP API: inference and feedback
================================

Send model requests, track startup, and report feedback using response receipts. See `HTTP API <http-api.rst>`__ for routes and headers.

Startup status
--------------

``GET /reef/scenarios/{scenario}/inference/status`` describes the selected
artifact, independently of any older route still answering requests.
``model`` defaults to ``active``; served artifact IDs also work.
Unknown names return 404; unreadable state returns 503.

- ``state``: ``pending`` (queued), ``starting``, ``ready`` (route published),
  or ``failed`` with ``error``. Creation and activation return before readiness.
- ``startup``: ``steps`` and ``estimate_seconds``, or null when unreported.
  Steps follow ``prepare`` → ``start`` → ``load``. Each has ``phase``,
  epoch-second ``started_at``/``ended_at``, and optional ``detail``.
  The current step has no end time.
- ``engine``: ``warm``, ``starting`` or ``idle``; null outside ready routes
  or when unreported. An idle engine starts on the next request.
- ``engine_replicas``: serving/starting counts and optional individual replicas.
  Each has ``id``, ``state``, ``started_at``, ``startup`` and ``load``.
  Serving load reports ``running_requests``, ``waiting_requests``,
  ``output_tokens_per_second`` and epoch-second ``measured_at``; otherwise null.
  ``engine_detail`` and top-level startup can describe an additional starting replica.

Without a usable route, inference returns 503: ``inference_starting`` with
``Retry-After: 30``, or ``inference_start_failed``.
Failed startup retries after 60 seconds, doubling to one hour.
Resend the complete ``served`` mapping to retry immediately.

``POST /reef/scenarios/{scenario}/redeploy`` restarts served engines without
changing artifact selection, returning ``{scenario}``. It also replaces pending startup.
Status reads start no work and are not health checks.

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

With `x-reef-record-training: 1 <http-api.rst#headers>`__, the record also keeps the engine's exact sample; the response does not.

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

Stream receipts follow record storage; incomplete streams have none and never train.
Engine/transport failures, timeouts and missing terminal events close the connection
without a terminal event, causing an SDK transport error. Aborted SGLang/vLLM
streams may keep a terminal event but are still incomplete. Streams cannot restart
once frames have been sent. See also `routed streams <http-api-routing.rst#routed-external-model-requests>`__.

Served Reef artifacts return ``x-reef-artifact-id``; routed external models do not. Clients find harness updates through `harness routes <http-api-harness.rst#harness-artifacts>`__.

Evaluation routes
-----------------

Use ``/reef/scenarios/{scenario}/evaluation/v1/{route}``, or ``.../components/{component}/evaluation/v1/{route}`` for composed snapshot components.

- Routes use the active model, store no record and return no ``training`` block.
- An unknown scenario or a scenario without a served model returns 404.
- Authenticated recipes use an evaluation-only startup token, renewed on restart.

Report
------

``POST /reef/report`` stores four fields and drops other top-level keys. Put harness-specific data inside ``metadata`` or ``feedback``.

- ``score``: finite number, excluding booleans.
- ``feedback``: string or object interpreted by the recipe.
- ``references``: last inference receipt of each scored trajectory in this scenario.
  Reef expands each to the trajectory's calls. Reported-feedback recipes require references.
- ``metadata``: recipe-defined object. ``task`` or ``harbor.task_name`` can identify
  the Harbor task. ``training.eligible`` is forbidden.

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
