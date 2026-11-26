HTTP API: inference and feedback
================================

See `HTTP API <http-api.rst>`__ for routes and authentication.

Startup status
--------------

``GET /reef/scenarios/{scenario}/inference/status`` accepts ``model`` as
``active`` (default) or a served artifact ID. Unknown names return 404.
Unreadable state returns 503. Reads start no work and are not health checks.
Published routes stay ready while other artifacts start.

.. code:: mermaid

   flowchart TB
       accTitle: Inference startup and retries
       accDescr: A worker takes pending work and starts the route. The route becomes ready on publication. After final failure, a new startup request returns it to pending.
       pending["pending"] -->|Worker starts| starting["starting"]
       starting -->|"Route<br/>published"| ready(["ready"])
       starting -->|"Final<br/>failure"| failed["failed"]
       failed -.->|"New<br/>request"| pending

Worker retries stay ``starting``. Failed work exposes ``error`` and retries
after 60 seconds, doubling to one hour. Resend the complete ``served`` mapping
for an immediate retry.

- ``startup``: ``{steps, estimate_seconds}``, or null when unreported.
  Phases run from ``prepare`` through ``start`` to ``load``. Steps contain ``phase``,
  ``started_at``, ``ended_at`` and optional ``detail``. The current step has no end time.
- ``engine``: ``warm``, ``starting`` or ``idle``. Null unless ready and reported.
  Idle engines start on request.
- ``engine_replicas``: serving/starting counts and replicas with ``id``, ``state``,
  ``started_at``, ``startup``, ``load``. Serving load reports ``running_requests``,
  ``waiting_requests``, ``output_tokens_per_second`` and ``measured_at``. Otherwise, ``load`` is null.
  ``engine_detail``/``startup`` can describe additional starting replicas.

Timestamps use epoch seconds. Until a route is ready, inference returns 503.
``inference_starting`` includes ``Retry-After: 30``. A failed startup returns ``inference_start_failed``.
``POST /reef/scenarios/{scenario}/redeploy`` replaces pending startup and restarts
served engines without changing selections, returning ``{scenario}``.

Request
-------

Provider bodies retain sampling parameters. Scenario-prefixed routes replace scenario headers.
Set ``"stream": true`` for SSE. See `serving <http-api-serving.rst>`__ for artifact
selection and traffic splits. Requests keep their route through activation.
Weight handlers replace ``lora_path`` with the served adapter.

One retry follows a non-streaming abort or model rejection before output.
Explicit artifacts and splits retain their target, returning 503 with ``Retry-After``
if unserved. Others can reach a new active artifact.

Providers' ``logprobs``/``top_logprobs`` pass through. For SGLang/vLLM, set
``logprobs: true`` and/or ``return_token_ids: true``.

.. list-table:: Returned fields
   :header-rows: 1

   * - API
     - Log probabilities
     - Token IDs
   * - Chat Completions
     - ``choices[0].logprobs.content``. Streams send entries with text.
     - ``prompt_token_ids``, ``choices[0].token_ids``. Streams send them last.
   * - Messages
     - ``logprobs.content`` (Reef extension).
     - ``prompt_token_ids``, ``token_ids``. Streams send IDs and log probabilities in ``message_delta``.

Sampled tokens follow output order. Records retain these fields. Token IDs require the flag.
Each ``top_logprobs`` contains only its sampled token. SGLang alternatives appear in
``choices[0].meta_info.output_top_logprobs``, using its format, in the last streaming
chunk. vLLM supplies no alternatives.

**Record receipts**

- Non-streaming: ``x-reef-agent-record-id`` header.
- OpenAI SSE: ``reef.agent_record_id`` in an empty-``choices`` chunk before ``[DONE]``.
- Anthropic SSE: ``reef.agent_record_id`` on ``message_stop``.

Receipts follow storage. Incomplete streams have no receipt and never train.
On engine failure, timeout or an incomplete upstream stream, clients can receive an SDK transport error.
An aborted SGLang/vLLM stream can still send a terminal event. Reef cannot restart a stream after sending a frame.
Only served Reef artifacts return ``x-reef-artifact-id``.

Evaluation routes
-----------------

``/reef/scenarios/{scenario}/evaluation/v1/{route}`` uses the active model without
records or ``training`` blocks. Composed snapshots also accept
``.../components/{component}/evaluation/v1/{route}``.
Unknown scenarios or absent served models return 404.
Recipe evaluation tokens renew on restart and authorize only these routes.

Report
------

``POST /reef/report`` stores these optional fields and drops other payload keys.

- ``score``: finite number. Booleans are rejected.
- ``feedback``: recipe-interpreted string/object.
- ``references``: each scored trajectory's last inference receipt in this scenario.
  Reef expands trajectories. Reported-feedback recipes require references.
- ``metadata``: recipe object. ``task`` or ``harbor.task_name`` identifies Harbor tasks.

.. code:: python

   client.report("hello-reef", {
       "agent_record_id": "myharness:run42:trial7",
       "score": 1.0,
       "references": ["abc123"],
       "metadata": {"harbor": {"trial_id": "run42:7"}},
   })

Returns ``{agent_record_id, scenario, request_type}``.
Optional ``agent_record_id`` makes identical retries idempotent, even after
reference cleanup. Changed content returns 409. Retries never restart training.

Reef returns 400 for schema violations, ``metadata.training.eligible``,
or missing, foreign, non-inference, duplicate or overlapping references. It stores no report.
Wait for pending inference records before resubmitting. Recipes validate declared
``score``/``metadata`` fields. Feedback and undeclared metadata pass through.
