HTTP API: training requests and jobs
====================================

Submit training requests, inspect or cancel jobs, and choose a trainer's starting artifact.

Manual training
---------------

``POST /reef/train`` queues a request in either mode; only requests start jobs in ``manual``. Every harness method accepts requests. No receipt or score is needed. A training commit consumes the queued request and stores ``training_request`` metrics; no record is created.

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: coding" \
     -H "Content-Type: application/json" \
     -d '{"agent_record_id":"change-001","text":"Add a skill that runs tests before answering", "session":"session-1", "artifact_id":"artifact-1"}'

The body takes these fields:

- ``text`` (required): the user's request, 1 to 4000 characters.
- ``session`` and ``artifact_id`` (required strings): the originating session and artifact. Training uses current committed state, without restoring this artifact.
- ``agent_record_id``: optional idempotency key. Identical retries return the existing request, including after eviction; changed content returns 409. Omission creates a new ID.
- ``samples``: the samples that the request's job trains (`Choose the samples that train <http-api-samples.rst#choose-the-samples-that-train>`__).
- ``train_from``: starting artifact ID, defaulting to the scenario's training start. Unknown scenario artifacts return 400.
- ``parameters``: a weight recipe only, the parameters of the training job (`Job parameters`_).
- ``requires``: what the change needs where the harness runs (`Requirements of a change`_).
- ``client``: optional machine information, described below.

HTTP 200 acknowledges durable acceptance, not training success: ``{agent_record_id, scenario, request_type: "train", page_path}``. ``page_path`` opens the `request page <http-api-harness.rst#request-page>`__. Normal authentication applies. Errors:

- 400 for empty text, text longer than 4000 characters, or missing or non-string ``session`` or ``artifact_id``;
- 400 for credential-shaped literals or instruction overrides; the error names the rule without storing or repeating the text;
- 400 from a weight recipe for a request without ``samples``;
- 404 for an unknown scenario; implicit creation never applies.

How a request trains
~~~~~~~~~~~~~~~~~~~~

In ``auto`` (default), jobs also start at ``job_samples`` or when ``training.manager`` requests one. Explicit requests take priority, oldest first, one per step, regardless of ``job_samples``:

- Weight recipes train ``samples`` and retain ``text`` in commit metrics.
- Harness methods read the request and its selected traces, possibly none.

``train_from`` decides how the job runs beside other jobs:

- Without ``train_from``, one job runs at a time and advances the training start at each checkpoint.
- With ``train_from``, jobs leave the training start unchanged and submit without waiting. They run concurrently when the backend supports it and ``training.max-jobs`` allows. Tinker supports this.
- Request order is preserved: a waiting request blocks later requests.

A running step ignores later requests. Evaluation and publication rules still apply.

Failed request steps are not retried. The next step commits ``skipped: "request failed"`` and ``error``, consuming only the request. Samples remain available; submit a new request to retry. ``evolution.max_steps`` and ``evolution.max_failure_streak`` count request steps but stop automatic steps only.

Job parameters
~~~~~~~~~~~~~~

``parameters`` is a weight-job override object, maximum 16 KiB. Nested objects merge by key; other values, including lists, replace defaults. Keys cannot be removed. The backend validates the result and fails rejected requests with its reason. An objective is required.

- The Tinker component takes ``objective``, ``learning_rate_schedule``, ``scheduling``, ``batch_size``, ``learning_rate``, ``kl_coef``, ``save_interval`` and ``rollout``; ``rollout`` only on a job of tasks.
- Slime takes ``objective``, ``learning_rate_schedule``, ``scheduling``, ``save_interval``, ``dataset`` and ``harbor``; ``harbor`` only on a job of tasks.

For Slime, ``save_interval`` counts optimizer steps for sample jobs and rollout iterations for both task and dataset jobs. Checkpoint ``step`` uses the same units. Omit ``save_interval`` for one checkpoint at the end.

``GET /reef/scenarios/{scenario}/contract`` exposes the parameter JSON Schema.

Requirements of a change
~~~~~~~~~~~~~~~~~~~~~~~~

Optional ``requires`` lists up to eight ``{name, kind, check, prompt}`` items.
Kinds are ``permission``, ``env``, ``service`` and ``binary``.
Clients run checks after user review; Reef never runs them.
Environment values must not enter requests or trees.
Optional ``check`` names an environment variable for ``env``; otherwise it is
a command whose exit code 0 means success. Optional ``prompt`` permits 200 characters.

The optional ``client`` object supplies ``platform``, ``arch``, ``release``
and up to 64 command-availability booleans. Invalid client fields are ignored.

Training jobs
-------------

``GET /reef/scenarios/{scenario}/jobs`` returns ``{scenario, jobs}``: the scenario's training jobs in the jobs queue, oldest first. The data service answers it, and an unknown scenario returns 404.

Jobs include ``id``, ``state``, ``parameters``, ``sample_count``, ``sample_kind``,
``train_from_artifact_id``, originating ``request``, ``checkpoints``, ``result`` and ``error``.
States are ``queued``, ``assigned``, ``running``, ``finished`` and ``failed``.
``created_at``, ``started_at`` and ``ended_at`` use ISO 8601 UTC; unset times are null.
Checkpoints are ordered by ``step`` and include content and metrics.
``experiment`` links the tracker run when available.

Control answers the two cancel routes below. Both return 404 for an unknown scenario.

``POST /reef/scenarios/{scenario}/jobs/{job_id}/cancel`` returns ``{scenario, job}`` after cancellation:

- A queued job fails at once, with the error ``cancelled by request``.
- A running job fails the same way, and its training worker cancels the run at its next poll.
- A job that ended stays as it is, so a second cancel changes nothing.

The trainer records the failed job's end. Unknown jobs return 404.

``POST /reef/scenarios/{scenario}/requests/{request_id}/cancel`` uses the request's ``agent_record_id``. It returns ``{scenario, request_id, queued, job}``:

- Queued requests fail with ``cancelled by request`` and ``queued: true``. The trainer commits ``skipped: "request failed"`` without reading samples or submitting a job.
- Unfinished jobs are cancelled. ``job`` is the request's newest job afterward, or null.
- Jobs still being prepared are cancelled on entering the queue.
- Completed requests and repeated cancellations change nothing.

A request that the scenario does not have returns 404.

Train from an artifact
----------------------

A trainer starts from its creation artifact and advances to its own outputs.
Harness sync also changes its start.

``POST /reef/scenarios/{scenario}/train-from`` sets the start to an existing artifact and cancels the running job:

Send ``{"artifact_id": "<id>", "component": "<name>"}``; omit ``component``
for a single-component scenario. The response is ``{scenario, component, artifact_id}``.
The active serving selection stays unchanged.

Tinker restores checkpoint/optimizer state; roots without it start a new optimizer.
Harness trainers use the artifact's tree. Slime rejects train-from because workers
continue their own latest checkpoints.

Malformed requests, unsupported artifacts/components and missing durable bytes return 400.
Unknown scenarios/artifacts return 404; pending training commits return 409.
