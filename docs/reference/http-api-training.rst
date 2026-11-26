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
- ``client``: a description of the requesting machine (`The requesting machine`_).

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

Commit ``training_request`` metrics retain ``id``, ``text``, ``session``, ``artifact_id`` and ``requires``. Nonempty ``method_summary`` (text) and ``method_notes`` (method-defined JSON) are informational. Reef truncates long text and redacts credential-shaped literals.

Job parameters
~~~~~~~~~~~~~~

``parameters`` is a weight-job override object, maximum 16 KiB. Nested objects merge by key; other values, including lists, replace defaults. Keys cannot be removed. The backend validates the result and fails rejected requests with its reason. An objective is required.

- The Tinker component takes ``objective``, ``learning_rate_schedule``, ``scheduling``, ``batch_size``, ``learning_rate``, ``kl_coef``, ``save_interval`` and ``rollout``; ``rollout`` only on a job of tasks.
- Slime takes ``objective``, ``learning_rate_schedule``, ``scheduling``, ``save_interval``, ``dataset`` and ``harbor``; ``harbor`` only on a job of tasks.

For Slime, ``save_interval`` counts optimizer steps for sample jobs and rollout iterations for both task and dataset jobs. Checkpoint ``step`` uses the same units. Omit ``save_interval`` for one checkpoint at the end.

``GET /reef/scenarios/{scenario}/contract`` serves the JSON Schema. For example, after an SFT job on uploaded trajectories, run an RL job on uploaded train tasks:

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: coding" \
     -H "Content-Type: application/json" \
     -d '{"agent_record_id":"rl-001","text":"RL on the uploaded tasks","session":"session-1","artifact_id":"artifact-1",
          "samples":["task-a","task-b"],
          "parameters":{"objective":"my_methods.objectives:MyObjective","rollout":{"rollouts_per_task":4}}}'

Requirements of a change
~~~~~~~~~~~~~~~~~~~~~~~~

``requires`` contains up to 8 ``{name, kind, check, prompt}`` items, default empty. After user review, clients run checks on the harness machine; Reef never runs them.

- ``kind`` is one of these:

  - ``permission``: an OS permission.
  - ``env``: an environment variable; its value never enters requests or trees.
  - ``service``: an account or endpoint.
  - ``binary``: a program named by ``name``, found on PATH.

- ``name``: display name matching ``^[A-Za-z0-9][A-Za-z0-9._-]*$``.
- ``check``: optional variable name for ``env``, otherwise a shell command whose exit code 0 means success. For ``env``, ``check`` or fallback ``name`` must match ``^[A-Za-z_][A-Za-z0-9_]*$``.
- ``prompt``: optional displayed instruction, maximum 200 characters; surrounding whitespace and blank values are removed.

The same credential and instruction checks apply to ``name``, ``check`` and ``prompt``. Violations return 400 naming the rule; malformed lists name the first invalid item.

The method may add items of its own in its step's result:

- Valid method items follow user items in ``training_request.requires``, the snapshot row and manifest. User-defined names win.
- Rejected items appear in ``training_request.refused_requires`` as ``{item, reason}``; the key is absent without rejections.
- More than 8 merged items skips the step.

The requesting machine
~~~~~~~~~~~~~~~~~~~~~~

The optional ``client`` describes the requesting machine:

- ``platform``, ``arch`` and ``release``: short words.
- ``commands``: up to 64 command names mapped to whether they exist on PATH. Clients check PATH without running commands.

This informs the method's target machine. ``training_request.client`` retains valid values; invalid values never reject the request.

Training jobs
-------------

``GET /reef/scenarios/{scenario}/jobs`` returns ``{scenario, jobs}``: the scenario's training jobs in the jobs queue, oldest first. The data service answers it, and an unknown scenario returns 404.

Job fields follow. Times use ISO 8601 UTC.

- ``id``: the job ID.
- ``state``: ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``. The last two are final.
- ``attempt_count``: worker assignments, starting at 0.
- ``checkpoints``: ``{step, path, content, metrics, changed_parts}`` in step order. ``path`` is on the training service; temporary artifact-backend ``content`` is removed after commit. ``changed_parts`` lists component parts changed since the job start, ``[]`` for weight jobs.
- ``result``: ``{"state": "finished", "metrics": {...}}`` once the job finished, else ``null``.
- ``error``: the error of a failed job, or the last retryable error of a queued job, else ``null``.
- ``parameters``: merged recipe and request parameters, defined by the backend.
- ``sample_count``: the number of samples that the job trains, an integer from 0.
- ``sample_kind``: ``trajectory`` or ``task``, the kind of the job's samples. It is ``null`` for a job with no sample.
- ``train_from_artifact_id``: the artifact that the job started from.
- ``request``: originating ``{id, text, parameters}``, or null for automatic jobs. Request parameters default to ``{}``.
- ``created_at``: when control added the job.
- ``started_at``: when a training worker first took the job; ``null`` until then, and for a job from a store that an older Reef wrote.
- ``ended_at``: when the job finished or failed; ``null`` until then.
- ``experiment``: ``{tracker, project, entity, run_id, url}``, null until the first tracked commit or without a tracker. Unknown ``entity`` and ``url`` are null. See `Commit correlation <configuration-observability.rst#commit-correlation>`__.

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

Each trainer has a training start: the starting artifact for its next weight-training job or harness evolution step.

- A new trainer starts from its component's creation artifact.
- After each job or step, the trainer moves its training start to its own newest output.
- A harness sync (`Importing a harness <http-api-harness.rst#importing-a-harness>`__) moves the harness trainer's training start to the artifact that the sync made.

``POST /reef/scenarios/{scenario}/train-from`` sets the start to an existing artifact and cancels the running job:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

The body is ``{"artifact_id": "<id>", "component": "<name>"}``. ``component`` names the trainer's component. Leave it out when the scenario has one component; a scenario with several components requires it. The answer is ``{scenario, component, artifact_id}``. Train-from never changes the active selection.

Whether a trainer can start from an artifact depends on the trainer:

- The Tinker component imports the artifact's checkpoint manifest, which names the saved training state with the optimizer state.
- The Slime component refuses every artifact, because its workers continue from their own latest checkpoint.
- A harness trainer starts its next evolve step from the artifact's tree.

A weight artifact without optimizer state, such as a root snapshot made from the default start, starts training with a new optimizer state.

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
