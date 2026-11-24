HTTP API: training requests and jobs
====================================

Queue a training request for a scenario in ``manual`` mode, read and cancel the scenario's training jobs, and set the artifact that a trainer starts from.

Manual training
---------------

``POST /reef/train`` queues one training request for a scenario in ``data.training_mode: manual``. In harness evolution, every method takes it. A request is not a record: it waits in the deployment's training request queue until a training commit consumes it, and the commit keeps it in its ``training_request`` metrics. The request needs no inference receipts and no score.

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: coding" \
     -H "Content-Type: application/json" \
     -d '{"agent_record_id":"change-001","text":"Add a skill that runs tests before answering", "session":"session-1", "artifact_id":"artifact-1"}'

The body takes these fields:

- ``text`` (required): the user's request, 1 to 4000 characters.
- ``session`` and ``artifact_id`` (required strings): the session and the artifact that the request came from. The backend operates on the current committed state; it does not restore that artifact.
- ``agent_record_id``: names the request and makes a retry safe. An identical resend returns the queued request without another step, including after capacity eviction. The same ID with another body returns 409. Without it, each submission gets a new ID.
- ``samples``: the samples that the request's job trains (`Choose the samples that train <http-api-samples.rst#choose-the-samples-that-train>`__).
- ``train_from``: the artifact ID that the job starts from. Without it, the job starts from the scenario's training start. A value that names no artifact of the scenario returns 400.
- ``parameters``: a weight recipe only, the parameters of the training job (`Job parameters`_).
- ``requires``: what the change needs where the harness runs (`Requirements of a change`_).
- ``client``: a description of the requesting machine (`The requesting machine`_).

The response is ``{agent_record_id, scenario, request_type: "train"}``, plus ``page_path``, the request's page with the query that a browser opens it by (`Request page <http-api-harness.rst#request-page>`__). HTTP 200 acknowledges durable acceptance, not successful training. The normal bearer authentication applies. Reef returns:

- 400 for empty text, text longer than 4000 characters, missing or non-string ``session`` or ``artifact_id``, or a request to an ``auto`` scenario;
- 400 for text that carries a credential-shaped literal or phrasing that overrides instructions. The reason names the rule, never the text, and Reef stores nothing;
- 400 from a weight recipe for a request without ``samples``;
- 404 for an unknown scenario. The route never creates a scenario, whatever ``reef.allow_implicit_scenario_creation`` says.

How a request trains
~~~~~~~~~~~~~~~~~~~~

``auto``, the default, also starts a job on its own: when ``job_samples`` samples wait, or when the deployment's training manager says so (``training.manager``, see `Configuration <configuration.rst>`__). ``manual`` starts a job only for a request. A request trains before the samples that wait, oldest request first, one per step. The step trains the samples that the request names, whatever ``job_samples``:

- A weight recipe trains the request's ``samples``. Its ``text`` goes into the commit's ``training_request`` metrics.
- Harness evolution runs the request with the traces that it names, possibly none. The method reads the request next to them.

``train_from`` decides how the job runs beside other jobs:

- A job without ``train_from`` continues the training start and moves it to each checkpoint that it saves. Reef runs one such job at a time.
- A job with ``train_from`` never moves the training start, and Reef does not wait for another job before it submits it. It runs beside the other jobs when the training component allows it and a training worker is free (``training.max-jobs``). The Tinker component allows it.
- Requests start in the order they came, so a request that waits for the running job also holds the requests after it.

A later request does not change a step that already runs. The evaluation and publication rules still decide whether the result becomes served.

A step that fails with a request, for example on a method error, is not retried. The next step consumes the request with a committed row whose ``skipped`` reads ``request failed`` and whose ``error`` carries the failure, and the queue moves on. Send the request again to run it again. That row consumes the request alone: the samples that were in the step stay held for the next job. The ``evolution.max_steps`` and ``evolution.max_failure_streak`` budgets count every step, request steps included, but stop automatic steps only; a request still runs past them.

The metrics of the commit that answered a request include ``training_request``, with the request ID, text, session, artifact ID and ``requires``:

.. code:: json

   {"training_request": {"id": "change-001", "text": "Text me when the run is blocked",
                         "session": "session-1", "artifact_id": "artifact-1",
                         "requires": [{"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID",
                                       "prompt": "The Twilio account SID"}]}}

The commit metrics also keep the method's summary as ``method_summary``, plain text, and its notes as ``method_notes``, the JSON object that it returned. Reef clips their long text and redacts credential-shaped literals in them, and never reads them to decide. Each is present only when the method returned it non-empty, and each method chooses its own note keys.

Job parameters
~~~~~~~~~~~~~~

``parameters`` sets the parameters of a weight recipe's training job: a JSON object of at most 16 KiB. Reef merges it into the recipe's parameters for the kind of the job's samples: nested objects merge key by key, and every other value of the request wins, lists included. A request cannot remove a key that the recipe sets. Reef does not read the parameters: the training component checks them, as it checks the recipe's. A refused job fails the request with the component's reason. A job with no objective, from the recipe or the request, is refused.

- The Tinker component takes ``objective``, ``learning_rate_schedule``, ``scheduling``, ``batch_size``, ``learning_rate``, ``kl_coef``, ``save_interval`` and ``rollout``; ``rollout`` only on a job of tasks.
- Slime takes ``objective``, ``learning_rate_schedule``, ``scheduling``, ``save_interval``, ``dataset`` and ``harbor``; ``harbor`` only on a job of tasks.

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

``requires`` holds at most 8 ``{name, kind, check, prompt}`` items, none by default. Reef never runs a check. The client runs the checks where the harness runs, after the person read them.

- ``kind`` is one of these:

  - ``permission``: an OS permission that the person grants.
  - ``env``: a variable that the person sets. The extension reads it from the environment, and its value never enters a request or the tree.
  - ``service``: an account or endpoint that the person connects.
  - ``binary``: a program that the person installs. Its ``name`` is the program, looked for on the PATH.

- ``name`` matches ``^[A-Za-z0-9][A-Za-z0-9._-]*$``. The pages list the item under it.
- ``check`` is optional. For ``env`` it is the variable name. For the other kinds it is a shell command whose exit status zero means satisfied. For ``env``, the variable named (the check, else the name) is a shell identifier, ``^[A-Za-z_][A-Za-z0-9_]*$``.
- ``prompt`` is optional: one sentence that tells the person what to enter or grant, at most 200 characters. Reef strips it and drops a blank one. The pages show it beside the item.

The credential and instruction-override checks run over every ``name``, ``check`` and ``prompt`` as they run over ``text``, with the same 400 and a reason that names the rule. A malformed list returns 400 and names the first bad item.

The method may add items of its own in its step's result:

- The backend runs the same checks over each added item. It merges the items that pass, by name, after the person's items into the commit's ``training_request.requires``. The snapshots row and the manifest carry the merged list.
- An added item with the name of a person's item is dropped.
- ``training_request.refused_requires`` lists each item that a check refused as ``{"item": <the item as written>, "reason": "..."}``. The key is absent when nothing was refused.
- A step whose merged list holds more than 8 items is skipped.

The requesting machine
~~~~~~~~~~~~~~~~~~~~~~

An optional ``client`` describes the requesting machine: ``platform``, ``arch`` and ``release`` (short words) and ``commands``, a map of command names to whether each is on the machine's PATH (at most 64). A client fills ``commands`` by looking each name up on the PATH; it runs nothing. The field only informs the method, which can then build for that machine and not for its evaluation sandbox. Reef drops what does not fit that shape and never refuses the request for it. ``training_request.client`` carries what was kept.

Training jobs
-------------

``GET /reef/scenarios/{scenario}/jobs`` returns ``{scenario, jobs}``: the scenario's training jobs in the jobs queue, oldest first. The data service answers it, and an unknown scenario returns 404.

A job object has these keys. A time is an ISO 8601 string in UTC, such as ``2026-10-31T08:15:00.250000+00:00``.

- ``id``: the job ID.
- ``state``: ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``. The last two are final.
- ``attempt_count``: how many times a training worker took the job, an integer from 0.
- ``checkpoints``: the saved checkpoints that the job reported, in step order, each ``{step, path, content, metrics, changed_parts}``. ``path`` is a directory of the training service. ``content`` is the name under which the training service stored the checkpoint's files in the artifact backend; Reef removes that content after it commits the checkpoint. ``changed_parts`` lists the parts of the scenario component that the checkpoint changed from the job's start, ``[]`` for a weight job.
- ``result``: ``{"state": "finished", "metrics": {...}}`` once the job finished, else ``null``.
- ``error``: the error of a failed job, or the last retryable error of a queued job, else ``null``.
- ``parameters``: the job's parameters, a JSON object: the recipe's parameters for the kind of the job's samples, with the request's parameters merged in. The training component owns their names, such as ``objective`` and ``scheduling``.
- ``sample_count``: the number of samples that the job trains, an integer from 0.
- ``sample_kind``: ``trajectory`` or ``task``, the kind of the job's samples. It is ``null`` for a job with no sample.
- ``train_from_artifact_id``: the artifact that the job started from.
- ``request``: ``{id, text, parameters}`` of the training request (``POST /reef/train``) that the job trains. ``parameters`` is what the request gave, ``{}`` when it gave none. It is ``null`` for a job that the ``auto`` training mode made.
- ``created_at``: when control added the job.
- ``started_at``: when a training worker first took the job; ``null`` until then, and for a job from a store that an older Reef wrote.
- ``ended_at``: when the job finished or failed; ``null`` until then.
- ``experiment``: ``{tracker, project, entity, run_id, url}`` of the experiment run that records the job, such as ``{"tracker": "wandb", ...}``. Control writes it when its tracker records the first commit of the job. ``entity`` and ``url`` are ``null`` when the tracker does not know them. It is ``null`` until then, and always ``null`` without an experiment tracker. See `Configuration <configuration.rst>`__, "Commit correlation", for how W&B gives the run's URL.

For example:

.. code-block:: json

   {"id": "4f2a9c1e", "state": "finished", "attempt_count": 1,
    "checkpoints": [{"step": 1, "path": "/jobs/4f2a9c1e/step-1", "content": "4f2a9c1e:1",
                     "metrics": {"loss": 0.41}, "changed_parts": []}],
    "result": {"state": "finished", "metrics": {"loss": 0.41}}, "error": null,
    "parameters": {"objective": "sft", "learning_rate": 1e-05},
    "sample_count": 8, "sample_kind": "trajectory",
    "request": {"id": "train-1", "text": "Prefer tests first", "parameters": {"learning_rate": 1e-05}},
    "created_at": "2026-10-31T08:15:00.250000+00:00",
    "started_at": "2026-10-31T08:15:01.020000+00:00",
    "ended_at": "2026-10-31T08:21:44.900000+00:00",
    "experiment": {"tracker": "wandb", "project": "reef", "entity": "acme",
                   "run_id": "0d4c3b2a19f8e7d6c5b4a39281706f5e",
                   "url": "https://wandb.ai/acme/reef/runs/0d4c3b2a19f8e7d6c5b4a39281706f5e"}}

Control answers the two cancel routes below. Both return 404 for an unknown scenario.

``POST /reef/scenarios/{scenario}/jobs/{job_id}/cancel`` cancels one training job of the scenario. It answers ``{scenario, job}``, with the job as ``GET /reef/scenarios/{scenario}/jobs`` lists it after the cancel:

- A queued job fails at once, with the error ``cancelled by request``.
- A running job fails the same way, and its training worker cancels the run at its next poll.
- A job that ended stays as it is, so a second cancel changes nothing.

The trainer reads the failed job and records its end. A job that the scenario does not have returns 404.

``POST /reef/scenarios/{scenario}/requests/{request_id}/cancel`` cancels one training request of the scenario (``POST /reef/train``). ``request_id`` is the ``agent_record_id`` of the request. It answers ``{scenario, request_id, queued, job}``:

- A request that waits in the queue fails with the error ``cancelled by request``. The trainer consumes it with a commit that has ``skipped: "request failed"`` and that error, reads none of its samples, and submits no job. ``queued`` is ``true``.
- Each job of the request that has not ended is cancelled, as ``POST /reef/scenarios/{scenario}/jobs/{job_id}/cancel`` does. ``job`` is the newest job of the request after the cancel, or ``null`` for a request with no job.
- A job that control adds for the request after the cancel, because it was building the job's data, is cancelled when it enters the queue.
- A request whose job ended stays as it is, so a second cancel changes nothing.

A request that the scenario does not have returns 404.

Train from an artifact
----------------------

Each trainer has a training start: the artifact that its next training job (a weight trainer) or evolve step (a harness trainer) starts from:

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
