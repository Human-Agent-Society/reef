HTTP API: evaluate an artifact
==============================

These routes score one artifact of a scenario on the scenario's eval tasks. The artifact is one component of one of the scenario's artifacts, such as the ``weights`` of a model checkpoint. A policy is a harness and a model: the artifact is one side, and the request configures the other. Reef runs the rollouts and stores the result with the scenario. The evaluation never changes what the scenario serves. Its model calls are not records: Reef keeps them only in the evaluation's call log (`Model calls of a rollout <http-api-evaluation-calls.rst#model-calls-of-a-rollout>`__). A scenario can also hold settings for an `Automatic evaluation`_ of each new artifact.

Reef evaluates a model artifact (a ``model`` or ``model_adapter`` component) with a Harbor agent, on a runtime that serves any artifact on request (``InferenceRuntime.serves_any_artifact``). Tinker is the built-in runtime that does. Reef returns 400 for a harness artifact, for a model artifact on a runtime without that capability, and for a scenario whose artifacts also hold a harness.

Start an evaluation
-------------------

First create the tasks as task items in the scenario's ``eval`` split (`Create task items <http-api-data-upload.rst#create-task-items>`__). Then start the evaluation:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/scenarios/my-agent/evaluations \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"artifact_id": "8f3c...", "harness": {"name": "terminus-2"}, "rollouts_per_task": 2}'

Body fields:

- ``artifact_id`` (required): one of the scenario's artifacts, as ``GET /reef/scenarios/{scenario}/artifacts`` lists it.
- ``start_id``: a client ID for safely repeating a start request in this scenario. Use a non-blank string of at most 256 characters without NUL or surrogate code points.
- ``component``: the component to evaluate. Leave it out when the artifacts hold one component with a kind.
- ``harness``: the agent that runs a model artifact (`The harness field`_).
- ``model``: the model binding that a harness artifact runs on. A request sets exactly one of ``harness`` and ``model``.
- ``task_names``: the eval tasks to run, every eval task by default. The stored request names the tasks that the evaluation runs.
- ``collection``: a collection of the scenario's owner (`Benchmarks and collections <http-api-data-sets.rst#benchmarks-and-collections>`__). Reef runs the collection's eval tasks, each one once. A request sets ``collection`` or ``task_names``, not both. An unknown collection, or one with no eval task, returns 400.
- ``num_samples``: the most tasks to run, a positive integer, or null (the default) to run every task that ``task_names`` or ``collection`` selects. Reef ranks tasks by the SHA-256 digests of their names unless you set ``sample_seed``. For a benchmark or a collection with the ``task`` average, Reef takes the first ``num_samples`` in rank order. The stored ``task_names`` names the chosen tasks, and a stored ``collection`` leaves out a benchmark with no chosen task.

  For the ``benchmark`` average, Reef first selects the highest-ranked eval task of each benchmark. A shared task counts once toward ``num_samples`` and covers every benchmark it belongs to. If those choices exceed the budget, Reef starts over and chooses tasks that cover the most uncovered benchmarks first. It uses rank to break ties. Reef returns 400 if this selection cannot cover all benchmarks with eval tasks within the budget. After it covers the benchmarks, Reef fills the remaining places by rank.
- ``sample_seed``: an integer, or null (the default). With a seed, Reef runs the tasks in a random order that the seed fixes: the order of the SHA-256 digests of ``"<sample_seed>:<task name>"``. With ``num_samples``, the seed sets the rank for selection, including benchmark coverage. Without ``num_samples``, Reef runs every task in rank order. With the same tasks, membership and settings, each run selects the same tasks in the same order. The stored ``task_names`` are in run order.
- ``rollouts_per_task``: the rollouts of each task, 1 by default.
- ``environment``: the Harbor environment type, ``docker`` by default.
- ``max_concurrent_rollouts``: the most rollouts of this evaluation that run at once, or null (the default) for no limit. Reef adds the next rollouts to the queue as earlier ones end. On ``e2b``, Reef adds a task's rollouts only after it built the task's sandbox templates, and a build holds no place of this limit.
- ``max_failed_fraction``: the largest fraction of failed rollouts, a number greater than 0 and less than 1, or null (the default) for no limit. A rollout fails when its work fails or when it ends with no score. After at least 10 rollouts end (or all of them, for a smaller evaluation), Reef stops the evaluation when more than this fraction of the ended rollouts failed. It fails with an error such as ``too many rollouts failed: 37 of 45 ended (82%), over the limit of 50%``. Each rollout that did not end fails with ``stopped: too many rollouts failed``, and the rollouts that wait never start.
- ``log_probs``: ``true`` to keep the sampled tokens' log probabilities and token IDs in each model call of the evaluation; ``false`` by default (`Log probabilities of an evaluation`_).

Reef returns:

- 202 with the stored evaluation, whose initial ``state`` is ``running``;
- 409 for a ``start_id`` already in use with other inputs;
- 400, and runs nothing, for a request that the scenario cannot run: an unknown field or task, the wrong side configured, a component that is no side of a policy, an artifact or a runtime that Reef refuses, or no eval task;
- 404 for an unknown scenario, artifact or evaluation.

Repeat the request with the same ``start_id`` to get the same evaluation in its current state, including after a restart. JSON object key order and whitespace can differ. Changing a value or adding or removing a field returns 409 for a start ID already in use. If the request omits ``start_id``, each successful POST creates an evaluation. If an interrupted start has no initial work after the queue retention period, Reef marks the evaluation failed. A replay still returns its original ID.

``GET /reef/scenarios/{scenario}/evaluation-capabilities`` on control returns ``{"evaluation_start_id": true}`` when the scenario store supports keyed starts. An unsupported store returns false and rejects requests with ``start_id`` with HTTP 501.

Reef sets no limit on the evaluations that run at once: the evaluations queue holds the rollouts that wait. An evaluation process runs at most ``evaluation.max_sandboxes`` rollouts at once, across all evaluations (`Configuration <configuration.rst>`__).

The harness field
-----------------

``harness`` has one of three forms:

- ``{"name": "terminus-2"}`` runs Reef's default terminus-2 configuration.
- Any other Harbor agent is its whole Harbor configuration, with ``{model}``, ``{base_url}`` and ``{api_key}`` where Reef fills them in, as the Tinker job parameter ``rollout.agent`` takes it. A Harbor agent takes variables in its own ``env`` key.
- ``{"item": "<item id>"}`` runs a harness tree: a published artifact item of the scenario's owner, also from another scenario.

Each agent runs in the task's container. On ``docker``, the container reaches the model at ``evaluation.agent_host``. In a remote environment, such as ``modal``, the agent reaches the model through Reef's sandbox relay, also for a task with several compose services.

These rules apply to an item:

- The item has the kind ``harness_plugin``, and ``adapter`` in its metadata. Its adapter must install with npm, uv or git and call the runtime's API dialect.
- Reef refuses a task that declares MCP servers or skills.
- The evaluation does not reach the activation rule, and an automatic evaluation cannot use an item.
- ``"env": {"NAME": "value"}`` gives each command of the agent these variables. A name is a shell variable name, and a value is text without ``${``. ``REEF_CONTEXT_WINDOW`` and ``REEF_MAX_OUTPUT_TOKENS`` also set the model's token limits in the rendered tree, each a positive number of tokens (`Token limits <../developer-guide/harness-adapters.rst#token-limits>`__).
- ``"config": {"<target>": {...}}`` merges each JSON object into the tree's ``config/<target>.json``, as if the tree held it. The tree with the config must pass the adapter's validation (`Config checks <../developer-guide/harness-adapters.rst#config-checks>`__), so the config sets no key that Reef's model binding writes. Otherwise Reef returns 400.

Log probabilities of an evaluation
----------------------------------

With ``log_probs: true``, Reef adds the request fields that ask for log probabilities and token IDs to each call that any agent sends, over the agent's own values:

- ``/v1/chat/completions``: ``logprobs``, ``return_token_ids`` and ``return_prompt_token_ids``, all ``true``.
- ``/v1/messages``: ``logprobs`` and ``return_token_ids``, all ``true`` (Reef's extension; see `Inference <http-api-inference.rst>`__).
- ``/v1/responses``: ``message.output_text.logprobs`` in ``include``.

The kept call holds the engine's answer with these fields, and its ATIF trajectory step holds them in ``metrics``. Each call's body grows with one entry for each sampled token.

The candidate's engine must accept the fields. Reef's SGLang and vLLM handlers accept all of them. SGLang's and vLLM's OpenAI servers accept the Chat Completions and Responses fields. A provider such as Anthropic's API refuses unknown fields, so do not set ``log_probs`` for an artifact that a provider serves.

Read an evaluation
------------------

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}`` returns the evaluation, also while it runs:

.. code:: json

   {"evaluation_id": "1d0e...", "state": "finished",
    "request": {"artifact_id": "8f3c...", "component": "weights", "harness": {"name": "terminus-2"},
                "model": null, "task_names": ["sum-391"], "collection": null, "rollouts_per_task": 2,
                "environment": "docker",
                "max_concurrent_rollouts": null, "max_failed_fraction": 0.5},
    "created_at": 1793169552.2, "finished_at": 1793169840.7,
    "policy_evaluation": {"harness": {"base": "terminus-2", "plugin": {}},
                          "model": {"name": "Qwen/Qwen3-8B", "adapter": "tinker://.../sampler"},
                          "scores": [{"task": "sum-391", "scores": [1.0, 0.0]}]},
    "mean_score": 0.5, "benchmark_scores": [], "error": null,
    "progress": {"total": 2, "ended": 2, "scored": 2, "failed": 0, "cancelled": 0, "running": 0,
                 "mean_score": 0.5, "eta_seconds": null, "benchmarks": []}}

- ``state`` is ``queued``, ``running``, ``finished`` or ``failed``.
- A finished evaluation holds its ``policy_evaluation``: the harness and the model that its rollouts ran, and each rollout's score by task (`Rollout <glossary.rst#rollout>`__). A failed one holds its ``error``.
- ``mean_score`` is the mean of the rollouts that have a score: a failed rollout does not count as 0.
- An evaluation whose rollouts the evaluations queue lost when the service stopped, as an in-memory queue does, reads as ``failed``. Reef does not resume it.

``GET /reef/scenarios/{scenario}/evaluations`` lists the evaluations, newest first, at most ``limit`` (50 by default).

For an evaluation on a collection, the stored ``collection`` is ``{"name", "average", "benchmarks"}``. ``benchmarks`` gives ``{"benchmark", "task_names"}`` for each benchmark that has a task in the evaluation. ``benchmark_scores`` gives ``{"benchmark", "task_count", "mean_score"}`` for each of them: the number of its tasks in the evaluation, and the mean of their rollouts that have a score. A task in two benchmarks counts in both. With the average ``task``, ``mean_score`` is the mean of all rollouts that have a score. With ``benchmark``, it is the mean of the benchmark scores. Reef skips benchmarks without a score. The built-in ``evaluated`` activation rule compares this mean score with ``min_mean_score``. ``benchmark_scores`` is empty for another evaluation.

``progress`` counts the rollouts of the evaluation, also while it runs. It is null when the evaluations queue no longer holds them:

- ``total`` is the evaluation's tasks times ``rollouts_per_task``.
- ``ended`` counts the rollouts that finished or failed. ``scored`` counts the ended rollouts with a score, and ``failed`` the others.
- ``cancelled`` counts the rollouts that a cancel or a stop for failed rollouts ended. They are not in ``ended``.
- ``running`` counts the rollouts that a worker holds now.
- ``mean_score`` applies the evaluation's weighting to the scores so far. It is null when there are no scores.
- ``eta_seconds`` estimates the seconds until every rollout ends: the rollouts left at the rate at which rollouts ended in the last 20 minutes. It is null when no rollout ended in that time, or when the evaluation ended.
- ``benchmarks`` gives the progress of each benchmark of an evaluation on a collection, in the collection's order: ``{"benchmark", "total", "ended", "scored", "failed", "cancelled", "running", "mean_score"}``, with the same meaning for the rollouts of the benchmark's tasks. A task in two benchmarks counts in both. It is empty for another evaluation.

Use ``score_levels=true`` when reading one evaluation to add ``score_levels`` to its progress and the progress of each benchmark. It counts rollouts with scores of at least one (``full``), between zero and one (``part``), and zero or below (``zero``). ``failed`` counts failed rollouts. ``pending`` counts rollouts that have yet to end. The separate ``cancelled`` progress count covers cancellations and early stops.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts`` answers ``{scenario, evaluation_id, rollouts}``, also while the evaluation runs: each rollout in task and repeat order, as ``{task, task_index, repeat, state, attempt_count, started_at, ended_at, score, rewards, error, calls, verifier_calls}``.

- ``state`` is the rollout's state in the evaluations queue: ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``. It is ``waiting`` while ``max_concurrent_rollouts`` keeps the rollout out of the queue, also when a retry runs the rollout again. On ``e2b``, it is also ``waiting`` while the task's sandbox templates build.
- A rollout whose sandbox the sandbox service refused for want of capacity (HTTP 503 or 429, or no placement before a gateway's 504 or the request's timeout) goes back to ``queued``, and its ``error`` starts with ``waiting for sandbox capacity``. The attempt does not count toward the retry limit or the evaluation's failures. The rollout waits 30 seconds, and twice as long after each later refusal, at most 10 minutes. In that time its worker starts no other rollout, so the service can add capacity. After 24 such waits, a refusal counts as a failed attempt. A separate verifier, which starts after the agent, first waits for capacity in place, for up to 80% of its start time limit. When it still finds none, its rollout goes back to the queue in the same way and runs again from the start, with no score from that attempt.
- ``attempt_count`` is how many times a worker took the rollout.
- ``started_at`` is when a worker first took the rollout, and ``ended_at`` when it finished or failed, in epoch seconds. Each is null until then.
- ``score`` is the rollout's score once it finished, else null.
- ``rewards`` holds every finite reward that the rollout's verifier wrote, by name, for example ``{"reward": 0.9, "rule": 1.0, "judge": 0.8}``. The score is the primary reward among them. It is null for a rollout that has not finished, that failed, whose verifier wrote no reward, or that finished before Reef kept rewards. Reef keeps at most 64. Nothing else reads them.
- ``error`` is why a failed rollout failed, or what a queued rollout waits for, else null.
- ``calls`` holds the totals of the agent's model calls over all attempts of the rollout: ``{count, failed, prompt_tokens, completion_tokens}``. ``failed`` counts the calls whose status is not 200. The token totals add the known counts only. It is null when Reef kept no call of the rollout.
- ``verifier_calls`` holds the same totals of the calls that the rollout's verifier wrote (`Calls that a verifier writes <http-api-evaluation-calls.rst#calls-that-a-verifier-writes>`__), or null.

When a provider rate-limits a model call, Reef retries it in the same sandbox, up to five attempts in total. Reef honors ``Retry-After``. If the next attempt cannot start within 60 seconds of the original call and before its inference timeout expires, the agent receives the refusal.

If a rate or payment limit blocks the last model call and the trial fails or receives no model answer, the rollout ends with a null ``score``. Its ``error`` gives the HTTP status and the reason. Use ``retry`` to run failed rollouts again. For ``rate_limit``, wait before retrying. For ``payment_required``, check the provider balance, spending cap or request size first.

When the inference service fails model calls of a rollout (HTTP 500 or more), the attempt has no score if the last call failed, or if the failed calls took at least 5% of the time from the rollout's first model call to the end of its last. Its ``error`` says how many calls the inference service failed. The rollout runs again as after any failed attempt, and when no attempt is left, it ends with a null ``score``. Fewer or shorter failures leave the score as it is.

Add ``limit`` to get 1 to 500 rollouts. ``offset`` skips that many rollouts and defaults to zero. The response also includes ``offset`` and ``total``, the number of rollouts that match the filters. ``state`` defaults to ``all``. It can also be ``running`` (assigned or running) or ``failed``. The Failed filter includes rollouts with a failed score and excludes cancellations and early stops. ``benchmark`` selects the tasks of one recorded collection benchmark. It matches no task if the evaluation has no mapping for that benchmark. ``sort=score`` orders tasks by their mean score, with unscored tasks last. The default, ``sort=task``, keeps task order. Each task's rollouts keep repeat order. Filtered pages and pages sorted by score include ``task_means``. It gives the mean of all scored repeats for each task on the page, by task index. For each page, call totals use only the rollouts it returns. The default page reads only the selected rollout rows. A state filter or score sort reads all rollout rows. A benchmark filter alone reads all repeats for the tasks on the page to compute their means. Without ``limit``, the route gives the original full response.

For the totals, Reef reads every kept call of the evaluation. The query ``calls=false`` leaves out ``calls`` and ``verifier_calls``, which keeps the read fast. The query value is ``true`` (the default) or ``false``. Another value returns 400. An unknown evaluation returns 404.

Cancel, retry and resume
------------------------

Control answers these three routes. An unknown scenario or evaluation returns 404.

``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/cancel`` cancels one evaluation. It answers the evaluation after the cancel, as ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}`` returns it:

- A queued or running evaluation fails with the error ``cancelled by request``. Each of its rollouts that did not end fails with the same error.
- A worker that runs one of these rollouts stops its trial and the trial's sandbox within seconds.
- The rollouts that wait for ``max_concurrent_rollouts`` do not start. They stay ``waiting`` in the rollouts list until a resume.
- An evaluation that ended stays as it is, so a second cancel changes nothing.

Two routes run rollouts of an evaluation again. Each one answers 202 with the evaluation, which is ``running``:

+------------+----------------------------------------------------------------+
| Route      | Rollouts that it runs again                                    |
+============+================================================================+
| ``retry``  | The failed rollouts: the rollouts whose own run ended with no  |
|            | score. The run failed, or it finished with a failed score.     |
+------------+----------------------------------------------------------------+
| ``resume`` | The rollouts that a cancel or a ``max_failed_fraction`` stop   |
|            | ended, also in the middle of their run, and the rollouts that  |
|            | never started.                                                 |
+------------+----------------------------------------------------------------+

The routes are ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/retry`` and ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/resume``. Both work in the same way:

- Each rollout runs from the start of its task, also a rollout that a cancel stopped in its run.
- Its earlier attempts stay, so its ``attempt_count`` goes on and the calls of each attempt stay readable.
- Both keep ``max_concurrent_rollouts``, unless the body changes it. When the evaluation has fewer free places than rollouts to run, the others are ``waiting`` and go back to the queue as earlier rollouts end.
- A rollout that has a score keeps it. A rollout that has not ended stays as it is.
- An evaluation that runs goes on, with those rollouts in the queue again.
- An evaluation that ended is ``running`` again, with no result and no error.
- When the rollouts are final, Reef ends the evaluation as it ends a first run: it stores the scores, and it gives a finished evaluation to the scenario's activation rule, which can activate the artifact.

The body is optional. With no body, or ``{}``, the evaluation keeps its request. ``{"max_concurrent_rollouts": 8}`` sets the limit, and ``{"max_concurrent_rollouts": null}`` removes it. The value is a positive integer or ``null``, as in a start request. The evaluation keeps the new value in its ``request`` for the rest of its run, also for the rollouts that wait:

.. code-block:: bash

   curl -X POST \
       -H "Authorization: Bearer $REEF_TOKEN" \
       -H 'Content-Type: application/json' \
       -d '{"max_concurrent_rollouts": 8}' \
       "$REEF_URL/reef/scenarios/$SCENARIO/evaluations/$EVALUATION_ID/resume"

A higher limit sends more waiting rollouts to the queue at once. A lower limit does not stop a rollout that runs. Fewer rollouts start until the rollouts that are not final are fewer than the limit.

After a cancel or a ``max_failed_fraction`` stop:

- A retry does not run the cancelled rollouts and does not start the other rollouts. When the failed rollouts end, the evaluation fails again. Its error is ``cancelled by request`` after a cancel. After a stop, the error starts with ``too many rollouts failed``.
- While a cancelled rollout waits for a resume, ``max_failed_fraction`` does not count it.
- To finish an evaluation that too many failures stopped, retry it first, and then resume it. A resume first counts the same failures, so it can stop again.

The two routes return:

- 409 from ``retry`` for an evaluation with no failed rollout;
- 409 from ``resume`` for an evaluation with no cancelled rollout and no rollout that never started;
- 409 for an ended evaluation whose rollouts Reef no longer keeps: Reef deletes the rollouts of an ended evaluation after their retention;
- 400, and change nothing, for a body with another key or with a ``max_concurrent_rollouts`` that is not a positive integer or ``null``.

Automatic evaluation
--------------------

``POST /reef/scenarios/{scenario}/update`` with ``artifact_evaluation`` stores the settings that Reef evaluates each new artifact of the scenario with. The settings are the body of an evaluation request without ``artifact_id``:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/scenarios/my-agent/update \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"artifact_evaluation": {"harness": {"name": "terminus-2"}, "rollouts_per_task": 2}}'

Reef checks the settings against the scenario's active artifact, as it checks a request, but the eval tasks may still be missing. The answer holds the stored settings, or 400 for settings that the scenario cannot run. Send ``"artifact_evaluation": null`` to stop the automatic evaluations.

Each training step that publishes a checkpoint makes a new artifact. Reef then stores an evaluation of it in the ``queued`` state; it is ``running`` once a worker takes one of its rollouts. The evaluations routes list it like any other. An automatic evaluation that cannot run, for example because the eval split holds no task, is not stored; the service logs why.
