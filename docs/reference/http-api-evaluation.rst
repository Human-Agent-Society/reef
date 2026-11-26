HTTP API: evaluate an artifact
==============================

Score an artifact on the scenario's eval tasks. A policy combines a model and a harness: select the artifact and configure its counterpart.

Reef runs the candidate separately from served inference. On completion, the scenario's activation rule can activate it, except when the harness comes from an item. Model calls stay outside training records (`Model calls of a rollout <http-api-evaluation-calls.rst#model-calls-of-a-rollout>`__). Use `Automatic evaluation`_ to score each new artifact.

Evaluation supports ``model`` and ``model_adapter`` components with Harbor agents. The runtime must support ``InferenceRuntime.serves_any_artifact``; Tinker is the built-in option. Unsupported runtimes, harness artifacts, and scenarios whose artifacts also contain a harness return 400.

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
- ``num_samples``: maximum tasks, a positive integer or null (default) for all selected tasks. Unless ``sample_seed`` is set, rank is the SHA-256 digest of each task name. Benchmarks and collections with ``task`` averaging take the first ``num_samples`` by rank. Stored ``task_names`` lists the selection; stored ``collection`` omits benchmarks without selected tasks.

  With ``benchmark`` averaging, Reef first selects each benchmark's highest-ranked eval task. Shared tasks count once and cover all their benchmarks. If this exceeds the budget, Reef instead prioritizes tasks covering the most uncovered benchmarks, breaking ties by rank. If it cannot cover every benchmark with eval tasks, it returns 400. Remaining places are filled by rank.
- ``sample_seed``: an integer or null (default). A seed ranks tasks by SHA-256 of ``"<sample_seed>:<task name>"`` for both selection and run order. Without ``num_samples``, all tasks run in that order. Identical tasks, membership and settings produce the same selection and order. Stored ``task_names`` preserves run order.
- ``rollouts_per_task``: the rollouts of each task, 1 by default.
- ``environment``: the Harbor environment type, ``docker`` by default.
- ``max_concurrent_rollouts``: maximum concurrent rollouts for this evaluation, or null (default) for no limit. New rollouts enter the queue as earlier ones end. On ``e2b``, sandbox templates must build first; builds use no rollout slots.
- ``max_failed_fraction``: a number strictly between 0 and 1, or null (default) for no limit. Failed work and rollouts without scores count as failures. Reef checks after 10 rollouts end, or all rollouts for smaller evaluations. If the fraction exceeds the limit, evaluation fails with ``too many rollouts failed`` and the counts. Unfinished rollouts fail with ``stopped: too many rollouts failed``; waiting rollouts never start.
- ``log_probs``: ``true`` to keep the sampled tokens' log probabilities and token IDs in each model call of the evaluation; ``false`` by default (`Log probabilities of an evaluation`_).

Reef returns:

- 202 with the stored evaluation, whose initial ``state`` is ``running``;
- 409 for a ``start_id`` already in use with other inputs;
- 400, and runs nothing, for a request that the scenario cannot run: an unknown field or task, the wrong side configured, a component that is no side of a policy, an artifact or a runtime that Reef refuses, or no eval task;
- 404 for an unknown scenario, artifact or evaluation.

Repeating the same ``start_id`` and inputs returns the existing evaluation's current state, including after restarts. Key order and whitespace can differ; changed, added or removed fields return 409. Without ``start_id``, every successful POST creates an evaluation.

If an interrupted start has no initial work after queue retention expires, Reef marks it failed. Replays still return its original ID.

``GET /reef/scenarios/{scenario}/evaluation-capabilities`` on control returns ``{"evaluation_start_id": true}`` when the scenario store supports keyed starts. An unsupported store returns false and rejects requests with ``start_id`` with HTTP 501.

Concurrent evaluations are unlimited. Each evaluation process runs at most ``evaluation.max_sandboxes`` rollouts across all evaluations; others wait in the queue (`Configuration <configuration.rst>`__).

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

For collection evaluations:

- Stored ``collection`` has ``{"name", "average", "benchmarks"}``. Each benchmark with selected tasks contributes ``{"benchmark", "task_names"}``.
- ``benchmark_scores`` contains ``{"benchmark", "task_count", "mean_score"}``: selected task count and mean of scored rollouts. Shared tasks count in every benchmark they belong to.
- With ``average: task``, ``mean_score`` averages all scored rollouts. With ``average: benchmark``, it averages benchmark scores, excluding benchmarks without scores.
- The built-in ``evaluated`` activation rule compares this mean with ``min_mean_score``.

Other evaluations return an empty ``benchmark_scores``.

``progress`` counts the rollouts of the evaluation, also while it runs. It is null when the evaluations queue no longer holds them:

- ``total`` is the evaluation's tasks times ``rollouts_per_task``.
- ``ended`` counts the rollouts that finished or failed. ``scored`` counts the ended rollouts with a score, and ``failed`` the others.
- ``cancelled`` counts the rollouts that a cancel or a stop for failed rollouts ended. They are not in ``ended``.
- ``running`` counts the rollouts that a worker holds now.
- ``mean_score`` applies the evaluation's weighting to the scores so far. It is null when there are no scores.
- ``eta_seconds`` estimates the seconds until every rollout ends: the rollouts left at the rate at which rollouts ended in the last 20 minutes. It is null when no rollout ended in that time, or when the evaluation ended.
- ``benchmarks`` gives the progress of each benchmark of an evaluation on a collection, in the collection's order: ``{"benchmark", "total", "ended", "scored", "failed", "cancelled", "running", "mean_score"}``, with the same meaning for the rollouts of the benchmark's tasks. A task in two benchmarks counts in both. It is empty for another evaluation.

Use ``score_levels=true`` when reading one evaluation to add ``score_levels`` to its progress and the progress of each benchmark. It counts rollouts with scores of at least one (``full``), between zero and one (``part``), and zero or below (``zero``). ``failed`` counts failed rollouts. ``pending`` counts rollouts that have yet to end. The separate ``cancelled`` progress count covers cancellations and early stops.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts`` answers ``{scenario, evaluation_id, rollouts}``, also while the evaluation runs: each rollout in task and repeat order, as ``{task, task_index, repeat, state, attempt_count, stopped_attempt_count, deferred_attempt_count, started_at, ended_at, score, rewards, error, calls, verifier_calls}``.

- ``state`` is the rollout's state in the evaluations queue: ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``. It is ``waiting`` while ``max_concurrent_rollouts`` keeps the rollout out of the queue, also when a retry runs the rollout again. On ``e2b``, it is also ``waiting`` while the task's sandbox templates build.
- If the sandbox service has no capacity, the rollout returns to ``queued``. Its ``error`` starts with ``waiting for sandbox capacity``.
  Other rollouts can start while it waits. The retry delay starts at 30 seconds and doubles up to 10 minutes.
  After 24 capacity waits, another refusal counts as a failed attempt.
  If a separate verifier cannot get a sandbox, the rollout starts over without a score from that attempt.
  On ``e2b``, the verifier first waits for up to 80% of its start time limit.
- If the service rejects the sandbox request as invalid, the rollout fails without a score. Its ``error`` gives the service's reason.
- ``attempt_count`` is how many times a worker took the rollout.
- ``stopped_attempt_count`` is how many of those attempts a worker gave back: it found no sandbox capacity, or the worker stopped.
  These attempts do not count toward the attempt limit. ``attempt_count`` minus ``stopped_attempt_count`` is the number of attempts that count.
- ``deferred_attempt_count`` is how many of the given-back attempts found no sandbox capacity.
- ``started_at`` is when a worker first took the rollout, and ``ended_at`` when it finished or failed, in epoch seconds. Each is null until then.
- ``score`` is the rollout's score once it finished, else null.
- ``rewards`` holds every finite reward that the rollout's verifier wrote, by name, for example ``{"reward": 0.9, "rule": 1.0, "judge": 0.8}``. The score is the primary reward among them. It is null for a rollout that has not finished, that failed, whose verifier wrote no reward, or that finished before Reef kept rewards. Reef keeps at most 64. Nothing else reads them.
- ``error`` is why a failed rollout failed, or what a queued rollout waits for, else null.
- ``calls`` holds the totals of the agent's model calls over all attempts of the rollout: ``{count, failed, prompt_tokens, completion_tokens}``. ``failed`` counts the calls whose status is not 200. The token totals add the known counts only. It is null when Reef kept no call of the rollout.
- ``verifier_calls`` holds the same totals of the calls that the rollout's verifier wrote (`Calls that a verifier writes <http-api-evaluation-calls.rst#calls-that-a-verifier-writes>`__), or null.

When a provider rate-limits a model call, Reef retries it in the same sandbox, up to five attempts in total. Reef honors ``Retry-After``. If the next attempt cannot start within 60 seconds of the original call and before its inference timeout expires, the agent receives the refusal.

If a rate or payment limit blocks the last model call, the rollout ends with a null ``score``. When a rollout makes tool-bearing calls, auxiliary calls without tools (such as title generation) do not mask a main-loop refusal, and a refused trailing call without tools fails the rollout only if the trial itself ended with an error. Its ``error`` gives the HTTP status and the reason. Use ``retry`` to run failed rollouts again. For ``rate_limit``, wait before retrying. For ``payment_required``, check the provider balance, spending cap or request size first.

When the inference service fails model calls of a rollout (HTTP 500 or more), the attempt has no score if the last call failed, or if the failed calls took at least 5% of the time from the rollout's first model call to the end of its last. Its ``error`` says how many calls the inference service failed. The rollout runs again as after any failed attempt, and when no attempt is left, it ends with a null ``score``. Fewer or shorter failures leave the score as it is.

Use these queries to page and filter rollouts:

- ``limit``: 1 to 500. Without it, the route returns the original full response.
- ``offset``: number of rollouts to skip, default 0. Paginated responses add ``offset`` and ``total``, the count matching the filters.
- ``state``: ``all`` (default), ``running`` (assigned or running), or ``failed``. Failed scores count as failed; cancellations and early stops do not.
- ``benchmark``: tasks from one recorded collection benchmark. An unknown mapping matches no tasks.
- ``sort``: ``task`` (default) or ``score``. Score sorting orders tasks by mean score, unscored last. Repeats stay in order within each task.

Filtered or score-sorted pages include ``task_means``, keyed by task index, averaging all scored repeats of each displayed task. Call totals cover only returned rollouts. State filters and score sorting read all rollout rows; benchmark filtering reads all repeats of displayed tasks. Default pages read only selected rows.

Call totals require reading the evaluation's retained calls. Set ``calls=false`` to omit ``calls`` and ``verifier_calls`` for faster reads. ``calls`` accepts ``true`` (default) or ``false``; other values return 400. An unknown evaluation returns 404.

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

Omit the body or send ``{}`` to keep existing settings. Set ``max_concurrent_rollouts`` to a positive integer, or null to remove the limit. The stored request and waiting rollouts use the new value:

.. code-block:: bash

   curl -X POST \
       -H "Authorization: Bearer $REEF_TOKEN" \
       -H 'Content-Type: application/json' \
       -d '{"max_concurrent_rollouts": 8}' \
       "$REEF_URL/reef/scenarios/$SCENARIO/evaluations/$EVALUATION_ID/resume"

A higher limit queues more waiting rollouts. Lowering it stops no running rollout; new rollouts wait until capacity becomes available.

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
