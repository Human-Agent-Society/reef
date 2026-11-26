HTTP API: evaluate an artifact
==============================

Evaluate ``model`` or ``model_adapter`` artifacts with a Harbor agent on eval tasks. The runtime must support ``InferenceRuntime.serves_any_artifact`` (built-in: Tinker). Unsupported runtimes, harness artifacts and scenarios also containing a harness return 400.

Candidates run separately from served inference. Completion can trigger activation, except with an item-based harness. Calls remain outside training records (`Evaluation calls <http-api-evaluation-calls.rst>`__). `Automatic evaluation`_ scores new artifacts.

Start an evaluation
-------------------

First create the tasks as task items in the scenario's ``eval`` split (`Create task items <http-api-data-upload.rst#create-task-items>`__). Then start the evaluation:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/scenarios/my-agent/evaluations \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"artifact_id": "8f3c...", "harness": {"name": "terminus-2"}, "rollouts_per_task": 2}'

Body fields:

- ``artifact_id`` (required): a listed artifact of this scenario.
- ``start_id``: optional idempotency key, nonblank, maximum 256 characters, without NUL or surrogate code points.
- ``component``: omit when exactly one component has an artifact kind.
- ``harness``: the model's agent (`The harness field`_).
- ``model``: a harness artifact's model binding. Supply exactly one of ``harness`` and ``model``; harness artifact evaluation is currently unsupported.
- ``task_names``: selected eval tasks, default all; stored requests record the selection.
- ``collection``: an owner's `collection <http-api-data-sets.rst#benchmarks-and-collections>`__, mutually exclusive with ``task_names``. Deduplicates tasks shared by benchmarks. Unknown collections or collections without eval tasks return 400.
- ``num_samples``: positive task limit or null (default). Tasks rank by SHA-256 of their names unless seeded. With ``task`` averaging, take the first ranked tasks. Stored ``collection`` omits benchmarks without selected tasks.

  ``benchmark`` averaging first selects each benchmark's highest-ranked task; shared tasks count once. If over budget, select tasks covering most uncovered benchmarks, breaking ties by rank. Incomplete coverage returns 400. Fill remaining places by rank.
- ``sample_seed``: integer or null (default). Seeds rank by SHA-256 of ``"<sample_seed>:<task name>"``. Identical inputs reproduce selection and order. Stored ``task_names`` preserves run order, with or without ``num_samples``.
- ``rollouts_per_task``: repeats per task, default 1.
- ``environment``: Harbor environment, default ``docker``.
- ``max_concurrent_rollouts``: positive concurrency limit or null (default). Completed rollouts release slots. E2B template builds precede queueing and consume no slots.
- ``max_failed_fraction``: strict fraction between 0 and 1, or null (default). After 10 rollouts end, or all for smaller evaluations, exceeding it stops evaluation. Failed work and missing scores count. Errors start with ``too many rollouts failed``; unfinished rollouts get ``stopped: too many rollouts failed``. Waiting rollouts never start.
- ``log_probs``: retain token IDs and log probabilities, default false (`Log probabilities of an evaluation`_).

Reef returns 202 with the stored evaluation in state ``running``.
An unknown scenario, artifact or evaluation returns 404.
Invalid fields, task selections, model–harness combinations, components and runtimes return 400 before any rollout starts.

Reuse ``start_id`` to retrieve the same evaluation, including after a restart.
Changing a field returns 409. Whitespace and key order can change.
Omit the ID to start a new evaluation. If the initial queued work expired after an interruption, replay keeps the ID and reports failure.

``GET /reef/scenarios/{scenario}/evaluation-capabilities`` returns ``{"evaluation_start_id": true}`` for keyed-start support. Otherwise it returns false, and ``start_id`` requests return 501.

Concurrent evaluations are unlimited. Each evaluation process runs at most ``evaluation.max_sandboxes`` rollouts across all evaluations; others wait in the queue (`Configuration <configuration.rst>`__).

The harness field
-----------------

``harness`` has one of three forms:

- ``{"name": "terminus-2"}`` runs Reef's default terminus-2 configuration.
- For another Harbor agent, use its full Harbor configuration, as in Tinker's ``rollout.agent``. Reef fills in ``{model}``, ``{base_url}`` and ``{api_key}``. Put environment variables in the agent's ``env``.
- ``{"item": "<item id>"}`` runs a harness tree: a published artifact item of the scenario's owner, also from another scenario.

Agents run in task containers. Docker uses ``evaluation.agent_host``; remote environments use Reef's sandbox relay, including multi-service tasks.

These rules apply to an item:

- The item has the kind ``harness_plugin``, and ``adapter`` in its metadata. Its adapter must install with npm, uv or git and call the runtime's API dialect.
- Reef refuses a task that declares MCP servers or skills.
- The evaluation does not reach the activation rule, and an automatic evaluation cannot use an item.
- ``env`` supplies command environment variables: shell-variable names and text without ``${``. ``REEF_CONTEXT_WINDOW`` and ``REEF_MAX_OUTPUT_TOKENS`` set positive integer token limits in the rendered tree (`Token limits <../developer-guide/harness-adapters.rst#token-limits>`__). ``REEF_AGENT_TIMEOUT_SECONDS`` caps agent execution and cloud sandbox lifetime in positive finite seconds.
- ``config`` merges JSON objects into ``config/<target>.json``. Adapter `validation <../developer-guide/harness-adapters.rst#config-checks>`__ still applies: model-binding keys cannot be overridden. Invalid config returns 400.

Log probabilities of an evaluation
----------------------------------

With ``log_probs: true``, Reef sets these fields on each agent request, overriding the agent's values:

- ``/v1/chat/completions``: ``logprobs``, ``return_token_ids`` and ``return_prompt_token_ids``, all ``true``.
- ``/v1/messages``: ``logprobs`` and ``return_token_ids``, all ``true`` (Reef's extension; see `Inference <http-api-inference.rst>`__).
- ``/v1/responses``: ``message.output_text.logprobs`` in ``include``.

Evaluation calls stream the engine's own response, including with ``log_probs``. Capture combines streamed token IDs and log probabilities into the stored response and ATIF step ``metrics``. Size grows per sampled token. Handlers without native streaming supply their JSON response as one piece.

Reef's SGLang/vLLM handlers accept all fields; their OpenAI servers accept Chat Completions and Responses fields. Do not enable this for providers rejecting extensions, such as Anthropic.

Read an evaluation
------------------

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}`` returns current state:

- ``evaluation_id``, normalized ``request``, and epoch-second ``created_at`` and ``finished_at``.
- ``state`` is ``queued``, ``running``, ``finished`` or ``failed``.
- ``policy_evaluation`` holds ``harness``, ``model`` and ``scores`` (``{task, scores}`` per task) on success; failed evaluations hold ``error``.
- ``mean_score`` is the mean of the rollouts that have a score: a failed rollout does not count as 0.
- If a service restart loses queued rollouts, Reef marks the evaluation ``failed``. This can happen with an in-memory queue.

``GET /reef/scenarios/{scenario}/evaluations`` lists the evaluations, newest first, at most ``limit`` (50 by default).

For collection evaluations:

- Stored ``collection`` has ``{"name", "average", "benchmarks"}``. Each benchmark with selected tasks contributes ``{"benchmark", "task_names"}``.
- ``benchmark_scores`` contains ``{"benchmark", "task_count", "mean_score"}``: selected task count and mean of scored rollouts. Shared tasks count in every benchmark they belong to.
- With ``average: task``, ``mean_score`` averages all scored rollouts. With ``average: benchmark``, it averages benchmark scores, excluding benchmarks without scores.
- The built-in ``evaluated`` activation rule compares this mean with ``min_mean_score``.

Other evaluations return an empty ``benchmark_scores``.

``progress`` is null after rollout queue retention; otherwise:

- ``total``: tasks × ``rollouts_per_task``.
- ``ended``: finished or failed; ``scored`` have scores, ``failed`` do not.
- ``cancelled``: cancellations and early stops, excluded from ``ended``.
- ``running``: worker-held rollouts.
- ``mean_score``: weighted score so far, null without scores.
- ``eta_seconds``: remaining rollouts divided by completion rate over the last 20 minutes; null without recent completions or after evaluation ends.
- ``benchmarks``: collection-ordered ``{benchmark, total, ended, scored, failed, cancelled, running, mean_score}``. Shared tasks count in each benchmark; non-collection evaluations return ``[]``.

``score_levels=true`` adds counts to evaluation and benchmark progress: ``full`` (score ≥1), ``part`` (0–1 exclusive), ``zero`` (≤0), ``failed`` and ``pending``. Cancellations remain in separate ``cancelled`` counts.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts`` answers ``{scenario, evaluation_id, rollouts}``, also while the evaluation runs: each rollout in task and repeat order, as ``{task, task_index, repeat, state, attempt_count, stopped_attempt_count, deferred_attempt_count, started_at, ended_at, score, rewards, error, calls, verifier_calls}``.

- ``state``: ``queued``, ``assigned``, ``running``, ``finished`` or ``failed``. ``waiting`` means concurrency-limited, including retries, or E2B templates are building.
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
- ``rewards``: up to 64 finite verifier rewards by name; ``score`` is the primary reward. Null before completion, on failure, without rewards or for older evaluations. Informational only.
- ``error`` is why a failed rollout failed, or what a queued rollout waits for, else null.
- ``calls``: ``{count, failed, prompt_tokens, completion_tokens}`` across all attempts. Non-200 calls fail; token totals include known counts only. Null without retained calls.
- ``verifier_calls``: equivalent totals for `verifier calls <http-api-evaluation-calls.rst#calls-that-a-verifier-writes>`__, or null.

Rate-limited calls retry in the same sandbox, at most five attempts, honoring ``Retry-After``. Retries must start within 60 seconds of the original call and before inference timeout; otherwise the agent receives the refusal.

Last-call rate/payment limits produce null scores and an error with HTTP status and reason. Tool-less auxiliary calls cannot mask tool-bearing refusals. A refused trailing tool-less call fails only an errored trial. Before ``retry``, wait for ``rate_limit`` or resolve provider balance, spending cap or request size for ``payment_required``.

HTTP 5xx calls fail the attempt if the last call failed or failed-call duration reached 5% of the first-to-last-call interval. The error reports failed-call count. Normal attempt retries apply; exhaustion leaves a null score. Service failures below both thresholds leave the score unchanged.

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

- Selected rollouts restart their tasks. Earlier calls remain readable and ``attempt_count`` continues.
- Concurrency limits apply; excess rollouts wait. Scored and unfinished rollouts stay unchanged.
- Running evaluations continue; ended evaluations return to ``running`` with result and error cleared.
- Completion stores scores and applies activation as on the first run.

Omit the body or send ``{}`` to keep existing settings. Set ``max_concurrent_rollouts`` to a positive integer, or null to remove the limit. The stored request and waiting rollouts use the new value:

.. code-block:: bash

   curl -X POST \
       -H "Authorization: Bearer $REEF_TOKEN" \
       -H 'Content-Type: application/json' \
       -d '{"max_concurrent_rollouts": 8}' \
       "$REEF_URL/reef/scenarios/$SCENARIO/evaluations/$EVALUATION_ID/resume"

A higher limit queues more waiting rollouts. Lowering it stops no running rollout; new rollouts wait until capacity becomes available.

After a cancel or a ``max_failed_fraction`` stop:

- ``retry`` leaves cancelled and unstarted rollouts untouched. Evaluation fails again afterward with ``cancelled by request`` or ``too many rollouts failed``.
- Cancelled rollouts awaiting resume do not count toward ``max_failed_fraction``.
- After too many failures, retry first, then resume; resuming first can hit the same failure limit.

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

Settings validate against the active artifact, allowing missing eval tasks. Success returns stored settings; invalid settings return 400. Send null to disable.

Published checkpoints queue evaluations, which become ``running`` when a worker takes a rollout. Unrunnable evaluations, such as those without eval tasks, are logged without being stored.
