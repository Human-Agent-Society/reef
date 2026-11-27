HTTP API: evaluate an artifact
==============================

Evaluate a model artifact on Harbor tasks without changing served inference.
The runtime must support ``serves_any_artifact``; Tinker supports it.
Harness artifacts and scenarios containing a harness return 400.

Start an evaluation
-------------------

Assign `task items <http-api-data-upload.rst#create-task-items>`__ to the scenario's
``eval`` split, then send:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/scenarios/my-agent/evaluations \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"artifact_id": "8f3c...", "harness": {"name": "terminus-2"}, "rollouts_per_task": 2}'

Fields:

- ``artifact_id``: required scenario artifact. ``component`` is required when
  more than one component has an artifact kind.
- ``harness``: agent configuration below. ``model`` is reserved for unsupported
  harness-artifact evaluation; do not send both.
- ``task_names``: selected eval tasks, default all. Alternatively, ``collection``
  selects an owner's collection and deduplicates shared tasks.
- ``num_samples``: positive task limit or null. ``sample_seed``: integer or null.
  Tasks rank by SHA-256 of their name, or ``"<sample_seed>:<name>"``.
  Stored ``task_names`` preserves selection and order.
- ``rollouts_per_task``: repeats, default 1. ``environment``: default ``docker``.
- ``max_concurrent_rollouts``: positive limit or null. Global capacity is
  ``evaluation.max_sandboxes`` per process; excess rollouts wait.
- ``max_failed_fraction``: fraction strictly between 0 and 1, or null.
  After 10 rollouts end, or all for smaller runs, exceeding it stops evaluation.
- ``log_probs``: default false; see below.
- ``start_id``: optional idempotency key, nonblank, at most 256 characters,
  without NUL or surrogates. Identical retries return the same evaluation,
  including after restart; changed fields return 409. Omission starts another run.

For collection ``benchmark`` averaging, sampling first covers every benchmark.
Shared tasks count once. If necessary, choose tasks covering the most uncovered
benchmarks, breaking ties by rank; fill remaining places by rank.
Insufficient coverage returns 400. ``task`` averaging takes the first ranked tasks.

Success returns 202 with a ``running`` evaluation. Unknown scenarios/artifacts
return 404; invalid fields, tasks or unsupported runtimes return 400 before execution.
``GET /reef/scenarios/{scenario}/evaluation-capabilities`` returns
``{evaluation_start_id: true}``; unsupported stores return false and reject keys with 501.

The harness field
-----------------

Choose one form:

- ``{"name": "terminus-2"}``: default agent.
- A Harbor agent configuration, with ``{model}``, ``{base_url}`` and ``{api_key}``
  placeholders. Put command variables in ``env``.
- An owner's published ``harness_plugin`` item. Its adapter must install through
  npm, uv or git and support the runtime's dialect. This pi harness sets token
  and config overrides:

  .. code:: json

     {"item": "<pi-item-id>",
      "env": {"REEF_CONTEXT_WINDOW": "32768"},
      "config": {"primary": {"defaultTools": []}}}

Item harnesses cannot use tasks declaring MCP servers or skills, automatic
evaluation, or activation. Optional item fields:

- ``env``: shell-variable names and text without ``${``.
  ``REEF_CONTEXT_WINDOW``/``REEF_MAX_OUTPUT_TOKENS`` set positive integer token limits.
  ``REEF_AGENT_TIMEOUT_SECONDS`` caps agent execution and cloud sandbox lifetime
  in positive finite seconds.
- ``config``: objects merged into ``config/<target>.json``. Adapter validation
  still applies; overriding model-binding keys returns 400.

Log probabilities of an evaluation
----------------------------------

``log_probs: true`` overrides agent requests with:

- Chat Completions: ``logprobs``, ``return_token_ids``, ``return_prompt_token_ids`` = true.
- Messages: ``logprobs``, ``return_token_ids`` = true, using Reef extensions.
- Responses: ``message.output_text.logprobs`` in ``include``.

Calls stream normally. Stored responses and ATIF metrics combine streamed
token IDs and log probabilities. Handlers without streaming return one JSON piece.
Use providers supporting these fields; Anthropic rejects Reef's extensions.

Read an evaluation
------------------

``GET /reef/scenarios/{scenario}/evaluations`` lists newest first, with ``limit``
defaulting to 50. Add ``/{evaluation_id}`` for one evaluation:

- ``state``: ``queued``, ``running``, ``finished`` or ``failed``.
- ``request``: normalized inputs. ``created_at``/``finished_at`` use epoch seconds.
- ``policy_evaluation``: ``{harness, model, scores}`` when finished.
  Failed evaluations contain ``error``.
- ``mean_score``: mean of scored rollouts; missing scores never count as zero.
  Collection ``benchmark`` averaging averages benchmark means; ``task`` averages rollouts.
  ``benchmark_scores`` lists each benchmark's task count and mean; otherwise empty.
- ``progress``: ``total``, ``ended``, ``scored``, ``failed``, ``cancelled``,
  ``running``, ``mean_score``, ``eta_seconds`` and per-benchmark counts.
  Cancellations are excluded from ``ended``. Progress becomes null after retention.

On the list, ``summary=true`` gives each evaluation without its large parts:
``request`` has no ``task_names``, its ``collection`` has no ``benchmarks``, and
``policy_evaluation`` keeps only ``harness``. ``tasks_digest`` is the SHA-256 of the
compact JSON of ``[task_names, collection name, collection benchmarks, collection average]``
with sorted keys. Evaluations with the same digest ran the same tasks with the same averaging.

``score_levels=true`` adds ``full`` (≥1), ``part`` (0 < score < 1), ``zero`` (≤0),
``failed`` and ``pending`` counts. Shared tasks count in every containing benchmark.

``GET .../evaluations/{evaluation_id}/rollouts`` returns
``{scenario, evaluation_id, rollouts}``, ordered by task/repeat. Each row includes
task/index/repeat, state, attempts, epoch-second times, score, named verifier
``rewards``, error, and agent/verifier call totals. Missing scores/rewards are null.
Calls stay outside training; inspect them through `evaluation calls <http-api-evaluation-calls.rst>`__.

Queries:

- ``limit``: 1–500; omit for all rows. ``offset`` defaults to 0.
  Paginated results add ``offset`` and matching ``total``.
- ``state``: ``all``, ``running`` or ``failed``; cancellations are separate.
- ``benchmark``: recorded collection member; unknown names match nothing.
- ``sort``: ``task`` or ``score``; scores sort by task mean, unscored last.
  Filtered/sorted pages add ``task_means`` by task index.
- ``calls=false`` omits expensive call totals; other than true/false returns 400.

Rollouts wait for concurrency and sandbox capacity. Capacity refusals back off
from 30 seconds to 10 minutes; after 24 waits they count as failed attempts.
Given-back attempts increase ``stopped_attempt_count`` but do not consume the attempt limit.

Rate-limited calls retry up to five attempts in the same sandbox, honoring
``Retry-After``, within 60 seconds and the inference timeout.
Final rate/payment refusals leave null scores. Auxiliary title calls cannot
mask main-loop refusals; a refused trailing auxiliary call fails only an errored trial.
HTTP 5xx failures invalidate an attempt when the last call fails or failed calls
consume at least 5% of the first-to-last-call interval. Normal attempt retries apply.

Cancel, retry and resume
------------------------

Under ``POST /reef/scenarios/{scenario}/evaluations/{evaluation_id}/``:

- ``cancel`` stops unfinished work and sandboxes, setting ``cancelled by request``.
  Completed evaluations stay unchanged. Returns the evaluation.
- ``retry`` restarts failed, unscored rollouts.
- ``resume`` restarts cancelled, early-stopped and never-started rollouts.

Retry/resume return 202 with ``running`` state. Earlier calls and attempt counts
remain; scored or unfinished rollouts are unchanged. Completion applies the
normal activation rule. An omitted body or ``{}`` keeps concurrency; a positive
``max_concurrent_rollouts`` changes it, and null removes the limit.
Lowering the limit does not stop running work.

After ``max_failed_fraction`` stops evaluation, retry failures before resuming.
Retry alone leaves cancellations untouched and the evaluation fails again.
Unknown evaluations return 404; no eligible work or expired rollout history
returns 409; invalid bodies return 400 without changes.
Lost queued work after restart is reported as failure, not silently resumed.

Automatic evaluation
--------------------

Set ``artifact_evaluation`` through the scenario update route using the request
fields above, except ``artifact_id``. Set null to disable it.
Settings validate against the active artifact; missing eval tasks are allowed.
Published checkpoints queue evaluations. Unrunnable evaluations are logged
without a stored run. The ``evaluated`` activation rule uses their mean score.
