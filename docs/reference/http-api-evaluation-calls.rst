HTTP API: calls and trajectories of an evaluation
=================================================

Inspect agent calls, verifier calls and ATIF trajectories from an `artifact evaluation <http-api-evaluation.rst>`__.

Model calls of a rollout
------------------------

Agent calls become readable when they end, including failures. Verifier calls appear after each attempt.

These `record items <http-api-data-items.rst#record-items>`__ belong to the scenario's owner, never train and survive record retention. Archiving removes evaluation routes; data-store access remains.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls`` lists calls without bodies:

- ``caller`` query: ``agent`` (default) or ``verifier``; other values return 400.
- ``label``: verifier label, else null.
- ``attempt`` starts at 1; ``index`` at 0 per caller/attempt. Results sort by attempt, then completion index.
- ``status``: received HTTP status. ``started_at``: epoch seconds. ``duration_s``: seconds.
- ``prompt_tokens``, ``completion_tokens``, ``finish_reason``: provider values; null if absent or failed. Messages includes cache tokens in ``prompt_tokens``. Responses uses ``incomplete_details.reason`` or, if absent, ``status`` for ``finish_reason``.
- ``tool_calls``: reply's tool-call count, including zero.

No kept calls returns an empty list; unknown evaluations return 404.

``GET .../rollouts/{task_index}/{repeat}/calls/{attempt}/{index}`` adds ``request``/``response`` JSON bodies, including errors; non-JSON bodies become null. It accepts the same ``caller`` query. Missing calls return 404.

Trajectories of an evaluation
-----------------------------

Trajectories use status-200 calls within each attempt. A call continues a conversation if its messages begin with all the messages from an earlier call.

Calls sort by index, excluding replaced retries; conversations sort by first call. No status-200 call means no trajectory.

- ``extra.reef.records`` contains each call's ``agent_record_id``, ``created_at`` (``started_at``) and ``payload``. Payload excludes conversation, tools and response (``call_summary``); steps hold the conversation.
- ``extra.reef.reward`` holds the rollout score for scored conversations in the final attempt. It is null for other conversations or a failed score.
  In an attempt with multiple conversations, single-turn title generation without tools receives no score. Required tool-less subagents and roles keep the reward.
- ``extra.reef.evaluation`` is ``{evaluation_id, task, task_index, repeat, attempt, caller, conversation, scored}``. ``conversation`` starts at 0; ``scored`` is true only for scored conversations of the final attempt.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/trajectories`` returns ``{scenario, evaluation_id, task_index, repeat, trajectories}`` in attempt, then conversation order.

Use ``?caller=verifier`` for verifier trajectories; other nondefault values return 400.

- Each attempt lists files by name, then trajectories from verifier calls.
- Verifier trajectories have no reward.
- Each verifier file's ``extra.reef.evaluation`` is ``{evaluation_id, task, task_index, repeat, attempt, caller, file, scored}``. ``file`` names the file; ``scored`` is false.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories`` streams ``application/x-ndjson``: one agent trajectory per line, ordered by task/repeat. Defaults to last attempts; ``?attempts=all`` includes all, other values return 400. Files within upload limits can become `trajectory items <http-api-data-upload.rst#create-trajectory-items>`__.

Both routes return 404 for an unknown evaluation.
