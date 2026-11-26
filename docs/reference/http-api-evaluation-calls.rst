HTTP API: calls and trajectories of an evaluation
=================================================

Inspect agent calls, verifier calls and ATIF trajectories from an `artifact evaluation <http-api-evaluation.rst>`__.

Model calls of a rollout
------------------------

Reef stores each agent call when it ends, including failed calls. You can read completed calls while the rollout continues. After each attempt, Reef also stores calls written by the verifier, such as judge calls (`Calls that a verifier writes`_).

Calls are `record items <http-api-data-items.rst#evaluation-calls>`__ owned by the scenario's owner. They do not train, and record retention does not delete them. Archiving the scenario removes its evaluation routes. The items remain available through the data store.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls`` answers the calls of one rollout, without their bodies:

.. code:: json

   {"scenario": "my-agent", "evaluation_id": "1d0e...", "task_index": 3, "repeat": 1,
    "calls": [{"caller": "agent", "label": null, "attempt": 1, "index": 0,
               "path": "/v1/chat/completions", "status": 200, "started_at": 1762590001.2,
               "duration_s": 1.84, "prompt_tokens": 2100, "completion_tokens": 340,
               "finish_reason": "tool_calls", "tool_calls": 1}]}

- ``caller`` selects ``agent`` (default) or ``verifier`` calls. Use ``?caller=verifier`` to read verifier calls. Other values return 400.
- ``label`` is the label that the verifier gave the call, else null.
- ``attempt`` starts at 1. ``index`` starts at 0 and orders completed calls within that caller and attempt. Results sort by attempt, then index.
- ``status`` is the HTTP status that the agent got. ``started_at`` is in epoch seconds, and ``duration_s`` in seconds.
- ``prompt_tokens``, ``completion_tokens`` and ``finish_reason`` come from the provider response. Missing fields and failed calls return null. Messages adds cache tokens to ``prompt_tokens``. Responses uses ``incomplete_details.reason`` for ``finish_reason``, or ``status`` when that reason is absent.
- ``tool_calls`` is the number of tool calls in the reply, 0 when none.

The list is empty for a rollout without a kept call. An unknown evaluation returns 404.

``GET .../rollouts/{task_index}/{repeat}/calls/{attempt}/{index}`` returns one call with the fields above plus ``request`` and ``response`` JSON bodies. ``response`` includes failed calls' error bodies. Either field is null when its body was not JSON. Use ``?caller=verifier`` for verifier calls. Missing calls return 404.

Calls that a verifier writes
----------------------------

A verifier can record calls to its own judge model under ``/logs/verifier/judge/``. Reef stores these files with the attempt after verification. Both formats are optional:

- ``calls.jsonl``: one object per call, in call order: ``{"request", "response", "status", "started_at", "duration_s", "path", "label"}``. ``request`` and ``response`` are JSON bodies; ``response`` can be null. ``status`` defaults to 200 and ``path`` to ``/v1/chat/completions``. ``started_at`` uses epoch seconds; ``duration_s`` uses seconds. Optional ``label`` has 1 to 200 characters, such as ``judge:character_fidelity``. Other keys are ignored.
- ``trajectory*.json``: ATIF documents, one in each file.

Reef reads these files even when verification fails. They do not change the score. Keep ``judge/`` when setting the verifier's ``include_logs`` or ``exclude_logs``.

Each attempt can supply up to 32 MiB of files and 2000 calls. Reef skips the following and logs the counts:

- invalid call lines and calls above the limit;
- invalid ATIF files and files above the limit;
- a symbolic link.

Trajectories of an evaluation
-----------------------------

Reef builds trajectories from calls with status 200, separately for each attempt. A call continues a conversation when its messages start with an earlier call's messages. Separate histories, such as those of different agent roles, produce separate conversations.

Calls appear in index order, excluding calls replaced by a retry. Conversations sort by their first calls. An attempt without a status-200 call has no trajectory.

- Each call is one entry of ``extra.reef.records``. Its ``agent_record_id`` is the call's record ID (`Evaluation calls <http-api-data-items.rst#evaluation-calls>`__), and its ``created_at`` is the call's ``started_at``. Its ``payload`` is the request body without the conversation, the tools and the response (``call_summary``), such as ``model`` and ``chat_template_kwargs``. The steps hold the conversation.
- ``extra.reef.reward`` is the rollout's score for each trajectory of the attempt that was scored: the last attempt of a finished rollout. It is null for every other attempt, and for a failed score.
- ``extra.reef.evaluation`` is ``{evaluation_id, task, task_index, repeat, attempt, caller, conversation, scored}``. ``conversation`` counts the conversations of the attempt from 0. ``scored`` is true only for the attempt that was scored.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/trajectories`` returns ``{scenario, evaluation_id, task_index, repeat, trajectories}`` in attempt, then conversation order.

For verifier trajectories, use ``?caller=verifier``. Other nondefault ``caller`` values return 400.

- Each attempt lists verifier files in name order, followed by trajectories reconstructed from verifier calls.
- Verifier trajectories have no reward.
- Each verifier file's ``extra.reef.evaluation`` is ``{evaluation_id, task, task_index, repeat, attempt, caller, file, scored}``. ``file`` names the file; ``scored`` is false.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories`` streams agent trajectories as ``application/x-ndjson``, one per line in task and repeat order. It defaults to each rollout's last attempt. Use ``?attempts=all`` for all attempts; other values return 400. Exported files within upload limits can be uploaded as `trajectory items <http-api-data-upload.rst#create-trajectory-items>`__.

Both routes return 404 for an unknown evaluation.
