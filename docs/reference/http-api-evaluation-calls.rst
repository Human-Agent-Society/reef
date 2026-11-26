HTTP API: calls and trajectories of an evaluation
=================================================

Read the model calls that Reef keeps of each rollout of an `artifact evaluation <http-api-evaluation.rst>`__, the calls that a task's verifier writes, and the ATIF trajectories that Reef makes from them.

Model calls of a rollout
------------------------

Reef keeps each model call that a rollout's agent makes to the candidate, failed calls too. It keeps a call when the call ends, so the calls of a running rollout are readable. After each attempt, Reef also keeps the calls that the task's verifier wrote, for example the calls of a judge (`Calls that a verifier writes`_).

Each call is a record item of the scenario's owner (`Evaluation calls <http-api-data-items.rst#evaluation-calls>`__). No scenario trains on it, and record retention does not delete it. After an archive of the scenario, the items stay in the data store, but these routes no longer reach them, because the evaluation is gone.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/calls`` answers the calls of one rollout, without their bodies:

.. code:: json

   {"scenario": "my-agent", "evaluation_id": "1d0e...", "task_index": 3, "repeat": 1,
    "calls": [{"caller": "agent", "label": null, "attempt": 1, "index": 0,
               "path": "/v1/chat/completions", "status": 200, "started_at": 1762590001.2,
               "duration_s": 1.84, "prompt_tokens": 2100, "completion_tokens": 340,
               "finish_reason": "tool_calls", "tool_calls": 1}]}

- ``caller`` is ``agent`` for a call of the agent, and ``verifier`` for a call that the verifier wrote. The route answers the agent's calls by default, and the verifier's calls with ``?caller=verifier``. Another value returns 400.
- ``label`` is the label that the verifier gave the call, else null.
- ``attempt`` is the 1-based attempt of the rollout that made the call, and ``index`` the 0-based order in which the calls of that caller and attempt ended. The list is in attempt and then index order.
- ``status`` is the HTTP status that the agent got. ``started_at`` is in epoch seconds, and ``duration_s`` in seconds.
- ``prompt_tokens``, ``completion_tokens`` and ``finish_reason`` come from the reply of a Chat Completions, Responses or Messages call. Each is null when the reply does not give it, and for a failed call. A Messages call adds its cache tokens to ``prompt_tokens``. The ``finish_reason`` of a Responses call is its ``incomplete_details.reason``, else its ``status``.
- ``tool_calls`` is the number of tool calls in the reply, 0 when none.

The list is empty for a rollout without a kept call. An unknown evaluation returns 404.

``GET .../rollouts/{task_index}/{repeat}/calls/{attempt}/{index}`` answers one call: the fields above, plus ``request``, the JSON body that the agent sent, and ``response``, the JSON body that it got or the error body of a failed call. Each is null when the call had no JSON body. Add ``?caller=verifier`` for a call that the verifier wrote. A call that Reef did not keep returns 404.

Calls that a verifier writes
----------------------------

A judge calls its own model, so its calls do not reach the candidate's endpoint. A task's verifier can write these calls, and the trajectories of an agent judge, to ``/logs/verifier/judge/``. After the verifier runs, Reef keeps the files with the attempt. Both files are optional:

- ``calls.jsonl``: one JSON object for each call, in call order: ``{"request", "response", "status", "started_at", "duration_s", "path", "label"}``. ``request`` is the JSON body that the judge sent, and ``response`` the JSON body that it got, or null. ``status`` is the HTTP status (default 200). ``started_at`` is in epoch seconds, and ``duration_s`` in seconds. ``path`` defaults to ``/v1/chat/completions``. ``label`` is an optional string of 1 to 200 characters, for example ``judge:character_fidelity``. Reef ignores other keys.
- ``trajectory*.json``: ATIF documents, one in each file.

Reef reads the files for a failed verifier too. The files never change the score. A task that sets the verifier's ``include_logs`` or ``exclude_logs`` must keep ``judge/``.

Reef reads at most 32 MiB of these files and keeps at most 2000 calls for each attempt. It skips these and logs how many it skipped:

- a line that is no such call, and a call over the limit;
- a file that is no ATIF document, and a file over the limit;
- a symbolic link.

Trajectories of an evaluation
-----------------------------

Reef makes ATIF trajectories from the kept calls of each attempt of a rollout. It links the calls with status 200 into conversations, as it links the records of training samples: a call continues the earlier call whose messages its own messages start with. An agent that plays several roles, each with its own messages, has one conversation for each role. Reef joins the calls of each conversation in index order, without the calls that a retry replaced, and orders the conversations by their first calls. An attempt without a 200 call has no trajectory.

- Each call is one entry of ``extra.reef.records``. Its ``agent_record_id`` is the call's record ID (`Evaluation calls <http-api-data-items.rst#evaluation-calls>`__), and its ``created_at`` is the call's ``started_at``. Its ``payload`` is the request body without the conversation, the tools and the response (``call_summary``), such as ``model`` and ``chat_template_kwargs``. The steps hold the conversation.
- ``extra.reef.reward`` is the rollout's score for each scored trajectory of the final attempt of a finished rollout. When that attempt includes tool-bearing calls, auxiliary conversations without tools (such as title generation) are not scored and have a null reward. The reward is also null for every other attempt and for a failed score.
- ``extra.reef.evaluation`` is ``{evaluation_id, task, task_index, repeat, attempt, caller, conversation, scored}``. ``conversation`` counts the conversations of the attempt from 0. ``scored`` is true only for scored conversations of the final attempt.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/rollouts/{task_index}/{repeat}/trajectories`` answers ``{scenario, evaluation_id, task_index, repeat, trajectories}``: the trajectories of each attempt of one rollout, in attempt and then conversation order. Add ``?caller=verifier`` for the verifier's trajectories; another ``caller`` value returns 400. In each attempt, the verifier's trajectories are the trajectory files that the verifier wrote, in name order, and then the trajectories that Reef makes from the verifier's calls. A verifier trajectory has no reward. Each verifier file has ``extra.reef.evaluation`` as ``{evaluation_id, task, task_index, repeat, attempt, caller, file, scored}``, where ``file`` is the name of its file and ``scored`` is false.

``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}/trajectories`` streams the trajectories of the whole evaluation as ``application/x-ndjson``: one trajectory on each line, in task and repeat order. It sends the agent's trajectories only, and by default the last attempt of each rollout. Add ``?attempts=all`` to send every attempt; another ``attempts`` value returns 400. A file within the upload limits uploads as it is as trajectory items (`Create trajectory items <http-api-data-upload.rst#create-trajectory-items>`__).

Both routes return 404 for an unknown evaluation.
