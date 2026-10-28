HTTP API: inference, feedback, and releases
===========================================

Reef serves the provider's own inference routes: OpenAI Chat Completions at
``/v1/chat/completions``, OpenAI Responses at ``/v1/responses``, and Anthropic
at ``/v1/messages``. It forwards each
request to the runtime unchanged. It adds a small set of ``/reef/*`` routes for
feedback, scenarios, artifacts, and status.

The service accepts request bodies up to 64 MiB so coding agents can send
large contexts. Record import routes accept request bodies up to 10 MiB.

For a complete request, receipt, and feedback example, start with the
`inference and feedback quickstart <../getting-started/quickstart.rst>`__.
To put the release routes into practice, follow the `agent harness tutorial
<../user-guide/evolve-your-harness.rst>`__ or the `model weight training guide
<../user-guide/evolve-your-model.rst>`__.

.. code:: bash

   export REEF_TOKEN=reef-local
   curl -f http://127.0.0.1:8900/healthz     # {"ok": true}

Routes
------

+----------------------------------------------------------------------------------+---------------------------------------------------+
| Route                                                                            | Response                                          |
+==================================================================================+===================================================+
| ``GET /healthz``                                                                 | readiness; the only unauthenticated route         |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /v1/chat/completions``                                                    | OpenAI-format inference                           |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /v1/responses``                                                           | OpenAI Responses-format inference                 |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /v1/messages``                                                            | Anthropic-format inference                        |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /v1/messages/count_tokens``                                               | count request tokens; recorded like any inference |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluation/v1/{route}``                        | one of the inference routes above, served for     |
|                                                                                  | the named scenario and kept by nobody: what a     |
|                                                                                  | recipe's evaluation rollouts call; a flat         |
|                                                                                  | release's request hooks stay out, since the       |
|                                                                                  | rollout runs a candidate of it. It serves a       |
|                                                                                  | scenario loaded in this process and never creates |
|                                                                                  | one: 404 for a scenario deleted under its rollout |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/components/{component}/evaluation/v1/{route}`` | the same for a candidate of one component of a    |
|                                                                                  | composed release: every other component's hooks   |
|                                                                                  | run (weights, request defaults), the named one's  |
|                                                                                  | stay out; 404 for a component it does not serve   |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /v1/images``, ``/v1/embeddings``,                                         | multimodal call, relayed by the recipe to its     |
| ``/v1/audio/speech``, ``/v1/decisions``                                          | gateway; not recorded, 501 when it offers none    |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/records``                                                           | import one existing inference or report           |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/records/batch``                                                     | atomically import a batch of records              |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/trajectories``                                                      | upload ATIF trajectories to train on directly     |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/report``                                                            | submit feedback about one or more receipts        |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/train``                                                             | enqueue one training instruction                  |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios``                                                          | every known scenario, active release and harness  |
|                                                                                  | adapter                                           |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios``                                                         | create a scenario explicitly                      |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/update``                                       | update the scenario training mode                 |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract``                                      | what this scenario accepts                        |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/records``                                       | retained record metadata                          |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/commits``                                       | paginated committed metadata                      |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/records/{record_id}``                           | one retained record and its trace payload         |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/artifacts``                                     | ``{scenario, artifacts}``, newest first           |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/activate``                                     | make an existing artifact the one that serves     |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/train-from``                                   | set the artifact a trainer starts its next job or |
|                                                                                  | step from                                         |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/tasks``                                        | add Harbor tasks to the scenario's imported tasks |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/tasks``                                         | imported tasks, their splits, which trained       |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/scenarios/{scenario}/evaluations``                                  | evaluate one artifact of a release                |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations``                                   | artifact evaluations, newest first                |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}``                   | one evaluation: its state and result              |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``DELETE /reef/scenarios/{scenario}``                                            | remove a scenario; its state is archived          |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness``                                                            | the harness tree of the active release            |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/releases``                                                   | the harness release catalog, oldest first         |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/releases/{step}/page``                                       | one HTML page per catalog step: why, design, what |
|                                                                                  | changed, review, result, setup, chain             |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/requests/{record_id}/page``                                  | one HTML page per filed harness request: its      |
|                                                                                  | step's state, then the result; reloads itself     |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/requests/{record_id}/progress``                              | the same reading as JSON: the step's phase, for a |
|                                                                                  | client with no browser to open the page           |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/releases/{step}/records``                                    | retained raw step file inventory or file body     |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``POST /reef/harness/proposals``                                                 | an agent's proposed tree change, admitted or not  |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/install``                                                    | a shell script that installs the tree             |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/harness/adapters``                                                   | every harness adapter this process resolves       |
+----------------------------------------------------------------------------------+---------------------------------------------------+
| ``GET /reef/status``                                                             | training, serving, and storage state              |
+----------------------------------------------------------------------------------+---------------------------------------------------+

Headers
-------

+-----------------------------------+---------------------------------------------------------+
| Header                            | Required for                                            |
+===================================+=========================================================+
| ``x-reef-scenario``               | inference, report, harness manifest, releases and       |
|                                   | proposals; optional on harness install. Names the       |
|                                   | workload a record belongs to.                           |
+-----------------------------------+---------------------------------------------------------+
| ``Authorization: Bearer <token>`` | every route except ``GET /healthz``, when auth is       |
|                                   | configured; with no Authorization header, ``x-api-key`` |
|                                   | carries the token. The evaluation routes also accept    |
|                                   | the token the service issues the recipe at startup.     |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-release-id``             | optional: bind a new scenario to this starting release; |
|                                   | on an existing scenario it must name the bound starting |
|                                   | release, or the request is HTTP 409. Inference always   |
|                                   | answers from the scenario's active release; to pull a   |
|                                   | specific release, use ``?release_id=`` on the harness   |
|                                   | manifest or install route.                              |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-tag-<name>``             | optional on inference: context stored under             |
|                                   | ``metadata.tags``. The ``release`` tag can select the   |
|                                   | record's release; see `Client release tags`_ below.     |
+-----------------------------------+---------------------------------------------------------+
| ``x-reef-rollout: 1``             | optional on inference: a call of a rollout that Reef    |
|                                   | runs itself. It is served from the scenario's           |
|                                   | release like any other call, but no record is kept, and |
|                                   | the response keeps its ``training`` block with the      |
|                                   | runtime load ID that answered. The call must not        |
|                                   | stream (HTTP 400). Only the service token opens it.     |
+-----------------------------------+---------------------------------------------------------+

Client release tags
~~~~~~~~~~~~~~~~~~~

``x-reef-tag-release`` identifies the harness release installed by the client.
Reef uses it as the inference record's ``artifact_ref`` only when all three
conditions hold:

- The scenario serves files for clients to pull.
- Its surface has no inference hooks, and it has no training component.
- The tag names a release in that scenario's catalog.

Otherwise the record names the active release, including when the tag
is absent or unknown. The tag does not change which release serves inference.

For example, a harness client installed release A before the service published
B. If these conditions hold, its calls tagged with A are recorded against A;
an untagged call or a call tagged with an unknown release is recorded against B.
This tag is separate from ``x-reef-release-id``, which binds a scenario to its
starting release.

Scenario model settings
-----------------------

``POST /reef/scenarios`` accepts an optional ``model`` object with ``url``,
``model``, ``api`` and ``api_key``. It applies only when creating a new
scenario. Existing scenarios retain their settings on repeated creation.
The creation response carries the scenario's ``model`` view.
``POST /reef/scenarios/{scenario}/update`` accepts ``model`` alongside the
existing ``training_mode`` field, and ``artifact_evaluation`` (`Automatic
evaluation`_). Send ``model: null`` to restore deployment
defaults. Responses redact the API key and report ``has_api_key`` instead.

See `Scenario model configuration <../user-guide/scenario-models.rst>`__ for
protocols, persistence and model bindings used throughout evolution.

Import records for cold-start learning
--------------------------------------

``POST /reef/records`` accepts an existing inference or report without calling
an inference provider. It uses the usual Bearer token and ``x-reef-scenario``
header. Submit one record per request and use a stable ``agent_record_id``:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/records \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'x-reef-scenario: my-agent' \
     -H 'Content-Type: application/json' \
     -d '{
       "agent_record_id": "offline-0001",
       "request_type": "inference",
       "payload": {
         "model": "example-model",
         "messages": [{"role": "user", "content": "What is 2 + 2?"}],
         "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]}
       }
     }'

The response is ``{"agent_record_id":"offline-0001","scenario":"my-agent",
"request_type":"inference"}``. HTTP 200 acknowledges record admission; it does
not mean a training step has finished. Identical retries return the existing
receipt, including after capacity eviction; reusing the ID with different content
returns HTTP 409. Authentication failures return 401 and invalid envelopes,
report schemas or references return 400. Scenario selection comes only from
headers; envelope fields other than the three shown above are rejected.

``request_type`` may be ``inference`` or ``report``. Inference payloads follow
the selected recipe's recorded-input contract: native provider request fields
with an existing ``response``, or another complete example that its processor
understands. Import does not synthesize missing training tensors, run inference,
or assign the active artifact to externally generated examples.
Reports use the same schema and reference validation as ``POST /reef/report``;
import referenced inferences first. Training instructions use ``POST /reef/train``.
To upload ATIF trajectories that are already complete, see
`Upload trajectories <#upload-trajectories>`__.

For existing datasets, use the batch endpoint and resumable JSONL importer
below. Both endpoints accept request bodies up to 10 MiB.

Imported records enter the same scenario storage and wake the same training
worker as live traffic. Keep the same scenario header when subsequently calling
``POST /v1/chat/completions``: its records continue through the existing
processor, with no processor replacement or end-of-dataset signal. A processor
that needs reward/report feedback still needs it for both imported and live
records; submit each example's feedback promptly so completed jobs can drain.
For example, feedback for the imported record can be sent as:

.. code:: json

   {"agent_record_id": "offline-score-0001", "request_type": "report",
    "payload": {"references": ["offline-0001"], "score": 1.0}}

Job size, buffering, retention and training policy remain owned by the selected
recipe and processor. Uploading can overlap training; this API adds no dataset
epochs, cold-start completion barrier, or automatic algorithm switch. To start
chat traffic after cold-start training has finished, use the recipe's progress
and commit status to determine when its imported examples have been consumed.

Batch imports and resume
~~~~~~~~~~~~~~~~~~~~~~~~

``POST /reef/records/batch`` accepts ``{"records": [record, ...]}``, where each
record uses the same three-field envelope as ``POST /reef/records``. The batch
must contain 1 to 1000 records and fit within the 10 MiB request-body limit.
All records share the request's scenario, release and tag headers. Put an
inference before reports that reference it, including within the same batch.

Reef validates the entire batch and persists it in one database transaction.
Malformed records or missing references return 400; conflicting IDs return
409. Neither failure persists any new records from the batch. Identical retries
are accepted, including after training or capacity eviction. Oversized requests return
413. The response is ``{"records": [receipt, ...]}``, in input order, with the
same receipt fields as the single-record endpoint. A successful import wakes
the existing consumer once after commit. Atomicity applies to one request,
not the entire dataset or subsequent training.

On the user machine, install the standalone ``reef-client`` package; no Reef
service or training dependencies are needed. For a large local dataset, save
one envelope per line in an immutable UTF-8 JSONL file. Keep original record IDs
and report references; the importer does not generate new IDs or convert
arbitrary dataset schemas.

.. code:: bash

   # Authentication is read from REEF_TOKEN when set.
   reef-client import records.jsonl --url http://127.0.0.1:8900 --scenario my-agent

The CLI reads incrementally and sends batches limited by both ``--batch-size``
(default 128 records) and ``--max-batch-bytes`` (default 524288 bytes). Only a
bounded batch and one look-ahead record are buffered. The defaults are starting
settings, not throughput guarantees. A single JSONL line must fit the byte
limit; increase it, below 1048576, for larger records.

Progress is saved to ``records.jsonl.reef-import.json``, or ``--progress PATH``.
It contains the destination, scenario, source checksum and last acknowledged
byte offset/count; it contains no token or record payloads. Each invocation
first hashes the complete file with bounded memory to verify its identity,
then seeks to the saved offset. The source must remain unchanged during import.
A local file lock prevents simultaneous use of the same checkpoint.

Run the same command after interruption to continue. Transient connection
failures and HTTP 408, 429, 500, 502, 503 or 504 are retried up to ``--retries``
(default 3) with backoff and unchanged IDs. If a reply is lost after commit,
server deduplication prevents the replayed batch from being inserted again.
Permanent errors stop without advancing that batch's checkpoint. Earlier
acknowledged batches remain stored. A modified source or different destination
requires a separate progress file; existing IDs still deduplicate at the server.
The reported count includes every acknowledged source row, including retries
of records already present, and does not count completed training samples.

This importer supports local JSONL on platforms with POSIX file locking, including
Linux and macOS. Parquet/Hugging Face schema conversion and server-side S3 import
jobs are not part of this interface. Import completion acknowledges storage;
use the scenario's training progress before treating cold-start learning as done.

The upload buffer limit does not bound processor memory. Report-based processors
may retain inference records while waiting for feedback; placing all inferences
before all reports can therefore build a large in-memory backlog. Where possible,
place reports close to their referenced inferences. Large-dataset throughput and
end-to-end memory use require measurement with the selected recipe.

Upload trajectories
-------------------

Use this route when your data is already a complete ATIF trajectory. An
uploaded trajectory is not a record. Reef keeps it in the scenario storage.
The recipe's record-to-sample stage renders it as a training sample, and the
sample-to-job stage puts it in the next job's data. The scenario's recipe
must accept uploaded trajectories. If it does not, Reef refuses the upload with
400. For the recipe side, see
`Uploaded trajectories <../developer-guide/processors.rst#uploaded-trajectories>`__.

The route uses the usual Bearer token and ``x-reef-scenario`` header. Send 1
to 1000 trajectories in one request to ``POST /reef/trajectories``. Each one
has a ``trajectory_id`` that you choose. Each document must have an ``ATIF-v``
``schema_version``, an ``agent`` object and at least one step. Reef saves all
of them in one transaction, or none. The request body can be at most 10 MiB:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/trajectories \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'x-reef-scenario: my-agent' \
     -H 'Content-Type: application/json' \
     -d '{"trajectories": [{
       "trajectory_id": "trajectory-0001",
       "trajectory": {
         "schema_version": "ATIF-v1.4",
         "session_id": "trajectory-0001",
         "agent": {"name": "my-agent", "version": "1"},
         "steps": [{"step_id": 1, "source": "user", "message": "What is 2 + 2?"}]
       }
     }]}'

The response is ``{"scenario": "my-agent", "trajectory_ids":
["trajectory-0001"]}``. An identical retry returns the same receipt, also
after training. The same ID with a different trajectory returns 409. An
invalid upload returns 400 and saves nothing. The recipe also checks each
trajectory: a trajectory that it cannot turn into a training sample refuses
the upload with 400, and the message names the trajectory. The SFT recipe
renders each upload with every agent turn trained, except the steps marked
``is_copied_context: true`` (`SFT
<../user-guide/recipes/sft.rst#train-on-whole-conversations>`__).

Each uploaded trajectory counts as one sample toward the pipeline's
``job_samples``. It does not count toward the pipeline's ``buffer_size``, so Reef never
releases it to make room. Job data takes uploaded trajectories first, in
upload order, and then the units of recorded traffic. Thus a large upload can delay the
training of live traffic. The delay continues until the jobs use all of
the upload. Archiving the scenario retires its uploaded trajectories with its
records. Record retention does not delete them.

Upload Harbor tasks
-------------------

Use this route to give a scenario your own Harbor tasks, for training or as
held-out tasks. It adds the tasks to the scenario's task store
``imported-tasks``, the same store that the `task-import section
<../user-guide/operate.rst#import-harbor-tasks>`__ fills. An upload to the
``train`` split needs a recipe that trains on imported tasks, such as the
SFT-then-RL recipe (`SFT then RL <configuration.rst#sft-then-rl>`__). If the
recipe does not, Reef refuses the upload with 400. The ``eval`` split takes
tasks for any recipe, because an `artifact evaluation <#evaluate-an-artifact>`__
runs them.

The body is a gzip-compressed tar archive with ``Content-Type:
application/gzip``. The archive and the files it unpacks to are each at most
256 MiB, in at most 10,000 entries. Each top-level entry is one Harbor task
directory, named by its task name: a lowercase letter or a digit, then
lowercase letters, digits, ``.``, ``_`` and ``-``. The query parameter
``split`` is ``train`` or ``eval``, for every task of the archive:

.. code:: bash

   tar -czf tasks.tar.gz -C /data/benchmark/tasks sum-391 sort-17
   curl -f "http://127.0.0.1:8900/reef/scenarios/my-agent/tasks?split=train" \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/gzip' \
     --data-binary @tasks.tar.gz

The response is ``{"scenario": "my-agent", "split": "train", "tasks":
[{"name": "sort-17", "files_digest": "..."}, {"name": "sum-391",
"files_digest": "..."}]}``, in name order. An identical retry returns the
same receipt. Reef returns:

- 400, and adds nothing, for an archive that is not tar.gz, a link or a device
  in it, a path outside its task directories, a file at the top level, a name
  that is not a task name, more than 1000 tasks or 10,000 entries, more than
  256 MiB unpacked, or a directory that Harbor's task loader refuses;
- 413 for a body larger than 256 MiB;
- 409 for a task name that the scenario holds with other files or in the other
  split. The tasks before it in name order are added; to change a task, give
  it a new name;
- 415 for another content type.

``GET /reef/scenarios/{scenario}/tasks`` returns ``{"scenario", "tasks":
[{"name", "split", "files_digest", "trained"}]}`` in the order the tasks were
added. ``trained`` is true once a task job trained the task.

Evaluate an artifact
--------------------

Use these routes to score one artifact of a scenario on the scenario's eval
tasks. An artifact is one component of one of the scenario's artifacts, such
as the ``weights`` of a model checkpoint. A policy is a harness and a model,
and the artifact is one side of it, so the request configures the other side.
Reef runs the rollouts itself and stores the result with the scenario. The
evaluation never changes what the scenario serves, and its model calls are not
records. A scenario can also hold settings that Reef evaluates each new
artifact with (`Automatic evaluation`_).

This version evaluates a model artifact (a ``model`` or ``model_adapter``
component) with a Harbor agent, on a runtime that serves any artifact on
request: the Tinker runtime. Reef refuses a harness artifact, a model artifact
on another runtime, and a scenario whose artifacts also hold a harness, with
400.

First upload the tasks to the ``eval`` split (`Upload Harbor tasks
<#upload-harbor-tasks>`__). Then start the evaluation:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/scenarios/my-agent/evaluations \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"artifact_id": "8f3c...", "harness": {"name": "terminus-2"}, "rollouts_per_task": 2}'

The body takes these fields:

- ``artifact_id`` (required): one of the scenario's artifacts, by the
  ``artifact_id`` that ``GET /reef/scenarios/{scenario}/artifacts`` lists.
- ``component``: the component to evaluate. Leave it out when the artifacts
  hold one component with a kind.
- ``harness``: the Harbor agent that runs a model artifact. ``{"name":
  "terminus-2"}`` runs Reef's default terminus-2 configuration. Any other
  agent is its whole Harbor configuration, with ``{model}``, ``{base_url}``
  and ``{api_key}`` where Reef fills them in, as the Tinker job parameter
  ``rollout.agent`` takes it.
- ``model``: the model binding that a harness artifact runs on. A request
  sets exactly one of ``harness`` and ``model``.
- ``task_names``: the eval tasks to run. Leave it out to run every eval task.
- ``rollouts_per_task``: the rollouts of each task, 1 by default.
- ``environment``: the Harbor environment type, ``docker`` by default.

The answer is 202 with the stored evaluation, whose ``state`` is ``running``.
``GET /reef/scenarios/{scenario}/evaluations/{evaluation_id}`` returns it as
it runs and after it ends:

.. code:: json

   {"evaluation_id": "1d0e...", "state": "finished",
    "request": {"artifact_id": "8f3c...", "component": "weights", "harness": {"name": "terminus-2"},
                "model": null, "task_names": [], "rollouts_per_task": 2, "environment": "docker"},
    "created_at": 1793169552.2, "finished_at": 1793169840.7,
    "policy_evaluation": {"harness": {"base": "terminus-2", "plugin": {}},
                          "model": {"name": "Qwen/Qwen3-8B", "adapter": "tinker://.../sampler"},
                          "scores": [{"task": "sum-391", "scores": [1.0, 0.0]}]},
    "mean_score": 0.5, "error": null}

``state`` is ``queued``, ``running``, ``finished`` or ``failed``. A finished
evaluation holds its ``policy_evaluation``: the harness and the model that its
rollouts ran, and each rollout's score by task (see `Rollout
<glossary.rst#rollout>`__). A failed one holds its ``error``. An evaluation
that was queued or running when the service stopped reads as ``failed``; Reef
does not resume it. ``GET /reef/scenarios/{scenario}/evaluations`` lists the
evaluations, newest first, at most ``limit`` (50 by default).

Reef returns:

- 400, and runs nothing, for a request that the scenario cannot run: an
  unknown field or task, the wrong side configured, a component that is no
  side of a policy, an artifact or a runtime that this version refuses, or no
  eval task;
- 404 for an unknown scenario, artifact or evaluation;
- 429 when the service runs two evaluations already.

Automatic evaluation
~~~~~~~~~~~~~~~~~~~~

``POST /reef/scenarios/{scenario}/update`` with ``artifact_evaluation`` stores
the settings that Reef evaluates each new artifact of the scenario with. The
settings are the body of an evaluation request without ``artifact_id``:

.. code:: bash

   curl -f http://127.0.0.1:8900/reef/scenarios/my-agent/update \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H 'Content-Type: application/json' \
     -d '{"artifact_evaluation": {"harness": {"name": "terminus-2"}, "rollouts_per_task": 2}}'

Reef checks the settings against the scenario's active artifact, as it checks
a request, but the eval tasks may still be missing. The answer holds the
stored settings, or 400 for settings that the scenario cannot run. Send
``"artifact_evaluation": null`` to stop the automatic evaluations.

Each training step that publishes a checkpoint is a new artifact. Reef then
stores an evaluation of it in the ``queued`` state and runs it when fewer than
two evaluations run. The evaluations routes list it like any other. An
automatic evaluation that cannot run, for example because the eval split holds
no task, is not stored; the service logs why.

Manual training
---------------

``POST /reef/train`` queues one training instruction for a scenario in
``data.training_mode: manual`` or ``hybrid`` (harness evolution with a
proposer that accepts ``requests``). It takes the user's ``text``,
originating ``session`` and ``release_id``. These fields identify the session
and release the request came from. The backend operates on the current
committed state; it does not restore the originating release. The API
requires no inference receipts or score.

A request may also carry ``requires``: what the change needs from the
person's machine, at most 8 ``{name, kind, check, prompt}`` items, default
none. ``kind`` is ``permission`` (an OS permission the person grants),
``env`` (a variable the person sets; the extension reads it from the
environment, and its value never enters a request or the tree) or
``service`` (an account or endpoint the person connects). ``name`` matches
the entry name pattern and is what a check off is recorded under.
``check`` is optional: the variable name for ``env``; for ``permission``
and ``service`` a shell command whose exit status zero means satisfied.
For ``env`` the variable named (the check, else the name) is a shell
identifier, ``^[A-Za-z_][A-Za-z0-9_]*$``. ``prompt`` is optional: one
sentence telling the person what to enter or grant, at most 200
characters, stripped and dropped when blank; ``reef-<adapter> setup`` and
the pages show it beside the item. The credential and directive screens
run over every ``name``, ``check`` and ``prompt`` as they run over
``text``, with the same HTTP 400 and a reason that names the rule; a
malformed list is HTTP 400 naming the first bad item. The method's ``propose`` may add items of its
own to the mapping it received (an extension that reads a variable, say);
the backend merges them by name into the commit's
``training_request.requires`` after the same screens (a bad item of the
method's is dropped alone, named in the service log), the person's items
first and the list capped at 8 with the dropped items named in the service
log, so the releases row and the manifest carry what the person named and
what the change added. Nothing on the service runs a check:
``reef-<adapter> setup`` runs one after the person read it and confirmed,
the install script only reads the check offs, and the update notice prints
the setup list instead of the update while an item is unmet.

The three modes differ in what starts a step. ``auto``, the default,
trains on traffic and refuses an instruction with HTTP 400. ``manual``
runs instructions only and never trains on traffic; harness evolution
runs an instruction alone, without samples. ``hybrid`` trains on traffic and
runs instructions: a queued instruction goes first, oldest first, one per
step, and the units that automatic job data would take next, up to
``job_samples`` traces (by default ``evolution.batch_size``) and possibly none, ride beside it
as the samples of its data (scored traces, or unscored records when the
``pipeline`` section names the inference stages), so the proposer reads the
request next to them; with none queued the step trains as ``auto`` does.

.. code:: bash

   curl -sS http://127.0.0.1:8900/reef/train \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: coding" \
     -H "Content-Type: application/json" \
     -d '{"agent_record_id":"change-001","text":"Add a skill that runs tests before answering", "session":"session-1", "release_id":"release-1"}'

The response is ``{agent_record_id, scenario, request_type: "train"}``,
plus ``page_path``, the request's page with the query a browser opens it by
(see Request page).
HTTP 200 acknowledges durable acceptance, not successful training. Requests
are executed one at a time by the normal training worker; later requests
do not change a step already in flight. A step that fails with an
instruction (a proposer error, for one) is not retried: the next step
consumes the instruction with a committed row whose ``skipped`` reads
``instruction failed`` and whose ``error`` carries the failure, and the
queue moves on. Send the instruction again to run it again. That row
consumes the instruction alone: in ``hybrid`` the units that rode beside it
stay held for the next job, so no failing trace is consumed unread. The
``evolution.max_steps`` and ``evolution.max_failure_streak`` budgets count
every step, instruction steps included, and stop automatic steps only; an
instruction still runs past them. The existing evaluation and publication
rules still determine whether the result becomes served.
``GET /reef/status`` reports ``training_mode``. Processors using the shared
instruction queue also report ``buffered_requests`` (requests already
read into the processor; later records may still wait in storage) and
``pending_instructions`` (accepted instructions not yet consumed: the
buffered ones plus those still unread in storage).
Committed step metrics include ``training_request`` with the instruction
id, text, session, release id and ``requires``; the releases row of a step
that answered a request carries, for example:

.. code:: json

   {"training_request": {"id": "change-001", "text": "Text me when the run is blocked",
                         "session": "session-1", "release_id": "release-1",
                         "requires": [{"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID",
                                       "prompt": "The Twilio account SID"}]}}

When the backend dropped an item the method added (malformed, or credential
or directive shaped), ``training_request.refused_requires`` lists each as
``{"item": <the item as written>, "reason": "..."}``; the key is absent when
nothing was dropped. A method may also record notes beside its proposal;
they land under ``proposal_notes``, a JSON mapping the backend writes as
given and never reads, present only when non-empty. Reefine writes
``design`` (the proposer's plan, a few sentences), ``review``
(``{"result": "complete" | "partial", "covered": [...], "uncovered": [...]}``,
the proposer's own reading of its entries against the request; absent when
the review call failed), ``refused_requires`` (the items the method itself
dropped, in the same ``{item, reason}`` shape) and ``undeclared_env`` (the
variables a written extension reads through ``process.env`` that no
``requires`` item names; nothing adds them, the version page shows them).
A request step that produced nothing records ``failure``, why: the model
call failed (how long it took, the reply budget and the endpoint's error)
or the reply held no usable entry. Other methods may write other keys.

An optional ``client`` reports the requesting machine, so a proposer builds
for it rather than for the sandbox it tries changes in: ``platform``,
``arch`` and ``release`` (short words) and ``commands``, a map of command
names to whether each is on the machine's PATH (at most 64). ``reef-pi`` and
pi's ``/reefine`` send one, reading the PATH without running anything.
It only informs the proposer: what does not fit that shape is dropped, never
a reason to refuse the request, and ``training_request.client`` carries what
was kept.

Supply ``agent_record_id`` to retry safely: an identical request is accepted
without another step, including after capacity eviction; reusing the id with
different content returns HTTP 409. Without it, each submission gets a fresh
id. Empty text, text longer than 4000 characters, missing or non-string
session/release fields, or a request to an ``auto`` scenario returns HTTP 400.
Text that carries a credential shaped literal or an instruction override
phrasing is refused with HTTP 400 and a reason that names the rule, never
the text; nothing is stored. The scenario must exist: an unknown scenario
answers HTTP 404 and creates nothing, whatever the implicit-scenario-creation
setting says. The normal bearer authentication applies.

Scenarios
---------

A scenario isolates the records, trainer, and release chain for one workload.
The first inference request or report carrying a new ``x-reef-scenario`` creates
the scenario using the deployment's configured recipe. Requests never select a
recipe.

.. code:: bash

   curl -sS -i http://127.0.0.1:8900/v1/chat/completions \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: hello-reef" \
     -H "Content-Type: application/json" \
     -d '{"model": "m", "messages": [{"role": "user", "content": "fix it"}]}'

The bindings never change; a request naming a different recipe with the same
scenario returns HTTP 409. This means the surface, runtime, inference backend,
and optional report schema chosen when the recipe is constructed are fixed with
it.

A deployment can limit its scenarios. See `Limit the scenarios a service holds
<../user-guide/operate.rst#limit-the-scenarios-a-service-holds>`__. A request
that would create a scenario past ``reef.max_scenarios`` returns HTTP 409.
A request can also need to load a scenario when the service already holds
``reef.max_loaded_scenarios`` loaded scenarios. If none of them is idle, the
request returns HTTP 503 with a ``Retry-After`` header.

If the deployment sets ``reef.allow_implicit_scenario_creation: false``, an
unknown scenario returns HTTP 404 and you create it first:

+---------------------------------------------+---------------------------------------------+
| Route                                       | Body and response                           |
+=============================================+=============================================+
| ``POST /reef/scenarios``                    | ``{"name", "release_id"?}``                 |
|                                             | → ``{scenario, release_id,                  |
|                                             | content_id}``; 201 created, 200 already     |
|                                             | existed                                     |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios``                     | every known scenario, its active release    |
|                                             | once loaded and, for a harness recipe, the  |
|                                             | ``adapter`` its tree is rendered for, also  |
|                                             | while the scenario loads                    |
+---------------------------------------------+---------------------------------------------+
| ``GET /reef/scenarios/{scenario}/contract`` | ``{scenario, processor,                     |
|                                             | required_request_types, training_mode,      |
|                                             | status}``                                   |
+---------------------------------------------+---------------------------------------------+
| ``DELETE /reef/scenarios/{scenario}``       | → ``{scenario, archived}``; 404 unknown     |
+---------------------------------------------+---------------------------------------------+

Deleting a scenario
-------------------

``DELETE /reef/scenarios/{scenario}`` removes the scenario from the running
service and frees its name. Nothing is destroyed: the record store and commit
log move under ``<agent_record_dir>/archived/``, the recipe's own directories
(a proposal inbox, step records) move under an ``archived/`` sibling, and the
scenario's ref in the artifact repository is renamed into
``refs/reef/archived/``, so every release it published stays reachable. The
base artifact, the shared head and releases other scenarios may fork from
stay where they are. A local step in flight for the scenario ends without a
commit. Only a local artifact repository can be archived; a remote one
answers 501.

For a scenario that trains weights, the delete also cancels the scenario's
running training job at its training component, since nothing would read the
job's checkpoints afterwards. On a runtime that serves per-scenario adapters,
the scenario's adapter stays resident in the serving engines until the
adapter residency evicts it.

Scenario updates
~~~~~~~~~~~~~~~~

``POST /reef/scenarios/{scenario}/update`` updates an existing scenario.
Currently, only ``training_mode`` is supported; unknown fields are rejected.
To change the data processor's mode:

.. code:: json

   {"training_mode": "manual"}

HTTP 200 returns ``{"scenario": "agents", "training_mode": "manual"}``.
The values are ``auto``, ``manual`` and ``hybrid``: ``"auto"`` resumes recipe
training alone, ``"hybrid"`` keeps it and takes instructions too. The change
selects subsequent job data; job data already reserved or running completes
in its original mode. The modes share buffered data, so auto and hybrid can
train on traffic collected while manual was selected. Accepted instructions
wait for a mode that takes them, including instructions not yet read when
the selector changes to auto.

A Reef process runs at most one scenario that trains full weights, on a single
thread, so preparation, remote execution, and commit never interleave. It may
run any number of scenarios that produce no updates or that update text
artifacts in process. Each one grows and commits on its own background thread,
so record acceptance never waits for artifact evolution.

Unknown scenarios return ``404`` without implicit creation; invalid payloads
return ``400``. A processor that does not support the requested mode returns
``501`` without changing its state. Harness ``manual`` and ``hybrid`` also
require a proposer that explicitly accepts ``requests``.

``GET /reef/status`` reports the selected ``training_mode``. This selector
is runtime state, not persisted scenario configuration: a service restart
uses the recipe's configured mode again, while a scenario reload after a
failed step keeps the selected mode.

Inference
---------

Request
~~~~~~~

Send the same body you would send to the provider. Reef never touches your
sampling parameters. 

Before calling the model, Reef reads the ref of the scenario's active release
and builds the request against that release. The stored exchange normally uses
that frozen ref, even if an update completes mid-request. For a scenario that
only delivers files to clients, `Client release tags`_ describes when the
record instead names the client's installed harness release.

On a weight-serving deployment it adds engine
bookkeeping keys: ``lora_path`` to address the served adapter and
``return_meta_info`` so the record proves which weights answered; a body
naming a different ``lora_path`` is refused. Set ``"stream": true`` and read
the SSE response for streaming.

The receipt identifies the stored record:

+---------------+-----------------------------------------------------------+
| Response kind | Where the receipt is                                      |
+===============+===========================================================+
| non-streaming | the ``x-reef-agent-record-id`` response header            |
+---------------+-----------------------------------------------------------+
| OpenAI SSE    | ``reef.agent_record_id`` in a final empty-``choices``     |
|               | chunk, immediately before ``data: [DONE]``                |
+---------------+-----------------------------------------------------------+
| Anthropic SSE | the same field on ``message_stop``                        |
+---------------+-----------------------------------------------------------+

Streams carry it only after the record is stored. A stream that ends without
its terminal event carries no receipt. A stream from Reef's SGLang or vLLM
backend that the engine aborts also carries no receipt, because Reef cannot
restart a stream after it sent frames. In both cases a harness that reports by
receipt does not reference the record. Reef stores it as an incomplete delivery
record and preserves the captured response, including any training tensors.
The record-to-record stage excludes aborted records from sample creation and
training, including when a report references their IDs directly or retained
records are replayed after a restart.

On a scenario that serves harness files, every inference response also
carries ``x-reef-release-id``: the active release when the response is
written, so an activation during the call shows on the next one. A resident
``reef-native serve`` process compares it with the release it mounted and
learns of a new active release on its next model call, with no extra
request. A weight serving scenario sends no such header, and neither does the
evaluation route, whose caller runs a candidate and follows no active
release.

A composite recipe's evaluation calls come through these routes; a recipe of
one component calls its runtime's endpoint directly, as before composites.
The evaluation calls do not present a service token. When auth is configured,
the service issues a random token at startup that opens the evaluation routes
alone, and hands it to the recipe: the rollouts and proposer run candidate
code, which can then sample the active release but reach no other route. A
restart issues a new one.

Response
~~~~~~~~

Reef validates a response before recording it. ``prepare_request`` transforms
the outgoing payload, and Reef forwards *and records* the transformed payload.
``verify_response`` checks the provider's answer against the frozen release. On
failure Reef records nothing and returns the error.

Serving version of a training response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a training scenario, Reef records which engine weights answered. The engine
reports the ``runtime_load_id`` of the response, or of each generated token
span when a weight load lands mid-generation. The spans must be contiguous and
must not return to an earlier version. A training response that reports no
``runtime_load_id``, or reports inconsistent spans, is a backend contract error
and returns HTTP 409. A streamed training response must be buffered and
recorded as a whole, so that its versions can be read.

Report
------

Reef stores four fields and drops other top-level keys. Put harness-specific
data inside ``metadata`` or ``feedback``.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Required
     - Notes
   * - ``score``
     - number
     - no
     - Must be finite; booleans are rejected.
   * - ``feedback``
     - string or object
     - no
     - Opaque feedback interpreted by the recipe.
   * - ``references``
     - list of strings
     - no
     - Unique existing inference receipts in this scenario. Reported-feedback
       processors require at least one reference; several form one trajectory.
   * - ``metadata``
     - object
     - no
     - Recipe metadata; ``training.eligible`` is not supported. ``metadata.task``
       (``name``, with ``path`` and ``digest`` when known) or
       ``metadata.harbor.task_name`` names the Harbor task the rollout ran;
       the assembled training sample carries it as ``task``, so a recipe can
       group samples by task.

It answers ``{agent_record_id, scenario, request_type}``.

.. code:: python

   client.report("hello-reef", {
       "agent_record_id": "myharness:run42:trial7",
       "score": 1.0,
       "references": ["abc123"],
       "metadata": {"harbor": {"trial_id": "run42:7"}},
   })

The optional top-level ``agent_record_id`` makes posting retry-safe: Reef uses
it as the report's own record id, so an identical resend returns the stored
record instead of reprocessing it, while the same id with different content is
HTTP 409. It is not a receipt. Receipts go in ``references``.

A recipe may declare a report schema. `tttd
<../user-guide/recipes/tttd.rst#the-report-contract>`__ declares one. In that case, Reef
validates the declared ``score`` and ``metadata`` fields at ingress and answers
HTTP 400 on a violation; ``feedback`` and undeclared ``metadata`` keys pass
through, except the removed ``metadata.training.eligible`` field, which is
rejected with HTTP 400 regardless of its value. Reports cannot opt out of training.

Every supplied reference must identify an already stored inference record in the
same scenario. Missing, foreign, non-inference, or duplicate references produce
HTTP 400 without persisting the report. Reef does not wait for future records;
the producer may resubmit after the referenced inference has been stored.
Identical retries of previously accepted reports retain their original behavior
after reference cleanup; they do not restart training.

Record
------

Reef stores every exchange of inference and every report as ``AgentRecord``.
These records are referenced via a record id. In ``/reef/train`` and
``/reef/report`` API calls, users can provide an optional ``agent_record_id``
field for retry-safety. Appending the same id with different content returns a
conflict rather than overwriting. Reef keeps track of consumed records so
retried reports and late reports whose references already trained are not
counted twice.

Training records remain readable after consumption. Storage evicts the oldest
bodies only when the shared capacity budget is exceeded (20 GiB by default),
with warnings and durable loss totals. Audit endpoints expose retained requests,
responses and references. Per-step ``consumed_ids`` records processed
inputs, including intentional skips, rather than proving a model update.
See `Configuration <configuration.rst>`__ and `Python API <python-api.rst>`__.


Receiving an update
-------------------

For weight-training scenarios there is nothing to do: keep calling the same
inference endpoint and it serves the active release. The artifact
lives in the inference runtime, so requests reach the new release directly.

A scenario whose artifact is the harness tree works the other way: the client
pulls. The harness should fetch the tree of the active release with
``GET /reef/harness``, or the install script built on it, and run the agent on
that release. So the harness it uses is the one whose receipts it will later
report against.

Harness artifacts
~~~~~~~~~~~~~~~~~

+--------------------------------+---------------------------------------------------------------+
| Route                          | Response                                                      |
+================================+===============================================================+
| ``GET /reef/harness``          | ``{release_id, content_id, parent_release_id, files,          |
|                                | evaluation, requires}``, plus ``components`` (each component  |
|                                | name to its content id) when the release binds several, and   |
|                                | an ``x-reef-release-id`` response header                      |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/releases`` | ``{scenario, releases}``, oldest first: the creation and      |
|                                | every training step, also a step that published nothing       |
|                                | (``checkpoint: false``). Each row has ``artifact_id``,        |
|                                | ``component``, ``active`` (true on exactly one row, the       |
|                                | active release) and ``activated`` (true when the row's        |
|                                | artifact has been active at some time); each training row     |
|                                | carries the evaluation metrics of its step and ``page_path``, |
|                                | its step's page with the query a browser opens it by          |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/install``  | a self-contained POSIX shell script that installs the vendor  |
|                                | binary, writes the tree, and writes the adapter's model       |
|                                | binding at the address the request reached (a gateway in      |
|                                | front names it in ``x-forwarded-host`` and                    |
|                                | ``x-forwarded-proto``), the token filled from ``REEF_TOKEN``  |
|                                | when the script runs; it records the sha256 of each tree file |
|                                | it wrote (for pi's ``settings.json``, the value of each key   |
|                                | but pi's own preferences) and that address in                 |
|                                | ``~/.reef/installs``, and writes the ``reef-<adapter>``       |
|                                | wrapper beside that record, which checks it before a session  |
|                                | starts                                                        |
+--------------------------------+---------------------------------------------------------------+
| ``GET /reef/harness/adapters`` | ``{adapters}`` — every harness adapter this process resolves, |
|                                | each with ``name``, ``binary``, ``trajectory_format``,        |
|                                | ``model_bindings``, and the pinned ``install`` spec           |
+--------------------------------+---------------------------------------------------------------+

The first three are read-only and take ``x-reef-scenario``. Install also requires
``?adapter=``, whose value may be ``pi``, ``opencode``, ``claude``, ``codex``,
``dsh``, ``hermes``, or an external descriptor. Only an adapter whose descriptor
declares an install section can be named here: ``native`` and ``terminus`` ship
with reef and pin no vendor binary, so they answer HTTP 400 rather than a
script, naming ``GET /reef/harness`` for the tree and ``POST /reef/train`` for
a request. If install omits ``x-reef-scenario``, Reef creates a scenario with a
generated ``harness-`` name and embeds that assignment in the wrapper script;
when exactly one configured recipe serves harness files, it selects that recipe
automatically. With ``reef.allow_implicit_scenario_creation: false``, install
creates no scenario and returns HTTP 404. Create the scenario first with
``POST /reef/scenarios``, then name it in ``x-reef-scenario``.

Creating that scenario takes a few seconds, and ``curl ... | bash`` shows
nothing until the first bytes arrive. So when the script is not ready within
half a second, the response starts at once with a short preamble that shows
``reef: preparing the harness install`` (a spinner on a terminal). The script
follows inside one ``{ ... }`` group that first stops the spinner, so a
connection that drops mid-script runs none of it. In that case the HTTP
status is already 200: a failure while preparing arrives as a script that
prints ``reef: the harness install failed (HTTP <status>): <message>`` and
exits 1. A script ready within the half second is sent as before, with the
failure's own HTTP status.

``files`` is the rendered tree, path to text. An adapter whose descriptor
declares ``files.tree`` (``native`` does: ``native/tree.json``) adds one more
file: the release's entries list, the same ``{id, name, config}`` objects the
commit history persists, as one JSON array. A resident ``reef-native serve``
process mounts that list entry by entry; an older ``reef-native`` ignores the
file and reads the rendered files as before. ``pi`` declares none: a pi
release is its rendered files, and the entries stay in the commit history, where
the proposals route and the evolve step read them.

``requires`` is what the release needs from the person who installs it:
every ``training_request.requires`` item (`Manual training
<#manual-training>`__ gives the shape) over the release's chain, merged by
name with the newest definition winning, since a request's items are per
release and a later release whose request named nothing still carries the
extension an earlier one added. The chain follows ``parent_release_id``
through the catalog; a releases row carries its own step's list alone. The install
script embeds the list (the union of a chain is not bounded by one
request's cap of 8) and refuses, before it installs the binary or makes a
directory, while an item is not checked off in the ``.reef-harness-release``
release metadata file on disk: it prints the setup list and the newest release in the
chain that requires nothing, the one that installs on a machine with
nothing set up (``?release_id=<id>``), and exits 1. ``reef-<adapter> setup``
records the check offs; ``--release <id>`` names an inactive release so its
items are checked off before you activate it. The release metadata file the script writes
carries ``requires`` (the list) and ``setup`` (the check offs, ``{name,
checked_at, check}``, carried over from the previous release metadata file by name; an
item whose check is not the recorded one counts as unmet); a release metadata file the
stdlib client pull wrote carries neither, which reads as nothing required.

Use ``?release_id=`` on the manifest or install route to request a specific
catalog release. An unknown or unrestorable release returns HTTP 404.

Catalog and manifest reads do not wait for an evolve step's proposer or
evaluation rollouts: they continue serving the existing releases while a
candidate is being prepared. Reads still serialize with publication and
activation so a manifest's artifact and evaluation metrics come from the same
release: reads do not interleave with a change of the active release and its
commit-history update.

Proposals
~~~~~~~~~

An agent running on the tree of the active release can propose a change to it. The
proposal enters the same evaluation as the method's own: nothing it says is served
until paired rollouts settle it.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: code-repair" -H "Content-Type: application/json" \
     -d '{"mutations": [{"op": "create", "id": "check", "options": {"name": "rules", "config": {"text": "Run the tests before you answer."}}}],
          "reason": "three of five sessions answered before the tests ran",
          "session": "3f1c2a9d0b7e", "release_id": "rel-12"}' \
     "$REEF_URL/reef/harness/proposals"

+----------------+----------------------------------------------------------------------+
| Field          | Meaning                                                              |
+================+======================================================================+
| ``mutations``  | a non-empty list of ``{op, id, options}``: ``create`` and ``update`` |
|                | carry ``options`` (``{name, config, disabled?}``, the entry without  |
|                | its id), ``remove`` carries none                                     |
+----------------+----------------------------------------------------------------------+
| ``reason``     | the proposer's own account, stored with the proposal                 |
+----------------+----------------------------------------------------------------------+
| ``session``    | the session that proposed, named in the commit that settles it       |
+----------------+----------------------------------------------------------------------+
| ``release_id`` | the release the proposer was running                                 |
+----------------+----------------------------------------------------------------------+

The service admits the mutations against the active release's entries with the
rules every mutation meets (a create on an existing id, an update on a missing
id or one that changes the entry's kind, a remove on a missing id, a config the
kind's admission refuses, a kind the adapter does not render, a tree that does
not render, any op on one of reef's own entries: ``reef-version-check``,
``reef-requests`` and ``reef-pi-extension-api`` are reserved ids) and answers
``{proposal_id, admitted, reason, release_id}``:
``reason`` is the rule that refused, else ``null``; ``release_id`` is the
active release that the proposal was admitted against. An admitted proposal waits in the
scenario's inbox (``evolution.proposals_dir``) until the next evolve step takes
it, oldest first, before the method's own ``propose`` is asked; the step admits
it again against the entries of its training start, which can differ from the
active release, and the evaluation
settles it like any mutation. The commit that settles it carries ``proposal:
{id, session, release_id, reason}`` in its metrics, and the releases row
carries that commit. When ``evolution.max_pending_proposals`` already
wait, the answer is ``admitted: false`` with reason ``inbox full``; on a
scenario in ``data.training_mode: manual`` it is ``admitted: false`` with
reason ``manual mode takes instructions only``, since no automatic step runs
there to take the inbox. A malformed
body is HTTP 400; a scenario whose recipe is not a harness evolution recipe is
HTTP 404 naming that. `Operate a deployment
<../user-guide/operate.rst#read-the-proposal-inbox>`__ describes the inbox
directories.

Harness requests
~~~~~~~~~~~~~~~~

``reef-<adapter> harness "<request>"`` and pi's ``/reefine <request>``
submit the user's instruction through ``POST /reef/train``, described under
`Manual training <#manual-training>`__. Set ``data.training_mode: hybrid``
(the deployment keeps learning from failures) or ``manual``, or switch an
existing scenario with ``POST /reef/scenarios/{scenario}/update``.
The commands send ``text``, ``session`` and the installed ``release_id``;
HTTP 200 acknowledges durable acceptance and returns ``agent_record_id``.
They require no inference receipts and leave captured receipts available for
feedback. A scenario in ``auto`` refuses the request; the commands surface
that error and leave mode switching to the caller.

The existing trainer delivers the request to a proposer that explicitly
accepts ``requests``, then evaluates and publishes under the same policy as
automatic evolution. Its commit metrics carry ``training_request:
{id, text, session, release_id, requires}`` (with ``refused_requires`` when
the backend dropped an item the method added) and, for a method that records
notes beside its proposal, ``proposal_notes``, both visible through the
release catalog.

Artifacts
~~~~~~~~~

An artifact is the content of one component of a release, such as a
checkpoint's weights or a harness tree. Training makes artifacts: each
checkpoint that a weight trainer saves, and each tree that a harness trainer
keeps. The release that created a scenario holds the creation artifact of each
component. ``artifact_id`` names an artifact:

- In a scenario of one component, it is the ``release_id`` of the release
  that made the artifact. Every shipped recipe serves one component, so there
  ``artifact_id`` and ``release_id`` are equal.
- In a scenario of several components, it is ``<release_id>:<component>``. A
  release that carries a component forward keeps that component's
  ``artifact_id``.

``GET /reef/scenarios/{scenario}/artifacts`` answers ``{scenario,
artifacts}``, **newest first**. Each row is one artifact of one component:

+-----------------------+------------------------------------------------------+
| Field                 | Meaning                                              |
+=======================+======================================================+
| ``artifact_id``       | the artifact                                         |
+-----------------------+------------------------------------------------------+
| ``component``         | the component whose content it is                    |
+-----------------------+------------------------------------------------------+
| ``release_id``        | the release that made it                             |
+-----------------------+------------------------------------------------------+
| ``parent_release_id`` | the parent of that release                           |
+-----------------------+------------------------------------------------------+
| ``content_id``        | its content                                          |
+-----------------------+------------------------------------------------------+
| ``operation``         | ``creation`` for the creation artifact,              |
|                       | ``training``, or ``recovery`` for the artifact that  |
|                       | a store without a durable commit log served at a     |
|                       | restart                                              |
+-----------------------+------------------------------------------------------+
| ``recorded_at``       | the commit time, ``null`` for the creation artifact  |
+-----------------------+------------------------------------------------------+
| ``active``            | true for the active artifact of each component       |
+-----------------------+------------------------------------------------------+
| ``activated``         | true when the artifact has been active at some time  |
+-----------------------+------------------------------------------------------+
| ``metrics``,          | training rows only: the step's metrics and the       |
| ``record_progress``   | record high-water mark that the step consumed        |
+-----------------------+------------------------------------------------------+

The list holds artifacts only. A training step that made no artifact, such
as a rejected or skipped step, a job submission or a job's end, has no row;
``GET /reef/scenarios/{scenario}/commits`` holds its record (see `Record and
commit history`_). Activations and train-from operations have no rows either.

Activate an artifact
~~~~~~~~~~~~~~~~~~~~

Pulling an older release changes only your local copy. Reef serves the
scenario's *active release*: inference answers from it, and
``GET /reef/harness`` and the install script serve its tree. The active
release binds the active artifact of each component. To serve another
artifact, send ``POST /reef/scenarios/{scenario}/activate``:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/activate"

The body is ``{"artifact_id": "<id>", "component": "<name>"}``. Leave out
``component`` when the scenario has one component. The answer is
``{scenario, component, artifact_id, release_id, content_id}``:
``artifact_id`` names the activated artifact, ``component`` names its
component, ``release_id`` is the active release afterwards, and
``content_id`` is the artifact's content.

- In a scenario of one component, activation writes no artifact or release.
  It appends an ``activate`` record, and the release that made the artifact
  serves again.
- In a scenario of several components, activation publishes a release: the
  active release with ``component`` bound to the artifact, and the other
  components as the active release binds them. It makes no new content and
  does not move the `chain tip
  <../advanced_topics/state-model.rst#the-release-chain>`__.

Activating the active artifact changes nothing. Activation never changes a
trainer's training start. A full-weight serving runtime cannot activate the
creation artifact after it served a checkpoint, and answers an error.

Train from an artifact
~~~~~~~~~~~~~~~~~~~~~~

Each trainer has a training start: the artifact that its next training job
(a weight trainer) or evolve step (a harness trainer) starts from. After each
job or step, the trainer moves its training start to its own newest output.
``POST /reef/scenarios/{scenario}/train-from`` sets the start to an existing
artifact and cancels the running job:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"artifact_id": "<id>"}' \
     "$REEF_URL/reef/scenarios/code-repair/train-from"

The body is ``{"artifact_id": "<id>", "component": "<name>"}``.
``component`` names the trainer's component. Leave it out when the scenario
has one component; a scenario with several components requires it. The
answer is ``{scenario, component, artifact_id}``. Train-from never changes
the active release.

Whether a trainer can start from an artifact depends on the trainer:

- The Tinker component imports the artifact's checkpoint manifest, which
  names the saved training state with the optimizer state.
- The Slime component refuses every artifact, because its workers continue
  from their own latest checkpoint.
- A harness trainer starts its next evolve step from the entries of the
  artifact's tree.

A weight artifact without optimizer state, such as the creation artifact,
starts training with a new optimizer state.

Both routes answer these statuses:

+--------+--------------------------------------------------------------------+
| Status | Cause                                                              |
+========+====================================================================+
| 200    | the artifact is active (``activate``) or is the trainer's start    |
|        | (``train-from``)                                                   |
+--------+--------------------------------------------------------------------+
| 400    | a malformed body; the artifact has no durable bytes; an unknown    |
|        | ``component``, or ``component`` is missing in a scenario with      |
|        | several components; ``activate``: the runtime cannot serve the     |
|        | artifact; ``train-from``: the trainer cannot start from the        |
|        | artifact                                                           |
+--------+--------------------------------------------------------------------+
| 404    | unknown scenario or artifact                                       |
+--------+--------------------------------------------------------------------+
| 409    | a training result waits for its commit; send the request again     |
|        | after the commit                                                   |
+--------+--------------------------------------------------------------------+

`Activate an artifact <../user-guide/operate.rst#activate-an-artifact>`__
explains when Reef activates a new artifact automatically. `Go back to an
older artifact <../user-guide/operate.rst#go-back-to-an-older-artifact>`__
explains how to go back to an older one for good.

Inactive artifacts
~~~~~~~~~~~~~~~~~~

With ``evolution.publish: review``, the step makes the winning tree as a
release but does not activate it. When a successful evaluation touches a node
kind listed in ``evolution.review_kinds``, the step does the same. Its row
carries ``active: false`` and ``activated: false``. The manifest and install
routes keep serving the active release, and ``?release_id=`` can pull the
inactive tree for a trial install. To make it serve, send the row's
``artifact_id`` to ``POST /reef/scenarios/{scenario}/activate``.

In a scenario with several components, ``GET /reef/harness/releases``, the
release pages and ``GET /reef/harness`` speak of the releases that changed the
pulled tree. Another component's step is not listed and is no new release of
that tree. In the rows listed, ``parent_release_id`` names the previous listed release, the
one the tree descends from. When the release's own parent differs,
``composed_parent_release_id`` keeps it. A rejected or skipped step made no
release, so its row names the listed release that the active release carried,
and ``composed_release_id`` keeps the active release's own id. The tree
that Reef serves is that of the active release, or of the listed release it
carried.
Every release stays addressable by id through ``?release_id=`` on the
manifest and install routes.

Version page
~~~~~~~~~~~~

``GET /reef/harness/releases/{step}/page`` answers one self contained HTML
page (``text/html``, no asset, its data inline) for one catalog row. ``step``
is the row's position in ``GET /reef/harness/releases`` oldest first, the
creation row being 0, which is the commit step: a rejected step publishes
nothing and its row carries the active release's id, so the step is what
names it. The page shares its chrome with the request page below: the same branded
header, palette, card layout, automatic light and dark themes and responsive
behavior, and one status vocabulary, so a step reads ``Published``,
``Waiting for activation``, ``Not selected`` or ``No changes`` in place of the
record's own field names. The page has up to seven sections, each its own
card, in this order: Why (the request the
step read, else the claimed proposal's reason, else a failure in the job data),
Design (the proposer's plan, ``proposal_notes.design``; only when the method
recorded one), What changed (the step's mutations, each with its operation,
node id and kind; an extension's file as text for a create, and for an update
a line diff against the release the
candidate ran on when that release is restorable, else the new text), Review
(``proposal_notes.review``: the proposer's result on its entries against the
request, ``complete`` or ``partial``, then the points it covered and the ones
it left uncovered, then ``proposal_notes.undeclared_env``,
the variables a written extension reads that no ``requires`` item names;
only when the row carries a review or that list), Result (the result and what
it means, then the evaluation's numbers: ``wins``, ``losses``, ``ties``,
``passed``, ``failed``,
``floor_score``, ``evaluation_sides``, ``current_score``, ``candidate_score`` and
``rollout_failures``, each when the row carries it, so a ``floor`` evaluation,
which runs no current side, shows ``passed``, ``failed`` and ``floor_score``
and no ``current_score``, then the skip reason, the selection reason, the step
record directory when ``evolution.step_record_dir`` is set, and
``proposal_notes.failure`` as the proposer failure
when the step recorded one), Setup (the request's ``requires`` with
name, kind, check and prompt, then the items the release carries from
earlier steps in its chain, the same union the install script and
``reef-<adapter> setup`` read; a rejected or skipped row lists only its own
items, since its release id is the active release's; then, under "Refused by the
step", the items the step dropped from ``training_request.refused_requires``
and ``proposal_notes.refused_requires``, each as written with its reason;
nothing when all are empty) and Chain (the parent release, this release,
and its children: the steps evaluated on it, won, lost or inactive, each
linking that step's own page; a rejected or
skipped step published nothing, so its
Chain names the release it ran on and no children).
Above the sections a step walk links the neighbouring steps, and the logo and
the "Harness" crumb lead to the active release's page; a dead end, and the
active release's own page, leave the link as plain text. Every link either page draws is one
of the two ``/page`` routes, the only ones a browser can open, since the
catalog answers JSON and refuses a query token.
The line under the title carries the release id and reads the commit time as
a date in UTC. On the active release, the row marked ``active``, it also
carries the chip ``Active release``. A step that made an artifact that was
never activated (``activated: false``) reads ``Waiting for activation``. A step outside the
catalog is HTTP 404 naming the range; a step that is not a number, or longer
than nine digits, is HTTP 404 too. The row itself rides in a
``<script type="application/json">`` block at the end of the page, every
``<`` escaped.

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" -H "x-reef-scenario: code-repair" \
     "$REEF_URL/reef/harness/releases/3/page" > harness-step-3.html

On pi, ``/versions`` in a ``reef-pi`` session lists the chain, and
``/versions v3`` offers to open this page in the browser, printing the URL
when the offer is declined. ``/versions v3 install`` installs that release,
and activates it first when it is inactive.

Request page
~~~~~~~~~~~~

``GET /reef/harness/requests/{record_id}/page`` answers one self contained
HTML page (``text/html``, no asset, ``Cache-Control: no-store``) for a filed
harness request, ``record_id`` being the ``agent_record_id`` that
``POST /reef/train`` answered; ``reef-pi evolve`` and pi's ``/reefine``
print the link. Until the step settles the page reloads itself every five
seconds. A four-stage progress strip and a status badge summarize the
request. The responsive layout places Request beside Progress on desktop
and stacks them on narrow screens, with automatic light and dark themes; it
shares that chrome and its status wording with the version page above.
Request highlights the instruction; session, release, request ID, submission
time and machine requirements are available in expandable details.
Progress shows
the request's state and what it means: ``queued`` while no step has taken
the request, ``proposing`` while the served model writes the change,
``evaluating`` while the candidate's rollouts run (with their count and the
step record directory when the backend reports them, and the time into the
step), ``running`` while the trainer holds the request and the backend
reports no phase, and ``settling`` while the row that consumed the record
lands. While a step holds the request, Activity lists what the proposer has
done so far, newest first, each line at its time into the step and the
newest with how long ago it happened: every model call as it starts and
as it answers (its seconds and tokens, or its error), and for the agent
proposer each tool the agent calls, each admission check, each trial with
its exit and multimodal calls, and each multimodal call with its status;
failed lines are marked. The page lists the latest 80; the step record
keeps every call. The catalog row whose ``metrics.training_request.id`` is the record
id settles the page: the reload stops and Progress gives way to Result
(the result as the version page words it, what it means and the next
action, a failed instruction's ``error``, ``proposal_notes.failure`` as
``proposer failure``, the release id and a link to the version page), What
changed (each mutation's op, id and kind; labeled Proposed changes for
inactive, rejected, skipped or failed steps) and, when the step recorded a review,
Review (its result and the points it left uncovered). Published and inactive
results show the session command to install, or to activate and install, when
the person is ready, alongside a link to the version page. An unknown
id, or one that is not a training instruction, is HTTP 404 naming it.

``GET /reef/harness/requests/{record_id}/progress`` answers the same reading
as JSON (``Cache-Control: no-store``), for a client that polls rather than a
browser that renders: ``request_id``, ``settled``, ``step`` (the step the
row landed as once it settles, else null), ``state`` (the page's own
``queued``, ``proposing``, ``evaluating``, ``running`` or ``settling``, and
the settled row's result once a row answers the request, including ``failed``
when a skipped step records a proposer failure or execution error), ``meaning`` (the
words the page prints beside the state, null once settled), and, while a
step holds this request, ``started_at``, ``rollouts_total``,
``step_record`` and ``activity`` (the Activity lines oldest first, each
``{at, kind, text}`` with ``failed: true`` on a failed one; ``kind`` is
``model``, ``agent``, ``check``, ``trial``, ``provider`` or ``proposer``;
empty otherwise) from the backend's progress. The phase is what the pi
extension's spinner names while the step runs, and opening the spinner lists
the latest four activity lines. Unlike the two pages this is
an ordinary route: it reads the headers alone, and a ``?key=`` is HTTP
401. An unknown id, or one that is not a training instruction, is HTTP 404
naming it.

Evaluation metadata uses ``evaluation``, ``evaluation_sides``,
``evaluation_task_count`` and ``evaluation_context``. Running rollouts use the
phase ``evaluating``. Reviews and settled proposals store their outcome under
``result``. A harness manifest carries the evaluation metrics under
``evaluation`` only.

Both pages are links a person opens in a browser, which sends no header, so
they also take the scenario and a credential as query parameters,
``?scenario=<name>&key=<page key>``, in place of ``x-reef-scenario`` and
``Authorization: Bearer``; a header wins when present, and each page's links
to the other carry the parameters it was opened with. Clients never build
these links: ``POST /reef/train`` answers ``page_path`` for the request's
page and ``GET /reef/harness/releases`` one per row for its step's page,
and a client prints its service URL followed by that path. The path's query
holds the page key when the request presented a service token (an
evaluation token gets none, and with authentication off the query names
the scenario alone). The key opens these two pages of that one scenario and
no other route, a request whose ``x-reef-scenario`` header names another
scenario is HTTP 401, and the token cannot be read back from it (the
service derives it as an HMAC SHA-256 of the scenario keyed by the token's
digest). The links ``reef-<adapter>`` and pi's extension print carry it,
since a session's model reads them and a session's traffic is captured.
The token itself is never read from a query: ``?token=`` is HTTP 401 on
every route, the pages included. Every other route reads the headers alone;
a ``?key=`` elsewhere is HTTP 401.

.. code:: text

   $REEF_URL/reef/harness/requests/<record_id>/page?scenario=<scenario>&key=<page key>

Retained step files
~~~~~~~~~~~~~~~~~~~

``GET /reef/harness/releases/{step}/records`` returns the raw file inventory
for the same catalog step, authenticated and scenario-scoped like the version
page. The response is ``{"status": "retained", "files": [{"path": "proposer.json",
"bytes": 123}]}``. Add ``?path=proposer.json`` (or an inventory path under
``rollouts/``) to read ``{"status": "retained", "path": "...", "text": "..."}``.
The service does not interpret proposer replies or agent events. A console can
render those persisted formats without changing their learning semantics.

The backend reads only the step directory referenced by the selected catalog
row under its configured scenario record root. Absolute paths, traversal and
symlinks are rejected. Only JSON and JSONL files are exposed; inventories are
limited to 1000 filesystem entries and individual files to 4 MiB. Exceeding
these limits returns HTTP 400 instead of silently truncating records. A missing
file or catalog step is HTTP 404. A step with no archive metadata returns
``status: not_recorded``, disabled recording returns ``status: disabled``, and
an absent archive directory returns ``status: missing``. These states carry an
empty files list. All successful reads use ``Cache-Control: no-store``.

Proposer records contain recorded messages and replies. Native sessions retain
request headers, assistant messages and tool events; reconstructed inputs are
not exact provider request bodies. They do not retain provider response IDs,
and a compaction event may prevent complete input reconstruction. Reads neither
copy records into another store nor change retention. Step files remain separate
from online record-body capacity eviction.

Status
------

Read ``GET /reef/status`` when inference is still serving an older release while
a training job runs or an update is being published.

.. code:: json

   {
     "error": null,
     "loaded_scenarios": 1,
     "preload_errors": {},
     "scenarios": {
       "hello-reef": {
         "scenario_step": 3,
         "artifact_head_sync": {"state": "synchronized", "release_id": "...", "error": null},
         "last_committed_step": {
           "step": 3,
           "recorded_at": 1756400000.0,
           "metrics": {"selected": false, "selection": {"reason": "candidate lost"}}
         },
         "current_runtime_load_id": "7f2a:12",
         "job_data_ready": false,
         "training_mode": "auto",
         "processor": {"...": "..."},
         "inference_admission": {"...": "..."},
         "training_job": {"job_id": "9c41...", "backend": "slime", "checkpoints_handled": 0},
         "finished_jobs": 2
       }
     },
     "serving": {"...": "..."}
   }

``error`` and ``preload_errors`` report asynchronous training and preload
failures. ``loaded_scenarios`` counts the scenarios loaded in this process.
``job_data_ready`` says whether the processor has a job's data waiting.
``processor`` includes ``ready_samples``, the samples of the held record sets
and uploaded trajectories; ``job_samples``, the samples that automatic job
data needs (``pipeline.sample2job.config.job_samples``); and ``buffer_size``,
the most samples the held record sets keep.

A scenario that trains weights also reports its training job.
``training_job`` is the job that the scenario's training component runs. It
holds the ``job_id``, the ``backend`` that runs the job, and
``checkpoints_handled``. ``checkpoints_handled`` is the number of the job's
saved checkpoints that Reef has evaluated and committed. ``training_job`` is
``null`` while no job runs. ``finished_jobs`` counts the jobs of the scenario
that ended, finished or failed. A runtime that serves each scenario its own
adapter adds ``adapter_runtime_load_id``. It is the runtime load ID of the
scenario's served adapter, or ``null`` while the scenario samples the base
model.

A scenario whose recipe runs one trainer per recipe component adds a
``components`` object. For each recipe component, it holds the component's
own ``job_data_ready``, ``training_mode``, ``processor`` status and
``last_committed_step``. The scenario-wide ``job_data_ready``, ``training_mode``
and ``processor`` describe the first trainer that trains, else the first
trainer. The scenario-wide ``last_committed_step`` is the latest commit of
any recipe component.

Every row of ``GET /reef/scenarios/{scenario}/artifacts`` names the
``component`` whose artifact it is.

``last_committed_step`` reports the latest durable training step number,
commit time, and its recipe-owned metrics; it is ``null`` before the first
training commit. This distinguishes a step still in flight from a completed
step that skipped or rejected its candidate. An activation or a train-from
advances ``scenario_step`` without replacing the latest training outcome.

``serving`` is runtime-wide but recipe-shaped, keyed by recipe name. A runtime
that serves per-scenario adapters reports the engines' shared adapter
residency there, under ``adapters``.

Status codes
------------

+--------+-------------------------------------------------------------+
| Status | Cause                                                       |
+========+=============================================================+
| 400    | malformed body, a missing or empty ``x-reef-scenario`` on a |
|        | scenario-scoped route, a report violating the recipe's      |
|        | declared schema, or an activation or a train-from naming an |
|        | artifact that Reef cannot serve or train from               |
+--------+-------------------------------------------------------------+
| 401    | missing or wrong bearer token, or ``x-api-key`` when no     |
|        | Authorization header is sent; the two harness pages also    |
|        | read ``?key=`` (see Request page)                           |
+--------+-------------------------------------------------------------+
| 403    | relayed from the upstream provider. Reef issues none of its |
|        | own: an unaccepted token is 401, and per-scenario           |
|        | authorization belongs to the gateway in front of Reef.      |
+--------+-------------------------------------------------------------+
| 404    | unknown scenario (with implicit creation off, or on         |
|        | ``POST /reef/train``), unknown release or artifact, unknown |
|        | adapter, no configured harness recipe, or a scenario that   |
|        | serves no files                                             |
+--------+-------------------------------------------------------------+
| 409    | a base artifact conflicting with the scenario registration, |
|        | record id resent with different content, an activation or a |
|        | train-from while a training result waits for its commit, or |
|        | an engine that reports no serving runtime load ID           |
+--------+-------------------------------------------------------------+
| 502    | the upstream provider failed on its own account             |
+--------+-------------------------------------------------------------+
| 503    | the artifact store is unreachable, the deployment serves no |
|        | inference (``inference.backend: none``), no loaded scenario |
|        | is idle at ``reef.max_loaded_scenarios`` (with              |
|        | ``Retry-After``), or inference attempts kept being aborted  |
|        | until the retry deadline                                    |
+--------+-------------------------------------------------------------+

Reef relays upstream 4xx failures with the provider's original message; the
common client statuses (400, 401, 403, 404, 408, 409, 422, 429) keep their
status code, and any other upstream 4xx comes back as 400.

Browser consoles
----------------

A browser console sends requests to this HTTP API directly, as every other
client does. The browser must be able to open the service URL. To let a console
connect, add ``reef.console-origins`` to the serve YAML. List each trusted
console origin explicitly, with no path, trailing slash, credentials, or
wildcard:

.. code-block:: yaml

   reef:
     console-origins:
       - "https://api.reefinfra.ai"
       - "http://localhost:3000"

Restart the service after changing the configuration. Without this option, Reef
does not add CORS headers. With it, unlisted browser origins are rejected before
a route runs. CORS preflight requests from listed origins do not need a service
token; actual requests retain the configured Bearer authentication. Browser
requests may use GET, POST or DELETE with Authorization, Content-Type and
x-reef-scenario headers. Cookies are not enabled through CORS.

CORS headers also cover errors and streamed responses, exposing
``x-reef-agent-record-id``, ``x-reef-release-id`` and ``x-reef-artifact-version``
to the browser. Clients without an Origin header keep their existing behavior.
A browser can also require local network permission or HTTPS. An origin that
Reef allows does not override browser policy.

A connected console acts with the existing permissions of the service token.
Its requests operate on the scenarios of this service. The CORS connection
does not create a cloud deployment or upload history.

Record and commit history
-------------------------

``GET /reef/scenarios/{scenario}/records`` reads retained record metadata,
including consumed records. ``after_sequence`` defaults to 0 and ``limit``
defaults to 50 (1 to 100). ``request_type`` (``inference``, ``report`` or
``train``) lists one type only; another value is HTTP 400. Records are oldest
first; ``next_after_sequence`` is null at the end. Each row contains
``sequence``, ``agent_record_id``, ``request_type``, ``created_at``,
``references``, the recorded ``artifact_ref``, and the payload's ``score``
field. No record payload or learning classification is included. An inference
record's ``artifact_ref`` names the release that served the call, except when
the conditions in `Client release tags`_ select the client's installed harness
release.

``GET /reef/scenarios/{scenario}/records/{record_id}`` returns that metadata
and the stored ``payload``. A missing body returns 404: it may have expired or
never been retained. Reading never reactivates records or changes retention.

``GET /reef/scenarios/{scenario}/commits`` reads committed metadata, oldest
first, with ``after_step`` (default 0), ``limit`` (default 50, range 1–100),
and ``next_after_step`` (null at the end). Repeat the optional ``record_id``
query parameter to filter commits whose recorded ``consumed_ids`` contain
any requested ID (at most 100 IDs of 1–256 characters). This is exact set
membership, not an eligibility decision. Each row exposes ``step``,
``operation``, ``operation_verified``, ``recorded_at``, ``artifact_ref``,
``pending``, ``consumed_ids``, and recorded ``metrics``. Algorithm state and
private checkpoint recovery data are excluded. Commit metadata remains
available when trace bodies expire.

``operation`` is one of three values:

- ``training``: a training step. ``artifact_ref`` is the release the step
  made, or the active release when the step made none. ``pending`` is true
  for an artifact that the step made but did not activate. The field keeps
  this name until the activation rule moves out of training (RFC #965,
  step 3).
- ``activate``: an activation. ``artifact_ref`` is the release that serves
  afterwards.
- ``train_from``: a new training start. ``artifact_ref`` names the artifact
  that the trainer starts from.

The active release is the release of the newest activation in the log. A
training step that made an artifact and activated it counts as an
activation.

The existing ``GET /reef/scenarios/{scenario}/contract`` also reports
``training_mode`` and processor ``status`` alongside ``processor`` and
``required_request_types``. These are live runtime facts, not policy versions
or historical rule decisions.

All these reads require service authentication and return
``Cache-Control: no-store``. Payloads and recorded metrics can contain request
content. A platform must bind the upstream scenario to the authenticated
workspace. Each response is a live read; concurrent training and retention
can change subsequent pages. Reading commit pages may scan the scenario's
cached commit history; the response is bounded, not a new persisted index.

Reef does not join these endpoints into learning links or assign learning
states, human-readable explanations, policy capability flags, or evaluation
results. The console owns that interpretation. Consumption is not proof that a
release made from the records serves.
