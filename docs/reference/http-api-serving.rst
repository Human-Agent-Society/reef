HTTP API: what a scenario serves
================================

Choose the artifacts that a scenario serves, divide its requests between them, call one served artifact, stop serving, and restore an earlier selection.

Set what a scenario serves
--------------------------

A scenario serves one or more artifacts of each component, and one of them is the active artifact. ``GET /reef/harness`` serves the active harness tree. The default inference routes answer from the active artifact that completed startup. Pulling an older artifact changes only your local copy.

The ``served`` setting of ``POST /reef/scenarios/{scenario}/update`` sets what the scenario serves of one component. It maps artifact IDs to weights:

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {"<id>": 100}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

These examples use the artifacts ``a`` and ``b`` of one component, where ``a`` is the active artifact before the request:

+--------------------------+-------------------------------------------------+
| ``served``               | Result                                          |
+==========================+=================================================+
| ``{"b": 100}``           | ``b`` becomes the active artifact, and Reef     |
|                          | stops every other served artifact               |
+--------------------------+-------------------------------------------------+
| ``{"a": 95, "b": 5}``    | a traffic split: ``b`` answers 5% of the        |
|                          | requests                                        |
+--------------------------+-------------------------------------------------+
| ``{"a": 100, "b": 0}``   | Reef loads ``b``, which answers only the        |
|                          | requests that name it                           |
+--------------------------+-------------------------------------------------+
| ``{"a": 100}``           | Reef stops serving ``b``                        |
+--------------------------+-------------------------------------------------+
| ``{}``                   | the scenario serves nothing (`Serve nothing`_)  |
+--------------------------+-------------------------------------------------+

The body is ``{"served": {"<id>": <weight>, ...}, "component": "<name>"}``:

- ``served`` holds one or more artifact IDs of one component. Each weight is an integer from 0 to 100, and the weights sum to 100. An empty object makes the scenario serve nothing.
- ``component`` names the component. A scenario with several components requires it; otherwise leave it out. Leave it out with an empty ``served``, which applies to the whole scenario.

Reef reads the mapping as follows:

- **The largest weight selects the active artifact.** Among equal weights, the first ID in sorted order is active. The active artifact answers each request that names no artifact, unless a traffic split gives the request to another artifact.
- **More than one weight above 0 is a traffic split** (`Traffic split`_).
- **Weight 0 loads an artifact.** It answers only the requests that name it (`Call one served artifact`_).
- **An artifact that the mapping leaves out stops serving.**

Reef checks the mapping before it changes anything, so a refused mapping changes nothing. `Runtime limits <../user-guide/operate.rst#set-what-a-scenario-serves>`__ gives the number of weight artifacts that the runtime keeps loaded at once, and the wait before a stopped checkpoint frees its place.

The answer is ``{"scenario": "<name>", "served": {...}}`` with the mapping that Reef accepted for the component. It does not show that the artifacts serve: inference loads them in the background, and publishes the new routes when the startup completes.

- Poll `Startup status <http-api-inference.rst#startup-status>`__ for ``ready`` or ``failed`` before you send requests that require a new artifact. Name an artifact that serves beside the active one with ``?model=<artifact-id>``.
- A startup can fail after Reef accepted the mapping, for example when a new checkpoint finds no free inference component.
- A repeated request retries a failed startup at once without another operation, and sets the wait of the automatic retries back to 60 seconds.

Reef appends one operation for each artifact that changes: ``activate`` for a new active artifact, ``serve`` for an artifact that starts to serve beside it, and ``stop_serving`` for one that stops. Each operation holds the complete component mapping, so a restart keeps what the scenario serves. Other components keep their artifacts. The request creates no artifact or repository revision, and never changes a trainer's training start. A full-weight serving runtime that makes the root snapshot active after it served a checkpoint loads the root's files.

+--------+--------------------------------------------------------------------+
| Status | Cause                                                              |
+========+====================================================================+
| 200    | Reef accepted the mapping; the artifacts load in the background    |
+--------+--------------------------------------------------------------------+
| 400    | a malformed body; a ``served`` that is no object, a weight that is |
|        | not an integer from 0 to 100, or weights that do not sum to 100;   |
|        | an empty ``served`` with ``component``;                            |
|        | ``component`` without ``served``; an unknown ``component``, or     |
|        | ``component`` is missing in a scenario with several components;    |
|        | an artifact has no durable bytes; the runtime cannot serve the     |
|        | artifact; the mapping names more weight artifacts than the         |
|        | runtime keeps loaded at once, or several artifacts of another      |
|        | component in a scenario whose weights the runtime loads; the       |
|        | mapping names several artifacts, and the store keeps no durable    |
|        | commit history                                                     |
+--------+--------------------------------------------------------------------+
| 404    | unknown scenario; an artifact that is not one of the component     |
+--------+--------------------------------------------------------------------+
| 409    | a training result waits for its commit; send the request again     |
|        | after the commit                                                   |
+--------+--------------------------------------------------------------------+

`Activation rule <../user-guide/operate.rst#activation-rule>`__ explains when Reef makes a new artifact active automatically. `Go back to an older artifact <../user-guide/operate.rst#go-back-to-an-older-artifact>`__ explains how to go back to an older one for good.

Traffic split
-------------

When ``served`` gives more than one artifact a weight above 0, the weights divide the scenario's model requests between those artifacts (`Set what a scenario serves`_). For example, this mapping sends 5% of the requests to a new artifact:

.. code:: json

   {"served": {"artifact:a": 95, "artifact:b": 5}}

Reef starts each artifact of the mapping that does not serve yet. Until an artifact has started, the active artifact answers its requests. Without a ready active route, as at the first startup or when the scenario serves again after serving nothing, they get HTTP 503 until a route is ready. Inference replicas use new weights within a few seconds, without a restart. A mapping with one weight above 0 ends the split.

The activation rule can make a newer artifact active during a split. The new active artifact then answers the requests of each artifact of the split that the scenario no longer serves, and the other artifacts keep their shares.

The weights apply to a model request that names ``active``, or a model that the scenario does not serve under that name:

- **One conversation stays on one artifact.** Send the ``x-reef-session-key`` header with an identifier of the conversation. Reef then computes a bucket from the scenario and that identifier.
- **Without the header, Reef reads the first messages.** It computes the bucket from the scenario and the text of the system message, the first message and the first user message. Later turns repeat these messages, so a client that shortens its history can move to another artifact. Conversations that start with the same messages get the same artifact, so fixed prompts can change the observed percentages.
- **A changed weight moves few conversations.** A change from 95/5 to 80/20 moves only the conversations of the 15 buckets that go to another artifact.
- **Some requests do not use the weights** and keep their usual target:

  - a request without a message;
  - a request with ``previous_response_id`` and no ``x-reef-session-key``, which goes to the active artifact;
  - a request that names an artifact or an artifact's route;
  - a rollout call or an evaluation call.

- **Reef does not move a failed request.** A request to a failed artifact fails. To stop new requests to an artifact, set its weight to 0.
- **A retry keeps its artifact.** Reef does not compute the bucket again when it retries a request (see `Request <http-api-inference.rst#request>`__). When the artifact has no route at the retry, Reef answers HTTP 503 with ``Retry-After``.

The weights select the artifact, and the artifact's engine selects its own replica.

The record names the artifact that answered in ``artifact_id``, and holds the bucket and the weights in ``metadata.traffic``. While the scenario serves an artifact, the samples service does not mark its records' samples stale, whatever its weight (`Call one served artifact`_). A job can then hold samples of several artifacts. Each sample keeps its own ``artifact_id`` and recorded log probabilities.

Serve nothing
-------------

A scenario can serve nothing, for example to train on uploaded trajectories without an engine. A scenario enters this state in two ways:

- ``POST /reef/scenarios`` with ``"serves": false`` creates it in this state.
- ``{"served": {}}`` without ``component`` puts an existing scenario in it.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

The state applies to the whole scenario, with every component. While the scenario serves nothing:

- Reef withdraws the scenario's routes and its traffic split, and stops the artifacts that it served beside the active one. The engines stop after the delay that a replaced route gets.
- The inference routes and `Startup status <http-api-inference.rst#startup-status>`__ answer HTTP 404 for the scenario.
- ``GET /reef/scenarios/{scenario}/settings`` returns an empty ``served`` mapping for each component.
- The scenario keeps its active selection, and the artifact list still marks it ``active``. ``GET /reef/harness`` still serves the active harness tree.
- Training and imports run as before. An activation, also an automatic one, changes the active selection and starts no engine.

The scenario serves again when ``served`` names one or more artifacts. The other components then serve the artifacts of the active selection. The state holds after a restart.

A nonempty mapping can follow at once, without a wait for the delayed stop. The runtime can reuse a verified binding; a retired checkpoint loads again. Poll the `Startup status <http-api-inference.rst#startup-status>`__ of each requested artifact before calling it. A checkpoint that the scenario did not serve before still needs a free inference component (`Replace a checkpoint <../user-guide/operate.rst#set-what-a-scenario-serves>`__).

Call one served artifact
------------------------

Each served artifact also answers at its own URL, beside the default inference routes. Give an artifact weight 0 to:

- keep an older artifact answering while a newer one is active;
- try a new artifact before it is active.

.. code:: bash

   curl -sS -X POST -H "Authorization: Bearer $REEF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"served": {"<active id>": 100, "<id>": 0}}' \
     "$REEF_URL/reef/scenarios/code-repair/update"

Send an inference call to ``/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route}``, where ``{route}`` is ``chat/completions``, ``responses``, ``messages`` or ``messages/count_tokens``. The call needs no ``x-reef-scenario`` header and never creates a scenario. Reef records it like a call to ``/v1/{route}``, and the record names the ``artifact_id`` that answered. A served artifact also answers a call to ``/v1/{route}`` whose ``model`` is its ``artifact_id``. An artifact that the scenario does not serve has no URL: its calls answer HTTP 404.

The same prefix with ``/evaluation/v1/{route}`` makes the same call without a record.


These rules apply to an artifact that serves beside the active one:

- In a scenario with several components, it stays bound to the artifacts of the other components that were active when it started to serve. A later activation does not change that binding.
- The activation rule and a restore change the active artifact alone. The artifacts served beside it stay served. Training continues from each trainer's own training start.
- The samples service does not mark a sample stale while the scenario serves the artifact that produced it. After that artifact stops serving, the check counts only the artifacts that the scenario served as active: an on-policy stage marks a sample whose artifact is not the active one.

- A scenario whose components load nothing into a runtime, such as a harness scenario, can serve any of its artifacts together. `Set what a scenario serves`_ gives the limits of every other scenario.


Bind one chat attempt
~~~~~~~~~~~~~~~~~~~~~

Use this optional contract when the caller must select a target before sending a model request.
It supports non-streaming Chat Completions on the artifact routes, including the unrecorded evaluation route.
Existing calls without the new headers keep their routing, streaming and retry behavior.

First read ``GET /reef/scenarios/{scenario}/inference/target?model={name}`` with the existing service token.
``model`` names an exact serving row and defaults to ``active``. An unknown name returns 404, without fallback.
This read does not execute a model, reserve an engine, or extend an artifact's lifetime.

.. list-table:: Target response
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``contract_version``
     - ``1`` for this contract.
   * - ``model``
     - The selected row name. Use this name in the artifact route.
   * - ``artifact_id``
     - The artifact associated with that row.
   * - ``binding_id``
     - An opaque SHA-256 identifier of the scenario, row, engine address, engine model name and serving settings.
   * - ``protocol``
     - ``openai_chat_completions``.
   * - ``streaming``
     - ``false``. This contract does not yet support streams.

Then send the ordinary provider body to ``/reef/scenarios/{scenario}/artifacts/{model}/v1/chat/completions/attempt`` with both headers:

- ``X-Reef-Binding-Id``: the returned 64-character lowercase hexadecimal identifier.
- ``X-Reef-Attempt-Id``: a caller-owned identifier of 1 to 128 letters, digits or ``._:-``.

For example, resolve ``model=artifact-1``, then call the ``artifacts/artifact-1`` route with the returned binding.
The response preserves the provider body, including any reported usage.
It adds the two request headers and the existing ``X-Reef-Artifact-Id`` header on success.
Recorded calls also keep their existing record receipt.
Use ``/artifacts/{model}/evaluation/v1/chat/completions/attempt`` under the same scenario prefix for an unrecorded call.

Reef reads the current row and settings without its stale-read fallback.
A missing row returns 404. A changed binding returns 409 before execution.
An unavailable or concurrently changing serving table returns 503 before execution.
Malformed headers, default routes and streaming requests return 400 before execution.
Unsupported attempt routes return 404.
An accepted request keeps its selected row and settings, even if another artifact becomes active.
An engine binding refusal returns 503; an engine abort returns 409. The replica does not retry either response.
It invokes the configured inference handler at most once per accepted HTTP request.

The binding describes a serving configuration, not new engine guarantees.
Weight-bearing runtimes must already pin admitted requests to the selected model name or use an immutable provider revision.
A record-only artifact identifies its configured model, not immutable weights at an external provider.
The response does not invent token counts or token limits. Callers must validate their handler's usage and configured limits.

``attempt_id`` correlates a request; Reef does not persist it or deduplicate repeated HTTP requests.
The caller owns durable admission, retry decisions and result recovery.
After a timeout or lost response, it must treat execution as uncertain rather than resend the same attempt.
This addition changes no stored format, service configuration, dependencies or payment behavior.
It can be rolled out before callers opt in. Older servers reject both new paths without calling a model.
Thus a mixed deployment can return an unsupported-route error, but cannot silently run a strict call through the old retry loop.


Restore a scenario selection
----------------------------

To restore every component's active artifact together, send ``POST /reef/scenarios/{scenario}/restore`` with ``{"commit_number": 12}``. The commit number names an existing operation, and Reef restores the active selection as it stood after that commit. Use ``{"commit_number": 0}`` for the starting artifacts.

The answer is ``{scenario, components}``, where ``components`` maps every component name to its restored ``artifact_id``. A changed selection appends a ``restore`` operation. It creates no artifact or repository revision and leaves each trainer's start, the other served artifacts and the traffic split unchanged. Restoring the current selection changes nothing. Poll `Startup status <http-api-inference.rst#startup-status>`__ before you use the restored inference artifact.

Reef returns:

- 400 for a body that holds anything but a non-negative integer ``commit_number``, for an unknown commit number, and for one whose selection cannot be reconstructed from the retained history;
- 404 for an unknown scenario;
- 409 while a training result waits for its commit.

Artifact validation and runtime capabilities still apply.

Serving settings
----------------

An artifact holds only its content, not which model serves it or how. A deployment that serves artifacts of several base models gives each scenario a `serving config <http-api-scenarios.rst#serving-config>`__ field that chooses the model of each artifact. The deployment's inference runtime defines and checks that field. An artifact without a setting is served with the deployment's model.

You can change a setting at any time. The runtime applies it when inference next serves the scenario. An artifact that runs keeps its engine.

An artifact that training makes names its training starts in its own metadata, ``training_starts``: the artifact ID of its training start, then that start's own training starts. A runtime uses them to serve a training artifact as its start is served, unless the artifact has its own setting.
