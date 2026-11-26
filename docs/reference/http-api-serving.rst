HTTP API: what a scenario serves
================================

Choose served artifacts, split traffic, call a specific artifact, or restore an earlier selection.

Set what a scenario serves
--------------------------

A scenario can serve multiple artifacts per component, with one active artifact. ``GET /reef/harness`` returns the active harness tree. Default inference routes use the active artifact once startup completes. Pulling an older artifact changes only your local copy.

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

Reef validates the mapping before applying it. Rejected mappings change nothing. See `Runtime limits <../user-guide/operate.rst#set-what-a-scenario-serves>`__ for artifact capacity and the delay before stopped checkpoints release their slots.

The response is ``{"scenario": "<name>", "served": {...}}`` with the accepted component mapping. Inference loads artifacts asynchronously and publishes routes after startup.

- Poll `Startup status <http-api-inference.rst#startup-status>`__ for ``ready`` or ``failed`` before you send requests that require a new artifact. Name an artifact that serves beside the active one with ``?model=<artifact-id>``.
- A startup can fail after Reef accepted the mapping, for example when a new checkpoint finds no free inference component.
- A repeated request retries a failed startup at once without another operation, and sets the wait of the automatic retries back to 60 seconds.

Reef records one operation per changed artifact:

- ``activate`` selects a new active artifact.
- ``serve`` starts serving another artifact beside it.
- ``stop_serving`` removes a served artifact.

Each operation stores the complete component mapping, which survives restarts. Other components keep their artifacts. The request creates no artifact or repository revision and preserves training starts. Reactivating a root snapshot on a full-weight runtime loads the root's files.

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

Reef starts artifacts that are not already served. The active artifact handles their requests until startup completes. Without a ready active route, these requests return 503. Inference replicas apply new weights within seconds, without restarting. One positive weight ends the split.

The activation rule can make a newer artifact active during a split. The new active artifact then answers the requests of each artifact of the split that the scenario no longer serves, and the other artifacts keep their shares.

The weights apply to a model request that names ``active``, or a model that the scenario does not serve under that name:

- **One conversation stays on one artifact.** Send the ``x-reef-session-key`` header with an identifier of the conversation. Reef then computes a bucket from the scenario and that identifier.
- **Without the header, Reef uses the initial messages.** The bucket uses the scenario and text from the system, first and first user messages. Shortening history can move a conversation to another artifact. Identical initial messages select the same artifact, so fixed prompts can skew observed percentages.
- **A changed weight moves few conversations.** A change from 95/5 to 80/20 moves only the conversations of the 15 buckets that go to another artifact.
- **Some requests do not use the weights** and keep their usual target:

  - a request without a message;
  - a request with ``previous_response_id`` and no ``x-reef-session-key``, which goes to the active artifact;
  - a request that names an artifact or an artifact's route;
  - a rollout call or an evaluation call.

- **Reef does not move a failed request.** A request to a failed artifact fails. To stop new requests to an artifact, set its weight to 0.
- **A retry keeps its artifact.** Reef does not compute the bucket again when it retries a request (see `Request <http-api-inference.rst#request>`__). When the artifact has no route at the retry, Reef answers HTTP 503 with ``Retry-After``.

The weights select the artifact, and the artifact's engine selects its own replica.

Records identify the responding artifact in ``artifact_id`` and store the bucket and weights in ``metadata.traffic``. Samples remain non-stale while their artifact is served, regardless of weight (`Call one served artifact`_). Jobs can include samples from several artifacts, each with its own artifact ID and recorded log probabilities.

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

You can submit a nonempty mapping immediately. The runtime reuses verified bindings or reloads retired checkpoints. Poll each artifact's `Startup status <http-api-inference.rst#startup-status>`__ before calling it. New checkpoints still need a free inference component (`Replace a checkpoint <../user-guide/operate.rst#set-what-a-scenario-serves>`__).

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

Call ``/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route}``, where ``{route}`` is ``chat/completions``, ``responses``, ``messages`` or ``messages/count_tokens``.

- No ``x-reef-scenario`` header is needed. The call never creates a scenario.
- Reef records the call as for ``/v1/{route}``, including the responding ``artifact_id``.
- You can also call ``/v1/{route}`` with the artifact ID as ``model``.
- An artifact that is not served returns 404.

The same prefix with ``/evaluation/v1/{route}`` makes the same call without a record.


These rules apply to an artifact that serves beside the active one:

- In a scenario with several components, it stays bound to the artifacts of the other components that were active when it started to serve. A later activation does not change that binding.
- The activation rule and a restore change the active artifact alone. The artifacts served beside it stay served. Training continues from each trainer's own training start.
- The samples service does not mark a sample stale while the scenario serves the artifact that produced it. After that artifact stops serving, the check counts only the artifacts that the scenario served as active: an on-policy stage marks a sample whose artifact is not the active one.

- A scenario whose components load nothing into a runtime, such as a harness scenario, can serve any of its artifacts together. `Set what a scenario serves`_ gives the limits of every other scenario.


Bind one chat attempt
~~~~~~~~~~~~~~~~~~~~~

Select a target before sending a non-streaming Chat Completions request on an artifact route.
This also supports unrecorded evaluation calls. Calls without binding headers keep their usual behavior.

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

The binding identifies the serving configuration.
Weight-bearing runtimes must pin admitted requests to the selected model name or use an immutable provider revision.
A record-only artifact identifies the configured model; it cannot guarantee an external provider's weights.
Callers must validate their handler's reported usage and configured token limits.

``attempt_id`` correlates requests but is neither persisted nor used for deduplication.
The caller owns durable admission, retries and result recovery.
After a timeout or lost response, treat execution as uncertain. Do not resend the same attempt.
Older servers reject both attempt paths without calling a model, so mixed deployments cannot silently use the ordinary retry behavior.


Restore a scenario selection
----------------------------

Send ``POST /reef/scenarios/{scenario}/restore`` with ``{"commit_number": 12}`` to restore all components' active artifacts after that commit. Use ``{"commit_number": 0}`` for starting artifacts.

The answer is ``{scenario, components}``, where ``components`` maps every component name to its restored ``artifact_id``. A changed selection appends a ``restore`` operation. It creates no artifact or repository revision and leaves each trainer's start, the other served artifacts and the traffic split unchanged. Restoring the current selection changes nothing. Poll `Startup status <http-api-inference.rst#startup-status>`__ before you use the restored inference artifact.

Reef returns:

- 400 for a body that holds anything but a non-negative integer ``commit_number``, for an unknown commit number, and for one whose selection cannot be reconstructed from the retained history;
- 404 for an unknown scenario;
- 409 while a training result waits for its commit.

Artifact validation and runtime capabilities still apply.

Serving settings
----------------

Artifacts contain content; serving configuration selects their model and runtime settings. For multiple base models, use each scenario's `serving config <http-api-scenarios.rst#serving-config>`__. The inference runtime defines and validates these fields. Artifacts without an override use the deployment's model.

You can change a setting at any time. The runtime applies it when inference next serves the scenario. An artifact that runs keeps its engine.

An artifact that training makes names its training starts in its own metadata, ``training_starts``: the artifact ID of its training start, then that start's own training starts. A runtime uses them to serve a training artifact as its start is served, unless the artifact has its own setting.
