HTTP API: what a scenario serves
================================

Choose served artifacts, split traffic, call a specific artifact, or restore an earlier selection.

Set what a scenario serves
--------------------------

A component can serve several artifacts, one active. ``GET /reef/harness`` returns the active tree; default inference uses the ready active artifact. Pulling older trees changes only local files.

``POST /reef/scenarios/{scenario}/update`` sets artifact traffic weights:

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

Body: ``{served: {artifact_id: weight}, component?}``. Weights are integers 0–100 totaling 100. Multiple components require ``component``; omit it for a single component or empty ``served``.

The highest weight selects the active artifact; sorted ID order breaks ties. Multiple positive weights split traffic. Zero-weight artifacts accept explicit requests only. Omitted artifacts stop serving. Empty mappings stop the whole scenario.

Validation is atomic. `Runtime limits <../user-guide/operate.rst#set-what-a-scenario-serves>`__ govern capacity and delayed slot release. Success returns ``{scenario, served}`` before asynchronous startup finishes.

- Poll `startup status <http-api-inference.rst#startup-status>`__ with ``?model=<artifact-id>``; accepted mappings can still fail startup.
- Repeating the mapping retries failures immediately, records no operation, and resets automatic retry delay to 60 seconds.

Each changed artifact records ``activate``, ``serve`` or ``stop_serving`` with the complete component mapping. Selections survive restarts; other components and training starts remain unchanged. No artifact or repository revision is created. Full-weight runtimes reload root files when reactivating root snapshots.

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

See `Activation rule <../user-guide/operate.rst#activation-rule>`__ for automatic selection.

Traffic split
-------------

Multiple positive weights divide model requests; this sends 5% to ``artifact:b``:

.. code:: json

   {"served": {"artifact:a": 95, "artifact:b": 5}}

During startup, the active route handles each new artifact's share; without a ready active route, requests return 503. Weights apply within seconds. One positive weight ends the split. Later activations absorb shares of artifacts no longer served; remaining artifacts keep theirs.

Splits apply to ``model: active`` or unrecognized model names:

- With ``x-reef-session-key``, scenario and session determine the bucket, keeping a conversation on one artifact.
- Otherwise, scenario and system/first/first-user message text determine it. Shortened history can change artifacts; identical prompts can skew percentages.
- Weight changes move only reassigned buckets: 95/5→80/20 moves 15 buckets.
- **Some requests do not use the weights** and keep their usual target:

  - a request without a message;
  - a request with ``previous_response_id`` and no ``x-reef-session-key``, which goes to the active artifact;
  - a request that names an artifact or an artifact's route;
  - a rollout call or an evaluation call.

- Failed requests do not move to another artifact. Set weight 0 to stop new split requests.
- Retries keep their artifact; missing routes return 503 with ``Retry-After`` (`Request <http-api-inference.rst#request>`__).

The weights select the artifact, and the artifact's engine selects its own replica.

Records keep ``artifact_id`` and bucket/weights in ``metadata.traffic``. Served artifacts' samples remain non-stale regardless of weight. Jobs can mix artifacts while preserving each sample's ID and log probabilities.

Serve nothing
-------------

A scenario can train without serving. Use either:

- ``POST /reef/scenarios`` with ``"serves": false`` creates it in this state.
- ``{"served": {}}`` without ``component`` puts an existing scenario in it.

This affects all components:

- Routes and traffic splits are withdrawn; engines stop after the normal replacement delay.
- Inference and startup status return 404; scenario settings return empty component ``served`` mappings.
- Active selections remain marked ``active`` and accessible through ``GET /reef/harness``.
- Training, imports and activation continue without starting engines.

This state survives restarts. Submit a nonempty mapping anytime to resume; other components serve their active selections. Verified bindings can be reused; retired checkpoints reload. Poll startup status. New checkpoints still need capacity.

Call one served artifact
------------------------

Weight 0 keeps an artifact available for explicit requests without split traffic.

Call ``/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route}``, where ``{route}`` is ``chat/completions``, ``responses``, ``messages`` or ``messages/count_tokens``.

- No ``x-reef-scenario`` header is needed. The call never creates a scenario.
- Reef records the call as for ``/v1/{route}``, including the responding ``artifact_id``.
- You can also call ``/v1/{route}`` with the artifact ID as ``model``.
- An artifact that is not served returns 404.

The same prefix with ``/evaluation/v1/{route}`` makes the same call without a record.


Additional artifacts:

- Keep other-component bindings from their original serving selection, despite later activations.
- Stay served through activation or restore; training starts are unchanged.
- Keep samples non-stale until stopped. Afterward, staleness counts artifacts previously served as active; on-policy stages mark non-active samples stale.
- Have no concurrency limit for scenarios loading no runtime weights, such as harness scenarios. Other runtime limits still apply.


Bind one chat attempt
~~~~~~~~~~~~~~~~~~~~~

Bind non-streaming Chat Completions calls, recorded or evaluation, to a selected target.
Ordinary calls remain unchanged.

Read ``GET /reef/scenarios/{scenario}/inference/target?model={name}`` with the service token.
``model`` defaults to ``active``; unknown exact row names return 404 without fallback.
This reserves nothing, executes no model and does not extend artifact lifetime.

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

Success preserves provider body/usage and returns both headers, ``X-Reef-Artifact-Id`` and the recorded-call receipt.
For unrecorded calls, use ``/artifacts/{model}/evaluation/v1/chat/completions/attempt`` under the scenario prefix.

Before execution: missing rows/unsupported routes return 404, changed bindings 409, and unavailable or changing serving tables 503. Malformed headers, default routes and streams return 400. No stale-read fallback applies.

Accepted calls keep their row/settings through activation and invoke the handler at most once. Binding refusal returns 503; engine abort returns 409. Neither retries.

Bindings identify configuration. Weight runtimes must pin requests to the selected model or immutable provider revision. Record-only artifacts cannot guarantee provider weights. Callers validate usage and token limits.

``attempt_id`` is unpersisted correlation, not deduplication. Callers own admission, retries and recovery. Timeouts/lost responses mean uncertain execution; do not resend the attempt. Older servers reject attempt paths without execution or fallback to ordinary retries.


Restore a scenario selection
----------------------------

Send ``POST /reef/scenarios/{scenario}/restore`` with ``{"commit_number": 12}`` to restore all components' active artifacts after that commit. Use ``{"commit_number": 0}`` for starting artifacts.

Response: ``{scenario, components}``, mapping names to restored IDs. Changed selections append ``restore`` without creating artifacts or revisions. Training starts, additional served artifacts and splits remain unchanged. Restoring current selections is a no-op. Poll startup before inference.

Reef returns:

- 400 for a body that holds anything but a non-negative integer ``commit_number``, for an unknown commit number, and for one whose selection cannot be reconstructed from the retained history;
- 404 for an unknown scenario;
- 409 while a training result waits for its commit.

Artifact validation and runtime capabilities still apply.

Serving settings
----------------

Runtime-defined `serving config <http-api-scenarios.rst#serving-config>`__ selects model/settings, defaulting to deployment values. Changes apply when inference next serves the scenario; running artifacts keep their engines.

Training artifacts' ``training_starts`` metadata lists their start and its ancestors. Without an override, runtimes inherit serving settings from these starts.
