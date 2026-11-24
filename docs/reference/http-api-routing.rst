HTTP API: scenario routing
==========================

The routes of a deployment that routes each request to an external model: how Reef selects the target, which errors it answers, what the record holds, and how the deployment controller opens and closes a managed deployment.

Routed external-model requests
------------------------------

When `Scenario routing <configuration-routing.rst#scenario-routing-between-students-and-teachers>`__ is enabled, clients send ``POST /v1/chat/completions`` directly to Reef with a service token. Only the base record-only Recipe is supported. A routed Reef serves only this route, ``POST /reef/report`` and ``GET /healthz``; other routes return 404. With the `external router lifecycle <configuration-routing.rst#external-router-lifecycle>`__, the ``inference`` service applies this section to the manifest's Scenarios only, and also serves ``GET /readyz``. Send ``POST /reef/report`` to the ``control`` service there. Calls with ``x-reef-rollout: 1`` or ``true`` return 400 (``unsupported_rollout``). Rollout calls remain available when routing is disabled.

.. list-table:: Client fields
   :header-rows: 1

   * - Field
     - Required value
   * - ``x-reef-scenario``
     - A Scenario listed in the manifest.
   * - JSON ``model``
     - The manifest's automatic model name, or one of the Scenario's target IDs.

For the automatic model, Reef sends one ``POST /api/v1/routing/preview`` to the Scenario's private Semantic Router entrypoint, with the request's messages and the router runtime hash Reef verified at startup. Reef accepts the reply only if it names that hash, the Scenario's entrypoint, and either ``default`` or one of the Scenario's rules with its configured target. Reef resolves a pin locally and never calls the router. Reef then sets the provider ``model``; the rest of the JSON body is the client's original body, not the router's re-encoded copy. The target is frozen for every attempt of the request. These checks precede Scenario creation and provider execution:

* An unconfigured Scenario returns 403 (``unknown_scenario``), and so does a pin to a target outside the Scenario (``target_not_allowed``).
* A missing Scenario or model, an unknown target, a private router entrypoint as ``model``, a repeated ``x-reef-*`` header, a client-supplied ``x-reef-routing-*``, ``x-reef-sr-decision`` or ``x-selected-model`` header, and a nonempty ``x-reef-artifact-id`` return 400.
* A disabled target returns 503.
* A managed deployment returns 408 when the request body does not finish uploading within 30 seconds.
* For the automatic model only: messages the router cannot classify return 400, a preview body over the router's 10 MiB limit returns 413 without calling the router, and a full router returns 429. A router connection that is refused, reset or closed before any reply arrives, as when SR stops or restarts, returns 503 (``routing_unavailable``), and so do an unverified runtime (``routing_unpublished``) and a runtime that changed (``routing_runtime_changed``). A router timeout returns 504. A reply that starts but is malformed or cut off, or a selection that does not match the manifest and runtime, returns 502 (``routing_protocol`` or ``routing_selection``).

Reef does not retry these failures or send the request to another target. See the `Semantic Router tutorial <../../tutorials/semantic-router/README.md>`__.

Each dispatched primary attempt produces an INFERENCE record in the request's Scenario. Server-generated ``payload.metadata.routing`` replaces caller-supplied routing metadata and contains ``target_id``, ``role``, ``provider_model``, ``model_revision``, ``snapshot_version``, ``router_config_hash`` (``null`` for a pin), ``original_selector``, the selection ``reason`` (``default``, ``rule`` or ``explicit_pin``) and ``rule_name`` (``null`` for pins). ``request_id``, ``attempt_index`` and ``prior_attempt_ids`` link retries; the successful attempt's normal receipt remains the reference for Scenario-local REPORT feedback. Primary attempts have ``purpose: serve``.

When a Scenario has a `shadow target <configuration-routing.rst#shadow-targets>`__, a shadow whose provider call started produces a separate INFERENCE record if the store accepts the write. Its routing metadata includes ``purpose: shadow``, ``reason: shadow_target``, ``rule_name: null``, ``client_visible: false`` and ``delivery_outcome: not_started``. The shadow gets its own ``request_id`` and links to the primary's ``request_id`` through ``primary_request_id``. Both carry the primary decision's ``original_selector``, ``snapshot_version`` and ``router_config_hash``. The client receives only the primary receipt. The shadow record has its own trajectory and no token-level training payload. The next conversation turn continues the primary trajectory.

A shadow writes to the Scenario instance or storage generation it started with. A managed service rejects a late write after that instance is unloaded, replaced or deleted. An external replica rejects writes to an archived generation. A skipped shadow or a shadow cancelled before its provider call starts leaves no record.

On routed requests, Reef replaces caller-supplied routing metadata. Native inference routes return 400 for a client-supplied ``metadata.routing`` with ``purpose: shadow`` or a ``primary_request_id`` field. Imported records can retain these fields, and a custom processor controls whether to use those records for training.

For both primary and shadow records, ``snapshot_version`` identifies the manifest used for the request. Look up ``target_id`` in that manifest to find the endpoint and provider model. The manifest supplies ``model_revision`` (``identity_source: deployment_manifest``). To compare specific checkpoints, confirm which weights each provider loaded. Reef sends no artifact identity headers, and the record has no Reef ``artifact_id``. Reef preserves the provider's returned log probabilities in the response and the record.

``execution_outcome`` is ``completed``, ``aborted``, ``error``, ``timeout`` or ``cancelled``; interrupted streams whose execution cannot be established use ``unknown``. Failed attempts are diagnostic records without training data. ``delivery_outcome`` is ``not_started`` for failures before delivery, or ``incomplete`` for interrupted streams. Successful generation uses ``unconfirmed``: Reef persists the record before returning a buffered response or writing the SSE receipt and terminal event, so it cannot confirm client receipt. ``client_visible`` likewise describes the serving attempt, not a delivery acknowledgement.

Persistence failure produces no success receipt. An SSE response that has already started ends without a successful terminal event. This recording mode does not train models or supply the sampled tensors required for OPD.

Managed routing lifecycle
-------------------------

With ``reef.routing_manifest`` set, the deployment controller commands the routed Reef child through ``GET /reef/lifecycle``, ``POST /reef/lifecycle`` and ``POST /reef/lifecycle/check``. This interface is private to Reef:

- The child serves it on a private Unix socket (directory 0700, socket 0600), never on ``reef.port``.
- It requires a separate per-generation control credential. Service and evaluation tokens cannot open it, and the control credential opens no public route.
- Operators use ``reef serve`` and the deployment status and cleanup commands instead (`Managed routing lifecycle <configuration-routing.rst#managed-routing-lifecycle>`__).

A client of the public routes sees these effects:

- Managed inference starts closed. A closed deployment rejects new inference with 503 without invoking a provider. ``GET /healthz`` remains a liveness check; it does not mean that the deployment accepts inference.
- Reef fixes admission and the request's routing policy together before it reads the body. An admitted request keeps the manifest and router runtime hash that it captured, also while the router previews it and when a policy is published meanwhile.
- An accepted request remains active through local response EOF or connection termination, including SSE terminal frames. Local EOF does not establish client receipt.
- Reports can finish during a ``close``. A ``stop`` rejects new mutations and is terminal for the generation.

The controller's side:

- ``GET /reef/lifecycle`` returns ``deployment_id``, ``generation``, ``snapshot_version``, ``router_config_hash`` (empty until the owner publishes a verified runtime), ``control_revision``, ``accepting``, ``stopping``, ``policy_closed``, ``active_requests``, ``pending_selections``, ``pending_shadows`` (active shadow calls, including record finalization), ``pending_writes``, ``drain_pending_writes`` and the diagnostic counts ``forced_requests`` and ``failed_writes``.
- ``POST /reef/lifecycle`` takes the deployment ID, the generation, a nonnegative integer ``control_revision`` and an ``action``: ``open``, ``close``, ``close_policy``, ``abort_policy``, ``activate`` or ``stop``. Revisions increase monotonically. The same revision and command again is idempotent. Stale revisions, conflicting commands and identity mismatches return 409.
- ``open`` and ``abort_policy`` require ``open_deadline_monotonic``. ``open`` returns 409 until a router runtime hash is published and until the writes of a close's drain finish. On a routed deployment it first checks every enabled provider except those used only for shadows and sends one preview per Scenario; a failed check, or a missed deadline, returns 503 and changes nothing.
- ``activate`` publishes ``manifest_path``, ``snapshot_version``, ``expected_snapshot_version`` and a 64-hex ``router_config_hash`` together, and leaves admission closed.
- ``close`` and ``stop`` accept ``cancel_active: true`` to cancel accepted inference requests after their drain deadline.
- ``POST /reef/lifecycle/check`` checks a candidate manifest and changes nothing. It returns 200 with ``{"snapshot_version": ...}``, 409 with the fixed text ``routing candidate rejected`` for a candidate that the child would not publish, 503 when a provider fails or the deadline passes, and 400 for a malformed body.

``reef/api/lifecycle.py`` defines the fields and the drain accounting of each command.
