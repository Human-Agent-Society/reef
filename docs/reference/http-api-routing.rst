HTTP API: scenario routing
==========================

Route inference to external models, inspect routing records, and control admission on managed deployments.

Routed external-model requests
------------------------------

With `Scenario routing <configuration-routing.rst#scenario-routing-between-students-and-teachers>`__ enabled, send ``POST /v1/chat/completions`` to Reef with a service token. Only the base record-only Recipe is supported.

- Routed Reef serves this route, ``POST /reef/report`` and ``GET /healthz``. Other routes return 404.
- With the `external router lifecycle <configuration-routing.rst#external-router-lifecycle>`__, inference applies these rules only to manifest Scenarios and also serves ``GET /readyz``. Send reports to control.
- ``x-reef-rollout: 1`` or ``true`` returns 400 (``unsupported_rollout``). Rollout calls remain available when routing is disabled.

.. list-table:: Client fields
   :header-rows: 1

   * - Field
     - Required value
   * - ``x-reef-scenario``
     - A Scenario listed in the manifest.
   * - JSON ``model``
     - The manifest's automatic model name, or one of the Scenario's target IDs.

For the automatic model, Reef sends one ``POST /api/v1/routing/preview`` to the Scenario's private Semantic Router entrypoint. It sends the request messages and the router runtime hash verified at startup. The reply must match that hash, the Scenario's entrypoint, and ``default`` or a configured rule and target.

An explicit target pin needs no router call. Reef sets the provider ``model`` and preserves the rest of the client's original JSON body. Every attempt uses the same target.

Before creating a Scenario or calling a provider, Reef checks the request:

* An unconfigured Scenario returns 403 (``unknown_scenario``), and so does a pin to a target outside the Scenario (``target_not_allowed``).
* A missing Scenario or model, an unknown target, a private router entrypoint as ``model``, a repeated ``x-reef-*`` header, a client-supplied ``x-reef-routing-*``, ``x-reef-sr-decision`` or ``x-selected-model`` header, and a nonempty ``x-reef-artifact-id`` return 400.
* A disabled target returns 503.
* A managed deployment returns 408 when the request body does not finish uploading within 30 seconds.

The automatic model also returns these router errors:

* 400 if the router cannot classify the messages.
* 413 if the preview body exceeds 10 MiB. Reef does not call the router.
* 429 if the router is full.
* 503 for a connection refused, reset or closed before a reply (``routing_unavailable``), an unverified runtime (``routing_unpublished``), or a changed runtime (``routing_runtime_changed``).
* 504 for a router timeout.
* 502 for a malformed or truncated reply, or a selection inconsistent with the manifest and runtime (``routing_protocol`` or ``routing_selection``).

Reef does not retry these failures or send the request to another target. See the `Semantic Router tutorial <../../tutorials/semantic-router/README.md>`__.

Each dispatched attempt creates an INFERENCE record in the request's Scenario. Reef replaces caller-supplied ``payload.metadata.routing`` with:

- ``target_id``, ``role``, ``provider_model``, ``model_revision`` and ``snapshot_version``;
- ``router_config_hash``, or ``null`` for a pin;
- ``original_selector`` and ``reason`` (``default``, ``rule`` or ``explicit_pin``);
- ``rule_name``, or ``null`` for a pin.

``request_id``, ``attempt_index`` and ``prior_attempt_ids`` link retries. Use the successful attempt's receipt for Scenario-local REPORT feedback.

The deployment manifest declares ``model_revision``, with ``identity_source: deployment_manifest``. The declared revision identifies the configured target; it does not prove which weights the provider loaded. The record has no Reef ``artifact_id``, and Reef sends the target no artifact identity. Reef preserves the provider's returned log probabilities in the response and the record.

``execution_outcome`` is ``completed``, ``aborted``, ``error``, ``timeout`` or ``cancelled``. Interrupted streams use ``unknown`` when Reef cannot determine execution. Failed attempts contain diagnostics without training data.

``delivery_outcome`` is ``not_started`` before delivery, ``incomplete`` for an interrupted stream, or ``unconfirmed`` after successful generation. Reef stores the record before sending the response or SSE receipt and terminal event. It cannot confirm client receipt. ``client_visible`` also describes the serving attempt, not a delivery acknowledgement.

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

Controller commands:

- ``GET /reef/lifecycle`` returns ``deployment_id``, ``generation``, ``snapshot_version``, ``router_config_hash`` (empty until the owner publishes a verified runtime), ``control_revision``, ``accepting``, ``stopping``, ``policy_closed``, ``active_requests``, ``pending_selections``, ``pending_writes``, ``drain_pending_writes`` and the diagnostic counts ``forced_requests`` and ``failed_writes``.
- ``POST /reef/lifecycle`` takes the deployment ID, the generation, a nonnegative integer ``control_revision`` and an ``action``: ``open``, ``close``, ``close_policy``, ``abort_policy``, ``activate`` or ``stop``. Revisions increase monotonically. The same revision and command again is idempotent. Stale revisions, conflicting commands and identity mismatches return 409.
- ``open`` and ``abort_policy`` require ``open_deadline_monotonic``. ``open`` returns 409 until a router runtime hash is published and until the writes of a close's drain finish. On a routed deployment it first checks every provider and sends one preview per Scenario; a failed check, or a missed deadline, returns 503 and changes nothing.
- ``activate`` publishes ``manifest_path``, ``snapshot_version``, ``expected_snapshot_version`` and a 64-hex ``router_config_hash`` together, and leaves admission closed.
- ``close`` and ``stop`` accept ``cancel_active: true`` to cancel accepted inference requests after their drain deadline.
- ``POST /reef/lifecycle/check`` checks a candidate manifest and changes nothing. It returns 200 with ``{"snapshot_version": ...}``, 409 with the fixed text ``routing candidate rejected`` for a candidate that the child would not publish, 503 when a provider fails or the deadline passes, and 400 for a malformed body.

``reef/api/lifecycle.py`` defines the fields and the drain accounting of each command.
