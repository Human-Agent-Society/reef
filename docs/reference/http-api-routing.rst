HTTP API: scenario routing
==========================

Route inference to external models, inspect routing records, and control admission on managed deployments.

Routed external-model requests
------------------------------

`Scenario routing <configuration-routing.rst#scenario-routing-between-students-and-teachers>`__ requires the base record-only Recipe and a service token.

- Supported routes: ``POST /v1/chat/completions``, ``POST /reef/report``, ``GET /healthz``. Others return 404.
- With the `external router lifecycle <configuration-routing.rst#external-router-lifecycle>`__, these rules cover manifest Scenarios; inference also serves ``GET /readyz`` and control receives reports.
- ``x-reef-rollout: 1`` or ``true`` returns 400 (``unsupported_rollout``). Rollout calls remain available when routing is disabled.

.. list-table:: Client fields
   :header-rows: 1

   * - Field
     - Required value
   * - ``x-reef-scenario``
     - A Scenario listed in the manifest.
   * - JSON ``model``
     - The manifest's automatic model name, or one of the Scenario's target IDs.

Automatic selection sends messages and the verified runtime hash to the Scenario's private ``POST /api/v1/routing/preview``. Replies must match the hash, entrypoint and configured default or rule target. Pins bypass the router. Reef changes only ``model`` in the original body; all attempts keep the target.

Before Scenario creation or provider execution:

* 403: unconfigured Scenario (``unknown_scenario``) or foreign target pin (``target_not_allowed``).
* 400: missing Scenario/model, unknown target, private entrypoint as ``model``, repeated ``x-reef-*``, supplied ``x-reef-routing-*``/``x-reef-sr-decision``/``x-selected-model``, or nonempty ``x-reef-artifact-id``.
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

``model_revision`` has ``identity_source: deployment_manifest``: configured identity, not proof of loaded weights. Records have no Reef ``artifact_id``; targets receive no artifact identity. Provider log probabilities are preserved.

``execution_outcome``: ``completed``, ``aborted``, ``error``, ``timeout``, ``cancelled``, or ``unknown`` for indeterminate interrupted streams. Failed attempts contain diagnostics only.

``delivery_outcome``: ``not_started``, ``incomplete`` or successful-but-``unconfirmed``. Records precede response delivery, so neither this field nor ``client_visible`` acknowledges client receipt. Persistence failures produce no success receipt or successful SSE terminal event. Recording does not train or provide OPD tensors.

Managed routing lifecycle
-------------------------

Managed deployments start closed: inference returns 503 until admission opens.
``/healthz`` checks liveness, not admission. Policy changes do not alter admitted requests.
Reports can finish during close; stop rejects new mutations permanently for that generation.

Operators use ``reef serve``, status and cleanup commands.
The controller's private Unix-socket API uses separate generation credentials,
not service tokens. See `Managed routing lifecycle
<configuration-routing.rst#managed-routing-lifecycle>`__.
