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

With ``reef.routing_manifest``, the controller uses private ``GET /reef/lifecycle``, ``POST /reef/lifecycle`` and ``POST /reef/lifecycle/check``:

- Unix socket only, never ``reef.port``: directory 0700, socket 0600.
- Per-generation control credentials only; service/evaluation tokens cannot access it, and control credentials cannot access public routes.
- Operators use ``reef serve``, status and cleanup commands (`Managed routing lifecycle <configuration-routing.rst#managed-routing-lifecycle>`__).

A client of the public routes sees these effects:

- Closed by default: inference returns 503 without calling providers. ``/healthz`` checks liveness, not admission.
- Admission fixes the manifest and runtime hash before reading the body; policy changes never alter admitted requests.
- Requests remain active through local EOF or termination, including SSE terminal frames; EOF does not confirm receipt.
- Reports can finish during ``close``. ``stop`` rejects new mutations and permanently stops that generation.

Controller commands:

- ``GET /reef/lifecycle`` returns ``deployment_id``, ``generation``, ``snapshot_version``, ``router_config_hash`` (empty until the owner publishes a verified runtime), ``control_revision``, ``accepting``, ``stopping``, ``policy_closed``, ``active_requests``, ``pending_selections``, ``pending_writes``, ``drain_pending_writes`` and the diagnostic counts ``forced_requests`` and ``failed_writes``.
- ``POST /reef/lifecycle`` requires deployment ID, generation, nonnegative ``control_revision`` and ``action``: ``open``, ``close``, ``close_policy``, ``abort_policy``, ``activate`` or ``stop``. Revisions increase monotonically; identical repeats are idempotent. Stale revisions, conflicts or identity mismatches return 409.
- ``open`` and ``abort_policy`` require ``open_deadline_monotonic``. ``open`` returns 409 until runtime publication and close-drain writes finish. It checks providers and one preview per Scenario; failures or deadline expiry return 503 without changes.
- ``activate`` publishes ``manifest_path``, ``snapshot_version``, ``expected_snapshot_version`` and a 64-hex ``router_config_hash`` together, and leaves admission closed.
- ``close`` and ``stop`` accept ``cancel_active: true`` to cancel accepted inference requests after their drain deadline.
- ``POST /reef/lifecycle/check`` changes nothing: 200 with ``{snapshot_version}``, 409 (``routing candidate rejected``), 503 for provider/deadline failure, or 400 for malformed input.

``reef/api/lifecycle.py`` defines the fields and the drain accounting of each command.
