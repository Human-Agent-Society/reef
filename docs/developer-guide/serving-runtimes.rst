Serving runtimes
================

Implement ``InferenceRuntime`` to serve published weights through an engine or provider.

Production inference contract
-----------------------------

Every admitted request uses its selected artifact until generation ends.
Its response and stored record identify that artifact.
Reject unavailable or uncertain bindings before generation.

Use immutable revisions or receiver-side pinning that prevents weight replacement
until requests finish. Headers, health checks and pre-request load-ID checks
alone do not guarantee identity. Prefix caches must preserve identity too.

Required methods
----------------

- ``inference_handler`` returns the request handler.
- ``activate_checkpoint`` establishes a binding and returns its load ID before route publication.
- ``served_artifact`` reads an established route without loading; return ``None`` for uncertain or unavailable bindings.

Optional methods cover ownership, multiple artifacts and adoption.
On takeover, ``handler_for(url)`` must recognize retained endpoints before ``open``.
Reef removes unrecognized routes, opens the runtime, then calls ``adopt(served)``.
Return false unless adoption can establish the binding.
See `InferenceRuntime <../../reef/inference/interfaces.py>`__ for full signatures.

Use ``REQUEST_SESSION_KEY`` for conversation affinity; evaluations assign one key per rollout.
Automatic training waits for the selected artifact's route; manual requests do not.

Record-only/harness snapshots use ``endpoint_model_name`` without weight activation.
Their IDs identify snapshots, not verified provider weights.
External Semantic Router targets carry no Reef artifact identity.

Request pinning
~~~~~~~~~~~~~~~

Saved-weight runtimes require ``ServingEngines.request_pinning``:

- Distinct model identities reject stale routes before output.
- Updates stop admission and drain requests up to ``inference.timeout-s``.
  Disconnects release pins only after engine generation ends.
- Drain timeouts interrupt remaining requests and require recovery.
- Prefix sharing clears caches after draining, before weight changes.
- Ambiguous updates stop admission until replacement and verified reload.
  Explicit refusals preserve the old binding.
- Newer owners fence old control calls with HTTP 412.

Retirement completes pinned requests or interrupts them explicitly.
Never retry on another artifact after streaming starts.
Incomplete streams receive no receipt.
Evaluation relays ``inference_stream`` bytes unchanged and holds admission until closure.
Without native streaming, handlers return their JSON answer as one piece.

Keep native engine addresses private: all generation and updates pass through
the component. External KV stores and prefix sharing with SGLang disaggregation
are unsupported. See `serving restrictions <../reference/configuration-inference.rst>`__.

Saved-weight recovery
---------------------

Built-ins serve PEFT adapters, one full-weight checkpoint, or several checkpoint slots.
Verify load IDs on every engine; never reuse IDs for different weights.
Capacity refusal preserves current bindings.

Uncertain bindings trigger replacement and reload. Resume only after verification.
Without verified weights, remain paused and return no route.
Saved-weight takeover starts fresh engines and interrupts in-flight requests.
Integrations must preserve pause intent across partial failures.

Integration qualification
~~~~~~~~~~~~~~~~~~~~~~~~~

Test real engines with two artifacts whose generated outputs distinguish them.
Check content, response identity and stored identity together.
Fake-engine tests cannot establish weight identity.

Exercise promotion, takeover, unavailable bindings, interrupted streams, wrong load IDs,
partial loads, adapters and prefix caches.
Record integration versions, uncovered cases and performance under concurrency.

Provide an isolated ``InferenceQualification`` context-manager factory with a
content-based ``OutputOracle``:

.. code:: bash

   python -m tests.smoke.inference_cutover --target my_qualification:target --output /scratch/cutover

The output must be an absolute, new/empty directory outside the checkout.
``report.json`` lists passed and uncovered checks.
For native one-GPU tests, use ``python -m tests.smoke.native_inference --help``.
These checks are outside normal CI and do not qualify managed-router lifecycle.
