Serving runtimes
================

Implement ``reef.inference.interfaces.InferenceRuntime`` to serve published
weights through an engine or provider. A runtime establishes artifact
bindings, routes requests and manages serving resources.

Production inference contract
-----------------------------

Every admitted request must use its selected artifact until generation ends.
The response and stored record must identify that artifact.
Reject unavailable or uncertain bindings before generation.

Use one of these mechanisms:

- An immutable revision, such as a provider deployment or Tinker sampler revision.
- Receiver-side request pinning that prevents loading, unloading or replacing
  the selected weights until the request ends.

Published routes, echoed headers, health checks and pre-request load-ID checks
alone do not establish this guarantee. Prefix-cache reuse must preserve
artifact identity too.

What a runtime implements
~~~~~~~~~~~~~~~~~~~~~~~~~

Three members are required:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Contract
   * - ``inference_handler``
     - Return the request handler. HTTP integrations can reuse Reef's handler.
   * - ``activate_checkpoint``
     - Establish the binding and return its runtime load ID before publishing a route.
       Raise on failure.
   * - ``served_artifact``
     - Return an established binding's route without loading it.
       Return ``None`` for unknown, unavailable or uncertain bindings.

Optional members cover resource ownership, adoption and multiple artifacts.
See `InferenceRuntime <../../reef/inference/interfaces.py>`__ for their contracts.

Handlers can use ``REQUEST_SESSION_KEY`` to keep a conversation on one replica.
Evaluations assign one key per rollout, including rollouts with identical prompts.
Automatic weight training waits for the selected artifact's served route;
manual requests do not.

On takeover, ``handler_for(url)`` must recognize retained endpoints before ``open``.
Reef removes routes without a handler, then opens the runtime and calls
``adopt(served)`` for the remaining routes.
Return false when adoption cannot establish the binding.
Keep uncertain bindings unavailable until verified recovery.

Retirement must finish pinned requests or interrupt them explicitly.
Never retry a stream on another artifact after output starts.
Incomplete streams receive no receipt.
Evaluation relays ``inference_stream`` bytes unchanged and holds admission until closure.
A handler without native streaming returns its JSON answer as one piece.

Routes that bind no weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Record-only and harness snapshots use ``endpoint_model_name(scenario)``
without weight activation. Their artifact IDs identify snapshots, not verified
endpoint weights. ``inference_proxy`` supports this mode and rejects weight snapshots.

Semantic Router external targets are not Reef artifacts. Their requests and
records carry no artifact identity and require an OpenAI chat endpoint.
See `Routed external-model requests
<../reference/http-api-routing.rst#routed-external-model-requests>`__.

Serving saved weights
---------------------

Built-in saved-weight runtimes load published artifacts when a scenario starts
or its selection changes:

- ``AdapterServingRuntime`` serves per-scenario PEFT adapters on one base model.
  ``adapter_capacity`` bounds residency; active adapters are not eviction targets.
- ``FullWeightServingRuntime`` serves one scenario's full weights.
  Engines remain paused until the first verified activation.
  Base roots load ``model_path``.
- ``CheckpointSlotsRuntime`` serves several full-weight checkpoints using
  separate inference components from ``url`` and ``slot-urls``.
  It refuses new checkpoints when all slots still serve artifacts.
  Released slots retain their engines.

Verify each load ID on every engine; never reuse IDs for different weights.
Loading served content is a no-op. The artifact format selects full-weight
or adapter serving; inference reserves GPUs separately from training.

**Recovery.** Lost engines or uncertain bindings trigger replacement and reload.
Resume only after load-ID verification. Route queries return none until recovery.
Full-weight runtimes without verified checkpoints stay paused.
Adapter recovery preserves targets until restoration completes.

**Takeover.** Built-in saved-weight runtimes start fresh engines, interrupt
in-flight requests and reactivate completed selections.
Old load IDs cannot prove retained weight identity.
See `Startup status <../reference/http-api-inference.rst#startup-status>`__.

Request pinning
~~~~~~~~~~~~~~~

Saved-weight runtimes require ``ServingEngines.request_pinning``.
There is no setting to disable it.

- **Admission:** full-weight loads and adapters have distinct model identities.
  Stale identities fail before output; the replica can retry once.
- **Drain:** updates stop admission and wait up to ``inference.timeout-s``.
  Client disconnect does not release a pin until engine generation ends.
  Timeout interrupts remaining requests and requires recovery.
- **Prefix sharing:** when enabled, the component clears engine prefix caches
  after draining and before changing weights.
- **Uncertainty:** ambiguous loads, pauses or cache clears stop admission
  until replacement and verified reload. Explicit engine refusal retains the old binding.
- **Ownership:** newer owners fence older control calls with HTTP 412.

Built-in protection covers standard SGLang and vLLM handlers.
Other engines need a qualified custom runtime.
Keep native engine addresses private; all generation and updates pass through
the component.

External KV stores and prefix sharing with SGLang disaggregation are unsupported.
See `serving restrictions <../reference/configuration-inference.rst>`__.
Reef trusts engine acknowledgements; it does not inspect weights or caches.
Training rollout engines use separate synchronization.

Integration qualification
~~~~~~~~~~~~~~~~~~~~~~~~~

Use two small artifacts with distinguishable generated outputs.
Check actual content, response identity and stored identity together.
CPU fake-engine tests cannot establish real weight identity.

Test promotion and takeover with delayed requests, unavailable bindings,
interrupted streams, wrong load IDs and partial loads.
Check adapter and prefix-cache identity, and refuse mismatched adoption.
Record integration versions, uncovered cases and performance under concurrency.

**Cutover check.** Provide a context-manager factory yielding
``InferenceQualification`` from ``tests.support.inference_qualification``.
Its ``OutputOracle`` must identify artifacts from generated content alone.
The factory owns isolated resources:

.. code:: bash

   python -m tests.smoke.inference_cutover --target my_qualification:target \
     --output /scratch/cutover

The output must be an absolute, new or empty directory outside the checkout.
It receives records and ``report.json``. Exit status 0 means every reported
check passed; the report also lists uncovered cases.

**Native check.** ``python -m tests.smoke.native_inference --help`` describes
one-GPU checks. Its ``lifecycle`` profile includes control and multiple artifacts.
Neither command qualifies managed-router lifecycle or images; neither runs in normal CI.

Engine supervision
------------------

``InferenceControl`` pauses, resumes and recovers an engine group.
Integrations implement ``InferenceEngineGroup`` and per-engine
``InferenceEngine`` operations.
Pause intent survives partial barrier failures; replacement engines remain
paused. Failed recovery never resumes generation automatically.

Independent SGLang backend
--------------------------

Install ``reef-infra[sglang]`` plus native SGLang in the selected GPU environment.
The extra does not install Slime or Megatron.
The integration uses SGLang's native server and file-based weight/adapter loads;
its plugin name is ``reef``.
CPU tests cover launch contracts, parsing and recovery.
GPU execution and supported backend combinations require acceptance testing.
