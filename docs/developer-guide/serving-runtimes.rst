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

Optional capabilities include resource ownership (``open``, ``shutdown``,
``set_owner_sequence``), retained-endpoint adoption (``adopt``), multiple
artifacts, adapters and arbitrary-artifact serving.
Read each member's default and constraints in ``InferenceRuntime``.

Handlers can use ``REQUEST_SESSION_KEY`` to keep a conversation on one replica.
Evaluations assign one key per rollout, including rollouts with identical prompts.
Automatic weight training waits for the selected artifact's served route;
manual requests do not.

Keep uncertain bindings unavailable until a checked restore succeeds.
``adopt`` must return false when it cannot establish a retained binding.
Retirement must finish pinned requests or interrupt them explicitly.
Never retry a stream on another artifact after output starts.
Incomplete streams receive no receipt, though diagnostic records may remain.

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

Each load has a runtime load ID verified on every engine.
Fresh runtime incarnations cannot reuse IDs for different weights.
Loading already served content is a no-op.

``inference.serving: engines`` uses a local inference component unless
``inference.remote`` names an external one. The training component's artifact
format selects adapter or full-weight serving.
Serving engines reserve GPUs separately from training workers.

**Recovery.** Health checks run every 30 seconds.
Lost engines or uncertain bindings trigger engine-stack replacement and
reload. Resume only after the load-ID check succeeds.
Until then, existing published routes may wait or fail; new route queries
return none. Recovery retries on later health checks.

A full-weight runtime without a verified checkpoint has nothing to restore
and stays paused. Adapter recovery retains previous adapter targets until
the complete restore succeeds. Capacity refusal leaves current adapters unchanged.

**Takeover.** A new serving owner starts fresh engines and interrupts in-flight
requests. It reactivates completed selections automatically.
Old load IDs cannot prove that retained weights stayed unchanged.
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

Built-in protection covers SGLang and vLLM with their standard handlers.
Other engines need a custom runtime that meets the production contract.
Keep native engine addresses private: generation and updates must pass through
the component, which exposes only generation routes.

External KV stores cannot be cleared by this mechanism and are rejected.
Prefix sharing also excludes SGLang prefill/decode disaggregation.
See `Serving configuration
<../reference/configuration-inference.rst#serving-of-a-weight-training-deployment>`__
for option restrictions.

Reef trusts engine acknowledgements; it does not inspect GPU weights or caches.
Qualify each engine version. The trainer's own rollout engines use separate
synchronization and do not use production request pinning.

Integration qualification
~~~~~~~~~~~~~~~~~~~~~~~~~

Use two small artifacts with distinguishable generated outputs.
Check actual content, response identity and stored identity together.
CPU fake-engine tests cannot establish real weight identity.

Test these cases on each integration version:

1. Delay requests and activations across promotion, including before route
   publication and after owner takeover.
2. Make a binding unavailable. Requests must fail without selecting another artifact.
3. Interrupt a stream after its first token. It must receive no receipt or cross-artifact retry.
4. For load-ID engines, inject mixed or uniformly wrong IDs. Neither may produce a valid route.
5. For mutable engines, fail one replica during load. Refuse requests until verified recovery.
6. For adoption, present missing or mismatched bindings and require refusal.
7. For adapters and prefix caches, verify that both preserve request identity.

Record the version and tested capabilities. Separately measure throughput,
p95 latency, drain time and unavailability under representative concurrency.

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
one-GPU SGLang/vLLM checks with synthetic checkpoints.
Only its ``lifecycle`` profile covers control publication and multi-artifact
deployment. Neither command qualifies managed-router lifecycle or images.
These commands are outside normal pytest and CI runs.

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
