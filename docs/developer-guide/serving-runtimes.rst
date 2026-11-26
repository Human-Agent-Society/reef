Serving runtimes
================

An inference runtime (``reef.inference.interfaces.InferenceRuntime``) admits requests and serves the weights that Reef published. It never trains.

:For: developers who add or change an inference runtime or engine integration
:Before you start: `Codebase structure <../contributing/codebase-structure.rst#domain-layer>`__
:You will have: the production inference contract, the built-in runtimes, request pinning, and the qualification cases

Production inference contract
-----------------------------

A runtime that serves weights must keep each admitted request on the artifact that the request selected:

- A request admitted for artifact A uses A's model and adapter until its generation ends.
- Its response and its stored record identify A.
- The receiver refuses a request for an unavailable or uncertain binding before generation. No request runs on weights that it did not select.

The contract applies to local engines, to hosted providers, and to a gateway in front of several engine replicas. Training and the trainer's own rollout engines have separate contracts.

A published route does not meet the contract by itself: it describes the binding when the runtime made it, and reserves that binding for no request. An echoed artifact header, a health check, or a load-ID check before the request does not meet it either. The serving path needs one of these:

- **An immutable revision** behind the route, such as a provider deployment or a Tinker sampler revision. Nothing replaces its weights, so a hosted provider needs no native GPU controls.
- **A receiver that pins each request**, for engines whose weights change while they run. In one atomic step, the receiver admits the request under the weights that it selected and holds those weights against loading, unloading and replacement. Generation and prefix-cache reuse keep that binding until the last token. Reef's saved-weight runtimes always use such a receiver (`Request pinning`_).

What a runtime implements
~~~~~~~~~~~~~~~~~~~~~~~~~

Every ``InferenceRuntime`` implements three members:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Required member
     - What it does
   * - ``inference_handler``
     - Returns the handler that executes a request. An HTTP integration can reuse Reef's HTTP handler.
   * - ``activate_checkpoint``
     - Establishes the binding before any route exposes it, and returns its runtime load ID. It raises on failure.
   * - ``served_artifact``
     - Reports the route of a binding that already holds: a verified load, or an immutable revision. It loads nothing. It returns ``None`` for an unknown artifact and for an unavailable or uncertain binding; Reef then publishes no route.

Every other member has a default. Override one only for a capability that the runtime has:

- Engines of its own: ``open``, ``shutdown``, ``set_owner_sequence``.
- Takeover of a retained endpoint: ``adopt``, and ``handler_for`` for an endpoint at a URL other than ``base_url``.
- Several artifacts at the same time: ``serve_artifact``, ``served_artifact_capacity``, ``stop_serving``, ``release_unreferenced_artifacts``, ``handler_for``.
- Any artifact on request: ``serves_any_artifact``, ``artifact_model``.
- Snapshots without weights: ``endpoint_model_name``.
- Adapters: ``served_weight_kind``, ``per_scenario_adapters``, ``shared_by_training_scenarios``.

The docstring of each member in ``reef.inference.interfaces`` gives its default, its result and its errors.

A handler in front of several engine replicas can read ``REQUEST_SESSION_KEY`` (``reef.inference.interfaces``) to send the calls of one conversation to one replica. An evaluation sets it to each rollout, so rollouts whose first messages are the same still go to different replicas.

Automatic weight training waits while the selected artifact has no served route. Manual training does not wait, because a request names its samples.

When a binding is in doubt, the runtime must fail safely:

- After an ambiguous load, the affected engines stay unavailable until a known binding is restored and checked.
- On takeover, the new owner removes each route at a URL that ``handler_for`` gives no handler for, before ``open``. After ``open``, ``adopt(served)`` gets each other route, and returns false when the retained endpoint cannot establish its binding.
- Retirement lets admitted requests finish on their pinned weights, or it interrupts them explicitly.
- A stream that has sent output never retries on another artifact. An incomplete stream can keep a diagnostic record, but it gets no receipt.

`Integration qualification`_ lists the cases to test before an engine or provider serves weights for production requests.

Routes that bind no weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two modes bind no weights and claim no weight identity. The contract and request pinning do not apply to them:

- **A snapshot without a weight component**, such as a record-only or harness snapshot, gets no activation. Its route names the model that ``endpoint_model_name(scenario)`` returns, under the snapshot's own artifact ID. That ID identifies the snapshot, not weights: Reef does not check what the endpoint serves under the name. The built-in ``inference_proxy`` runtime returns its configured model, and it refuses weight snapshots. A scenario with a weight component never uses this mode.
- **A routed target** of the Semantic Router (``reef.routing_manifest``) is not a Reef artifact: its requests and records name no artifact (`Routed external-model requests <../reference/http-api-routing.rst#routed-external-model-requests>`__). A target must expose an OpenAI chat endpoint, which the component's native receiver (``reef.inference.remote``) is not.

Serving saved weights
---------------------

Serving loads only the weights that a job saved and Reef published. It loads them when a scenario starts, when a committed snapshot is activated, and when a person activates a snapshot. No trainer holds a serving engine.

Three runtimes serve saved weights on inference components:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Runtime
     - What it serves
   * - ``AdapterServingRuntime`` (``reef.inference.serving``)
     - Each scenario's own PEFT adapter on one base model. Requests name the adapter per scenario, and a scenario that serves a base root names the base model. At most ``adapter_capacity`` adapters are resident. ``AdapterResidencyManager`` evicts a leaked adapter first, then the oldest-activated adapter that is no scenario's current one.
   * - ``FullWeightServingRuntime`` (``reef.inference.serving``)
     - One scenario's full Hugging Face weights. The engines start with the base model and stay paused until the first activation succeeds. Each activation loads the artifact's files from the artifact repository. A base root loads the base model's directory, the runtime's ``model_path``.
   * - ``CheckpointSlotsRuntime`` (``reef.inference.checkpoint_slots``)
     - Several full-weight checkpoints of one training scenario at the same time. It holds one ``FullWeightServingRuntime`` for each inference component that ``inference.remote`` names (``url`` and ``slot-urls``), so each checkpoint has its own engines, load IDs, health checks and recovery. It never replaces a checkpoint that still serves: when every component holds one, it refuses a further checkpoint. A checkpoint that stops serving releases its component, which keeps its engines. The runtime evaluates no artifact that is not deployed.

Each load gets a new runtime load ID, and every engine reports it afterwards. The IDs of one runtime belong to a fresh incarnation, so a restart never reuses an ID for other weights. Loading content that the engines already serve does nothing.

**Selection.** The ``inference`` section selects the serving, not the training backend (``reef.api.serving``). With ``inference.serving: engines``, ``reef serve`` starts a local inference component (``python -m reef.inference.remote``) as the ``inference`` service, unless ``inference.remote`` names a component that runs elsewhere. Reef reaches the component through ``RemoteInferenceBackend`` and the runtime kind ``remote_serving``. The training component's ``artifact_format`` chooses the runtime (``adapter`` or ``full``) and the engines' adapter settings. The local component joins the deployment's Ray cluster, so its engines and the training workers reserve separate GPUs.

**Health and recovery.** The serving runtime checks the component's health every 30 seconds. The component names its process in each health report, so the runtime sees a component that restarted and runs no engines.

- After a failed check, the runtime pauses generation. When an engine is lost or dead, or the component reports an uncertain binding, the runtime replaces the whole engine stack, loads what it served again, and resumes.
- Generation resumes only after the reload and its load-ID check succeed. Until then, the runtime reports no route. A route that is already published stays published, and its requests wait or fail. The runtime retries the restore at each health check.
- A ``FullWeightServingRuntime`` without a verified checkpoint has nothing to restore: before its first successful activation, and after it released its checkpoint. A failed health check then only replaces the engines, which stay paused until the next activation.
- An ``AdapterServingRuntime`` keeps every scenario's previous adapter as the restore target until the whole restore succeeds. A capacity refusal changes no adapter, and the current adapters keep serving.

**Takeover.** A new serving owner always starts a new engine stack, because the old engines' load IDs cannot prove that the weights are unchanged. The owner changes when the owning inference worker restarts or loses its lease. The new stack ends the requests in flight, and the worker activates each completed selection again, so a takeover needs no operator action. `Startup status <../reference/http-api-inference.rst#startup-status>`__ gives the retries of a failed startup.

Request pinning
~~~~~~~~~~~~~~~

The inference component, local or ``inference.remote``, pins every production request for the saved-weight runtimes. No setting turns this protection off: the runtimes refuse engines whose receiver does not declare it (``ServingEngines.request_pinning``). The trainer's rollout engines do not use it, because the trainer synchronizes its own rollouts with its weight updates.

The component gives these guarantees:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Guarantee
     - Behavior
   * - Admission
     - Each full-weight load serves under its own model name, which the published route names. An adapter keeps its immutable adapter name. On an adapter stack, the base model serves under the load ID that the owner gave the new stack, for the route of a base root. The component refuses a request for a load that it no longer serves, before any output, and the replica retries once (`Request <../reference/http-api-inference.rst#request>`__).
   * - Drain
     - Before it changes weights or adapters, the component stops admitting requests and waits for the admitted ones to finish, for at most ``inference.timeout-s``. A request whose client disconnected keeps its pin until the engine finishes it. A drain that times out interrupts the remaining requests, and the engines then need recovery.
   * - Prefix sharing
     - With ``inference.prefix-sharing: true``, requests on the same weights share the engine's prefix cache. After the drain, the component clears the cache before each change, so no cache entry outlives its weights.
   * - Uncertain binding
     - A failed load, an ambiguous pause or a failed cache clear stops admission. Admission resumes after the component replaces its engine stack, and the intended artifacts are loaded and checked again. An explicit refusal from the engine leaves the binding in place.
   * - Ownership
     - A newer serving owner fences the older one: the component refuses the older owner's control calls with HTTP 412. The new owner's engine stack ends the requests in flight.

The protection has these limits:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Limit
     - Scope
   * - Engines
     - It covers the built-in SGLang and vLLM engines with their built-in request handlers. Engine serving refuses another ``inference.backend`` or ``inference.handler-factory``. Another engine needs its own serving runtime (``inference.serving: module:Factory``) that meets the `Production inference contract`_.
   * - Topology
     - The component starts its own engines and router. Generation and weight changes must go through the component, so the native engine addresses stay private. The component forwards only the engine's generation route, and other inference paths return HTTP 404.
   * - Caches
     - The component clears only the engine's own prefix cache. It cannot clear KV that an engine keeps outside itself, so engine serving refuses the engine options that select such a store. `Serving of a weight-training deployment <../reference/configuration-inference.rst#serving-of-a-weight-training-deployment>`__ lists them, and what the check does not cover. Prefix sharing needs regular SGLang engines: prefill/decode-disaggregated engines cannot use the clearing pause.
   * - Trust
     - The component trusts the engine's acknowledgements of loads and cache clears. It does not inspect the weights or the cache in engine memory, so qualify the engine version that you deploy (`Integration qualification`_).

`Run the services in separate processes <../user-guide/split-services.rst#inference>`__ gives the operating rules for the component.

Integration qualification
~~~~~~~~~~~~~~~~~~~~~~~~~

CPU tests with fake engines cannot verify which weights a real engine serves. Before an engine or provider serves production requests, check these cases on the version that you deploy. Cases 1 to 3 apply to every integration, and the others under their condition:

#. Delay a request or an activation across a promotion. Check the actual output, the response identity and the stored identity. Include the interval before the new route is published, and a late operation from a former owner.
#. Make a binding unavailable; for an immutable provider, select a revision that it does not serve. The request must fail, and must not run on another artifact.
#. Interrupt a stream after its first token. It must get no receipt and no retry on another artifact.
#. If its engines report load IDs, make them report mixed IDs after a load, and uniformly wrong ones. Neither load may give a route for the requested artifact.
#. If it changes the weights of running engines, fail one replica while it loads, then recover. Requests must be refused while its weights are uncertain, and succeed after a checked reload.
#. If it adopts retained routes, present a missing or mismatched binding. Adoption must be refused.
#. If it supports adapters or prefix-cache reuse, check that each preserves the selected artifact.

Use two small artifacts whose outputs differ, and identify each answer from its generated content, not from labels or version metadata. Record the engine version and the capabilities checked. These cases verify behavior. Use a representative concurrent workload to measure throughput, 95th-percentile latency, drain time and unavailable time.

Two commands run a real integration through the real inference HTTP app and record store. They are not pytest tests, so ``pytest tests/`` and CI do not run them.

**Cutover check.** ``tests.support.inference_qualification`` runs your integration's two artifacts through a promotion. It keeps A's route published after it activates B. Each answer must then come from A and agree with its artifact header and stored record, or fail without a receipt. Write a factory that takes the output directory and returns a context manager. The context manager owns the run's isolated resources and yields an ``InferenceQualification``, whose ``OutputOracle`` names the artifact from the generated content alone. Run the command from the checkout, with the factory as ``--target`` (``module:factory``):

.. code-block:: bash

   python -m tests.smoke.inference_cutover --target my_qualification:target \
     --output /scratch/cutover

``--output`` must be an absolute path to a new or empty directory outside the checkout. The command stores the records and ``report.json`` there, and exits with status 0 only when every check passes. The report lists the checked cases and the cases that the run does not cover. The CPU suite runs the same check against toy SGLang and vLLM components that pin each request.

**Native check.** ``tests/smoke/native_inference.py`` checks Reef's own SGLang and vLLM integrations on one GPU, with two synthetic checkpoints. Its module docstring describes the profiles; run ``python -m tests.smoke.native_inference --help`` for options. Only its ``lifecycle`` profile covers control-plane publication and a public deployment of several artifacts. Neither command covers a managed router lifecycle, or validation of the router's deployment image.

Engine supervision
------------------

``reef.inference.recovery.InferenceControl`` pauses, resumes and recovers the engines of one inference service. The engine integration implements its hook, ``InferenceEngineGroup``, which pauses, resumes, replaces dead engines and retires engines. Each engine actor implements ``reef.inference.interfaces.InferenceEngine``, the per-engine vocabulary that the group and the backend call by name.

- ``InferenceControl`` records the pause intent before the engine barrier, so a partial pause failure still fences replacement engines.
- Recovery replaces dead engines, and pauses them again when generation was paused.
- A failed recovery keeps generation paused. Only ``resume`` lets the engines serve again.

Independent SGLang backend
--------------------------

``reef/inference/sglang/`` owns the chat backend, the SGLang plugin, the native engine process, the router, the Ray engine groups and the inference lifecycle. Native ``ServerArgs`` and ``launch_server`` come directly from SGLang, not from Slime.

Install ``reef-infra[sglang]`` for the Python-side inference dependencies, and install native SGLang in the selected GPU environment. This extra does not install Slime or Megatron.

Engines load published weights from files. The engine control worker's ``load_weights_from_disk`` reloads full weights through SGLang's ``update_weights_from_disk``. Its ``load_adapter_from_disk`` loads an adapter through ``load_lora_adapter_from_disk``. The plugin name is ``reef``.

CPU tests cover native launch bindings, shared placement, response parsing, LoRA requirements and recovery. Actual GPU execution and backend combinations still require acceptance testing.
