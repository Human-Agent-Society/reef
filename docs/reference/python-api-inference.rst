Inference API
=============

The contracts of the inference service: the runtime that loads and names published weights, the handler that answers a request, the backend that controls engines, and the two built-in runtimes for saved weights.

`Serving runtimes <../developer-guide/serving-runtimes.rst>`__ gives the production inference contract, request pinning and the qualification of an integration.

Backend and runtime contracts
-----------------------------

``TrainingComponent`` and ``InferenceRuntime`` are independent interfaces. Neither inherits from the other, neither holds the other, and no trainer holds a serving engine. A recipe holds neither: the process holds both beside the recipe, in ``LiveObjects`` (`Live objects <python-api-recipe.rst#live-objects>`__).

.. code:: python

   from reef.inference.interfaces import InferenceBackend, InferenceHandler, InferenceRuntime

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Contract
     - Role
   * - ``InferenceRuntime``
     - Executes requests, owns admission, and loads published snapshots into its engines (`Runtime responsibilities`_).
   * - ``InferenceHandler``
     - Executes one buffered or streaming request (`Inference handlers`_).
   * - ``InferenceBackend``
     - The native integration under a runtime. It controls the serving engines of one deployment: loads ``WeightFiles`` into every engine (``load_weight_files``), pauses and resumes generation, unloads adapters, and reports the runtime load IDs that the engines serve. ``SGLangInferenceBackend`` (``reef.inference.sglang.backend``) is the built-in one.
   * - ``InferenceEngine``
     - The operations of one engine actor. ``ReefSGLangEngine`` and ``ReefVLLMEngine`` implement it.
   * - ``InferenceEngineGroup``
     - The engine supervision hook of an inference integration, in ``reef/inference/recovery.py``.
   * - ``TrainingComponent``
     - Runs bounded training jobs with its own framework, such as Slime or Tinker (`Training components <python-api-training.rst#training-components>`__).

The deployment selects native backends with ``training.backend`` and ``inference.backend``.

Inference handlers
~~~~~~~~~~~~~~~~~~

``InferenceHandler.inference(artifact, path, payload)`` answers one request. ``artifact`` is an ``InferenceArtifact``: the component ``artifact_id``, an optional ``local_path``, and ``served_model``, the model name of the selected route or ``None``. Native handlers send that name to the component, which pins the request to it. The name alone does not show which weights an engine holds. Handlers receive no repository or snapshot locator.

A handler reaches Reef in one of two ways:

- **Configured.** ``inference.handler-factory`` names an ``InferenceHandler`` subclass as ``package.module:ClassName``, and ``inference.handler-config`` holds its config. Reef calls the class method ``from_config(upstream_url, *, model_path, timeout_s, **config)``. Arbitrary functions are not accepted.
- **Injected.** ``create_app`` and ``build_app`` accept ``inference_handler`` (`Embed the service <python-api-recipe.rst#embed-the-service>`__). Such a handler needs only the inference methods.

Runtime and recipe objects expose the handler as ``inference_handler``. Reef's engines serve published weights only with the engine's built-in handler (`Request pinning <../developer-guide/serving-runtimes.rst#request-pinning>`__).

**Streaming.** By default, ``inference_stream`` wraps the buffered ``inference`` answer as one JSON chunk. Override it for incremental streaming.

- A streaming handler returns an ``InferenceStream``. Its ``record_response`` and ``record_response_pending`` fields carry buffered or deferred recording state. Initialize them through the stream constructor.
- A stream's ``chunks`` must end only after the handler has read the upstream to its end and finalized its capture. A terminal frame alone is not a successful completion, and it does not identify the weights that answered. `Request <http-api-inference.rst#request>`__ describes the receipts that clients get.
- An SGLang handler subclass that preserves raw reasoning tags sets ``SPLIT_REASONING = False``.

**Serving config.** A scenario's `serving config <http-api-scenarios.rst#serving-config>`__ reaches three methods. Each has a default for a handler or runtime without serving fields:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Method
     - Contract
   * - ``InferenceHandler.serving_config(override)``, ``InferenceRuntime.serving_config(override)``
     - Return the effective value of each serving field under one scenario's ``override``. Raise ``ValueError`` for a field or value that they do not take. Control calls the runtime's method to check an update; its default gives the handler's fields.
   * - ``InferenceHandler.request_payload(payload, *, serving_config)``
     - Gets the scenario's override with each request, applies its own fields, and leaves the rest.
   * - ``InferenceRuntime.apply_serving_config(scenario, override)``
     - Applies the engine fields. The inference worker calls it before each activation, before each ``serve_artifact(artifact, scenario=...)``, and when only the override changed. The worker then reads the scenario's route again with ``served_artifact``, so the runtime can move the scenario to another engine.

A runtime that serves several models declares a serving field that chooses the model of each artifact (`Serving settings <http-api-serving.rst#serving-settings>`__). It reads an artifact's training starts with ``reef.artifact.artifact.training_starts_of``, nearest first, so that a training artifact can use the setting of its start.

**Startup status.** These optional members feed the scenario's startup status:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Member
     - Contract
   * - ``InferenceHandler.engine_state(served)``
     - For engines that scale to zero: ``"warm"``, ``"starting"``, ``"idle"`` or ``None``. The status of a ready artifact returns it as ``engine``. It must not start an engine.
   * - ``InferenceHandler.engine_detail(served)``
     - While an engine is ``"starting"``, one short line for a person. The status returns it as ``engine_detail``.
   * - ``InferenceHandler.engine_steps(served)``
     - The ``start`` and ``load`` phases of a starting engine as ``reef.inference.interfaces.StartupSteps``.
   * - ``reef.inference.startup.report_startup(phase, detail=None, *, estimate_seconds=None)``
     - A runtime calls it while it serves an artifact, when the phase or its detail changes: ``prepare`` (deploy, convert or quantize), ``start`` (a machine and the weights) or ``load`` (GPU memory, kernels, warm-up). The status gives the reports as ``startup``. Outside a serving work, the call does nothing.

**Ownership.** The serving worker calls ``InferenceRuntime.set_owner_sequence(sequence)`` before it opens its runtime, with the stored sequence number of its ownership. The default does nothing. The inference component then refuses the control calls of an earlier owner, whose runtime raises ``InferenceOwnerLostError``. The worker treats that error as a lost lease, not as a failed artifact selection. Without a sequence, a ``remote_serving`` runtime starts no engine.

Runtime responsibilities
------------------------

``ScenarioCommitter`` publishes checkpoints and records artifact selections, and a dispatcher queues each selection for the inference worker. The worker opens the runtime, loads the published snapshot, and publishes routes when activation returns. Route publication and startup completion share one transaction, and a superseded request cannot publish routes.

An ``InferenceRuntime`` implements ``inference_handler`` and the two members marked *required*. It overrides another member only when it has the capability. The `production inference contract <../developer-guide/serving-runtimes.rst#production-inference-contract>`__ gives the requirements of an integration.

These members bind weights and name routes:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Member
     - Default
     - Result and errors
   * - ``activate_checkpoint(artifact, *, scenario=None)``
     - required
     - Serves the weight component of a published snapshot. Returns its runtime load ID (``str``). Gets ``scenario`` only when ``per_scenario_adapters`` is true. Raises on failure, and promises no rollback.
   * - ``served_artifact(artifact_id, scenario)``
     - required
     - Returns ``ServedArtifact | None``: the engine's HTTP(S) URL and model name of a weight binding that already holds. Loads nothing. ``None`` for an unknown artifact, an unavailable or uncertain binding, or an engine without such a URL; the startup then fails.
   * - ``endpoint_model_name(scenario)``
     - ``None``
     - Returns the model name (``str``) for the routes of a snapshot without a weight component, and of each artifact served beside it. These routes identify no weights. ``None`` refuses such a snapshot: its startup fails.
   * - ``handler_for(url)``
     - ``inference_handler`` for ``base_url``, else ``None``
     - Returns the handler of the engine at ``url``, for a service that did not open the runtime. After ``None``, that service uses a plain proxy.
   * - ``open()``, ``shutdown()``
     - do nothing
     - ``open`` starts the engines before the inference worker adopts or activates artifacts; no other process opens the runtime. ``shutdown`` releases what the runtime owns.
   * - ``adopt(served)``
     - ``False``
     - Returns whether the engine still serves ``served``, a route from before the inference worker restarted. After ``True``, the route stays, and the next activation of that artifact loads nothing. ``False`` removes the route until serving resumes.
   * - ``stop_serving(served)``
     - does nothing
     - Releases the binding of ``served``, which no route names any more. The engines can stay allocated. An unknown value is no error.
   * - ``release_unreferenced_artifacts(retained)``
     - does nothing
     - Releases each binding whose URL and model name no value of ``retained`` has, such as one that a failed startup left. ``retained`` holds every published route and every pending stop. The inference worker calls it between two serving works, only when ``serving_slots`` is 1. Raises on a failed release; the next call tries again.

These members state what the runtime serves:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Member
     - Default
     - Result and errors
   * - ``model_path``
     - ``""``
     - The default model name for harness bindings. Empty for a runtime whose requests select the model.
   * - ``served_weight_kind``
     - ``model``
     - The kind of the weights that the runtime serves: ``model`` for full weights, or ``model_adapter`` for a LoRA adapter on its base model. A runtime that serves adapters must override it.
   * - ``per_scenario_adapters``
     - ``False``
     - True for a runtime that serves each scenario its own adapter.
   * - ``shared_by_training_scenarios``
     - ``False``
     - True for a runtime that serves each training scenario its own weights.
   * - ``serves_any_artifact``
     - ``False``
     - True for a runtime whose handler answers each call from the artifact that the call names, without a load, as the Tinker runtime does. Such a runtime overrides ``artifact_model`` and ``serve_artifact``.
   * - ``artifact_model(artifact)``
     - raises ``ReefError``
     - Returns the ``PolicyModel`` that samples an artifact on request, without a load, for the `artifact evaluation <http-api-evaluation.rst#http-api-evaluate-an-artifact>`__ of a model artifact. ``ValueError`` for an artifact that the runtime cannot serve.
   * - ``serve_artifact(artifact, *, scenario)``
     - raises ``ReefError``
     - Deploys an artifact beside the scenario's active one. Returns ``ServedArtifact | None`` when the engine is ready: ``None`` when no other process can call the engine. Consumes one artifact slot until ``stop_serving``. ``ReefError`` when every slot holds another artifact, ``ValueError`` for an artifact that the runtime cannot serve.
   * - ``served_artifact_capacity``
     - ``None`` when ``serves_any_artifact`` is true, else 1
     - The artifact slots: how many weight artifacts of one scenario the runtime keeps bound at once. A positive integer, or ``None`` for no limit. A capacity above 1 needs ``serve_artifact``. It does not enable the evaluation of a model artifact.

No component changes an inference request: the inference routes forward each request to the engine that ``served_models`` names.

A scenario reads what serves:

- ``Scenario.serving_artifact_id()`` names the selected artifact.
- ``Scenario.served_artifact_id()`` reads the published route when a dispatcher owns the scenario.
- ``Scenario.served_artifacts()`` reads the artifacts that the scenario served from the commit history. The samples service judges samples against them.

Automatic weight training waits while the selected artifact has no matching served route. Manual training does not wait, since a request names its samples. Startup status is available through the inference service's ``GET /reef/scenarios/{scenario}/inference/status`` route.

Saved-weight runtimes
~~~~~~~~~~~~~~~~~~~~~

``reef.inference.serving`` holds the two runtimes for saved weights, both over an ``InferenceBackend``: ``AdapterServingRuntime``, whose snapshots hold the adapter in ``adapter/``, and ``FullWeightServingRuntime``. `Serving saved weights <../developer-guide/serving-runtimes.rst#serving-saved-weights>`__ describes what each one serves and how it recovers its engines.

- **Engines.** Both runtimes replace weights in engines that keep running. They take the engines as a ``ServingEngines`` value, and its ``request_pinning`` property must return true. The default is false, and the runtimes then raise ``RuntimeContractError`` when they are built. Return true only for a receiver that enforces the rules in the property's description (`Request pinning <../developer-guide/serving-runtimes.rst#request-pinning>`__).
- **Loading.** Each runtime passes a snapshot's directory as a ``WeightFiles`` value, with the ``"adapter"`` or ``"full"`` layout, to ``InferenceBackend.load_weight_files``, under a new runtime load ID that every engine then reports.
- **Health.** Each runtime checks the health of its engines every ``health_interval_s`` seconds (default 30), and calls ``recover_engines`` after a failed check. Before its first successful activation, a ``FullWeightServingRuntime`` has nothing to restore: a failed check only replaces the engines, and a call of ``recover_engines`` replaces them and then raises ``ReefError``.

Each runtime answers a `base root <python-api-artifacts.rst#base-roots>`__ (``base.json``) with the base model and asks ``ArtifactWeights`` for nothing:

- ``AdapterServingRuntime`` loads nothing. The scenario serves no adapter, and ``activate_checkpoint`` returns ``base_version``. The base root's route names the base model under that load ID, which the engines' receiver admits. No route names the artifact of the adapter that the scenario served before. That adapter stays resident until its route stops or another adapter needs its slot.
- ``FullWeightServingRuntime`` loads the directory ``model_path`` as full weights under a new version. It raises ``ReefError`` when ``model_path`` is no directory on its host.
- ``TinkerInferenceRuntime`` samples the base model. It raises ``ValueError`` for a base root of another model or another service.

A root snapshot that an earlier Reef made from the default start has the layout of every other snapshot of its component. An adapter runtime loads its untrained adapter, and the Tinker runtime samples its untrained adapter.

Factories and modules
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Name
     - Role
   * - ``RuntimeFactory``, ``RuntimeRegistry`` (``reef.inference.registry``)
     - Define and build the ``InferenceRuntime``.
   * - ``reef.service.deploy.serving.serving_runtime_config``
     - Describes the serving runtime from the ``inference`` settings and the backend's ``artifact_format``.

The inference contracts are the modules at the root of ``reef.inference``. Values that both training and inference read are in ``reef.core.weight_layout`` (``ArtifactFormat``, ``WeightFiles``, ``ArtifactWeights``). See the `codebase map <../contributing/codebase-structure.rst>`__. `Training <python-api-training.rst#factories-and-modules>`__ lists the training component factories and the rule for two training scenarios in one deployment.
