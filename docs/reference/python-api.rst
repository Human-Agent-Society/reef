Python API for custom learning methods
======================================

The Python API is the set of extension points a learning method plugs into. Reef
owns everything around them: accepting and replaying records, holding a job's data
until it is acknowledged, committing algorithm state, submitting training jobs,
and publishing the saved checkpoints.

`Write a recipe <../developer-guide/write-a-recipe.rst>`__ is the executable tutorial.

What to implement
-----------------

Start with Recipe and add only what the method actually needs.

.. list-table::
   :header-rows: 1
   :widths: 34 34 32

   * - Need
     - Component
     - Required for
   * - configure deployment behavior
     - `Recipe <#recipe>`__
     - every deployment
   * - validate feedback at ingress
     - `Report <#report>`__
     - methods whose signal arrives in reports
   * - turn records into typed job data
     - `Pipeline stages <#pipeline-stages>`__
     - every method producing updates
   * - carry data to the backend
     - `Job data <#job-data>`__
     - every method producing updates; subclass only for a new shape
   * - turn job data into a signal
     - `Training objective <#training-objective>`__
     - weight-training methods
   * - declare a new snapshot component
     - `Scenario components <#scenario-components>`__
     - only a new evolution medium
   * - evolve a harness tree and score a task run
     - `Harness method <#harness-method>`__
     - harness-evolution methods
   * - evaluate a produced candidate
     - `Candidate evaluation <#candidate-evaluation>`__
     - optional, any recipe
   * - a new tensor objective
     - `Loss family <../developer-guide/loss-families.rst>`__
     - rarely

Method code should depend only on what this page documents. Anything else under
``reef.`` is an implementation detail and may change.

Every extension contract is an abstract base class. Inherit it explicitly and
implement each abstract member; Reef rejects an object that only has methods
of the same names. Optional capabilities are separate base classes: inherit
only those that the implementation supports.

Backend and runtime contracts
-----------------------------

Training and inference are separate contracts. Neither holds the other, and
there is no aggregate runtime:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Layer
     - Training
     - Inference
   * - Reef contract
     - ``TrainingComponent``
     - ``InferenceRuntime``
   * - Native integration
     - the component's own framework (Slime, Tinker)
     - ``InferenceBackend``

.. code:: python

   from reef.inference.interfaces import InferenceBackend, InferenceHandler, InferenceRuntime
   from reef.train.component import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Contract
     - Role
   * - ``TrainingComponent``
     - Runs bounded training jobs (`Training jobs and components`_).
   * - ``InferenceRuntime``
     - Executes requests, owns admission, and loads published snapshots into
       its engines (`Runtime responsibilities`_).
   * - ``InferenceBackend``
     - Controls the serving engines of one deployment: loads ``WeightFiles``
       into every engine (``load_weight_files``), pauses and resumes
       generation, unloads adapters, and reports the runtime load IDs that
       the engines serve. ``SGLangInferenceBackend`` is the built-in one.
   * - ``InferenceEngine``
     - The operations of one engine actor. ``ReefSGLangEngine`` and
       ``ReefVLLMEngine`` implement it.
   * - ``InferenceEngineGroup``
     - The engine supervision hook of an inference integration, in
       ``reef/inference/recovery.py``.
   * - ``InferenceHandler``
     - Executes one buffered or streaming request (below).
   * - ``reef.train.jobs.JobLoop``
     - How a ``Trainer`` trains: ``Trainer.build`` takes it as ``job_loop``,
       and a trainer without one trains nothing (`Training jobs and
       components`_).

Inference handlers
~~~~~~~~~~~~~~~~~~

``InferenceHandler.inference(artifact, path, payload)`` answers one request.
``artifact`` is an ``InferenceArtifact``: the component ``artifact_id``, an
optional ``local_path``, and ``served_model``, the model name of the selected
route or None. Native handlers send that name to the component, which pins the
request to it; the name alone does not show which weights an engine holds.
Handlers receive no repository or snapshot locator.

A handler reaches Reef in one of two ways:

- **Configured.** ``inference.handler-factory`` names an ``InferenceHandler``
  subclass as ``package.module:ClassName``, and ``inference.handler-config``
  holds its config. Reef calls the class method ``from_config(upstream_url, *,
  model_path, timeout_s, **config)``. Arbitrary functions are not accepted.
- **Injected.** ``create_app`` and ``build_app`` accept ``inference_handler``
  (see `Embed the service`_). Such a handler needs only the inference methods.

Runtime and recipe objects expose the handler as ``inference_handler``. The
deployment selects native backends with ``training.backend`` and
``inference.backend``. Reef's engines serve published weights only with the
engine's built-in handler (see `Request pinning
<../developer-guide/serving-runtimes.rst#request-pinning>`__).

**Streaming.** By default, ``inference_stream`` wraps the buffered
``inference`` answer as one JSON chunk; override it for incremental streaming.
A streaming handler returns an ``InferenceStream``. Its ``record_response``
and ``record_response_pending`` fields carry buffered or deferred recording
state; initialize them through the stream constructor. A stream's ``chunks``
must end only after the handler has read the upstream to its end and finalized
its capture. A terminal frame alone is not a successful completion, and it
does not identify the weights that answered. `Request <http-api.rst#request>`__
describes the receipts that clients get. An SGLang handler subclass that
preserves raw reasoning tags sets ``SPLIT_REASONING = False``.

**Serving config.** A scenario's `serving config
<http-api.rst#serving-config>`__ reaches three methods. Each has a default for
a handler or runtime without serving fields:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Method
     - Contract
   * - ``InferenceHandler.serving_config(override)``,
       ``InferenceRuntime.serving_config(override)``
     - Return the effective value of each serving field under one scenario's
       ``override``. Raise ``ValueError`` for a field or value that they do
       not take. Control calls the runtime's method to check an update; its
       default gives the handler's fields.
   * - ``InferenceHandler.request_payload(payload, *, serving_config)``
     - Gets the scenario's override with each request, applies its own
       fields, and leaves the rest.
   * - ``InferenceRuntime.apply_serving_config(scenario, override)``
     - Applies the engine fields. The inference worker calls it before each
       activation, before each ``serve_artifact(artifact, scenario=...)``,
       and when only the override changed. The worker then reads the
       scenario's route again with ``served_artifact``, so the runtime may
       move the scenario to another engine.

A runtime that serves several models declares a serving field that chooses the
model of each artifact (`Serving settings <http-api.rst#serving-settings>`__).
It reads an artifact's training starts with
``reef.artifact.artifact.training_starts_of``, nearest first, so that a training
artifact can use the setting of its start.

**Startup status.** These optional members feed the scenario's startup status:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Member
     - Contract
   * - ``InferenceHandler.engine_state(served)``
     - For engines that scale to zero: ``"warm"``, ``"starting"``, ``"idle"``
       or None. The status of a ready artifact returns it as ``engine``. It
       must not start an engine.
   * - ``InferenceHandler.engine_detail(served)``
     - While an engine is ``"starting"``, one short line for a person; the
       status returns it as ``engine_detail``.
   * - ``InferenceHandler.engine_steps(served)``
     - The ``start`` and ``load`` phases of a starting engine as
       ``reef.inference.interfaces.StartupSteps``.
   * - ``reef.inference.startup.report_startup(phase, detail=None, *,
       estimate_seconds=None)``
     - A runtime calls it while it serves an artifact, when the phase or its
       detail changes: ``prepare`` (deploy, convert or quantize), ``start``
       (a machine and the weights) or ``load`` (GPU memory, kernels,
       warm-up). The status gives the reports as ``startup``. Outside a
       serving work, the call does nothing.

**Ownership.** The serving worker calls
``InferenceRuntime.set_owner_sequence(sequence)`` before it opens its runtime,
with the stored sequence number of its ownership. The inference component then
refuses the control calls of an earlier owner, whose runtime raises
``InferenceOwnerLostError``. The worker treats that error as a lost lease, not
as a failed artifact selection. The default does nothing. Without a sequence, a
``remote_serving`` runtime starts no engine.

Recipe
------

.. code:: python

   from reef.recipe import Recipe, WeightTrainingRecipe, config_field
   from reef.recipe.harness import HarnessEvolutionRecipe
   from reef.recipe.sft import SFTRecipe

A recipe is one frozen dataclass that configures the serving and evolution
behavior of every scenario in a deployment. Every member is a function of its
configuration: a recipe has no runtime, no training component and no store,
and it builds none.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Base class
     - Subclasses
   * - ``Recipe`` (``reef.recipe``)
     - record-only by default
   * - ``WeightTrainingRecipe`` (``reef.recipe``)
     - ``SFTRecipe``, ``DatasetRecipe`` and ``GeneratedTaskRecipe`` under
       ``reef.recipe``; ``SAORecipe``, ``TTTDRecipe``, ``SDFTRecipe``,
       ``SDPORecipe``, ``OPDRecipe`` and ``OpenClawRLRecipe`` under
       ``recipes``; ``CoralRecipe`` (beta)
   * - ``HarnessEvolutionRecipe`` (``reef.recipe.harness``)
     - ``ReefineRecipe`` (beta), which only sets its defaults
   * - ``CompositeRecipe`` (``reef.recipe.composite``)
     - none; it maps component names to child recipes

A harness method is a ``Method`` that ``HarnessEvolutionRecipe`` runs, not a
recipe: GEPA, Meta-Harness, ACE and ReasoningBank name the base recipe and set
``evolution.method``.

Choose the narrowest class whose assumptions all hold:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - What you are building
     - Where to start
   * - record traffic, no updates
     - ``Recipe`` as-is; its trainer only reads records
   * - train and publish weights
     - subclass ``WeightTrainingRecipe``
   * - use Reef's harness loop
     - configure ``HarnessEvolutionRecipe`` with a ``Method`` in
       ``evolution.method``; subclass it only when the method changes more
       than the tree, such as the files a client pulls
   * - evolve a different artifact
     - subclass ``WeightTrainingRecipe``, deploy it with a
       ``TrainingComponent`` that makes the artifact, and override
       ``build_components()``
   * - serve an externally produced artifact
     - subclass ``Recipe`` and override ``build_components()`` only
   * - evolve weights and a harness (or configuration) together
     - configure ``CompositeRecipe`` with one recipe per component; each
       trainer runs as its own worker

Common members
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Contract
   * - ``name``
     - ``str`` instance field; the default registry key.
   * - ``report_type``
     - The ``ReportBase`` subclass that the recipe's reports parse as.
       ``None`` keeps ingress open. A composite declares none.
   * - ``pipeline_spec()``
     - Class method. The default class of each pipeline stage, a
       ``PipelineSpec`` (`Pipeline stages`_).
   * - ``stages()``
     - ``PipelineSettings``: the two stage classes and their configs, the
       deployment's ``pipeline`` section over ``pipeline_spec()``.
   * - ``build_components(scenario)``
     - ``ScenarioComponents``: the components that the configuration declares
       for the scenario's snapshots, each with its kind and admission check
       (default ``AcceptAnyArtifact()``).
   * - ``default_start()``
     - ``DefaultStart``: the content of a new scenario's root snapshot that
       the configuration gives; empty by default.
   * - ``activation_rule()``
     - The rule that decides whether an artifact that the recipe's trainers
       make serves. The deployment's ``activation`` section names it
       (`Activation rule`_).

``WeightTrainingRecipe`` adds:

- ``default_parameters()``: class method. The recipe's own parameters of each
  training job by the kind of its samples, normally the ``objective``.
- ``parameters_by_kind()``: those parameters with the ``job_parameters``
  config over them (`Training objective`_).

``HarnessEvolutionRecipe`` adds:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Member
     - Value
   * - ``evolution``, ``evaluation``
     - the parsed ``evolution`` section (``EvolutionSettings``, in
       ``reef.train.harness_backend.settings``) and its ``evaluation``
       section (``EvaluationSettings``); they name the method, the scorer
       and the collector as plugin settings
   * - ``seed``, ``models``, ``model_metadata``, ``worker_executor``,
       ``model_name``
     - the first tree, the method's named models, the model metadata, the
       placement of the run workers and the served model's name
   * - ``harness_settings()``
     - the ``HarnessSettings`` that the harness training component builds
       itself from
   * - ``runs_offline()``, ``task_directories(descriptor)``,
       ``configured_tasks()``
     - functions of the configuration

.. code:: python

   @dataclass(frozen=True, kw_only=True)
   class MyMethodRecipe(WeightTrainingRecipe):
       name: str = "my_method"

       @property
       def report_type(self) -> type[MyMethodReport]:
           return MyMethodReport

       @classmethod
       def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
           return {kind: {"objective": "my_method"} for kind in SAMPLE_KINDS}

       @classmethod
       def pipeline_spec(cls) -> PipelineSpec:
           return PipelineSpec(record2sample=MyMethodSamples, sample2job=MyMethodJobData)

The base requires ``frozen=True``. ``kw_only=True`` keeps later fields
keyword-only. Call ``super().__post_init__()`` first when you add validation.

A ``CompositeRecipe`` computes nothing from its children. Its
``__post_init__`` checks the mapping: at least two entries, valid component
names, no nested composite, one ``training_mode``, at most one
``WeightTrainingRecipe`` and at most one ``HarnessEvolutionRecipe``.

Code that holds the recipe of a deployment reads its recipes with
``reef.recipe.composite.component_recipes(recipe)``: the pairs ``(component
name, child)`` of a composite, and the one pair ``(None, recipe)`` of any
other recipe.

Live objects
~~~~~~~~~~~~

.. code:: python

   from reef.scenario.live_objects import LiveObjects, ServedEndpoint

The process that runs a service builds the objects that run the recipe, and
holds them in one frozen ``LiveObjects`` value.
``reef.service.assembly.load_deployment(settings, env, runs_training=...)``
returns the recipe and the value, and ``Dispatcher(recipe, backend_factory,
live=live, ...)`` receives it. Only a process that runs the training service
gets a training component.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Value
   * - ``runtime``
     - the serving ``InferenceRuntime``, or ``None``
   * - ``training_facts``
     - the ``TrainingBackendFacts`` of the backend of the recipe that trains
       weights, or ``None``; a ``WeightTrainingRecipe`` needs the runtime and
       these facts (``check_recipe_objects`` raises ``RecipeConfigError``)
   * - ``training_component``
     - the ``TrainingComponent`` that runs the jobs of that recipe, in a
       process that runs the training service, or ``None``; a value with a
       component has the facts of the same backend
   * - ``harness_facts``
     - the ``HarnessFacts`` of the deployment's harness recipe, or ``None``
   * - ``harness_component``
     - the training component that runs the jobs of the harness recipe, in a
       process that runs the training service, or ``None``
   * - ``data_store``
     - the deployment's ``DataStore``, which keeps the data directory of each
       training job, or ``None``
   * - ``served_endpoint``
     - a ``ServedEndpoint(url, token)`` where this Reef answers inference
       itself, or ``None``

``dispatcher.live`` holds the value as the trainers see it: each component is
the jobs queue's view of the backend whose facts the process holds
(``reef.train.job_queue.JobQueueComponent``). ``dispatcher.job_components``
are the components that the process built, which is none outside the training
service. Read the serving runtime as ``dispatcher.live.runtime``.

A recipe builds no trainer. The scenario factory builds one trainer for each
component from the recipe and the live objects, with ``Trainer.build``.
Nothing replaces this path: a deployment changes its trainers through its
recipe and its live objects. A recipe that is not a composite has one trainer,
so its scenario serves at most one component.

Before Reef registers a scenario with an item start, the facts of each
recipe's backend check the root (``TrainingBackendFacts.check_root_start``).
A refusal raises ``ValueError`` and leaves the scenario unregistered.

The training service builds the harness training component with
``HarnessTrainingComponent.from_settings``. Code that already holds the
method, executor and scorer, such as an example driver, calls
``HarnessTrainingComponent(settings, models, method=..., executor=...,
scorer=..., collector=..., agent=...)`` and gives the dispatcher
``LiveObjects(runtime=runtime, harness_component=component)``.

Configuration
~~~~~~~~~~~~~

``config_field()`` is a ``dataclasses.field`` carrying a default, a type-aware
parser, and an optional environment fallback:

.. code:: python

   clip_ratio: float = config_field(0.2, env="REEF_MY_METHOD_CLIP_RATIO")

Precedence is explicit configuration, then environment, then the default.

A recipe class does not load itself and has no method that the loader calls.
``reef.recipe.loading.load_recipe(recipe_class, config, environ)`` reads the
declarations of the class and builds the recipe; ``build_recipe(reference,
environ, config)`` does the same for a class reference. The declarations are:

- ``config_field`` fields;
- ``section_field`` fields, one parsed config section each;
- ``reader_field`` fields, each read by its ``ConfigReader``;
- ``section_defaults``, the class's defaults for its sections, under the
  deployment's keys;
- ``refused_sections``, the config sections that the class refuses, each with
  its error message.

For a weight-training deployment,
``reef.service.deploy.training.training_recipe_config`` forwards the declared
fields from the service configuration and refuses a key that the class does
not declare. A setting that a pipeline stage reads belongs in the stage's
settings, under the deployment's ``pipeline`` section, not in a recipe field.
Never read deployment YAML from inside a stage.

Report
------

.. code:: python

   from reef.core.reports import ReportBase, ReportValidationError, ScoredRolloutReport, TeacherContextReport

A report type declares the feedback a method accepts, so malformed input fails
at ingress with HTTP 400. Two contracts are shared:

- ``ScoredRolloutReport``: score only.
- ``TeacherContextReport``: the contract of the distilling recipes. It holds a
  rollout's receipt and the ``teacher_context`` its teacher sees (a
  demonstration, environment feedback, or nothing for a separate teacher).

Method-specific report contracts live in their method package; Reef does not
import or re-export them.

.. code:: python

   @dataclass(frozen=True)
   class MyMethodReport(ReportBase):
       score: float
       task_id: str
       rubric: str = ""

       def validate(self) -> None:
           if not self.task_id.strip():
               raise ReportValidationError("metadata.task_id must be non-empty")

Declare it through the recipe's ``report_type``. ``score`` uses the top-level
``score`` channel; every other field uses ``metadata.<field>``. Producers may
attach extra fields the schema does not declare.

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - Annotation
     - Accepted JSON
     - Validation
   * - ``float``
     - number
     - must be finite; ints normalize
   * - ``int``
     - number
     - integral value; booleans rejected
   * - ``str``
     - string
     - no coercion
   * - ``bool``
     - boolean
     - no numeric substitutes
   * - ``Mapping[str, str]``
     - object
     - every key and value a string

Each supported type may also be written as ``T | None``. Any other annotation,
any other Union included, is a declaration error and raises ``TypeError`` when
Reef first inspects the type. A field without a default is required; a field
with one may be absent; JSON ``null`` is accepted only when the annotation
permits it.

The same type serves both sides: a producer constructs it and calls
``to_dict()``; a processor receives the parsed instance as
``context.parsed_report``.

Embed the service
-----------------

An application can run the complete Reef service in its own process.
``reef.service.assembly.build_app`` makes the dispatcher and the HTTP app from
a ``ServiceSettings``. The app includes record retention, the inference routes,
the evaluation token, readiness of local cycles, and cleanup.

.. code:: python

   from aiohttp import web

   from reef.service.app import get_dispatcher
   from reef.service.assembly import build_app

   app = build_app(settings, inference_handler=handler)

   async def scenario_count(request: web.Request) -> web.Response:
       dispatcher = get_dispatcher(request.app)
       return web.json_response({"scenarios": len(dispatcher.list_scenarios())})

   app.router.add_get("/my-app/scenario-count", scenario_count)
   web.run_app(app, host=settings.host, port=settings.port)

- ``inference_handler`` is optional. If you give one, it runs every
  inference request in place of the recipe runtime's handler.
- Add middleware and routes before the app starts. The service tokens also
  protect the routes that you add.
- ``get_dispatcher(app)`` returns the dispatcher of an app that ``build_app``
  or ``create_app`` made.
- ``build_app`` owns the dispatcher that it makes. The app closes it on
  cleanup. ``create_app(dispatcher, ...)`` closes the dispatcher only if you
  set ``close_dispatcher=True``. It never creates a default storage backend.
- You own the inference handler that you give. Reef does not close it.
  Release it after the app stops.
- If ``build_app`` fails, it closes the resources that it made and raises the
  original error.

Scenario stores
---------------

Three abstract bases hold a scenario's state. A storage implementation
subclasses them explicitly and overrides every abstract member.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Base
     - Module
     - Role
   * - ``RecordStore``
     - ``reef.storage.records``
     - Appends, reads and audits records (`Record storage and audit`_).
   * - ``ScenarioStore``
     - ``reef.storage.scenario``
     - One scenario's session: a ``RecordStore`` with the committed scenario
       state. It settles each step and its record progress together.
   * - ``ScenarioStorage``
     - ``reef.storage.scenario``
     - Opens the sessions and keeps all per-scenario state.

``reef.storage.scenario`` also exports ``ScenarioSettings`` and
``ScenarioStoreConflictError``. The bundled implementations are
``ItemRecordStore``, ``CommitHistoryScenarioStore``, and
``SQLiteScenarioStorage`` and ``PostgresScenarioStorage``.

.. code:: python

   from pathlib import Path

   from reef.dispatcher import Dispatcher
   from reef.storage.sql_coordination import memory_coordination
   from reef.storage.sqlite import SQLiteScenarioStorage

   storage = SQLiteScenarioStorage(Path(".reef/agent-record"))
   dispatcher = Dispatcher(
       recipe,
       repository_backend_factory,
       scenario_storage=storage,
       coordination=memory_coordination(),
   )

``Dispatcher`` and ``build_default_dispatcher`` require both arguments:

- ``scenario_storage``: the dispatcher passes it to scenario creation, closes
  it on shutdown, and never selects a storage implementation itself.
  Deployment assembly builds the storage that ``storage.record_backend``
  names: SQLite, PostgreSQL, or a ``ScenarioStorage`` plugin.
  ``SQLiteScenarioStorage()`` without a directory keeps records, commits and
  settings in memory. With a directory, each scenario has one SQLite database
  file for its records, commits, settings, tasks and uploaded trajectories.
- ``coordination``: the store of the services' work queues, served models and
  control lease. One process that runs every service and keeps no queue state
  across restarts passes ``memory_coordination()``. Deployment assembly opens
  the shared store that ``storage.coordination_backend`` names.

Direct ``Scenario`` construction requires ``store=...``, one session that a
``ScenarioStorage`` opened. The supplied trainer must use that store's
``records``. Read committed history through ``scenario.store.history()`` and
the persistence capability through ``scenario.store.durable``.

The storage contract is:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``from_config(config, *, directory, task_files, data=None)``
     - Class method that builds the storage a deployment names in
       ``storage.record_backend``. ``config`` is the plugin's own mapping;
       ``directory`` is ``storage.agent_record_dir``. ``task_files`` is the
       ``TaskFileStore`` of the deployment (see `Task file stores`_). The
       storage owns it: ``archive`` and ``close`` also apply to it. ``data``
       is the deployment's data store, which keeps each scenario's records
       as record items; the storage shares it and does not close it.
   * - ``durable``
     - Whether new sessions and settings survive a process restart.
   * - ``open(scenario)``
     - Open an independently owned session (the ``ScenarioStore`` below).
       Raise ``ScenarioDeletedError`` for a scenario that ``archive``
       archived, so a late record makes no new store of a deleted scenario.
   * - ``read_settings(scenario)``
     - The scenario's ``ScenarioSettings``: the selected training mode, the
       artifact evaluation, the activation rule and the owner. Defaults for
       a scenario with none.
   * - ``write_settings(scenario, settings)``
     - Replace the settings durably. The caller serializes writers of one scenario.
       Raise ``ScenarioDeletedError`` for a scenario that ``archive`` archived.
   * - ``sample_store(scenario, session)``
     - The data items that the scenario trains on, with the training state
       of its open ``session``, as a ``SampleStore``. The default returns
       ``None``: a storage without a data store has none, and its trainers
       keep their samples in memory.
   * - ``archive(scenario)``
     - Retire the scenario's records, commits, tasks and settings, and
       remove the scenario from the uses of its owner's items; return the
       archive locations. A repeated call changes nothing more. A scenario
       ID is not used again after its delete.
   * - ``close()``
     - Release shared resources; repeated calls are safe.

The session contract is:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``records``
     - A ``RecordStore`` implementation preserving append deduplication,
       ordered replay, scenario isolation, and audit reads.
   * - ``durable``
     - Whether committed history survives session/process restart. A durable
       store requires a repository backend supporting staged snapshots.
   * - ``tasks``
     - The ``TaskStorage`` that opens the task stores of the
       scenario (see `Task stores`_). The stores close with the
       session and survive restarts when ``durable``.
   * - ``training_state``
     - The ``TrainingState`` of the scenario: the train counts of its samples,
       the number of committed jobs of each trainer that trained each one, and
       the records that each trainer's stages consumed. It closes with the
       session and survives restarts when ``durable``.
   * - ``evaluations``
     - The ``EvaluationStore`` of the artifact evaluations that clients asked
       the scenario for. It closes with the session and survives restarts
       when ``durable``.
   * - ``use_snapshots(snapshots)``
     - Locate the snapshot of each record that ``history`` reads. The default
       does nothing, for a store that holds the snapshots itself.
   * - ``history()``
     - ``CommitRecord`` values in commit number order: training steps
       (including inactive artifacts), activations, and new training starts.
       This is authoritative after an ambiguous commit error.
   * - ``training_run_position()``
     - The commit number of the last ``train_from`` commit and the count of training
       commits after it, for experiment run numbering.
   * - ``append_commit(expected_commit_number=..., commit=...)``
     - Persist the ``CommitRecord`` and its consumption progress, returning the
       canonical accepted record. Its ``commit_number`` must equal
       ``expected_commit_number + 1``.
   * - ``recover(checkpoint=...)``
     - Accept a checkpoint ``CommitRecord`` (``None`` at initial registration),
       reconcile it with history, and return the head ``CommitRecord`` or ``None`` for a fresh
       scenario.
   * - ``close()``
     - Release the session's record and commit resources; repeated calls are
       safe. Each session closes after its trainer, also when construction
       or recovery fails.

A ``CommitRecord`` carries the operation, component selection, private storage
locator, algorithm state, record watermark, consumed IDs, checkpoint/pending
flags, metrics and training job identity. The operations include ``training``,
``activate``, ``train_from``, ``import``, ``sync``, ``serve``,
``stop_serving`` and ``restore``:

- A selection maps component names directly to artifact IDs.
- ``activate`` and ``serve`` record a complete selection without publishing
  another artifact snapshot. ``activate`` and ``train_from`` carry no trainer
  state.
- ``sync`` selects its new artifact for serving and training in one commit.
- ``restore`` records a complete prior active selection and leaves training
  starts unchanged.

``append_commit`` must validate the current commit number atomically:

- Two different commits prepared from the same commit number cannot both
  succeed.
- An identical retry returns the original record, even after later commits.
  Only ``recorded_at`` is excluded from the comparison.
- A new commit with a stale expected commit number, or a retry with
  conflicting content, raises ``ScenarioStoreConflictError`` without advancing
  the store.
- A failure after the append can leave the commit stored. Retry the exact
  commit or recover the session; do not infer rollback from an exception.

``CommitHistoryScenarioStore`` takes a ``RecordStore``, a ``TaskStorage``
(``tasks=``) and a ``CommitHistory``. The bundled record stores give them from
``task_storage()`` and ``commit_history()``. The store appends each commit,
and the train count of each sample that the commit lists, in one write
transaction of the scenario's database. Direct writes through a
``CommitHistory`` bypass this contract. Recovery uses committed
``consumed_ids`` and separate skipped-batch receipts to rebuild processor
memory without repeating work. It does not modify stored bodies.

Artifact bytes and backend head movement remain outside ``ScenarioStore``.
``ScenarioCommitter`` owns their order around store settlement, including
inactive artifacts, activation, new training starts, and checkpoint
reconciliation. See `Commit ordering
<../advanced_topics/state-model.rst#commit-ordering>`__. Step validation does not
provide distributed leases or authorize multiple active training writers.

Task stores
~~~~~~~~~~~

A ``TaskStore`` (``reef.storage.tasks``) holds published Harbor tasks, each in
the ``train`` or ``eval`` split. A record-to-sample stage that generates tasks
saves them in a task store before any rollout. Thus generation and rollout can
recover independently after a restart.

To open a store, call ``task_storage.open(name)``. A pipeline stage gets
``task_storage`` from ``PipelineContext.task_storage``, and each
record-to-sample stage uses its own ``name``. For a composite recipe, each
component's ``TaskStorage`` opens ``<component>/<name>``, so two components
never share a store.

A store holds generations. A generation holds its source record ids and its
candidates. A task is a Harbor task directory: the database keeps the state of
the store and the digest of each directory, and the task file store keeps the
files, text or binary, under that digest. Each ``TaskCandidate`` holds the
task ``name``, the ``files_digest`` of its directory, and its outcome:

- A published candidate has a ``split``, ``train`` or ``eval``. The stage
  chooses the split, and the store keeps it.
- A rejected candidate has an ``error`` that gives the reason.
- A pending candidate has neither. After a restart, the stage validates the
  candidates in ``GenerationState.pending_indices``.

The published tasks are the candidates with a split. A ``StoredTask`` holds
the task ``name``, its ``split`` and its ``files_digest``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``begin_generation(generation=..., source_record_ids=...)``
     - Create the checkpoint of a generation. A retry with the same source
       record ids returns the checkpoint.
   * - ``write_candidates(generation=..., directories=...)``
     - Save a copy of each candidate task directory once, before validation or
       rollouts. The name of a candidate is the name of its directory.
   * - ``publish_task(generation=..., index=..., split=...)``
     - Publish one candidate and return its ``StoredTask``. Each task name is
       published once. A later candidate with the same files must give the
       same split. It then returns the published task.
   * - ``reject_task(generation=..., index=..., reason=...)``
     - Save the rejection of one candidate.
   * - ``complete_generation(generation=...)``
     - Mark the generation complete. Each candidate must have an outcome.
   * - ``generation(generation)``, ``generations()``
     - Return one checkpoint or all checkpoints.
   * - ``tasks(split=None)``
     - Return the published tasks in generation and candidate order, each
       name once. Give a ``split`` to return only the tasks of that split.
   * - ``path(files_digest)``
     - Return the local directory of a candidate, a published task or a
       ``write_files`` call, for example for ``read_harbor_task`` or
       ``TaskRolloutRunner.run``. Do not change it.
   * - ``write_document(key=..., document=...)``, ``documents()``
     - Save a text document under a non-empty key, or return every document by
       key. Each key holds one document; another document under a used key
       raises ``TaskStoreError``. The store does not read a document: the stage
       that writes the store keeps its own state about its tasks here.
   * - ``write_state_document(key=..., document=...)``,
       ``delete_state_document(key=...)``, ``state_document(key)``,
       ``state_document_keys(prefix="")``
     - Save, remove or read a state document, or return the keys that start
       with ``prefix`` in increasing string order. A state document is text
       under a non-empty key. Unlike a document, its key can be written again
       and removed. The two kinds of document have separate keys. The stage
       keeps there the state that changes in place: the generated-task stage
       gives it to its task generator as ``GeneratorState``.
   * - ``write_files(directory)``
     - Save a copy of a directory in the task files of the scenario and return
       the digest of its files, which ``path`` takes. The call makes no
       candidate and changes no generation.

Each write is durable when it returns. An identical retry of a write changes
nothing. A write that conflicts with the stored state raises
``TaskStoreError``. If a task name is published with other files or in the
other split, the store raises ``TaskConflictError``, a subclass of
``TaskStoreError``. The caller serializes the writers of one store.

SQLite keeps the task tables in the database file of the scenario.
PostgreSQL keeps them in the deployment schema, with a key for each store
generation. ``archive`` retires the task tables and the task files with the
records and commits. Record retention does not delete tasks.

The scenario's imported tasks are not a task store. They are task items that
the scenario uses, and each trainer reads them from
``PipelineContext.samples`` (a ``SampleStore``). The service takes a task item
in the ``train`` split of a scenario (``POST
/reef/namespaces/{owner}/data/tasks``) only when a trainer of the scenario
trains imported tasks: its sample-to-job stage sets the class attribute
``accepts_imported_tasks``. An upload to the ``eval`` split is open to every
recipe, because an evaluation runs those tasks.

Task file stores
~~~~~~~~~~~~~~~~

A ``TaskFileStore`` (``reef.storage.tasks``) keeps the directories of the
published tasks, addressed by the digest of their files. The digest covers the
directory name, and the path and bytes of each file. Thus a repeated save of
the same files changes nothing. A deployment selects the store in
``storage.task_file_backend`` (see `Choose the task file store
<../user-guide/storage.rst#choose-the-task-file-store>`__).
``LocalTaskFileStore`` is the built-in store. An embedded caller can give a
store to ``SQLiteScenarioStorage(..., task_files=...)`` or
``PostgresScenarioStorage(..., task_files=...)``. Without one, the storage
uses a ``LocalTaskFileStore`` in ``task-files`` under its directory.

A store keeps its files in partitions: one for each scenario, and
``data/<owner>`` for the task items of each data owner. A client can write a
task directory into a partition itself, in the layout of the store, and then
create the task item by reference (`Create task items
<http-api.rst#create-task-items>`__). The layout belongs to each store.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``from_config(config, *, directory)``
     - Class method that builds the store a deployment names. ``directory``
       is ``storage.task_file_dir``. A remote store can use it as a local cache.
   * - ``write(scenario, source)``
     - Save a copy of the task directory ``source`` and return the digest.
       The copy is durable when the method returns. Raise ``TaskStoreError``
       for a symbolic link or a special file.
   * - ``path(scenario, digest)``
     - Return a local directory with the files of ``digest``. The directory
       has the name of the saved directory. Raise ``TaskStoreError`` if the
       store does not hold the digest.
   * - ``contains(partition, digest, name)``
     - Return whether ``partition`` holds a complete task directory named
       ``name`` with the files of ``digest``. A directory that a client
       writes is complete when the client has written its last part. Do not
       read the files. Raise ``ValueError`` for a digest that is not 64
       lowercase hexadecimal digits.
   * - ``archive(scenario)``
     - Retire the files of the scenario and return the archive locations.
   * - ``close()``
     - Release the resources of the store. A second call does nothing.

The module also gives the digest and the limits that every client uses for a
task directory:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Name
     - Contract
   * - ``files_digest(source)``
     - Return the SHA-256 hex digest of the local task directory ``source``:
       its name, and the path and bytes of each file. Directories add
       nothing. Raise ``TaskStoreError`` for a symbolic link or a special
       file.
   * - ``FilesDigest(directory_name)``
     - Compute ``files_digest`` from files given one at a time. Call
       ``add_file(relative_path, size_bytes)`` for each file, then
       ``update(chunk)`` until the file's bytes are given, then
       ``hexdigest()``. The files come in increasing ``task_file_order``.
       A file out of order, or a file with more or fewer bytes than its size,
       raises ``TaskStoreError``.
   * - ``task_file_order(relative_path)``
     - Return the sort key of a file path relative to the task directory: the
       tuple of its parts, so ``a/b`` comes before ``a-b``. Raise
       ``ValueError`` for an empty part, ``.`` or ``..``.
   * - ``task_limit_errors(entries)``
     - Return a ``TaskLimitError`` for each limit that a task breaks, from its
       ``(relative_path, size_bytes)`` entries (``None`` as the size of a
       directory). Each error has a ``code``, the ``path`` (``""`` for the
       task directory) and an English ``message``. The codes are
       ``file_too_large`` (a file over ``MAX_TASK_FILE_BYTES``, 500 GB),
       ``too_many_entries`` (a directory with more than
       ``MAX_TASK_DIRECTORY_ENTRIES``, 10,000, entries) and
       ``too_many_files`` (more than ``MAX_TASK_FILES``, 100,000, files).

Data stores
~~~~~~~~~~~

A ``DataStore`` (``reef.storage.data``) keeps the deployment's data items. An
item is ``(id, content, metadata)``:

- Its content never changes: the JSON object of a trajectory or a record, or
  the directory of a task, which the task file store keeps in the owner's
  partition.
- Its metadata is a JSON object. Reef reads ``kind`` (``trajectory``, ``task``
  or ``record``), ``owner``, ``scenarios`` (scenario name to ``train`` or
  ``eval``) and, for a record or artifact, ``origin`` (``{scenario,
  artifact_id}``). The keys in ``FIXED_KEYS`` (``kind``, ``owner``, ``origin``
  and ``evaluation_id``) never change.

A deployment selects the store in ``storage.data_store``. ``SQLiteDataStore``
and ``PostgresDataStore`` (``reef.storage.sql_data``) and ``MemoryDataStore``
are built in. Every operation names one owner and reads or writes only its
items. Every store links the record items of each owner into trajectories.
The SQL stores keep only the messages that each item adds; reads return the
whole content. With a segment store, the SQL stores keep the content of
trajectory items and record messages in segments (see `Segment stores`_).

The table gives the members that a caller uses most. A plugin implements
every abstract member of the class, including the scenario uses, holds,
collections and job data directories that ``reef/storage/data.py`` describes.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``from_config(config, *, directory, task_files, segments,
       job_directory_ttl_seconds=...)``
     - Class method that builds the store a deployment names. ``directory``
       is ``storage.agent_record_dir``. The caller owns ``task_files``.
       ``segments`` is the ``storage.segment_backend`` store or ``None``; the
       data store owns it and closes it. ``job_directory_ttl_seconds`` is
       ``storage.job_directory_ttl_s``.
   * - ``create(owner, items, *, generation=None)``
     - Stores ``NewItem`` values, all or none. The same ID and content again
       changes nothing; other content is ``ItemConflictError``. Bad metadata is
       ``ValueError``. A task's content is a local task directory, which the
       store copies into the owner's task file partition, or a
       ``TaskFilesReference(digest)`` (``reef.storage.item_kinds``) to a
       directory that the partition holds; a reference that it does not hold
       is ``TaskFilesNotFoundError``.
   * - ``read(owner, item_id)``
     - The ``StoredItem``, or ``None``.
   * - ``query(owner, *, kind, scenario, after, limit, content=True, ...)``
     - The owner's items after the sequence ``after``, in arrival order, at
       most ``limit`` (1 to 1000), of ``kind`` and used by ``scenario`` when
       given. With ``content=False``, each item's ``content`` is ``None`` and
       the store does not read it. The method takes more optional filters;
       see its signature.
   * - ``update(owner, item_ids, patch)``
     - Applies one JSON merge patch to each item's metadata, all or none.
       Changing a fixed key is ``ValueError``; an unknown ID is
       ``ItemNotFoundError``.
   * - ``delete(owner, item_id)``
     - Removes the item; ``False`` for an unknown ID. The store keeps it as a
       ``DeletedItem`` (the item without content, and ``deleted_at``), one for
       each ID; a second delete of the ID replaces the earlier one. A deleted
       item is in no read, query, set, use or record link, and ``create`` can
       use its ID again.
   * - ``deleted_items(owner, *, kind=None, before=None, limit=1000)``
     - The owner's ``DeletedItem`` values, of ``kind`` when given, by item
       sequence newest first, below ``before`` when given.
   * - ``read_deleted(owner, item_ids)``
     - The ``DeletedItem`` of each given ID that has one, in the order given.
   * - ``restore(owner, item_ids)``
     - Makes the deleted items items again, all or none, with their sequence,
       metadata and content, and gives them the uses that their metadata
       gives. Each use gets a new use sequence. An ID without a deleted item
       is ``ItemNotFoundError``; an ID that the owner stores is
       ``ItemConflictError``.
   * - ``purge(owner, item_id)``
     - Removes the deleted item for good; ``False`` when there is none.
   * - ``purge_deleted(time)``
     - Purges each deleted item, across owners, deleted before ``time``, and
       returns their number.
   * - ``task_path(item)``
     - A local directory with the files of a task item.
   * - ``record_links(owner, item_ids)``
     - The ``RecordLink`` of each given record item that has a message list:
       ``parent``, the nearest ancestor in its trajectory that is still
       stored (``None`` for a root), and ``entry_count``. A trajectory ends at
       a call that no stored call names as parent; its calls are
       ``reef.core.trajectory_index.trajectory_calls`` of the path from the
       root. Reports and unknown IDs are left out.
   * - ``groups(owner, *, kind, key, scenario=None, before=None, limit=1000)``
     - The ``ItemGroup`` values of the owner's items of ``kind`` by the string
       under the metadata ``key``: ``value`` (``None`` without a string),
       ``item_count``, ``first_sequence``, ``last_sequence`` and the first
       item's ``metadata``. Newest first, below ``before`` when given. The
       default reads every page of ``query``; the SQL stores group in SQL.
   * - ``maintain()``
     - The store's periodic work. The data service calls it every 5 minutes.
       It changes no item. The default deletes the job data directories that
       no job used for the time to live, and purges the items deleted 30 days
       ago or earlier (``DELETED_ITEM_TTL_SECONDS``).
   * - ``close()``
     - Releases the store's resources; a second call does nothing.

The store checks the form of the metadata. The service checks that
``scenarios`` names only scenarios of the owner.

Segment stores
~~~~~~~~~~~~~~

A ``SegmentStore`` (``reef.storage.segments``) keeps segments: immutable
JSON Lines files, each in one partition. The SQL data stores use one
partition per owner. A deployment selects the store in
``storage.segment_backend`` (see `Keep large content in segment files
<../user-guide/storage.rst#keep-large-content-in-segment-files>`__).
``LocalSegmentStore`` is the built-in store.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``from_config(config, *, directory)``
     - Class method that builds the store a deployment names. ``directory``
       is ``<storage.agent_record_dir>/segments``. A remote store can use it
       as a local cache.
   * - ``write(partition, segment_id, data)``
     - Store the bytes ``data``. The segment is durable when the method
       returns. The data store never writes one ID twice.
   * - ``read(partition, lines)``
     - The bytes of each ``SegmentLine`` (``segment_id``, ``offset``,
       ``length``), in order. Raise ``SegmentNotFoundError`` for a missing
       segment or a line past its end.
   * - ``delete(partition, segment_id)``
     - Remove the segment. A missing segment is not an error.
   * - ``read_directory_uri(key)``
     - The URI of the directory ``key``, or ``None`` when the store does not
       hold all of it. A key is ``/``-separated names. The data store keeps
       each job data directory as ``job-data/<key>``.
   * - ``write_directory(key, source)``
     - Keep a copy of the files under ``source`` as the directory ``key``,
       and return its URI. A write of a key that the store holds keeps the
       first copy.
   * - ``open_directory(uri)``
     - A local directory with the files of the directory at ``uri``. A remote
       store copies them into its local cache.
   * - ``delete_directory(key)``
     - Remove the directory. A missing directory is not an error.
   * - ``close()``
     - Release the resources of the store.

Artifact backends
-----------------

An artifact backend stores the snapshots of each scenario. By default, Reef
uses Git LFS (``GitLFSRepositoryBackendFactory``). A deployment selects a
different backend in ``storage.artifact_backend`` (see `Choose the artifact
backend <../user-guide/storage.rst#choose-the-artifact-backend>`__).

A backend plugin is a subclass of
``reef.artifact.ConfigurableRepositoryBackendFactory``. Calling the factory
with a scenario name returns the ``RepositoryBackend`` of that scenario. The
plugin implements one class method:

.. code:: python

   @classmethod
   def from_config(cls, config, *, context): ...

- ``config`` is the plugin's own ``config`` mapping. Raise ``ValueError`` for
  an invalid value.
- ``context`` is a ``RepositoryBackendContext``. ``work_dir`` and ``cache_dir``
  are ``storage.artifact_work_dir`` and ``storage.artifact_cache_dir``.

Reef does not call a close method on a backend factory.

A snapshot's revision and checksum are private locators; the public identity
belongs to each component's ``artifact_id``. Activating or serving an existing
artifact updates scenario selections without calling ``publish``. Every
scenario owns its root snapshot, which Reef writes when it creates the
scenario (`Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__).
A backend shares no snapshot between scenarios.

Snapshots and artifact IDs
~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``RepositoryBackend`` implements these members and keeps each component
entry's ``artifact_id`` in the manifests it stores:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Member
     - Contract
   * - ``register(root, *, metadata)``
     - ``root`` is an unpublished ``Artifact`` whose ``local_path`` is a
       directory. When the scenario has no head, store the files of ``root``
       as a snapshot with no parent, preserve its artifact IDs and
       ``metadata``, and make it the head in one atomic step. When the
       scenario has a head, also one that another caller registered first,
       store nothing and return that head.
   * - ``artifact_snapshot(artifact_id)``
     - Return the ``SnapshotRef`` of the stored snapshot that holds the
       artifact, or raise ``ArtifactNotFoundError``. The ID must keep naming
       the same files and metadata across publications and restarts. Reject
       an attempt to bind it to different content. The inherited
       ``resolve_artifact(artifact_id)`` materializes that snapshot and
       returns the artifact's component.
   * - ``bind_artifact_id(artifact_id, ref)``
     - Bind an unbound ID to a stored snapshot and return any earlier
       binding. A binding never changes.
   * - ``artifact_content(ref, artifact_id)``
     - Return a value that is equal for two snapshots exactly when they hold
       the same files and metadata for the artifact. The inherited
       ``bind_artifact_ids(snapshot)`` uses this and ``bind_artifact_id``,
       and raises ``ArtifactConflictError`` for an ID that names other
       content. Call it before a head moves.
   * - ``resolve_snapshot(revision)``
     - Return a ``SnapshotRef`` for a stored snapshot, or raise
       ``ArtifactNotFoundError``. Its ``revision`` and ``parent_revision``
       are backend locators, separate from component IDs.

Stored content
~~~~~~~~~~~~~~

Stored content is a named file tree of one scenario with no parent, no
metadata and no artifact ID, and no head moves to it. The training service
stores each checkpoint as content, and control commits it by name without
reading a file. A backend implements five operations on it:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Member
     - Contract
   * - ``store_content(name, directory)``
     - Store the files of ``directory`` as the content ``name``: exactly
       those files, whatever their names. Storing a name again replaces its
       files in one step for a reader, so a repeated call after a failure is
       safe.
   * - ``materialize_content(name)``
     - Return a local directory with exactly those files, which stays
       readable when the name is stored again.
   * - ``discard_content(name)``
     - Remove the content. A name that the backend does not hold changes
       nothing, and every snapshot keeps its files.
   * - ``discard_all_content()``
     - Remove every content of the scenario. Control's sweep calls it after
       the scenario was deleted and its registration archived, so the call
       must find the content of a scenario that has no registration. The
       sweep repeats a call that failed.
   * - ``publish_parts(parts, *, metadata, expected_parent, advance_head)``
     - Store a snapshot of files that the backend holds already, and move
       the head as ``publish`` does. ``parts`` maps the directory of each
       component to its ``StoredFiles``, or the one key ``None`` for a flat
       snapshot. A ``StoredFiles`` names stored content, or an artifact ID
       whose files an earlier snapshot holds. Carry a component's files byte
       for byte. ``metadata`` is complete: it holds the component manifest
       and the artifact IDs. Bind them with ``bind_artifact_ids`` before a
       head moves. Raise ``ArtifactNotFoundError`` for content or an
       artifact that the backend does not hold. A failure can leave the
       snapshot stored with its IDs bound and no head on it. Make the
       snapshot inside the storage, for example with a server-side copy, so
       that the calling process downloads no file.

Callers use the inherited ``publish_stored``, which checks first that the head
is ``expected_parent``, also for ``advance_head=False``. A backend that moves
the head checks it again in the same step as the move.

The factory has two cleanup hooks, each with a default that does nothing:

- ``reclaim_discarded()``: frees the files of content whose discard deleted
  only the name. Control's sweep of ended work calls it once per sweep.
- ``forget_scenario(scenario)``: drops what the process keeps of a deleted
  scenario beside the store, such as a cached backend, a clone or the cached
  files of its snapshots. It changes nothing for a scenario that was not
  deleted. Each service calls it when it learns of the delete.
  ``CachedRepositoryBackendFactory`` checks ``has_archived_registration``,
  drops its cached backend, and calls ``forget_local_state(scenario)``, which
  a subclass overrides to remove its own files.

Optional capabilities
~~~~~~~~~~~~~~~~~~~~~

Inherit only the base classes that the backend supports:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Base class
     - Capability
   * - ``RegistrationAwareRepositoryBackendFactory``
     - ``has_registration(scenario)``: find a stored scenario without making
       its backend.
   * - ``EnumerableRepositoryBackendFactory``
     - ``list_registrations()``: list the stored scenarios. The service
       loads them when it starts.
   * - ``CachedRepositoryBackendFactory``
     - Both capabilities, and one cached backend for each scenario. A
       subclass must implement ``has_archived_registration(scenario)``: True
       exactly when ``archive_registration(scenario)`` has completed in any
       process that shares the store, and never False again after that. Reef
       refuses to create a scenario for which it is true, because a deleted
       scenario's ID is never used again.
   * - ``StagedSnapshotRepositoryBackend``
     - The backend side: ``publish(advance_head=False)`` stores a snapshot
       without moving the head, and ``commit_snapshot`` moves the head
       after the scenario commit. A durable scenario store requires it.

A backend overrides these ``RepositoryBackend`` members when its storage
allows it:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Member
     - Contract
   * - ``snapshot_metadata(ref)``
     - Read a snapshot's metadata without its files. The default
       materializes the whole snapshot.
   * - ``materialize_component(ref, name)``
     - Make one component of a snapshot local without the files of the
       others. Control reads a harness tree through it, so that it fetches
       no weights. The default materializes the whole snapshot.
   * - ``file_downloads(ref, expires_in_seconds)``
     - For storage that can give clients a URL for each file, such as an
       object store. Return every file of the snapshot as a
       ``FileDownload``: ``path`` in the snapshot, ``byte_count``, and a
       ``url`` that opens the file without a Reef credential until
       ``expires_at``. It must not materialize the snapshot. The artifact
       file list of the HTTP API gives these URLs to clients. The default
       returns None, and Reef then serves the files itself.
   * - ``snapshot_uri(ref)``
     - The URI of the snapshot's directory, ending with ``/``, in a scheme
       that ``ArtifactSources`` reads. The `artifact items
       <http-api.rst#artifact-items>`__ of the data store carry it, so that a
       scenario, also in another deployment, imports the snapshot from
       there. The default returns None.

``tests/artifact/test_reef_artifacts.py`` and
``tests/artifact/test_reef_git_lfs.py`` show the publication and recovery
behavior that a backend must have.

Artifact imports
----------------

``POST /reef/scenarios/{scenario}/artifacts`` imports a directory as a new
artifact of one component (`Import an artifact
<../user-guide/operate.rst#import-an-artifact>`__). A source reader opens the
files that the URI names, the component's layout checks them where the source
keeps them, and then Reef stores a copy, or a reference to a Hugging Face
revision that cannot change. After the layout, the component's admission check
runs on what Reef stores (see `Scenario components
<../developer-guide/components.rst#admission-checks>`__). Two contracts decide
what an import accepts.

Layouts
~~~~~~~

A component's layout is the ``layout`` of its ``ComponentSpec``, an
``reef.artifact.components.ArtifactLayout``. ``None``, the default, means that
the component takes no import.

``check(files)`` returns when ``files`` are an artifact in the layout of the
component's snapshots, and the remote weights that they name exist. Otherwise
it raises ``ValueError``, and Reef answers HTTP 400 before it copies anything.
``files`` is a ``reef.artifact.ArtifactFiles``. It gives ``paths()``, and
``sha256(path)`` and ``read_bytes(path)`` for each path. Read only the files
that the check needs: on an S3 prefix or a Hugging Face revision, each read
downloads that file.

The built-in recipes bind one:

- A weight recipe's component gets ``TrainingArtifactLayout(facts, component)``
  (``scenario_components``), which calls the facts' ``check_artifact`` (see
  `Training jobs and components`_). For full weights and adapters it reads
  only file names and ``adapter/adapter_config.json``.
- A harness recipe binds ``HarnessLayout``. Every file must be UTF-8 text,
  and the files must form a ``HarnessTree`` that passes ``validate_tree`` for
  the adapter: the source tree a harness artifact stores.

Source readers
~~~~~~~~~~~~~~

A source reader is a subclass of ``reef.artifact.ArtifactSourceReader`` with
two members:

.. code:: python

   @property
   def schemes(self) -> tuple[str, ...]: ...

   def open(self, uri: str) -> SourceFiles: ...

``open`` returns the ``reef.artifact.SourceFiles`` that ``uri`` names, and
reads at most a listing. Raise ``ValueError`` for a URI that names nothing the
reader can read, and ``ArtifactSourceError`` when the source fails.
``SourceFiles`` adds three members to ``ArtifactFiles``:

- ``version``: the version that the source served, or ``""`` when it has none.
- ``reference``: ``None`` when Reef stores a copy. Otherwise, the files, as
  paths and bytes, that Reef stores instead, for a source that keeps the
  files fixed.
- ``copy_into(directory)``: write the files into the empty ``directory``.

Reef ships ``HuggingFaceReader`` (``hf``), ``FileReader`` (``file``) and
``S3Reader`` (``s3``), whose files are ``HuggingFaceRevisionFiles``,
``DirectorySource`` and ``S3PrefixFiles``. A deployment adds readers in
``storage.artifact_sources``. A configured reader replaces the built-in reader
of a scheme that it reads.

``HuggingFaceRevisionFiles`` has a reference when the URI names a full
40-digit commit SHA. Reef stores the reference instead of a copy only when the
component's layout says that its readers resolve one
(``ArtifactLayout.resolves_references``; ``TrainingArtifactLayout`` does for snapshots
that hold weight files). Then Reef stores only ``hf-reference.json``,
``{"uri": "hf://org/model", "version": "<commit>"}``, and
``reef.artifact.read_reference(directory)`` reads it back. The built-in
serving runtimes load such a snapshot from the revision's snapshot, which
Hugging Face downloads once into its cache. A runtime that loads models from
the Hub can read the reference and download nothing.

Base roots
~~~~~~~~~~

An artifact of a weight component has one more form, the base root. It holds
one file, ``base.json``, and no weight files. The artifact stands for the
deployment's base model with no adapter:

.. code:: json

   {"format": "reef-base-root-1",
    "inputs": {"model": "/models/qwen", "lora_rank": "8"}}

``inputs`` are the inputs of the training backend's default start
(``TrainingBackendFacts.default_start_inputs``). ``model`` is always there.
Slime with LoRA adds ``lora_rank``, ``lora_alpha`` and ``lora_target_modules``.
Tinker adds ``lora_rank``, and ``service`` when the deployment names a base
URL. ``reef.artifact.BaseRoot`` is the value. ``read_base_root(directory)``
reads it from an artifact directory and returns ``None`` for an artifact
without the file. ``base_root_of(files)`` reads it from ``ArtifactFiles`` and
refuses an artifact that holds other files beside it. A reader refuses a file
with another ``format``.

The default start of the Slime and the Tinker backends is a base root:
``SlimeFacts.default_start()`` and ``TinkerFacts.default_start()`` write it,
and a new scenario's root holds it. The ``spindle_full`` backend has no base
root. Its default start is a reference to the base revision, which
``SpindleFullFacts.default_start()`` writes.

Record storage and audit
------------------------

``reef.storage.records`` defines ``RecordStore``, the record result and error
types, and ``RecordRetention``. ``reef.RecordStore`` is the same class.

``reef.storage.record_items.ItemRecordStore`` is the ``RecordStore`` of one
scenario over the data store. It appends each record as its record item, and
reads the record items that the scenario made, in item sequence order.
``SQLiteScenarioStorage`` and ``PostgresScenarioStorage`` give one to each
session. Both take a data store (``data=``). Without one, the SQLite storage
keeps its records in ``<directory>/data``, and the PostgreSQL storage in the
schema ``<schema>_data``.

``PostgresScenarioStorage(url, directory, schema=...)`` keeps commits (the
``scenario_commit`` table) and settings (the ``scenario_setting`` table) in
the schema. Install ``reef-infra[postgres]`` for the PostgreSQL driver. A
deployment selects it with ``storage.record_backend: postgres``; see `Choose
the storage backend <../user-guide/storage.rst#choose-the-storage-backend>`__.

An older database keeps its records in the tables ``agent_record``,
``record_node``, ``consumed_agent_record``, ``record_consumption`` and
``record_eviction``. The storage drops these tables when the record item of
the newest record of each active scenario is in the data store: the PostgreSQL
storage when it starts, the SQLite storage at the first open of each scenario.
Otherwise it keeps them and logs a warning: run Reef 83a6f3ab once to copy the
records.

Appending
~~~~~~~~~

``RecordStore.append_many(items)`` atomically appends an ordered batch from one
scenario and returns an ``AppendResult`` for each input. Conflicts roll back the
batch; identical retries and evicted records follow ``append_result`` semantics.
SQLite and PostgreSQL implement this with one write transaction. Other adapters
must implement atomic batch append to support ``POST /reef/records/batch``;
the default raises ``NotImplementedError`` before writing.

Training commits persist consumption progress without changing record
visibility or deleting bodies. Record-to-sample stages expose
``releasable_record_ids()`` and ``release_records(ids)``, which the samples
service calls for memory management. Storage controls capacity independently
of these buffers.

Reading
~~~~~~~

``read``, ``replay``, ``replay_page``, and ``count`` read all retained records.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Result
   * - ``read_for_audit(scenario, agent_record_id)``
     - A ``StoredRecord``, or ``None`` if no body is retained in that scenario.
   * - ``audit_page(scenario, after_sequence=0, limit=256)``
     - A bounded tuple of ``StoredRecord`` entries, in append order, including
       consumed bodies. Advance the cursor using the last entry's ``sequence``.
   * - ``loss(scenario)``
     - Durable ``RecordLoss`` totals: ``record_count``, ``body_bytes``,
       ``first_sequence`` and ``last_sequence`` of the record items that
       retention deleted (``DataStore.record_loss``). Counts include consumed
       and unconsumed records.
   * - ``trajectory(scenario, agent_record_id)``
     - The record IDs of the trajectory that the record ends, oldest first: the
       earlier calls with the same API format and tools whose messages the
       record's messages extend, one per prefix, and then the record. A retry
       with the same messages replaces the call that it retries. A call that
       retention deleted is left out. A record without a message list, or a
       store that keeps no trajectories, gives a trajectory of one call; ``()``
       for a record that is not stored.

``StoredRecord`` contains ``sequence`` and ``item`` (the original
``AgentRecord``). Reading a record does not change a consumer's progress. A
missing body may have been purged or never stored; the read API does not guess
which. Use the commit history's per-step ``consumed_ids`` to determine
consumption, including intentional skips; it does not prove that every named
record produced a model update.

For example, inspect one trace without making it available to training again:

.. code:: python

   entry = scenario.records.read_for_audit(scenario.name, record_id)
   if entry is not None:
       payload = entry.item.payload
       references = entry.item.references

Retention
~~~~~~~~~

The HTTP service runs background retention at startup and every 60 seconds. It
deletes the oldest record items, across owners, only when they hold more than a
shared 20 GiB budget (``DataStore.prune_records``). The budget counts each record
item's row and the trajectory nodes that only it needs. It counts and deletes
only the record items whose ``origin.source`` is ``inference``; it keeps the
calls of evaluation rollouts. `Configuration <configuration.rst>`__ gives the
limits.

Retention logs nothing per record. A trainer reports the loss of its scenario
in its metrics and status. Concurrent writes can exceed the budget between
sweeps, and indexes and other metadata also need disk space.

An embedded deployment has two options:

- ``dispatcher.prune_record_archives(RecordRetention(max_bytes=20 * 1024**3))``
  runs one sweep over the dispatcher's data store and serializes it with
  scenario archival. ``Dispatcher`` construction starts no maintenance task.
- ``create_app(dispatcher, record_retention=RecordRetention(...))`` enables
  the service's background retention.

Pipeline stages
---------------

.. code:: python

   from reef.record2sample import (
       GroupDecision,
       JudgedSamples,
       RecordJudging,
       ReportContext,
       ReportGrouping,
       ReportSamples,
       SkippedReport,
   )
   from reef.sample2job import DistillJobData, Sample2Job

A training pipeline (``reef.train.pipeline.TrainingPipeline``) turns durable
records into typed job data through two stages. A recipe names a default class
for each in ``pipeline_spec()`` (``PipelineSpec(record2sample=...,
sample2job=...)``). `Training pipeline <../developer-guide/write-a-recipe.rst>`__
owns the explanation; this section gives the signatures.

Reef owns replay, retention, deduplication, pending job data, and exactly-once
consumption; a method implements only the hooks below. The hooks run
synchronously on the trainer thread, so they must not block on network or
model latency. Reports must reference existing inference records in the same
scenario.

The record-to-sample stage
~~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.record2sample.Record2Sample`` decides which records train and which
train together. The default stage, ``NoSamples``, makes no sample, so nothing
trains.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Contract
   * - ``push(record)``
     - The pipeline pushes every record to the stage.
   * - ``form(record_set)``, ``make_samples(record_set)``
     - ``form`` makes the samples of each record set (``make_samples``) and
       holds them in a record set unit until the unit is ready. A sample is
       a ``Sample`` value.
   * - ``advance(now)``
     - A stage with slow work keeps its records and forms the set here when
       the work ends (see `Slow work
       <../developer-guide/recipe-samples-service.rst#slow-work>`__).
   * - ``is_stale(samples, served)``
     - Whether a group of the stage's stored samples may still train against
       the artifacts that the scenario served (``ServedArtifacts``). The
       samples service marks the stale ones, and the automatic jobs leave
       them out. The default is on-policy (see `the staleness check
       <../developer-guide/recipe-samples-service.rst#staleness>`__).
   * - ``uploaded_sample(trajectory) -> TrajectorySample``
     - Turns an uploaded ATIF trajectory into its training sample. It must
       be pure, and it raises ``ValueError`` to skip the trajectory (see
       `Uploaded trajectories
       <../developer-guide/recipe-control-service.rst#uploaded-trajectories>`__).
   * - ``job_ended(outcome)``
     - Called once the commit of a job's end is durable (`Training jobs and
       components`_). The default does nothing.

A ``Record2Sample`` keeps its state in memory and in the store, not in a
directory of its own, so a scenario delete archives nothing for it.

The sample-to-job stage
~~~~~~~~~~~~~~~~~~~~~~~

``reef.sample2job.Sample2Job.make_job_data(samples, data_id)`` shapes the
samples of the next job, in use order, into one job's data with the ID
``data_id``. The pipeline assigns the ID: ``<scenario>:data:<number>`` for
automatic job data and ``<scenario>:request:<request id>`` for a request's
data. The number counts the job data of one pipeline. It is not a training
step.

``Sample2Job.__init__(context, buffer_size, job_samples, *, max_job_samples=None)``
sets the most samples that one automatic job takes and the fewest samples of
one job (see `Job size <../developer-guide/recipe-control-service.rst#job-size>`__):

- A ``job_samples`` of ``None`` is a stage that sets no job size.
  ``Trainer.build`` raises ``PluginConfigError`` for a trainer that trains
  with such a stage.
- A ``buffer_size`` of ``None`` takes ``job_samples``.
- The constructor raises ``PluginConfigError`` for a ``buffer_size`` smaller
  than ``job_samples``, and for a ``job_samples`` above ``max_job_samples``
  (an SDPO or TTTD grid).
- ``Sample2JobSettings`` parses the ``buffer_size`` and ``job_samples`` keys,
  and a stage with more keys extends it.

A ``Sample2Job`` declares ``supported_training_modes``.
``TrainingPipeline.training_mode`` selects automatic (``auto``) or
request-triggered (``manual``) training. In ``manual``, a request's data holds
the samples that the request names; ``Sample2Job.reads_requests`` says whether
a request that names no sample runs.

Building the stages
~~~~~~~~~~~~~~~~~~~

Reef builds the stages for each scenario, in this order, with the class
methods below; the sample-to-job stage receives the record-to-sample stage:

- ``Record2Sample.from_config(config, *, context, environ)``
- ``Sample2Job.from_config(config, *, context, record2sample, environ)``

``config`` is the stage's config from the deployment's ``pipeline`` section;
``environ`` holds the environment the deployment was read with.

- Declare the stage's keys in a frozen ``reef.core.plugins.PluginSettings``
  subclass with ``reef.core.config.config_option`` fields, and validate them
  in ``__post_init__``.
- In ``from_config``, parse the config with ``Settings.parse(config,
  plugin_class=cls)`` (``NoSettings`` for a stage without config) and give the
  settings object to the constructor. A bad config fails when the first
  scenario builds its pipeline.
- A sample-to-job stage that needs the record-to-sample stage as a specific
  class checks it with ``isinstance`` and raises ``PluginConfigError``
  otherwise.
- A value that both stages need belongs to the record-to-sample stage; the
  sample-to-job stage reads it from the stage object that it receives.

Every stage gets the scenario's experiment logger as
``context.experiment_logger`` (``self.experiment_logger`` on a ``Sample2Job``).
Log finite numeric metrics under the method's namespace; stage code never
imports W&B, and the logger is a no-op when tracking is off.

Reported feedback
~~~~~~~~~~~~~~~~~

A reported-feedback recipe implements these hooks:

.. list-table::
   :header-rows: 1

   * - Hook
     - Contract
   * - ``ReportSamples.make_sample(context) -> Sample | SkippedReport``
     - Assemble one valid report and its resolved inferences; raise on data errors. Return
       ``SkippedReport(reason)`` for a valid report that cannot train; the stage releases it.
       The default makes the report's trajectories one ATIF trajectory sample with the report's score.
   * - ``Sample2Job.make_job_data(samples, data_id) -> JobData``
     - Shape the flat tuple of selected tasks into job data with the ID ``data_id``.

The recipe owns the report policy. Its record-to-sample stage is a subclass of
``ReportSamples(context, grouping=None, *, exclusive_sources=False)``:

- With no ``grouping``, it trains every report by itself.
- With ``exclusive_sources=True``, a report that ends owns its inference
  records. Use it when each report has its own inferences.
- A report references the last call of each trajectory that it scores. The
  stage expands each reference to the calls of its trajectory
  (``reef.core.trajectory_index``) and adds the calls and the report id to
  the sample's ``source_agent_record_ids``.
- No inference is in two record sets. A report whose trajectory holds an
  inference of another report ends without training, and so does a stored
  report that lists calls inside one trajectory.
- An invalid reference raises at admission. An incomplete group waits for
  more valid reports. A training input violation raises instead of
  silently filtering reports.

A ``ReportGrouping`` runs in the record-to-sample stage and reads only records:

.. list-table::
   :header-rows: 1

   * - Method
     - Contract
   * - ``placement(context) -> (group_key, slot)``
     - Optional collection group and retry slot. The default, ``(None, None)``, is an independent report. A group
       key collects reports, and a slot deduplicates retries within that group (None uses the report id).
   * - ``decide(key, members) -> GroupDecision``
     - Required when ``placement`` supplies a group key. ``members`` are the group's
       ``ReportContext`` values in arrival order. Return READY, INCOMPLETE, or DISCARD.
   * - ``is_training_report(report) -> bool``
     - Optional; ``False`` for a valid report that is not this method's training data. The stage releases it and
       never assembles it; its sources go with it only under ``exclusive_sources`` or when it references more than
       one inference.
   * - ``status() -> Mapping``
     - Optional JSON-safe group outcomes, such as steps that can never train. The pipeline adds them to its status.

``ReportContext`` carries ``report``, ``inferences`` (the calls of the report's
trajectories, oldest first), optional ``score``, and the recipe's
``parsed_report``. ``ends`` holds the referenced last calls, in reference order.
``require_score()`` returns a finite reward or raises if the training method
cannot use the supplied feedback.

A group is fixed when it is first ready; later reports for its key form a new
group. This collection group differs from ``TrajectorySample.group_id``, the
comparison group inside one job's data. The job's commit consumes every sample
that ``make_job_data`` receives, even when the recipe omits some from its
output. See `Reported feedback
<../developer-guide/write-a-recipe.rst>`__ and `the next job
<../developer-guide/recipe-control-service.rst#the-next-job>`__.

``TrajectorySamples(context, *, trajectory_idle_seconds=300.0)`` is the stage
of a method that needs no report. It forms a ``TrajectorySet`` of a
trajectory's calls once the trajectory's last call has had no continuing call
and no retry for ``trajectory_idle_seconds``, counted from the call's
``created_at``, and makes it one ATIF ``TrajectorySample``
(``trajectory_sample``). Its ``next_due()`` is the time at which the earliest
waiting trajectory goes idle. Every other record ends at once.

Computed feedback
~~~~~~~~~~~~~~~~~

Use this engine when later traffic completes an earlier record and judging it
calls a model or another slow service. The recipe writes a ``RecordJudging``.
Its record-to-sample stage is a ``JudgedSamples(judging, worker)`` subclass that implements ``make_sample``, and its
``Sample2Job`` shapes the judged samples into job data.

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - Hook
     - Contract
   * - ``RecordJudging.observe(record, now, stage)``
     - correlate one record with the stage's verbs; must not block
   * - ``async RecordJudging.judge(job)``
     - slow judgment, on the worker thread
   * - ``RecordJudging.expire(now)``
     - optionally return receipts whose completion window ended
   * - ``JudgedSamples.make_sample(record, judgment)``
     - a ``TrajectorySample``, or ``None`` to retire the record
   * - ``make_job_data(samples, data_id)``
     - the declared job data type
   * - ``required_request_types``
     - optionally restrict which record types reach the pipeline
   * - ``accepts_trajectories``
     - optionally set True to train on uploaded ATIF trajectories

Before each ``observe``, the stage retires expired records and absorbs finished
judgments. ``observe`` then uses the stage's ``track(record)``,
``tracked_record(receipt)``, ``dispatch(job)``, ``retire(receipt)``, and
``abandon(receipt)``. A failed judgment or a ``None`` sample retires the record
instead of failing a training step. All correlation state must be
reconstructible from replay.

Task generation
~~~~~~~~~~~~~~~

.. code:: python

   from reef.core.tasks import HarborTask
   from reef.recipe.generated_tasks import TaskGenerator, GeneratedTaskResult, TaskValidationResult
   from reef.record2sample.generator import TaskGenerationRequest

``TaskGenerator`` is the plugin base class that a deployment names in
``pipeline.record2sample.config.generator``. The built-in generated-task
recipe, ``reef.recipe.generated_tasks:GeneratedTaskRecipe``, calls its
methods. See `the task generation contract
<../developer-guide/task-generators.rst>`__ for the round
and its failure behavior.

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Member
     - Contract
   * - ``from_config(config, *, scenario, environ, state)``
     - Class method; returns the generator of one scenario. ``state`` is a
       ``GeneratorState``.
   * - ``async generate(request) -> tuple[HarborTask, ...]``
     - Generate tasks for one ``TaskGenerationRequest``.
   * - ``async validate(task) -> TaskValidationResult``
     - Check one ``HarborTask``. ``TaskValidationResult(errors=())`` exposes
       ``is_valid``; each error is a non-empty explanation of a task defect.
       A failure to execute the checks raises an exception instead.
   * - ``async record_results(results, *, generation)``
     - Optional. Receives the ``GeneratedTaskResult`` values of one earlier
       ``generate`` call, named by ``TaskGenerationRequest.generation``,
       possibly after later calls and after a restart. A second call for the
       same generation must have no effect. The default keeps nothing.

``GeneratorState`` (``reef.recipe.generated_tasks``) is the durable state of
the scenario's generator, kept in the scenario's task store. It holds text
documents by key (``read``, ``write``, ``remove``, ``keys``) and saved
directories by the digest of their files (``save_files``, ``files``). A
generator keeps its state there and keeps no file of its own.

``TaskGenerationRequest(source_records, description, assets=(), asset_files={}, generation=0)``
carries source records, non-empty requirements text, and the number of the
generation:

- Each ``AgentRecord`` must have a distinct ID and belong to the same
  scenario. Use an empty record tuple for description-only generation.
- For local materials, pass a tuple of ``Path`` values in ``assets``. For
  materials in memory, use ``asset_files`` to map relative file names to
  UTF-8 text. Supplying both fields is an error.
- The constructor validates fields without reading files.

Record2dataset accepts the same input through ``DesignerRequest(inputs=request,
...)``. Its HTTP client reads local materials and sends their contents. The
service rebuilds ``TaskGenerationRequest`` from the JSON fields. The generated
task preserves source record IDs in order.

``GeneratedTaskResult`` is ``RejectedTask(task, reason)``, ``HeldOutTask(task)``,
``RolledOutTask(task, rollout_rewards)`` or ``FailedTask(task, error,
infrastructure_failure)``. ``rollout_rewards`` is a non-empty tuple of finite
numbers; ``reason`` and ``error`` are non-empty.

Job data
--------

.. code:: python

   from pathlib import Path
   from reef.train.types import (
       TaskSample, Sample, JobData, TrajectorySample,
       trajectories, trajectory_groups,
   )

``Sample = TrajectorySample | TaskSample``. Every sample-to-job stage returns a
frozen ``JobData(data_id, samples, request=None)`` with an ordered tuple of
samples:

- One job's data holds samples of one kind. Both readers,
  ``trajectories(data)`` and ``trajectory_groups(data)``, reject an
  unsupported sample kind instead of filtering mixed job data.
- ``request`` is keyword-only and carries an optional explicit training
  request; request-only job data may be empty.
- The pipeline returns the same pending job data until its id is
  acknowledged. Reef submits one training job for each value. The training
  component cuts the samples into its own optimizer batches.
- The component reads job data as a job data directory, so only the fields of
  ``JobData`` reach it. Keep job data serializable, with no handles to
  services, files, threads or models.
- Both sample kinds may carry ``source_agent_record_ids``. These record the
  derivation inputs; the pipeline's acknowledgement remains authoritative for
  consumption.

.. code:: python

   data = JobData(
       "scenario:data:1",
       (
           TrajectorySample(atif_document, source_agent_record_ids=("inference-1", "report-1")),
           TrajectorySample(other_atif_document, source_agent_record_ids=("inference-2", "report-2")),
       ),
   )

Trajectory samples
~~~~~~~~~~~~~~~~~~

A ``TrajectorySample.trajectory`` is an
`ATIF document <https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md>`__
as a JSON-compatible mapping. The item checks the envelope; consumers validate
the fields their algorithms require. Standard steps describe messages, tool
calls and observations. Captured records use ATIF-v1.7, matching the pinned
Harbor version.

Reef-specific fields live inside ``trajectory.extra.reef``:

- ``reward`` and ``feedback`` hold the supplied or computed training signal;
- ``source_agent_record_id`` identifies the primary source;
- ``records`` preserves ordered source ids, timestamps and original provider
  payloads, including provider-specific fields;
- ``teacher`` is a distillation stage's teacher request (``messages``, or
  null for a teacher that reads the student's own sequence, ``tools``,
  ``max_prompt_tokens`` and ``max_tokens``); OpenClaw-RL writes one request
  for each hint in ``teacher_candidates``;
- ``max_tokens`` is the longest sample, in tokens, that a stage lets train;
- ``training`` holds what a recipe sets for its objective, such as
  ``rollout_created_at`` or ``distill_sample_weight``. A sample that serving
  recorded has no ``tokens`` and no log-probabilities in it.

A sample that the record-to-sample stage makes holds the provider exchange
and no tokens. The training component renders it (`Chat rendering`_). The
rendered sample then holds ``tokens`` and ``loss_mask`` in ``training``, and
``teacher_tokens`` or ``teacher_candidates`` when it has a teacher request. A
task rollout of a training component carries its exact ``tokens``,
``loss_mask`` and ``rollout_log_probs``, ``artifact_id`` (the artifact that
sampled it), and, when the component records them, the sampler's
``topk_indices`` and ``topk_log_probs``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Contract
   * - ``item.metadata``, ``item.training``
     - Read the extension objects. Training arrays are JSON lists.
   * - ``with_metadata``, ``with_training``
     - Return updated documents and preserve the other extensions.
   * - ``item.artifact_id``
     - The artifact that produced the trajectory: ``training.artifact_id``,
       else the ``artifact_id`` of the last recorded call
       (``records[-1].payload``), or None for an upload. The samples
       service's staleness check reads it.
   * - ``group_id``
     - An optional string, scoped to the job data. None means an independent
       trajectory. Group-relative policy algorithms require an explicit id
       on every trajectory and contiguous samples for each group, so
       advantages and rows stay aligned. ``trajectory_groups(data)`` reads
       these comparison groups.

TTTD and CORAL keep their group barriers and ordering. SAO and OpenClaw-RL
schedule each trajectory independently, even when group metadata is present.

The ``reef.core.trajectories`` helpers construct and consume ATIF:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Helper
     - Contract
   * - ``make_trajectory(records, reward, feedback)``
     - Converts the exchanges with ``reef.core.atif_payloads``, as the record
       stores do, and writes the last record's ``tools`` as the agent's
       ``tool_definitions``. It keeps a summary of each call in
       ``extra.reef.records`` (``call_summary``): its ``agent_record_id``,
       ``created_at`` and ``artifact_id``, and its ``payload`` without the
       conversation, the tools and the response (``CONVERSATION_FIELDS``),
       such as ``model`` and ``chat_template_kwargs``.
   * - ``recorded_payloads``
     - Each call's stored payload.
   * - ``recorded_payload``
     - The last call as served, with the conversation that the steps
       project, for harness methods and for external ATIF without captured
       records.
   * - ``trajectory_reward``
     - A finite policy reward.

``SampleAssembly.build`` (``reef.record2sample.assembly``) makes a report's
sample: the trajectory's last call, or every call with
``accept_multi_turn_policy_samples``. A task rollout of a training component
uses ``make_policy_trajectory`` or ``make_multi_turn_policy_trajectory`` from
``reef.record2sample.samples`` to keep the tensors that its engine returned.

Completed text-only streaming responses contribute their aggregated
``response.message`` to the ATIF agent step. The original SSE body remains
in the captured record. Matching assistant messages in later request history
are reused as context without duplicating those steps. An explicit
``training.response_message`` takes precedence over the stream summary.

Task samples
~~~~~~~~~~~~

A ``TaskSample`` names a `Harbor task directory <https://www.harborframework.com/docs/tasks>`__
with ``task_path: Path``: the directory contains the instruction, environment
and verifier configuration/files. The path is interpreted in the consuming
algorithm's execution environment. The value also accepts optional ``metadata``.
``files`` is a ``TaskFiles(partition, digest)`` that names the directory in the
task file store, or ``None``. Only a task with ``files`` can go into a training
job (see `The job data directory <../developer-guide/training-backends.rst#the-job-data-directory>`__).

A ``TaskSample`` performs no file I/O or rollout. Algorithms supporting tasks
own validation, environment execution and conversion to trajectories. The
Tinker component, and the Slime component with its own rollout engines, roll
task samples out in task jobs; the harness backends refuse them. Accepting
``JobData`` does not imply that every algorithm supports both sample kinds.

Chat rendering
~~~~~~~~~~~~~~

Serving returns no tokens and no log-probabilities. A training component
renders each trajectory sample with its model's chat template, and computes
the log-probabilities that its objective needs with the weights of the job's
start. `Rendering in the training component
<../developer-guide/write-a-recipe.rst>`__
explains segments and the samples that a component leaves out.
``reef.core.chat_rendering`` holds the shared rendering:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Name
     - Contract
   * - ``ChatTemplate``,
       ``TokenizerChatTemplate(tokenizer, template_kwargs)``
     - Render messages and tools to token IDs; the second is the chat
       template of a Hugging Face tokenizer. Its ``template_kwargs`` must
       equal the chat template arguments that the serving engine renders
       with.
   * - ``render_trajectory(template, trajectory)``
     - One ``RenderedTrajectory`` for each segment. The agent's
       ``tool_definitions`` are the tools. Each agent step that is not
       ``is_copied_context`` trains: the loss mask is 1 on its tokens. A
       turn whose call's prompt extends the tokens of the turn before it
       continues that segment; otherwise it starts a new one. Raises
       ``ValueError`` when the trajectory has no trained step, or when the
       template does not render a turn after its own generation prompt.
   * - ``TeacherRequest``
     - The ``extra.reef.teacher`` value. Its ``teacher_tokens`` is the
       rendered teacher prompt followed by the student's reply tokens
       verbatim, or the student's own tokens when it names no messages.
   * - ``rendered_job_data(data, template)``
     - Renders each trajectory sample that has no ``tokens``; a trajectory
       of several segments becomes one sample for each segment, with
       ``segment``: ``{"index": i, "count": n}`` in its
       ``extra.reef.training``. Returns the job data and the positions of
       the samples that it leaves out. The Slime and Tinker components count
       them in the job metric ``unrenderable_samples``.

Training objective
------------------

Each training backend owns its objectives: ``SlimeObjective`` in
``reef.train.slime_backend.objectives`` and ``TinkerObjective`` in
``reef.train.tinker_backend.objectives``. An objective prepares the training
signal over a job's full data and declares the backend's loss family. A job
names it in its ``objective`` parameter, usually from the recipe's
``default_parameters()``: a registered name, or a dotted
``my_method.slime:MyObjective`` reference. The dotted reference names a class
with a zero-argument constructor; its module must be importable in both the
service and training processes.

.. code:: python

   from reef.train.slime_backend.objectives import SlimeObjective
   from reef.train.slime_backend.schedule import next_steps
   from reef.train.slime_backend.signals import TrainingSignal
   from reef.train.types import JobData, trajectories
   from reef.core.trajectories import trajectory_reward

   class MyObjective(SlimeObjective):
       name = "my_method"
       loss_family = "my_method"

       def prepare(self, data, state) -> TrainingSignal:
           samples = trajectories(data)
           steps = next_steps(state)
           return TrainingSignal(
               action="train",
               advantages=tuple(trajectory_reward(sample) for sample in samples),
               next_algorithm_state={"steps": steps},
               metrics={"steps": steps},
           )

For a short name, a method package's public half calls
``register_objective(name, "my_method.slime:MyObjective")`` from the backend's
``objectives`` module. The call records the reference and imports nothing; the
backend imports the module when a job first names the objective. The same
function accepts an instance in place of the reference. Each backend has its
own ``register_objective``, ``resolve_objective``, ``TrainingSignal``,
``next_steps`` and ``StepScheduling``; the examples here show Slime's. Core
never imports method packages. A built-in ``sft`` objective exists on each
backend, and ``slime_dataset`` on Slime.

``prepare`` runs on the complete reserved job data before optimizer or worker
partitioning. Compute group-relative statistics here. Keep this entry point
free of torch, Ray, Slime and Tinker imports. It must not mutate the job data,
execute training, or commit state. Model-dependent terms (critic-based GAE,
reference-model KL) stay in the backend implementation after the required
forward passes. Slime and Tinker use the same objective preparation; the
selected backend resolves its own loss implementation lazily.

``TrainingSignal`` carries:

- ``action``: ``train`` runs backend training; ``skip`` commits a state-only
  transition.
- ``advantages``: optional per-trajectory values, in the job data's order.
- ``next_algorithm_state``: proposed state committed through the trainer's
  existing step lifecycle.
- ``metrics``: method telemetry carried to the commit record.

Step scheduling
~~~~~~~~~~~~~~~

The training component cuts each job's data into optimizer steps. The
backend's batch size (Slime's ``--global-batch-size``, Tinker's
``batch_size``) is the rollouts in one step, or the packs in one step when the
job sets ``packing_length``. Two class attributes of the
objective set what the component may do with the job's data:

- ``rollout_unit`` is the rows that one step keeps together. ``"sample"``, the
  default, makes each sample its own rollout. ``"comparison_set"`` keeps the
  samples that share a ``group_id`` in one step.
- ``supports_multiple_epochs`` (default ``False``) says whether passes after
  the first are valid. Those passes train off-policy against log-probs
  computed once, which a clipped ratio tolerates.

A job's optional ``scheduling`` parameter is a ``StepScheduling``
(``reef.train.slime_backend.schedule`` or ``reef.train.tinker_backend.schedule``) as an object:

.. container:: config

   ``epochs`` : 1
       the passes over the job's data, a positive int

   ``shuffle`` : ``false``
       reorder the rollouts in every pass, deterministically from the data's rows

   ``remainder`` : ``"partial"``
       a last step with fewer rollouts (or packs) than the batch size: ``"partial"`` trains it as a smaller step, ``"drop"`` leaves it out, ``"error"`` refuses the job's data

   ``packing_length`` : ``null``
       pack the rollouts into packs of at most this many tokens (``training["tokens"]`` of each trajectory) before the steps are cut; one step then trains batch-size packs. A pack never splits a rollout, and a longer rollout is a pack of its own. ``shuffle`` and ``remainder`` apply to the packs.

``pack_rollouts(token_counts, packing_length)`` of each backend's ``schedule``
module is the packing: best-fit decreasing, which puts each rollout, longest
first, into the pack whose free tokens it fills most closely. The packs decide
only which rollouts share a step. The loss and the micro-batches stay the
component's.

Job parameters
~~~~~~~~~~~~~~

``WeightTrainingRecipe.parameters_by_kind()`` returns the parameters of each
training job by the kind of the job's samples: ``trajectory`` or ``task``. By
default, the parameters of both kinds are the recipe's
``default_parameters()`` with its ``job_parameters`` config over them, key by
key, such as ``scheduling`` or ``save_interval``. The parameters are part of
the job's ID, so ``parameters_by_kind()`` must return the same value at every
call. See `Write a recipe
<../developer-guide/recipe-training-service.rst#job-parameters>`__
for a recipe that trains each kind with its own method.

The trainer gives the parameters to its pipeline
(``PipelineContext.job_parameters``). When a job's data is ready, the
sample-to-job stage gives the job's parameters,
``Sample2Job.job_parameters(data)``:

- They are those of the kind of the data's samples, with the request's
  ``parameters`` (``TrainingRequest.parameters``) merged in by
  ``reef.core.values.deep_merge``: nested objects merge key by key, and every
  other value of the request wins.
- The stage refuses data that holds both kinds, and a kind that the recipe
  does not name when no request gives parameters. A kind that the recipe does
  not name never trains in ``auto``.
- Reef reads none of the parameters. It submits them to the deployment's
  training component, whose name is the job's ``backend``.

Each component parses a job's parameters into its own typed value:
``reef.train.slime_backend.parameters.SlimeJobParameters`` and
``reef.train.tinker_backend.parameters.TinkerJobParameters``. Slime and Tinker
parse when they accept a job and again when they run it. The parser raises
``ValueError``:

- for an unknown key;
- when ``objective`` is missing or does not resolve;
- when ``scheduling`` has another key or a value out of range;
- when the objective's ``validate_scheduling`` refuses the schedule:
  ``epochs > 1`` without ``supports_multiple_epochs``.

``TrainingComponent.parameters_schema(kind)`` returns the JSON Schema of those
parameters for a sample kind, or None; the scenario contract serves it.

Harness method
--------------

This section lists the harness method API. `Write a harness method
<../developer-guide/write-a-harness-method.rst>`__ explains the job, the
plugin settings (``from_config`` and ``PluginSettings``), and when a job is
skipped or fails.

.. code:: python

   from reef.evaluation.harness import TaskScorer, History, TaskCollector
   from reef.harness.adapters import get_adapter
   from reef.harness.adapters.parts import Format, Part, TreeLayout
   from reef.harness.tree import HarnessTree, TreePatch
   from reef.train.harness_backend import (
       AgentExecution, AgentMethod, AgentOutcome, Method, MethodJob, ModelCalls, Services, Trace,
       Workspace,
   )

   class MyMethod(Method):
       def run(self, job: MethodJob) -> None: ...

   class MyScorer(TaskScorer):
       def score(self, task, result) -> float: ...

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Member
     - Contract
   * - ``Method.run(job)``
     - edit ``job.workspace.harness`` into the job's candidate, or leave
       it unchanged to skip the job; say what the job did with
       ``job.workspace.write_output``
   * - ``Method.runs_agent``
     - a class attribute, ``False`` by default; ``True`` for a method that
       calls ``Services.run_agent`` or reads ``Services.agent``, which then
       gets ``evolution.agent`` and is refused at startup without it
   * - ``Method.runs_without_agent``
     - a class attribute, ``False`` by default; ``True`` for a method that
       ``runs_agent`` and also runs when the deployment configures no agent:
       it starts without ``evolution.agent`` and reads ``Services.agent`` as
       ``None``
   * - ``Method.from_config(config)``
     - builds the method from ``evolution.method.config`` alone. A method
       takes its own settings, such as a time limit, from ``config``, and
       everything that runs (a model, the coding agent, the evaluation) from
       ``job.services``; its constructor receives no service
   * - ``AgentMethod``
     - a method that is a folder of prompts (``config``: ``prompts``, the
       folder, and ``model``, ``served`` by default): ``START.md``, an
       optional ``RESUME.md``, and the agent's own tree; each job runs the
       agent once and reads ``output/summary.md``, ``notes.json`` and
       ``requires.json``
   * - ``MethodJob``
     - ``workspace``, the job's ``Workspace``, and ``services``, the job's
       ``Services``; the backend makes one per job
   * - ``Workspace``
     - ``root``, ``history_dir`` and ``step`` (1 or more); the directory
       paths ``harness``, ``material``, ``state``, ``scratch`` and ``output``;
       ``tree()``, ``write_tree(tree)``, ``traces()``, ``request()`` (a
       ``TrainingRequest`` or ``None``), ``tasks()``,
       ``evaluation_repeats()`` (the runs per task and tree, 0 when the
       deployment configured no scorer), ``host()`` (the host facts of
       ``material/host.json``: ``adapter``, ``parts``, ``config_targets`` and
       ``tasks_are_directories``), ``history()`` and ``state_files()`` (every
       file of ``state/`` as a path mapped to its text; ``ValueError`` past
       16 MiB or for a file that is not UTF-8 text);
       ``write_output(summary="", notes=None, requires=())`` writes
       ``output/summary.md`` (the commit's ``method_summary``),
       ``output/notes.json`` (a JSON object, the commit's ``method_notes``)
       and ``output/requires.json`` (``reef.core.requirements`` items, read on
       a request job only), and raises ``ValueError`` before writing when
       ``notes`` or ``requires`` is not valid
   * - ``TreeLayout``
     - the parts of an adapter's tree, from the ``evolve`` section of its ``adapter.yaml``; a method gets
       its adapter's as ``get_adapter(workspace.host()["adapter"]).layout``.
       ``parts`` maps each part name to a ``Part`` (``name``, ``path``,
       ``format``, a ``Format``, ``renders``, ``holds`` and ``review``);
       ``locate(path)`` returns the ``TreeLocation`` of a tree path
       (``part``, and ``name``, the name in the part's ``{name}``, ``""`` for
       a part without one), or ``None`` when no part takes the path
   * - ``Trace``
     - ``directory``, ``source`` (``traffic`` or ``run``), ``task``,
       ``reply``, ``score``, ``feedback``, ``failure``, ``record`` and
       ``client``; ``trajectory()`` reads the ATIF trajectory of a traffic
       trace, and returns ``None`` for a run, whose run files the
       history keeps
   * - ``Services.models``
     - a ``ModelBindings``: ``served`` and every ``evolution.models`` entry;
       ``chat(messages, *, timeout_s=None, **params)`` returns the assistant
       text, ``complete(body)`` the provider's response. Every call counts
       against ``evolution.max_model_calls_per_step`` and lands in
       ``calls.json``
   * - ``Services.calls``
     - the job's ``ModelCalls``: ``spend()`` counts one call and raises
       ``RuntimeError`` past the budget, ``record(entry)`` appends one call
       to ``calls.json``, and ``note(kind, text, *, failed=False)`` adds one
       line to the job's activity
   * - ``Services.check(tree)``
     - the ``ValidationResult`` of ``tree`` for the adapter: ``passed``,
       ``problems`` and ``message()``
   * - ``Services.evaluate(tree, tasks, *, repeats=1)``
     - ``repeats`` ``RunRecord`` values per task of the list ``tasks``
       (a single string is refused), task by task; reuses the scored
       runs the history holds; ``RuntimeError`` without a scorer, or
       when the call needs more runs than
       ``evolution.max_runs_per_step`` leaves, and then no task runs;
       ``ValueError`` for a tree that fails ``check``
   * - ``Services.agent``
     - the deployment's coding agent bound for the job's scenario, an
       ``AgentExecution`` (``descriptor``, the agent's adapter; ``executor``,
       its isolation bound to the adapter's program; ``binary``; and
       ``timeout_s``, the limit of one run), or ``None`` when no agent runs
       for the method. A method that needs its own gateway or tools starts
       the agent with it; ``run_agent`` is the usual way
   * - ``Services.run_agent(prompt, *, tree, model)``
     - one run of the deployment's coding agent in the workspace, its own
       harness ``tree`` rendered after Reef's protocol rules and its model
       calls on the binding ``model`` names, through a gateway that serves
       ``check`` and ``evaluate`` too; an ``AgentOutcome``. ``RuntimeError``
       when no agent runs for the method, ``KeyError`` for an unknown
       ``model``, ``ValueError`` for a tree that does not render
   * - ``AgentOutcome``
     - ``exit_code`` (``None`` when the agent did not exit by itself),
       ``timed_out``, ``error`` (why it could not start) and ``seconds``;
       ``succeeded`` is an exit with status 0
   * - ``History``
     - ``candidates`` (``CandidateRecord`` by tree checksum), ``runs``
       and ``decisions`` in file order; ``tree(checksum)``,
       ``runs_of(checksum)``, ``scores(checksum, tasks, *, repeats=1)``,
       ``decision(step)`` and ``run_dir(run_id)``
   * - ``CandidateRecord``
     - ``tree``, ``step``, ``origin`` (``seed``, ``method``, ``proposal`` or
       ``release``), ``parent``, ``summary`` and ``notes``
   * - ``RunRecord``
     - ``run_id``, ``tree``, ``task``, ``task_key``, ``repeat``,
       ``context``, ``step``, ``score`` (a ``RolloutScore``: a finite value, or
       failed when the environment could not run the task), ``failure``,
       ``reply``, ``residue``, ``agents``, ``path`` and ``transcript_read``
   * - ``DecisionRecord``
     - ``step``, ``tree`` (the candidate's checksum), ``head`` (the checksum
       of the head tree that the job started from), ``selected``
       (published; the activation rule decides whether it serves),
       ``policy`` and ``reason``
   * - ``TaskScorer.score(task, result)``
     - a finite float for one finished run, higher is better; ``result``
       holds ``exit_code``, ``stdout``, ``stderr``, the parsed
       ``trajectory`` and ``residue``. The evaluation calls
       ``score_with_models(task, result, models)``, whose default calls
       ``score``. A model-based judge overrides ``score_with_models`` to use
       the job's model bindings, and still implements the abstract
       ``score``, for example by raising. A scorer that needs a local GPU
       overrides ``execution_requirements()``
   * - ``TaskCollector.collect(samples)``
     - the prompts to collect from the job's trace samples, with
       ``evolution.evaluation.collect_failures``
   * - ``evaluation.metrics``
     - holds ``candidate_scores`` and ``current_scores`` under
       ``policy: compare`` and ``policy: floor``: ``RolloutScores``
       (``reef.core.evaluation``) in JSON, one entry per task in task order
       with the scores of its runs in run order, each a number or
       ``"failed"``; read them with ``RolloutScores.from_json``.
       ``current_scores`` is empty under ``floor``, which runs the candidate
       alone. Under ``policy: none``, which runs no task, it holds no
       score

``reef.evaluation.harness`` holds the publication policies that
``evolution.evaluation.policy`` names: ``ComparePolicy`` (``compare``, with
``statistic`` ``wins`` or ``mean`` and ``min_margin``), ``FloorPolicy``
(``floor``, with ``floor_score``) and ``NoEvaluationPolicy`` (``none``). A
harness recipe takes no other policy.

Candidate evaluation
--------------------

.. code:: python

   from reef import CandidateEvaluationPlugin, EvaluationResult, SelectionDecision

A harness recipe's plugin explicitly inherits ``CandidateEvaluationPlugin``,
measures a candidate tree, and decides whether the search keeps it.
``CandidateEvaluator`` and ``CandidateSelector`` remain separate abstract
capabilities; implementing only one does not make an object a plugin. Reef
enforces the fixed evaluate-then-decide order and verifies that the decision
kept the exact result it was given.

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Member
     - Contract
   * - ``evaluate(candidate)``
     - returns an ``EvaluationResult``: evaluator name, version, and a
       metrics mapping
   * - ``decide(candidate, evaluation)``
     - returns a ``SelectionDecision`` whose ``selected`` is a boolean, and
       which must carry the evaluation it was given

A rejected candidate makes no artifact, and the kept tree stays the search's
current one. An exception is fail-closed: Reef does not publish the candidate.
Make evaluations idempotent by ``candidate.candidate_id`` because recovery may
repeat work whose result was not durably committed. Whether a kept tree serves
is the activation rule's decision, not the plugin's.

The harness training component builds its plugin for each job from
``evolution.evaluation.policy`` and takes no other plugin (see `Harness
method`_).

Activation rule
---------------

.. code:: python

   from reef.core.activation import ActivationRule, MadeArtifact

Training makes artifacts and never activates one. A scenario's
``ActivationRule`` decides whether an artifact serves:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Contract
   * - ``activates(made)``
     - True when Reef activates the artifact as soon as training publishes
       it. ``made`` is a ``MadeArtifact``: the ``component`` and the
       ``changed_kinds`` of a harness tree. The committer asks when a
       training step publishes an artifact; the commit record's
       ``activated`` keeps the answer.
   * - ``activates_evaluated(artifact_id, result)``
     - True when Reef activates the artifact once an evaluation of it
       finished with the ``PolicyEvaluation`` ``result``; false by default.
       The artifact evaluation service asks when an evaluation finishes.

The built-in rules are ``reef.evaluation.activation``'s ``AlwaysActivate``,
``NeverActivate`` and ``EvaluatedActivation``. The deployment names a rule in
its ``activation`` section (`Configuration
<configuration.rst#the-activation-section>`__). A rule of your own inherits
``ActivationRule``. Reef builds it from its config with the class method
``from_config(config)``.

Scenario components
-------------------

.. code:: python

   from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree

``Recipe.build_components(scenario)`` returns the components that every
snapshot of the scenario binds, as the recipe's configuration declares them.
``WeightTrainingRecipe`` declares a ``weights`` component, whose kind is the
deployment runtime's ``served_weight_kind``, and ``HarnessEvolutionRecipe``
declares a ``harness`` component that clients pull, so most methods never
touch this. See `Scenario components <../developer-guide/components.rst>`__
for where Reef reads each field.

``ScenarioComponents`` is a frozen dataclass mapping each component name to a
``ComponentSpec``. Bare ``ScenarioComponents()`` is the complete record-only
configuration. A snapshot of one component is flat; a snapshot of several
components has one directory per component. At most one component has a
weight kind, and at most one has ``files``. The field ``validator`` of
``ScenarioComponents`` is the snapshot's own check; a snapshot of several
components has none, so there it stays ``AcceptAnyArtifact()``.

A ``ComponentSpec`` has these fields:

.. list-table::
   :header-rows: 1
   :widths: 18 28 54

   * - Field
     - Type
     - Contract
   * - ``validator``
     - ``ArtifactValidator``
     - admit an artifact of the component before it is stored; the default
       accepts any artifact
   * - ``kind``
     - ``ArtifactKind | None``
     - the kind of every artifact of the component; ``model`` and
       ``model_adapter`` load into the inference runtime
   * - ``files``
     - ``FileTree | None``
     - back client pulls
   * - ``layout``
     - ``ArtifactLayout | None``
     - check an imported artifact; ``None`` takes no import

The inference runtime loads the weight component with
``InferenceRuntime.activate_checkpoint(artifact, scenario=...)``, which returns
the runtime load ID that serves it. No component changes an inference request:
the inference routes forward each request to the engine that ``served_models``
names.

Components do not decide which records train, compute candidates, execute a
training job, or mutate the snapshot chain. A method should not add an HTTP
proxy or copy Reef's record store.

Runtime responsibilities
------------------------

``TrainingComponent`` and ``InferenceRuntime`` are independent interfaces.
Neither inherits from the other, and no trainer holds a serving engine.

``ScenarioCommitter`` publishes checkpoints and records artifact selections.
A dispatcher queues each selection for the inference worker. The worker opens
the runtime, loads the published snapshot, and publishes routes when
activation returns. Route publication and startup completion share one
transaction. A superseded request cannot publish routes. The runtime loads a
snapshot when a scenario starts, when a committed snapshot is activated, and
when a person activates one.

An ``InferenceRuntime`` implements ``inference_handler`` and the two members
marked *required*. It overrides another member only when it has the
capability. The `production inference contract
<../developer-guide/serving-runtimes.rst#production-inference-contract>`__ gives
the requirements of an integration.

These members bind weights and name routes:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Member
     - Default
     - Result and errors
   * - ``activate_checkpoint`` ``(artifact, *, scenario=None)``
     - required
     - Serves the weight component of a published snapshot. Returns its
       runtime load ID (``str``). Gets ``scenario`` only when
       ``per_scenario_adapters`` is true. Raises on failure, and promises
       no rollback.
   * - ``served_artifact`` ``(artifact_id, scenario)``
     - required
     - Returns ``ServedArtifact | None``: the engine's HTTP(S) URL and
       model name of a weight binding that already holds. Loads nothing.
       ``None`` for an unknown artifact, an unavailable or uncertain
       binding, or an engine without such a URL; the startup then fails.
   * - ``endpoint_model_name(scenario)``
     - ``None``
     - Returns the model name (``str``) for the routes of a snapshot
       without a weight component, and of each artifact served beside it.
       These routes identify no weights. ``None`` refuses such a snapshot:
       its startup fails.
   * - ``handler_for(url)``
     - ``inference_handler`` for ``base_url``, else ``None``
     - Returns the handler of the engine at ``url``, for a service that
       did not open the runtime. After ``None``, that service uses a plain
       proxy.
   * - ``open()``, ``shutdown()``
     - do nothing
     - ``open`` starts the engines before the inference worker adopts or
       activates artifacts; no other process opens the runtime.
       ``shutdown`` releases what the runtime owns.
   * - ``adopt(served)``
     - ``False``
     - Returns whether the engine still serves ``served``, a route from
       before the inference worker restarted. After ``True``, the route
       stays, and the next activation of that artifact loads nothing.
       ``False`` removes the route until serving resumes.
   * - ``stop_serving(served)``
     - does nothing
     - Releases the binding of ``served``, which no route names any more.
       The engines can stay allocated. An unknown value is no error.
   * - ``release_unreferenced_artifacts`` ``(retained)``
     - does nothing
     - Releases each binding whose URL and model name no value of
       ``retained`` has, such as one that a failed startup left.
       ``retained`` holds every published route and every pending stop.
       The inference worker calls it between two serving works, only when
       ``serving_slots`` is 1. Raises on a failed release; the next call
       tries again.

These members state what the runtime serves:

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Member
     - Default
     - Result and errors
   * - ``model_path``
     - ``""``
     - The default model name for harness bindings. Empty for a runtime
       whose requests select the model.
   * - ``served_weight_kind``
     - ``model``
     - The kind of the weights that the runtime serves: ``model`` for full
       weights, or ``model_adapter`` for a LoRA adapter on its base model.
       A runtime that serves adapters must override it.
   * - ``per_scenario_adapters``
     - ``False``
     - True for a runtime that serves each scenario its own adapter.
   * - ``shared_by_training_scenarios``
     - ``False``
     - True for a runtime that serves each training scenario its own
       weights.
   * - ``serves_any_artifact``
     - ``False``
     - True for a runtime whose handler answers each call from the
       artifact that the call names, without a load, as the Tinker runtime
       does. Such a runtime overrides ``artifact_model`` and
       ``serve_artifact``.
   * - ``artifact_model(artifact)``
     - raises ``ReefError``
     - Returns the ``PolicyModel`` that samples an artifact on request,
       without a load, for the `artifact evaluation
       <http-api.rst#evaluate-an-artifact>`__ of a model artifact.
       ``ValueError`` for an artifact that the runtime cannot serve.
   * - ``serve_artifact(artifact, *, scenario)``
     - raises ``ReefError``
     - Deploys an artifact beside the scenario's active one. Returns
       ``ServedArtifact | None`` when the engine is ready: ``None`` when
       no other process can call the engine. Consumes one artifact slot
       until ``stop_serving``. ``ReefError`` when every slot holds another
       artifact, ``ValueError`` for an artifact that the runtime cannot
       serve.
   * - ``served_artifact_capacity``
     - ``None`` when ``serves_any_artifact`` is true, else 1
     - The artifact slots: how many weight artifacts of one scenario the
       runtime keeps bound at once. A positive integer, or ``None`` for
       no limit. A capacity above 1 needs ``serve_artifact``. It does not
       enable the evaluation of a model artifact.

``Scenario.serving_artifact_id()`` names the selected artifact.
``Scenario.served_artifact_id()`` reads the published route when a dispatcher
owns the scenario. ``Scenario.served_artifacts()`` reads the artifacts that the
scenario served from the commit history; the samples service judges samples
against them. Automatic weight training waits while the selected artifact has
no matching served route. Manual training does not wait, since a request names
its samples. Startup status is available through the inference service's
``GET /reef/scenarios/{scenario}/inference/status`` route.

Saved-weight runtimes
~~~~~~~~~~~~~~~~~~~~~

``reef.inference.serving`` holds the two runtimes for saved weights, both
over an ``InferenceBackend``: ``AdapterServingRuntime``, whose snapshots
hold the adapter in ``adapter/``, and ``FullWeightServingRuntime``.
`Serving saved weights
<../developer-guide/serving-runtimes.rst#serving-saved-weights>`__ describes what
each one serves and how it recovers its engines.

Both runtimes replace weights in engines that keep running. They take the
engines as a ``ServingEngines`` value, and its ``request_pinning`` property
must return true. The default is false, and the runtimes then raise
``RuntimeContractError`` when they are built. Return true only for a receiver
that enforces the rules in the property's description (see `Request pinning
<../developer-guide/serving-runtimes.rst#request-pinning>`__).

Each runtime passes a snapshot's directory as a ``WeightFiles`` value, with
the ``"adapter"`` or ``"full"`` layout, to
``InferenceBackend.load_weight_files``, under a new runtime load ID that every
engine then reports. Each checks the health of its engines every
``health_interval_s`` seconds (default 30), and calls ``recover_engines``
after a failed check. Before its first successful activation, a
``FullWeightServingRuntime`` has nothing to restore: a failed check only
replaces the engines, and a call of ``recover_engines`` replaces them and then
raises ``ReefError``.

Each runtime answers a base root (``base.json``, see `Artifact imports`_)
with the base model and asks ``ArtifactWeights`` for nothing:

* ``AdapterServingRuntime`` loads nothing. The scenario serves no adapter,
  and ``activate_checkpoint`` returns ``base_version``. The base root's route
  names the base model under that load ID, which the engines' receiver
  admits. No route names the artifact of the adapter that the scenario served
  before. That adapter stays resident until its route stops or another
  adapter needs its slot.
* ``FullWeightServingRuntime`` loads the directory ``model_path`` as full
  weights under a new version. It raises ``ReefError`` when ``model_path`` is
  no directory on its host.
* ``TinkerInferenceRuntime`` samples the base model. It raises ``ValueError``
  for a base root of another model or another service.

A root snapshot that an earlier Reef made from the default start has the
layout of every other snapshot of its component. An adapter runtime loads its
untrained adapter, and the Tinker runtime samples its untrained adapter.

Factories and modules
~~~~~~~~~~~~~~~~~~~~~

A weight recipe holds neither the ``TrainingComponent`` nor the
``InferenceRuntime``. The process holds both beside the recipe, in
``LiveObjects`` (`Live objects`_).

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Name
     - Role
   * - ``TrainingComponentFactory``, ``build_training_component``
       (``reef.train.component``)
     - Define and build the ``TrainingComponent``. A training deployment's
       component factory returns the component alone.
   * - ``TrainingComponentFactory.facts``, ``training_backend_facts``
     - State a backend's ``TrainingBackendFacts`` from the same config
       without a component: its name, artifact format, default start and
       its inputs, parameters schema, root start check, and whether
       scenarios and jobs share it. Every factory defines ``facts``.
   * - ``RuntimeFactory``, ``RuntimeRegistry``
       (``reef.inference.registry``)
     - Define and build the ``InferenceRuntime``.
   * - ``reef.service.deploy.serving.serving_runtime_config``
     - Describes the serving runtime from the ``inference`` settings and the
       backend's ``artifact_format``.

The inference service of a split deployment reads the ``artifact_format`` of
the facts and builds no component. Control and the training service hold the
facts, and control writes a new scenario's root from
``facts.default_start()``.

Two training scenarios can share a deployment only if its training component
allows it (``shares_scenarios``) and its runtime serves each scenario its own
weights. The Tinker component allows it, because each job carries its own
start. The Slime component does not: its workers hold one set of weights, so
a Slime deployment trains one scenario.

The inference contracts are the modules at the root of ``reef.inference``,
and ``reef.train.component`` holds the training component contract. Values
that both sides read are in ``reef.core.weight_layout`` (``ArtifactFormat``,
``WeightFiles``, ``ArtifactWeights``). See the `codebase map
<../contributing/codebase-structure.rst>`__.

Concrete integrations live with their owners: ``SlimeTrainingComponent`` in
``reef.train.slime_backend.component``, ``TinkerTrainingComponent`` in
``reef.train.tinker_backend.component``, and ``SGLangInferenceBackend`` in
``reef.inference.sglang.backend``.

Training jobs and components
----------------------------

``reef.train.component`` defines bounded training jobs.

Training jobs
~~~~~~~~~~~~~

A ``TrainingJob`` has these fields:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Value
   * - ``job_id``, ``backend``
     - The job's ID and the name of the training component.
   * - ``start``
     - The non-empty start: one that the component made, such as
       ``SavedCheckpoint.path``, or the component's start of a published
       snapshot. A new trainer starts at ``TrainingComponent.root_start``.
   * - ``data``
     - A ``JobDataRef``: a reference to the job data directory that the job
       trains. ``data_uri`` is its URI, and ``data_id``, ``sample_count``
       and ``kind`` describe the data; these copy nothing. ``read_data()``
       returns its samples and ``data_dir`` is its local path; with a remote
       store, the first read copies the directory, and the directories of
       its tasks, to the host.
   * - ``parameters``
     - The backend's parameters. Reef reads none of them; they override the
       backend's own configuration.
   * - ``scenario``
     - The scenario whose trainer submits the job. The jobs of one scenario
       run one at a time, in the order of their submission, unless the
       component has ``concurrent_jobs``.
   * - ``request_id``
     - The ID of the training request that the job trains, or ``None``.
   * - ``component``
     - The scenario component that the job trains, or ``None``.
   * - ``previous_job_id``
     - The last job whose end the trainer committed, or ``None`` before the
       first. A harness continues its state after that job, since a job
       that keeps no tree still advances it; a weight component ignores it.
   * - ``train_from_artifact_id``
     - The artifact that ``start`` came from, or ``None``. Reef's job
       listing reads it; a component does not need it.

``reef.train.jobs.job_id_for`` names a job by a hash of its backend, its
start, its data and its parameters. The weight trainer puts the scenario and
the records of the job's data into the data identity. A retry of the same
request is therefore the same job.

A ``JobStatus`` has ``state`` (``running``, ``finished`` or ``failed``), the
``SavedCheckpoint`` values so far in step order, the backend's ``metrics``,
and ``error``, set exactly when the job failed. A finished job without
checkpoints trained nothing, for example because its objective skipped the
job's data.

A ``SavedCheckpoint`` has a ``step`` within its job, a ``path`` that a later
job starts from, an ``artifact_path`` that Reef publishes and serving loads,
and the backend's ``metrics``. Both paths are local directories of the process
that runs the component, and the component does not change ``artifact_path``
after it reports the checkpoint. The training worker stores the files of
``artifact_path`` in the artifact backend (``RepositoryBackend.store_content``)
and writes the checkpoint into the jobs queue with the name of that
``content`` in place of ``artifact_path``.

Training components
~~~~~~~~~~~~~~~~~~~

A ``TrainingComponent`` runs whole jobs with its framework's own training
loop. It implements the members marked *required*:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Contract
   * - ``name`` (required)
     - The backend name that the component's jobs carry.
   * - ``artifact_format`` (required)
     - An ``ArtifactFormat``: the weight files that each saved checkpoint's
       snapshot holds, ``layout`` ``full`` (a Hugging Face checkpoint) or
       ``adapter`` (a PEFT adapter in ``adapter/``, with ``adapter_rank`` and
       ``adapter_target_modules``). ``None`` when the snapshots hold no weight
       files that an engine loads, such as Tinker's manifests. The serving
       assembly configures the engines from it and refuses a serving runtime
       that cannot load it.
   * - ``submit(job)`` (required)
     - Record the job durably and return; the job runs in the background.
       Submitting a job ID again does nothing. Raise ``ValueError`` for a
       job that the component refuses and ``RuntimeError`` after ``close``;
       nothing is recorded then.
   * - ``status(job_id)`` (required)
     - The job's ``JobStatus``. Raise ``KeyError`` for a job the component
       never saw. A restarted component resumes each unfinished job or
       reports it failed.
   * - ``cancel(job_id)`` (required)
     - Stop a running job, which then reports ``failed``. Nothing for an
       unknown or ended job.
   * - ``import_start(artifact_dir)`` (required)
     - Copy a published snapshot into the component's own storage and return
       it as a start. The training service calls it when it takes a job
       whose start names a snapshot, such as the first job after a
       ``train-from``. Raise ``ValueError`` when no job can start from those
       weights.
   * - ``root_start(root)``
     - The start of the first job of a trainer without committed state, a
       non-empty string. ``root`` is a ``ScenarioRoot``: the trainer's
       component of the scenario's root snapshot and the root's
       ``RootSource``. The default is ``import_start`` of the root's files.
       A root of a new scenario holds the facts' default start, and the job
       then starts at the base model. Tinker starts a new LoRA from its
       ``seed`` for a base root; with ``seed: null`` it saves one untrained
       adapter in ``state_dir`` when the first such job runs, and each such
       job restores it. Slime, which imports no snapshot, returns its startup
       weights when the root's source is its default start with equal
       ``inputs``, and raises ``ValueError`` naming both otherwise.
   * - ``snapshot_start(artifact)``, ``start_files``
     - ``snapshot_start`` is the start of a job that trains from the
       trainer's component of a published snapshot; the default is
       ``import_start`` of its directory. ``start_files`` are the only
       paths that ``root_start`` and ``snapshot_start`` read, or ``None``,
       the default, for the whole component.
   * - ``shares_scenarios``, ``concurrent_jobs``
     - Whether jobs of several scenarios can run on the component, and
       whether several jobs of one scenario can run at the same time. Both
       default to ``False``. Return true only when a job depends on nothing
       but its own ``start``, data and parameters.
   * - ``check_job(job)``
     - Raise ``ValueError`` for a job that the component cannot train. The
       training service calls it before control adds the job to the queue.
       The default accepts every job.
   * - ``check_task_jobs(parameters)``
     - Raise ``ValueError`` unless the component can train jobs of
       ``TaskSample`` values with ``parameters``. The training service calls
       it at its start for a recipe whose jobs train tasks. The default
       refuses.
   * - ``parameters_schema(kind)``
     - The JSON Schema of a job's parameters for a sample kind, or ``None``,
       the default (`Training objective`_).
   * - ``check_remote_artifact(names)``
     - Raise ``ValueError`` unless what ``names`` names resolves on the
       component's service. ``names`` is what the facts' ``check_artifact``
       returned for an import. Tinker looks up the manifest's ``state_path``
       on its service. The Tinker API describes no sampler or
       exported-adapter path, so those are checked only for their
       ``tinker://`` form.
   * - ``changed_parts(start, checkpoint)``
     - The parts of the scenario component that the checkpoint changed from
       the job's start, empty by default. The scenario's activation rule
       reads them as ``changed_kinds``.
   * - ``job_progress(job_id)``, ``job_records(job_id)``
     - Where a running job stands, as a JSON object, and a local directory
       of the files that record how an ended job ran. Both default to
       ``None``. The training worker copies them to where control reads
       them.
   * - ``archive_scenario(scenario)``
     - Move aside what the component keeps for a deleted scenario alone. The
       default keeps nothing for one scenario.
   * - ``start_jobs()``
     - Start running jobs in this process: the training service's job worker
       calls it once. No other process builds the component.
       The default does nothing.
   * - ``close()``
     - Release the component's resources; safe to call repeatedly.

The backend's ``TrainingBackendFacts`` check an import's files with
``check_artifact(files)``. The default reads ``artifact_format``: ``full``
needs ``config.json`` and weight files, ``adapter`` needs a PEFT adapter in
``adapter/`` with a rank of at most ``adapter_rank`` and target modules within
``adapter_target_modules``. ``None`` refuses. Tinker checks its manifest and
returns it for ``check_remote_artifact``. The default and Tinker first accept a
base root whose ``inputs`` equal ``default_start_inputs``, and raise
``ValueError`` that names each input that differs.

``reef.train.local_jobs.LocalJobComponent`` is the base of the built-in
components. It keeps one directory per job (``job.json``,
``checkpoints.json``, and ``result.json`` once the job ended). It runs one job
at a time in a worker thread, and it resumes unfinished jobs after a restart.
A subclass implements ``check_job(job)`` and ``run(job, directory)``. It must
call ``start_jobs()`` once it can train. Until then no job runs, including
the jobs that a restart left.

The job loop
~~~~~~~~~~~~

``reef.train.jobs.JobLoop`` is the control plane's side. It holds the
trainer's one training component, and weight training and harness evolution
share it. It returns the result that the trainer commits:

1. When job data is ready, the trainer reserves it. The pipeline's
   sample-to-job stage gives the job's parameters, the loop submits the job,
   and the trainer commits the submission, which consumes the records of the
   job's data.
2. While the job runs, the trainer commits each new saved checkpoint: its
   stored ``content`` becomes a new artifact, with the checkpoint's training
   metrics. The backend makes the snapshot of the stored files
   (``RepositoryBackend.publish_stored``), so control reads none of them. The
   scenario's activation rule decides whether it serves.
3. When the job ends, the trainer commits its outcome.

What a job needs beside its data comes from its stage's ``job_parameters``:
the harness stage (``TraceJobData``) adds the training request and the task
paths of the scenario's eval split.

Training is linear: the next job starts from the newest saved checkpoint,
whether or not it serves. A restarted job replaces the trainer's start
with its fresh start. A job continues the objective state of its start only
when it trains the same objective. The trainer commits the loop's state as its
algorithm state. This ``JobLoopState`` has four fields:

- ``start``: the training start as the component reads it, a non-empty
  string.
- ``jobs``: the running jobs as ``RunningJob`` values, in the order of their
  submission.
- ``finished_jobs``: the number of jobs that ended, finished or failed.
- ``last_job``: the outcome of the last job that ended, or ``None`` before
  the first one.

A failed job ends like a finished one. Reef consumed the records of its data when
it submitted the job, so they do not train again.

Once the commit of a job's end is durable, the samples service calls
``Record2Sample.job_ended(outcome)`` with ``last_job``. A trainer that starts
with no running job calls it again for the last job that ended, before it
reads a record. A stage whose job data holds ``TaskSample`` values reads each
task's rollout results from the outcome's ``metrics``: the Tinker and Slime
components report ``rollout_rewards``, the rewards of each task's rollouts, by
task name. A component that cannot roll out tasks refuses such a job in
``check_job``.

For a ``train-from``, control first asks the backend's facts
(``check_import_start``) and cancels the running job. It then writes the
``train_from`` commit, and the training start names the snapshot. The training
service imports it with ``import_start`` when it takes the next job. A backend
that cannot load a snapshot raises ``ValueError`` in the facts, and Reef
refuses the request with HTTP 400.

Tinker integration
------------------

See `Train with Tinker <../user-guide/tinker.rst>`__ for the supported
contracts.

``reef.train.tinker_backend.launch.TinkerDeployment`` implements the optional
``tinker`` backend as an ``InProcessTrainingDeployment``. Its
``component_factory`` builds a ``TinkerTrainingComponent`` from
``TinkerSettings``, only after deployment selection. Each job trains remotely
from the job's start manifest. A saved checkpoint is both the start of a later
job and the snapshot Reef publishes. Train-from works: the component imports
the snapshot's manifest as a start. A job takes these parameters:

- ``objective``;
- optional ``learning_rate_schedule`` and ``scheduling``;
- optional ``batch_size``, ``kl_coef`` and ``learning_rate`` overrides;
- ``save_interval``, the optimizer steps between saved checkpoints (by
  default, one checkpoint when the job ends);
- ``rollout``, the rollouts of each task, for a job of Harbor tasks.

``TrainingDeployment`` (``reef.train.deployment``) is a training backend's
deployment definition:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Member
     - Contract
   * - ``default_serving``
     - The serving of a deployment that names none and sets no engine
       setting: ``engines`` unless the backend says otherwise. Tinker's is
       ``trainer``.
   * - ``trainer_serving_config(settings)``
     - The runtime config of the backend's own serving, or ``None`` without
       one. Tinker's selects ``reef.inference.tinker.TinkerInferenceRuntime``
       (runtime kind ``tinker``), which reads each snapshot's
       ``tinker-checkpoint.json`` manifest and samples from the immutable
       sampler it names. A snapshot holds that manifest alone, so the
       component's ``artifact_format`` is ``None``.
   * - ``engine_weights(settings)``
     - An ``EngineWeights`` for a backend whose snapshots hold no weight
       files: the ``ArtifactFormat`` that Reef's engines load and the
       ``ArtifactWeights`` plugin (``reef.core.weight_layout``) that gives
       each snapshot's files. The default is ``None``: engines cannot serve
       such snapshots. ``ArtifactWeightFiles``, the default plugin, reads a
       snapshot's own files. Tinker names
       ``reef.tinker.weights.TinkerSamplerWeights``, which downloads the
       adapter at each snapshot's ``sampler_path`` and converts it into a
       PEFT adapter, so Reef's engines can serve it (``inference.serving:
       engines``).
   * - ``requires_local_model``
     - True by default. Hosted integrations set it to false to preserve
       remote model identifiers during deployment resolution.

Methods implement ``TinkerLoss`` from ``reef.train.tinker_backend.losses``
with a ``loss_family`` name, and decorate the class with ``@register_loss_family``
from ``reef.train.tinker_backend.loss_families``, in the module that defines
their ``TinkerObjective``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Contract
   * - ``loss_fn``
     - The Tinker built-in loss the family trains with.
   * - ``inputs(rows, base_logprobs, kl_coef=...)``
     - That function's token-aligned ``loss_fn_inputs``, one dictionary per
       row. For ``importance_sampling``, ``TokenRow.inputs`` builds the
       dictionary from response-token advantages, shifting prediction
       targets by one token and padding prompt positions with zeros.
   * - ``needs_base_logprobs``
     - Requests frozen-base scores for a nonzero KL coefficient.
   * - ``check_kl_coef(kl_coef)``
     - Raises ``ValueError`` for a coefficient the family cannot train. Reef
       calls it when it prepares a job, before any remote work. The default
       accepts every coefficient.
   * - ``TinkerCustomLoss.loss(data, logprobs)``
     - The SDK boundary passes it to ``forward_backward_custom``, so the
       family computes its loss tensor on the Reef host. ``Datum`` objects
       are constructed only after this shaping.
