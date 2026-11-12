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

+---------------------------------------------+--------------------------------------------------+------------------------------+
| Need                                        | Component                                        | Required for                 |
+=============================================+==================================================+==============================+
| configure deployment behavior               | `Recipe <#recipe>`__                             | every deployment             |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| validate feedback at ingress                | `Report <#report>`__                             | methods whose signal arrives |
|                                             |                                                  | in reports                   |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| turn records into typed job data            | `Pipeline stages <#pipeline-stages>`__           | every method producing       |
|                                             |                                                  | updates                      |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| carry data to the backend                   | `Job data <#job-data>`__                         | every method producing       |
|                                             |                                                  | updates; subclass only for a |
|                                             |                                                  | new shape                    |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| turn job data into a signal                 | `Training objective <#training-objective>`__     | weight-training methods      |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| declare a new snapshot component            | `Scenario components <#scenario-components>`__   | only a new evolution medium  |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| evolve a harness tree and score a task run  | `Harness method <#harness-method>`__             | harness-evolution methods    |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| evaluate a produced candidate               | `Candidate evaluation <#candidate-evaluation>`__ | optional, any recipe         |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| a new tensor objective                      | `Loss family                                     | rarely                       |
|                                             | <../developer-guide/loss-families.rst>`__        |                              |
+---------------------------------------------+--------------------------------------------------+------------------------------+

Method code should depend only on what this page documents. Anything else under
``reef.`` is an implementation detail and may change.

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

   from reef.inference.interfaces import InferenceBackend, InferenceRuntime
   from reef.train.component import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob

A ``TrainingComponent`` runs bounded training jobs (see `Training jobs and
components`_). An ``InferenceRuntime`` executes requests, owns admission, and
loads published snapshots into its engines (see `Runtime responsibilities`_).
``InferenceBackend`` controls the serving engines of one deployment: it loads
``WeightFiles`` into every engine (``load_weight_files``), pauses and resumes
generation, unloads adapters, and reports the runtime load IDs that the engines
serve. ``InferenceEngine`` names the operations one engine actor offers, which
``ReefSGLangEngine`` and ``ReefVLLMEngine`` implement. The engine supervision
hook an inference integration also implements (``InferenceEngineGroup``) lives
in ``reef/inference/recovery.py``, next to ``InferenceControl``, which drives it.
Implementations inherit the corresponding interface and provide every abstract
operation. SGLang's ``SGLangInferenceBackend`` inherits ``InferenceBackend``
through ``EngineInferenceBackend``. ``InferenceService`` in
``reef/inference/deployment.py`` owns engine start and shutdown. It also requires
explicit inheritance; inference factory discovery rejects objects that only
happen to expose similarly named methods.

Two other extension points have separate names and purposes:

- ``reef.train.jobs.JobLoop`` is how a ``Trainer`` trains: ``Trainer.build``
  takes it as ``job_loop`` (see `Training jobs and components`_), and a
  trainer without one trains nothing. Every training step is a job of a
  training component; the harness recipe's ``HarnessTrainingComponent``
  (``reef.train.harness_backend.component``) runs each harness job on a
  ``HarnessBackend``, in the training service.
  The trainer's job size is its sample-to-job stage's ``job_samples``.
- ``reef.inference.interfaces.InferenceHandler`` executes one buffered or streaming
  request. Its ``artifact`` argument is an ``InferenceArtifact``: the component
  ``artifact_id`` and an optional ``local_path``.
  Handlers receive no repository or snapshot locator. Runtime and recipe objects
  expose the handler as ``inference_handler``;
  ``create_app`` and ``build_app`` accept the same keyword for an injected
  handler (see `Embed the service`_).

The deployment selects native backends with ``training.backend`` and ``inference.backend``.
Custom request adapters use ``inference.handler-factory`` and
``inference.handler-config``. The factory setting names an ``InferenceHandler``
subclass as ``package.module:ClassName``. Reef calls its ``from_config(upstream_url, *, model_path, timeout_s,
**config)`` class method; arbitrary functions are not accepted. Handlers injected
directly into ``create_app`` or ``build_app`` need only implement the inference methods.

A scenario's `serving config <http-api.rst#serving-config>`__ reaches three
methods. Each has a default for a handler or runtime without serving fields:

- ``InferenceHandler.serving_config(override)`` and
  ``InferenceRuntime.serving_config(override)`` return the effective value of
  each serving field under one scenario's ``override``, and raise ``ValueError``
  for a field or value that they do not take. Control calls the runtime's method
  to check an update. The runtime's default gives its handler's fields.
- ``InferenceHandler.request_payload(payload, *, serving_config)`` gets the
  scenario's override with each request. It applies its own fields and leaves
  the rest.
- ``InferenceRuntime.apply_serving_config(scenario, override)`` applies the
  engine fields. The inference worker calls it before each activation, before
  each ``serve_artifact(artifact, scenario=...)``, and when only the override
  changed. After it, the worker reads the scenario's route again with
  ``served_artifact``, so the runtime may move the scenario to another engine.

An ``InferenceHandler`` whose engines scale to zero can report the state of
the engine behind a route with ``engine_state(served)``: ``"warm"``,
``"starting"``, ``"idle"`` or None (RFC #1269). The startup status of a ready
artifact returns it as ``engine``. It must not start an engine. While an
engine is ``"starting"``, ``engine_detail(served)`` can give one short line
for a person, which the status returns as ``engine_detail``, and
``engine_steps(served)`` its ``start`` and ``load`` phases as
``reef.inference.interfaces.StartupSteps`` (RFC #1283).

While an ``InferenceRuntime`` serves an artifact, it can report the phase of
its startup with ``reef.inference.startup.report_startup(phase, detail,
estimate_seconds=...)``: ``prepare`` (deploy, convert or quantize), ``start``
(a machine and the weights) or ``load`` (GPU memory, kernels, warm-up). The
serving worker adds each report to the serving work's progress for that
artifact, and the startup status gives them as ``startup``. Report when the
phase or its detail changes. Outside a serving work, the call does nothing.

A runtime that serves several models declares a serving field that chooses the
model of each artifact (`Serving settings <http-api.rst#serving-settings>`__).
It reads an artifact's training starts with
``reef.artifact.artifact.training_starts_of``, nearest first, so that a training
artifact can use the setting of its start.

Streaming handlers return ``InferenceStream``; its ``record_response`` and
``record_response_pending`` fields carry buffered or deferred recording state.
Implementations and test doubles must initialize these fields through the
stream constructor. SGLang handler subclasses that preserve raw reasoning tags
can set ``SPLIT_REASONING = False`` (renamed from ``_SPLIT_REASONING``).

Python extension contracts use abstract base classes and explicit inheritance.
This also applies to repository factories, candidate evaluation plugins, harness
plugins, and component file trees. Implement optional capabilities only when
supported: adapter weight residency, artifact activation, and request leases
remain separate interfaces. CI rejects ``typing.Protocol``,
``typing_extensions.Protocol``, and ``runtime_checkable`` in all first-party
Python files, including scripts and tutorials; these findings cannot be baselined.
Third-party, generated result and golden fixture exclusions follow the
contribution policy.

Recipe
------

.. code:: python

   from reef.recipe import Recipe, WeightTrainingRecipe, config_field
   from reef.recipe.harness import HarnessEvolutionRecipe
   from reef.recipe.sft import SFTRecipe

A recipe is one frozen dataclass configuring the serving and evolution behavior
for every scenario in a deployment.

.. code:: text

   Recipe                       record-only by default   reef.recipe
   ├── WeightTrainingRecipe     training component, inference runtime, objective
   │   ├── SFTRecipe                                        reef.recipe.sft
   │   ├── SAORecipe                                        recipes.sao.recipe
   │   ├── TTTDRecipe                                       recipes.tttd.recipe
   │   ├── SDFTRecipe                                       recipes.sdft.recipe
   │   └── OpenClawRLRecipe                                 recipes.openclawrl.recipe
   └── HarnessEvolutionRecipe   harness tree + task runs reef.recipe.harness
       └── ReefineRecipe (beta)                           recipes.beta.reefine

A harness method is a ``Method`` that ``HarnessEvolutionRecipe`` runs, not a
recipe: GEPA, Meta-Harness, ACE and ReasoningBank name the base recipe and set
``evolution.method``. The beta Reefine subclasses it only to set its defaults.

``HarnessEvolutionRecipe`` holds configuration and builds nothing that runs:

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

The recipe holds no runtime, no served endpoint and no training component.
The process holds them in ``LiveObjects`` (see `Live objects`_), and
``reef.scenario.live_objects.harness_models(recipe, live)`` gives the models
of the runs: the runtime as the model under test, and the recipe's named
models.

The service builds the ``HarnessTrainingComponent``
(``reef.service.assembly.with_harness_components`` calls
``HarnessTrainingComponent.from_settings`` and returns the ``LiveObjects``
that hold the component). The component builds the
``Method``, the ``TaskScorer``, the ``TaskCollector``, the run executor and
the agent settings from the sections, then one ``HarnessBackend`` with its
evaluator for each scenario component, and the selection policy for each job.
Code that holds these objects already, such as an example driver, calls
``HarnessTrainingComponent(settings, models, method=..., executor=...,
scorer=..., collector=..., agent=...)`` and gives the dispatcher
``LiveObjects(runtime=runtime, harness_component=component)``.

Choose the narrowest class whose assumptions all hold. Inheriting ``Recipe``
starts without the extra contracts of a specialized base; the trainer of such a
recipe only reads records.

+---------------------------------+------------------------------------------------------+
| What you are building           | Where to start                                       |
+=================================+======================================================+
| record traffic, no updates      | ``Recipe`` as-is                                     |
+---------------------------------+------------------------------------------------------+
| train and publish weights       | subclass ``WeightTrainingRecipe``                    |
+---------------------------------+------------------------------------------------------+
| use Reef's harness loop         | configure ``HarnessEvolutionRecipe`` with a          |
|                                 | ``Method`` in ``evolution.method``; subclass it only |
|                                 | when the method changes more than the tree, such as  |
|                                 | the files a client pulls                             |
+---------------------------------+------------------------------------------------------+
| evolve a different artifact     | subclass ``WeightTrainingRecipe``, deploy it with a  |
|                                 | ``TrainingComponent`` that makes the artifact, and   |
|                                 | override ``build_components()``                      |
+---------------------------------+------------------------------------------------------+
| serve an externally produced    | subclass ``Recipe``, override ``build_components()`` |
| artifact                        | only                                                 |
+---------------------------------+------------------------------------------------------+
| evolve weights and a harness    | configure ``CompositeRecipe`` with one recipe per    |
| (or configuration) together     | component; each trainer runs as its own worker       |
+---------------------------------+------------------------------------------------------+

Common members
~~~~~~~~~~~~~~

+---------------------------------------------------+-----------------------------+--------------------------------+
| Member                                            | Type                        | Contract                       |
+===================================================+=============================+================================+
| ``name``                                          | ``str``                     | instance field; the default    |
|                                                   |                             | registry key                   |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``stages()``                                      | ``PipelineSettings``        | the two stage classes and      |
|                                                   |                             | their configs: the deployment's|
|                                                   |                             | ``pipeline`` section over      |
|                                                   |                             | ``pipeline_spec()``            |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``build_components(scenario)``                    | ``ScenarioComponents``      | the components that the        |
|                                                   |                             | configuration declares for one |
|                                                   |                             | named scenario's snapshots,    |
|                                                   |                             | each with its kind and         |
|                                                   |                             | admission check (default       |
|                                                   |                             | ``AcceptAnyArtifact()``)       |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``default_start()``                               | ``DefaultStart``            | the content of a new scenario's|
|                                                   |                             | root snapshot that the         |
|                                                   |                             | configuration gives; empty by  |
|                                                   |                             | default                        |
+---------------------------------------------------+-----------------------------+--------------------------------+

A recipe has no runtime, no training component and no store, and it builds
none. Every member is a function of the recipe's configuration.

Live objects
~~~~~~~~~~~~

.. code:: python

   from reef.scenario.live_objects import LiveObjects, ServedEndpoint

The process that runs a service builds the objects that run the recipe, and
holds them in one frozen ``LiveObjects`` value. The service assembly builds it
(``reef.service.assembly.load_deployment`` returns the recipe and the value),
and ``Dispatcher(recipe, backend_factory, live=live, ...)`` receives it. The
inference service calls ``load_serving_deployment``, which builds no training
component for a backend that states its facts.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Value
   * - ``runtime``
     - the serving ``InferenceRuntime``, or ``None``
   * - ``training_component``
     - the ``TrainingComponent`` that runs the jobs of the recipe that trains
       weights, or ``None``; a ``WeightTrainingRecipe`` needs the runtime,
       and this component or ``training_facts`` (``check_recipe_objects``
       raises ``RecipeConfigError``)
   * - ``training_facts``
     - the ``TrainingBackendFacts`` of that recipe's backend, in a process
       that built them in place of the component, or ``None``; at most one of
       ``training_component`` and ``training_facts`` is set
   * - ``harness_component``
     - the training component that runs the jobs of the harness recipe, or
       ``None``
   * - ``data_store``
     - the deployment's ``DataStore``, or ``None``
   * - ``served_endpoint``
     - a ``ServedEndpoint(url, token)`` where this Reef answers inference
       itself, or ``None``

``dispatcher.live`` holds the value as the trainers see it: each job
component is the jobs queue's view of it. Read the serving runtime of a
dispatcher as ``dispatcher.live.runtime``.

The same module holds the functions that need a recipe and a live object:

- ``scenario_components(recipe, live, scenario)`` gives the components of a
  deployment's scenario: what the configuration declares
  (``declared_components(recipe, scenario)``), with the kind
  of the weight component from the runtime (``served_weight_kind``) and its
  layout from the training component (``TrainingArtifactLayout``). A process
  that holds ``training_facts`` gets a ``TrainingFactsLayout``: it answers
  ``resolves_references`` and checks no import. The
  declaration is ``recipe.build_components(scenario)``; for a composite, it is
  each child's one declared component under the child's component name.
- ``default_start(recipe, live)`` gives the root content of a new scenario:
  the training component's untrained checkpoint for a weight recipe, each
  component's start under its directory for a composite, and
  ``recipe.default_start()`` for any other recipe.
- ``check_root_start(recipe, live, root)`` checks an item start; see below.
- ``harness_models(recipe, live)`` and
  ``harness_served_model(recipe, live, scenario, component)`` give the models
  of a harness recipe's runs.

A recipe builds no trainer. The scenario factory reads a ``TrainerSpec`` from
the recipe and the live objects
(``reef.scenario.factory.read_trainer_specs``): the recipe's ``stages()``,
``report_type``, ``training_mode`` and, for a ``WeightTrainingRecipe``, its
``parameters_by_kind()``, with the training component of ``LiveObjects``. A ``CompositeRecipe`` gives
one spec for each component, with the scenario's report contract
(``scenario_report_type``, below). The factory's ``TrainerBuilder``
(``reef.train.trainer_builder``) then builds one trainer for each spec with
``Trainer.build``, and binds it to its snapshot component. A recipe that is
not a composite has one trainer, so its scenario serves at most one component.

``check_root_start(recipe, live, root)`` checks an item start before scenario
registration. ``root`` is the scenario's ``ScenarioRoot``, with its artifact
and source. Each recipe of the deployment has its root checked by the
component that runs its jobs: the function calls ``root_start`` of the
training component in ``live`` for a weight recipe, and of the harness
component in ``live`` for a harness recipe. A recipe of a ``CompositeRecipe``
has its own component's view of the root checked, the same view as trainer
construction uses. The function checks nothing for a recipe that submits no
job, and for a component that ``live`` does not hold. A refusal raises
``ValueError`` and leaves the scenario unregistered.

A ``CompositeRecipe`` is a mapping from component name to child recipe, and
it computes nothing from the children. Its ``__post_init__`` checks the
mapping: at least two entries, valid component names, no nested composite,
one ``training_mode``, at most one ``WeightTrainingRecipe`` and at most one
``HarnessEvolutionRecipe``. Code that holds the recipe of a deployment reads
its recipes with ``reef.recipe.composite.component_recipes(recipe)``: the
pairs ``(component name, child)`` of a composite, and the one pair
``(None, recipe)`` of any other recipe. These functions answer for a
deployment from its configuration alone:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Function
     - Result
   * - ``reef.recipe.deployment.harness_recipe(recipe)``
     - the deployment's one harness recipe with its component name, or
       ``None``
   * - ``reef.recipe.deployment.harness_adapter(recipe)``
     - the adapter that ``GET /reef/harness`` renders the harness artifacts
       for (``evolution.adapter`` of the harness recipe), or ``None``; the
       scenario list names it for every scenario
   * - ``reef.recipe.deployment.harness_state_dir(recipe, scenario, component)``
     - the directory of the component's harness evolution state, or ``None``
   * - ``reef.recipe.deployment.scenario_state_dirs(recipe, scenario)``
     - the directories that a scenario delete archives
   * - ``reef.scenario.factory.scenario_report_type(recipe)``
     - the report contract that ingress checks: the one that the recipes
       share, or a contract that a report meets when any recipe's parses it
       (``reef.core.reports.shared_report_type``)
   * - ``reef.scenario.factory.with_record2sample_configs(recipe, components, configs)``
     - the recipe with a scenario's own record-to-sample keys, by component

Every recipe other than a composite may declare ``report_type``, the
``ReportBase`` subclass its reports parse as (``None`` keeps ingress open).
Every recipe declares
``pipeline_spec()``, the default class of each pipeline stage (see `Pipeline
stages <#pipeline-stages>`__). Weight-training recipes add these members:

- ``default_parameters()`` gives the recipe's own parameters of each training
  job by the kind of its samples, normally the ``objective``.
- ``parameters_by_kind()`` returns them with the ``job_parameters`` config over
  them (see `Training objective`_).

Every recipe's ``activation_rule()`` returns the rule that decides whether an
artifact that its trainers make serves; the deployment's ``activation``
section names it (`Activation rule`_).

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

``frozen=True`` is required by the base. ``kw_only=True`` keeps later fields
keyword-only while the training component stays the positional dependency. Call
``super().__post_init__()`` first when adding validation.

Configuration
~~~~~~~~~~~~~

``config_field()`` is a ``dataclasses.field`` carrying a default, a type-aware
parser, and an optional environment fallback:

.. code:: python

   clip_ratio: float = config_field(0.2, env="REEF_MY_METHOD_CLIP_RATIO")

Precedence is explicit configuration, then environment, then the default.
A recipe class does not load itself: ``reef.recipe.loading.load_recipe(recipe_class, config, environ)``
reads the declarations of the class and builds the recipe, and
``build_recipe(reference, environ, config)`` does the same for a class
reference. The declarations are the ``config_field`` fields, the
``section_field`` fields (one parsed config section each), the
``reader_field`` fields (each read by its ``ConfigReader``),
``section_defaults`` (the class's defaults for its sections, under the
deployment's keys) and ``refused_sections`` (the config sections that the
class refuses, each with its error message). A recipe class has no method
that the loader calls. For a weight-training deployment,
``reef.service.deploy.training.training_recipe_config`` forwards the declared
fields from the service configuration and refuses a key that the class does
not declare. A setting that a pipeline stage reads belongs
in the stage's settings, under the deployment's ``pipeline`` section, not in a
recipe field. Never read deployment YAML from inside a stage.

Report
------

.. code:: python

   from reef.core.reports import ReportBase, ReportValidationError, ScoredRolloutReport, TeacherContextReport

``ScoredRolloutReport`` is the shared score-only contract and
``TeacherContextReport`` the shared contract of the distilling recipes: a
rollout's receipt and the ``teacher_context`` its teacher sees (a demonstration,
environment feedback, or nothing for a separate teacher).
Method-specific report contracts live in their method package; Reef does not
import or re-export them.

A report type declares the feedback a method accepts, so malformed input fails
at ingress with HTTP 400.

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

+-----------------------+---------------------+------------------------------------+
| Annotation            | Accepted JSON       | Validation                         |
+=======================+=====================+====================================+
| ``float``             | number              | must be finite; ints normalize     |
+-----------------------+---------------------+------------------------------------+
| ``int``               | number              | integral value; booleans rejected  |
+-----------------------+---------------------+------------------------------------+
| ``str``               | string              | no coercion                        |
+-----------------------+---------------------+------------------------------------+
| ``bool``              | boolean             | no numeric substitutes             |
+-----------------------+---------------------+------------------------------------+
| ``Mapping[str, str]`` | object              | every key and value a string       |
+-----------------------+---------------------+------------------------------------+

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
  set ``close_dispatcher=True``.
- You own the inference handler that you give. Reef does not close it.
  Release it after the app stops.
- If ``build_app`` fails, it closes the resources that it made and raises the
  original error.

Scenario stores
---------------

``RecordStore`` in ``reef.storage.records`` is the abstract base for appending,
reading and auditing records. ``ScenarioStore`` composes a ``RecordStore`` with committed
scenario state, settling each step and its record progress together. Recipes
and trainers use the record interface; ``Scenario`` coordinates training and
artifact publication through its supplied scenario store.

``RecordStore.append_many(items)`` atomically appends an ordered batch from one
scenario and returns an ``AppendResult`` for each input. Conflicts roll back the
batch; identical retries and evicted records follow ``append_result`` semantics.
SQLite and PostgreSQL implement this with one write transaction. Other adapters
must implement atomic batch append to support ``POST /reef/records/batch``;
the default raises ``NotImplementedError`` before writing. Single-record
callers remain compatible with existing adapters.

Storage implementations explicitly subclass ``RecordStore``,
``ScenarioStore``, and ``ScenarioStorage`` and override their abstract
methods and properties. Incomplete subclasses cannot be instantiated. The
bundled ``ItemRecordStore``, ``CommitHistoryScenarioStore``, and
``SQLiteScenarioStorage`` inherit those bases, respectively.

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

``Dispatcher`` requires ``scenario_storage`` and passes it to scenario
creation. It never selects or imports a record storage implementation. It
also requires ``coordination``, the store of the services' work queues,
served models and control lease. One process that runs every service and
keeps no queue state across restarts passes ``memory_coordination()``;
deployment assembly opens the shared store that
``storage.coordination_backend`` names. The ``build_default_dispatcher``
convenience helper also requires both arguments.
Deployment assembly builds the storage that ``storage.record_backend`` names:
SQLite, PostgreSQL, or a ``ScenarioStorage`` plugin; embedded callers supply
their chosen storage service explicitly. For ephemeral records, commits and
settings, pass ``SQLiteScenarioStorage()`` without a directory. With a
directory, each scenario has one SQLite database file for its records,
commits, settings, tasks and uploaded trajectories.

The storage service keeps all per-scenario state. This includes the
``ScenarioSettings`` of each scenario: the selected training mode, the
artifact evaluation, the activation rule and the owner. The scenario
registry uses ``read_settings`` and ``write_settings``. The per-scenario state also includes the published
Harbor tasks of its task stores (see `Task stores`_). ``archive``
retires the settings and tasks with the records and commits.

Direct ``Scenario`` construction requires ``store=...``. Replace the former
``records=...`` and ``commit_log=...`` arguments with one store opened by a
``ScenarioStorage``. The supplied trainer must use that store's ``records``.
``Scenario`` does not choose a storage backend. Read committed history through
``scenario.store.history()`` and persistence capability through
``scenario.store.durable``; the former ``scenario.commit_log`` property has been
removed. ``CommitHistoryScenarioStore.commit_history`` is for backend diagnostics.

The abstract bases ``ScenarioStore`` and ``ScenarioStorage`` are exported from
``reef.storage.scenario``, alongside ``ScenarioSettings`` and ``ScenarioStoreConflictError``.
Concrete storage classes live under ``reef.storage``:
``commit_history.CommitHistoryScenarioStore`` accepts a ``RecordStore``, a
``TaskStorage`` (``tasks=``), and a ``CommitHistory``. The bundled record
stores give their ``TaskStorage`` from ``task_storage()`` and their
``sql_commits.SQLCommitHistory`` from ``commit_history()``; the commits are a
``scenario_commit`` table in the same database as the records. ``sqlite.SQLiteScenarioStorage`` and
``postgres.PostgresScenarioStorage`` assemble their respective backends. The
scenario package depends only on storage contracts. The storage contract is:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``from_config(config, *, directory, task_files)``
     - Class method that builds the storage a deployment names in
       ``storage.record_backend``. ``config`` is the plugin's own mapping;
       ``directory`` is ``storage.agent_record_dir``. ``task_files`` is the
       ``TaskFileStore`` of the deployment (see `Task file stores`_). The
       storage owns it: ``archive`` and ``close`` also apply to it.
   * - ``durable``
     - Whether new sessions and settings survive a process restart.
   * - ``open(scenario)``
     - Open an independently owned session (the ``ScenarioStore`` below).
   * - ``read_settings(scenario)``
     - The scenario's ``ScenarioSettings``; defaults for a scenario with none.
   * - ``write_settings(scenario, settings)``
     - Replace the settings durably. The caller serializes writers of one scenario.
   * - ``archive(scenario)``
     - Retire the scenario's commits, tasks and settings, and remove the
       scenario from the uses of its owner's items, so a recreated scenario
       starts empty; return the archive locations.
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
     - Release the session's record and commit resources; repeated calls are safe.

``CommitRecord`` carries the operation, component selection, private storage
locator, algorithm state, record watermark, consumed IDs, checkpoint/pending
flags, metrics and training job identity. Operations include ``training``,
``activate``, ``train_from``, ``import``, ``sync``, ``serve``,
``stop_serving`` and ``restore``. Restore records a complete prior active
selection and leaves training starts unchanged. A sync selects its new
artifact for serving and training in one commit. Activation and serving record a complete selection without
publishing another artifact snapshot. A selection maps component names directly to artifact IDs.
An ``activate`` or ``train_from`` record carries no trainer state. The store must atomically validate
the current commit number before accepting a new successor: two different commits
prepared from the same commit number cannot both succeed. An identical recorded retry
returns the original record, even after later commits; only
``recorded_at`` is excluded from retry comparison. A new commit with a stale
expected commit number, or a retry with conflicting content, raises
``ScenarioStoreConflictError`` without advancing the store.

``CommitHistoryScenarioStore`` serializes writers through ``CommitHistory.locked()``.
``SQLCommitHistory`` holds one write transaction of the record store, which
SQLite locks with ``BEGIN IMMEDIATE`` and PostgreSQL with a row lock on the
store generation. It reads the history and inserts the commit row in that
transaction. In the same transaction, it adds one to the train count of the
commit's trainer for each sample whose ``sample:<sample_id>`` the commit lists. The commit is
durable when the transaction ends; an exception before then stores no commit
and counts no sample. Direct writes through a ``CommitHistory``
bypass this store contract.
A failure after append can leave the commit stored. Retry the exact commit or
recover the session; do not infer rollback from an exception. Recovery uses
committed ``consumed_ids`` and separate skipped-batch receipts to rebuild
processor memory without repeating work. It does not modify stored bodies.

Artifact bytes and backend head movement remain outside ``ScenarioStore``.
``ScenarioCommitter`` owns their order around store settlement, including
inactive artifacts, activation, new training starts, and checkpoint
reconciliation. See `Commit ordering
<../advanced_topics/state-model.rst#commit-ordering>`__. Step validation does not
provide distributed leases or authorize multiple active training writers.

The storage service exposes ``durable``, ``open(scenario)``, ``archive(scenario)``,
and ``close()``. Archive retires that scenario's stored state so its name can be
reused. Custom backends implement those operations instead
of exposing local file paths. ``from_config`` takes the deployment's data store,
which keeps each scenario's records as record items. Each opened session belongs to its scenario and
closes after its trainer, including on construction or recovery failure. The
dispatcher closes its storage service on shutdown. Application assembly should use
the scenario storage service; direct record store construction is described below.

Task stores
~~~~~~~~~~~

A ``TaskStore`` holds published Harbor tasks, each in the ``train`` or
``eval`` split. A record-to-sample stage that generates tasks saves them in a
task store before any rollout. Thus generation and rollout can recover
independently after a restart. The contract and its values are in
``reef.storage.tasks``.

To open a store, call ``task_storage.open(name)``. Each record-to-sample stage
uses its own ``name``.

A pipeline stage gets ``task_storage`` from ``PipelineContext.task_storage``.
The scenario factory gives ``session.tasks`` to its trainer builder, which
gives it to ``Trainer.build``. For a composite recipe, the builder gives each
component a ``TaskStorage`` that opens ``<component>/<name>``. Thus two
components never share a store. The scenario's imported tasks are not a task
store: they are task items that the scenario uses. The factory reads them from
the data store with the session's ``training_state`` as ``SampleStore``
(RFC #1140). It gives them to its trainer builder as ``samples``. Each
trainer reads them as itself (``SampleStore.for_trainer``) from
``PipelineContext.samples``.

The service takes a task item in the ``train`` split of a scenario
(``POST /reef/namespaces/{owner}/data/tasks``) only when a trainer of the
scenario trains imported tasks (``Trainer.accepts_imported_tasks``). A sample-to-job stage that trains
them sets the class attribute ``accepts_imported_tasks``. An upload to the
``eval`` split is open to every recipe, because an evaluation runs those
tasks.

A store holds generations. A generation holds its source record ids and its
candidates. ``write_candidates`` saves a copy of each candidate task directory
in the task file store. Each ``TaskCandidate`` holds the task ``name``, the
``files_digest`` of its directory, and its outcome:

- A published candidate has a ``split``, ``train`` or ``eval``. The stage
  chooses the split, and the store keeps it.
- A rejected candidate has an ``error`` that gives the reason.
- A pending candidate has neither. After a restart, the stage validates the
  candidates in ``GenerationState.pending_indices``.

A task is a Harbor task directory. The database keeps the state of the store
and the digest of each task directory. The task file store keeps the files,
text or binary, under that digest. ``publish_task`` gives a saved candidate
its split. The published tasks are the candidates with a split. A
``StoredTask`` holds the task ``name``, its ``split`` and its ``files_digest``.

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

Task file stores
~~~~~~~~~~~~~~~~

A ``TaskFileStore`` (``reef.storage.tasks``) keeps the directories of the
published tasks, addressed by the digest of their files. The digest covers the
directory name, and the path and bytes of each file. Thus a repeated save of
the same files changes nothing. A deployment selects the store in
``storage.task_file_backend`` (see `Choose the task file store
<../user-guide/operate.rst#choose-the-task-file-store>`__).
``LocalTaskFileStore`` is the built-in store. An embedded caller can give a
store to ``SQLiteScenarioStorage(..., task_files=...)`` or
``PostgresScenarioStorage(..., task_files=...)``. Without one, the storage
uses a ``LocalTaskFileStore`` in ``task-files`` under its directory.

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
   * - ``archive(scenario)``
     - Retire the files of the scenario and return the archive locations.
   * - ``close()``
     - Release the resources of the store. A second call does nothing.

Data stores
~~~~~~~~~~~

A ``DataStore`` (``reef.storage.data``) keeps the deployment's data items
(RFC #1028). An item is ``(id, content, metadata)``. Its content never
changes: the JSON object of a trajectory or a record, or the directory of a
task, which the task file store keeps in the owner's partition. Its metadata
is a JSON object. Reef reads ``kind`` (``trajectory``, ``task`` or
``record``), ``owner``, ``scenarios`` (scenario name to ``train`` or
``eval``) and, for a record or artifact, ``origin``
(``{scenario, artifact_id}``). ``kind``, ``owner`` and ``origin`` never change. A deployment
selects the store in ``storage.data_store``; ``SQLiteDataStore`` and
``PostgresDataStore`` (``reef.storage.sql_data``) and ``MemoryDataStore`` are
built in. Every operation names one owner and reads or writes only its items.
Every store links the record items of each owner into trajectories, as the
record stores do. The SQL stores keep only the messages that each item adds;
reads return the whole content. With a segment store, the SQL stores keep the
content of trajectory items and record messages in segments (see `Segment
stores`_).

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``from_config(config, *, directory, task_files, segments)``
     - Class method that builds the store a deployment names. ``directory``
       is ``storage.agent_record_dir``. The caller owns ``task_files``.
       ``segments`` is the ``storage.segment_backend`` store or ``None``; the
       data store owns it and closes it.
   * - ``create(owner, items)``
     - Stores ``NewItem`` values, all or none. The same ID and content again
       changes nothing; other content is ``ItemConflictError``. Bad metadata is
       ``ValueError``.
   * - ``read(owner, item_id)``
     - The ``StoredItem``, or ``None``.
   * - ``query(owner, *, kind, scenario, after, limit, content=True)``
     - The owner's items after the sequence ``after``, in arrival order, at
       most ``limit`` (1 to 1000), of ``kind`` and used by ``scenario`` when
       given. With ``content=False``, each item's ``content`` is ``None`` and
       the store does not read it.
   * - ``update(owner, item_ids, patch)``
     - Applies one JSON merge patch to each item's metadata, all or none.
       Changing ``kind``, ``owner`` or ``origin`` is ``ValueError``; an
       unknown ID is ``ItemNotFoundError``.
   * - ``delete(owner, item_id)``
     - Removes the item; ``False`` for an unknown ID.
   * - ``task_path(item)``
     - A local directory with the files of a task item.
   * - ``record_links(owner, item_ids)``
     - The ``RecordLink`` of each given record item that has a message list:
       ``parent``, the nearest ancestor in its trajectory that is still
       stored (``None`` for a root), and ``entry_count``. A trajectory ends at
       a call that no stored call names as parent; its calls are
       ``reef.core.trajectory_index.trajectory_calls`` of the path from the
       root. Reports and unknown IDs are left out.
   * - ``groups(owner, *, kind, key, before=None, limit=1000)``
     - The ``ItemGroup`` values of the owner's items of ``kind`` by the string
       under the metadata ``key``: ``value`` (``None`` without a string),
       ``item_count``, ``first_sequence``, ``last_sequence`` and the first
       item's ``metadata``. Newest first, below ``before`` when given. The
       default reads every page of ``query``; the SQL stores group in SQL.
   * - ``maintain()``
     - The store's periodic work. The data service calls it every 5 minutes.
       It changes no item. The default does nothing.
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
<../user-guide/operate.rst#keep-large-content-in-segment-files>`__).
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
uses Git LFS. A deployment selects a different backend in
``storage.artifact_backend`` (see `Choose the artifact backend
<../user-guide/operate.rst#choose-the-artifact-backend>`__).

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

Repository backends store snapshots. A snapshot's
revision and checksum are private locators; the public identity belongs to
each component's ``artifact_id``. Activating or serving an existing artifact
updates scenario selections without calling ``publish``.

Every scenario owns its root snapshot, which Reef writes when it creates the
scenario (`Starting artifacts <../user-guide/operate.rst#starting-artifacts>`__).
A backend shares no snapshot between scenarios. It must:

- Implement ``register(root, *, metadata)``. ``root`` is an unpublished
  ``Artifact`` whose ``local_path`` is a directory. When the scenario has no
  head, store the files of ``root`` as a snapshot with no parent, preserve its
  artifact IDs and ``metadata``, and make it the head in one atomic
  step. When the scenario has a head, also one that another caller registered
  first, store nothing and return that head.
- Implement ``artifact_snapshot(artifact_id)``. Return the ``SnapshotRef`` of
  the stored snapshot that holds the artifact, or raise ``ArtifactNotFoundError``
  when this repository has no such ID. The ID must keep naming the same files
  and metadata across publications and restarts. Reject an attempt to bind it
  to different content. The inherited ``resolve_artifact(artifact_id)``
  materializes that snapshot and returns the artifact's component.
- Implement ``bind_artifact_id(artifact_id, ref)`` and ``artifact_content(ref, artifact_id)``.
  The first binds an unbound ID to a stored snapshot and returns any earlier binding; a binding never
  changes. The second returns a value that is equal for two snapshots exactly when they hold the same
  files and metadata for the artifact. The inherited ``bind_artifact_ids(snapshot)`` uses both, and
  raises ``ArtifactConflictError`` for an ID that names other content. Call it before a head moves.
- Implement ``resolve_snapshot(revision)``. Return a ``SnapshotRef`` for a
  stored snapshot, or raise ``ArtifactNotFoundError``. Its ``revision`` and
  ``parent_revision`` are backend locators, separate from component IDs.
- Keep each component entry's ``artifact_id`` in the manifests it stores.

The optional capabilities are separate base classes. Inherit only the ones
that the backend supports:

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
     - Both capabilities, and one cached backend for each scenario.
   * - ``StagedSnapshotRepositoryBackend``
     - The backend side: ``publish(advance_head=False)`` stores a snapshot
       without moving the head, and ``commit_snapshot`` moves the head
       after the scenario commit. A durable scenario store requires it.

A backend whose storage can give clients a URL for each file, such as an
object store, overrides ``file_downloads(ref, expires_in_seconds)``. It
returns every file of the snapshot as a ``FileDownload``: ``path`` in the
snapshot, ``byte_count``, and a ``url`` that opens the file without a Reef
credential until ``expires_at``. It must not materialize the snapshot. The
default returns None, and Reef then lists and serves the files of the
materialized snapshot itself. The artifact file list of the HTTP API gives
these URLs to clients, so that a large artifact downloads from the storage.

Such a backend also overrides ``snapshot_uri(ref)``: the URI of the snapshot's
directory, ending with ``/``, in a scheme that ``ArtifactSources`` reads. The
`artifact items <http-api.rst#artifact-items>`__ of the data store carry it,
so that a scenario, also in another deployment, imports the snapshot from
there. The default returns None.

``tests/artifact/test_reef_artifacts.py`` and
``tests/artifact/test_reef_git_lfs.py`` show the publication and recovery
behavior that a backend must have. ``GitLFSRepositoryBackendFactory`` is the
built-in plugin. Reef does not call a close method on a backend factory.

Artifact imports
----------------

``POST /reef/scenarios/{scenario}/artifacts`` imports a directory as a new
artifact of one component (`Import an artifact
<../user-guide/operate.rst#import-an-artifact>`__). A source reader opens the
files that the URI names, the component's layout checks them where the source
keeps them, and then Reef stores a copy, or a reference to a Hugging Face
revision that cannot change. Two contracts decide what an import accepts.

A component's layout is the ``layout`` of its ``ComponentSpec``, an
``reef.artifact.components.ArtifactLayout``. ``check(files)`` returns when ``files`` are
an artifact in the layout of the component's snapshots, and the remote weights
that they name exist. Otherwise it raises ``ValueError``, and Reef answers
HTTP 400 before it copies anything. ``None``, the default, means that the
component takes no import. ``files`` is a ``reef.artifact.ArtifactFiles``. It
gives ``paths()``, and ``sha256(path)`` and ``read_bytes(path)`` for each path.
Read only the files that the check needs: on an S3 prefix or a Hugging Face
revision, each read downloads that file. The built-in recipes bind one:

- A weight recipe's component gets ``TrainingArtifactLayout(training_component)``
  (``scenario_components``), which calls
  the component's ``check_artifact`` (see `Training jobs and components`_).
  For full weights and adapters it reads only file names and
  ``adapter/adapter_config.json``.
- A harness recipe binds ``HarnessLayout``. Every file must be UTF-8 text,
  and the files must form a ``HarnessTree`` that passes ``validate_tree`` for
  the adapter: the source tree a harness artifact stores.

After the layout, the snapshot's own admission check runs on what Reef stores,
as at a training publication.

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

Reef reads a base root but does not write one yet: a default start still
holds the files that ``default_start()`` gives. The ``spindle_full`` backend
has no base root. Its default start is a reference to the base revision.

Record storage and audit
------------------------

``reef.storage.records`` defines ``RecordStore``, record result and error types, and
``RecordRetention``. It has no dependency on scenario coordination, training,
or concrete storage adapters. Both ``reef.RecordStore`` and
``reef.storage.records.RecordStore`` refer to this abstract base.

``reef.storage.record_items.ItemRecordStore`` is the ``RecordStore`` of one
scenario over the data store (RFC #1219). It appends each record as its record
item with the scenario's generation, and reads the record items that the
scenario made, in item sequence order. ``SQLiteScenarioStorage`` and
``PostgresScenarioStorage`` give one to each session. Both take a data store
(``data=``); without one, the SQLite storage keeps its records in
``<directory>/data``, and the PostgreSQL storage in the schema
``<schema>_data``. Deployment assembly gives them the deployment's data store.

``reef.storage.sql_records.SQLScenarioDatabase`` is the database of one
scenario's state: its settings, and the transactions of its task, sample,
evaluation and commit stores. ``reef.storage.sqlite.SQLiteScenarioDatabase`` and
``reef.storage.postgres.PostgresScenarioDatabase`` supply the connections,
schemas and conflict insertion. Install ``reef-infra[postgres]`` to enable the
PostgreSQL driver.

``PostgresScenarioStorage(url, directory, schema=...)`` keeps commits (the
``scenario_commit`` table) and settings (the ``scenario_setting`` table) in the
schema. Pass it through ``Dispatcher(..., scenario_storage=...)`` for embedded
use. Scenario code continues to depend only on the abstract store contracts.
Deployment configuration selects this storage service with
``storage.record_backend: postgres``; see
`Choose the storage backend <../user-guide/operate.rst#choose-the-storage-backend>`__
for connection and storage setup.

A database of a version before RFC #1219 keeps its records in the tables
``agent_record``, ``record_node``, ``consumed_agent_record``,
``record_consumption`` and ``record_eviction``. That version copied each record
to its record item at the first open of its scenario. The storage drops these
tables when the record item of the newest record of each active scenario is in
the data store: the PostgreSQL storage when it starts, the SQLite storage at the
first open of each scenario. Otherwise it keeps them and logs a warning: run
Reef 83a6f3ab once to copy the records.

Training commits persist consumption progress without changing record
visibility or deleting bodies. Record-to-sample stages expose
``releasable_record_ids()`` and ``release_records(ids)``, which the samples
service calls for memory management. Storage controls capacity
independently of these buffers.

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

``StoredRecord`` contains ``sequence`` and ``item`` (the original ``AgentRecord``).
Reading a record does not change a consumer's progress. A missing body may have been purged or never stored;
the read API does not guess which. Use the commit history's per-step
``consumed_ids`` to determine consumption, including intentional skips; it is
not proof that every named record produced a model update.

For example, inspect one trace without making it available to training again:

.. code:: python

   entry = scenario.records.read_for_audit(scenario.name, record_id)
   if entry is not None:
       payload = entry.item.payload
       references = entry.item.references

The HTTP service runs background retention at startup and every 60 seconds. It
deletes the oldest record items, across owners, only when they hold more than a
shared 20 GiB budget (``DataStore.prune_records``). The budget counts each record
item's row and the trajectory nodes that only it needs. It counts and deletes
only the record items whose ``origin.source`` is ``inference``; it keeps the
calls of evaluation rollouts. Limits are configurable
in `Configuration <configuration.rst>`__.

For embedded Python deployments, use
``dispatcher.prune_record_archives(RecordRetention(max_bytes=20 * 1024**3))``
with ``RecordRetention`` imported from ``reef.storage.records``. This runs one
sweep over the dispatcher's data store and serializes it with scenario archival.
``Dispatcher`` construction does not start a maintenance task.
``create_app(dispatcher, ...)`` requires an existing dispatcher and accepts
``record_retention=RecordRetention(...)`` to enable service maintenance. It
closes the dispatcher on cleanup only when ``close_dispatcher=True``. The
deployment assembly selects the storage backend and passes the dispatcher to the
HTTP app; ``create_app`` never creates a default backend.

Retention logs nothing per record. A trainer reports the loss of its scenario in
its metrics and status. Concurrent writes can exceed the budget between sweeps.
Indexes and other metadata also require disk space. HTTP audit routes remain a
separate integration.

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
sample2job=...)``):

- ``record2sample``: the record-to-sample stage
  (``reef.record2sample.Record2Sample``). The pipeline pushes every record to
  it (``push``). It decides which records train and which train together: it
  forms record sets (``form``), groups reports, judges records, and ends
  records that can no longer train. ``form`` makes the samples of each set
  (``make_samples``) and holds them in a record set unit until the unit is
  ready. A sample is a ``Sample`` value. Reported feedback supplies a
  ``make_sample`` hook on ``ReportSamples``; computed feedback supplies one on
  ``JudgedSamples``, after an asynchronous ``judge`` derives new feedback. A
  stage with slow work keeps its records and forms the set in
  ``advance(now)`` when the work ends (see `Slow work
  <../developer-guide/processors.rst#slow-work>`__). The stage's staleness
  check (``is_stale``) keeps samples with data more than ``max_staleness``
  artifacts behind the served artifact out of the next automatic job. ``max_staleness``
  is the stage's own setting, from ``Record2SampleSettings`` (see `the
  staleness check <../developer-guide/processors.rst#staleness>`__). The
  default, ``NoSamples``, makes no sample, so nothing trains.
  ``uploaded_sample(trajectory) -> TrajectorySample`` turns an uploaded ATIF
  trajectory into its training sample. It must be pure, and it raises
  ``ValueError`` to skip the trajectory (see `Uploaded trajectories
  <../developer-guide/processors.rst#uploaded-trajectories>`__).
- ``sample2job``: the sample-to-job stage (``reef.sample2job.Sample2Job``).
  Its ``make_job_data(samples, data_id)`` shapes the samples of the next job,
  in use order, into one job's data with the ID ``data_id``. The pipeline assigns the
  ID. Automatic job data gets ``<scenario>:data:<number>``. A request's
  data gets ``<scenario>:request:<request id>``. The number counts the
  job data of one pipeline. It is not a training step.
  ``Sample2Job.__init__(context, buffer_size, job_samples, *, max_job_samples=None)``
  sets the most samples that one automatic job takes and the fewest
  samples of one job. A ``job_samples`` of ``None`` is a stage that sets no
  job size, which ``Trainer.build`` refuses for a trainer that trains; a
  ``buffer_size`` of ``None`` takes ``job_samples``. The constructor raises ``PluginConfigError`` for
  a ``buffer_size`` smaller than ``job_samples`` and for a ``job_samples`` above
  ``max_job_samples`` (an SDPO or TTTD grid).
  ``Trainer.build`` raises ``PluginConfigError`` for a training trainer whose
  stage has no ``job_samples``. ``Sample2JobSettings`` parses the
  ``buffer_size`` and ``job_samples`` keys, and a stage with more keys
  extends it. See `Job size <../developer-guide/processors.rst#job-size>`__.

Reef builds the stages for each scenario, in this order, with the class
methods below; the sample-to-job stage receives the record-to-sample stage:

- ``Record2Sample.from_config(config, *, context, environ)``
- ``Sample2Job.from_config(config, *, context, record2sample, environ)``

``config`` is the stage's config from the deployment's ``pipeline`` section;
``environ`` holds the environment the deployment was read with. A stage
declares its keys in a frozen ``reef.core.plugins.PluginSettings`` subclass
with ``reef.core.config.config_option`` fields and validation in
``__post_init__``. In ``from_config``, parse the config with
``Settings.parse(config, plugin_class=cls)`` (``NoSettings`` for a stage
without config) and give the settings object to the constructor. A bad config
fails when the first scenario builds its pipeline. A sample-to-job stage that
needs the record-to-sample stage as a specific class checks it with
``isinstance`` and raises ``PluginConfigError`` otherwise. A value that both
stages need belongs to the record-to-sample stage; the sample-to-job stage
reads it from the stage object that it receives. A ``Record2Sample`` that keeps
per-scenario state also implements ``scenario_state_dirs(config, scenario)``,
so a scenario delete archives it. Reef owns replay, retention, deduplication, pending job data,
and exactly-once consumption; the method implements only the hooks below.
They run synchronously on the trainer thread, so they must not block on
network or model latency. Reports must reference existing inference records
in the same scenario.

``TrainingPipeline.training_mode`` selects automatic (``auto``) or
request-triggered (``manual``) training. A ``Sample2Job`` declares
``supported_training_modes``. In ``manual`` a request's data holds the
samples that the request names; ``Sample2Job.reads_requests`` says
whether a request that names no sample runs. Record push, sample storage, acknowledgement, retention
and background derivation are shared. See
`Training pipeline <../developer-guide/processors.rst>`__ for the training request queue and job data contract.

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
   * - ``Sample2Job.make_job_data(samples, data_id) -> JobData``
     - Shape the flat tuple of selected tasks into job data with the ID ``data_id``.

The recipe owns the report policy. Its record-to-sample stage is a subclass of
``ReportSamples(context, grouping=None, *, exclusive_sources=False, max_staleness=0)``.
With no ``grouping``, it trains every report by itself. With
``exclusive_sources=True``, a report that ends owns its inference records.
A report references the last call of each trajectory that it scores;
the stage expands each reference to the calls of its trajectory
(``reef.core.trajectory_index``). No inference is in two record sets: a report
whose trajectory holds an inference of another report ends without training,
and so does a stored report that lists calls inside one trajectory. A
``ReportGrouping`` runs in the record-to-sample stage and reads only records:

.. list-table::
   :header-rows: 1

   * - Method
     - Contract
   * - ``placement(context) -> (group_key, slot)``
     - Optional collection group and retry slot; defaults to an independent report.
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

``make_sample`` returns a trajectory or task directly. Its default makes the
report's trajectories one ATIF trajectory sample with the report's score.
``ReportSamples`` adds the calls and the report id to
``source_agent_record_ids``.

``TrajectorySamples(context, *, trajectory_idle_seconds=300.0,
max_staleness=0)`` is the stage of a method that needs no report. It forms a
``TrajectorySet`` of a trajectory's calls once the trajectory's last call has
had no continuing call and no retry for ``trajectory_idle_seconds``, counted
from the call's ``created_at``, and makes
it one ATIF ``TrajectorySample`` (``trajectory_sample``). Its ``next_due()``
is the time at which the earliest waiting trajectory goes idle. Every other
record ends at once. Grouping and
consumption state stay in the stage; recipes never receive buffered report objects.
``placement`` returns ``(None, None)`` by default. A group key collects reports,
and a slot deduplicates retries within that group (None uses the report id).
This collection group differs from ``TrajectorySample.group_id``: TTTD waits for
a complete step, then trains several comparison groups within it.
The pipeline stores the samples of a report or a ready group when it is
ready, and that consumes their records. A group is fixed when it is first
ready; later reports for its key form a new group. ``make_job_data``
receives the samples of the `next job
<../developer-guide/processors.rst#the-next-job>`__ in use order: units in
the order they became ready, interleaved with uploads by arrival. The job's commit consumes every sample that it receives, even
when the recipe omits constant-reward groups from its output. Samples stored
after the job data was built wait for later job data.
On the stage, ``exclusive_sources``
controls source release for terminal group/duplicate reports.

There is no report-level ``judge``, ``WAIT``, ``NEVER``, or eligibility flag.
Reference validation runs at admission; incomplete groups wait for more valid
samples. Training input violations raise instead of silently filtering reports.

Computed feedback
~~~~~~~~~~~~~~~~~

Use this engine when later traffic completes an earlier record and judging it
calls a model or another slow service. The recipe writes a ``RecordJudging``.
Its record-to-sample stage is a ``JudgedSamples(judging, worker, *,
max_staleness=0)`` subclass that implements ``make_sample``, and its
``Sample2Job`` shapes the judged samples into job data.

+------------------------------------------+-----------------------------------+
| Hook                                     | Contract                          |
+==========================================+===================================+
| ``RecordJudging.observe(record, now,``   | correlate one record with the     |
| ``stage)``                               | stage's verbs; must not block     |
+------------------------------------------+-----------------------------------+
| ``async RecordJudging.judge(job)``       | slow judgment, on the worker      |
|                                          | thread                            |
+------------------------------------------+-----------------------------------+
| ``RecordJudging.expire(now)``            | optionally return receipts whose  |
|                                          | completion window ended           |
+------------------------------------------+-----------------------------------+
| ``JudgedSamples.make_sample(record,``    | a ``TrajectorySample``, or        |
| ``judgment)``                            | ``None`` to retire the record     |
+------------------------------------------+-----------------------------------+
| ``make_job_data(samples, data_id)``      | the declared job data type        |
+------------------------------------------+-----------------------------------+
| ``required_request_types``               | optionally restrict which record  |
|                                          | types reach the pipeline          |
+------------------------------------------+-----------------------------------+
| ``accepts_trajectories``                 | optionally set True to train on   |
|                                          | uploaded ATIF trajectories        |
+------------------------------------------+-----------------------------------+

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
``pipeline.record2sample.config.generator``. Subclasses implement:

- the class method ``from_config(config, *, scenario, environ, state)``, which
  returns the generator of one scenario;
- ``async generate(request: TaskGenerationRequest) -> tuple[HarborTask, ...]``;
- ``async validate(task: HarborTask) -> TaskValidationResult``.

``async record_results(results: tuple[GeneratedTaskResult, ...], *, generation: int)``
is optional. It receives the results of one earlier ``generate`` call, named by
``TaskGenerationRequest.generation``, possibly after later calls and after a
restart. A second call for the same generation must have no effect. The
default keeps nothing. The built-in generated-task recipe,
``reef.recipe.generated_tasks:GeneratedTaskRecipe``, calls these methods.
See `Training pipeline <../developer-guide/processors.rst#task-generation-contract>`__
for the round and its failure behavior.

``state`` is a ``GeneratorState`` (``reef.recipe.generated_tasks``): the
durable state of the scenario's generator, kept in the scenario's task store.
It holds text documents by key (``read``, ``write``, ``remove``, ``keys``) and
saved directories by the digest of their files (``save_files``, ``files``). A
generator keeps its state there and keeps no file of its own. A generator
written for an earlier version must accept the ``state`` argument.

``TaskGenerationRequest(source_records, description, assets=(), asset_files={}, generation=0)``
carries source records, non-empty requirements text, and the number of the generation.
Each ``AgentRecord`` must have a distinct ID and belong to the same scenario.
Use an empty record tuple for description-only generation.
For local materials, pass a tuple of ``Path`` values in ``assets``.
For materials in memory, use ``asset_files`` to map relative file names to
UTF-8 text. Supplying both fields is an error. The constructor validates
fields without reading files.

Record2dataset accepts the same input through ``DesignerRequest(inputs=request,
...)``. Its HTTP client reads local materials and sends their contents. The
service rebuilds ``TaskGenerationRequest`` from the JSON fields. The generated
task preserves source record IDs in order.

``GeneratedTaskResult`` is ``RejectedTask(task, reason)``, ``HeldOutTask(task)``,
``RolledOutTask(task, rollout_rewards)`` or ``FailedTask(task, error,
infrastructure_failure)``. ``rollout_rewards`` is a non-empty tuple of finite
numbers; ``reason`` and ``error`` are non-empty. See `the task generation
contract <../developer-guide/processors.rst#task-generation-contract>`__ for
when the recipe sends each case.

``TaskValidationResult(errors=())`` exposes ``is_valid``. Each error is a
non-empty explanation of a task defect. Failures to execute the checks raise
exceptions instead.

Migration: ``reef.train.processors.TaskGenerationProcessor`` is removed. Move
its ``generate`` and ``validate`` methods to a ``TaskGenerator`` subclass:

- ``generate`` returns a tuple of ``HarborTask`` values instead of one task.
- ``validate`` receives the ``HarborTask`` instead of a directory path.
- Import ``TaskValidationResult`` from ``reef.recipe.generated_tasks``.

Name the class in ``pipeline.record2sample.config.generator`` and use the generated-task recipe. The recipe runs the tasks and trains, so the processor half of the method is
not needed.

Job data
--------

.. code:: python

   from pathlib import Path
   from reef.train.types import (
       TaskSample, Sample, JobData, TrajectorySample,
       trajectories, trajectory_groups,
   )

``Sample = TrajectorySample | TaskSample``. Every processor returns a frozen
``JobData(data_id, samples, request=None)`` with an ordered tuple of samples.
One job's data holds samples of one kind. ``request`` is keyword-only and carries
an optional explicit training request; request-only job data may be empty.
The pipeline returns the same pending job data until its id is acknowledged.
Reef submits one training job for each value. The training component cuts the
samples into its own optimizer batches.

.. code:: python

   data = JobData(
       "scenario:data:1",
       (
           TrajectorySample(atif_document, source_agent_record_ids=("inference-1", "report-1")),
           TrajectorySample(other_atif_document, source_agent_record_ids=("inference-2", "report-2")),
       ),
   )

A ``TrajectorySample.trajectory`` is an
`ATIF document <https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md>`__
as a JSON-compatible mapping. All shipped processors and recipes use this
representation. The item checks the envelope; consumers validate the fields
their algorithms require. Standard steps describe messages, tool calls and
observations. Captured records use ATIF-v1.7, matching the pinned Harbor version.

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
and no tokens. The training component renders it with the model's chat
template and computes the log-probabilities with the weights of the job's
start (`Chat rendering`_). The rendered sample then holds ``tokens`` and
``loss_mask`` in ``training``, and ``teacher_tokens`` or
``teacher_candidates`` when it has a teacher request. A task rollout of a
training component carries its exact ``tokens``, ``loss_mask`` and
``rollout_log_probs``, ``artifact_id`` (the artifact that sampled it), and,
when the component records them, the sampler's ``topk_indices`` and
``topk_log_probs``.

``item.metadata`` and ``item.training`` read these extension objects.
``with_metadata`` and ``with_training`` return updated documents while preserving
other extensions. Training arrays are JSON lists. ``item.artifact_id`` is the
artifact that produced the trajectory: ``training.artifact_id``, else the
``artifact_id`` of the last recorded call (``records[-1].payload``), or None
for an upload. The staleness check reads it.

``TrajectorySample.group_id`` is an optional string, scoped to the job data.
None means an independent trajectory. Group-relative policy algorithms require
an explicit id on every trajectory and contiguous samples for each group, so
advantages and rows stay aligned. TTTD and CORAL preserve their existing group
barriers and ordering. SAO and OpenClaw-RL explicitly schedule each trajectory
independently, even when group metadata is present. ``trajectory_groups(data)`` reads these comparison groups;
``trajectories(data)`` reads the trajectory samples in order.

A ``TaskSample`` names a `Harbor task directory <https://www.harborframework.com/docs/tasks>`__
with ``task_path: Path``: the directory contains the instruction, environment
and verifier configuration/files. The path is interpreted in the consuming
algorithm's execution environment. The value also accepts optional ``metadata``.
``files`` is a ``TaskFiles(partition, digest)`` that names the directory in the
task file store, or ``None``. Only a task with ``files`` can go into a training
job (see `The job data directory <../developer-guide/processors.rst#the-job-data-directory>`__).
It performs no file I/O or rollout. Algorithms supporting tasks own validation,
environment execution and conversion to trajectories. The Tinker component,
and the Slime component with its own rollout engines, roll task samples out in
task jobs; the harness backends refuse them. Accepting ``JobData`` does not imply that every algorithm
supports both sample kinds.

Both sample kinds may carry ``source_agent_record_ids``. These record derivation
inputs; processor acknowledgement remains authoritative for consumption,
including inputs removed by a recipe's existing group rules.

The ``reef.core.trajectories`` helpers construct and consume ATIF:
``make_trajectory(records, reward, feedback)`` converts the exchanges with
``reef.core.atif_payloads``, as the record stores do, writes the last record's
``tools`` as the agent's ``tool_definitions``, and retains original exchanges;
``recorded_payload`` / ``recorded_payloads`` project exchanges for harness
methods, including external ATIF without captured records; and
``trajectory_reward`` reads a finite policy reward. ``SampleAssembly.build``
(``reef.record2sample.assembly``) makes a report's sample. Without
``accept_multi_turn_policy_samples``, the sample is the trajectory's last
call: its earlier agent steps are ``is_copied_context: true``, and only the
last reply trains. With it, the sample holds every call, and every reply
trains. A task rollout of a training component uses
``make_policy_trajectory`` or ``make_multi_turn_policy_trajectory`` from
``reef.record2sample.samples`` to keep the tensors that its engine
returned. Both job data readers reject unsupported sample kinds
instead of silently filtering mixed job data.

Completed text-only streaming responses contribute their aggregated
``response.message`` to the ATIF agent step. The original SSE body remains
in the captured record. Matching assistant messages in later request history
are reused as context without duplicating those steps. An explicit
``training.response_message`` takes precedence over the stream summary.

Migration: the former policy/trace sample types and their batch subclasses are
removed. Construct ATIF trajectory samples directly and return ``JobData(id, samples)``.
Use ``group_id`` on each member for grouped job data. Read training fields from
``item.training`` and rewards through ``trajectory_reward(item)``.
The component reads job data as a job data directory, so only the fields of
``JobData`` reach it. Keep job data serializable, with no handles to
services, files, threads or models.

Chat rendering
~~~~~~~~~~~~~~

Serving returns no tokens and no log-probabilities. A training component
renders each trajectory sample with its model's chat template, and computes
the log-probabilities that its objective needs with the weights of the job's
start. These are exact when ``max_staleness`` is 0 and an approximation above
it. ``reef.core.chat_rendering`` holds the shared rendering:

- ``ChatTemplate`` renders messages and tools to token IDs.
  ``TokenizerChatTemplate(tokenizer, template_kwargs)`` is the chat template of
  a Hugging Face tokenizer. Its ``template_kwargs`` must equal the chat
  template arguments that the serving engine renders with.
- ``render_trajectory(template, trajectory)`` renders each trained turn as
  its call rendered it: the conversation before the turn with the generation
  prompt, then the turn. The agent's ``tool_definitions`` are the tools. Each
  agent step that is not ``is_copied_context`` trains: the loss mask is 1 on
  its tokens. A turn whose call's prompt extends the tokens of the turn
  before it continues that segment. Else it starts a new segment. The result
  is one ``RenderedTrajectory`` for each segment. It raises ``ValueError``
  when the trajectory has no trained step, or when the template does not
  render a turn after its own generation prompt.
- ``TeacherRequest`` is the ``extra.reef.teacher`` value. Its
  ``teacher_tokens`` is the rendered teacher prompt followed by the student's
  reply tokens verbatim, or the student's own tokens when it names no
  messages.
- ``rendered_job_data(data, template)`` renders each trajectory sample that
  has no ``tokens``. A trajectory of several segments becomes one sample for
  each segment, with ``segment``: ``{"index": i, "count": n}`` in its
  ``extra.reef.training``. It returns the job data and the positions of the
  samples that it leaves out: a sample that cannot render, that is longer
  than its ``max_tokens``, whose teacher sequence is longer than its teacher
  request allows, or that has a teacher and several segments. The Slime and Tinker components count them in the job
  metric ``unrenderable_samples``.

Training objective
------------------

Each training backend owns its objectives: ``SlimeObjective`` in
``reef.train.slime_backend.objectives`` and ``TinkerObjective`` in
``reef.train.tinker_backend.objectives``. An objective
owns signal preparation over a job's full data and declares the backend's loss family.
A job names it in its ``objective`` parameter, usually from the recipe's
``default_parameters()``: a registered name, or a dotted
``my_method.slime:MyObjective`` reference.
The dotted reference names a class with a zero-argument
constructor; its module must be importable in both the
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
``register_objective(name, "my_method.slime:MyObjective")`` from
``reef.train.slime_backend.objectives`` (or from
``reef.train.tinker_backend.objectives`` for a ``TinkerObjective``). The call
records the reference and imports nothing; the backend imports the module when
a job first names the objective. The same function accepts an instance in
place of the reference. Each backend has its own ``register_objective``,
``resolve_objective``, ``TrainingSignal``, ``next_steps`` and
``StepScheduling``; the examples here show Slime's. Core never imports method
packages. A built-in
``sft`` objective exists on each backend, and ``slime_dataset`` on Slime.

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

Each component parses a job's parameters into its own typed value:
``reef.train.slime_backend.parameters.SlimeJobParameters`` and
``reef.train.tinker_backend.parameters.TinkerJobParameters``. The parser raises
``ValueError`` for an unknown key, when ``objective`` is missing or does not
resolve, when ``scheduling`` has another key or a value out of range, and when the
objective's ``validate_scheduling`` refuses the schedule: ``epochs > 1``
without ``supports_multiple_epochs``. Slime and Tinker parse when they
accept a job and again when they run it.
``TrainingComponent.parameters_schema(kind)`` returns the JSON Schema of those
parameters for a sample kind, or None; the scenario contract serves it.

``pack_rollouts(token_counts, packing_length)`` of each backend's ``schedule`` module is
the packing: best-fit decreasing, which puts each rollout, longest first, into
the pack whose free tokens it fills most closely. The packs decide only which
rollouts share a step. The loss and the micro-batches stay the component's.

``WeightTrainingRecipe.parameters_by_kind()`` returns the parameters of each
training job by the kind of the job's samples: ``trajectory`` or ``task``.
The trainer gives them to its pipeline (``PipelineContext.job_parameters``).
When a job's data is ready and no job runs, the pipeline's sample-to-job stage
gives the job's parameters, ``Sample2Job.job_parameters(data)``: those of the
kind of the data's samples, with the request's ``parameters``
(``TrainingRequest.parameters``) merged in by ``reef.core.values.deep_merge``:
nested objects merge key by key, and every other value of the request wins. It refuses data that holds
both kinds, and a kind that the recipe does not name when no request gives
parameters. A kind that the recipe does not name never trains in ``auto``. Reef reads none of the parameters; it submits
them to the deployment's training component, whose name is the job's
``backend``.

By default, the parameters of both kinds are the recipe's
``default_parameters()`` with its ``job_parameters`` config over them, key by key,
such as ``scheduling`` or ``save_interval``.

The parameters are part of the job's ID, so ``parameters_by_kind()`` must
return the same value at every call. See `Write a recipe
<../developer-guide/write-a-recipe.rst#train-each-kind-of-sample-with-its-own-method>`__
for a recipe that trains each kind with its own method.

``TrainingSignal`` carries:

- ``action``: ``train`` runs backend training; ``skip`` commits a state-only
  transition.
- ``advantages``: optional per-trajectory values, in the job data's order.
- ``next_algorithm_state``: proposed state committed through the trainer's
  existing step lifecycle.
- ``metrics``: method telemetry carried to the commit record.

``prepare`` runs on the complete reserved job data before optimizer or worker
partitioning. Compute group-relative statistics here. Keep this entry point
free of torch, Ray, Slime and Tinker imports. It must not mutate the job data,
execute training, or commit state. Model-dependent terms (critic-based GAE,
reference-model KL) stay in the backend implementation after the required
forward passes. Both Slime and Tinker use the same objective preparation;
the selected backend resolves its own loss implementation lazily.

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
       ``decision(step)`` and ``run_dir(run)``
   * - ``CandidateRecord``
     - ``tree``, ``step``, ``origin`` (``seed``, ``method``, ``proposal`` or
       ``release``), ``parent``, ``summary`` and ``notes``
   * - ``RunRecord``
     - ``run``, ``tree``, ``task``, ``task_key``, ``repeat``,
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
measures a candidate tree, and decides whether the search keeps it. ``CandidateEvaluator`` and
``CandidateSelector`` remain separate abstract capabilities; implementing only
one does not make an object a plugin. Reef
enforces the fixed evaluate-then-decide order and verifies the decision kept the
exact result it was given.

+-----------------------------------+-----------------------------------------------+
| Member                            | Contract                                      |
+===================================+===============================================+
| ``evaluate(candidate)``           | returns an ``EvaluationResult``: evaluator    |
|                                   | name, version, and a metrics mapping          |
+-----------------------------------+-----------------------------------------------+
| ``decide(candidate, evaluation)`` | returns a ``SelectionDecision`` whose         |
|                                   | ``selected`` is a boolean, and which must     |
|                                   | carry the evaluation it was given             |
+-----------------------------------+-----------------------------------------------+

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

+----------------------------------------------+------------------------------------------------+
| Member                                       | Contract                                       |
+==============================================+================================================+
| ``activates(made)``                          | true when Reef activates the artifact as soon  |
|                                              | as training publishes it. ``made`` is a        |
|                                              | ``MadeArtifact``: the ``component`` and the    |
|                                              | ``changed_kinds`` of a harness tree            |
+----------------------------------------------+------------------------------------------------+
| ``activates_evaluated(artifact_id, result)`` | true when Reef activates the artifact once an  |
|                                              | evaluation of it finished with the             |
|                                              | ``PolicyEvaluation`` ``result``; false by      |
|                                              | default                                        |
+----------------------------------------------+------------------------------------------------+

The committer asks ``activates`` when a training step publishes an artifact.
The commit record's ``activated`` keeps what the rule decided. The artifact
evaluation service asks ``activates_evaluated`` when an evaluation finishes,
and then activates the artifact. The built-in rules are
``reef.evaluation.activation``'s ``AlwaysActivate``, ``NeverActivate`` and
``EvaluatedActivation``. The deployment names a rule in its ``activation``
section (`Configuration <configuration.rst#the-activation-section>`__). A rule
of your own inherits ``ActivationRule``. Reef builds it from its config with
the class method ``from_config(config)``.

Scenario components
-------------------

.. code:: python

   from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree

``Recipe.build_components(scenario)`` returns the components that every
snapshot of the scenario binds, as the recipe's configuration declares them.
``WeightTrainingRecipe`` declares a ``weights`` component, whose kind is the
deployment runtime's ``served_weight_kind``
(``reef.scenario.live_objects.scenario_components`` adds it), and
``HarnessEvolutionRecipe`` declares a ``harness`` component that clients
pull, so most methods never touch this. See `Scenario components
<../developer-guide/components.rst>`__ for where Reef reads each field.

``ScenarioComponents`` is a frozen dataclass mapping each component name to a
``ComponentSpec``. Bare ``ScenarioComponents()`` is the complete record-only
configuration. A snapshot of one component is flat; a snapshot of several
components has one directory per component. At most one component has a
weight kind, and at most one has ``files``.

+---------------+-----------------------------+----------------------------------------------+
| Field         | Type                        | Contract                                     |
+===============+=============================+==============================================+
| ``validator`` | ``ArtifactValidator``       | admit the component before publication and   |
|               |                             | activation; the default accepts any artifact |
+---------------+-----------------------------+----------------------------------------------+
| ``kind``      | ``ArtifactKind | None``     | the kind of every artifact of the component; |
|               |                             | ``model`` and ``model_adapter`` load into    |
|               |                             | the inference runtime                        |
+---------------+-----------------------------+----------------------------------------------+
| ``files``     | ``FileTree | None``         | back client pulls                            |
+---------------+-----------------------------+----------------------------------------------+
| ``layout``    | ``ArtifactLayout | None``   | check an imported artifact; ``None`` takes   |
|               |                             | no import                                    |
+---------------+-----------------------------+----------------------------------------------+

The inference runtime loads the weight component with
``InferenceRuntime.activate_checkpoint(artifact, scenario=...)``, which returns
the runtime load ID that serves it. No component changes an inference request:
the inference routes forward each request to the engine that ``served_models``
names.

Components do not decide which records train, compute candidates, execute a
training job, or mutate the snapshot chain. Native streaming behavior stays
unchanged. A method should not add an HTTP proxy or copy Reef's record store.

Runtime responsibilities
------------------------

``TrainingComponent`` and ``InferenceRuntime`` are independent interfaces.
Neither inherits from the other, and no trainer holds a serving engine.

* A ``TrainingComponent`` runs training jobs and saves checkpoints. It cuts
  each job's data into optimizer batches by its own batch size, from its
  training options (Slime's ``--global-batch-size``, Tinker's
  ``batch_size``). Reef does not read that size; the deployment sets the job
  size in ``pipeline.sample2job.config.job_samples``.
* An ``InferenceRuntime`` executes requests, manages admission, and loads
  published snapshots. It loads a snapshot when a
  scenario starts, when a committed snapshot is activated, and when a person
  activates one.
  Its ``model_path`` property supplies the default model name for harness
  bindings; the base implementation returns an empty string for runtimes whose
  requests select the model. ``per_scenario_adapters`` is true for a runtime
  that serves each scenario its own adapter. ``served_weight_kind`` is the
  kind of the weights that the runtime serves from a snapshot: ``model``, the
  default, for full weights, or ``model_adapter`` for a LoRA adapter on its
  base model. A runtime that serves adapters must override it.
  ``served_artifact(artifact_id, scenario)`` gives the engine's HTTP(S) URL
  and the model name (``served_model_name(scenario)``) that inference writes to
  ``served_models`` for the scenario. It returns None when the engine has no
  such URL; startup then fails without publishing a route.
  ``artifact_model(artifact)`` is the ``PolicyModel`` that the runtime serves
  for an artifact on request, without loading it, or None, the default, for a
  runtime that serves only its loaded artifact. A runtime whose handler answers
  each call from the artifact that the call names overrides it, as the Tinker
  runtime does; the `artifact evaluation <http-api.rst#evaluate-an-artifact>`__
  of a model artifact needs it. Such a runtime also returns True from
  ``serves_any_artifact``.
  ``shared_by_training_scenarios`` is true for a runtime that serves each
  training scenario its own weights.
* ``ScenarioCommitter`` publishes checkpoints and records artifact selections.
  A dispatcher queues each selection for the inference worker. The worker
  opens the runtime, loads the published snapshot, and publishes routes when
  activation returns. Route publication and startup completion share one
  transaction. A superseded request cannot publish routes.
  ``train-from`` still imports the artifact into the trainer before it writes
  a ``train_from`` record.

``Scenario.serving_artifact_id()`` names the selected artifact.
``Scenario.served_artifact_id()`` reads the published route when a dispatcher
owns the scenario. Training staleness checks use this served identity.
Automatic weight training waits while the selected artifact has no matching
served route. Manual training does not wait, since a request names its samples.
Deployments without inference keep their existing training behavior.
Startup status is available through the inference service's
``GET /reef/scenarios/{scenario}/inference/status`` route.

``reef.inference.serving`` holds the two runtimes for saved weights, both over an
``InferenceBackend``:

* ``AdapterServingRuntime`` serves each scenario its own PEFT adapter on one
  base model. A snapshot holds the adapter in ``adapter/``. At most
  ``adapter_capacity`` adapters stay resident.
  ``reef.inference.residency.AdapterResidencyManager`` evicts a leaked
  adapter first, then the oldest-activated adapter that is no scenario's
  current one.
* ``FullWeightServingRuntime`` serves one scenario's full Hugging Face
  weights. The engines start with the base model. Each snapshot loads its
  files, a root snapshot's included, unless the engines serve its content
  already.

A root snapshot made from the default start has the layout of every other
snapshot of its component. An adapter runtime loads its untrained adapter,
and the Tinker runtime samples its untrained adapter.

Each runtime answers a base root (``base.json``, see `Artifact imports`_)
with the base model and asks ``ArtifactWeights`` for nothing:

* ``AdapterServingRuntime`` loads nothing. The scenario serves no adapter, so
  its requests name the base model, and ``activate_checkpoint`` returns
  ``base_version``. The adapter that the scenario served before stays
  resident until its route stops or another adapter needs its slot.
* ``FullWeightServingRuntime`` loads the directory ``model_path`` as full
  weights under a new version. It raises ``ReefError`` when ``model_path`` is
  no directory on its host.
* ``TinkerInferenceRuntime`` samples the base model. It raises ``ValueError``
  for a base root of another model or another service.

Two training scenarios can share a deployment only if its training component
allows it (``shares_scenarios``) and its runtime serves each scenario its own
weights. The Tinker component allows it, because each job carries its own
start. The Slime component does not: its workers hold one set of weights, so
a Slime deployment trains one scenario.

Each runtime passes a snapshot's directory as a ``WeightFiles`` value, with
the ``"adapter"`` or ``"full"`` layout, to
``InferenceBackend.load_weight_files``, under a new runtime load ID that every
engine then reports. The runtime kind ``remote_serving``
(``reef.inference.remote.runtime``) builds one of them over the engines of an
inference component, as the artifact format says: ``AdapterServingRuntime``
for ``adapter`` snapshots, ``FullWeightServingRuntime`` for ``full`` ones.
``serving_engine_config`` builds the engines' input from the ``inference``
settings and the artifact format. With a ``service``, each runtime checks its
health every ``health_interval_s`` seconds (default 30). After a failed check
it calls ``recover_engines``: it pauses generation, lets the backend replace
lost or dead engines (``InferenceBackend.recover``, which for an inference
component also covers a restarted component), loads what it serves into them
again (``reload_served``), and resumes.

A weight recipe holds neither the ``TrainingComponent`` nor the
``InferenceRuntime``. The process holds both beside the recipe:

.. code:: python

   recipe = SAORecipe()
   live = LiveObjects(runtime=inference, training_component=training)

A training deployment's component factory returns the ``TrainingComponent``
alone; a serving factory returns an ``InferenceRuntime``.
``TrainingComponentFactory`` and ``build_training_component`` in
``reef.train.component`` define and build the first; ``RuntimeFactory`` and
``RuntimeRegistry`` in ``reef.inference.registry`` define and build the second.
``TrainingComponentFactory.facts`` and ``training_backend_facts`` state a
backend's ``TrainingBackendFacts`` from the same config without a component:
its name, artifact format, default start inputs, parameters schema, root start
check, and whether scenarios and jobs share it.
The HTTP service builds the training component first, then the serving
runtime that ``reef.service.deploy.serving.serving_runtime_config`` describes
from the ``inference`` settings and the component's ``artifact_format``. The
inference service of a split deployment reads the ``artifact_format`` of the
facts and builds no component; it builds the component only for a backend
whose ``facts`` returns ``None``.

Each side owns its contracts. The inference contracts are the modules at the
root of ``reef.inference``: ``interfaces`` (inference contracts and runtime
load IDs), ``deployment`` (the inference service contracts), ``registry``
(runtime factories), ``serving`` (the saved-weight serving runtimes),
``residency`` (adapter residency) and ``recovery`` (engine supervision).
``reef.train.component`` holds the training component contract. Values that both sides read
are in ``reef.core.weight_layout`` (``ArtifactFormat``, ``WeightFiles``,
``ArtifactWeights``). ``reef.execution.executor`` contains worker transports.
Concrete integrations live with their owners:
``SlimeTrainingComponent`` in ``reef.train.slime_backend.component``,
``TinkerTrainingComponent`` in ``reef.train.tinker_backend.component``, and
``SGLangInferenceBackend`` in ``reef.inference.sglang.backend``.

Training jobs and components
----------------------------

``reef.train.component`` defines bounded training jobs. A ``TrainingJob`` has
these fields:

- ``job_id`` and ``backend``, the name of the training component.
- ``data``, a ``JobDataRef``: the job data directory that the job trains.
  ``read_data()`` returns its samples, ``data_dir`` is its local path, and
  ``data_uri`` its URI. ``data_id``, ``sample_count`` and ``kind`` describe
  the data.
- ``request_id``: the ID of the training request that the job trains, or
  ``None``.
- ``component``: the scenario component that the job trains, or ``None``.
- ``previous_job_id``: the last job whose end the trainer committed, or
  ``None`` before the first.
- ``start``: the directory of the weights the job starts from, or ``None``
  for the backend's base model.
- ``parameters``: the backend's parameters. Reef reads none of them; they
  override the backend's own configuration.
- ``scenario``: the scenario whose trainer submits the job. The jobs of one
  scenario run one at a time, in the order of their submission.

``job_id_for`` names a job by a hash of its backend, its start, its data and
its parameters. The weight trainer puts the scenario and the records of the
job's data into the data identity, so the ID covers the scenario, the
backend, the start, those records and the parameters. A retry of the same request is
therefore the same job.

A ``TrainingComponent`` runs whole jobs with its framework's own training
loop:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Contract
   * - ``name``
     - The backend name that the component's jobs carry.
   * - ``artifact_format``
     - A ``ArtifactFormat``: the weight files that each saved checkpoint's
       snapshot holds, ``layout`` ``full`` (a Hugging Face checkpoint) or
       ``adapter`` (a PEFT adapter in ``adapter/``, with ``adapter_rank`` and
       ``adapter_target_modules``). ``None`` when the snapshots hold no weight
       files that an engine loads, such as Tinker's manifests. The serving
       assembly configures the engines from it and refuses a serving runtime
       that cannot load it.
   * - ``default_start()``
     - The component's untrained checkpoint as a ``DefaultStart``, laid out
       as one of its snapshots, with the ``inputs`` that decide its content.
       A scenario created without an artifact item holds it in its root
       snapshot. Slime gives the ``hf_checkpoint`` directory, or with LoRA an
       adapter whose weights are zero. Tinker gives the checkpoint of its
       untrained adapter, which it makes only when a scenario's root needs it
       or the training process runs jobs.
   * - ``root_start(root)``
     - The start of the first job of a trainer without committed state, a
       non-empty string. ``root`` is a ``ScenarioRoot``: the trainer's
       component of the scenario's root snapshot and the root's
       ``RootSource``. The default is ``import_start`` of the root's files.
       Slime, which imports no snapshot, returns its startup weights when the
       root's source is its default start with equal ``inputs``, and raises
       ``ValueError`` naming both otherwise.
   * - ``submit(job)``
     - Record the job durably and return; the job runs in the background.
       Submitting a job ID again does nothing.
   * - ``status(job_id)``
     - A ``JobStatus``: ``state`` (``running``, ``finished`` or ``failed``),
       the ``SavedCheckpoint`` values so far in step order, the backend's
       ``metrics``, and ``error``, set exactly when the job failed. Raise
       ``KeyError`` for a job the component never saw. A restarted component
       resumes each unfinished job or reports it failed.
   * - ``cancel(job_id)``
     - Stop a running job, which then reports ``failed``.
   * - ``import_start(artifact_dir)``
     - Copy a published snapshot into the component's own storage and return
       it as a start. Reef calls it
       for a ``train-from``. Raise ``ValueError`` when no job can start from
       those weights.
   * - ``start_jobs()``
     - Start running jobs in this process: the training service's job worker
       calls it once. Other processes use the component without running jobs.
       The default does nothing.
   * - ``check_artifact(directory)``
     - Raise ``ValueError`` unless ``directory`` holds a snapshot in the
       layout of this component's snapshots. An artifact import calls it.
       The default reads ``artifact_format``: ``full`` needs ``config.json``
       and weight files, ``adapter`` needs a PEFT adapter in ``adapter/``
       with a rank of at most ``adapter_rank`` and target modules within
       ``adapter_target_modules``. ``None`` refuses. Tinker checks its
       manifest and looks up its ``state_path`` on its service. The Tinker
       API describes no sampler or exported-adapter path, so those are
       checked only for their ``tinker://`` form. The default and Tinker
       first accept a base root whose ``inputs`` equal
       ``default_start_inputs``, and raise ``ValueError`` that names each
       input that differs.
   * - ``close()``
     - Release the component's resources; safe to call repeatedly.

A ``SavedCheckpoint`` has a ``step`` within its job, a ``path`` that a later
job starts from, a ``artifact_path`` that Reef publishes and serving loads, and
the backend's ``metrics``. A finished job without checkpoints trained nothing,
for example because its objective skipped the job's data.

``reef.train.local_jobs.LocalJobComponent`` is the base of the built-in
components. It keeps one directory per job (``job.json``,
``checkpoints.json``, and ``result.json`` once the job ended). It runs one job
at a time in a worker thread, and it resumes unfinished jobs after a restart.
A subclass implements ``check_job(job)`` and ``run(job, directory)``. It must
call ``start_jobs()`` once it can train. Until then no job runs, including
the jobs that a restart left.

``reef.train.jobs.JobLoop`` is the control plane's side. It holds the
trainer's one training component. It returns the result that the trainer
commits:

1. When no job runs, the trainer reserves ready job data. The pipeline's
   sample-to-job stage gives the job's parameters, the loop submits it, and the
   trainer commits the submission, which consumes the records of the job's
   data.
2. While the job runs, the trainer commits each new saved checkpoint: its
   ``artifact_path`` becomes a new artifact, with the checkpoint's training
   metrics. The scenario's activation rule decides whether it serves.
3. When the job ends, the trainer commits its outcome.

Weight training and harness evolution share the loop: every job carries the
same fields. The loop sets ``TrainingJob.component``, the scenario component
that the trainer trains, and ``TrainingJob.previous_job_id``, the last job
whose end the trainer committed. A harness continues its state after that
job, since a job that keeps no tree still advances it; a weight component
ignores it. What a job needs beside its data comes from its stage's
``job_parameters``: the harness stage (``TraceJobData``) adds the training
request and the task paths of the scenario's eval split. On each checkpoint's
commit, the loop asks the component for the parts that the checkpoint changed
from the job's start (``TrainingComponent.changed_parts``, empty by default),
which the scenario's activation rule reads (``changed_kinds``).

Training is linear: the next job starts from the newest saved checkpoint,
whether or not it serves. A restarted job replaces the trainer's start
with its fresh start. A job continues the objective state of its start only
when it trains the same objective. The trainer commits the loop's state as its
algorithm state. This ``JobLoopState`` has four fields:

- ``start``: the start of the next job, or ``None`` for the base model.
- ``job``: the running job as a ``RunningJob``, or ``None``.
- ``finished_jobs``: the number of jobs that ended, finished or failed.
- ``last_job``: the outcome of the last job that ended, or ``None`` before
  the first one.

A failed job ends like a finished one. Reef consumed the records of its data when
it submitted the job, so they do not train again.

Once the commit of a job's end is durable, the samples service calls
``Record2Sample.job_ended(outcome)`` with ``last_job``. The default does nothing.
A trainer that starts with no running job calls it again for the last job
that ended, before it reads a record. The stage takes it as the
end of the last job that may have trained its samples.
A stage whose job data holds ``TaskSample`` values (Harbor tasks) reads each
task's rollout results from the outcome's ``metrics``: the Tinker and Slime
components report ``rollout_rewards``, the rewards of each task's rollouts, by
task name. A component that cannot roll out tasks refuses such a job in
``check_job``.

For a ``train-from``, Reef first imports the snapshot with ``import_start``
and cancels the running job. It then writes the ``train_from`` commit. The
next job starts from the imported snapshot. A component that cannot load a
snapshot raises ``ValueError``, and Reef refuses the request with HTTP 400.

Tinker integration
------------------

``reef.train.tinker_backend.launch.TinkerDeployment`` implements the optional
``tinker`` backend as an ``InProcessTrainingDeployment``. Its
``component_factory`` builds, from ``TinkerSettings`` and only after deployment
selection, a ``TinkerTrainingComponent``. Each job
trains remotely from the job's start manifest. Its parameters are ``objective``,
the optional ``learning_rate_schedule`` and ``scheduling``, optional ``batch_size``, ``kl_coef`` and
``learning_rate`` overrides, and ``save_interval``, the optimizer steps between saved
checkpoints (by default, one checkpoint when the job ends). A saved checkpoint
is both the start of a later job and the snapshot Reef publishes. A job of
Harbor tasks also takes ``rollout``, the rollouts of each task.

``TinkerDeployment.default_serving`` is ``trainer``: its
``trainer_serving_config`` selects ``reef.inference.tinker.TinkerInferenceRuntime``
(runtime kind ``tinker``), which reads each snapshot's
``tinker-checkpoint.json`` manifest and samples from the immutable sampler it
names. A snapshot holds that manifest alone, so the component's
``artifact_format`` is ``None``. Reef's engines can still serve the adapters
(``inference.serving: engines``): ``TinkerDeployment.engine_weights`` names
``reef.tinker.weights.TinkerSamplerWeights``, which downloads the adapter at
each snapshot's ``sampler_path`` and converts it into a PEFT adapter.
Train-from works: the component imports the snapshot's manifest as a start.

``TrainingDeployment`` (``reef.train.deployment``) is a training backend's
deployment definition. ``default_serving`` (``engines`` unless the backend
says otherwise) is the serving of a deployment that names none and sets no
engine setting, and ``trainer_serving_config(settings)`` the runtime config
of the backend's own serving, or ``None`` without one.
``engine_weights(settings)`` is an ``EngineWeights`` for a backend whose
snapshots hold no weight files: the ``ArtifactFormat`` that Reef's engines load
and the ``ArtifactWeights`` plugin (``reef.core.weight_layout``) that gives each
snapshot's files. The default is ``None``: engines cannot serve such
snapshots. ``ArtifactWeightFiles``, the default plugin, reads a snapshot's own files. ``requires_local_model``
defaults to true; hosted integrations set it to false to preserve remote
model identifiers during deployment resolution.

Methods implement ``TinkerLoss`` from ``reef.train.tinker_backend.losses``
with a ``loss_family`` name, and decorate the class with ``@register_loss_family``
from ``reef.train.tinker_backend.loss_families``, in the module that defines
their ``TinkerObjective``. ``loss_fn`` names the Tinker built-in loss the
family trains with; ``inputs(rows, base_logprobs, kl_coef=...)`` returns that
function's token-aligned ``loss_fn_inputs``, one dictionary per row; and
``needs_base_logprobs`` requests frozen-base scores for a nonzero KL
coefficient. ``check_kl_coef(kl_coef)`` raises ``ValueError`` for a
coefficient the family cannot train; Reef calls it when it prepares a job,
before any remote work. The default accepts every coefficient. For ``importance_sampling``, ``TokenRow.inputs`` builds the
dictionary from response-token advantages, shifting prediction targets by one
token and padding prompt positions with zeros. A ``TinkerCustomLoss`` adds
``loss(data, logprobs)``, which the SDK boundary passes to
``forward_backward_custom`` so the family computes its loss tensor on the
Reef host. ``Datum`` objects are constructed only after this shaping. See
`Train with Tinker <../user-guide/tinker.rst>`__ for supported contracts.
