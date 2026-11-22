Recipe and report API
=====================

The contracts that configure a deployment: the recipe classes and their members, the objects that run a recipe, recipe configuration, the report type, the activation rule, and the embedded service.

`Write a recipe <../developer-guide/write-a-recipe.rst>`__ teaches how to choose and write a recipe. `Control service <../developer-guide/recipe-control-service.rst>`__ explains report types and activation.

Recipe
------

.. code:: python

   from reef.recipe import Recipe, WeightTrainingRecipe, config_field
   from reef.recipe.harness import HarnessEvolutionRecipe
   from reef.recipe.sft import SFTRecipe

A recipe is one frozen dataclass that configures the serving and evolution behavior of every scenario in a deployment. Every member is a function of its configuration. A recipe has no runtime, no training component and no store, and it builds none.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Base class
     - Use and subclasses
   * - ``Recipe`` (``reef.recipe``)
     - Records traffic and makes no update: its trainer only reads records. Subclass it and override ``build_components()`` to serve an externally produced artifact.
   * - ``WeightTrainingRecipe`` (``reef.recipe``)
     - Trains and publishes weights. To evolve a different artifact, deploy the subclass with a ``TrainingComponent`` that makes the artifact, and override ``build_components()``. Subclasses: ``SFTRecipe``, ``DatasetRecipe`` and ``GeneratedTaskRecipe`` under ``reef.recipe``; ``SAORecipe``, ``TTTDRecipe``, ``SDFTRecipe``, ``SDPORecipe``, ``OPDRecipe`` and ``OpenClawRLRecipe`` under ``recipes``; ``CoralRecipe`` (beta).
   * - ``HarnessEvolutionRecipe`` (``reef.recipe.harness``)
     - Runs Reef's harness loop with the ``Method`` that ``evolution.method`` names. GEPA, Meta-Harness, ACE and ReasoningBank are methods, not recipes. Subclass the recipe only when the method changes more than the tree, such as the files that a client pulls. ``ReefineRecipe`` (beta) only sets its defaults.
   * - ``CompositeRecipe`` (``reef.recipe.composite``)
     - Maps component names to child recipes, to evolve weights and a harness (or configuration) together. Each trainer runs as its own worker. It has no subclass.

Choose the narrowest class whose assumptions all hold.

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
     - The ``ReportBase`` subclass that the recipe's reports parse as. ``None`` keeps ingress open. A composite declares none.
   * - ``pipeline_spec()``
     - Class method. The default class of each pipeline stage, a ``PipelineSpec`` (`Pipeline stages <python-api-pipeline.rst>`__).
   * - ``stages()``
     - ``PipelineSettings``: the two stage classes and their configs, the deployment's ``pipeline`` section over ``pipeline_spec()``.
   * - ``build_components(scenario)``
     - ``ScenarioComponents``: the components that the configuration declares for the scenario's snapshots, each with its kind and admission check (default ``AcceptAnyArtifact()``). See `Scenario components <python-api-artifacts.rst#scenario-components>`__.
   * - ``default_start()``
     - ``DefaultStart``: the content of a new scenario's root snapshot that the configuration gives; empty by default.
   * - ``activation_rule(reference=None)``
     - The rule that decides whether an artifact that the recipe's trainers make serves. The deployment's ``activation`` section names it (`Activation rule`_).

``WeightTrainingRecipe`` adds:

- ``default_parameters()``: class method. The recipe's own parameters of each training job by the kind of its samples, normally the ``objective``.
- ``parameters_by_kind()``: those parameters with the ``job_parameters`` config over them (`Job parameters <python-api-training.rst#job-parameters>`__).

``HarnessEvolutionRecipe`` adds:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Member
     - Value
   * - ``evolution``, ``evaluation``
     - The parsed ``evolution`` section (``EvolutionSettings``, in ``reef.train.harness_backend.settings``) and its ``evaluation`` section (``EvaluationSettings``). They name the method, the scorer and the collector as plugin settings.
   * - ``seed``, ``models``, ``model_metadata``, ``worker_executor``, ``model_name``
     - The first tree, the method's named models, the model metadata, the placement of the run workers and the served model's name.
   * - ``harness_settings()``
     - The ``HarnessSettings`` that the harness training component builds itself from.
   * - ``runs_offline()``, ``task_directories(descriptor)``, ``configured_tasks()``
     - Functions of the configuration.

`A minimal weight recipe <../developer-guide/write-a-recipe.rst#a-minimal-weight-recipe>`__ shows a subclass. The base requires ``frozen=True``. ``kw_only=True`` keeps later fields keyword-only. Call ``super().__post_init__()`` first when you add validation.

A ``CompositeRecipe`` computes nothing from its children. Its ``__post_init__`` checks the mapping: at least two entries, valid component names, no nested composite, one ``training_mode``, at most one ``WeightTrainingRecipe`` and at most one ``HarnessEvolutionRecipe``.

``reef.recipe.composite.component_recipes(recipe)`` returns the recipes of a deployment: the pairs ``(component name, child)`` of a composite, and the one pair ``(None, recipe)`` of any other recipe.

Live objects
~~~~~~~~~~~~

.. code:: python

   from reef.control.scenario.live_objects import LiveObjects, ServedEndpoint

The process that runs a service builds the objects that run the recipe, and holds them in one frozen ``LiveObjects`` value. ``reef.service.assembly.load_deployment(settings, env, runs_training=...)`` returns the recipe and the value, and ``Dispatcher(recipe, backend_factory, live=live, ...)`` receives it.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Value
   * - ``runtime``
     - The serving ``InferenceRuntime``, or ``None``.
   * - ``training_facts``
     - The ``TrainingBackendFacts`` of the backend of the recipe that trains weights, or ``None``. A ``WeightTrainingRecipe`` needs the runtime and these facts; otherwise ``check_recipe_objects`` raises ``RecipeConfigError``.
   * - ``training_component``
     - The ``TrainingComponent`` that runs the jobs of that recipe, or ``None``. Only a process that runs the training service has one. A value with a component has the facts of the same backend.
   * - ``harness_facts``
     - The ``HarnessFacts`` of the deployment's harness recipe, or ``None``.
   * - ``harness_component``
     - The training component that runs the jobs of the harness recipe, in a process that runs the training service, or ``None``.
   * - ``data_store``
     - The deployment's ``DataStore``, which keeps the data directory of each training job, or ``None``.
   * - ``served_endpoint``
     - A ``ServedEndpoint(url, token)`` where this Reef answers inference itself, or ``None``.

``dispatcher.live`` holds the value as the trainers see it: each component is the jobs queue's view of the backend whose facts the process holds (``reef.train.job_queue.JobQueueComponent``). ``dispatcher.job_components`` are the components that the process built, which is none outside the training service. Read the serving runtime as ``dispatcher.live.runtime``.

A recipe builds no trainer. The scenario factory builds one trainer for each component from the recipe and the live objects, with ``Trainer.build``. Nothing replaces this path. A recipe that is not a composite has one trainer, so its scenario serves at most one component.

Before Reef registers a scenario with an item start, the facts of each recipe's backend check the root (``TrainingBackendFacts.check_root_start``). A refusal raises ``ValueError`` and leaves the scenario unregistered.

The training service builds the harness training component with ``HarnessTrainingComponent.from_settings``. Code that already holds the method, executor and scorer, such as an example driver, calls ``HarnessTrainingComponent(settings, models, method=..., executor=..., scorer=..., collector=..., agent=...)`` and gives the dispatcher ``LiveObjects(runtime=runtime, harness_component=component)``.

Configuration
~~~~~~~~~~~~~

``config_field()`` is a ``dataclasses.field`` that carries a default, a type-aware parser, and an optional environment fallback:

.. code:: python

   clip_ratio: float = config_field(0.2, env="REEF_MY_METHOD_CLIP_RATIO")

Precedence is explicit configuration, then environment, then the default.

A recipe class does not load itself. ``reef.recipe.loading.load_recipe(recipe_class, config, environ)`` reads the declarations of the class and builds the recipe. ``build_recipe(reference, environ, config)`` does the same for a class reference. The declarations are:

- ``config_field`` fields;
- ``section_field`` fields, one parsed config section each;
- ``reader_field`` fields, each read by its ``ConfigReader``;
- ``section_defaults``, the class's defaults for its sections, under the deployment's keys;
- ``refused_sections``, the config sections that the class refuses, each with its error message.

For a weight-training deployment, ``reef.service.deploy.training.training_recipe_config`` forwards the declared fields from the service configuration and refuses a key that the class does not declare. A setting that a pipeline stage reads belongs in the stage's settings, under the deployment's ``pipeline`` section, not in a recipe field. Never read deployment YAML from inside a stage.

Report
------

.. code:: python

   from reef.core.reports import ReportBase, ReportValidationError, ScoredRolloutReport, TeacherContextReport

A report type declares the feedback that a method accepts, so malformed input fails at ingress with HTTP 400. Two contracts are shared:

- ``ScoredRolloutReport``: score only.
- ``TeacherContextReport``: the contract of the distilling recipes. It holds a rollout's receipt and the ``teacher_context`` that its teacher sees (a demonstration, environment feedback, or nothing for a separate teacher).

Method-specific report contracts live in their method package. Reef does not import or re-export them.

A report type is a frozen dataclass that inherits ``ReportBase``. Override ``validate()`` for rules between fields, and raise ``ReportValidationError`` (`Report type <../developer-guide/recipe-control-service.rst#report-type>`__ has an example). Declare it through the recipe's ``report_type``. ``score`` uses the top-level ``score`` channel; every other field uses ``metadata.<field>``. Producers can attach extra fields that the schema does not declare.

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

- Each supported type can also be written as ``T | None``. JSON ``null`` is accepted only when the annotation permits it.
- Any other annotation, any other Union included, is a declaration error. It raises ``TypeError`` when Reef first inspects the type.
- A field without a default is required. A field with one can be absent.

The same type serves both sides: a producer constructs it and calls ``to_dict()``; a stage receives the parsed instance as ``context.parsed_report``.

Activation rule
---------------

.. code:: python

   from reef.core.activation import ActivationRule, MadeArtifact

Training makes artifacts and never activates one. A scenario's ``ActivationRule`` decides whether an artifact serves:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - Contract
   * - ``activates(made)``
     - True when Reef activates the artifact as soon as training publishes it. ``made`` is a ``MadeArtifact``: the ``component`` and the ``changed_kinds`` of a harness tree. The committer asks when a training step publishes an artifact; the commit record's ``activated`` keeps the answer.
   * - ``activates_evaluated(artifact_id, evaluation)``
     - True when Reef activates the artifact once an evaluation of it finished with the ``PolicyEvaluation`` ``evaluation``; false by default. The artifact evaluation service asks when an evaluation finishes.

The built-in rules are ``AlwaysActivate``, ``NeverActivate`` and ``EvaluatedActivation`` in ``reef.evaluation.activation``. The deployment names a rule in its ``activation`` section (`Configuration <configuration-recipe.rst#the-activation-section>`__). A rule of your own inherits ``ActivationRule``. Reef builds it from its config with the class method ``from_config(config)``.

Embed the service
-----------------

An application can run the complete Reef service in its own process. ``reef.service.assembly.build_app`` makes the dispatcher and the HTTP app from a ``ServiceSettings``. The app includes record retention, the inference routes, the evaluation token, readiness of local cycles, and cleanup.

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

- ``inference_handler`` is optional. If you give one, it runs every inference request in place of the recipe runtime's handler. You own it: Reef does not close it. Release it after the app stops.
- Add middleware and routes before the app starts. The service tokens also protect the routes that you add.
- ``get_dispatcher(app)`` returns the dispatcher of an app that ``build_app`` or ``create_app`` made.
- ``build_app`` owns the dispatcher that it makes. The app closes it on cleanup. If ``build_app`` fails, it closes the resources that it made and raises the original error.
- ``create_app(dispatcher, ...)`` closes the dispatcher only if you set ``close_dispatcher=True``. It never creates a default storage backend.
