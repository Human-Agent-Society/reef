Python API for custom learning methods
======================================

The Python API is the set of base classes that a learning method, a storage plugin or a serving integration implements. This page says where each contract is.

The docstring of each class is the reference for its contract: it gives the signature, the default and the errors of each member. The guide in the last column explains how to use the contract. Reef owns everything around the contracts: it accepts and replays records, holds a job's data until it is acknowledged, commits algorithm state, submits training jobs, and publishes the saved checkpoints.

Every contract is an abstract base class. Inherit it explicitly and implement each abstract member. Reef rejects an object that only has methods of the same names. Optional capabilities are separate base classes: inherit only those that the implementation supports.

Method code must depend only on the classes that this page and the guides name, and on what the docstrings of those classes name. Anything else under ``reef.`` is an implementation detail and can change.

Learning methods
----------------

Start with a recipe and add only what the method needs.

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - Need
     - Classes
     - Guide
   * - Configure the behavior of a deployment
     - ``Recipe``, ``WeightTrainingRecipe`` (``reef.recipe.base``), ``HarnessEvolutionRecipe`` (``reef.recipe.harness``), ``CompositeRecipe`` (``reef.recipe.composite``)
     - `Write a recipe <../developer-guide/write-a-recipe.rst>`__
   * - Validate the feedback of a report
     - ``ReportBase`` (``reef.core.reports``)
     - `Report type <../developer-guide/recipe-control-service.rst#report-type>`__
   * - Decide which artifacts serve
     - ``ActivationRule`` (``reef.core.activation``)
     - `Activation rule <../developer-guide/recipe-control-service.rst#activation-rule>`__
   * - Turn records into samples
     - ``Record2Sample``, ``ReportSamples``, ``JudgedSamples``, ``TrajectorySamples`` (``reef.samples``)
     - `Samples service <../developer-guide/recipe-samples-service.rst>`__
   * - Generate tasks
     - ``TaskGenerator`` (``reef.recipe.generated_tasks.generator``)
     - `Task generators <../developer-guide/task-generators.rst>`__
   * - Shape samples into the data of a job
     - ``Sample2Job`` (``reef.train.sample2job``); ``JobData``, ``TrajectorySample``, ``TaskSample``, ``DelayedSample`` (``reef.core.job_data``)
     - `Sample-to-job stage <../developer-guide/recipe-control-service.rst#sample-to-job-stage>`__
   * - Turn the data of a job into a training signal
     - ``SlimeObjective`` (``reef.train.slime_backend.objectives``), ``TinkerObjective`` (``reef.train.tinker_backend.objectives``)
     - `The objective <../developer-guide/recipe-training-service.rst#the-objective>`__
   * - Add a tensor loss
     - ``register_loss_family`` (``reef.train.slime_backend.loss_families``), ``TinkerLoss`` (``reef.train.tinker_backend.losses``)
     - `Loss families <../developer-guide/loss-families.rst>`__
   * - Evolve a harness tree, and score a run
     - ``Method``, ``Workspace``, ``Services`` (``reef.train.harness_backend``), ``TaskScorer`` (``reef.evaluation.harness``)
     - `Write a harness method <../developer-guide/write-a-harness-method.rst>`__
   * - Declare a new component of a snapshot
     - ``ComponentSpec`` (``reef.artifact.components``)
     - `Scenario components <../developer-guide/components.rst>`__

Integrations and plugins
------------------------

A deployment names each of these in its config, as a built-in name or as ``package.module:ClassName``.

.. list-table::
   :header-rows: 1
   :widths: 22 38 18 22

   * - Need
     - Classes
     - Setting
     - Guide
   * - Run training jobs on a new backend
     - ``TrainingComponent``, ``TrainingBackendFacts`` (``reef.train.component``), ``TrainingDeployment`` (``reef.train.deployment``)
     - ``training.backend``
     - `Training backends <../developer-guide/training-backends.rst>`__
   * - Decide when a scenario trains
     - ``TrainingManager`` (``reef.train.manager``)
     - ``training.manager``
     - `Decide what a scenario trains <../developer-guide/training-backends.rst#decide-what-a-scenario-trains>`__
   * - Serve published weights
     - ``InferenceRuntime``, ``InferenceBackend``, ``InferenceHandler`` (``reef.inference.interfaces``), ``InferenceService`` (``reef.inference.deployment``)
     - ``inference.backend``
     - `Serving runtimes <../developer-guide/serving-runtimes.rst>`__
   * - Start workers in another way
     - ``Executor`` (``reef.execution.executor.base``)
     - ``backend`` of an ``execution`` setting
     - `Executors <../developer-guide/executors.rst>`__
   * - Store commits, settings and records
     - ``ScenarioStorage``, ``ScenarioStore`` (``reef.storage.scenario.store``), ``RecordStore`` (``reef.storage.scenario.records``)
     - ``storage.record-backend``
     - `Choose the storage backend <../user-guide/storage.rst#choose-the-storage-backend>`__
   * - Store data items
     - ``DataStore`` (``reef.storage.data.store``)
     - ``storage.data-store``
     - `Choose the data store <../user-guide/storage.rst#choose-the-data-store>`__
   * - Store the files of tasks
     - ``TaskFileStore`` (``reef.storage.data.tasks``)
     - ``storage.task-file-backend``
     - `Choose the task file store <../user-guide/storage.rst#choose-the-task-file-store>`__
   * - Store large content
     - ``SegmentStore`` (``reef.storage.data.segments``)
     - ``storage.segment-backend``
     - `Keep large content in segment files <../user-guide/storage.rst#keep-large-content-in-segment-files>`__
   * - Store artifacts
     - ``ConfigurableRepositoryBackendFactory``, ``RepositoryBackend`` (``reef.artifact.repository``)
     - ``storage.artifact-backend``
     - `Choose the artifact backend <../user-guide/storage.rst#choose-the-artifact-backend>`__
   * - Read an artifact that a client imports
     - ``ArtifactSourceReader`` (``reef.artifact.sources``)
     - ``storage.artifact-sources``
     - `Import an artifact <../user-guide/operate.rst#import-an-artifact>`__
   * - Limit the scenarios that a service holds
     - ``ScenarioAdmission`` (``reef.control.scenario.admission``)
     - ``reef.scenario-admission``
     - `Limit the scenarios a service holds <../user-guide/operate.rst#limit-the-scenarios-a-service-holds>`__
   * - Make the sandbox templates of a task
     - ``TaskTemplateBuilder`` (``reef.evaluation.task_templates``)
     - ``evaluation.task-template-builder``
     - `Build templates before the trials <../developer-guide/sandboxes.rst#build-templates-before-the-trials>`__
   * - Keep more of an evaluation trial
     - ``TrialObserver``, ``EvaluationTrial`` (``reef.evaluation.trials``)
     - ``evaluation.trial-observer``
     - `Keep the files of a trial <../developer-guide/sandboxes.rst#keep-the-files-of-a-trial>`__

An agent connects through an adapter, which is a file and not a class: `Harness adapters <../developer-guide/harness-adapters.rst>`__.

Embed the service
-----------------

An application can run the complete Reef service in its own process. ``reef.api.assembly.build_app`` makes the dispatcher and the HTTP app from a ``ServiceSettings``. The application adds its routes before it starts the app, and the service tokens protect them too.

.. code:: python

   from aiohttp import web

   from reef.api.app import get_dispatcher
   from reef.api.assembly import build_app

   app = build_app(settings, inference_handler=handler)

   async def scenario_count(request: web.Request) -> web.Response:
       dispatcher = get_dispatcher(request.app)
       return web.json_response({"scenarios": len(dispatcher.list_scenarios())})

   app.router.add_get("/my-app/scenario-count", scenario_count)
   web.run_app(app, host=settings.host, port=settings.port)

``inference_handler`` is optional: it gives the app an ``InferenceHandler`` that the application made. Such a handler needs only the inference methods.

The app of ``build_app`` also runs record retention and the data store's maintenance. An application that uses a ``Dispatcher`` without this app runs them itself, at intervals: ``Dispatcher.prune_record_archives`` with a ``RecordRetention`` (``reef.storage.scenario.records``), and ``DataStore.maintain``.
