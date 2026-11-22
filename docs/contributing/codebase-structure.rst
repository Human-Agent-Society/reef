Codebase structure
==================

This page says which package owns a change, and maps the packages to the seven services of a deployment.

Each package's ``__init__`` docstring states the boundaries that it holds and how to extend it. There are no READMEs under ``reef/``. Read the docstring of the package that you change; this page only helps you find it.

Packages
--------

`Architecture <architecture.rst>`__ gives the four layers of the packages and what each package is for. The table below says which package owns a change.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Package
     - Owns
     - Does not own
   * - ``reef/core/``
     - Shared value types, wire shapes, artifact identity, root errors
     - Storage, I/O, runtime behavior
   * - ``reef/api/``
     - HTTP routes, auth, streaming, the app of each service
     - Training methods, domain logic tied to aiohttp
   * - ``reef/deploy/``
     - ``reef serve``: the configuration of a deployment, the startup of its processes, and their lifecycle
     - HTTP routes, request handling
   * - ``reef/control/``
     - The loaded scenarios. ``scenario/`` holds scenario binding, commit ordering, recovery and lifecycle, and ``LiveObjects``: the runtime, training components and data store that one process builds for the recipe. The dispatcher coordinates the loaded scenarios, and ``workers/`` holds control's background workers. ``operations/`` holds the operations that a client asks for, without HTTP: data items, artifact evaluations, artifact imports, the harness tree and its sync, and the reads of a scenario's files and samples.
     - A concrete record backend, HTTP, training algorithms, repository implementations
   * - ``reef/recipe/``
     - The contract that a method implements, dotted resolution, the recipe loader, the base harness evolution recipe, the built-in SFT and generated-task recipes. A recipe is configuration only.
     - Cookbook methods, runtimes, training components, stores
   * - ``reef/train/``
     - The trainer, the training job loop, the training pipeline, job data types, training components and their backends
     - HTTP endpoints, deployment configuration parsing
   * - ``reef/samples/``
     - Which records Reef uses, which train together, and their samples (``Record2Sample``, ``ReportSamples``, ``JudgedSamples``); the samples service; the generator service that writes, checks and runs Harbor tasks; the context that the training pipeline and its stages read (``context``)
     - Job data shape, record storage, training loops
   * - ``reef/train/sample2job/``
     - Shaping the samples of the next job into one job's data (``Sample2Job``), and the job size
     - Which records train, the training loop, step signals
   * - ``reef/evaluation/``
     - Candidate evaluation: running and scoring evaluation runs, the harness history and its score cache, the harness publication policies, the activation rule
     - Making candidates, training loops, artifact publication
   * - ``reef/inference/``
     - Inference contracts, serving runtimes and their selection, saved-weight serving, concrete inference backends, native engine control, request adaptation
     - Training algorithms, publication decisions
   * - ``reef/routing/``
     - The Semantic Router integration: the routing manifest and its checks of a request, the routing preview, and the router container of a managed deployment
     - HTTP routes, the recording of a routed request, the ``reef serve`` process
   * - ``reef/tinker/``
     - The Tinker SDK pieces that the Tinker training component and the Tinker serving runtime share
     - Either side's own logic
   * - ``reef/artifact/``
     - Artifact bytes, repositories, materialization, snapshot heads, and the components that a recipe declares
     - Commit policy, weight loading
   * - ``reef/storage/``
     - The storage contracts and their SQLite and PostgreSQL implementations (``sql/``): the record, commit, sample and evaluation stores of a scenario (``scenario/``), the data store with its tasks (``data/``), and the coordination store with the work queues, ``served_models`` and the control lease (``coordination/``)
     - Scenario coordination, training
   * - ``reef/harness/``
     - Adapters (the parts of each agent's tree and its descriptor), the source tree (patches, validation, rendering), task runs, runners, trajectories, and the loopback gateway of an agent process
     - Recipe policy, the snapshot chain
   * - ``reef/execution/``
     - Job scheduling, process ownership, worker executors, sandbox isolation, file transfer, and the queue worker loop
     - Harness rendering, evaluation, recipe policy
   * - ``recipes/``
     - One method per package: the recipe, processor and objective of a weight method, or the ``Method`` of a harness method, with its runnable examples. Beta methods live under ``recipes/beta/``.
     - Shared machinery, another method
   * - ``tests/``
     - Repository-level tests grouped by responsibility
     - Tests hidden inside an integration subtree
   * - ``docker/``
     - Container and GPU environment setup
     - Python dependency declarations

Nothing under ``reef/`` imports a method package. A weight method keeps its objective and loss family in a ``slime/`` subpackage that only the training service imports. A harness method holds its ``Method``, its prompts, and a scorer when it needs its own.

Services and the shared store
-----------------------------

A deployment runs seven services (``SERVICE_NAMES`` in ``reef/execution/workers.py``). ``reef serve`` runs all of them in one process, and ``reef serve --component NAME`` runs one. No service calls another: each reads and writes the shared store. `Run the services in separate processes <../user-guide/split-services.rst>`__ describes the layout for operators.

``reef/api/services.py`` holds one app builder per service (``SERVICE_APPS``). Control, data, harness, samples and evaluation each build a ``Dispatcher`` (``build_dispatcher`` in ``reef/api/assembly.py``) that runs only that service's workers. Inference and training build no dispatcher.

.. list-table::
   :header-rows: 1
   :widths: 11 17 30 21 21

   * - Service
     - Builder
     - Code that it runs
     - Takes from
     - Writes
   * - ``control``
     - ``ControlApp``
     - ``reef/api/routes/`` (all but the data routes), ``reef/control/``; the trainers' job loops (``reef/train/jobs.py``)
     - The control lease; the ``data_sync``, ``harness_sync``, ``scenario_deletes`` and ``training_requests`` queues; the results of every other queue
     - Commits and settings; items of the ``serving``, ``training_checks``, ``jobs``, ``samples``, ``evaluations`` and ``data`` queues
   * - ``inference``
     - ``InferenceApp``
     - ``reef/api/inference_service.py``, ``ServingWorker`` in ``reef/control/workers/serving.py``, the runtime in ``reef/inference/``
     - The ``serving`` queue; the ``inference-owner`` lease; ``served_models``
     - ``served_models`` routes and stops; records
   * - ``samples``
     - ``QueueServiceApp``
     - ``SamplesWorker`` in ``reef/samples/service/``
     - The ``samples`` queue
     - Samples
   * - ``training``
     - ``TrainingApp``
     - ``TrainingJobWorker`` and ``TrainingCheckWorker`` in ``reef/train/job_queue.py``; the training component of ``reef/train/``
     - The ``jobs`` and ``training_checks`` queues; job data directories; start snapshots
     - Job progress and results; checkpoint content in the artifact backend
   * - ``evaluation``
     - ``QueueServiceApp``
     - ``EvaluationRolloutWorker`` and ``TaskTemplateWorker`` in ``reef/evaluation/rollout_worker.py``
     - The ``evaluations`` queue
     - Rollout results; ``serve_artifact`` items of the ``serving`` queue; the calls of each rollout
   * - ``data``
     - ``DataApp``
     - ``reef/api/routes/data.py``, ``reef/api/routes/scenario_files.py``, ``reef/control/operations/data.py``, ``DataWorker`` in ``reef/control/operations/data_service.py``
     - The ``data`` queue (artifact copies, imports, exports)
     - Data items and segments; ``data_sync`` items
   * - ``harness``
     - ``HarnessApp``
     - ``reef/api/routes/harness.py``
     - The commit history and the artifact backend, read-only
     - ``harness_sync`` items

The shared store has four parts. Each row names the module that defines its contract:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Part
     - Contract
     - Content
   * - Coordination store
     - ``reef/storage/coordination/store.py``, with SQL tables in ``reef/storage/sql/coordination.py``
     - The work queues (``reef/storage/coordination/work_queue.py``, table ``work_item``), ``served_models`` (``reef/storage/coordination/served_models.py``, tables ``served_model`` and ``scheduled_stop``), and the control lease (``reef/storage/coordination/control_lease.py``)
   * - Scenario storage
     - ``reef/storage/scenario/store.py``
     - The commit history, settings, tasks and evaluations of each scenario
   * - Data store
     - ``reef/storage/data/store.py``
     - Data items: records, trajectories, tasks, samples, artifact items, and job data directories
   * - Artifact backend
     - ``reef/artifact/``
     - The snapshots of each scenario, and the stored content of checkpoints

A queue name is a constant beside the worker that takes its items: ``JOBS_QUEUE`` and ``CHECKS_QUEUE`` in ``reef/train/job_queue.py``, ``SAMPLES_QUEUE`` in ``reef/samples/service/worker.py``, ``EVALUATIONS_QUEUE`` in ``reef/evaluation/rollout_worker.py``, ``SERVING_QUEUE`` in ``reef/storage/coordination/served_models.py``, ``DATA_QUEUE`` and ``DATA_SYNC_QUEUE`` in ``reef/storage/data/queue.py``, ``HARNESS_SYNC_QUEUE`` in ``reef/control/operations/harness_imports.py``, ``DELETES_QUEUE`` in ``reef/control/workers/scenario_deletes.py``, and ``TRAINING_REQUESTS_QUEUE`` in ``reef/train/training_requests.py``. ``QueueWorker`` in ``reef/execution/workers.py`` is the loop that takes, leases and finishes items.

Choose a destination
--------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - The change
     - Goes in
   * - A value or error that unrelated layers need, without behavior
     - ``reef/core/``
   * - Scenario state, commit ordering, recovery, activation, training starts
     - ``reef/control/scenario/``. The persisted ``CommitRecord`` is in ``reef/storage/scenario/commits.py``.
   * - A record, task, sample or data store contract, or its implementation
     - ``reef/storage/``. Implementations depend on the contracts, never the reverse.
   * - Persisting or materializing versioned bytes, or declaring a snapshot component
     - ``reef/artifact/``. Loading those bytes into an engine is ``reef/inference/``.
   * - An inference contract
     - A module at the root of ``reef/inference/``. A concrete engine goes in ``reef/inference/<integration>/``.
   * - A training component contract
     - ``reef/train/component.py``. A concrete training stack goes in ``reef/train/<integration>/``.
   * - A value that both training and inference read
     - ``reef/core/``, such as ``reef/core/weight_layout.py``
   * - Which records train, in which groups, and how a record set becomes samples
     - ``reef/samples/``
   * - Shaping samples into job data
     - ``reef/train/sample2job/``. ``reef/train/pipeline.py`` drives both stages; a recipe selects and binds them.
   * - Turning job data into a step signal
     - The training backend that trains it: ``reef/train/slime_backend/`` or ``reef/train/tinker_backend/``. The backends share no code.
   * - Writing, checking or running generated Harbor tasks
     - ``reef/samples/generator/``. The built-in recipe that trains on them is ``reef/recipe/generated_tasks/``.
   * - One service telling another what to do
     - A queue item or a table of the shared store, not a call (`Services and the shared store`_)
   * - A data route
     - ``reef/control/operations/data.py``, with the aiohttp adapter in ``reef/api/routes/data.py``
   * - HTTP-specific code
     - ``reef/api/routes/``. Transport-independent behavior goes in a service or domain object.
   * - A benchmark, task, grader or external environment
     - ``recipes/<name>/examples/``, or the external harness

**Harness evolution.** Each part has one owner. Data, evaluation and publication are shared with weight training; evolve and sync are the harness's own.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Part
     - Owner
   * - Tree
     - ``reef/harness/tree/`` holds the tree that every part reads. ``reef/harness/adapters/parts.py`` defines the parts that a tree may hold, and each adapter declares its own in its ``adapter.yaml`` (`Harness adapters <../developer-guide/harness-adapters.rst>`__).
   * - Data
     - The training pipeline turns the agent's recorded calls, their reports and the uploaded tasks into a job's data. The harness backend only reads job data.
   * - Evolve
     - ``reef/train/harness_backend/backend.py`` runs one job through evolve, evaluation and publication. It calls ``Method.run`` once in the job's workspace and reads back the tree. The method contract is in the package docstring (`Write a harness method <../developer-guide/write-a-harness-method.rst>`__).
   * - Evaluation
     - ``reef/evaluation/`` holds the contract that every candidate passes, weights and trees alike. ``reef/evaluation/harness/evaluator.py`` scores trees by running tasks, and ``reef/evaluation/harness/policy.py`` decides.
   * - Publication
     - The harness backend turns a selected tree into the job's snapshot. The scenario committer publishes it, and the activation rule (``reef/evaluation/activation.py``) decides whether it serves.
   * - Sync
     - ``reef/harness/sync.py`` moves a tree between Reef and a person's machine, in both directions. The client half is ``reef_client.harness``.

Package import direction
------------------------

Imports between the top-level Reef packages form a directed acyclic graph. ``tests/contributing/test_dependency_boundaries.py`` scans every Python file, including relative imports and imports inside functions. Importing the root ``reef`` facade from an internal package also counts as a dependency; internal code imports the owning module directly.

Concrete integrations may depend on shared contracts. Shared contracts never import a concrete integration.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Package
     - Imports
   * - ``core``
     - None
   * - ``storage``, ``artifact``, ``tinker``
     - ``core``
   * - ``execution``
     - ``storage``, ``core``
   * - ``observability``
     - ``artifact``, ``storage``, ``core``
   * - ``inference``
     - ``tinker``, ``execution``, ``artifact``, ``core``
   * - ``routing``
     - ``inference``, ``storage``, ``core``
   * - ``harness``
     - ``inference``, ``execution``, ``artifact``, ``core``
   * - ``samples``
     - ``harness``, ``observability``, ``execution``, ``storage``, ``core``
   * - ``evaluation``
     - ``inference``, ``harness``, ``execution``, ``artifact``, ``storage``, ``core``
   * - ``train``
     - ``evaluation``, ``samples``, ``harness``, ``tinker``, ``observability``, ``execution``, ``artifact``, ``storage``, ``core``
   * - ``recipe``
     - ``train``, ``evaluation``, ``samples``, ``harness``, ``execution``, ``artifact``, ``storage``, ``core``
   * - ``control``
     - ``recipe``, ``train``, ``evaluation``, ``samples``, ``inference``, ``harness``, ``observability``, ``execution``, ``artifact``, ``storage``, ``core``
   * - ``api``
     - ``control``, ``recipe``, ``train``, ``evaluation``, ``inference``, ``routing``, ``harness``, ``observability``, ``execution``, ``artifact``, ``storage``, ``core``
   * - ``deploy``
     - ``api``, ``recipe``, ``samples``, ``inference``, ``routing``, ``execution``, ``artifact``, ``core``
   * - ``cli``
     - ``deploy``, ``core``

Three rules follow from the graph:

- **Training and inference never import each other.** ``train/`` never imports ``inference/``, and ``inference/`` never imports ``train/``. The values that both read are in ``reef/core/``: the weight layout (``core/weight_layout.py``), chat rendering (``core/chat_rendering.py``) and the reasoning split (``core/reasoning.py``). ``execution/executor/`` owns worker launch and control RPC for both sides. No training backend builds serving: ``service/deploy/serving.py`` selects it from the ``inference`` section and the component's snapshot format.
- **The inference contracts import no integration.** The modules at the root of ``inference/`` are the contracts: ``interfaces.py`` (``InferenceRuntime``, ``InferenceBackend``, runtime load IDs), ``deployment.py``, ``registry.py``, ``serving.py``, ``recovery.py`` and ``residency.py``. Engines live in ``inference/sglang/``, ``inference/vllm/`` and ``inference/remote/``.
- **The execution base stays small.** ``execution/lifecycle.py`` uses only the standard library. ``execution/processes.py`` and ``execution/jobs.py`` add only ``reef.core.errors``. None of them imports ``execution.sandbox``.

The extension points that the packages expose are in `Python API <../reference/python-api.rst>`__.

Deployment modules
------------------

``reef/deploy/`` separates configuration input from process startup:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - Responsibility
   * - ``config_utils.py``
     - Reads YAML, expands environment and config references, and locates recipe source packages. It downloads no model and validates no process graph.
   * - ``service_config.py``
     - Converts the effective values of a deployment into ``ServiceSettings`` for app assembly. ``reef/api/settings.py`` declares the shared HTTP, storage and runtime fields.
   * - ``deployment_config.py``
     - Declares the ``DeploymentSettings`` defaults, loads the selected recipe and runtime declarations, translates the versioned public layout, and validates component values.
   * - ``cli.py``
     - Builds help and applies dotted command-line overrides before the shared type conversion. CLI values take precedence over YAML.
   * - ``inference.py``, ``training.py``
     - Assemble the inference processes and the training deployment that ``training.backend`` names. They read the serving mode and the training backend from ``reef/api/serving.py`` and ``reef/api/training.py``.
   * - ``execution.py``, ``process.py``, ``guard.py``
     - Validate process definitions and dependency order, select executors, own worker processes, and clean up a remote process group when its Ray owner disappears.
   * - ``diagnostics.py``
     - Lists the resolved settings with their sources for the startup log, and masks credentials.
   * - ``orchestrator.py``
     - Coordinates configuration resolution, launch, readiness, supervision and shutdown.

Shared parsing types come from ``reef.core.config``. Native backend argument encoding comes from ``reef.execution.executor.arguments``. The exports of ``reef.deploy`` are the entrypoints for app assembly and launching; internal module names are not a compatibility API.

Repository-level homes
----------------------

Code for a concrete integration lives together under its package, but its surrounding files stay at repository level:

- component and integration tests in ``tests/<owner>/``
- cookbook method tests in ``tests/recipes/<method>/``
- import and packaging contracts in ``tests/plugin_contracts/``
- service-facing contracts in ``tests/api/``
- the learn-nothing deployment stacks, and the smallest example around them, in ``recipes/basic/``
- the configuration of one runnable deployment under ``recipes/<name>/examples/``
- container and environment setup under ``docker/``
- Python dependencies, package data and plugin entry points in ``pyproject.toml``

The `test directory map <https://github.com/Human-Agent-Society/reef/blob/main/tests/AGENTS.md>`_ defines the owner directories, the shared support, and the rules for moving tests.

Do not copy third-party source into an integration subtree. Pin or declare the dependency in ``pyproject.toml`` and keep Reef-owned adapters local to the integration.

Adding a new subpackage under ``reef/`` or a new method under ``recipes/`` requires an RFC that states which layer owns the behavior.

Where the detailed rules live
-----------------------------

- Design pages: `Scenario components <../developer-guide/components.rst>`__, `Training pipeline <../developer-guide/write-a-recipe.rst>`__, `Training backends <../developer-guide/training-backends.rst>`__, `Serving runtimes <../developer-guide/serving-runtimes.rst>`__ and `Executors <../developer-guide/executors.rst>`__.
- Extension points: `Python API <../reference/python-api.rst>`__.
- The public wire contract: `HTTP API <../reference/http-api.rst>`__.
- How to add each kind of component: `Adding components <adding-components.rst>`__.
- How the cookbook methods sit beside ``reef/``: the `top-level README <../../README.md>`__.
