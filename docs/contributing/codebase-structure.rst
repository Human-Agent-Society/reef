Codebase structure
==================

This page says which package should own a change.


Choose a destination
--------------------

``reef/`` holds every shared mechanism, including the harness evolution engine
at ``reef/train/cordis_backend/``. The built-in Reefine recipe lives under
``reef/recipe/reefine/``. The built-in SFT recipe lives under
``reef/recipe/sft/``, and its Slime loss family is in
``reef/train/slime_backend/sft.py``. Paper-backed methods live in separate
packages under ``recipes/`` (``sao``, ``tttd``, ``openclawrl``, ``skillclaw``)
with that method's recipe, processor, training objective, and, for weight methods,
the ``slime/`` subpackage only the training plane imports. Nothing under
``reef/`` imports a method package.

Reef is organized around an application kernel and three capability domains,
not a strict stack of top-level packages. The arrows below show the primary
composition and use paths; they are not an exhaustive Python import graph:

.. code:: mermaid

   flowchart TD
       accTitle: Reef application kernel, capability domains, and adapters

       Methods("<b>Method plug-ins</b><br/><code>recipes/*</code>")
       Entry("<b>Entrypoints</b><br/>HTTP · CLI")
       Policy("<b>Policy</b><br/><code>reef/recipe</code>")
       Service("<b>Delivery &amp; composition</b><br/><code>reef/service</code>")
       Kernel(["<b>Application kernel</b><br/><code>reef/dispatcher</code> · <code>reef/scenario</code>"])

       subgraph Domains["Capability domains"]
           direction LR
           Serving("<b>Serving</b><br/><code>runtime</code> · <code>surface</code>")
           Evolution("<b>Evolution</b><br/><code>train</code> · <code>evaluation</code> · <code>harness</code>")
           State("<b>State</b><br/><code>artifact</code> · <code>records</code>")
       end

       subgraph Adapters["Concrete adapters"]
           direction LR
           ServingAdapters("runtime adapters<br/>surface implementations")
           EvolutionAdapters("training backends<br/>harness adapters")
           StateAdapters("Git/LFS repositories<br/>SQLite")
       end

       Observability(["<b>Cross-cutting</b><br/><code>reef/observability</code>"])
       Core(["<b>Shared kernel</b><br/><code>reef/core</code>"])

       Methods --> Policy --> Kernel
       Entry --> Service --> Kernel
       Kernel --> Serving & Evolution & State
       Serving --> ServingAdapters
       Evolution --> EvolutionAdapters
       State --> StateAdapters
       Kernel -. telemetry .-> Observability
       ServingAdapters & EvolutionAdapters & StateAdapters --> Core

Method packages provide policy through ``reef/recipe`` and may bind
``reef/train`` machinery directly. HTTP and CLI entrypoints compose Reef
through ``reef/service``; the transport-free dispatcher and scenario aggregate
coordinate serving, evolution, and state. ``reef/service`` also imports
``reef/artifact`` directly to stream artifact bytes.

Concrete integrations may depend on shared contracts; shared contracts never
import a concrete integration.

+-------------------------+----------------------------------------------------------+--------------------------------------------+
| Package                 | Owns                                                     | Does not own                               |
+=========================+==========================================================+============================================+
| ``reef/core/``          | shared value types, wire shapes, artifact                | storage, I/O, runtime behavior             |
|                         | identity, root errors                                    |                                            |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/service/``       | HTTP routes, auth, streaming, process                    | training methods or domain logic           |
|                         | lifecycle                                                | tied to aiohttp                            |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/scenario/``      | scenario binding, commit ordering,                       | training algorithms, repository            |
|                         | recovery, and lifecycle                                  | implementations                            |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/recipe/``        | the contract a method implements, dotted                 | external cookbook methods                  |
|                         | resolution, runtime binding, and the built-in Reefine    |                                            |
|                         | and generated-task recipes                               |                                            |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/train/``         | the trainer, the training job loop, the training         | HTTP endpoints, deployment                 |
|                         | pipeline, job data types, training components            | configuration parsing                      |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/record2record/`` | which records Reef uses and in what order: report        | sample and job data assembly, record       |
|                         | grouping, judging, eligibility checks, selection order   | storage, task generation                   |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/record2task/``   | turning a record set into tasks (``Record2Task``,        | which records train, job data shape,       |
|                         | ``ReportTasks``, ``JudgedTasks``), and the generator     | training loops                             |
|                         | service that writes, checks, and runs Harbor tasks       |                                            |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/evaluation/``    | candidate evaluation: running and scoring evaluation     | proposing candidates, training loops,      |
|                         | episodes, built-in selection policies, and the           | artifact publication                       |
|                         | ``evaluation`` plugin configuration                      |                                            |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/task2job/``      | shaping the tasks of the selected record set units into  | which records train, the training loop,    |
|                         | one job's data (``Task2Job``)                            | step signals                               |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/runtime/``       | backend-neutral contracts: training components,          | concrete inference and training            |
|                         | inference runtimes, saved-weight serving, executors      | integrations                               |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/inference/``     | concrete inference backends, native engine control,      | training algorithms and publication        |
|                         | request adaptation and loading saved weights             | decisions                                  |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/surface/``       | delivering a published artifact to the                   | proposing, evaluating, or                  |
|                         | process or client that uses it                           | selecting updates                          |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/artifact/``      | artifact bytes, repositories,                            | commit policy or delivery                  |
|                         | materialization, release heads                           | behavior                                   |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/harness/``       | harness descriptors, tree rendering,                     | recipe policy, the release chain           |
|                         | episodes, trajectories                                   |                                            |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/sandbox/``       | process execution, isolation, file transfer              | harness rendering, evaluation, recipe      |
|                         | and environment lifecycle through Harbor                 | policy                                     |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``recipes/``            | one method per package: recipe, processor, objective,    | shared machinery, or another method        |
|                         | and its runnable examples                                |                                            |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``tests/``              | repository-level tests grouped by responsibility         | tests hidden inside an integration subtree |
+-------------------------+----------------------------------------------------------+--------------------------------------------+
| ``docker/``             | container and GPU environment setup                      | Python dependency declarations             |
+-------------------------+----------------------------------------------------------+--------------------------------------------+


Package import direction
------------------------

Imports between the top-level Reef packages form a directed acyclic graph.
``tests/contributing/test_dependency_boundaries.py`` scans every Python file,
including relative imports and imports inside functions. Importing the root
``reef`` facade from an internal package also counts as a dependency; internal
code imports the owning module directly.

The following table records direct package dependencies (excluding each
package's own submodules and third-party libraries):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Package
     - Imports
   * - ``core``
     - None
   * - ``storage``, ``artifact``
     - ``core``
   * - ``observability``
     - ``storage``, ``core``
   * - ``surface``
     - ``artifact``, ``core``
   * - ``runtime``
     - ``surface``, ``artifact``, ``core``
   * - ``inference``
     - ``runtime``, ``artifact``, ``core``
   * - ``sandbox``
     - ``core``
   * - ``harness``
     - ``sandbox``, ``runtime``, ``core``
   * - ``evaluation``
     - ``harness``, ``sandbox``, ``runtime``, ``core``
   * - ``train``
     - ``evaluation``, ``harness``, ``sandbox``, ``runtime``, ``artifact``, ``storage``, ``observability``, ``core``
   * - ``recipe``
     - ``train``, ``evaluation``, ``harness``, ``sandbox``, ``inference``, ``runtime``, ``surface``, ``artifact``, ``storage``, ``observability``, ``core``
   * - ``scenario``
     - ``recipe``, ``train``, ``inference``, ``runtime``, ``surface``, ``artifact``, ``storage``, ``observability``, ``core``
   * - ``dispatcher``
     - ``scenario``, ``recipe``, ``train``, ``harness``, ``runtime``, ``artifact``, ``storage``, ``observability``, ``core``
   * - ``service``
     - ``dispatcher``, ``scenario``, ``recipe``, ``train``, ``inference``, ``harness``, ``runtime``, ``surface``, ``artifact``, ``storage``, ``observability``, ``core``
   * - ``cli``
     - ``service``, ``core``

Shared job data and candidate evaluation contracts live in ``core/job_data.py``
and ``core/evaluation.py``. ``core/training_request.py`` validates training-request
payloads and holds the queued instruction carried by a job's data. Request
requirements and their release-chain interpretation live in ``core/requirements.py``.
Storage owns commit record encoding; scenario owns commit ordering and recovery. Artifact admission lives
with surface contracts.
``recipe/cordis.py`` assembles the harness training backend.
``service/deploy/training.py`` selects the training deployment definition
that ``training.backend`` names and assembles the services it needs;
``service/deploy/inference.py`` adds the local inference component when the
deployment serves saved weights on one.
``runtime/`` is a namespace package with exactly seven Python modules and the
``executor/`` package; it has no ``__init__.py`` or import facade:

.. code:: text

   runtime/
     interfaces.py
     jobs.py
     durable_files.py
     deployment.py
     recovery.py
     residency.py
     serving.py
     executor/

``interfaces.py`` defines the ``InferenceRuntime`` and ``InferenceBackend``
contracts, ``TrainingMethod`` and its learning-rate schedule, and runtime load
IDs. ``jobs.py`` defines bounded training jobs and the ``TrainingComponent``
interface. ``deployment.py`` owns runtime factories, ``RuntimeRegistry`` and
the inference service contracts. ``durable_files.py`` writes JSON files and
directories durably, for the training components' job records. ``recovery.py``
owns ``InferenceControl`` and ``EngineHealthMonitor``. ``residency.py`` holds the
``AdapterResidencyManager``. ``serving.py`` loads saved weights into serving
engines that the control plane controls directly. ``executor/`` owns worker
launch and control RPC.

``jobs.py``, ``interfaces.py`` and ``durable_files.py`` import no other runtime module; ``serving.py``
may import every other module except ``executor/``.
``tests/contributing/test_dependency_boundaries.py`` checks this order.
Runtime contracts have no direct imports of concrete inference or training
code; the deployment registry resolves the selected factory explicitly.

In ``train/``, ``trainer.py`` holds the ``Trainer``; ``jobs.py`` holds the
``JobLoop``, the recipe's ``JobPlan`` and the committed ``JobLoopState``;
``local_jobs.py`` holds ``LocalJobComponent``, the base of the built-in
training components; ``deployment.py`` holds the ``TrainingDeployment``
definitions; and ``backend.py`` holds ``CandidateBackend``, the candidate
lifecycle of harness evolution. ``train/slime_backend/component.py`` and
``train/tinker_backend/component.py`` supply the Slime and Tinker training
components, and each ``launch.py`` supplies its deployment definition and
runtime factory. No training backend builds serving:
``service/deploy/serving.py`` selects and configures it from the
``inference`` section and the component's release format.

HTTP provider handling lives in ``inference/http.py``.
``inference/chat.py`` holds the engine-neutral OpenAI and Anthropic chat facade
over a token-native generate route: it renders the prompt, builds the training
record from a ``CapturedGeneration`` and streams protocol frames. Each engine
supplies a ``NativeGenerateClient`` that shapes its request, parses its response
and builds its tool-call parser. ``inference/engine_control.py`` holds the
control classes that SGLang and vLLM share: the inference backend, the
inference service, the control actor and its default serving executor.
``inference/sglang/`` owns SGLang engine launch, capture and control
independently of training; ``inference/vllm/`` owns vLLM capture,
the KV connector that stamps tokens with their weight version, and engine launch and
control over vLLM's HTTP control routes. ``inference/remote/`` is the inference
component: ``python -m reef.inference.remote`` runs the engines and serves
their control operations and requests over HTTP. ``RemoteInferenceBackend``
is the control plane's client for it, and ``inference/remote/runtime.py``
supplies the ``remote_serving`` runtime kind and builds the engines' input
(``serving_engine_config``). ``inference/null.py`` is the
runtime of ``inference.backend: none``: it runs no engine and refuses every
request. ``inference/tinker.py`` serves Tinker's sampler. ``tinker/`` holds
the Tinker SDK pieces that the Tinker training component and the Tinker
serving runtime share: ``tinker/sampling.py`` renders and samples text chat
with Reef's training capture, for serving and for the component's own
rollouts.

The extension points those packages expose are in `Python API
<../reference/python-api.rst>`__.


- Is it a value or error needed by unrelated layers without behavior attached?
  Put it in ``reef/core/``.
- Does it own scenario state, commit ordering, recovery, or rollback? Put it in
  ``reef/scenario/``. ``reef/storage/commits.py`` defines the persisted ``CommitRecord`` and
  ``RecordProgress`` values without storage behavior. Checkpoint recovery uses
  the same ``CommitRecord`` type; initial registration has no commit.
  ``reef/storage/scenario.py`` defines the ``ScenarioStore`` and ``ScenarioStorage``
  abstract bases. ``storage/commits.py`` also encodes artifact metadata and decodes existing
  checkpoint metadata into registration information and a ``CommitRecord``.
  ``ScenarioFactory`` receives the storage service from deployment assembly
  and handles registration, release selection, and recovery. It opens a
  session, restores committed artifacts, builds the trainer, replays records,
  and returns a complete ``Scenario``; failure closes its owned resources. ``committer.py`` owns writes and retry ordering;
  ``releases.py`` owns release and artifact queries under the same publication
  lock. ``history.py`` pages retained records and commits. ``registry.py`` owns
  loaded instances, model configuration caching, updates, and scenario
  archival coordination. Recipes, scenarios, and the factory use the concrete
  ``ModelConfig`` in ``reef/inference/model_config.py``. The factory receives
  one configuration per creation or recovery; it owns no configuration cache.
  The registry reads and writes each scenario's settings (model configuration
  and training mode) through the injected ``ScenarioStorage``, like records and
  commits; it imports no storage implementation.
  ``Dispatcher`` calls the storage service directly for retention and closes
  it after closing the loaded scenarios. It never imports or chooses a
  concrete record backend.
- Does it define shared record operations or retention limits? Put the contract
  or value in ``reef/storage/records.py``. It defines the ``RecordStore`` abstract base
  without importing scenario coordination, training, or concrete adapters.
  ``ScenarioStore`` combines a ``RecordStore`` with committed scenario state
  and opens the scenario's task stores (``reef/storage/tasks.py``);
  ``ScenarioStorage`` owns scenario settings, archival, and retention.
- Does it turn a designer's replies into Harbor tasks, check them with Harbor
  or run them with an agent? Put it in ``reef/record2task/generator/``, the
  generator service ``reef serve`` starts beside the HTTP service from a
  ``generator`` section. ``reef/service/deploy/generator.py`` assembles that
  child, and ``reef/service/deploy/generator_process.py`` runs it. The built-in generated-task recipe, which runs generated tasks
  through that service and trains on them, is in
  ``reef/recipe/generated_tasks/``, with the ``TaskGenerator`` contract
  (``generator.py``). The request, ``TaskGenerationRequest``, is in
  ``reef/record2task/generator/inputs.py``, because the generator service
  reads it too. ``reef/core/tasks/`` keeps the Harbor task values and the
  split manifests. What a method asks the designer for, and what it does
  with the results, stays in the method's task generator under ``recipes/``.
- Does it implement storage? Put it in ``reef/storage/``. ``sql_records.py``
  shares SQL record and retention operations, and ``ScopedTable``, one table
  with the namespace of one store. ``sql_tasks.py``,
  ``sql_trajectories.py`` and ``sql_commits.py`` share the task, trajectory
  and commit tables and their operations in the same database.
  ``tasks.py`` holds the local ``TaskFileStore`` next to the task contracts.
  ``sqlite.py`` supplies SQLite schema, connections, transactions, and file
  maintenance. ``postgres.py`` supplies PostgreSQL tables, pooled transactions,
  and retention. ``commit_history.py`` owns the ``CommitHistory`` base and
  ``CommitHistoryScenarioStore``, which accepts any ``RecordStore``,
  ``TaskStorage`` and ``CommitHistory``. ``migrations.py`` holds one-time
  upgrades of older schemas and files. ``sqlite.py`` and ``postgres.py`` assemble their respective
  scenario storage services against the interfaces in ``scenario.py``.
  Storage implementations depend on domain contracts, never the reverse.
- Does it persist or materialize versioned bytes? Put it in
  ``reef/artifact/``. If it decides how consumers activate those bytes, put
  that behavior in ``reef/surface/`` instead.
- Does it define a backend-neutral model-service contract? Put it in
  ``reef/runtime/``, beside the training component and serving contracts.
  Put concrete inference engines and their request/control adapters in
  ``reef/inference/<integration>/``; put implementation tied to a concrete
  training stack in its own ``reef/train/<integration>/`` subtree.
  ``reef/train/cordis_backend/`` is the general harness evolution engine;
  the shared composition engine lives in ``reef/harness/compose/``. It derives
  from cordis 4.0.0-rc.8; see ``reef/harness/compose/UPSTREAM.md``. ``reef/train/slime_backend/`` is the
  weights counterpart.
- Does it decide which records train, or in which groups? Put it in
  ``reef/record2record/``. Does it turn one record set into tasks? Put the
  shared contract in ``reef/record2task/``. Does it shape tasks into job data?
  Put it in ``reef/task2job/``. Does it turn job data into a step signal? Put
  it in ``reef/train/algos/``. ``reef/train/pipeline.py`` drives the three
  stages. A recipe selects and binds that machinery; it should
  not reimplement it.
- Is it HTTP-specific? Keep the aiohttp adapter in ``reef/service/routes/``
  and put transport-independent behavior in a service or domain object.
- Does it orchestrate a benchmark, task, grader, or external environment?
  Keep it under ``recipes/<name>/examples/`` or in the external harness.

Repository-level homes
----------------------

Code for a concrete training integration lives together under
``reef/train/<integration>/``, but its surrounding files remain at repository
level:

- component and integration tests in ``tests/<owner>/``;
- cookbook method tests in ``tests/recipes/<method>/``;
- import and packaging contracts in ``tests/plugin_contracts/``;
- service-facing contracts in ``tests/reef_service/``;
- the learn-nothing deployment stacks, and the smallest example around them, in
  ``recipes/basic/``;
- configuration for one runnable deployment under ``recipes/<name>/examples/``;
- container and environment setup under ``docker/``; and
- Python dependencies, package data, and plugin entry points in
  ``pyproject.toml``.

The `test directory map <https://github.com/Human-Agent-Society/reef/blob/main/tests/AGENTS.md>`_
defines the owner directories, shared support, and rules for moving tests.

Do not copy third-party source into an integration subtree. Pin or declare the
dependency in ``pyproject.toml`` and keep Reef-owned adapters local to the
integration.

Where the detailed rules live
-----------------------------

Each package's ``__init__`` docstring states the boundaries it holds and
how to extend it; there are no READMEs under ``reef/``. Design pages for the
two packages that need more than a docstring are `Surfaces
<../developer-guide/surface.rst>`__ and `Training pipeline
<../developer-guide/processors.rst>`__; the extension points every package
exposes are in `Python API <../reference/python-api.rst>`__, and the public
harness wire contract is `HTTP API <../reference/http-api.rst>`__. The
`top-level README <../../README.md>`__ shows how the cookbook methods sit
beside ``reef/``.

``reef/train/cordis_backend/`` is the general harness evolution engine; the
shared composition engine derives from cordis 4.0.0-rc.8 with the conformance
map in ``reef/harness/compose/UPSTREAM.md``. ``reef/train/slime_backend/`` is the weights
counterpart.

Adding a new subpackage under ``reef/`` or a new method under ``recipes/``
requires an RFC that states which layer owns the behavior.


Deployment modules
------------------

``reef/service/deploy/`` separates configuration input from process startup:

* ``config_utils.py`` reads YAML, expands environment/config references and locates
  recipe source packages. It does not download models or validate process graphs.
* ``service_config.py`` declares shared HTTP, storage and runtime fields and
  converts effective values into ``ServiceConfig`` for app assembly. With a
  ``task-import`` section, assembly wraps the scenario storage in
  ``reef/service/task_import.py``, so that each scenario that opens imports the
  deployment's Harbor tasks.
* ``deployment_config.py`` declares ``DeploymentConfig`` defaults, loads selected
  recipe/runtime declarations, translates the versioned public layout and
  validates component values.
* ``cli.py`` builds help and applies dotted command-line overrides before shared
  type conversion. CLI values take precedence over YAML.
* ``inference.py`` assembles provider or local inference processes and resolves
  model snapshots; ``training.py`` selects training deployment definitions and
  assembles the training dependencies and HTTP process.
* ``execution.py`` validates process definitions and dependency order and selects
  executors. ``process.py`` owns worker processes; ``guard.py`` cleans up a remote
  process group when its Ray owner disappears.
* ``diagnostics.py`` lists the resolved settings with their sources (file,
  command line, environment, automatic, default) for the startup log, masking
  credentials.
* ``orchestrator.py`` coordinates configuration resolution, launch, readiness,
  supervision and shutdown, including the internal HTTP child entrypoint.

Shared parsing types come directly from ``reef.core.config``. Native backend
argument encoding comes from ``reef.runtime.executor.arguments``. The package
exports in ``reef.service.deploy`` remain the entrypoints for app assembly and
launching; internal module names are not a compatibility API.
