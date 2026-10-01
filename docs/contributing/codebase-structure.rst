Codebase structure
==================

This page says which package should own a change.


Choose a destination
--------------------

``reef/`` holds every shared mechanism, including the harness evolution engine
at ``reef/train/cordis_backend/``. The built-in Reefine recipe lives under
``reef/recipe/reefine/``. Paper-backed methods live in separate
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
           Evolution("<b>Evolution</b><br/><code>train</code> · <code>harness</code>")
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

+----------------------+----------------------------------------------------------+--------------------------------------------+
| Package              | Owns                                                     | Does not own                               |
+======================+==========================================================+============================================+
| ``reef/core/``       | shared value types, wire shapes, artifact                | storage, I/O, runtime behavior             |
|                      | identity, root errors                                    |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/service/``    | HTTP routes, auth, streaming, process                    | training methods or domain logic           |
|                      | lifecycle                                                | tied to aiohttp                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/scenario/``   | scenario binding, commit ordering,                       | training algorithms, repository            |
|                      | recovery, and lifecycle                                  | implementations                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/recipe/``     | the contract a method implements, dotted                 | external cookbook methods                  |
|                      | resolution, runtime binding, and built-in Reefine        |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/train/``      | the trainer loop, processor engines, batch               | HTTP endpoints, deployment                 |
|                      | types, backend integrations                              | configuration parsing                      |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/runtime/``    | backend-neutral runtime contracts, scheduling,           | concrete inference and training            |
|                      | weight synchronization and publication coordination      | integrations                               |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/inference/``  | concrete inference backends, native engine control,      | training algorithms and publication        |
|                      | request adaptation and weight reception                  | decisions                                  |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/surface/``   | delivering a published artifact to the                   | proposing, evaluating, or                   |
|                      | process or client that uses it                           | selecting updates                          |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/artifact/``  | artifact bytes, repositories,                            | commit policy or delivery                   |
|                      | materialization, release heads                           | behavior                                   |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``reef/harness/``    | harness descriptors, tree rendering,                     | recipe policy, the release chain           |
|                      | episodes, trajectories                                   |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``recipes/``         | one method per package: recipe, processor, objective,    | shared machinery, or another method        |
|                      | and its runnable examples                                |                                            |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``tests/``           | repository-level tests grouped by responsibility         | tests hidden inside an integration subtree |
+----------------------+----------------------------------------------------------+--------------------------------------------+
| ``docker/``          | container and GPU environment setup                      | Python dependency declarations             |
+----------------------+----------------------------------------------------------+--------------------------------------------+


Package import direction
------------------------

Imports between the top-level Reef packages form a directed acyclic graph.
``tests/reef_service/test_dependency_boundaries.py`` scans every Python file,
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
   * - ``harness``
     - ``runtime``, ``core``
   * - ``train``
     - ``harness``, ``inference``, ``runtime``, ``surface``, ``artifact``, ``storage``, ``observability``, ``core``
   * - ``recipe``
     - ``train``, ``harness``, ``runtime``, ``surface``, ``storage``, ``observability``, ``core``
   * - ``scenario``
     - ``recipe``, ``train``, ``runtime``, ``surface``, ``artifact``, ``storage``, ``observability``, ``core``
   * - ``dispatcher``
     - ``scenario``, ``recipe``, ``train``, ``harness``, ``runtime``, ``artifact``, ``storage``, ``observability``, ``core``
   * - ``service``
     - ``dispatcher``, ``scenario``, ``recipe``, ``train``, ``inference``, ``harness``, ``runtime``, ``surface``, ``artifact``, ``storage``, ``observability``, ``core``
   * - ``cli``
     - ``service``, ``core``

Shared batches and candidate evaluation contracts live in ``core/batches.py``
and ``core/evaluation.py``. Request requirements and their release-chain
interpretation live in ``core/requirements.py``. Storage owns commit record
encoding; scenario owns commit ordering and recovery. Artifact admission lives
with surface contracts, while checkpoint cadence is recipe policy.
``recipe/cordis.py`` assembles the harness training backend, and
``service/training_driver.py`` independently selects training and inference
definitions; ``runtime/deployment.py`` owns their startup, attachment and shutdown.
``runtime/`` is a namespace package with exactly five Python modules and the
``executor/`` package; it has no ``__init__.py`` or import facade:

.. code:: text

   runtime/
     interfaces.py
     scheduler.py
     deployment.py
     publication.py
     recovery.py
     executor/

``interfaces.py`` defines the independent ``TrainingRuntime`` and
``InferenceRuntime`` scheduling interfaces, the native ``TrainingBackend`` and
``InferenceBackend`` contracts, and shared values and storage interfaces.
``scheduler.py`` owns recipe-facing candidate/commit coordination, remote job
serialization, staleness admission and train/checkpoint ordering.
``publication.py`` owns weight publication, version allocation, LoRA residency
and transfer locking. ``recovery.py`` owns durable markers, scenario history,
engine monitoring and restart recovery. ``deployment.py`` owns component
lifecycle, runtime configuration and factory resolution. ``executor/`` owns
worker launch, control RPC and coordinator connections.

Dependencies run from deployment through scheduling, recovery and publication
to interfaces. Executor implementations depend only on interfaces and their own
package. Runtime coordination has no direct imports of concrete inference or
training code; the deployment registry resolves the selected factory explicitly.
Generic connection implementations live in ``inference/runtime.py`` and
``train/runtime.py``; HTTP provider handling lives in ``inference/http.py``.
``inference/sglang/runtime.py`` supplies ``SGLangInferenceRuntime``,
``inference/vllm/runtime.py`` supplies ``VLLMInferenceRuntime``, and
``train/slime_backend/runtime.py`` supplies ``SlimeTrainingRuntime``. These
implement the scheduling interfaces; their native Backend implementations retain
model operations, checkpoint and tensor I/O.
``inference/chat.py`` holds the engine-neutral OpenAI and Anthropic chat facade
over a token-native generate route: it renders the prompt, builds the training
record from a ``CapturedGeneration`` and streams protocol frames. Each engine
supplies a ``NativeGenerateClient`` that shapes its request, parses its response
and builds its tool-call parser. ``inference/sglang/`` owns SGLang engine launch,
capture and control independently of training; ``inference/vllm/`` owns vLLM capture,
the KV connector that stamps tokens with their weight version, and engine launch and
control over vLLM's HTTP control routes. ``train/slime_backend/inference.py`` only
translates Slime options into plain launch data; the selected inference factory
constructs its own configuration.
``train/slime_backend/driver.py`` supplies Slime component definitions;
``service/slime_driver.py`` preserves the legacy explicit-process entrypoint.

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
  The registry calls ``reef/storage/model_config.py`` functions directly for
  the fixed local JSON files. This is its only storage implementation import:
  records and commits still use an injected ``ScenarioStorage``.
  ``Dispatcher`` calls the storage service directly for retention and closes
  it after closing the loaded scenarios. It never imports or chooses a
  concrete record backend.
- Does it define shared record operations or retention limits? Put the contract
  or value in ``reef/storage/records.py``. It defines the ``RecordStore`` abstract base
  without importing scenario coordination, training, or concrete adapters.
  ``ScenarioStore`` combines a ``RecordStore`` with committed scenario state;
  ``ScenarioStorage`` owns archival and retention.
- Does it turn a designer's replies into Harbor tasks, check them with Harbor
  or play them with an agent? Put it in ``reef/record2dataset/``, the
  generator service ``reef serve`` starts beside the HTTP service from a
  ``generator`` section; ``reef/service/deploy/generator.py`` assembles that
  child. What a method asks the designer for, and what it does with the
  episodes, stays in the method's processor under ``recipes/``.
- Does it implement storage? Put it in ``reef/storage/``. ``sql_records.py``
  shares SQL record and retention operations; ``sqlite.py`` supplies SQLite
  schema, connections, transactions, and file maintenance. ``postgres.py`` supplies
  PostgreSQL tables, pooled transactions, and retention. ``commit_log.py`` owns
  the JSONL ``CommitLog`` and ``CommitLogScenarioStore``, which accepts any
  ``RecordStore``. ``sqlite.py`` and ``postgres.py`` assemble their respective
  scenario storage services against the interfaces in ``scenario.py``.
  Storage implementations depend on domain contracts, never the reverse.
- Does it persist or materialize versioned bytes? Put it in
  ``reef/artifact/``. If it decides how consumers activate those bytes, put
  that behavior in ``reef/surface/`` instead.
- Does it define a backend-neutral model-service contract? Put it in
  ``reef/runtime/``, alongside Reef's scheduling and publication coordination.
  Put concrete inference engines and their request/control adapters in
  ``reef/inference/<integration>/``; put implementation tied to a concrete
  training stack in its own ``reef/train/<integration>/`` subtree.
  ``reef/train/cordis_backend/`` is the general harness evolution engine;
  the shared composition engine lives in ``reef/harness/compose/``. It derives
  from cordis 4.0.0-rc.8; see ``reef/harness/compose/UPSTREAM.md``. ``reef/train/slime_backend/`` is the
  weights counterpart.
- Does it turn records and feedback into a batch or step signal? Put it in
  ``reef/train/processors/`` or ``reef/train/algos/``. A recipe selects and
  binds that machinery; it should not reimplement it.
- Is it HTTP-specific? Keep the aiohttp adapter in ``reef/service/routes/``
  and put transport-independent behavior in a service or domain object.
- Does it orchestrate a benchmark, task, grader, or external environment?
  Keep it under ``recipes/<name>/examples/`` or in the external harness.

Repository-level homes
----------------------

Code for a concrete training integration lives together under
``reef/train/<integration>/``, but its surrounding files remain at repository
level:

- internal integration tests in ``tests/<integration>/``;
- import and packaging contracts in ``tests/plugin_contracts/``;
- service-facing contracts in ``tests/reef_service/``;
- the learn-nothing deployment stacks, and the smallest example around them, in
  ``recipes/basic/``;
- configuration for one runnable deployment under ``recipes/<name>/examples/``;
- container and environment setup under ``docker/``; and
- Python dependencies, package data, and plugin entry points in
  ``pyproject.toml``.

Do not copy third-party source into an integration subtree. Pin or declare the
dependency in ``pyproject.toml`` and keep Reef-owned adapters local to the
integration.

Where the detailed rules live
-----------------------------

Each package's ``__init__`` docstring states the boundaries it holds and
how to extend it; there are no READMEs under ``reef/``. Design pages for the
two packages that need more than a docstring are `Surfaces
<../developer-guide/surface.rst>`__ and `Processors
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
  converts effective values into ``ServiceConfig`` for app assembly.
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
