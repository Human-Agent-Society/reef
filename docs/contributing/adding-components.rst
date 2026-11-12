Adding components
=================

First use the `Codebase structure <codebase-structure.rst>`__ to choose the
owner, then use the matching playbook below.

Core and external extensions
----------------------------

Recipes and learning methods outside ``reef/recipe/`` are external packages
selected by dotted reference; Reef does not import those packages at boot. Changes to shared runtime or
training machinery start with an `RFC issue
<https://github.com/Human-Agent-Society/reef/issues/new?template=rfc.yml>`__.
A new top-level package, persisted format, wire contract, or incompatible
public API also requires an RFC.

Add a recipe
------------

Use this playbook when a new method binds record processing, step preparation,
training jobs, and artifact delivery. Read `Write a recipe
<../developer-guide/write-a-recipe.rst>`__ and the `Python API <../reference/python-api.rst>`__ first.

Implementation
~~~~~~~~~~~~~~

- Keep a recipe with its method package and select its frozen recipe dataclass
  as ``package.module:ClassName``. There is no ``register_kind`` step and no
  import from ``reef/__init__.py``. The contract it implements is
  ``reef/recipe/``.
- Declare recipe-owned settings with ``config_field``. Do not parse those
  settings again in ``reef/service/``. Do not add a loading method to the
  recipe class: ``reef.recipe.loading.load_recipe`` builds the recipe from its
  declarations.
- Put the method's record-to-sample and sample-to-job stages in
  ``<method-package>/processor.py``, subclassing ``ReportSamples`` or
  ``JudgedSamples`` (``reef/record2sample/``) and ``Sample2Job``
  (``reef/sample2job/``), and its record stage in the recipe module.
  Extend those contracts only when they do not already express it.
- Put the method's training objective with the backend that it targets: a
  ``SlimeObjective`` (``reef/train/slime_backend/objectives.py``) in
  ``<method-package>/slime/__init__.py``, or a ``TinkerObjective``
  (``reef/train/tinker_backend/objectives.py``) in its Tinker module. The
  package's ``__init__.py`` registers it by reference with that backend's
  ``register_objective``, so its public half imports none of the method's
  backend code. Name it in the recipe's ``default_parameters()``.
- A method with its own tensor objective adds a loss family in
  ``<method-package>/slime/`` (spec in ``__init__.py`` under
  ``@register_loss_family``, hooks in ``objective.py`` under ``@worker_hook``)
  and names it in the objective's ``loss_family``;
  the `Loss families
  <../developer-guide/loss-families.rst>`__ lists what a family declares.
- Override ``build_components`` only when the method's snapshots bind
  components beyond those of the recipe it extends. It reads the recipe's
  configuration alone: a recipe has no runtime and no training component.

Surrounding changes
~~~~~~~~~~~~~~~~~~~

- Add dotted-resolution and configuration tests in
  ``tests/reef_recipe/test_reef_recipes.py`` and
  ``tests/reef_recipe/test_recipe_config_fields.py``.
- Add method behavior tests under ``tests/recipes/<method>/``. Add an integration
  contract when the method calls into a concrete backend.
- Put every runnable configuration beside its method under
  ``recipes/<name>/examples/``; only a stack that binds no method belongs in
  ``recipes/basic/``.
- Add ``docs/user-guide/recipes/<name>.rst`` and update the public README recipe
  table when the cookbook carries the method. Update the relevant pipeline stage, objective, or component
  reference if its public contract changes.

Add a harness method
--------------------

Use this playbook for a method that evolves a harness tree. Read `Write a
harness method <../developer-guide/write-a-harness-method.rst>`__ first.

Implementation
~~~~~~~~~~~~~~

- Put the ``Method`` in ``recipes/<name>/method.py`` and export it from the
  package's ``__init__.py``. A deployment runs
  ``reef.recipe.harness:HarnessEvolutionRecipe`` and names the class in
  ``evolution.method``, so the method needs no recipe of its own.
- Parse the method's settings into a ``PluginSettings`` dataclass in
  ``from_config``.
- Keep what the method learns across steps in files under
  ``workspace.state``. Read earlier candidates, scores and decisions from the
  history instead of keeping copies of them.
- Put a prompt longer than a sentence or two in a ``.md`` file beside the
  method.
- Subclass ``HarnessEvolutionRecipe`` only to change configuration, such as
  the defaults of a deployment or the files a client pulls. A recipe builds
  no backend, evaluator or method; a method takes what it needs from
  ``job.services``.

Surrounding changes
~~~~~~~~~~~~~~~~~~~

- Add method tests under ``tests/recipes/<name>/``. Run a job as the trainer
  does with ``backend_step`` in ``tests/support/harness_service.py``, or call ``run``
  on a workspace from ``Workspace.create`` with a fake model binding.
- Put runnable deployments under ``recipes/<name>/examples/``. The method's
  ``README.md`` says what it does, how it uses the workspace, and where it
  departs from its paper.
- Add the method to the list in `Evolve your harness
  <../user-guide/evolve-your-harness.rst#choose-or-write-a-method>`__.

Add a training integration
--------------------------

Use this playbook for a concrete external training stack. It must implement the
contracts described by the `Python API <../reference/python-api.rst>`__; do not let
framework types leak into backend-neutral modules.

Implementation
~~~~~~~~~~~~~~

- Start an accepted integration in ``reef/train/<integration>/``. Keep its
  implementation, framework adapters, and plugins inside that subtree.
- Implement ``TrainingComponent`` from ``reef/train/component.py``. A component
  runs whole training jobs with the framework's own training loop and saves
  checkpoints; Reef submits jobs and reads their status. Derive from
  ``reef.train.local_jobs.LocalJobComponent`` to get durable job directories,
  one job at a time, and resumption after a restart.
- Own the job's parameters: parse them into your own typed value, refuse an
  unknown key with ``ValueError``, and declare them in
  ``TrainingComponent.parameters_schema(kind)``. Reef reads none of them. Own your
  objectives, their table and your step schedule too: the Slime and Tinker
  backends each have their own (``objectives.py``, ``loss_families.py``,
  ``schedule.py``, ``signals.py``) and share none. Write each
  ``SavedCheckpoint.artifact_path`` in a form that serving loads: a Hugging Face
  checkpoint for full weights, or a PEFT adapter in an ``adapter/`` directory.
- Implement ``import_start`` to load a published snapshot as the start of the
  next job. Raise ``ValueError`` when the framework cannot do this; Reef then
  refuses a ``train-from`` for the scenario with HTTP 400.
- Change these shared interfaces only when the
  existing backend-neutral contract is insufficient for more than one
  integration. Contract changes need focused compatibility tests.
- Implement ``TrainingDeployment`` in the integration to describe process
  preparation, and a ``TrainingComponentFactory`` that returns the
  ``TrainingComponent``. Declare the component's ``artifact_format``; the
  ``inference`` section builds the serving from it, so the integration builds
  no serving runtime. A backend with its own serving returns its runtime
  config from ``trainer_serving_config``. In-process integrations can extend
  ``InProcessTrainingDeployment``. Expose the class through its
  ``package.module:ClassName`` reference or the ``reef.training_backends``
  entry-point group; see
  `Training backend deployment
  <../developer-guide/write-a-recipe.rst#training-backend-deployment>`__.
- Implement ``facts(config, model_path, environ)`` on the factory. It returns
  the backend's ``TrainingBackendFacts`` from the same config, without
  building or starting the component, and it imports no training framework or
  service SDK. Define the facts type in the integration's own package, from
  its own settings, and let the component answer the same members from a value
  of it, so the two cannot differ. Share no facts code with another backend.
  With the facts, ``reef serve --component inference`` builds the serving
  runtime from ``facts.artifact_format`` and never builds the component, so
  its environment does not need the framework. A factory that inherits
  ``facts``, which returns ``None``, makes the inference service build the
  component.
  Add a test that compares the facts with the component for two configs, and
  one that builds the facts without the framework. See `Facts without a
  component <../developer-guide/executors.rst#facts-without-a-component>`__.
- Keep the signal that an objective returns inside the integration
  (``signals.py``), and translate it into framework payloads there.
- Declare Python dependencies and source pins in ``pyproject.toml``. Add
  plugin entry points and package-data rules there when the framework loads
  Reef assets or hooks dynamically.
- Put a runnable deployment under ``recipes/<name>/examples/`` (or
  ``recipes/basic/`` when it binds no method), and image or
  environment setup under ``docker/``.

Tests and documentation
~~~~~~~~~~~~~~~~~~~~~~~

- Put framework-internal tests in ``tests/<integration>/``.
- Put import-scope, packaging, and plugin-loading tests in
  ``tests/plugin_contracts/``.
- Put service-facing lifecycle and wire contracts in ``tests/reef_service/``.
- Prove that importing backend-neutral Reef modules does not import the
  framework. Extend ``tests/contributing/test_dependency_boundaries.py`` when
  a new package dependency rule is added.
- Update the ``reef/train`` package docstring, the Integration API,
  deployment guidance, and the configuration reference.

Add a runtime kind
------------------

Use this playbook for a new inference provider or a runtime implementation that
satisfies Reef's backend-neutral lifecycle. Read the runtime contract in
``reef/inference/interfaces.py`` before adding configuration.

- For an external runtime, subclass ``RuntimeFactory`` and use the reference
  ``package.module:ClassName`` of the class as the runtime ``type``. Reef
  makes the class with no arguments. A function or an instance is not
  accepted.
- Every setting that takes a name or a reference resolves it with
  ``resolve_plugin_class`` in ``reef/core/plugins.py``, in one order: a
  built-in name, then an installed entry point when the setting has an
  entry-point group, then ``package.module:ClassName``. Do not write another
  lookup; give the resolver your table of built-in names.
- For a concrete inference backend, add its native implementation under
  ``reef/inference/<integration>/``. Keep native engine launch, request
  adaptation, weight loading, and framework imports inside that package.
  The modules at the root of ``reef/inference/`` hold the inference contracts;
  they must not import a concrete integration or ``reef.train``. Implement
  ``InferenceBackend`` from ``reef/inference/interfaces.py``, including
  ``load_weight_files``, so that the serving runtimes in
  ``reef/inference/serving.py`` can load published snapshots. Implement
  ``InferenceService`` with its ``from_config`` class method, and expose the
  class through its ``package.module:ClassName`` reference or the
  ``reef.inference_backends`` entry-point group, so the inference component
  can start the engines. A bundled engine adds its service class and its
  request handler class to ``BUILTIN_ENGINES`` in
  ``reef/inference/deployment.py``. Shared monitoring and engine recovery belong in
  ``reef/inference/recovery.py``; adapter residency belongs in
  ``reef/inference/residency.py``. Shared contracts and version values belong in
  ``reef/inference/interfaces.py``.
- For a runtime that Reef bundles, subclass ``RuntimeFactory``, set its
  ``kind``, implement ``__call__``, and add the kind with the reference of
  the class to ``BUILTIN_FACTORIES``. ``RuntimeFactory`` and
  factory resolution live in ``reef.inference.registry``. Importing
  ``reef.inference`` imports no integration and re-exports no class. A
  training component is not a runtime: its
  ``TrainingDeployment.component_type`` names a ``TrainingComponentFactory``
  from ``reef.train.component``.
- A ``RuntimeFactory`` can expose ``config_type()`` returning a configuration
  dataclass whose fields use ``reef.core.config.config_option``. The registry
  parses that selected schema with the shared CLI/YAML rules and runs the
  dataclass's validation before calling the factory. Keep schema imports
  lightweight. A factory without a schema receives the section as a mapping.
  Opaque backend payloads must be declared as object fields and validated by
  the adapter; do not enumerate backend-native flags in the service layer.
- Keep provider authentication and provider-native request handling in the
  adapter. Do not import ``reef.train`` from ``reef/inference/``.
- Test built-in name and reference resolution in
  ``tests/reef_service/test_runtime_registry.py``. Add request, recovery, and
  malformed-result contracts for every capability the runtime implements.
- Add or update the stack that exercises the runtime under ``recipes/`` and
  document every public setting. Never commit credentials or real provider tokens.

Add a worker executor
---------------------

Use an ``Executor`` subclass when worker launch or control transport changes
while the training lifecycle stays the same. Implement the common interface
under ``reef/execution/executor/`` for a general-purpose backend, or inside the
owning training integration for a backend-specific launcher. See
`Worker executors <../developer-guide/executors.rst>`__ for configuration,
ownership, rank ordering and the Slime extension contract.

Add a component or artifact type
--------------------------------

Artifacts own immutable bytes and snapshot heads. A recipe's components own
admission, import layouts, kinds and client file trees. Runtimes own loading.
Read `Scenario components <../developer-guide/components.rst>`__ and the
release-chain section of `Architecture <../getting-started/core-loop.rst>`__.

- Put storage, materialization, or version identity behavior in
  ``reef/artifact/``. A new storage backend implements ``RepositoryBackend``
  and is exposed through a repository backend factory. To let a deployment
  select it in ``storage.artifact_backend``, the factory also inherits
  ``ConfigurableRepositoryBackendFactory``.
- Put weight loading in ``InferenceRuntime.activate_checkpoint`` under
  ``reef/inference/``, and client-pull delivery in a
  ``FileTree`` (``reef/artifact/components.py``). Reef does not change a model
  request on its way to the engine: a client pulls the artifacts that shape
  its requests.
- Declare the components in the owning recipe's ``build_components``; do not
  make a component choose a method or evaluate a candidate. A fact that only
  a runtime or a training component knows belongs in
  ``reef.scenario.live_objects``, whose functions take the recipe and the
  process's ``LiveObjects``.
- Test component behavior in ``tests/artifact/test_scenario_components.py`` or
  ``tests/artifact/test_snapshot_components.py``. Test storage backends against
  the contracts in ``tests/artifact/test_reef_artifacts.py``.
- Add scenario commit, activation, train-from, and recovery tests whenever
  persistence or activation ordering changes. Persisted-format changes require an RFC and a
  compatibility contract.

Add a service route
-------------------

Routes adapt HTTP to transport-independent behavior. The ``reef/service``
package docstring and `HTTP API <../reference/http-api.rst>`__ define the
current request contract.

- Add or extend a module in ``reef/service/routes/`` with a
  ``register_*_routes`` function.
- Import and call it from ``reef/service/routes/__init__.py`` in
  ``register_routes``.
- Keep parsing and response adaptation in the route. Put reusable request or
  domain behavior in ``reef/service/request_service.py`` or the package that
  owns the concept.
- Raise domain errors. If a new error needs a non-default HTTP status, add it
  to ``ERROR_STATUS_TABLE`` in ``reef/service/errors.py`` instead of adding a
  route-local exception table.
- Add route and service tests under ``tests/reef_service/``. Update the wire
  guide for a public endpoint; the docs contract check requires the guide to
  name every registered route.
- Treat removal, incompatible payload changes, and changes to authentication
  rules as public API changes requiring an RFC and contract tests.

Add a harness adapter
---------------------

Use this playbook when Reef must render and run the same composition tree for
another agent harness. Benchmark, grader, and environment orchestration stays
under ``recipes/<name>/examples/`` or in the external harness.

- For a bundled adapter, add ``reef/harness/adapters/<name>/adapter.yaml``,
  whose ``evolve`` section declares the parts of the agent's tree, and its
  package initializer. Add a ``render.py`` with an
  ``AdapterRenderer`` subclass, named by the descriptor's ``renderer``, only
  for behavior the declarative files cannot express.
- Add an asset, such as Codex's ``bundled_models.json``, only when the
  adapter needs it, then include the asset in the package-data rules in
  ``pyproject.toml``.
- For an external adapter, publish an entry in the
  ``reef.harness_adapters`` entry-point group that resolves to an
  ``AdapterDescriptor`` or a zero-argument descriptor factory.
- Add descriptor and golden-render coverage in
  ``tests/harness/test_harness_render.py``. Add run, cleanup, install,
  and trajectory tests for the capabilities the descriptor declares.
- Put golden trees under ``tests/harness/data/harness_goldens/`` and a
  real-binary smoke test under ``tests/smoke/`` when CI can install the pinned
  harness version.
- Update `HTTP API <../reference/http-api.rst>`__ when the public
  install or composition contract changes.

Definition of done
------------------

For every playbook:

- the implementation is in the owning package and does not reverse an
  existing package dependency;
- registration happens through the documented registry or entry point;
- focused unit and contract tests cover the new interface;
- configuration, package data, and dependencies are declared at their
  repository-level homes;
- user, operator, and contributor documentation describe the public change;
  and
- ``pre-commit run --all-files`` and the relevant focused pytest command pass.

Run the full ``pytest tests/`` suite in the supported container environment
before merging an integration change.
