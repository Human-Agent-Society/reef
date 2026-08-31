Adding components
=================

First use the `Codebase structure <codebase-structure.rst>`__ to choose the
owner, then use the matching playbook below.

Core and external extensions
----------------------------

Recipes and learning methods are external packages selected by dotted
reference; Reef does not bundle or register them. Changes to shared runtime or
training machinery start with an `RFC issue
<https://github.com/Human-Agent-Society/reef/issues/new?template=rfc.yml>`__.
A new top-level package, persisted format, wire contract, or incompatible
public API also requires an RFC.

Add a recipe
------------

Use this playbook when a new method binds record processing, step preparation,
runtime execution, and artifact delivery. Read `Write a recipe
<../developer-guide/write-a-recipe.rst>`__ and the `Python API <../reference/python-api.rst>`__ first.

Implementation
~~~~~~~~~~~~~~

- Keep a recipe with its method package and select its frozen recipe dataclass
  as ``package.module:ClassName``. There is no ``register_kind`` step and no
  import from ``reef/__init__.py``. The contract it implements is
  ``reef/recipe/``.
- Declare recipe-owned settings with ``config_field``. Do not parse those
  settings again in ``reef/service/``.
- Put the method's record-to-batch behavior in
  ``<method-package>/processor.py``, subclassing a processor contract
  from ``reef/train/processors/``; extend those contracts only when they do
  not already express it.
- Put the method's backend-neutral step preparer in
  ``<method-package>/preparer.py`` (``reef/train/algos/`` holds the
  contract). Backend-specific payload
  construction belongs to the concrete integration.
- A method with its own tensor objective adds a loss family in
  ``<method-package>/slime/`` (spec in ``__init__.py``, hooks in
  ``objective.py``) and names it in the recipe's ``training_spec()``;
  the `Loss families
  <../developer-guide/loss-families.rst>`__ lists what a family declares.
- Override ``build_surface`` only when the produced artifact needs delivery
  behavior beyond the existing surfaces.

Surrounding changes
~~~~~~~~~~~~~~~~~~~

- Add dotted-resolution and configuration tests in
  ``tests/reef_service/test_reef_recipes.py`` and
  ``tests/reef_service/test_recipe_config_fields.py``.
- Add method behavior tests under ``tests/reef_service/``. Add an integration
  contract when the method calls into a concrete backend.
- Put every runnable configuration beside its method under
  ``recipes/<name>/examples/``; only a stack that binds no method belongs in
  ``recipes/basic/``.
- Add ``docs/user-guide/recipes/<name>.rst`` and update the public README recipe
  table when the cookbook carries the method. Update the relevant processor, preparer, or surface
  reference if its public contract changes.

Add a training integration
--------------------------

Use this playbook for a concrete external training stack. It must implement the
contracts described by the `Python API <../reference/python-api.rst>`__; do not let
framework types leak into backend-neutral modules.

Implementation
~~~~~~~~~~~~~~

- Start an accepted integration in ``reef/train/<integration>/``. Keep its
  implementation, framework adapters, bridge code, and plugins inside that
  subtree.
- Change ``reef/train/backend.py`` or ``reef/runtime/base.py`` only when the
  existing backend-neutral contract is insufficient for more than one
  integration. Contract changes need focused compatibility tests.
- Keep step signals in ``reef/train/algos/`` backend-neutral. Translate them
  into framework payloads inside the integration.
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
  framework. Extend ``tests/reef_service/test_dependency_boundaries.py`` when
  a new package dependency rule is added.
- Update the ``reef/train`` package docstring, the Integration API,
  deployment guidance, and the configuration reference.

Add a runtime kind
------------------

Use this playbook for a new inference provider or a runtime implementation that
satisfies Reef's backend-neutral lifecycle. Read the runtime contract in
``reef/runtime/base.py`` before adding configuration.

- For an external runtime, expose a factory as
  ``package.module:factory_name`` and use that dotted value as the runtime
  ``type``.
- For an accepted bundled runtime, add an adapter under
  ``reef/runtime/adapters/``. Subclass ``RuntimeFactory``, set its ``kind``,
  implement ``__call__``, decorate the class with ``@register_runtime_kind``,
  and import its module from ``reef/runtime/adapters/__init__.py``.
- Keep provider authentication and provider-native request handling in the
  adapter. Do not import a concrete training backend from ``reef/runtime/``.
- Test registered and dotted resolution in
  ``tests/reef_service/test_runtime_registry.py``. Add request, recovery, and
  malformed-result contracts for every capability the runtime implements.
- Add or update the stack that exercises the runtime under ``recipes/`` and
  document every public setting. Never commit credentials or real provider tokens.

Add a surface or artifact type
------------------------------

Artifacts own immutable bytes and version heads; surfaces own validation and
delivery. A change that needs both still keeps those responsibilities in
separate packages. Read the `Python API <../reference/python-api.rst#surface>`__ and the
release-chain section of `Architecture <../getting-started/architecture.rst>`__.

- Put storage, materialization, or version identity behavior in
  ``reef/artifact/``. A new storage backend implements ``RepositoryBackend``
  and is exposed through a repository backend factory.
- Put request preparation and response verification in ``InferenceHooks``,
  restore/recovery behavior in ``ArtifactLoader``, and client-pull delivery in
  ``FileTree`` under ``reef/surface/``.
- Build the surface from the owning recipe; do not make the surface choose a
  method or evaluate a candidate.
- Test surface behavior in ``tests/reef_service/test_surfaces.py`` or the
  closest artifact-specific surface test. Test storage backends against the
  contracts in ``tests/reef_service/test_reef_artifacts.py``.
- Add scenario commit, rollback, and recovery tests whenever persistence or
  activation ordering changes. Persisted-format changes require an RFC and a
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

- For a bundled adapter, add ``reef/harness/adapters/<name>/descriptor.yaml``
  and its package initializer. Use a ``quirks.py`` module only for behavior
  the declarative descriptor cannot express.
- Add an asset such as ``version_check.ts`` only when the adapter needs it,
  then include the asset in the package-data rules in ``pyproject.toml``.
- For an external adapter, publish an entry in the
  ``reef.harness_adapters`` entry-point group that resolves to an
  ``AdapterDescriptor`` or a zero-argument descriptor factory.
- Add descriptor and golden-render coverage in
  ``tests/reef_service/test_harness_render.py``. Add episode, cleanup, install,
  and trajectory tests for the capabilities the descriptor declares.
- Put golden trees under ``tests/reef_service/data/harness_goldens/`` and a
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
