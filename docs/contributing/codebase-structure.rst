Codebase structure
==================

Use the four layers below to find the package that owns your change.
Read its ``__init__.py`` docstring for the package contract.
See `Development <development.rst>`__ for setup and checks.

Layers
------

Reef has four code layers. Packages may import their own layer and lower
layers. They must not import a higher layer.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Layer
     - Main packages
     - Responsibility
   * - Entry
     - ``deploy``, ``api``
     - Start deployments and receive HTTP requests.
   * - Control
     - ``control``
     - Coordinate scenarios, their lifecycle, commits and recovery.
   * - Domain
     - ``samples``, ``train``, ``evaluation``, ``inference``, ``harness``, ``recipe``, ``routing``
     - Implement each capability.
   * - Foundation
     - ``core``, ``storage``, ``artifact``, ``execution``, ``observability``, ``tinker``
     - Provide shared types, persistence, artifact files and execution.

These layers describe code dependencies. The `seven services
<../user-guide/split-services.rst>`__ describe how a deployment runs that code
and exchanges work through the shared store.

Entry layer
-----------

``deploy`` starts and supervises processes. ``api`` parses HTTP requests and calls control operations.
For a route change, update ``reef/api/routes/``, ``tests/api/`` and the `HTTP reference <../reference/http-api.rst>`__.

Control layer
-------------

In ``reef/control/``, ``scenario/`` owns lifecycle, commits and recovery.
``operations/`` handles client requests and ``workers/`` runs background work.
Keep HTTP objects in the entry layer. Test scenario behavior in ``tests/control/``.

Domain layer
------------

- ``samples`` selects records and builds training samples. ``train`` groups samples into jobs and runs training backends.
- ``evaluation`` runs and scores candidates. ``inference`` serves requests and loads weights.
- ``harness`` renders, runs and syncs harness trees. ``routing`` integrates Semantic Router.
- ``recipe`` defines shared recipe contracts. Put method-specific code in ``recipes/<name>/`` and tests in ``tests/recipes/<name>/``.

Training and inference share types through ``core``. Keep concrete engines inside their integrations.

Foundation layer
----------------

- ``core`` defines shared values, wire contracts and errors.
- ``storage`` persists scenario state, data items and coordination queues. ``artifact`` stores versioned files.
- ``execution`` runs processes, executors and sandboxes. ``observability`` provides metrics and tracing.
- ``tinker`` provides SDK code for training and serving.

``tests/contributing/test_dependency_boundaries.py`` checks imports across these boundaries.

Extending Reef
--------------

- `Write a recipe <../developer-guide/write-a-recipe.rst>`__ or
  `a harness method <../developer-guide/write-a-harness-method.rst>`__.
- Add a `training backend <../developer-guide/training-backends.rst>`__ or
  `serving runtime <../developer-guide/serving-runtimes.rst>`__.
- Add a `harness adapter <../developer-guide/harness-adapters.rst>`__ or
  `task generator <../developer-guide/task-generators.rst>`__.
- Define a `snapshot component <../developer-guide/components.rst>`__ or
  `worker executor <../developer-guide/executors.rst>`__.

For review and RFC policy, read `CONTRIBUTING.md <../../CONTRIBUTING.md>`__.
