Codebase structure
==================

Find the package that owns your change, then read its ``__init__.py`` for the contract.

Layers
------

Packages may import their own layer and lower layers.

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
     - Select samples, train and evaluate candidates, serve models, run harnesses and route requests.
   * - Foundation
     - ``core``, ``storage``, ``artifact``, ``execution``, ``observability``, ``tinker``
     - Provide shared types, persistence, artifact files and execution.

These layers describe code dependencies. The `seven services
<../user-guide/split-services.rst>`__ describe how a deployment runs that code
and exchanges work through the shared store.

Keep HTTP parsing in ``reef/api/routes/``. Control operations live in ``reef/control/operations/`` and work without HTTP.
Put method-specific code in ``recipes/<name>/``. Use the owning package's tests.
``tests/contributing/test_dependency_boundaries.py`` checks the import boundaries.

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

For setup and checks, see `Development <development.rst>`__. For review and RFC policy, read `CONTRIBUTING.md <../../CONTRIBUTING.md>`__.
