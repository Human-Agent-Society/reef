Codebase structure
==================

Find the package that owns your change, then read its ``__init__.py`` for the contract.

Layers
------

A package can import from its own layer or any layer below it.

.. container:: layer-map

   .. container:: flow

      Entry
          ``deploy`` starts processes. ``api`` receives HTTP requests.

      Control
          ``control`` coordinates scenario changes, commits and recovery.

      Domain
          ``recipe`` defines learning methods. ``samples`` creates samples, ``train`` runs jobs and ``evaluation`` scores candidates.
          ``inference`` serves models, ``routing`` routes requests and ``harness`` runs agents.

      Foundation
          ``core`` defines shared types, ``storage`` persists state and ``artifact`` manages versioned files.
          ``execution`` runs workers, ``observability`` collects telemetry and ``tinker`` connects training and serving to the Tinker SDK.

Keep method-specific code in ``recipes/<name>/``. Use the owning package's tests.
See the `service diagram <../user-guide/split-services.rst>`__ for the runtime layout.

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
