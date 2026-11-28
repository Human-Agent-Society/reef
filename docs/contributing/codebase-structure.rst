Codebase structure
==================

Layers
------

Packages can import from their own layer and the layers below.

.. container:: layer-map

   .. container:: flow

      Entry
          ``deploy`` starts services. ``api`` handles HTTP.

      Control
          ``control`` manages scenario state, commits and recovery.

      Domain
          Learning (``recipe``, ``samples``, ``train``, ``evaluation``), model serving (``inference``, ``routing``) and agents (``harness``).

      Foundation
          Types (``core``), persistence (``storage``), files (``artifact``), workers (``execution``), telemetry (``observability``) and the Tinker SDK (``tinker``).

For runtime boundaries, see the `seven-service diagram <../user-guide/split-services.rst>`__.

Extending Reef
--------------

Read the package's ``__init__.py`` for its contract. Put method-specific code in ``recipes/<name>/``.

- `Recipes <../developer-guide/write-a-recipe.rst>`__ and `harness methods <../developer-guide/write-a-harness-method.rst>`__
- `Training backends <../developer-guide/training-backends.rst>`__ and `serving runtimes <../developer-guide/serving-runtimes.rst>`__
- `Harness adapters <../developer-guide/harness-adapters.rst>`__ and `task generators <../developer-guide/task-generators.rst>`__
- `Snapshot components <../developer-guide/components.rst>`__ and `worker executors <../developer-guide/executors.rst>`__
