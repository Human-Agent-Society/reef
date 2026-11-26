Codebase structure
==================

Use this page to find where a change belongs. Read that package's
``__init__.py`` docstring for its contracts, then extend the nearby code and tests.
For setup and checks, see `Development <development.rst>`__.

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

``reef/deploy/`` reads deployment configuration and starts, supervises and stops
processes. ``reef/api/`` assembles each service's HTTP app and handles routes,
authentication and streaming. The dependency runs from ``deploy`` to ``api``.

For an HTTP endpoint, start in ``reef/api/routes/``. Parse the request there,
then call the operation that owns the behavior. Update the
`HTTP reference <../reference/http-api.rst>`__ and its tests in ``tests/api/``.
Deployment configuration and process changes belong in ``deploy`` and
``tests/deploy/``.

Control layer
-------------

``reef/control/`` coordinates the scenarios loaded in a process:

- ``scenario/`` owns scenario state, commit ordering, recovery and lifecycle.
- ``workers/`` coordinates background work.
- ``operations/`` handles client operations, such as importing artifacts or
  starting evaluations, without HTTP request objects.

For example, a harness import route reads the scenario header and passes the
scenario name to an operation. A background sync worker passes the same name
directly. Neither caller needs to turn its work into a synthetic HTTP request.

Use ``tests/control/`` for scenario behavior. Keep HTTP-specific assertions in
``tests/api/``. Control imports neither the entry layer nor ``aiohttp``.

Domain layer
------------

Choose the capability that owns the behavior:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Owns
   * - ``reef/samples/``
     - Select records, group them and make training samples; generate Harbor tasks.
   * - ``reef/train/``
     - Shape samples into job data (``sample2job/``), run training jobs and connect training backends.
   * - ``reef/evaluation/``
     - Run and score candidates; apply harness selection and artifact activation rules.
   * - ``reef/inference/``
     - Serve model requests, start engines and load saved weights.
   * - ``reef/harness/``
     - Describe, render and run harness trees; read trajectories and sync files.
   * - ``reef/recipe/``
     - Define recipe configuration and shared recipe contracts.
   * - ``reef/routing/``
     - Integrate the Semantic Router.

A new learning method belongs in ``recipes/<name>/``. Keep its runnable
examples in ``recipes/<name>/examples/`` and its tests in
``tests/recipes/<name>/``. Core Reef never imports these method packages.
Shared mechanisms belong in ``reef/``.

Keep concrete engines and training frameworks inside their integration
packages. Training and inference do not import each other; shared values go
in ``core``.

``TrainingPipeline`` stays in ``reef/train/pipeline.py``: its two stages share
sample storage and state under the trainer's lock. Splitting that public
interface needs a separate design. The generated-task recipe and
``TaskGenerator`` stay in ``reef/recipe/generated_tasks/`` because
``recipes/`` is not included in the installed wheel.

Foundation layer
----------------

- ``reef/core/``: shared value types, wire contracts and errors; no storage or I/O.
- ``reef/storage/``: scenario state, data items and coordination queues.
  ``scenario/``, ``data/`` and ``coordination/`` define the stores;
  ``sql/`` implements them in SQLite and PostgreSQL.
- ``reef/artifact/``: versioned artifact files, repositories and materialization.
  Loading those files into an engine belongs in ``inference``.
- ``reef/execution/``: worker processes, executors, sandboxes and the queue worker loop.
- ``reef/observability/``: shared metrics, tracing and experiment logging.
- ``reef/tinker/``: SDK code shared by Tinker training and serving.

Storage contracts do not import database implementations. Dependency tests in
``tests/contributing/test_dependency_boundaries.py`` check package cycles and
the layer boundaries.

Extending Reef
--------------

Use the guide for the extension you are writing. Its contracts belong in the
named classes' docstrings.

- `Write a recipe <../developer-guide/write-a-recipe.rst>`__ or
  `a harness method <../developer-guide/write-a-harness-method.rst>`__.
- Add a `training backend <../developer-guide/training-backends.rst>`__ or
  `serving runtime <../developer-guide/serving-runtimes.rst>`__.
- Add a `harness adapter <../developer-guide/harness-adapters.rst>`__ or
  `task generator <../developer-guide/task-generators.rst>`__.
- Define a `snapshot component <../developer-guide/components.rst>`__ or
  `worker executor <../developer-guide/executors.rst>`__.

Read `CONTRIBUTING.md <../../CONTRIBUTING.md>`__ for review and RFC policy.
Keep dependencies and package data in ``pyproject.toml``, environments in
``docker/``, and tests under their `owning component
<https://github.com/Human-Agent-Society/reef/blob/main/tests/AGENTS.md>`__.
