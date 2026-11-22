Python API for custom learning methods
======================================

The Python API is the set of extension points that a learning method, a storage plugin or a serving integration implements. This page says which page holds each contract.

Reef owns everything around the extension points: it accepts and replays records, holds a job's data until it is acknowledged, commits algorithm state, submits training jobs, and publishes the saved checkpoints. `Write a recipe <../developer-guide/write-a-recipe.rst>`__ is the executable tutorial. These pages give the signatures, defaults and errors.

What to implement
-----------------

Start with a recipe and add only what the method needs.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Need
     - Contract
     - Required for
   * - configure deployment behavior
     - `Recipe <python-api-recipe.rst#recipe>`__, with its live objects and configuration
     - every deployment
   * - validate feedback at ingress
     - `Report <python-api-recipe.rst#report>`__
     - methods whose signal arrives in reports
   * - decide which artifacts serve
     - `Activation rule <python-api-recipe.rst#activation-rule>`__
     - optional, any recipe
   * - turn records into typed job data
     - `Pipeline stages <python-api-pipeline.rst>`__, with the report and judging engines and task generators
     - every method producing updates
   * - carry data to the backend
     - `Job data <python-api-job-data.rst>`__, with trajectory samples, task samples and chat rendering
     - every method producing updates; subclass only for a new shape
   * - turn job data into a signal
     - `Training objective <python-api-training.rst#training-objective>`__, with step scheduling and job parameters
     - weight-training methods
   * - evolve a harness tree and score a task run
     - `Harness method <python-api-harness.rst#harness-method>`__
     - harness-evolution methods
   * - evaluate a produced candidate
     - `Candidate evaluation <python-api-harness.rst#candidate-evaluation>`__
     - optional, any recipe
   * - declare a new snapshot component
     - `Scenario components <python-api-artifacts.rst#scenario-components>`__
     - only a new evolution medium
   * - a new tensor objective
     - `Loss family <../developer-guide/loss-families.rst>`__
     - rarely

Integrations and storage plugins implement these contracts:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Need
     - Contract
   * - run training jobs on a new backend
     - `Training jobs and components <python-api-training.rst#training-jobs-and-components>`__, the job loop, and the `Tinker integration <python-api-training.rst#tinker-integration>`__
   * - serve published weights
     - `Inference <python-api-inference.rst>`__: runtimes, handlers, backends and the saved-weight runtimes
   * - embed the service in an application
     - `Embed the service <python-api-recipe.rst#embed-the-service>`__
   * - store commits, settings and records
     - `Scenario and record stores <python-api-scenario-stores.rst>`__
   * - store tasks, task files, data items and segments
     - `Task and data stores <python-api-data-stores.rst>`__
   * - store or import artifacts
     - `Artifacts <python-api-artifacts.rst>`__: artifact backends, imports and base roots

Method code must depend only on what these pages document. Anything else under ``reef.`` is an implementation detail and can change.

Every extension contract is an abstract base class. Inherit it explicitly and implement each abstract member. Reef rejects an object that only has methods of the same names. Optional capabilities are separate base classes: inherit only those that the implementation supports.
