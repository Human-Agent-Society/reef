Configure Reef serving and training
===================================

Every setting of ``reef serve`` is a key in one section of a configuration file, and each key is also a command-line flag with the same path, such as ``--inference.model-path``. This page lists the sections and names the page that holds the keys of each one.

.. code:: bash

   reef serve -c reef.yaml --reef.port 9000 --print-config

Reef reads a file only when ``-c`` selects one, and it rejects an unknown section or key. ``--print-config`` prints every resolved setting with its source and starts nothing. `Configuration file layout <configuration-files.rst>`__ gives the layout, the value rules and the starts that need no file.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Section
     - Holds
     - Page
   * - ``reef``
     - HTTP host, port, authentication, run directory, readiness deadline and scenario limits
     - `Service and storage settings <configuration-service.rst>`__
   * - ``reef.routing_manifest``, ``reef.routing_deployment``
     - routing of a scenario's requests among external targets, and the Semantic Router lifecycle
     - `Scenario routing between students and teachers <configuration-routing.rst>`__
   * - ``harness``
     - how harness clients reach the deployment's services
     - `Service and storage settings <configuration-service.rst>`__
   * - ``storage``
     - artifact, record, data and task-file stores, and record retention
     - `Service and storage settings <configuration-service.rst#the-storage-section>`__
   * - ``data``
     - the data service's limits
     - `Service and storage settings <configuration-service.rst#the-data-section>`__
   * - ``evaluation``
     - sandbox limits and plugins of the evaluation service
     - `Service and storage settings <configuration-service.rst#the-evaluation-section>`__
   * - ``inference``
     - model path, backend, GPUs, provider connection, serving mode, request handlers and native engine ``options``
     - `The inference section <configuration-inference.rst>`__
   * - ``training``, ``teacher``
     - backend, timeouts, Ray connection, workload variables (``config``) and native training ``options``; a frozen SGLang teacher engine
     - `The training section <configuration-training.rst>`__
   * - ``execution``, ``executors``
     - executor selection for each role, and named executor profiles
     - `Stack execution backends <configuration-execution.rst>`__
   * - ``recipe``
     - ``implementation`` (the recipe class or preset), ``config`` (the recipe's declared fields) and ``runtime`` (a runtime type and its settings)
     - `Recipe configuration <configuration-recipe.rst>`__
   * - ``recipe.config.evolution``
     - the method, adapter, limits, run executor and evaluation of harness evolution
     - `Harness evolution keys <configuration-harness.rst>`__
   * - ``activation``
     - the rule that decides which new artifacts serve
     - `Recipe configuration <configuration-recipe.rst#the-activation-section>`__
   * - ``pipeline``
     - the config, and optionally the class, of each pipeline stage
     - `Recipe configuration <configuration-recipe.rst#the-pipeline-section>`__
   * - ``generator``, ``task-import``
     - the generator service that writes and checks Harbor tasks and runs their rollouts, and the Harbor tasks that each scenario imports
     - `Generated tasks <configuration-generated-tasks.rst>`__
   * - ``observability``
     - experiment tracking (``wandb``) and record tracing (``tracing``)
     - `Observability settings <configuration-observability.rst>`__

Some topics cover more than one section:

- **Serving of a weight-training deployment**, the engines' native options and the chat template arguments: `The inference section <configuration-inference.rst#serving-of-a-weight-training-deployment>`__.
- **Slime jobs** that run their own rollouts, and training without an inference engine: `Task jobs on Slime <configuration-training.rst#task-jobs-on-slime>`__, `Dataset jobs on Slime <configuration-training.rst#dataset-jobs-on-slime>`__ and `Train without an inference engine <configuration-training.rst#train-without-an-inference-engine>`__.
- **The generated-task recipe**, its rounds, its training pool and its rollout parameters: `Generated tasks <configuration-generated-tasks.rst>`__.
- **Operational metrics** of a scenario: `Observability settings <configuration-observability.rst#operational-metrics>`__.
- **Cookbook stacks** to start from: `Start from a cookbook stack <configuration-files.rst#start-from-a-cookbook-stack>`__.
