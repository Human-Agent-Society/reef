Configure a deployment
======================

Configure ``reef serve`` with a YAML file, command-line flags, or both.
Flags use the same paths as YAML keys, such as ``--inference.model-path``.
The `Settings reference <configuration-settings.rst>`__ lists types and defaults.

A typical file
--------------

This example uses SAO with Slime and local inference.
It requires a checkout and the `GPU training environment
<../getting-started/installation.rst#gpu-image-for-weight-training>`__.

.. code:: yaml

   reef:
     host: 127.0.0.1
     port: 8900
     token: ${REEF_TOKEN}             # the bearer token that clients send
   storage:                           # each path defaults under .reef/ in the launch directory
     agent-record-dir: /data/reef/agent-record
     artifact-repository: /data/reef/artifacts.git
     artifact-work-dir: /data/reef/artifact-work
     artifact-cache-dir: /data/reef/artifact-cache
   inference:
     model-path: Qwen/Qwen3-8B        # or upstream-url and upstream-model for a provider
     num-gpus: 1
     options:                         # native flags of the engine
       mem-fraction-static: 0.6
   training:
     backend: slime
     options:                         # native flags of the training backend
       actor-num-nodes: 1
       actor-num-gpus-per-node: 1
       global-batch-size: 32
   recipe:
     implementation: recipes.sao.recipe:SAORecipe
   pipeline:
     sample2job:
       config:
         job-samples: 32

Save the file as ``reef.yaml`` and run ``reef serve -c reef.yaml``.
To inspect resolved values without starting services, add ``--print-config``.
For complete examples, use the configurations under ``recipes/`` and ``tutorials/``.

Sections
--------

.. list-table::
   :header-rows: 1
   :widths: 24 40 36

   * - Section
     - Holds
     - Described in
   * - ``reef``, ``harness``
     - the HTTP service, its tokens, the scenario limits and request routing
     - `Operate a deployment <../user-guide/operate.rst>`__, `Scenario routing <configuration-routing.rst>`__
   * - ``storage``, ``data``
     - the stores of the deployment and record retention
     - `Choose where a deployment stores its data <../user-guide/storage.rst>`__
   * - ``inference``
     - the model or provider, and the serving engines
     - `Inference serving <configuration-inference.rst>`__
   * - ``training``, ``teacher``
     - the training backend and the teacher engine
     - `Slime training options and jobs <configuration-training.rst>`__, `Train with Tinker <../user-guide/tinker.rst>`__
   * - ``recipe``, ``activation``, ``pipeline``
     - the recipe, the rule that decides which artifacts serve, and the pipeline stages
     - `Recipe configuration <configuration-recipe.rst>`__
   * - ``recipe.config.evolution``
     - harness evolution
     - `Harness evolution keys <configuration-harness.rst>`__
   * - ``generator``, ``task-import``
     - the generator service and the imported Harbor tasks
     - `Generated tasks <configuration-generated-tasks.rst>`__
   * - ``evaluation``
     - the evaluation service
     - `Run the services in separate processes <../user-guide/split-services.rst>`__
   * - ``execution``, ``executors``
     - how Reef starts the workers of each role
     - `Executors <../developer-guide/executors.rst>`__
   * - ``observability``
     - experiment tracking and record tracing
     - `Observability settings <configuration-observability.rst>`__

Layout rules:

- Reef rejects unknown sections and keys.
- Section names and keys accept hyphens or underscores unless the settings reference requires underscores.
  Do not use both spellings of one key in a file.
- The recipe and backends define the processes. Reef rejects ``service``,
  ``services`` and ``execution.services`` in the file.

Start without a file
--------------------

Start a local model or connect an external provider:

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct
   reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model

- ``--inference.model-path`` cannot be combined with an upstream URL or model.
- Without ``-c``, Reef uses ``REEF_UPSTREAM_URL``, ``REEF_UPSTREAM_MODEL``,
  ``REEF_UPSTREAM_API_KEY`` and ``REEF_TOKEN`` as fallbacks. Flags take precedence.
  An exported ``REEF_UPSTREAM_URL`` conflicts with ``--inference.model-path``.
  With ``-c``, Reef does not use these automatic fallbacks.
- Reef does not manage the provider's lifecycle. Readiness does not validate
  its credentials or model IDs.
- ``REEF_PYTHON`` selects the service interpreter. Its default is the interpreter
  that launched Reef. Local inference requires the engine and GPU-enabled PyTorch in that environment.
- For weight training, set ``--recipe.implementation``, ``--inference.model-path``
  and the required ``--training.options.*`` flags. Reef resolves the recipe
  package from the working directory.

How Reef reads values
---------------------

Files and flags use the same type parser. ``reef serve --help`` lists flags.
Add ``-c reef.yaml`` to include the selected recipe's flags.

- **Precedence:** command line, file, declared environment fallback, default.
  For repeated flags, the last value wins. A missing or invalid selected file is an error.
- **Strings:** flags preserve text, so ``--inference.upstream-model 00123`` keeps
  ``00123``. In YAML, quote strings that resemble numbers or booleans.
- **Numbers:** values must match the declared type and be finite.
- **Booleans:** use ``true``, ``yes``, ``on`` or ``1`` for true;
  ``false``, ``no``, ``off`` or ``0`` for false.
  A bare flag means true. ``--no-<path>`` means false, as in
  ``--no-reef.allow-implicit-scenario-creation``.
- **Lists and objects:** pass one quoted JSON or YAML value.
  For example, ``--reef.tokens '[]'`` replaces the file's token list.
- **Empty strings:** empty values, including unset ``${VAR}`` references, count as omitted.
  Exceptions: ``reef.host`` and ``inference.model-path`` reject empty values.
- **Null:** YAML ``null`` clears an optional value. Other keys reject it.
  Omit a key to use its default.

**Interpolation.** File values can reference environment variables or other keys:

- ``${VAR}`` is the environment variable, or an empty string when the variable is unset.
- ``${VAR:?}`` requires a nonempty variable. If it is missing, startup fails
  with the variable name and key. A flag for that key overrides the reference.
- ``${dotted.path}`` references a key, such as ``${reef.port}``.
  Only explicit file or flag values resolve. A reference to a default value stays as text.

**Resolved settings.** Before downloads or startup, Reef logs values and their
sources: ``file``, ``command line``, ``environment``, ``automatic`` or ``default``.
``automatic`` identifies a value Reef chose for an omitted key.
Startup logs include defaults only for the recipe and HTTP bind.
``--print-config`` includes all defaults and exits with status 0, or 2 for invalid configuration.

.. warning::

   Reef masks tokens, API keys, passwords and database URLs by key name, at any depth.
   Credentials in other URLs can remain visible, including
   ``inference.upstream-url`` and ``storage.artifact-repository``.

Startup failures
----------------

If a service exits early or exceeds its readiness deadline, Reef stops the stack.
The error identifies the service, reason and log directory.
Service output goes to the terminal and logs under ``reef.run-dir``.

Exit status 1 means startup failed. Status 2 means invalid configuration.
See `Troubleshooting <../user-guide/troubleshooting.rst>`__ for common causes.
