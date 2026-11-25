Configure a deployment
======================

Every setting of ``reef serve`` is a key in one section of a YAML file, and each key is also a command-line flag with the same path, such as ``--inference.model-path``. This page gives the layout of the file and the rules by which Reef reads values. `Settings reference <configuration-settings.rst>`__ lists every key with its type, its default and what it does.

A typical file
--------------

Most deployments set only these keys. Every other key has a default.

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

Run it with ``reef serve -c reef.yaml``. ``reef serve -c reef.yaml --print-config`` prints every resolved setting with its source and starts nothing (`Reef CLI: serve <cli.rst>`__). The runnable stacks of the source checkout are under ``recipes/`` and ``tutorials/``.

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

Rules of the layout:

- Reef rejects an unknown section and an unknown key.
- A section name and a key can use hyphens or underscores. Both spellings of one key in one file are an error. The few sections that take underscores only say so in the settings reference.
- A file holds no process definitions: Reef rejects ``service``, ``services`` and ``execution.services``. The recipe and the backends decide which processes start.

Start without a file
--------------------

Local inference needs one flag, and an external provider needs two:

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct
   reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model

- ``--inference.model-path`` cannot be combined with an upstream URL or model.
- Without ``-c``, Reef reads ``REEF_UPSTREAM_URL``, ``REEF_UPSTREAM_MODEL``, ``REEF_UPSTREAM_API_KEY`` and ``REEF_TOKEN`` as fallbacks, and a flag wins over them. An exported ``REEF_UPSTREAM_URL`` therefore makes ``--inference.model-path`` fail. With ``-c``, Reef does not read them.
- Reef does not start or stop the provider, and readiness does not check the provider's credentials or models.
- ``REEF_PYTHON`` names the interpreter of the services; the default is the one that launched ``reef serve``. For a local engine, it must have the engine and GPU-enabled PyTorch installed.
- Weight training needs no file either: pass ``--recipe.implementation``, ``--inference.model-path`` and the ``--training.options.*`` flags that the recipe needs. Reef then looks for the recipe's package from the working directory.

How Reef reads values
---------------------

A file and the command line use the same parser. ``reef serve --help`` lists the flags, and ``reef serve -c reef.yaml --help`` adds the flags of the selected recipe.

- **Precedence.** The command line, then the file, then an environment fallback that a setting declares, then the default. The last occurrence of a flag wins. A selected file that is missing or invalid is an error.
- **Strings.** A command-line string keeps its text: ``--inference.upstream-model 00123`` stays ``00123``. In a file, quote a string that YAML reads as a number or a boolean.
- **Numbers and booleans.** Reef parses them by the declared type, and a number must be finite. A boolean takes ``true``, ``yes``, ``on`` or ``1``, or ``false``, ``no``, ``off`` or ``0``. A bare flag is true, and ``--no-<path>`` is false, for example ``--no-reef.allow-implicit-scenario-creation``.
- **Lists and objects.** A flag takes one quoted JSON or YAML value, such as ``--reef.tokens '[]'``, and that value replaces the value of the file.
- **Empty string.** An empty value, such as an unset ``${VAR}``, counts as omitted and keeps the default. ``reef.host`` and ``inference.model-path`` refuse an empty value.
- **Null.** In a file, ``null`` gives an optional key no value. For any other key, ``null`` is an error that names the key: omit the key to use its default.

**Interpolation.** A value of the file can take text from the environment or from the file itself:

- ``${VAR}`` is the environment variable, or an empty string when the variable is unset.
- ``${VAR:?}`` requires the variable. If it is unset or blank, Reef reports the variable and its key and starts nothing. A command-line value for that key supplies it instead.
- ``${dotted.path}`` is another key of the config, such as ``${training.config.global_batch_size}``. It resolves only a key that the file or the command line sets. A reference to a key at its default stays as text.

**The report of resolved settings.** Before any model download or process start, the launcher logs each setting with its source: ``file``, ``command line``, ``environment``, ``automatic`` (a value that Reef chose because the key was omitted) or ``default``. It lists a default only for the recipe and the HTTP bind; ``--print-config`` lists every default. ``--print-config`` exits with status 0, or with 2 for an invalid config.

.. warning::

   The report masks tokens, API keys, passwords and database URLs by the name of their key, at any depth. It does not mask a credential inside a URL of another key, such as ``inference.upstream-url`` or ``storage.artifact-repository``.

Startup failures
----------------

If a service exits before it is ready, or passes its readiness deadline, Reef stops the stack. It reports the service, the reason and the log directory. Service output goes to the logs under ``reef.run-dir`` and to the terminal.

``reef serve`` exits with status 1 for a startup failure and with status 2 for an invalid configuration. `Troubleshooting <../user-guide/troubleshooting.rst>`__ lists common causes.
