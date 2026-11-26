Configure a deployment
======================

Use a YAML file, command-line flags or both. Flags follow the YAML path, such as ``--inference.model-path``.
Find field types and defaults in the `Settings reference <configuration-settings.rst>`__.

A typical file
--------------

Save this as ``reef.yaml``. It connects a local Reef service to a hosted model.

.. code:: yaml

   inference:
     upstream-url: https://api.openai.com
     upstream-api-key: ${REEF_UPSTREAM_API_KEY:?}
     upstream-model: gpt-4o

With the provider key set, run ``reef serve -c reef.yaml``. Add ``--print-config`` to inspect settings before startup.

Sections
--------

- Configure `serving engines <configuration-inference.rst>`__, `routing <configuration-routing.rst>`__ and `storage <../user-guide/storage.rst>`__.
- Choose a `recipe and activation rule <configuration-recipe.rst>`__, then configure `Slime <configuration-training.rst>`__ or `Tinker <../user-guide/tinker.rst>`__.
- For agents, see `harness evolution <configuration-harness.rst>`__ and `generated tasks <configuration-generated-tasks.rst>`__.
- Manage `services <../user-guide/split-services.rst>`__, `worker executors <../developer-guide/executors.rst>`__ and `observability <configuration-observability.rst>`__.

Use hyphens or underscores consistently for each key. Some sections require underscores.
Reef rejects unknown layout keys and explicit ``service``, ``services`` or ``execution.services`` sections.
The recipe and backends select the processes.

Start without a file
--------------------

Start a local model or connect an external provider:

.. code:: bash

   reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct
   reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model

Without ``-c``, Reef reads ``REEF_UPSTREAM_URL``, ``REEF_UPSTREAM_MODEL``, ``REEF_UPSTREAM_API_KEY`` and ``REEF_TOKEN``. Flags take priority.
A local model path conflicts with any upstream URL or model, including an exported URL.
Readiness does not check provider credentials or model IDs.

``REEF_PYTHON`` defaults to the interpreter that launched Reef. Local inference requires the engine and GPU-enabled PyTorch in that environment.
For weight training, also set the recipe and its training options. Reef resolves recipe packages from the working directory.

How Reef reads values
---------------------

Run ``reef serve --help`` to list flags. Add ``-c reef.yaml`` to include recipe flags.

- Flags override file values, then declared environment fallbacks, then defaults. The last repeated flag wins.
- String flags preserve text. In YAML, quote strings such as ``"00123"``. Numeric fields require finite values of the declared type.
- Booleans accept ``true/false``, ``yes/no``, ``on/off`` and ``1/0``. A bare flag enables an option and ``--no-<path>`` disables it.
- Pass lists and objects as quoted JSON or YAML. For example, ``--reef.tokens '[]'`` replaces the token list.
- Empty strings count as omitted, except that ``reef.host`` and ``inference.model-path`` reject them. YAML ``null`` clears an optional value.

In a file, ``${VAR}`` expands to an environment value or an empty string.
Use ``${VAR:?}`` to require a nonempty value. A flag overriding that key also overrides the reference.
``${dotted.path}`` resolves explicit file or flag values. References to defaults remain literal.

``--print-config`` lists every resolved value and its source, including automatic choices and defaults.
Ordinary startup logs omit most defaults.

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
