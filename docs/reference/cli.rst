Reef CLI: serve
===============

Use ``reef serve`` to start a deployment from a config file or command-line flags.

.. code:: bash

   export REEF_TOKEN="$(openssl rand -hex 16)"
   reef serve -c recipes/basic/external-provider.yaml

This example requires a repository checkout. It binds all interfaces, so a token is required.

Reef starts processes in dependency order and waits for their readiness checks.
Independent processes can start together. An unexpected process exit stops the deployment.
Press Ctrl-C to stop gracefully. Press it again to skip the remaining 30-second grace period and force cleanup.

The package also installs ``reef-native`` for the native harness adapter. The adapter runs it automatically.

Options
-------

.. container:: config

   ``-c, --config``
       Optional config file, relative to the working directory. Reef does not discover ``$REEF_CONFIG`` or ``reef.yaml`` automatically.

   ``--model [PROVIDER/]MODEL``
       Upstream model. See `Selecting the model`_.

   ``--service NAME``
       Run one service: ``control``, ``inference``, ``samples``, ``training``, ``evaluation``, ``data`` or ``harness``. See `Running one service`_. The older ``--component`` alias remains supported.

   ``--print-config``
       Print resolved settings and their sources, then exit. This downloads no models and starts no services. Credentials are masked by key name; see `Resolved settings <configuration.rst#how-reef-reads-values>`__.

   ``--<section>.<key> VALUE``
       Override one config value. See `Overriding config values`_.

   ``--help``
       Show command help.

   ``-V, --version``
       Show the installed version. Use without a command: ``reef --version``.

Selecting the model
-------------------

Without ``-c``, configure inference with flags and the supported `environment fallbacks <configuration.rst#start-without-a-file>`__.
Pass ``-c`` to use a recipe's deployment file.

``--model`` is a shorthand:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Spelling
     - Effect
   * - ``ollama/MODEL``
     - Fills ``http://127.0.0.1:11434`` and a placeholder key.
   * - ``openai/MODEL``
     - Fills ``https://api.openai.com`` and reads ``REEF_UPSTREAM_API_KEY``.
   * - any other value
     - The model ID as is. It needs an upstream URL.

An explicit ``--inference.upstream-url`` or ``--inference.upstream-api-key``
wins over the prefix.

.. code:: bash

   reef serve --model ollama/gemma4:26b

Overriding config values
------------------------

Flags use the same paths and types as file values, and take priority.
Components validate overrides for their own options.

.. code:: bash

   reef serve -c path/to/training.yaml \
     --inference.model-path ~/models/Qwen2.5-1.5B-Instruct \
     --training.options.global-batch-size 32

Use public paths to relocate state. Older bare and ``reef.*`` aliases remain accepted.

.. code:: bash

   reef serve -c recipes/basic/external-provider.yaml \
     --storage.agent-record-dir .reef/agent-record \
     --storage.artifact-work-dir .reef/artifact-work \
     --storage.artifact-cache-dir .reef/artifact-cache

Running one service
-------------------

By default, one HTTP process runs all seven services.
With ``--service``, the process runs only the selected service and starts no other service.
Use the same config for every process so they share storage:

.. code:: bash

   reef serve -c deploy.yaml --service control --reef.port 8901
   reef serve -c deploy.yaml --service inference --reef.port 8902
   reef serve -c deploy.yaml --service samples --reef.port 8903

`Run the services in separate processes
<../user-guide/split-services.rst>`__ describes
what each service does and which routes it answers.

Where it writes
---------------

Managed process logs and PID files use ``reef.run-dir`` (default: ``.reef/run``).
Records, commits and artifacts use the configured ``storage.*`` paths.
