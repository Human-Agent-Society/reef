Reef CLI: serve
===============

Start a deployment from a checkout with a provider key in ``REEF_UPSTREAM_API_KEY``.

.. code:: bash

   export REEF_TOKEN="$(openssl rand -hex 16)"
   reef serve -c recipes/basic/external-provider.yaml

Reef starts each process when its dependencies are ready. An unexpected exit stops the deployment.
Press Ctrl-C to stop gracefully. Press it again to skip the remaining 30-second grace period.

Options
-------

.. container:: config

   ``-c, --config FILE``
       Read this file relative to the working directory. Without ``-c``, Reef uses flags and environment fallbacks.

   ``--model [PROVIDER/]MODEL``
       Select the upstream model.

   ``--service NAME``
       Run one of the seven services. Alias: ``--component``.

   ``--print-config``
       Print resolved settings and their sources, then exit. See `credential masking <configuration.rst#how-reef-reads-values>`__.

   ``--<section>.<key> VALUE``
       Override a file value.

Use ``reef serve --help`` for flags or ``reef --version`` for the installed version.

Selecting the model
-------------------

``ollama/MODEL`` selects ``http://127.0.0.1:11434`` with a placeholder key.
``openai/MODEL`` selects ``https://api.openai.com`` with ``REEF_UPSTREAM_API_KEY``.
Other model IDs require an upstream URL. Explicit URL and key flags take priority.

.. code:: bash

   reef serve --model ollama/gemma4:26b

Overriding config values
------------------------

Use the same paths as the configuration file. For example, change the batch size without editing the file:

.. code:: bash

   reef serve -c training.yaml --training.options.global-batch-size 32

Running one service
-------------------

By default one HTTP process runs all seven services. To split them, start each with the same config and a separate port.

.. code:: bash

   reef serve -c deploy.yaml --service control --reef.port 8901
   reef serve -c deploy.yaml --service inference --reef.port 8902

For service names and routing, see `Split services <../user-guide/split-services.rst>`__.

Where it writes
---------------

Logs and PID files use ``reef.run-dir`` (default ``.reef/run``). Records, commits and artifacts use ``storage.*``.
