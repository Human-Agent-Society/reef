Reef CLI: serve
===============

From a checkout, set your provider key in ``REEF_UPSTREAM_API_KEY`` and start Reef:

.. code:: bash

   export REEF_TOKEN="$(openssl rand -hex 16)"
   reef serve -c recipes/basic/external-provider.yaml

Use ``reef serve --help`` to list flags. Press Ctrl-C to stop. Logs and PID files default to ``.reef/run``.

On Linux and macOS, closing the terminal or the SSH session (SIGHUP) also
requests graceful shutdown. A second hangup does not skip the grace period. To
keep the stack running after you log out, start it with ``nohup reef serve``.
A deployment with managed routing uses SIGHUP to reload its policy instead.

Options
-------

- ``-c FILE`` reads deployment YAML. Override a value with ``--<section>.<key> VALUE``.
- ``--print-config`` prints resolved settings and their sources. See `credential masking <configuration.rst#how-reef-reads-values>`__.
- ``--service NAME`` runs one service. See `Split services <../user-guide/split-services.rst>`__ for the names, ports and shared stores.
- ``--model ollama/MODEL`` uses the local Ollama server. ``--model openai/MODEL`` uses OpenAI with ``REEF_UPSTREAM_API_KEY``.
  Other model IDs need an upstream URL. Explicit URL and key flags take priority.
